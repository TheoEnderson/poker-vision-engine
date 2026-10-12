"""Pote: conflitos na mesma captura não são resolvidos por maioria/histórico.

Erros 55->5, 155->55 e 105->15 são injetados na fronteira Tesseract:
não representam uma reprodução desses erros nas capturas originais.
Fixture 55: somente [210:305,850:1070] em cinza da captura 11/10/2026
23:07:53, sem jogadores, identificadores ou terminal.
"""

import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr import ocr_reader
from tests import test_hand_cycle_integration as cycle
from tests import test_pot_ocr_regression as previous


def pot_55_frame():
    data = (Path(__file__).parent / "fixtures/replay_pot_55.png.b64").read_bytes()
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[210:305, 850:1070] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return frame


class TestPotOCRConsistency(unittest.TestCase):
    def read(self, outputs, **kwargs):
        # A tentativa adicional, quando houver geometria suficiente, fica
        # ilegível nestes cenários: conflito NÃO é resolvido por concordância.
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=list(outputs) + [""]):
            return ocr_reader.extract_pot(pot_55_frame(), default_value=None, verbose=False, **kwargs)

    def test_real_55_and_previous_13(self):
        self.assertEqual(ocr_reader.extract_pot(pot_55_frame(), default_value=None, verbose=False), 55)
        self.assertEqual(ocr_reader.extract_pot(previous.real_pot_frame(), default_value=None, verbose=False), 13)

    def test_complete_numbers_are_not_rejected_based_on_size(self):
        for number in (5, 13, 55, 105, 155, 550, 1055):
            with self.subTest(number=number):
                self.assertEqual(self.read([str(number)] * 3), number)

    def test_native_number_does_not_skip_other_visual_evidence(self):
        self.assertIsNone(self.read(["5\n", "55\n", "55\n"]))

    def test_injected_first_middle_and_last_digit_loss_suspend(self):
        for short, full in (("5", "55"), ("55", "155"), ("15", "105"),
                            ("55", "550"), ("105", "1055")):
            with self.subTest(short=short, full=full):
                self.assertIsNone(self.read([short, short, full]))

    def test_conflicts_are_not_resolved_by_majority_or_largest_value(self):
        for outputs in (["5", "5", "55"], ["55", "5", "55"], ["55", "55", "5"]):
            with self.subTest(outputs=outputs):
                self.assertIsNone(self.read(outputs))

    def test_partial_or_malformed_numeric_tokens_are_not_extracted(self):
        for raw in ("1?5", "?55", "55?", "10 55", "-5", "1.2.3", "12,34", "13.5"):
            with self.subTest(raw=raw):
                self.assertIsNone(self.read([raw] * 3))

    def test_malformed_read_cannot_be_outvoted(self):
        self.assertIsNone(self.read(["55?", "55", "55"]))

    def test_full_labels_and_thousands_separators(self):
        self.assertEqual(self.read(["POTE\n1,055", "POT\n1.055", "1055"]), 1055)

    def test_empty_reads_do_not_disagree_with_enlarged_read(self):
        self.assertEqual(self.read([None, "", "13"]), 13)

    def test_no_result_remains_unknown(self):
        self.assertIsNone(self.read([None, "", ""]))

    def test_empty_variant_is_not_a_conflict_or_a_reason_to_freeze_valid_read(self):
        self.assertEqual(self.read(["5", "5", ""]), 5)

    def test_repeated_conflict_never_becomes_confirmed_and_valid_read_recovers(self):
        for _ in range(2):
            self.assertIsNone(self.read(["5", "5", "55"]))
        self.assertEqual(self.read(["55", "55", "55"]), 55)
        self.assertEqual(self.read(["70", "70", "70"]), 70)
        self.assertEqual(self.read(["5", "5", "5"]), 5)  # Sem monotonicidade inventada.

    def test_conflict_does_not_return_previous_default_value(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=["5", "55", "55", ""]):
            self.assertIsNone(ocr_reader.extract_pot(pot_55_frame(), default_value=100, verbose=False))

    def test_optional_diagnostics_explain_conflict_and_preserve_raw_evidence(self):
        diagnostics = {}
        self.assertIsNone(self.read(["5\n", "55\n", "55\n"], diagnostics=diagnostics))
        self.assertEqual(diagnostics["status"], "CONFLICT")
        self.assertEqual([r["raw_text"] for r in diagnostics["readings"][:3]], ["5\n", "55\n", "55\n"])
        self.assertEqual([r["value"] for r in diagnostics["readings"][:3]], [5, 55, 55])
        self.assertEqual(diagnostics["readings"][-1]["status"], "EMPTY")


class TestPotConflictFinancialIntegration(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp
    live = cycle.TestHandCycleIntegration.live

    def test_conflict_suspends_finances_and_recovers_without_repeating_equity(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=["5", "5", "55", ""]):
            conflict = ocr_reader.extract_pot(pot_55_frame(), default_value=None, verbose=False)
        hero = ["5d", "6s"]
        self.app.hero_cards = list(hero)
        frames = [cycle.capture(hero=hero, pot=p, call=15, stack=1235)
                  for p in (55, conflict, conflict, 70)]
        self.live(frames)
        first, invalid1, invalid2, recovered = self.analyses
        self.assertEqual(first["pot_size"], 55)
        for invalid in (invalid1, invalid2):
            self.assertFalse(invalid["financial_valid"])
            self.assertFalse(invalid["action_available"])
            self.assertIsNone(invalid["financial_observation"]["pot_size"])
        self.assertIs(self.snapshots[2]["cache"], first)
        self.assertEqual(self.snapshots[1]["weights"], self.snapshots[3]["weights"])
        self.assertEqual(recovered["pot_size"], 70)
        self.assertTrue(recovered["recalculated"])
        self.assertTrue(recovered["equity_cached"])
        self.assertTrue(recovered["action_available"])
        self.assertEqual(self.risk.call_count, 2)
        self.calculate_equity.assert_called_once()
        self.assertAlmostEqual(first["pot_odds"], 100 * 15 / 70)
        self.assertAlmostEqual(recovered["pot_odds"], 100 * 15 / 85)
        self.assertNotEqual(first["ev"], recovered["ev"])
        self.assertEqual(self.reset_count, 0)
        self.assertIs(self.app.last_analysis, recovered)


if __name__ == "__main__":
    unittest.main()
