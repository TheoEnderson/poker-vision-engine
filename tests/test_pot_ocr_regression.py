"""OCR do pote: recorte real anonimizado, parser e bloqueio financeiro.

Fixture: captura fornecida em 11/10/2026, 22:48:03, 1920x1080.
Somente ROI [210:305, 850:1070], convertida para cinza sem redimensionar.
PNG em base64 para preservar os pixels sem incluir jogadores/logs da tela.
"""

import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr import ocr_reader
from tests import test_hand_cycle_integration as cycle


def real_pot_frame():
    data = (Path(__file__).parent / "fixtures/replay_pot_13.png.b64").read_bytes()
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[210:305, 850:1070] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return frame


class TestPotOCRRegression(unittest.TestCase):
    def test_real_capture_reads_13_using_live_arguments(self):
        frame = real_pot_frame()
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertEqual(ocr_reader.extract_pot(frame, default_value=None, verbose=False), 13.0)

    def test_numeric_lines_labels_digits_and_separators(self):
        frame = real_pot_frame()
        for raw, expected in (("13\n", 13), ("POT\n13\n", 13), ("POTE\n13\n", 13),
                              ("7\n", 7), ("128\n", 128), ("4321\n", 4321),
                              ("POTE\n3,840\n", 3840), ("POT\n3.840\n", 3840)):
            with self.subTest(raw=raw), patch.object(ocr_reader.pytesseract, "image_to_string", return_value=raw) as ocr:
                self.assertEqual(ocr_reader.extract_pot(frame, default_value=None, verbose=False), expected)
                self.assertEqual(ocr.call_count, 3)  # Confronta a leitura com a mesma captura em outras condições.

    def test_unreadable_or_none_stays_unknown(self):
        for raw in ("", "   ", "POTE\n???", "??\n.\n", None):
            with self.subTest(raw=raw), patch.object(ocr_reader.pytesseract, "image_to_string", return_value=raw):
                self.assertIsNone(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False))

    def test_empty_native_reads_do_not_prevent_enlarged_read(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=["", "", "13\n"]) as ocr:
            self.assertEqual(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False), 13)
        self.assertEqual([c.args[0].shape for c in ocr.call_args_list], [(95, 220), (95, 220), (285, 660)])
        self.assertEqual(len({c.kwargs["config"] for c in ocr.call_args_list}), 1)

    def test_otsu_success_is_checked_against_enlarged_read(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=["", "13\n", "13\n"]) as ocr:
            self.assertEqual(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False), 13)
            self.assertEqual(ocr.call_count, 3)

    def test_recovery_has_no_persistent_none_or_previous_value(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", return_value=""):
            self.assertIsNone(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False))
        self.assertEqual(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False), 13)
        with patch.object(ocr_reader.pytesseract, "image_to_string", return_value=""):
            self.assertIsNone(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False))

    def test_existing_sanity_limit_still_rejects_false_positive(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", return_value="93700\n"):
            self.assertIsNone(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False))

    def test_enlarged_retry_preserves_sanity_limit(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=["", "", "93700\n"]):
            self.assertIsNone(ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False))

    def test_blank_real_ocr_does_not_invent_zero(self):
        self.assertIsNone(ocr_reader.extract_pot(np.zeros((1080, 1920, 3), dtype=np.uint8),
                                               default_value=None, verbose=False))


class TestPotFinancialCycle(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp
    live = cycle.TestHandCycleIntegration.live

    def test_flop_to_turn_repeated_missing_pot_does_not_run_math(self):
        flop = ["9c", "3h", "4s"]
        turn = flop + ["Kc"]
        hero = ["Ah", "Qh"]
        self.app.hero_cards = list(hero)
        frames = [cycle.capture(hero=hero, board=b, pot=None, stack=201, actions=["FOLD", "CHECK", "BET"])
                  for b in [flop] * 3 + [turn] * 3]
        initial_weights = dict(self.model.weights)
        output = self.live(frames)
        self.assertEqual(self.app.state_machine.board, turn)
        self.assertEqual(self.app.state_machine.current_state.value, "TURN")
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.reset_count, 0)
        self.assertEqual(self.model.weights, initial_weights)
        self.assertIsNone(self.app.last_analysis)
        self.assertEqual(len(self.analyses), 6)
        self.assertTrue(all(not a["financial_valid"] and not a["action_available"] for a in self.analyses))
        self.assertTrue(all(a["financial_observation"]["pot_size"] is None for a in self.analyses))
        self.calculate_equity.assert_not_called()
        self.risk.assert_not_called()
        self.assertIn("Pote não confirmado", output)
        self.assertNotIn("RECOMENDAÇÃO PARA A CAPTURA:", output)

    def test_invalid_captures_preserve_cache_then_recover_with_current_pot(self):
        flop = ["9c", "3h", "4s"]
        hero = ["Ah", "Qh"]
        self.app.hero_cards = list(hero)
        observed = ocr_reader.extract_pot(real_pot_frame(), default_value=None, verbose=False)
        frames = [cycle.capture(hero=hero, board=flop, pot=p, stack=201, actions=["FOLD", "CHECK", "BET"])
                  for p in (7, None, None, observed)]
        self.live(frames)
        initial, invalid1, invalid2, recovered = self.analyses
        for invalid in (invalid1, invalid2):
            self.assertFalse(invalid["action_available"])
            self.assertIsNone(invalid["financial_observation"]["pot_size"])
        self.assertIs(self.snapshots[2]["cache"], initial)
        self.assertEqual(self.snapshots[1]["weights"], self.snapshots[3]["weights"])
        self.assertEqual(recovered["pot_size"], 13)
        self.assertEqual(recovered["hero_stack"], 201)
        self.assertEqual(recovered["bet_to_call"], 0)
        self.assertTrue(recovered["recalculated"])
        self.assertIsNot(recovered.get("financial_valid"), False)
        self.assertIsNot(recovered.get("observation_valid"), False)
        self.assertNotIn("Pote não confirmado", recovered["action_status"])
        # Finanças válidas não dispensam a validação independente da ação:
        # o CFR pode sugerir BET, cujo tamanho ainda não é validado na interface.
        if recovered["action"] == "BET":
            self.assertFalse(recovered["action_available"])
            self.assertIn("tamanho/limites", recovered["action_status"])
        self.assertTrue(recovered["equity_cached"])
        self.calculate_equity.assert_called_once()
        self.assertEqual(self.risk.call_count, 2)
        self.assertEqual([c.kwargs["pot_size"] for c in self.risk.call_args_list], [7, 13])
        self.assertIs(self.app.last_analysis, recovered)


if __name__ == "__main__":
    unittest.main()
