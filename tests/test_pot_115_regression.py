"""115: Otsu da ROI inteira perde um dígito; não resolver por maioria.

Fixture: apenas ROI em cinza [210:305,850:1070], captura real fornecida
de 11/10/2026, 23:24:56. Sem jogadores, terminal ou identificadores.
"""

import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr import ocr_reader
from tests import test_hand_cycle_integration as cycle
from tests import test_pot_ocr_consistency as consistency


def pot_115_frame():
    data = (Path(__file__).parent / "fixtures/replay_pot_115.png.b64").read_bytes()
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[210:305, 850:1070] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return frame


class TestPot115Regression(unittest.TestCase):
    def read(self, outputs, frame=None):
        self.diagnostics = {}
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=outputs) as ocr:
            result = ocr_reader.extract_pot(pot_115_frame() if frame is None else frame,
                                           default_value=None, verbose=False, diagnostics=self.diagnostics)
        self.calls = ocr.call_args_list
        return result

    def test_real_115_recovers(self):
        self.assertEqual(ocr_reader.extract_pot(pot_115_frame(), default_value=None, verbose=False), 115)

    def test_conflict_recovery_requires_numeric_line_and_three_glyphs(self):
        self.assertEqual(self.read(["115", "15", "115", "115"]), 115)
        resolution = self.diagnostics["resolution"]
        self.assertEqual(resolution["glyph_count"], 3)
        self.assertEqual(resolution["status"], "RESOLVED")
        self.assertEqual(self.diagnostics["status"], "READ")
        self.assertEqual(self.diagnostics["readings"][1]["value"], 15)
        self.assertIn("glifos", self.diagnostics["readings"][1]["rejection_reason"])
        self.assertIn("--psm 7", self.calls[-1].kwargs["config"])

    def test_directed_read_that_still_loses_digit_does_not_resolve(self):
        self.assertIsNone(self.read(["115", "15", "115", "15"]))

    def test_directed_read_without_result_does_not_resolve(self):
        self.assertIsNone(self.read(["115", "15", "115", None]))

    def test_directed_read_cannot_introduce_a_new_value(self):
        self.assertIsNone(self.read(["115", "15", "115", "155"]))

    def test_same_length_contradictions_remain_conflicts(self):
        self.assertIsNone(self.read(["115", "155", "115"]))
        self.assertEqual(len(self.calls), 3)

    def test_three_distinct_numbers_remain_conflicts(self):
        self.assertIsNone(self.read(["115", "155", "550"]))

    def test_correct_read_and_empty_variant_is_not_conflict(self):
        self.assertEqual(self.read(["115", None, "115"]), 115)
        self.assertEqual(len(self.calls), 3)

    def test_three_empty_methods_remain_unknown(self):
        self.assertIsNone(self.read([None, "", ""]))
        self.assertEqual(len(self.calls), 3)

    def test_partial_token_is_not_resolved_by_two_correct_reads(self):
        self.assertIsNone(self.read(["115", "1?5", "115"]))

    def test_label_is_not_part_of_numeric_token(self):
        self.assertEqual(self.read(["POT\n115", "POTE\n115", "115"]), 115)

    def test_visual_recovery_can_select_smaller_value_not_largest(self):
        frame = consistency.pot_55_frame()  # Dois glifos, não três.
        self.assertEqual(self.read(["155", "55", "155", "55"], frame=frame), 55)

    def test_no_recovery_when_number_line_is_clipped_at_roi_edge(self):
        frame = pot_115_frame()
        # A linha final tocando a borda não comprova cobertura completa do número.
        frame[304, 850:1070] = 255
        self.assertIsNone(self.read(["115", "15", "115"], frame=frame))

    def test_noise_in_numeric_line_does_not_become_an_extra_digit(self):
        frame = pot_115_frame()
        frame[265, 980] = 255  # Componente sem a altura/baseline dos dígitos.
        self.assertIsNone(self.read(["115", "15", "115"], frame=frame))
        self.assertEqual(len(self.calls), 3)

    def test_no_isolation_proof_without_separate_label_and_number_lines(self):
        frame = pot_115_frame()
        frame[230:241, 850:1070] = 0  # Remove apenas a linha do rótulo.
        self.assertIsNone(self.read(["115", "15", "115"], frame=frame))
        self.assertEqual(len(self.calls), 3)


class TestPot115FinancialRecovery(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp
    live = cycle.TestHandCycleIntegration.live

    def test_uncertain_then_real_115_then_legitimate_change_preserve_equity_cache(self):
        with patch.object(ocr_reader.pytesseract, "image_to_string", side_effect=["115", "155", "115"]):
            uncertain = ocr_reader.extract_pot(pot_115_frame(), default_value=None, verbose=False)
        recovered_pot = ocr_reader.extract_pot(pot_115_frame(), default_value=None, verbose=False)
        hero = ["5s", "5h"]
        self.app.hero_cards = list(hero)
        self.live([cycle.capture(hero=hero, pot=p, stack=1210, call=15)
                   for p in (55, uncertain, recovered_pot, 135)])
        first, invalid, recovered, changed = self.analyses
        self.assertFalse(invalid["action_available"])
        self.assertFalse(invalid["financial_valid"])
        self.assertIsNone(invalid["financial_observation"]["pot_size"])
        self.assertIs(self.snapshots[2]["cache"], first)
        self.assertEqual(recovered["pot_size"], 115)
        self.assertEqual(changed["pot_size"], 135)
        self.assertTrue(recovered["equity_cached"])
        self.assertTrue(changed["equity_cached"])
        self.assertTrue(recovered["recalculated"])
        self.assertTrue(changed["recalculated"])
        self.calculate_equity.assert_called_once()
        self.assertEqual(self.risk.call_count, 3)
        self.assertEqual(self.reset_count, 0)
        self.assertIs(self.app.last_analysis, changed)


if __name__ == "__main__":
    unittest.main()
