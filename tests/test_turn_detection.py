"""Turno e ações exigem evidência dos botões, não apenas call igual a zero."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr.ocr_reader import detect_turn_and_bet_to_call, parse_action_button_text


class TestActionText(unittest.TestCase):
    def test_empty_noise_and_sizing_controls_do_not_confirm_check(self):
        for text in ("", "???", "100", "MIN. 1/2 POTE MAX", "LEVANTAR-SE"):
            with self.subTest(text=text):
                self.assertFalse(parse_action_button_text(text)["is_hero_turn"])

    def test_preselection_is_not_a_current_action(self):
        for text in ("CHECK/FOLD", "Check / Fold", "CHECK ANY", "Fold to any bet"):
            with self.subTest(text=text):
                self.assertFalse(parse_action_button_text(text)["is_hero_turn"])

    def test_explicit_check_is_free_including_preflop(self):
        result = parse_action_button_text("DESISTO PASSO APOSTO 100")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["bet_to_call"], 0.0)
        self.assertIn("CHECK", result["available_actions"])
        self.assertNotIn("CALL", result["available_actions"])

    def test_call_amount_is_associated_with_call_label(self):
        result = parse_action_button_text("MIN 10 DESISTO PAGO 40 AUMENTO PARA 100")
        self.assertEqual(result["bet_to_call"], 40.0)
        self.assertEqual(result["available_actions"], ["FOLD", "CALL", "RAISE"])

    def test_raise_amount_does_not_replace_unreadable_call(self):
        result = parse_action_button_text("FOLD CALL ??? RAISE TO 100")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["action_type"], "CALL")
        self.assertIsNone(result["bet_to_call"])

    def test_fold_or_raise_alone_does_not_prove_free_action(self):
        for text in ("FOLD", "RAISE TO 100", "BET 100"):
            with self.subTest(text=text):
                self.assertIsNone(parse_action_button_text(text)["bet_to_call"])

    def test_zero_call_is_not_check(self):
        result = parse_action_button_text("CALL 0 RAISE 100")
        self.assertIsNone(result["bet_to_call"])
        self.assertNotIn("CHECK", result["available_actions"])

    def test_conflicting_check_and_call_are_uncertain(self):
        self.assertFalse(parse_action_button_text("CHECK CALL 40")["is_hero_turn"])

    def test_call_numeric_formats_and_small_call(self):
        for text, amount in [("CALL 1", 1.0), ("CALL 1,000", 1000.0), ("PAGO 1.000", 1000.0), ("CALL 1,000.50", 1000.5), ("PAGO 1.000,50", 1000.5)]:
            with self.subTest(text=text):
                self.assertEqual(parse_action_button_text(text)["bet_to_call"], amount)
        self.assertIsNone(parse_action_button_text("CALL 1.2.3")["bet_to_call"])

    def test_fold_and_all_in_amount_does_not_establish_call_payment(self):
        result = parse_action_button_text("DESISTO ALL-IN 970")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["action_type"], "ALL-IN")
        self.assertIsNone(result["bet_to_call"])
        self.assertEqual(result["all_in_displayed_amount"], 970.0)
        self.assertEqual(result.get("all_in_nature"), "UNKNOWN")
        self.assertEqual(result["available_actions"], ["FOLD", "ALL-IN"])

    def test_fold_and_all_in_without_amount_has_unconfirmed_cost(self):
        result = parse_action_button_text("FOLD ALL-IN")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["action_type"], "ALL-IN")
        self.assertIsNone(result["bet_to_call"])
        self.assertEqual(result.get("all_in_nature"), "UNKNOWN")
        self.assertEqual(result["available_actions"], ["FOLD", "ALL-IN"])

    def test_call_and_all_in_distinguishes_raise_from_call_payment(self):
        result = parse_action_button_text("FOLD CALL 40 ALL-IN 500")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["action_type"], "CALL")
        self.assertEqual(result["bet_to_call"], 40.0)
        self.assertEqual(result.get("all_in_nature"), "RAISE")
        self.assertEqual(result["available_actions"], ["FOLD", "CALL", "ALL-IN"])

    def test_check_and_all_in_distinguishes_bet_from_call(self):
        result = parse_action_button_text("CHECK ALL-IN 500")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["action_type"], "CHECK")
        self.assertEqual(result["bet_to_call"], 0.0)
        self.assertEqual(result.get("all_in_nature"), "BET")
        self.assertEqual(result["available_actions"], ["CHECK", "ALL-IN"])

    def test_isolated_all_in_is_unconfirmed_nature(self):
        result = parse_action_button_text("ALL-IN 500")
        self.assertTrue(result["is_hero_turn"])
        self.assertEqual(result["action_type"], "UNKNOWN")
        self.assertIsNone(result["bet_to_call"])
        self.assertEqual(result.get("all_in_nature"), "UNKNOWN")

    def test_all_in_invalid_amounts_remain_unconfirmed(self):
        for text in ("FOLD ALL-IN 0", "FOLD ALL-IN 1.2.3", "FOLD ALL-IN ???"):
            with self.subTest(text=text):
                res = parse_action_button_text(text)
                self.assertIsNone(res["bet_to_call"])
                self.assertEqual(res.get("all_in_nature"), "UNKNOWN")


class TestVisualTurn(unittest.TestCase):
    def buttons(self, colors):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for index, color in enumerate(colors):
            cv2.rectangle(frame, (1210 + 210 * index, 990), (1390 + 210 * index, 1050), color, -1)
        return frame

    def test_no_buttons_means_waiting_without_ocr(self):
        with patch("pytesseract.image_to_string") as ocr:
            result = detect_turn_and_bet_to_call(self.buttons([]), verbose=False)
        self.assertFalse(result["is_hero_turn"])
        ocr.assert_not_called()

    def test_colored_controls_without_action_labels_are_not_a_turn(self):
        with patch("pytesseract.image_to_string", return_value="MIN 100"):
            result = detect_turn_and_bet_to_call(self.buttons([(0, 0, 220)]), verbose=False)
        self.assertFalse(result["is_hero_turn"])

    def test_call_and_raise_are_read_from_separate_buttons(self):
        with patch("pytesseract.image_to_string", side_effect=["FOLD", "CALL ???", "RAISE TO 100"]):
            result = detect_turn_and_bet_to_call(self.buttons([(0, 0, 220)] * 3), verbose=False)
        self.assertIsNone(result["bet_to_call"])
        self.assertEqual(result["available_actions"], ["FOLD", "CALL", "RAISE"])

    def test_disabled_gray_button_does_not_prove_an_action(self):
        with patch("pytesseract.image_to_string") as ocr:
            result = detect_turn_and_bet_to_call(self.buttons([(100, 100, 100)]), verbose=False)
        self.assertFalse(result["is_hero_turn"])
        ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
