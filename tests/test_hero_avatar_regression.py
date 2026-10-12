"""Pixels reais: nome/stack removidos; preservados cartas e avatar genérico.

ROI [412:563,1395:1640]. Origem: capturas 12/10 00:21:03, 00:22:40 e
11/10 22:48:03. A máscara não é uma reconstrução das cartas.
"""
import base64
from pathlib import Path
import unittest
import cv2
import numpy as np
from src.vision import detector


def real_hero_frame(name):
    raw = (Path(__file__).parent / "fixtures" / (name + ".png.b64")).read_bytes()
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
    frame = np.full((1080, 1920, 3), (40, 110, 40), np.uint8)
    frame[412:563, 1395:1640] = crop
    return frame


class TestHeroAvatarRegression(unittest.TestCase):
    def test_diagnostics_separate_localization_and_classification_failure(self):
        diagnostics = {}
        blank = np.full((1080, 1920, 3), (40, 110, 40), np.uint8)
        self.assertEqual(detector.find_and_detect_hero_cards(blank, verbose=False, save_unknown=False,
                                                          diagnostics=diagnostics), [])
        self.assertEqual(diagnostics["rejection"], "NO_CARD_PAIR_GEOMETRY")
        self.assertEqual(diagnostics["slots"], [])
        blank[424:482,1461:1539] = 255
        blank[424:482,1550:1629] = 255
        self.assertEqual(detector.find_and_detect_hero_cards(blank, verbose=False, save_unknown=False,
                                                          diagnostics=diagnostics), [])
        self.assertEqual(diagnostics["rejection"], "CLASSIFICATION_UNCERTAIN")
        self.assertEqual(len(diagnostics["slots"]), 2)

    def test_avatar_separation_is_reported_without_relaxing_card_geometry(self):
        diagnostics = {}
        cards = detector.find_and_detect_hero_cards(real_hero_frame("hero_avatar_4c_7c"),
                                                   verbose=False, save_unknown=False, diagnostics=diagnostics)
        self.assertEqual(cards, ["4c", "7c"])
        self.assertTrue(any(item["box"][3] == 128 for item in diagnostics["contours"]))
        self.assertTrue(diagnostics["separated_cards"])
        for _, _, width, height in diagnostics["separated_cards"]:
            self.assertLess(width, 120)
            self.assertLess(height, 120)
        self.assertIsNone(diagnostics["rejection"])

    def test_original_cards_next_to_avatar_and_under_name_panel(self):
        for name, expected in (("hero_avatar_4c_7c", ["4c", "7c"]),
                               ("hero_avatar_qh_ah", ["Qh", "Ah"]),
                               ("hero_4h_9d", ["4h", "9d"])):
            for hint in (None, dict(left=1461, top=424, width=168, height=58)):
                with self.subTest(name=name, hint=hint):
                    result = detector.find_and_detect_hero_cards(real_hero_frame(name), verbose=False,
                                                               save_unknown=False, hero_coords_hint=hint)
                    self.assertEqual(result, expected)

    def test_blank_or_truncated_suit_is_not_filled_from_previous_cards(self):
        frame = real_hero_frame("hero_4h_9d")
        hint = dict(left=1461, top=424, width=168, height=58)
        self.assertEqual(detector.find_and_detect_hero_cards(frame, verbose=False, save_unknown=False), ["4h", "9d"])
        frame[460:483,1460:1540] = (0,0,0)
        self.assertEqual(detector.find_and_detect_hero_cards(frame, verbose=False, save_unknown=False,
                                                          hero_coords_hint=hint), [])
        self.assertEqual(detector.find_and_detect_hero_cards(real_hero_frame("hero_4h_9d"), verbose=False,
                                                          save_unknown=False, hero_coords_hint=hint), ["4h", "9d"])


if __name__ == "__main__":
    unittest.main()
