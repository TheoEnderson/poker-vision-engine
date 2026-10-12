"""Cartas reais de 12/10 12:58; borda variável sintética, conforme caixas do JSONL.

A fixture contém somente os 168x58 pixels das cartas, sem nome/stack/avatar.
Os frames dos ciclos 38/40 não foram salvos; a borda reproduz sua geometria.
"""
import base64
from pathlib import Path
import unittest

import cv2
import numpy as np
from src.vision.detector import find_and_detect_hero_cards


def hero_frame(border_height=None):
    raw = (Path(__file__).parent / 'fixtures/hero_5h_7c.png.b64').read_bytes()
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
    frame = np.full((1080, 1920, 3), (40, 110, 40), np.uint8)
    frame[306:364, 1467:1635] = crop
    if border_height is not None:
        points = [(1467, 355), (1450, 355), (1412, 375),
                  (1412, 306 + border_height - 2), (1440, 306 + border_height - 2)]
        cv2.polylines(frame, [np.array(points)], False, (255, 255, 255), 2)
    return frame


class TestHeroAllInSession(unittest.TestCase):
    hint = dict(left=1467, top=306, width=168, height=58)

    def recognize(self, frame, hint=None, diagnostics=None):
        return find_and_detect_hero_cards(frame, verbose=False, save_unknown=False,
                                         hero_coords_hint=hint, diagnostics=diagnostics)

    def test_real_5h_7c_with_and_without_known_seat(self):
        for hint in (None, self.hint):
            with self.subTest(hint=hint):
                self.assertEqual(self.recognize(hero_frame(), hint), ['5h', '7c'])

    def test_avatar_connection_below_and_above_old_height_cutoff(self):
        for height in (106, 116, 127):
            for hint in (None, self.hint):
                with self.subTest(height=height, hint=hint):
                    diagnostics = {}
                    cards = self.recognize(hero_frame(height), hint, diagnostics)
                    self.assertIn([1411, 306, 134, height],
                                  [c['box'] for c in diagnostics['contours']])
                    self.assertEqual(cards, ['5h', '7c'])
                    self.assertIn([1467, 306, 78, 56], diagnostics['separated_cards'])

    def test_missing_suit_stays_uncertain_then_recovers(self):
        frame = hero_frame(116)
        frame[340:364, 1556:1635] = 0
        self.assertEqual(self.recognize(frame, self.hint), [])
        self.assertEqual(self.recognize(hero_frame(116), self.hint), ['5h', '7c'])

    def test_genuine_merged_card_pair_still_recognizes(self):
        frame = hero_frame()
        frame[310:355, 1540:1560] = 255
        self.assertEqual(self.recognize(frame, self.hint), ['5h', '7c'])
