"""Contagem visual de adversários na mão, não de assentos ocupados."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.config import DEFAULT_NUM_OPPONENTS, SAMPLES_DIR
from src.vision.detector import detect_active_opponents


class TestOpponentDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = cv2.imread(str(SAMPLES_DIR / "mesa_turno.png"))
        # Dois dorsos reais parcialmente encobertos pelo badge do jogador.
        cls.backs = cls.sample[432:493, 314:430].copy()
        cls.hero = cls.sample[652:713, 1208:1387].copy()

    def table(self, active_seats, occupied_seats=None):
        frame = np.full((1080, 1920, 3), (40, 110, 40), dtype=np.uint8)
        seats = [(350, 450), (570, 215), (1250, 215), (1500, 450)]
        if occupied_seats is None:
            occupied_seats = range(4)
        for index, (x, y) in enumerate(seats):
            text = "Jogador" if index in occupied_seats else "Assento vazio"
            cv2.putText(frame, text, (x, y + 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
            if index in active_seats:
                frame[y:y + self.backs.shape[0], x:x + self.backs.shape[1]] = self.backs
        # Hero aberto em baixo à direita; sexto assento sem jogador.
        frame[652:713, 1208:1387] = self.hero
        cv2.putText(frame, "Assento vazio", (570, 740), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
        return frame

    def test_heads_up_has_one_opponent(self):
        self.assertEqual(detect_active_opponents(self.table([0], [0])), 1)

    def test_five_players_including_hero_has_four_opponents(self):
        self.assertEqual(detect_active_opponents(self.table(range(4))), 4)

    def test_empty_seat_does_not_add_an_opponent(self):
        frame = self.table([0, 1, 3], [0, 1, 3])
        self.assertEqual(detect_active_opponents(frame), 3)

    def test_folded_player_remains_seated_but_is_not_counted(self):
        self.assertEqual(detect_active_opponents(self.table(range(4))), 4)
        self.assertEqual(detect_active_opponents(self.table([0, 1, 3])), 3)

    def test_existing_samples_ignore_hero_board_avatars_and_deck(self):
        for name, expected in [("mesa_turno.png", 2), ("mesa_par_de_6.png", 2), ("mesa_novas.png", 1)]:
            with self.subTest(name=name):
                self.assertEqual(detect_active_opponents(SAMPLES_DIR / name), expected)

    def test_no_evidence_is_configured_fallback_not_confirmed_zero(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertEqual(detect_active_opponents(frame), DEFAULT_NUM_OPPONENTS)
        self.assertEqual(detect_active_opponents("missing_opponents_regression.png"), DEFAULT_NUM_OPPONENTS)

    def test_known_hero_box_is_excluded_even_with_closed_cards(self):
        frame = self.table([0])
        frame[652:713, 1208:1387] = (40, 110, 40)
        frame[652:713, 1208:1324] = self.backs
        self.assertEqual(detect_active_opponents(
            frame, hero_coords={"left": 1208, "top": 652, "width": 179, "height": 61}
        ), 1)

    def test_two_fully_visible_backs_are_one_opponent(self):
        frame = self.table([], [0])
        card = self.sample[432:493, 346:430]
        for x in (350, 434):
            frame[450:511, x:x + 84] = card
        self.assertEqual(detect_active_opponents(frame), 1)

    def test_fallback_uses_configuration_instead_of_literal_four(self):
        with patch("src.vision.detector.DEFAULT_NUM_OPPONENTS", 3):
            self.assertEqual(detect_active_opponents(np.zeros((1080, 1920, 3), dtype=np.uint8)), 3)
            self.assertEqual(detect_active_opponents("missing_opponents_regression.png"), 3)


if __name__ == "__main__":
    unittest.main()
