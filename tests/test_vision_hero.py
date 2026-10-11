"""Regressões visuais: glifos reais, recortes ocluídos e assentos conhecidos.

As mesas sintéticas são montadas em memória com os templates do projeto.
Não representam a captura original do relato, que não está disponível.
"""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from main import PokerAnalyticsApp
from src.config import SAMPLES_DIR
from src.vision import detector


class TestHeroRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranks, cls.suits = detector.load_templates()

    def setUp(self):
        # Impede gravação de recortes incertos; reconhecimento continua real.
        saver = patch.object(detector, "save_unknown_card")
        self.save_unknown = saver.start()
        self.addCleanup(saver.stop)

    def card(self, rank, suit, scale=1.0, visible_height=61):
        card = np.full((visible_height, 84, 3), 255, dtype=np.uint8)
        for glyph, top in [(self.ranks[rank], 4), (self.suits[suit], 36)]:
            glyph = cv2.resize(
                glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            top = round(top * scale)
            height = min(glyph.shape[0], visible_height - top)
            if height > 0:
                region = card[top:top + height, 6:6 + glyph.shape[1]]
                region[glyph[:height] > 127] = (40, 40, 230) if suit in "hd" else (30, 30, 30)
        return card

    def table(self, pairs):
        frame = np.full((1080, 1920, 3), (40, 110, 40), dtype=np.uint8)
        for x, y, cards in pairs:
            for index, card in enumerate(cards):
                left = x + index * 90
                frame[y:y + card.shape[0], left:left + card.shape[1]] = card
        return frame

    def recognize(self, frame, **kwargs):
        return detector.find_and_detect_hero_cards(frame, verbose=False, **kwargs)

    def test_red_suits_for_4_and_9_at_multiple_scales(self):
        for rank in ("4", "9"):
            for suit in ("h", "d"):
                for scale in (0.75, 1.0, 1.25):
                    with self.subTest(rank=rank, suit=suit, scale=scale):
                        card = self.card(rank, suit, scale, round(61 * scale))
                        frame = self.table([(1200, 650, [card, card])])
                        self.assertEqual(self.recognize(frame), [rank + suit] * 2)

    def test_mixed_suits_are_not_reported_as_suited(self):
        frame = self.table([(1200, 650, [self.card("4", "h"), self.card("9", "d")])])
        self.assertEqual(self.recognize(frame), ["4h", "9d"])

    def test_black_suits_remain_recognizable(self):
        for suit in ("c", "s"):
            with self.subTest(suit=suit):
                frame = self.table([(1200, 650, [self.card("4", suit), self.card("9", suit)])])
                self.assertEqual(self.recognize(frame), ["4" + suit, "9" + suit])

    def test_existing_samples_with_partially_covered_cards(self):
        for filename, expected in [
            ("mesa_turno.png", ["5h", "7d"]),
            ("mesa_par_de_6.png", ["6h", "6d"]),
            ("mesa_novas.png", ["Qd", "7h"]),
            ("image_230008.jpg", ["Qd", "Ks"]),
            ("image_23da1f.jpg", ["3s", "4s"]),
        ]:
            with self.subTest(filename=filename):
                self.assertEqual(self.recognize(SAMPLES_DIR / filename), expected)

    def test_known_seat_selects_hero_among_revealed_opponents(self):
        frame = self.table([
            (1200, 650, [self.card("4", "h"), self.card("9", "d")]),
            (400, 250, [self.card("9", "h"), self.card("4", "d")]),
        ])
        cards, coords = self.recognize(
            frame, hero_coords_hint={"left": 1200, "top": 650}, return_coords=True
        )
        self.assertEqual(cards, ["4h", "9d"])
        self.assertEqual(coords["left"], 1200)
        self.assertEqual(coords["top"], 650)

    def test_multiple_revealed_hands_without_seat_are_uncertain(self):
        frame = self.table([
            (1200, 650, [self.card("4", "h"), self.card("9", "d")]),
            (400, 250, [self.card("9", "h"), self.card("4", "d")]),
        ])
        self.assertEqual(self.recognize(frame, return_coords=True), ([], None))

    def test_unreadable_hero_does_not_select_nearby_opponent(self):
        blank = np.full((61, 84, 3), 255, dtype=np.uint8)
        frame = self.table([
            (1200, 650, [blank, blank]),
            (1000, 650, [self.card("9", "h"), self.card("4", "d")]),
        ])
        self.assertEqual(self.recognize(
            frame, hero_coords_hint={"left": 1200, "top": 650}, return_coords=True
        ), ([], None))

    def test_clipped_suit_is_uncertain(self):
        card = self.card("4", "h", visible_height=48)
        self.assertEqual(self.recognize(self.table([(1200, 650, [card, card])])), [])

    def test_heart_point_at_crop_boundary_is_recognizable(self):
        # Reproduz o recorte salvo às 13:21: o coração termina na última linha.
        card = self.card("5", "h", visible_height=57)
        self.assertEqual(self.recognize(self.table([(1200, 650, [card, card])])), ["5h", "5h"])

    def test_black_suit_connected_to_card_border_is_recognizable(self):
        # A linha cinza inferior entra na máscara (<185) e une o naipe à borda.
        card = self.card("K", "s", visible_height=57)
        card[-1, :] = (130, 130, 130)
        frame = self.table([(1200, 650, [card, self.card("5", "h")])])
        self.assertEqual(self.recognize(frame), ["Ks", "5h"])

    def test_missing_hero_does_not_select_nearby_opponent(self):
        frame = self.table([(1000, 650, [self.card("9", "h"), self.card("4", "d")])])
        self.assertEqual(self.recognize(
            frame,
            hero_coords_hint={"left": 1200, "top": 650, "width": 174, "height": 61},
            return_coords=True,
        ), ([], None))

    def test_board_is_not_selected_as_a_players_hand(self):
        frame = self.table([
            (750, 430, [self.card("9", "h"), self.card("4", "d")]),
            (1200, 650, [self.card("4", "h"), self.card("9", "d")]),
        ])
        self.assertEqual(self.recognize(frame), ["4h", "9d"])

    def test_process_frame_uses_known_hero_seat(self):
        frame = self.table([
            (1200, 650, [self.card("4", "h"), self.card("9", "d")]),
            (400, 250, [self.card("9", "h"), self.card("4", "d")]),
        ])
        # Não inicializa logs/arquivos. OCR e análise são fronteiras deste teste;
        # a localização e o reconhecimento visual permanecem reais.
        app = PokerAnalyticsApp.__new__(PokerAnalyticsApp)
        app.auto_detect_hero = True
        app.auto_detect_pot = False
        app.auto_detect_turn = False
        app.hero_cards = []
        app.hero_position = "Desconhecida"
        app.hero_coords = {"left": 1200, "top": 650, "width": 174, "height": 61}
        with patch("main.extract_hero_position", return_value="Desconhecida"), \
                patch("main.detect_board", return_value=[]), \
                patch.object(app, "process_cards", return_value={}) as process_cards:
            app.process_frame(frame, pot_size=50.0, bet_to_call=0.0, show_dashboard=False)
        self.assertEqual(app.hero_cards, ["4h", "9d"])
        self.assertEqual(app.hero_coords["left"], 1200)
        process_cards.assert_called_once_with([], 50.0, 0.0, False)

    def test_explicit_threshold_is_respected(self):
        frame = cv2.imread(str(SAMPLES_DIR / "mesa_turno.png"))
        self.assertEqual(self.recognize(frame, threshold=0.99), [])

    def test_dimmed_cards_remain_uncertain(self):
        self.assertEqual(self.recognize(SAMPLES_DIR / "mesa_6.png"), [])

    def test_invalid_image_preserves_coordinate_return_contract(self):
        self.assertEqual(self.recognize(
            np.zeros((100, 100, 3), dtype=np.uint8), return_coords=True
        ), ([], None))

    def test_missing_image_preserves_coordinate_return_contract(self):
        self.assertEqual(self.recognize(
            SAMPLES_DIR / "missing_hero_regression.png", return_coords=True
        ), ([], None))


if __name__ == "__main__":
    unittest.main()
