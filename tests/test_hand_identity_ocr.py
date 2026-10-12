"""Regressão da captura 11/10/2026 23:54:56, sem dados pessoais.

Fixtures PNG/base64 preservam somente os pixels do painel Atual/Anterior
([147:194, 0:200]) e cartas Qs/4c ([424:482, 1461:1629]). Nada de nomes,
avatares completos, terminal ou desktop. Não simulam frames ausentes do log.
"""

import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr import hand_identity as ocr
from src.vision.detector import find_and_detect_hero_cards


TEXT = "Atual: #1425782381\nAnterior: #1425782190\n"
IDENTITY = ("1425782381", "1425782190")


def fixture(name):
    path = Path(__file__).parent / "fixtures" / (name + ".png.b64")
    return cv2.imdecode(np.frombuffer(base64.b64decode(path.read_bytes()), dtype=np.uint8), cv2.IMREAD_COLOR)


def identity_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[147:194, 0:200] = fixture("replay_hand_identity")
    return frame


class TestHandIdentityOCR(unittest.TestCase):
    def test_real_capture_native_and_enlarged_text_agree(self):
        diagnostics = {}
        self.assertEqual(ocr.extract_hand_identity(identity_frame(), diagnostics), IDENTITY)
        self.assertEqual([r["identity"] for r in diagnostics["readings"]], [IDENTITY, IDENTITY])
        self.assertEqual(diagnostics["status"], "CONFIRMED")

    def test_exact_labeled_pair_in_portuguese_and_english(self):
        for text in (TEXT, "Current: #1425782381\nPrevious: #1425782190"):
            with self.subTest(text=text):
                self.assertEqual(ocr._parse_hand_identity(text), IDENTITY)

    def test_partial_malformed_unlabeled_and_reversed_pairs_rejected(self):
        for text in (None, "", "Atual: #1425782381", "Atual: 41425782381\nAnterior: #1425782190",
                     "Atual: #1425782381\nAnterlor: #1425782190", "1425782381\n1425782190",
                     "Atual: #14?5782381\nAnterior: #1425782190",
                     "Atual: #10\nAnterior: #11", "Atual: #10\nAnterior: #10",
                     "Atual: #10\nAnterior: #0", TEXT + "lixo"):
            with self.subTest(text=text):
                self.assertIsNone(ocr._parse_hand_identity(text))

    def test_methods_disagree_do_not_vote_or_reconstruct_id(self):
        for second in (None, "", TEXT.replace("1425782381", "142578238")):
            with self.subTest(second=second), patch.object(ocr.pytesseract, "image_to_string", side_effect=[TEXT, second]):
                self.assertIsNone(ocr.extract_hand_identity(identity_frame()))

    def test_missing_native_read_is_not_completed_from_history(self):
        with patch.object(ocr.pytesseract, "image_to_string", return_value="") as reader:
            self.assertIsNone(ocr.extract_hand_identity(identity_frame()))
            reader.assert_called_once()

    def test_recovery_has_no_persistent_ocr_state(self):
        with patch.object(ocr.pytesseract, "image_to_string", side_effect=["", TEXT, TEXT, ""]):
            self.assertIsNone(ocr.extract_hand_identity(identity_frame()))
            self.assertEqual(ocr.extract_hand_identity(identity_frame()), IDENTITY)
            self.assertIsNone(ocr.extract_hand_identity(identity_frame()))

    def test_other_resolution_and_missing_input_are_not_guessed(self):
        for frame in (None, {}, np.zeros((720, 1280, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(frame, "shape", None)), patch.object(ocr.pytesseract, "image_to_string") as reader:
                self.assertIsNone(ocr.extract_hand_identity(frame))
                reader.assert_not_called()

    def test_real_hero_pixels_recognizable_with_and_without_old_seat_hint(self):
        frame = np.full((1080, 1920, 3), (40, 110, 40), dtype=np.uint8)
        frame[424:482, 1461:1629] = fixture("replay_hero_qs_4c")
        for hint in (None, dict(left=1461, top=424, width=168, height=58)):
            with self.subTest(hint=hint):
                cards, coords = find_and_detect_hero_cards(frame, verbose=False, return_coords=True,
                                                         hero_coords_hint=hint, save_unknown=False)
                self.assertEqual(cards, ["Qs", "4c"])
                self.assertEqual(coords["left"], 1461)
                self.assertLess(coords["height"], 120)


if __name__ == "__main__":
    unittest.main()
