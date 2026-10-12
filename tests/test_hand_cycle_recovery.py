"""4F-A.1: recuperação e diagnóstico sem presumir execução de FOLD."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from src.ocr import ocr_reader
from src.vision import detector

from tests import test_hand_cycle_integration as cycle


UNKNOWN = "Desconhecida"
capture = cycle.capture
HERO, OTHER, FLOP = cycle.HERO, cycle.OTHER, cycle.FLOP


class TestHandCycleRecovery(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp
    live = cycle.TestHandCycleIntegration.live
    assert_continuity = cycle.TestHandCycleIntegration.assert_continuity
    assert_redeal = cycle.TestHandCycleIntegration.assert_redeal

    def test_missing_hero_with_active_buttons_does_not_latch_a_boundary(self):
        self.live([capture(position=UNKNOWN), capture(hero=[], position=UNKNOWN),
                   capture(pot=25, call=10, position=UNKNOWN)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        self.assertTrue(self.analyses[-1]["action_available"])
        self.assertEqual(self.analyses[-1]["bet_to_call"], 10)
        self.assertEqual(self.risk.call_count, 2)

    def test_absence_incomplete_board_and_button_return_recovers_current_hand(self):
        self.live([capture(position=UNKNOWN),
                   capture(hero=[], board=[UNKNOWN] * 3, turn=False, position=UNKNOWN),
                   capture(pot=25, call=10, position=UNKNOWN)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        self.assertTrue(self.analyses[-1]["equity_cached"])

    def test_temporary_unreadable_hero_in_turn_does_not_break_hand_and_recovers_action(self):
        c1 = capture(pot=6, call=0, turn=False, position=UNKNOWN)
        c2 = capture(pot=970, call=970, stack=970, actions=["FOLD", "ALL-IN"], position=UNKNOWN)
        c2_context = dict(is_hero_turn=True, action_type="CALL", all_in_nature="CALL",
                          bet_to_call=970, available_actions=["FOLD", "ALL-IN"])
        c3 = capture(hero=[], pot=970, call=970, stack=970, actions=["FOLD", "ALL-IN"], position=UNKNOWN)
        c4 = capture(pot=970, call=970, stack=970, actions=["FOLD", "ALL-IN"], position=UNKNOWN)

        with patch("main.detect_turn_and_bet_to_call",
                   side_effect=lambda f, **kw: c2_context if f.get("actions") == ["FOLD", "ALL-IN"] else cycle.context(f)):
            self.live([c1, c2, c3, c4])

        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        for analysis in self.analyses:
            self.assertTrue(analysis["action_available"])
            self.assertEqual(analysis["action"], "FOLD")
            self.assertEqual(analysis["bet_to_call"], 970)

    def test_preflop_fold_then_public_board_and_new_deal_without_zero_or_dealer(self):
        new = capture(hero=OTHER, pot=15, position=UNKNOWN)
        away = capture(hero=[], board=FLOP, pot=25, turn=False, position=UNKNOWN)
        self.live([capture(pot=25, call=10, position=UNKNOWN), away, away, new, new])
        self.assert_redeal(OTHER)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(self.app.hero_position, UNKNOWN)

    def test_identical_cards_can_be_redealt_after_public_board_witness(self):
        away = capture(hero=[], board=FLOP, pot=25, turn=False, position=UNKNOWN)
        new = capture(pot=15, position=UNKNOWN)
        self.live([capture(position=UNKNOWN), away, away, new, new])
        self.assert_redeal(HERO)

    def test_one_false_public_board_does_not_confirm_redeal(self):
        away = capture(hero=[], board=FLOP, pot=25, turn=False, position=UNKNOWN)
        new = capture(hero=OTHER, pot=15, position=UNKNOWN)
        self.live([capture(position=UNKNOWN), away, new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_public_board_evidence_expires_when_same_hand_recovers(self):
        away = capture(hero=[], board=FLOP, turn=False, position=UNKNOWN)
        recovered = capture(board=FLOP, position=UNKNOWN)
        self.live([capture(position=UNKNOWN), away, away, recovered,
                   capture(position=UNKNOWN), capture(position=UNKNOWN), recovered])
        self.assert_continuity(FLOP, simulations=2)

    def test_public_board_conflicting_with_locked_hero_is_not_evidence(self):
        away = capture(hero=[], board=[HERO[0], "2d", "3s"], turn=False, position=UNKNOWN)
        new = capture(hero=OTHER, position=UNKNOWN)
        self.live([capture(position=UNKNOWN), away, away, new, new])
        self.assert_continuity()

    def test_candidate_survives_unreadable_frame_without_gaining_support(self):
        new = capture(hero=OTHER, pot=15, position="CO")
        unreadable = capture(hero=[], board=[UNKNOWN] * 3, turn=False, position=UNKNOWN)
        self.live([capture(), new, unreadable, new])
        self.assert_redeal(OTHER)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_candidate_survives_temporarily_unknown_dealer(self):
        new = capture(hero=OTHER, position="CO")
        self.live([capture(), new, dict(new, position=UNKNOWN), new])
        self.assert_redeal(OTHER)

    def test_known_old_hero_refutes_pending_candidate(self):
        new = capture(hero=OTHER, position="CO")
        self.live([capture(), new, capture(), new])
        self.assert_continuity()

    def test_false_hero_twice_without_independent_evidence_does_not_reset(self):
        wrong = capture(hero=["Kd", "4d"], position=UNKNOWN)
        self.live([capture(position=UNKNOWN), wrong, wrong, capture(position=UNKNOWN)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_indistinguishable_preflop_hands_remain_suspended_with_missing_evidence(self):
        new = capture(hero=OTHER, pot=15, position=UNKNOWN)
        output = self.live([capture(position=UNKNOWN),
                            capture(hero=[], turn=False, position=UNKNOWN), new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)
        self.assertIn("Faltam:", output)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)

    def test_same_cards_after_ambiguous_wait_remain_suspended(self):
        self.live([capture(position=UNKNOWN), capture(hero=[], turn=False, position=UNKNOWN),
                   capture(position=UNKNOWN), capture(position=UNKNOWN)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_five_preflop_hands_without_zero_or_dealer_and_with_incomplete_reads(self):
        hands = [HERO, OTHER, ["Jh", "6d"], ["7s", "7d"], ["Kh", "Qh"]]
        public = ["2c", "3d", "5s"]
        frames = [capture(position=UNKNOWN)]
        for hand in hands[1:]:
            away = capture(hero=[], board=public, pot=25, turn=False, position=UNKNOWN)
            new = capture(hero=hand, pot=15, position=UNKNOWN)
            frames += [away, away, new,
                       capture(hero=[], board=[UNKNOWN] * 3, turn=False, position=UNKNOWN), new]
        self.live(frames)
        self.assertEqual(self.app.state_machine.hand_generation, 4)
        self.assertEqual(self.reset_count, 4)
        self.assertEqual(self.calculate_equity.call_count, 5)
        self.assertEqual(self.app.hero_cards, hands[-1])

    def test_equivalent_real_session_then_recovery_and_new_distribution(self):
        first = ["Jh", "6d"]
        second = ["4h", "10h"]
        self.app.hero_cards = list(first)
        flop = ["3c", "10s", "Qc"]
        river = flop + ["4h", "8d"]
        frames = [capture(hero=first, board=b, pot=p, position=UNKNOWN)
                  for b, p in (([], 10), (flop, 25), (river[:4], 50), (river, 100))]
        deal = capture(hero=second, pot=10, call=2, stack=808, position=UNKNOWN)
        away = capture(hero=[], board=["2c", "3d", "5s"], pot=25, turn=False, position=UNKNOWN)
        third = capture(hero=OTHER, pot=15, position=UNKNOWN)
        frames += [deal, deal, capture(hero=[], position=UNKNOWN),
                   dict(deal, pot=14, call=4), third, third, away, away, third, third]
        self.live(frames)
        self.assertEqual(self.app.state_machine.hand_generation, 2)
        self.assertEqual(self.calculate_equity.call_count, 6)
        self.assertEqual(self.analyses[4]["action"], "FOLD")
        self.assertEqual(self.analyses[5]["hero_cards"], second)
        self.assertEqual(self.analyses[5]["bet_to_call"], 4)
        self.assertEqual(self.analyses[-1]["hero_cards"], OTHER)

    def test_repeated_uncertain_messages_are_consolidated(self):
        bad = capture(hero=OTHER, position=UNKNOWN)
        output = self.live([capture(position=UNKNOWN)] + [bad] * 30)
        self.assert_continuity()
        self.assertLessEqual(output.count("CAPTURA INCERTA -"), 2)
        self.assertIn("30", output)

    def test_debug_has_complete_structured_trace_for_every_capture(self):
        frames = [capture(position=UNKNOWN), capture(hero=[], position=UNKNOWN),
                  capture(pot=25, call=10, position=UNKNOWN)]
        output = self.live(frames, debug=True)
        logs = [json.loads(line.split("[DEBUG MÃO] ", 1)[1])
                for line in output.splitlines() if "[DEBUG MÃO] " in line]
        self.assertEqual([log["captura_id"] for log in logs], [1, 2, 3])
        for log in logs:
            self.assertTrue(log["timestamp"])
            for key in ("estado_anterior", "hero_confirmado", "hero_observado", "board_anterior",
                        "board_observado", "pote_observado", "call_observado", "botoes_disponiveis",
                        "posicao", "dealer", "adversarios_observados", "estado_espera", "candidato",
                        "criterios", "evidencias_ausentes", "motivo", "capturas_compativeis"):
                self.assertIn(key, log)
        self.assertIsNone(logs[0]["dealer"]["confidence"])
        self.assertEqual(logs[1]["estado_espera"], "SUSPENDED")
        self.assertEqual(logs[-1]["call_observado"], 10)

    def test_incomplete_candidate_keeps_count_without_reinforcing_it(self):
        new = capture(hero=OTHER, position="CO")
        partial = capture(hero=[], board=[UNKNOWN] * 3, turn=False, position=UNKNOWN)
        output = self.live([capture(), new, partial, new], debug=True)
        logs = [json.loads(line.split("[DEBUG MÃO] ", 1)[1])
                for line in output.splitlines() if "[DEBUG MÃO] " in line]
        self.assertEqual([log["capturas_compativeis"] for log in logs], [0, 1, 1, 2])
        self.assertEqual(logs[-1]["resultado"], "NEW_HAND")
        self.assertEqual(logs[-1]["hero_confirmado"], HERO)

    def test_varying_active_opponents_do_not_change_layout_or_hand_identity(self):
        self.live([dict(capture(position=UNKNOWN), opponents=3),
                   dict(capture(hero=[], position=UNKNOWN), opponents=1),
                   dict(capture(pot=25, call=10, position=UNKNOWN), opponents=1)])
        self.assert_continuity(simulations=2)
        self.assertEqual(self.analyses[-1]["num_opponents"], 1)
        self.assertEqual(self.app.hero_position, UNKNOWN)

    def test_financial_recovery_uses_current_values_not_cached_values(self):
        self.live([capture(pot=10, call=2, stack=808, position=UNKNOWN),
                   capture(hero=[], position=UNKNOWN),
                   capture(pot=14, call=4, stack=1, position=UNKNOWN),
                   capture(pot=18, call=4, stack=800, position=UNKNOWN)])
        self.assert_continuity()
        self.assertFalse(self.analyses[1]["observation_valid"])
        self.assertEqual(self.risk.call_count, 2)
        last = self.analyses[-1]
        self.assertEqual((last["pot_size"], last["bet_to_call"], last["hero_stack"]), (18, 4, 800))
        self.assertAlmostEqual(last["ev"], .41 * 22 - 4)


class TestNoPersistentCaptureDiagnostics(unittest.TestCase):
    def test_unknown_board_remains_unknown_without_saving_crop(self):
        blank_card = np.full((100, 90, 3), 255, dtype=np.uint8)
        with patch.object(detector, "save_unknown_card") as save:
            cards = detector.detect_board(blank_card, verbose=False, save_unknown=False)
        self.assertEqual(cards, [UNKNOWN, "Vazio", "Vazio", "Vazio", "Vazio"])
        save.assert_not_called()

    def test_unknown_hero_remains_unknown_without_saving_crop(self):
        frame = np.full((1080, 1920, 3), (40, 110, 40), dtype=np.uint8)
        frame[650:711, 1200:1284] = 255
        frame[650:711, 1290:1374] = 255
        with patch.object(detector, "save_unknown_card") as save:
            cards = detector.find_and_detect_hero_cards(
                frame, verbose=False, save_unknown=False,
                hero_coords_hint={"left": 1200, "top": 650}, return_coords=True)
        self.assertEqual(cards, ([], None))
        save.assert_not_called()


class TestDealerDiagnostics(unittest.TestCase):
    def position(self, score, hero=True, diagnostics=None):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(ocr_reader, "_resolve_image_input", return_value=image), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("cv2.imread", return_value=image[:4, :4]), \
             patch("cv2.matchTemplate", return_value=np.array([[score]], dtype=np.float32)), \
             patch.object(ocr_reader, "last_hero_name_coords", (70, 45, 80, 55) if hero else None), \
             patch("pytesseract.image_to_data", return_value={"text": []}):
            return ocr_reader.extract_hero_position(image, diagnostics=diagnostics)

    def test_unknown_dealer_exposes_score_without_guessing_position(self):
        diagnostic = {}
        self.assertEqual(self.position(.2, diagnostics=diagnostic), UNKNOWN)
        self.assertAlmostEqual(diagnostic["confidence"], .2)
        self.assertIn("abaixo", diagnostic["reason"])

    def test_dealer_found_without_hero_anchor_does_not_claim_known_position(self):
        diagnostic = {}
        self.assertEqual(self.position(.9, hero=False, diagnostics=diagnostic), UNKNOWN)
        self.assertAlmostEqual(diagnostic["confidence"], .9)
        self.assertIn("assento", diagnostic["reason"])

    def test_diagnostics_preserve_legacy_classification(self):
        diagnostic = {}
        self.assertEqual(self.position(.9), self.position(.9, diagnostics=diagnostic))
        self.assertEqual(diagnostic["layout"], "6-max")

    def test_dealer_score_between_052_and_060_is_recognized(self):
        diagnostic = {}
        pos = self.position(.55, hero=True, diagnostics=diagnostic)
        self.assertNotEqual(pos, UNKNOWN)
        self.assertAlmostEqual(diagnostic["confidence"], .55)
        self.assertIn("6-max", diagnostic["reason"])

    def test_dealer_score_below_threshold_is_rejected(self):
        diagnostic = {}
        pos = self.position(.50, hero=True, diagnostics=diagnostic)
        self.assertEqual(pos, UNKNOWN)
        self.assertAlmostEqual(diagnostic["confidence"], .50)
        self.assertIn("abaixo", diagnostic["reason"])


if __name__ == "__main__":
    unittest.main()
