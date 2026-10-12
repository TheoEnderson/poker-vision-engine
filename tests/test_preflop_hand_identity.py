"""Continuidade pré-flop com a identidade explícita Atual/Anterior da mesa.

Nenhuma recomendação de FOLD é tratada como ação efetivamente executada.
Captura/OCR são controlados; FSM, caches, ranges e validação são reais.
"""

import unittest
from unittest.mock import patch

from tests import test_hand_cycle_integration as cycle


OLD = ["4h", "8d"]
NEW = ["Qs", "4c"]
IDENTITY = ("1425782190", "1425782000")
NEXT = ("1425782381", IDENTITY[0])
UNKNOWN = "Desconhecida"


def capture(hero=OLD, identity=IDENTITY, **kwargs):
    frame = cycle.capture(hero=hero, position=kwargs.pop("position", "BB"),
                          stack=kwargs.pop("stack", 1240), **kwargs)
    frame["identity"] = identity
    return frame


class TestPreflopHandIdentity(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp
    assert_continuity = cycle.TestHandCycleIntegration.assert_continuity
    assert_redeal = cycle.TestHandCycleIntegration.assert_redeal

    def live(self, frames, **kwargs):
        self.app.hero_cards = list(OLD)
        with patch("main.extract_hand_identity",
                   side_effect=lambda frame, **kw: frame.get("identity")):
            return cycle.TestHandCycleIntegration.live(self, frames, **kwargs)

    def test_real_context_preflop_redeal_without_observed_zero_or_dealer_change(self):
        new = capture(NEW, NEXT, pot=15, call=10)
        self.live([capture(pot=120, call=60), new, new])
        self.assert_redeal(NEW)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(len(self.analyses), 2)
        self.assertEqual(self.analyses[-1]["pot_size"], 15)
        self.assertEqual(self.analyses[-1]["bet_to_call"], 10)

    def test_after_possible_fold_without_observing_flop_or_pot_zero(self):
        new = capture(NEW, NEXT, pot=15, call=10, position=UNKNOWN)
        self.live([capture(pot=120, call=60, position=UNKNOWN),
                   capture(hero=[], turn=False, pot=120, position=UNKNOWN), new, new])
        self.assert_redeal(NEW)

    def test_same_hole_cards_can_be_redealt(self):
        new = capture(identity=NEXT, pot=15, call=10)
        self.live([capture(pot=120), new, new])
        self.assert_redeal(OLD)
        self.assertEqual(len(self.analyses), 2)

    def test_no_pot_drop_required_for_explicit_redeal(self):
        new = capture(NEW, NEXT, pot=150)
        self.live([capture(pot=120), new, new])
        self.assert_redeal(NEW)

    def test_temporarily_absent_hero_does_not_confirm_candidate(self):
        new = capture(NEW, NEXT, pot=15)
        self.live([capture(), new, capture(hero=[], identity=NEXT, turn=False), new])
        self.assert_redeal(NEW)
        self.assertEqual(len(self.analyses), 2)
        self.assertEqual(self.snapshots[3]["generation"], 0)

    def test_missing_identity_does_not_confirm_candidate(self):
        new = capture(NEW, NEXT, pot=15)
        self.live([capture(), new, capture(NEW, None, pot=15), new])
        self.assert_redeal(NEW)
        self.assertEqual(len(self.analyses), 2)
        self.assertEqual(self.snapshots[3]["generation"], 0)

    def test_wrong_hero_without_new_identity_remains_uncertain(self):
        wrong = capture(NEW, pot=15)
        output = self.live([capture(pot=120), wrong, wrong, capture(pot=120)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)
        self.assertIs(self.analyses[0], self.analyses[1])

    def test_pot_ocr_drop_alone_preserves_hand_and_equity_cache(self):
        self.live([capture(pot=120), capture(pot=15), capture(pot=120)])
        self.assert_continuity()
        self.assertEqual(self.risk.call_count, 3)
        self.assertTrue(self.analyses[-1]["equity_cached"])

    def test_temporary_hero_loss_and_recovery_preserve_hand(self):
        self.live([capture(pot=120), capture(hero=[], pot=15, turn=False), capture(pot=120)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_one_false_identity_is_refuted_by_known_hand(self):
        new = capture(NEW, NEXT)
        self.live([capture(), new, capture(), new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_unlinked_identity_cannot_confirm_redeal(self):
        wrong = capture(NEW, (NEXT[0], "999"), pot=15)
        self.live([capture(pot=120), wrong, wrong])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_changed_identity_with_same_cards_is_not_current_analysis(self):
        wrong = capture(identity=(NEXT[0], "999"), pot=15)
        self.live([capture(pot=120), wrong, wrong])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_unlinked_identity_then_missing_ocr_does_not_revalidate_old_analysis(self):
        wrong = capture(identity=(NEXT[0], "999"), pot=15)
        self.live([capture(pot=120), wrong, capture(identity=None, pot=15), capture(pot=120)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        self.assertIs(self.analyses[0], self.analyses[-1])

    def test_new_identity_with_absent_hero_then_missing_id_does_not_revalidate_old_analysis(self):
        self.live([capture(pot=120), capture(hero=[], identity=NEXT, turn=False),
                   capture(identity=None, pot=15), capture(pot=120)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_financial_recovery_uses_only_new_capture(self):
        new = capture(NEW, NEXT, pot=None)
        self.live([capture(pot=120), new, new, capture(NEW, NEXT, pot=15, call=10)])
        self.assert_redeal(NEW)
        valid = [a for a in self.analyses if a.get("observation_valid") is not False]
        self.assertEqual([a["pot_size"] for a in valid], [120, 15])
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_ranges_and_both_caches_reset_once_only_after_confirmation(self):
        new = capture(NEW, NEXT, pot=15)
        self.live([capture(), new, new, new])
        self.assert_redeal(NEW)
        before, pending, confirmed = self.snapshots[1:]
        self.assertEqual(before["weights"], pending["weights"])
        self.assertIs(before["cache"], pending["cache"])
        self.assertEqual(before["equity_key"], pending["equity_key"])
        self.assertNotEqual(pending["equity_key"], confirmed["equity_key"])
        self.assertIs(self.analyses[-1], self.analyses[-2])
        self.assertEqual(self.risk.call_count, 2)

    def test_new_hand_can_be_seen_directly_on_flop(self):
        flop = ["9c", "3h", "4s"]
        new = capture(NEW, NEXT, board=flop, pot=45, position=UNKNOWN)
        self.live([capture(position=UNKNOWN), new, new])
        self.assert_redeal(NEW, flop)

    def test_identity_recovers_same_hand_after_absent_hero_and_unreadable_dealer(self):
        self.live([capture(position=UNKNOWN),
                   capture(hero=[], turn=False, position=UNKNOWN), capture(position=UNKNOWN)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_legitimate_dealer_change_can_still_confirm_without_identity(self):
        new = capture(NEW, None, position="CO", pot=15)
        self.live([capture(identity=None), new, new])
        self.assert_redeal(NEW)

    def test_single_dealer_false_positive_does_not_reset(self):
        self.live([capture(), capture(position="CO"), capture()])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_repeated_dealer_and_card_false_positives_with_same_identity_do_not_reset(self):
        wrong = capture(NEW, position="CO", pot=15)
        self.live([capture(pot=120), wrong, wrong, capture(pot=120)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_changed_identity_refutes_old_candidate_even_when_hero_unreadable(self):
        new = capture(NEW, NEXT)
        self.live([capture(), new, capture(hero=[], turn=False), new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_same_cards_with_missing_candidate_identity_suspend_old_recommendation(self):
        new = capture(identity=NEXT, pot=15)
        self.live([capture(pot=120), new, capture(identity=None, pot=15), new])
        self.assert_redeal(OLD)
        self.assertEqual(len(self.analyses), 2)

    def test_explicit_new_hand_does_not_override_duplicate_card_guard(self):
        wrong = capture(["Qs", "Qs"], NEXT, pot=15)
        self.live([capture(pot=120), wrong, wrong])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_missing_all_independent_evidence_remains_suspended(self):
        new = capture(NEW, None, position=UNKNOWN, pot=15)
        self.live([capture(identity=None, position=UNKNOWN, pot=120), new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_several_preflop_hands_without_observed_gap(self):
        third_id = ("1425782500", NEXT[0])
        second = capture(NEW, NEXT, pot=15)
        third = capture(identity=third_id, pot=20)
        self.live([capture(pot=120), second, second, third, third])
        self.assertEqual(self.app.state_machine.hand_generation, 2)
        self.assertEqual(self.reset_count, 2)
        self.assertEqual(self.calculate_equity.call_count, 3)
        self.assertEqual(self.app.hero_cards, OLD)


if __name__ == "__main__":
    unittest.main()
