"""Recortes reais 12/10 13:20: cartas, D, IDs e botões; sem nomes/terminal.

O terminal do teste é uma sobreposição sintética nas coordenadas observadas.
"""
import base64
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.ocr.ocr_reader import extract_hero_position, detect_turn_and_bet_to_call
from src.ocr.hand_identity import extract_hand_identity, _parse_hand_identity
from tests import test_hand_cycle_integration as cycle
from tests.test_same_street_decisions import HERO, FLOP, TURN


def frame():
    raw = (Path(__file__).parent/'fixtures/dealer_session_6c_5h.png.b64').read_bytes()
    return cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)


class TestDealerPixels(unittest.TestCase):
    def position(self, image):
        diagnostics = {}
        position = extract_hero_position(image, hero_coords=dict(left=1461, top=424, width=168, height=58),
                                         diagnostics=diagnostics)
        return position, diagnostics

    def test_real_dealer_instead_of_spade_or_club(self):
        pos, d = self.position(frame())
        self.assertEqual(pos, 'MP')  # Geometria 6-max existente: assento 2 -> assento 0.
        self.assertEqual(d['dealer_center'], [730, 698])
        self.assertEqual(d['dealer_seat'], 'Bot Left')
        self.assertEqual(d['hero_seat'], 'Mid Right')
        self.assertTrue(d['validated'])

    def test_cards_without_dealer_do_not_produce_position(self):
        image = frame(); image[684:712, 716:744] = (40,110,40)
        self.assertEqual(self.position(image)[0], 'Desconhecida')

    def test_real_dealer_at_hero_seat_is_btn(self):
        image = frame(); badge = image[684:712,716:744].copy()
        image[684:712,716:744] = (40,110,40)
        image[455:483,1420:1448] = badge
        self.assertEqual(self.position(image)[0], 'BTN')

    def test_terminal_overlay_does_not_change_visible_dealer(self):
        image = frame(); image[595:1080,15:673] = (48,48,48)
        cv2.putText(image, 'BTN CHECK CALL', (25,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        self.assertEqual(self.position(image)[0], 'MP')
        image[680:720,710:750] = (48,48,48)
        self.assertEqual(self.position(image)[0], 'Desconhecida')

    def test_multiple_visible_badges_remain_ambiguous(self):
        image = frame(); image[455:483,1420:1448] = image[684:712,716:744]
        self.assertEqual(self.position(image)[0], 'Desconhecida')

    def test_real_english_ids_and_actions(self):
        self.assertEqual(extract_hand_identity(frame()), ('1425916397','1425916110'))
        info = detect_turn_and_bet_to_call(frame(), verbose=False)
        self.assertEqual(info['action_type'], 'CHECK')
        self.assertEqual(info['available_actions'], ['FOLD','CHECK','BET'])
        self.assertEqual(info['bet_to_call'], 0)
        self.assertEqual(_parse_hand_identity('Atual: #1425916397\nAnterior: #1425916110'),
                         extract_hand_identity(frame()))


class TestPositionContinuity(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp

    def live(self, frames, validated=True):
        with patch('main.extract_hand_identity', side_effect=lambda f, **kw: f.get('identity')):
            # O harness controla os pixels. Injeta evidência validada pelo
            # detector sem transformar repetição de posições em validação.
            observe = self.app.observe_hand
            def observation(*args, **kw):
                kw['dealer_diagnostics'] = dict(validated=validated, dealer_seat='Bot Left',
                                                position_basis='geometria 6-max; ocupação não confirmada')
                return observe(*args, **kw)
            with patch.object(self.app, 'observe_hand', side_effect=observation):
                return cycle.TestHandCycleIntegration.live(self, frames)

    def capture(self, position='BTN', identity=('20','10'), **kw):
        return dict(cycle.capture(position=position, **kw), identity=identity)

    def test_consistent_identity_recovers_wrong_reference_without_reset(self):
        self.live([self.capture(), self.capture(position='MP', board=FLOP),
                   self.capture(position='MP', board=TURN), self.capture(position='MP', board=TURN)])
        self.assertEqual(self.reset_count, 0)
        self.assertEqual(self.app._hand_identity, ('20','10'))
        self.assertEqual(self.app.hero_position, 'MP')
        self.assertEqual(self.analyses[1]['hero_position'], 'Desconhecida')
        self.assertFalse(self.analyses[1]['action_available'])
        self.assertIn('posição', self.analyses[1]['action_status'])
        self.assertTrue(self.analyses[2]['position_valid'])
        self.assertIs(self.analyses[2], self.analyses[3])
        self.assertEqual(self.calculate_equity.call_count, 3)

    def test_oscillation_with_id_cannot_confirm_position_or_reset(self):
        self.live([self.capture(), self.capture(position='MP'), self.capture(position='CO'),
                   self.capture(position='MP'), self.capture()])
        self.assertEqual(self.reset_count, 0)
        for analysis in self.analyses[1:-1]:
            self.assertFalse(analysis['action_available'])
        self.assertEqual(self.app.hero_position, 'BTN')
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_without_id_divergence_stays_suspended_but_board_advances(self):
        output = self.live([self.capture(identity=None),
                            self.capture('MP', None, board=FLOP), self.capture('MP', None, board=TURN)])
        self.assertEqual(self.reset_count, 0)
        self.assertEqual(self.app.state_machine.board, TURN)
        self.assertEqual(len(self.analyses), 1)
        self.assertEqual(self.app.hero_position, 'Desconhecida')
        self.assertIn('posição', output)

    def test_linked_new_identity_still_resets_caches_and_ranges(self):
        new = self.capture('MP', ('30','20'))
        self.live([self.capture(), new, new])
        self.assertEqual(self.reset_count, 1)
        self.assertEqual(self.app._hand_identity, ('30','20'))
        self.assertEqual(self.app.hero_position, 'MP')
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_new_hand_clears_unresolved_position_conflict(self):
        new = self.capture('CO', ('30','20'))
        self.live([self.capture(), self.capture('MP'), new, new])
        self.assertEqual(self.reset_count, 1)
        self.assertFalse(self.app._position_uncertain)
        self.assertIsNone(self.app._position_candidate)
        self.assertEqual(self.app._position_candidate_count, 0)
        self.assertTrue(self.analyses[-1]['position_valid'])
        self.assertEqual(self.analyses[-1]['hero_position'], 'CO')
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_position_recovery_does_not_bypass_financial_validation(self):
        self.live([self.capture(), self.capture('MP', pot=None), self.capture('MP', pot=None),
                   self.capture('MP', pot=30)])
        self.assertEqual(self.app.hero_position, 'MP')
        self.assertFalse(self.analyses[1]['action_available'])
        self.assertFalse(self.analyses[2]['action_available'])
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_repeated_unvalidated_dealer_does_not_confirm_position(self):
        self.live([self.capture(), self.capture('MP'), self.capture('MP'), self.capture('MP')], validated=False)
        self.assertEqual(self.app.hero_position, 'Desconhecida')
        self.assertEqual(self.app._hand_position, 'BTN')
        self.assertEqual(self.reset_count, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)
        for analysis in self.analyses[1:]:
            self.assertFalse(analysis['position_valid'])
            self.assertFalse(analysis['action_available'])

    def test_position_repair_preserves_equity_and_ranges_but_updates_decision(self):
        output = self.live([self.capture(), self.capture('MP'), self.capture('MP'), self.capture('MP')])
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(self.risk.call_count, 3)
        self.assertEqual(self.snapshots[1]['weights'], self.model.weights)
        self.assertEqual(self.reset_count, 0)
        self.assertTrue(self.analyses[-1]['equity_cached'])
        self.assertIs(self.analyses[-1], self.analyses[-2])
        self.assertIn('MP (estimativa 6-max)', output)
