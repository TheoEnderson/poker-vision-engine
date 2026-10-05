"""
src/engine/state_machine.py
Máquina de Estados Finitos (FSM) para controle de ciclo de vida do jogo no Texas Hold'em.
"""

import datetime
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permite execução direta como script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class HandState(str, Enum):
    """
    Estados possíveis da mão de poker no Texas Hold'em.
    Conforme modelado no Documento de Arquitetura do Poker Analytics Engine (PAE).
    """
    WAITING_HAND = "WAITING_HAND"
    PRE_FLOP = "PRE_FLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"
    SHOWDOWN = "SHOWDOWN"


class PokerHandStateMachine:
    """
    Máquina de Estados Finitos (FSM) que controla o ciclo de vida sequencial
    da mão de poker, garantindo que não ocorram transições ilegais ou dados dessincronizados.
    """

    def __init__(self, initial_state: HandState = HandState.WAITING_HAND, allow_initial_sync: bool = True):
        self.current_state: HandState = initial_state
        self.board: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.allow_initial_sync: bool = allow_initial_sync
        self._initial_synced: bool = False
        self._record_transition(None, self.current_state, "Inicialização da máquina de estados")

    def _record_transition(self, from_state: Optional[HandState], to_state: HandState, reason: str):
        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "board": list(self.board),
            "reason": reason,
        }
        self.history.append(record)

    @staticmethod
    def filter_valid_cards(raw_cards: List[str]) -> List[str]:
        """Filtra apenas cartas comunitárias com identificação válida (exclui 'Vazio' e 'Desconhecida')."""
        return [
            card
            for card in raw_cards
            if card and card.lower() not in ("vazio", "desconhecida", "none", "--")
        ]

    def _is_transition_allowed(self, from_state: HandState, to_state: HandState, num_cards: int) -> bool:
        if from_state == to_state:
            return True

        if not self._initial_synced and self.allow_initial_sync and from_state == HandState.WAITING_HAND:
            if to_state in (HandState.PRE_FLOP, HandState.FLOP, HandState.TURN, HandState.RIVER):
                return True

        if to_state in (HandState.WAITING_HAND, HandState.PRE_FLOP) and num_cards == 0:
            return True

        if from_state == HandState.WAITING_HAND and to_state == HandState.PRE_FLOP:
            return num_cards == 0

        if from_state == HandState.PRE_FLOP and to_state == HandState.FLOP:
            return num_cards == 3

        if from_state == HandState.FLOP and to_state == HandState.TURN:
            return num_cards == 4

        if from_state == HandState.TURN and to_state == HandState.RIVER:
            return num_cards == 5

        if from_state == HandState.RIVER and to_state == HandState.SHOWDOWN:
            return num_cards == 5

        return False

    def update_board(self, detected_cards: List[str]) -> bool:
        valid_cards = self.filter_valid_cards(detected_cards)
        num_cards = len(valid_cards)

        if num_cards == 0:
            target_state = HandState.PRE_FLOP
        elif num_cards == 3:
            target_state = HandState.FLOP
        elif num_cards == 4:
            target_state = HandState.TURN
        elif num_cards == 5:
            target_state = HandState.RIVER
        else:
            print(
                f"[AVISO] Quantidade inconsistente de cartas comunitárias ({num_cards}): "
                f"{valid_cards}. Nenhuma transição realizada."
            )
            return False

        if target_state == self.current_state and valid_cards == self.board:
            return True

        if not self._is_transition_allowed(self.current_state, target_state, num_cards):
            if self.current_state == HandState.WAITING_HAND and target_state in (HandState.FLOP, HandState.TURN, HandState.RIVER):
                pass # Bot está apenas aguardando a mão atual (onde ele foldou) terminar na tela
            else:
                print(
                    f"[ERRO DE TRANSIÇÃO ILEGAL] Tentativa proibida: "
                    f"Estado atual: {self.current_state.value} -> Destino: {target_state.value} | "
                    f"Cartas recebidas ({num_cards}): {valid_cards}"
                )
            return False

        previous_state = self.current_state
        self.current_state = target_state
        self.board = valid_cards
        self._initial_synced = True
        reason = f"Detecção de {num_cards} cartas comunitárias: {valid_cards}"
        self._record_transition(previous_state, target_state, reason)

        print(
            f"[TRANSIÇÃO ACEITA] {previous_state.value} -> {target_state.value} | "
            f"Board atual ({num_cards} cartas): {self.board}"
        )
        return True

    def transition_to(self, target_state: HandState, reason: str = "") -> bool:
        if self._is_transition_allowed(self.current_state, target_state, len(self.board)):
            prev = self.current_state
            self.current_state = target_state
            self._record_transition(prev, target_state, reason or "Transição explícita")
            print(f"[TRANSIÇÃO MANUAL] {prev.value} -> {target_state.value}")
            return True
        print(f"[ERRO] Transição manual proibida: {self.current_state.value} -> {target_state.value}")
        return False

    def reset_hand(self):
        prev = self.current_state
        self.current_state = HandState.WAITING_HAND
        self.board = []
        self._record_transition(prev, self.current_state, "Reset de mão")
        print(f"[RESET] Máquina reiniciada: {prev.value} -> {self.current_state.value}")
