"""
main.py
Ponto de entrada do Poker Analytics Engine (PAE).
Orquestrador Mestre integrando Visão Computacional, OCR, Máquina de Estados e Motor de Risco.
"""

import os
import math
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Garante acesso à raiz do projeto e aos pacotes de src/
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_HERO_STACK,
    DEFAULT_NUM_OPPONENTS,
    MONTE_CARLO_ITERATIONS,
    POLL_INTERVAL,
    SAMPLES_DIR,
)
from src.engine import (
    HandState,
    PokerHandStateMachine,
    calculate_equity,
    detect_draws,
    evaluate_7_cards,
    get_hand_category_name,
    make_decision,
)
from src.engine.dataset_logger import DatasetLogger
from src.ocr import (
    detect_turn_and_bet_to_call,
    extract_hero_stack,
    extract_pot,
    extract_hero_position,
)
from src.vision import (
    detect_active_opponents,
    detect_board,
    find_and_detect_hero_cards,
    grab_screen,
)

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"


def _valid_chip_amount(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


class PokerAnalyticsApp:
    """
    Orquestrador Mestre do Poker Analytics Engine (PAE).
    
    Integra de ponta a ponta:
      1. Visão Computacional (src.vision) -> Reconhece cartas do board, hero e oponentes.
      2. Leitura OCR (src.ocr)            -> Extrai numericamente pote, turno e stack.
      3. Máquina de Estados (src.engine)  -> Valida o ciclo sequencial de Texas Hold'em.
      4. Motor de Probabilidades          -> Calcula Equity via Monte Carlo e Outs.
      5. Análise de Risco                 -> Calcula Pot Odds, EV e tomada de decisão ótima.
    """

    def __init__(
        self,
        hero_cards: Optional[List[str]] = None,
        hero_stack: float = DEFAULT_HERO_STACK,
        num_opponents: int = DEFAULT_NUM_OPPONENTS,
        iterations: int = MONTE_CARLO_ITERATIONS,
        allow_initial_sync: bool = True,
        auto_detect_hero: bool = True,
        auto_detect_pot: bool = True,
        auto_detect_turn: bool = True
    ):
        self.hero_position = "Desconhecida"
        self.state_machine = PokerHandStateMachine(allow_initial_sync=allow_initial_sync)
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        self.dataset_logger = DatasetLogger(str(data_dir / "dataset_partidas.csv"))
        self.hero_cards = list(hero_cards) if hero_cards is not None else []
        self.hero_stack = hero_stack
        self.num_opponents = num_opponents
        self.iterations = iterations
        self.auto_detect_hero = auto_detect_hero
        self.auto_detect_pot = auto_detect_pot
        self.auto_detect_turn = auto_detect_turn

        # Cache de estado para economia de CPU/GPU
        self.last_board: List[str] = []
        self.last_state: HandState = HandState.WAITING_HAND
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.total_frames_processed: int = 0
        self.total_recalculations: int = 0
        self.total_equity_calculations: int = 0
        self._equity_cache_key = None
        self._equity_cache_result = None
        self._hero_turn_observed = None
        self._decision_generation = 0
        self.total_uncertain_frames: int = 0
        self._new_hand_hero_cards_set: bool = False
        self._hand_generation = self.state_machine.hand_generation
        self._hand_boundary_seen = False
        self._preflop_end_seen = False
        self._redeal_gap_seen = False
        self._hand_position = None
        self._away_board_candidate = None
        self._away_board_count = 0
        self._away_board_witness = None
        self._new_hand_candidate = None
        self._new_hand_candidate_count = 0
        self.last_hand_observation: Optional[Dict[str, Any]] = None
        
        from src.engine.range_model import OpponentModel
        from src.engine.evaluator import FULL_DECK
        self.opponent_model = OpponentModel(FULL_DECK)

    def _clear_hand_context(self, hero_cards: Optional[List[str]] = None):
        """Descarta dados da mão anterior, mesmo quando as novas cartas são iguais."""
        self.hero_cards = list(hero_cards) if hero_cards is not None else []
        self.last_analysis = None
        self._equity_cache_key = None
        self._equity_cache_result = None
        self._hero_turn_observed = None
        self._decision_generation = 0
        self.last_board = []
        self.last_state = HandState.WAITING_HAND
        self.hero_position = "Desconhecida"
        self._new_hand_hero_cards_set = hero_cards is not None
        self._hand_generation = self.state_machine.hand_generation
        self.hero_cards_stable_frames = 0
        self.last_detected_hero = None
        self._hero_was_absent = False
        self._hand_boundary_seen = False
        self._preflop_end_seen = False
        self._redeal_gap_seen = False
        self._hand_position = None
        self._away_board_candidate = None
        self._away_board_count = 0
        self._away_board_witness = None
        self._new_hand_candidate = None
        self._new_hand_candidate_count = 0
        self.opponent_model.reset()

    def reset_hand(self, hero_cards: Optional[List[str]] = None, reason: str = "Reset de mão"):
        """Reset explícito; hero_cards só deve conter cartas adquiridas para a nova mão."""
        self.state_machine.reset_hand(reason=reason)
        self._clear_hand_context(hero_cards)

    def _observe_action_context(self, context: Optional[Dict[str, Any]]):
        """Registra oportunidades observadas, não ações supostamente executadas."""
        if context is None:
            return
        active = bool(context.get("is_hero_turn"))
        if active and getattr(self, "_hero_turn_observed", None) is not True:
            self._decision_generation = getattr(self, "_decision_generation", 0) + 1
        self._hero_turn_observed = active

    def observe_hand(self, hero_cards: List[str], board: List[str], pot: Optional[float], debug: bool = False,
                     action_context: Optional[Dict[str, Any]] = None,
                     hero_position: Optional[str] = None,
                     dealer_diagnostics: Optional[Dict[str, Any]] = None,
                     observed_opponents: Optional[int] = None) -> str:
        """Observa o ciclo mesmo fora do turno; não calcula nem executa ações.

        No pré-flop, board vazio e hole cards divergentes não corroboram nova
        mão: exige encerramento observado ou dealer alterado corroborado por
        redistribuição. A posição vem do reconhecedor existente (mesa 6-max).
        Após o flop, preserva a confirmação por board reiniciado/redistribuição.
        Board ausente isoladamente nunca confirma mão.
        """
        previous_state = self.state_machine.current_state
        known_board = list(self.state_machine.board)
        confirmed_hero = list(self.hero_cards)
        self._observation_sequence = getattr(self, "_observation_sequence", 0) + 1
        capture_id = self._observation_sequence
        timestamp = datetime.now(timezone.utc).isoformat()
        duplicates = PokerHandStateMachine.find_duplicate_cards(hero_cards + board)
        valid_board = PokerHandStateMachine.filter_valid_cards(board)
        unknown = any(not c or c.lower() in ("desconhecida", "none", "--") for c in board)
        complete = not unknown and len(valid_board) in (0, 3, 4, 5) and len(set(valid_board)) == len(valid_board)
        clear = complete and not valid_board
        in_preflop = previous_state == HandState.PRE_FLOP and not known_board
        positions = ("BTN", "SB", "BB", "UTG", "MP", "CO")
        position = hero_position if hero_position in positions else None
        hand_position = getattr(self, "_hand_position", None)
        dealer_changed = bool(position and hand_position and position != hand_position)
        dealer_unchanged = bool(position and position == hand_position)
        valid_hero = len(hero_cards) == 2 and len(set(hero_cards)) == 2 and all(
            c and c.lower() not in ("desconhecida", "vazio", "none", "--") for c in hero_cards
        )
        changed = valid_hero and bool(self.hero_cards) and set(hero_cards) != set(self.hero_cards)
        # Mais de uma carta nova completa não é simples ausência/regressão de OCR.
        restarted = (complete and len(valid_board) >= 3 and bool(known_board)
                     and len(valid_board) <= len(known_board)
                     and len(set(valid_board) - set(known_board)) >= 2)
        last_pot = self.last_analysis.get("pot_size", 0) if self.last_analysis else 0
        pot_changed = (_valid_chip_amount(pot) and pot > 0
                       and _valid_chip_amount(last_pot) and last_pot > 0 and pot != last_pot)
        active = bool(action_context and action_context.get("is_hero_turn"))
        waiting = bool(action_context and action_context.get("is_hero_turn") is False
                       and action_context.get("action_type") == "WAITING")
        if not hero_cards and not duplicates:
            self._hero_was_absent = True
            # Botões ativos são incompatíveis com confirmar encerramento por
            # mera falha da leitura das hole cards. UNKNOWN continua ambíguo.
            if clear and self.hero_cards and (not active or not in_preflop):
                self._hand_boundary_seen = True
                if waiting:
                    self._redeal_gap_seen = True
                # Board vazio é normal no pré-flop. Exige encerramento observado:
                # sem hole cards, pote zerado e botões explicitamente em espera.
                # UNKNOWN/valor ilegível não confirma esse intervalo.
                if (in_preflop and _valid_chip_amount(pot) and pot == 0
                        and waiting):
                    self._preflop_end_seen = True
        # Observa o board após possível fold sem atualizar FSM, ranges ou matemática.
        # Duas leituras completas compatíveis corroboram a existência dessa rodada;
        # board depois limpo + redistribuição corroborada não depende do dealer.
        away = getattr(self, "_away_board_candidate", None)
        if (not hero_cards and waiting and complete and len(valid_board) >= 3
                and not PokerHandStateMachine.find_duplicate_cards(confirmed_hero + valid_board)
                and valid_board[:len(known_board)] == known_board):
            compatible = away is not None and valid_board[:len(away)] == list(away)
            self._away_board_count = getattr(self, "_away_board_count", 0) + 1 if compatible else 1
            self._away_board_candidate = tuple(valid_board)
            if self._away_board_count >= 2:
                self._away_board_witness = tuple(valid_board)
        else:
            # Falta de leitura não reforça evidência. Conflitos não são testemunho.
            self._away_board_candidate = None
            self._away_board_count = 0
        witnessed_clear = clear and bool(getattr(self, "_away_board_witness", None))
        boundary = getattr(self, "_hand_boundary_seen", False)
        preflop_end = getattr(self, "_preflop_end_seen", False)
        # Posição/dealer NÃO basta isoladamente: exige novas hole cards ou
        # um intervalo observado sem Hero e com botões em espera. O candidato
        # precisa repetir as mesmas cartas, board E posição em duas capturas.
        dealer_redeal = (dealer_changed and (clear or in_preflop)
                         and (changed or getattr(self, "_redeal_gap_seen", False)))
        criteria = dict(hero_alterado=changed, board_completo=complete,
                        board_limpo=clear, board_reiniciado=restarted,
                        intervalo_observado=boundary, pote_alterado=pot_changed,
                        fim_preflop_observado=preflop_end,
                        posicao_da_mao=hand_position, posicao_observada=position,
                        dealer_alterado=dealer_changed,
                        intervalo_em_espera=getattr(self, "_redeal_gap_seen", False),
                        board_observado_sem_hero=list(getattr(self, "_away_board_witness", None) or []))

        def report(status, reason):
            missing = []
            if not valid_hero:
                missing.append("duas cartas do Hero reconhecidas")
            if not complete:
                missing.append("board completo reconhecido")
            if status == "UNCERTAIN" and valid_hero and complete and not proposal:
                if hand_position is None:
                    missing.append("referência do dealer")
                if position is None:
                    missing.append("dealer atual")
                missing.append("continuidade confirmada ou evidência independente de redistribuição")
            if status == "UNCERTAIN" and proposal:
                missing.append("segunda captura compatível da redistribuição")
            diagnostic = dict(captura_id=capture_id, timestamp=timestamp,
                              estado_anterior=previous_state.value, estado_atual=self.state_machine.current_state.value,
                              hero_confirmado=confirmed_hero, hero_observado=list(hero_cards),
                              board_anterior=known_board, board_observado=list(board), criterios=dict(criteria),
                              pote_anterior=last_pot, pote_observado=pot,
                              call_observado=action_context.get("bet_to_call") if action_context else None,
                              botoes_disponiveis=list(action_context.get("available_actions", [])) if action_context else [],
                              turno_confirmado=active, posicao=position,
                              dealer=dict(dealer_diagnostics or {"confidence": None, "reason": "detalhes não fornecidos"}),
                              adversarios_observados=observed_opponents,
                              adversarios_fonte="estimativa de ativos; detector possui fallback interno" if observed_opponents is not None else "leitura indisponível",
                              adversarios_confirmados_anteriormente=self.last_analysis.get("num_opponents") if self.last_analysis else None,
                              estado_espera="SUSPENDED" if status == "UNCERTAIN" else "READY" if active else "OUT_OF_TURN",
                              candidato=getattr(self, "_new_hand_candidate", None),
                              candidato_confirmado=(list(hero_cards), list(valid_board), position) if status == "NEW_HAND" else None,
                              capturas_compativeis=criteria.get("observacoes_compativeis", getattr(self, "_new_hand_candidate_count", 0)),
                              evidencias_ausentes=missing,
                              resultado=status, motivo=reason)
            if debug:
                print(f"[DEBUG MÃO] {json.dumps(diagnostic, ensure_ascii=False)}")
            self.last_hand_observation = diagnostic
            return status

        proposal = False
        if duplicates:
            # Uma captura impossível não corrobora redistribuição nem autoriza reset.
            self._new_hand_candidate = None
            self._new_hand_candidate_count = 0
            return report("UNCERTAIN", f"Cartas duplicadas na captura: {', '.join(duplicates)}")

        proposal = valid_hero and complete and (
            witnessed_clear or dealer_redeal or (in_preflop and preflop_end)
            or (not in_preflop and (
                (changed and (clear or restarted)) or restarted or (boundary and clear and pot_changed)
            ))
        )
        if proposal:
            candidate = (tuple(sorted(hero_cards)), tuple(valid_board), position if dealer_redeal else None)
            count = getattr(self, "_new_hand_candidate_count", 0) + 1 if candidate == getattr(self, "_new_hand_candidate", None) else 1
            self._new_hand_candidate = candidate
            self._new_hand_candidate_count = count
            criteria["observacoes_compativeis"] = count
            if count >= 2:
                reason = ("Nova mão confirmada: board observado sem Hero, mesa limpa e redistribuição corroborada"
                          if witnessed_clear else "Nova mão confirmada: dealer alterado e redistribuição corroborada"
                          if dealer_redeal else
                          "Nova mão confirmada: encerramento pré-flop observado e redistribuição corroborada"
                          if in_preflop else "Nova mão confirmada: redistribuição corroborada de Hero/board")
                print(f"\n[INFO] {reason}")
                self.reset_hand(hero_cards=hero_cards, reason=reason)
                self._hand_position = position
                if position:
                    self.hero_position = position
                self.state_machine.update_board(board, has_hero_cards=True)
                return report("NEW_HAND", reason)
            return report("UNCERTAIN", "Redistribuição candidata; aguardando observação compatível")

        candidate = getattr(self, "_new_hand_candidate", None)
        # Falta de dados não confirma nem refuta o candidato. Uma observação
        # positiva contraditória (cartas/board/dealer) o descarta imediatamente.
        hero_known = PokerHandStateMachine.filter_valid_cards(hero_cards)
        missing_dealer = bool(candidate and candidate[2] is not None and position is None)
        uncontradicted = (candidate is not None and (not valid_hero or not complete or missing_dealer)
                         and set(hero_known) <= set(candidate[0])
                         and (not valid_board or valid_board == list(candidate[1])[:len(valid_board)])
                         and (position is None or candidate[2] is None or position == candidate[2]))
        if not uncontradicted:
            self._new_hand_candidate = None
            self._new_hand_candidate_count = 0
        if not valid_hero or not complete:
            return report("UNCERTAIN", "Hero/board não reconhecido por completo; ciclo da mão não confirmado")
        if changed:
            return report("UNCERTAIN", "Cartas do Hero divergentes sem encerramento confirmado ou redistribuição corroborada")
        if in_preflop and boundary and clear and not dealer_unchanged:
            return report("UNCERTAIN", "Retorno do Hero após intervalo ambíguo: sem encerramento confirmado nem dealer compatível")
        if boundary and clear and not in_preflop:
            return report("UNCERTAIN", "Intervalo observado, mas redistribuição ainda indistinguível de falha visual")
        if not self.state_machine.update_board(board, has_hero_cards=True):
            return report("UNCERTAIN", "Board incompatível com a mão conhecida; reset não confirmado")
        if dealer_changed:
            return report("UNCERTAIN", "Dealer/posição divergente sem redistribuição corroborada")
        self._hero_was_absent = False
        self._hand_boundary_seen = False
        self._redeal_gap_seen = False
        # Continuidade recuperada encerra o intervalo; o testemunho antigo não
        # pode transformar uma futura falha isolada do board em redistribuição.
        self._away_board_candidate = None
        self._away_board_count = 0
        self._away_board_witness = None
        if hand_position is None:
            self._hand_position = position
            if position:
                self.hero_position = position
        return report("CURRENT", "Captura compatível com a mão conhecida")

    def process_frame(
        self,
        frame_or_path: Union[str, Path, np.ndarray],
        pot_size: Optional[float] = None,
        bet_to_call: Optional[float] = None,
        show_dashboard: bool = True
    ) -> Dict[str, Any]:
        """Processa um frame de vídeo ou imagem estática da mesa."""
        # 1. Extração dinâmica da mão do Hero
        if self.auto_detect_hero:
            hero_detected, detected_box = find_and_detect_hero_cards(
                frame_or_path,
                verbose=False,
                hero_coords_hint=getattr(self, 'hero_coords', None),
                return_coords=True,
            )
            if detected_box is not None:
                self.hero_coords = detected_box
            has_valid_detected = (
                hero_detected
                and len(hero_detected) == 2
                and all(c and c != "Desconhecida" for c in hero_detected)
            )

            already_has_valid_hand = (
                isinstance(self.hero_cards, list)
                and len(self.hero_cards) == 2
                and not PokerHandStateMachine.find_duplicate_cards(self.hero_cards)
                and all(
                    isinstance(c, str)
                    and len(c) >= 2
                    and c.lower() not in ("desconhecida", "vazio", "none", "--")
                    for c in self.hero_cards
                )
            )

            # Iron Lock Estrito: preserva a mão sólida até o fim da rodada
            if has_valid_detected and not already_has_valid_hand:
                if getattr(self, 'last_detected_hero', None) == hero_detected:
                    self.hero_cards_stable_frames = getattr(self, 'hero_cards_stable_frames', 0) + 1
                else:
                    self.hero_cards_stable_frames = 1
                    self.last_detected_hero = hero_detected
                    
                if self.hero_cards_stable_frames >= 1:
                    self.hero_cards = hero_detected
                    self._new_hand_hero_cards_set = True
            elif not already_has_valid_hand:
                self.hero_cards_stable_frames = 0
                self.hero_cards = []

        # 2. Leitura numérica do pote via OCR
        actual_pot = pot_size
        if self.auto_detect_pot and actual_pot is None:
            actual_pot = extract_pot(frame_or_path, default_value=None, verbose=False)

        if actual_pot is None and not self.auto_detect_pot:
            actual_pot = 85.0
        new_position = extract_hero_position(frame_or_path, verbose=False, hero_coords=getattr(self, 'hero_coords', None))
        if new_position != "Desconhecida" and self.hero_position == "Desconhecida":
            self.hero_position = new_position

        # 3. Leitura do turno e aposta a pagar (bet_to_call) via OCR
        actual_bet = bet_to_call
        turn_info = None
        if self.auto_detect_turn:
            turn_info = detect_turn_and_bet_to_call(frame_or_path, verbose=False)
            if actual_bet is None and turn_info["is_hero_turn"]:
                actual_bet = turn_info["bet_to_call"]

        if actual_bet is None and not self.auto_detect_turn:
            actual_bet = 30.0

        # 4. Detecção das cartas comunitárias do Board
        detected_cards = detect_board(frame_or_path, verbose=False)
        return self.process_cards(
            detected_cards, actual_pot, actual_bet, show_dashboard, action_context=turn_info
        )

    def _financial_context_error(self, pot, call, context) -> Optional[str]:
        """Não estima fichas nem converte aposta nominal em call all-in."""
        if not _valid_chip_amount(self.hero_stack):
            return "Stack não confirmado ou inválido na captura atual."
        if not _valid_chip_amount(pot):
            return "Pote não confirmado ou inválido na captura atual."
        if call is not None and not _valid_chip_amount(call):
            return "Valor do CALL inválido na captura atual."
        if self.hero_stack == 0:
            return "Stack zero: Hero pode já estar all-in; nenhuma ação atual validada."
        if call is not None and call > self.hero_stack:
            return (f"CALL de {call:g} excede o stack de {self.hero_stack:g}. Possível call all-in; "
                    "valor efetivo e pote elegível não confirmados. Não foi reduzido o CALL por estimativa.")
        if context and context.get("is_hero_turn") and call is not None:
            observed_call = context.get("bet_to_call")
            if not _valid_chip_amount(observed_call) or observed_call != call:
                return "Valor do CALL da análise difere do valor confirmado nos botões da captura."
        return None

    @staticmethod
    def _with_action_context(analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Preserva a ação do modelo; valida separadamente sua disponibilidade na captura."""
        analysis["action_context"] = context
        available = False
        status = "Turno e ações não verificados."
        stack = analysis.get("hero_stack")
        amount = analysis.get("recommended_amount")
        call = analysis.get("bet_to_call")
        analysis["call_is_all_in"] = (
            analysis["action"] == "CALL" and _valid_chip_amount(stack) and stack > 0
            and amount == call == stack
        )
        if analysis.get("observation_valid") is False:
            status = "Board da captura incerto; nenhuma recomendação atual validada."
            if analysis.get("duplicate_cards") or analysis.get("financial_valid") is False:
                status = analysis["reason"] + " Nenhuma recomendação atual validada."
        elif (not _valid_chip_amount(stack) or stack == 0
              or not _valid_chip_amount(amount) or amount > stack):
            status = "Stack/valor da ação inválido ou valor recomendado excede o stack; recomendação suspensa."
        elif (analysis.get("action_details") or {}).get("supported") is False:
            status = analysis["reason"] + " Contexto de aposta não suportado; recomendação suspensa."
        elif context:
            action = analysis["action"]
            observed = context.get("available_actions", [])
            bet = context.get("bet_to_call")
            if not context.get("is_hero_turn"):
                status = "Vez do Hero não confirmada."
            elif not analysis.get("recalculated") or analysis.get("bet_to_call") is None:
                status = "Dados insuficientes para validar uma recomendação."
            elif action not in observed and not (
                action == "CALL" and "ALL-IN" in observed
                and context.get("all_in_nature") == "CALL"
                and analysis.get("call_is_all_in", False)
            ):
                status = "Ação do modelo não observada nos botões desta captura."
            elif action in ("BET", "RAISE", "ALL-IN"):
                status = "Tipo de ação observado, mas tamanho/limites da aposta não validados."
            else:
                available = (
                    action == "FOLD"
                    or (action == "CHECK" and context.get("action_type") == "CHECK"
                        and bet == 0 and analysis["bet_to_call"] == 0)
                    or (action == "CALL" and bet is not None and bet > 0
                        and bet == analysis["bet_to_call"] == analysis["recommended_amount"])
                )
                status = "Compatível com os botões e o valor observados na captura." if available else "Valor/contexto da ação não confirmado."
                if available and analysis["call_is_all_in"]:
                    status = "CALL ALL-IN: valor efetivo observado no botão corresponde ao stack inteiro."
        analysis["action_available"] = available
        analysis["action_status"] = status
        return analysis

    def process_cards(
        self,
        detected_cards: List[str],
        pot_size: Optional[float] = 85.0,
        bet_to_call: Optional[float] = 30.0,
        show_dashboard: bool = True,
        action_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Processa as cartas detectadas na mesa com controle estrito de ciclo de vida,
        mecanismo de cache inteligente e avaliação de Valor Esperado (EV).
        """
        self.total_frames_processed += 1
        context = dict(action_context) if action_context is not None else None
        if context is not None:
            context["available_actions"] = list(context.get("available_actions", []))

        # Também reconhece resets explícitos feitos diretamente na FSM.
        if getattr(self, "_hand_generation", 0) != self.state_machine.hand_generation:
            self._clear_hand_context()
        self._observe_action_context(context)
        # Sinal de aquisição, não autorização persistente para preservar cartas antigas.
        self._new_hand_hero_cards_set = False

        # 1. Atualiza a Máquina de Estados Finitos (FSM)
        has_valid_hero = (
            isinstance(self.hero_cards, list)
            and len(self.hero_cards) == 2
            and all(
                isinstance(c, str)
                and len(c) >= 2
                and c.lower() not in ("desconhecida", "vazio", "none", "--")
                for c in self.hero_cards
            )
        )
        duplicates = PokerHandStateMachine.find_duplicate_cards(list(self.hero_cards or []) + detected_cards)
        if duplicates:
            # Validação conjunta ANTES da transição, do cache e de qualquer matemática.
            self.state_machine.board_read_status = "CONFLICT"
            board_is_valid = False
        else:
            board_is_valid = self.state_machine.update_board(detected_cards, has_hero_cards=has_valid_hero)
        current_state = self.state_machine.current_state
        current_board = self.state_machine.board

        financial_error = self._financial_context_error(pot_size, bet_to_call, context) if board_is_valid else None
        if not board_is_valid or financial_error:
            # Não contamina o cache válido nem atualiza ranges com uma captura rejeitada.
            self.total_uncertain_frames = getattr(self, "total_uncertain_frames", 0) + 1
            analysis = dict(self.last_analysis or {
                "hero_position": self.hero_position,
                "hero_hand": "Não avaliada",
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
                "num_opponents": self.num_opponents,
                "pot_odds": 0.0,
                "equity": 0.0,
                "p_win": 0.0,
                "p_tie": 0.0,
                "p_lose": 0.0,
                "ev": 0.0,
                "sim_time": 0.0,
            })
            analysis.update({
                "state": current_state.value,
                "board": list(current_board),
                "hero_cards": list(self.hero_cards),
                "board_observation": list(detected_cards),
                "board_read_status": self.state_machine.board_read_status,
                "observation_valid": False,
                "action": "AGUARDANDO RECONHECIMENTO DO BOARD",
                "recommended_amount": 0.0,
                "reason": "Leitura rejeitada; estado conhecido preservado sem recomendação atual.",
                "recalculated": False,
            })
            if duplicates:
                analysis.update({
                    "duplicate_cards": duplicates,
                    "hero_observation": list(self.hero_cards),
                    "hero_cards": list(self.last_analysis["hero_cards"]) if self.last_analysis else [],
                    "action": "AGUARDANDO RECONHECIMENTO DAS CARTAS",
                    "reason": f"Cartas duplicadas na captura: {', '.join(duplicates)}; estado conhecido preservado.",
                })
            if financial_error:
                analysis.update({
                    "financial_valid": False,
                    "financial_observation": dict(hero_stack=self.hero_stack, pot_size=pot_size, bet_to_call=bet_to_call),
                    "action": "AGUARDANDO CONTEXTO FINANCEIRO",
                    "reason": financial_error,
                })
            self._with_action_context(analysis, context)
            if show_dashboard:
                self.render_dashboard(analysis)
            return analysis

        # 2. Otimização de Performance: Verifica se houve alteração na mesa
        last_pot = self.last_analysis.get("pot_size") if self.last_analysis else None
        last_bet = self.last_analysis.get("bet_to_call") if self.last_analysis else None
        last_hero = self.last_analysis.get("hero_cards") if self.last_analysis else None
        last_pos = self.last_analysis.get("hero_position") if self.last_analysis else None
        last_stack = self.last_analysis.get("hero_stack") if self.last_analysis else None
        last_num_opponents = self.last_analysis.get("num_opponents") if self.last_analysis else None
        range_revision = getattr(self.opponent_model, "revision", 0)
        has_changed = (
            (current_board != self.last_board)
            or (current_state != self.last_state)
            or (pot_size != last_pot)
            or (bet_to_call != last_bet)
            or (self.hero_cards != last_hero)
            or (self.hero_position != last_pos)
            or (self.hero_stack != last_stack)
            or (self.num_opponents != last_num_opponents)
            or (self.last_analysis is not None and self.iterations != self.last_analysis.get("iterations"))
            or (self.last_analysis is not None and getattr(self, "_decision_generation", 0)
                != self.last_analysis.get("decision_generation", 0))
            or (context != (self.last_analysis.get("action_context") if self.last_analysis else None))
            or (self.last_analysis is not None
                and range_revision != self.last_analysis.get("range_revision", range_revision))
            or (self.last_analysis is not None
                and id(self.opponent_model) != self.last_analysis.get("range_model_identity", id(self.opponent_model)))
        )

        hero_is_missing = (
            not isinstance(self.hero_cards, list)
            or len(self.hero_cards) < 2
            or any(c.lower() in ("desconhecida", "vazio", "none", "--") for c in self.hero_cards)
        )

        if not has_changed and self.last_analysis is not None:
            if show_dashboard:
                self.render_dashboard(self.last_analysis, cached=True)
            return self.last_analysis

        # 3. Mudança real detectada: dispara novo recálculo matemático
        self.total_recalculations += 1
        self.last_board = list(current_board)
        self.last_state = current_state

        has_valid_hero = (
            isinstance(self.hero_cards, list)
            and len(self.hero_cards) == 2
            and all(
                isinstance(c, str)
                and len(c) >= 2
                and c.lower() not in ("desconhecida", "vazio", "none", "--")
                for c in self.hero_cards
            )
        )

        if not has_valid_hero or current_state == HandState.WAITING_HAND or bet_to_call is None:
            unknown_call = has_valid_hero and bet_to_call is None
            analysis = {
                "state": current_state.value,
                "board": current_board,
                "hero_cards": list(self.hero_cards) if unknown_call else [],
                "hero_position": self.hero_position,
                "hero_hand": "Valor do CALL desconhecido" if unknown_call else "Não identificadas",
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
                "num_opponents": self.num_opponents,
                "pot_odds": 0.0,
                "equity": 0.0,
                "p_win": 0.0,
                "p_tie": 0.0,
                "p_lose": 0.0,
                "ev": 0.0,
                "action": "AGUARDANDO VALOR DO CALL" if unknown_call else "AGUARDANDO LEITURA DAS CARTAS",
                "recommended_amount": 0.0,
                "reason": "Valor do CALL não identificado. Não foi presumido CHECK." if unknown_call else "Cartas do Hero não identificadas com precisão na mesa. Simulação fictícia bloqueada.",
                "sim_time": 0.0,
                "recalculated": False,
            }
            analysis["iterations"] = self.iterations
            analysis["decision_generation"] = getattr(self, "_decision_generation", 0)
            self._with_action_context(analysis, context)
            self.last_analysis = analysis
            if show_dashboard:
                sys.stdout.write(f"\r{C_YELLOW}[{analysis['action']}]{C_RESET}   ")
                sys.stdout.flush()
            return analysis

        is_evaluable = (
            current_state == HandState.PRE_FLOP
            or (current_state in (HandState.FLOP, HandState.TURN, HandState.RIVER) and len(current_board) >= 3)
        )

        if is_evaluable:
            if current_state == HandState.PRE_FLOP:
                c1, c2 = self.hero_cards[0], self.hero_cards[1]
                r1, r2 = c1[:-1], c2[:-1]
                if r1 == r2:
                    hero_hand_desc = f"Pocket Pair ({r1}{r2})"
                elif c1[-1].lower() == c2[-1].lower():
                    hero_hand_desc = f"Suited Hole Cards ({c1}, {c2})"
                else:
                    hero_hand_desc = f"Hole Cards ({c1}, {c2})"
            else:
                hero_eval = evaluate_7_cards(self.hero_cards + current_board)
                hero_hand_desc = get_hand_category_name(hero_eval)

            # Apenas capturas aceitas chegam aqui. O modelo deduplica evidências
            # por mão; recálculo por pote/stack não representa nova ação adversária.
            if current_state == HandState.PRE_FLOP:
                self.opponent_model.apply_action_heuristic(pot_size, "PRE_FLOP")
            else:
                self.opponent_model.apply_postflop_board(current_board, pot_size)

            # Cache probabilístico separado do contexto financeiro/da vez.
            # A revisão só é consultada DEPOIS de processar a evidência de range.
            equity_key = (self.state_machine.hand_generation, tuple(sorted(self.hero_cards)),
                          tuple(current_board), self.num_opponents, self.iterations,
                          id(self.opponent_model), getattr(self.opponent_model, "revision", 0))
            equity_cached = equity_key == getattr(self, "_equity_cache_key", None)
            if not equity_cached:
                result = calculate_equity(
                    hero_cards=self.hero_cards,
                    board_cards=current_board,
                    num_opponents=self.num_opponents,
                    iterations=self.iterations,
                    opp_range_percent=self.opponent_model
                )
                self._equity_cache_key = equity_key
                self._equity_cache_result = result
                self.total_equity_calculations = getattr(self, "total_equity_calculations", 0) + 1
            p_win, p_tie, p_lose, equity, elapsed = self._equity_cache_result

            outs, draw_name = detect_draws(self.hero_cards, current_board)

            # Motor de Risco e Tomada de Decisão
            decision = make_decision(
                loose_mode=("--loose" in sys.argv),
                p_win=p_win,
                p_lose=p_lose,
                p_tie=p_tie,
                equity=equity,
                num_opponents=self.num_opponents,
                pot_size=pot_size,
                bet_to_call=bet_to_call,
                hero_stack=self.hero_stack,
                state=current_state.value,
                hero_cards=self.hero_cards,
                board_cards=current_board,
                outs=outs,
                draw_name=draw_name
            )

            analysis = {
                "state": current_state.value,
                "board": current_board,
                "hero_cards": self.hero_cards,
                "hero_position": self.hero_position,
                "hero_hand": hero_hand_desc,
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
                "num_opponents": self.num_opponents,
                "pot_odds": decision["pot_odds"],
                "equity": decision["equity"],
                "opp_range_percent": "Bayesiano Dinâmico",
                "range_revision": getattr(self.opponent_model, "revision", 0),
                "range_model_identity": id(self.opponent_model),
                "iterations": self.iterations,
                "decision_generation": getattr(self, "_decision_generation", 0),
                "equity_cached": equity_cached,
                "p_win": p_win,
                "p_tie": p_tie,
                "p_lose": p_lose,
                "ev": decision["ev"],
                "action": decision["action"],
                "recommended_amount": decision["recommended_amount"],
                "reason": decision["reason"],
                "sim_time": elapsed,
                "recalculated": True,
            }
            analysis.update({key: decision[key] for key in
                             ("action_details", "selected_action", "selected_action_probability", "cfr_strategy")
                             if key in decision})

        else:
            action = "CHECK" if bet_to_call == 0 else "FOLD"
            analysis = {
                "state": current_state.value,
                "board": current_board,
                "hero_cards": self.hero_cards,
                "hero_position": self.hero_position,
                "hero_hand": "Aguardando cartas suficientes",
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
                "num_opponents": self.num_opponents,
                "pot_odds": 0.0,
                "equity": 0.0,
                "p_win": 0.0,
                "p_tie": 0.0,
                "p_lose": 0.0,
                "ev": 0.0,
                "action": action,
                "recommended_amount": 0.0,
                "reason": "Estágio sem cartas comunitárias suficientes para cálculo de probabilidade.",
                "sim_time": 0.0,
                "recalculated": False,
            }

        self._with_action_context(analysis, context)

        # Grava os dados da simulação no CSV de Machine Learning
        if analysis.get("recalculated") and analysis.get("state") != "WAITING_HAND":
            try:
                self.dataset_logger.log_state(analysis)
            except Exception as e:
                print(f"[AVISO] Falha ao gravar dataset: {e}")

        self.last_analysis = analysis

        if show_dashboard:
            self.render_dashboard(analysis, cached=False)

        return analysis

    def render_dashboard(self, a: Dict[str, Any], cached: bool = False):
        """Renderiza o painel visual formatado no terminal com cores ANSI."""
        if a.get("observation_valid") is False:
            print("\n[CAPTURA INCERTA - SEM RECOMENDAÇÃO ATUAL]")
            print(f"Último estado/board válido: {a['state']} | {a['board']}")
            if a.get("financial_valid") is False:
                print(f"Valores recebidos (não validados): {a['financial_observation']}")
            else:
                print(f"Leitura recebida ({a['board_read_status']}): {a['board_observation']}")
            print(a["action_status"])
            return
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"
        C_CYAN = "\033[96m"
        C_YELLOW = "\033[93m"
        C_GREEN = "\033[92m"
        C_RED = "\033[91m"
        C_WHITE = "\033[97m"

        act = a["action"]
        if act in ("RAISE", "ALL-IN", "BET"):
            action_color = f"{C_BOLD}{C_GREEN}"
        elif act in ("CALL", "CHECK"):
            action_color = f"{C_BOLD}{C_CYAN}"
        else:
            action_color = f"{C_BOLD}{C_RED}"

        ev_val = a["ev"]
        ev_color = C_GREEN if ev_val >= 0 else C_RED
        cache_status = f"{C_YELLOW}[CACHE ATIVO - Zero CPU]{C_RESET}" if cached else f"{C_GREEN}[RECÁLCULO EXECUTADO]{C_RESET}"
        if not cached and a.get("equity_cached"):
            cache_status = f"{C_GREEN}[DECISÃO ATUALIZADA - EQUITY EM CACHE]{C_RESET}"

        board_display = ", ".join(a["board"]) if a["board"] else "Nenhuma carta"
        hero_cards_str = ", ".join(a["hero_cards"]) if a["hero_cards"] else "NÃO IDENTIFICADAS"

        w = 81
        sep_double = "═" * w
        sep_single = "─" * w

        print(f"\n╔{sep_double}╗")
        print(f"║ {C_BOLD}{C_CYAN}POKER ANALYTICS ENGINE (PAE){C_RESET} - Painel de Decisão em Tempo Real  {cache_status}  ║")
        print(f"╠{sep_double}╣")
        print(
            f"║ {C_BOLD}ESTADO ATUAL:{C_RESET}     {a['state']:<14s} ║ "
            f"{C_BOLD}CARTAS DO HERO:{C_RESET}  [ {hero_cards_str} ]"
        )
        print(
            f"║ {C_BOLD}BOARD ATUAL:{C_RESET}      [{board_display:<22s}] ║ "
            f"{C_BOLD}JOGO ATUAL:{C_RESET}      {a['hero_hand']}"
        )
        pos_str = a.get("hero_position", getattr(self, "hero_position", "Desconhecida"))
        num_opponents = a.get("num_opponents", self.num_opponents)
        print(
            f"║ {C_BOLD}POSIÇÃO HERO:{C_RESET}     {pos_str:<14s} ║ "
            f"{C_BOLD}OPONENTES:{C_RESET}       {num_opponents}"
        )
        print(f"╠{sep_double}╣")
        print(f"║ {C_BOLD}DADOS DA RODADA:{C_RESET}")
        call_display = f"{a['bet_to_call']:8.1f}" if a["bet_to_call"] is not None else "Desconhecido"
        print(
            f"║   • Pote Atual:        {a['pot_size']:8.1f} fichas   │ "
            f"• Aposta a Pagar (Call): {call_display} fichas"
        )
        print(
            f"║   • Stack do Hero:     {a['hero_stack']:8.1f} fichas   │ "
            f"• Pot Odds Exigidas:     {a['pot_odds']:8.2f}%"
        )
        print(f"╠{sep_double}╣")
        range_val = a.get('opp_range_percent', 1.0)
        range_str = range_val if isinstance(range_val, str) else ("100% Aleatório" if range_val == 1.0 else f"Top {int(range_val*100)}% Range")
        print(f"║ {C_BOLD}PROBABILIDADES (MONTE CARLO - {self.iterations:,} iterações vs {range_str}):{C_RESET}")
        print(
            f"║   • Taxa de Vitória (P_win): {a['p_win']:6.2f}%  │ "
            f"• Taxa de Empate  (P_tie): {a['p_tie']:6.2f}%"
        )
        print(
            f"║   • Taxa de Derrota (P_lose): {a['p_lose']:5.2f}% │ "
            f"• Equity Consolidada:  {a['equity']:8.2f}%"
        )
        print(
            f"║   • Tempo de Simulação:       {a['sim_time']:8.3f}s"
        )
        print(f"╠{sep_double}╣")
        print(f"║ {C_BOLD}ANÁLISE DE RISCO & VALOR ESPERADO (EV):{C_RESET}")
        ev_label = "Parcela esperada do pote" if a["bet_to_call"] == 0 else "EV do CALL (base)"
        print(f"║   • {ev_label}: {ev_color}{ev_val:+8.2f} fichas{C_RESET}")
        if a["bet_to_call"] == 0:
            print("║   • Sem apostas futuras; não é EV estratégico de CHECK.")
        print("║   • Pote único, sem rake/side pots; não é EV de RAISE/ALL-IN.")
        print("║")
        available = a.get("action_available", False)
        label = "RECOMENDAÇÃO PARA A CAPTURA" if available else "ANÁLISE INFORMATIVA - AÇÃO DO MODELO"
        print(f"║   >>> {label}: {action_color}[ {a['action']} ]{C_RESET} ")
        print(f"║   • Contexto: {a.get('action_status', 'Turno e ações não verificados.')}")
        context = a.get("action_context") or {}
        observed = ", ".join(context.get("available_actions", [])) or "Não confirmadas"
        print(f"║   • Ações observadas nos botões: {observed}")
        details = a.get("action_details") or {}
        if a.get("selected_action"):
            print(f"║   • Ramo avaliado: {a['selected_action']} | Probabilidade: {a['selected_action_probability']:.3f}")
        if details and a["recommended_amount"] > 0:
            print(f"║   • Investimento adicional (modelo): {details['additional_amount']:.12g} fichas")
            if details.get("raise_to") is not None:
                print(f"║   • Total da aposta na street: {details['raise_to']:.12g} fichas")
            elif a["action"] in ("BET", "RAISE", "ALL-IN"):
                print("║   • Total da street não confirmado; contribuições anteriores não observadas.")
            if details.get("raise_by"):
                print(f"║   • Incremento sobre a aposta anterior: {details['raise_by']:.12g} fichas")
            if details.get("validation_reason"):
                print(f"║   • {details['validation_reason']}")
        elif a["recommended_amount"] > 0:
            if not available:
                print(f"║   • Valor do modelo (não validado): {a['recommended_amount']:.1f} fichas")
            elif a["action"] == "CALL":
                print(f"║   • Completar Apostas (Adicional):  {a['recommended_amount']:.1f} fichas")
            elif a["action"] == "RAISE":
                print(f"║   • Aumentar Para (Total na mesa):  {a['recommended_amount']:.1f} fichas")
            elif a["action"] == "BET":
                print(f"║   • Apostar (Valor Adicional):      {a['recommended_amount']:.1f} fichas")
            else:
                print(f"║   • Valor Recomendado:              {a['recommended_amount']:.1f} fichas")
        print("║")
        print(f"║   • Justificativa: {a['reason']}")
        print(f"╚{sep_double}╝\n")


def run_live(poll_interval: float = POLL_INTERVAL, debug_hands: bool = False):
    """Executa o Poker Analytics Engine em modo assistente ao vivo (tempo real)."""
    C_RESET = "\033[0m"
    C_BOLD = "\033[1m"
    C_CYAN = "\033[96m"
    C_YELLOW = "\033[93m"
    C_GREEN = "\033[92m"

    print("=" * 83)
    print(f"{C_BOLD}{C_CYAN}POKER ANALYTICS ENGINE (PAE){C_RESET} - ASSISTENTE AUTÔNOMO AO VIVO (--live)")
    print("Captura nativa de tela no Linux (Wayland / X11) | Decisão em tempo real")
    print(f"Taxa de verificação: {poll_interval}s | {C_YELLOW}Pressione Ctrl + C para encerrar.{C_RESET}")
    print("=" * 83 + "\n")

    app = PokerAnalyticsApp(
        hero_stack=DEFAULT_HERO_STACK,
        num_opponents=DEFAULT_NUM_OPPONENTS,
        iterations=MONTE_CARLO_ITERATIONS,
        allow_initial_sync=True,
        auto_detect_hero=True,
        auto_detect_pot=True,
        auto_detect_turn=True
    )

    uncertain_key = None
    uncertain_count = 0

    def flush_uncertain():
        nonlocal uncertain_key, uncertain_count
        if uncertain_count > 1:
            print(f"[CAPTURAS INCERTAS CONSOLIDADAS: {uncertain_count} com o mesmo motivo]")
        uncertain_key, uncertain_count = None, 0

    try:
        while True:
            frame = grab_screen()
            hero_res = find_and_detect_hero_cards(
                frame,
                verbose=False,
                hero_coords_hint=getattr(app, 'hero_coords', None),
                return_coords=True,
                save_unknown=False,
            )
            hero_detected, detected_box = hero_res
            if detected_box is not None:
                app.hero_coords = detected_box

            has_valid_detected = (
                isinstance(hero_detected, list)
                and len(hero_detected) == 2
                and all(c and c != "Desconhecida" for c in hero_detected)
                and not PokerHandStateMachine.find_duplicate_cards(hero_detected)
            )

            already_has_valid_hand = (
                isinstance(app.hero_cards, list)
                and len(app.hero_cards) == 2
                and all(
                    isinstance(c, str)
                    and len(c) >= 2
                    and c.lower() not in ("desconhecida", "vazio", "none", "--")
                    for c in app.hero_cards
                )
            )

            # IRON LOCK IMEDIATO (1 frame): Trava as cartas do Hero assim que forem identificadas na tela,
            # mesmo enquanto o Hero ainda aguarda sua vez de jogar!
            if has_valid_detected and not already_has_valid_hand:
                app.hero_cards = hero_detected
                app.last_detected_hero = hero_detected
                app.hero_cards_stable_frames = 1
                app._new_hand_hero_cards_set = True
                already_has_valid_hand = True

            # O ciclo da mão não depende de haver botões ativos nem de ler CALL.
            detected_cards = detect_board(frame, verbose=False, save_unknown=False)
            observed_pot = extract_pot(frame, default_value=None, verbose=False)
            turn_info = detect_turn_and_bet_to_call(frame, verbose=False)
            # A mudança de dealer é evidência do ciclo, inclusive fora da vez.
            dealer_diagnostics = {"confidence": None, "reason": "detalhes não fornecidos"}
            new_position = extract_hero_position(frame, verbose=False, diagnostics=dealer_diagnostics, hero_coords=getattr(app, 'hero_coords', None))
            try:
                observed_opponents = detect_active_opponents(frame, hero_coords=getattr(app, 'hero_coords', None))
            except Exception:
                observed_opponents = None
            app._observe_action_context(turn_info)
            hand_status = app.observe_hand(hero_detected, detected_cards, observed_pot, debug=debug_hands,
                                           action_context=turn_info, hero_position=new_position,
                                           dealer_diagnostics=dealer_diagnostics, observed_opponents=observed_opponents)
            if hand_status == "UNCERTAIN":
                app.total_frames_processed += 1
                app.total_uncertain_frames = getattr(app, "total_uncertain_frames", 0) + 1
                detail = "MUDANÇA DO HERO SEM NOVA MÃO CONFIRMADA" if has_valid_detected and hero_detected != app.hero_cards else "CICLO DA MÃO NÃO CONFIRMADO"
                reason = app.last_hand_observation["motivo"]
                missing = app.last_hand_observation["evidencias_ausentes"]
                key = (app.state_machine.hand_generation, detail, reason, tuple(missing))
                if key != uncertain_key:
                    flush_uncertain()
                    uncertain_key = key
                    print(f"\n{C_YELLOW}[CAPTURA INCERTA - {detail}]{C_RESET}")
                    print(f"SEM RECOMENDAÇÃO ATUAL: {reason}")
                    if missing:
                        print(f"Faltam: {'; '.join(missing)}.")
                uncertain_count += 1
                time.sleep(poll_interval)
                continue

            flush_uncertain()

            # Se a máquina está em WAITING_HAND e não há cartas válidas, mantém status limpo aguardando próxima mão
            if app.state_machine.current_state == HandState.WAITING_HAND and not already_has_valid_hand:
                sys.stdout.write(f"\r{C_YELLOW}[AGUARDANDO INÍCIO DA PRÓXIMA MÃO...]{C_RESET}   ")
                sys.stdout.flush()
                time.sleep(poll_interval)
                continue

            if not turn_info["is_hero_turn"]:
                hand_str = f" | Mão travada: {app.hero_cards}" if already_has_valid_hand else ""
                status = "VEZ DO HERO NÃO CONFIRMADA / BOTÕES ILEGÍVEIS" if turn_info.get("action_type") == "UNKNOWN" else "AGUARDANDO SUA VEZ DE JOGAR..."
                sys.stdout.write(f"\r{C_YELLOW}[{status}]{hand_str}{C_RESET}   ")
                sys.stdout.flush()
                time.sleep(poll_interval)
                continue

            bet_to_call = turn_info.get("bet_to_call")
            if bet_to_call is None or (turn_info.get("action_type") == "CALL" and bet_to_call <= 0):
                sys.stdout.write(f"\r{C_YELLOW}[AÇÃO/VALOR DO CALL INCERTO - CHECK NÃO PRESUMIDO]{C_RESET}   ")
                sys.stdout.flush()
                time.sleep(poll_interval)
                continue

            sys.stdout.write("\r" + " " * 75 + "\r")
            sys.stdout.flush()

            app.num_opponents = observed_opponents if observed_opponents is not None else DEFAULT_NUM_OPPONENTS

            pot_size = observed_pot
            
            if new_position != "Desconhecida" and app.hero_position == "Desconhecida":
                app.hero_position = new_position

            app.hero_stack = extract_hero_stack(
                frame,
                last_stack=app.hero_stack,
                hero_coords=getattr(app, 'hero_coords', None),
                require_confirmation=True,
            )
            analysis = app.process_cards(
                detected_cards=detected_cards,
                pot_size=pot_size,
                bet_to_call=bet_to_call,
                show_dashboard=True,
                action_context=turn_info,
            )
            if debug_hands and analysis.get("observation_valid") is False:
                print("[DEBUG ANÁLISE] " + json.dumps(dict(
                    captura_id=app.last_hand_observation["captura_id"],
                    resultado="SUSPENDED", motivo=analysis["reason"],
                    financeiro_observado=analysis.get("financial_observation")), ensure_ascii=False))
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        flush_uncertain()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        print("\n" + "=" * 83)
        print(f"{C_BOLD}{C_GREEN}[ENCERRADO]{C_RESET} Assistente ao vivo finalizado com sucesso pelo usuário (Ctrl + C).")
        print("MÉTRICAS DA SESSÃO AO VIVO:")
        print(f"  • Total de Ciclos Processados:       {app.total_frames_processed}")
        print(f"  • Recálculos Monte Carlo Executados:  {getattr(app, 'total_equity_calculations', 0)}")
        print(f"  • Análises de Decisão Atualizadas:    {app.total_recalculations}")
        uncertain = getattr(app, "total_uncertain_frames", 0)
        print(f"  • Capturas com Contexto Incerto:     {uncertain}")
        print(f"  • Ciclos Otimizados por Cache:       {app.total_frames_processed - app.total_recalculations - uncertain}")
        print("=" * 83 + "\n")


def main():
    print("=" * 83)
    print("POKER ANALYTICS ENGINE (PAE) - SUÍTE INTEGRADA DE VALIDAÇÃO ARQUITETURAL")
    print("Simulação do ciclo de vida: WAITING_HAND -> PRE_FLOP -> FLOP -> TURN -> RIVER")
    print("=" * 83 + "\n")

    app = PokerAnalyticsApp(
        hero_cards=["Ah", "Kh"],
        hero_stack=250.0,
        num_opponents=1,
        iterations=5000,
        allow_initial_sync=True,
        auto_detect_hero=False
    )

    # CENÁRIO 1: Início da mão (WAITING_HAND)
    print("[PASSO 1] Mesa limpa (0 cartas comunitárias) - WAITING_HAND")
    app.process_cards(
        detected_cards=["Vazio", "Vazio", "Vazio", "Vazio", "Vazio"],
        pot_size=15.0,
        bet_to_call=0.0,
        show_dashboard=True
    )

    # CENÁRIO 2: Chegada do FLOP
    print("\n[PASSO 2] Dealer distribui o FLOP: ['8h', 'Qh', 'Js']")
    app.process_cards(
        detected_cards=["8h", "Qh", "Js", "Vazio", "Vazio"],
        pot_size=45.0,
        bet_to_call=15.0,
        show_dashboard=True
    )

    # CENÁRIO 3: Chegada do TURN
    print("\n[PASSO 3] Dealer distribui o TURN: ['8h', 'Qh', 'Js', '2s']")
    app.process_cards(
        detected_cards=["8h", "Qh", "Js", "2s", "Vazio"],
        pot_size=85.0,
        bet_to_call=30.0,
        show_dashboard=True
    )

    # CENÁRIO 4: Chegada do RIVER
    print("\n[PASSO 4] Dealer distribui o RIVER: ['8h', 'Qh', 'Js', '2s', '3h']")
    analysis_river = app.process_cards(
        detected_cards=["8h", "Qh", "Js", "2s", "3h"],
        pot_size=150.0,
        bet_to_call=50.0,
        show_dashboard=True
    )
    assert analysis_river["state"] == "RIVER"
    assert analysis_river["action"] in ("CALL", "RAISE", "ALL-IN")
    print(f"[SUCESSO] River processado com perfeição! Ação: {analysis_river['action']}")

    # CENÁRIO 5: Detecção Dinâmica da Mesa Real (mesa_6.png)
    mesa_6_path = SAMPLES_DIR / "mesa_6.png"
    if not mesa_6_path.exists():
        mesa_6_path = PROJECT_ROOT / "mesa_6.png"
    if mesa_6_path.exists():
        print("\n" + "-" * 83)
        print("[PASSO 5] Processando captura real da mesa de 6 jogadores (mesa_6.png)...")
        app.auto_detect_hero = True
        app.reset_hand(reason="Reinicialização para captura independente: mesa_6.png")
        app.process_frame(
            frame_or_path=mesa_6_path,
            pot_size=None,
            bet_to_call=10.0,
            show_dashboard=True
        )

    # CENÁRIO 6: Turno Ativo do Hero com Extração de Pote e Ação (mesa_turno.png)
    mesa_turno_path = SAMPLES_DIR / "mesa_turno.png"
    if not mesa_turno_path.exists():
        mesa_turno_path = PROJECT_ROOT / "mesa_turno.png"
    if mesa_turno_path.exists():
        print("\n" + "-" * 83)
        print("[PASSO 6] Processando mesa_turno.png (Turno Ativo do Hero - Check/Passo)...")
        app.auto_detect_hero = True
        app.reset_hand(reason="Reinicialização para captura independente: mesa_turno.png")
        app.process_frame(
            frame_or_path=mesa_turno_path,
            pot_size=None,
            bet_to_call=None,
            show_dashboard=True
        )

    # CENÁRIO 7: Decisão Crítica no PRE_FLOP sob Aposta Alta (4c 2c vs Bet 945)
    print("\n" + "-" * 83)
    print("[PASSO 7] Simulação de Pre-Flop com Mão Marginal (4c 2c) sob Aposta Pesada (945.0)")
    app_preflop = PokerAnalyticsApp(
        hero_cards=["4c", "2c"],
        hero_stack=1000.0,
        num_opponents=4,
        iterations=5000,
        auto_detect_hero=False
    )
    analysis_preflop = app_preflop.process_cards(
        detected_cards=["Vazio", "Vazio", "Vazio", "Vazio", "Vazio"],
        pot_size=30.0,
        bet_to_call=945.0,
        show_dashboard=True
    )
    assert analysis_preflop["state"] == "PRE_FLOP"
    assert analysis_preflop["action"] == "FOLD"
    assert analysis_preflop["ev"] < 0
    print(f"[SUCESSO] Cenário de Pre-Flop validado! Ação: {analysis_preflop['action']} | EV: {analysis_preflop['ev']:.2f}")

    # CENÁRIO 8: Hero com Cartas Não Identificadas (Proteção Anti-Mock)
    print("\n" + "-" * 83)
    print("[PASSO 8] Verificação de Segurança: Hero com cartas não identificadas")
    app_no_hero = PokerAnalyticsApp(
        hero_cards=None,
        hero_stack=1000.0,
        num_opponents=4,
        iterations=5000,
        auto_detect_hero=False
    )
    analysis_no_hero = app_no_hero.process_cards(
        detected_cards=["8h", "Qh", "Js", "Vazio", "Vazio"],
        pot_size=50.0,
        bet_to_call=10.0,
        show_dashboard=True
    )
    assert analysis_no_hero["hero_cards"] == []
    assert analysis_no_hero["action"] == "AGUARDANDO LEITURA DAS CARTAS"
    assert analysis_no_hero["equity"] == 0.0
    print("[SUCESSO] Proteção contra Mock Fantasma validada com perfeição!\n")

    print("=" * 83)
    print("MÉTRICAS FINAIS DO SISTEMA INTEGRADO:")
    print(f"  • Total de Ciclos Processados:      {app.total_frames_processed}")
    print(f"  • Recálculos Monte Carlo Executados: {app.total_equity_calculations}")
    print(f"  • Análises de Decisão Atualizadas:   {app.total_recalculations}")
    uncertain = getattr(app, "total_uncertain_frames", 0)
    print(f"  • Capturas com Contexto Incerto:    {uncertain}")
    print(f"  • Ciclos Otimizados por Cache:      {app.total_frames_processed - app.total_recalculations - uncertain}")
    print("=" * 83 + "\n")


if __name__ == "__main__":
    if "--live" in sys.argv or "-l" in sys.argv:
        poll = POLL_INTERVAL
        for i, arg in enumerate(sys.argv):
            if arg in ("--poll", "-p") and i + 1 < len(sys.argv):
                try:
                    poll = float(sys.argv[i + 1])
                except ValueError:
                    pass
        run_live(poll_interval=poll, debug_hands="--debug-hands" in sys.argv)
    else:
        main()
