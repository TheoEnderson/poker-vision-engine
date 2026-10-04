"""
main.py
Ponto de entrada do Poker Analytics Engine (PAE).
Orquestrador Mestre integrando Visão Computacional, OCR, Máquina de Estados e Motor de Risco.
"""

import os
import sys
import time
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
)
from src.vision import (
    detect_active_opponents,
    detect_board,
    find_and_detect_hero_cards,
    grab_screen,
)


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
        self.state_machine = PokerHandStateMachine(allow_initial_sync=allow_initial_sync)
        self.dataset_logger = DatasetLogger(str(PROJECT_ROOT / "dataset_partidas.csv"))
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
            hero_detected = find_and_detect_hero_cards(frame_or_path, verbose=False)
            has_valid_detected = (
                hero_detected
                and len(hero_detected) == 2
                and all(c and c != "Desconhecida" for c in hero_detected)
            )

            already_has_valid_hand = (
                isinstance(self.hero_cards, list)
                and len(self.hero_cards) == 2
                and all(
                    isinstance(c, str)
                    and len(c) >= 2
                    and c.lower() not in ("desconhecida", "vazio", "none", "--")
                    for c in self.hero_cards
                )
            )

            # Iron Lock Estrito: preserva a mão sólida até o fim da rodada
            if has_valid_detected and not already_has_valid_hand:
                self.hero_cards = hero_detected
            elif not already_has_valid_hand:
                self.hero_cards = []

        # 2. Leitura numérica do pote via OCR
        actual_pot = pot_size
        if self.auto_detect_pot and actual_pot is None:
            extracted_pot = extract_pot(frame_or_path, verbose=False)
            if extracted_pot > 0:
                actual_pot = extracted_pot

        if actual_pot is None:
            actual_pot = 85.0

        # 3. Leitura do turno e aposta a pagar (bet_to_call) via OCR
        actual_bet = bet_to_call
        if self.auto_detect_turn and actual_bet is None:
            turn_info = detect_turn_and_bet_to_call(frame_or_path, verbose=False)
            if turn_info["is_hero_turn"]:
                actual_bet = turn_info["bet_to_call"]

        if actual_bet is None:
            actual_bet = 30.0

        # 4. Detecção das cartas comunitárias do Board
        detected_cards = detect_board(frame_or_path, verbose=False)
        return self.process_cards(detected_cards, actual_pot, actual_bet, show_dashboard)

    def process_cards(
        self,
        detected_cards: List[str],
        pot_size: float = 85.0,
        bet_to_call: float = 30.0,
        show_dashboard: bool = True
    ) -> Dict[str, Any]:
        """
        Processa as cartas detectadas na mesa com controle estrito de ciclo de vida,
        mecanismo de cache inteligente e avaliação de Valor Esperado (EV).
        """
        self.total_frames_processed += 1

        # 1. Atualiza a Máquina de Estados Finitos (FSM)
        prev_state = self.last_state
        self.state_machine.update_board(detected_cards)
        current_state = self.state_machine.current_state
        current_board = self.state_machine.board

        # Destrancamento Seguro entre Rodadas
        if self.auto_detect_hero:
            if current_state == HandState.WAITING_HAND:
                self.hero_cards = []
            elif current_state == HandState.PRE_FLOP and prev_state not in (HandState.PRE_FLOP, HandState.WAITING_HAND):
                self.hero_cards = []
            elif len(current_board) == 0 and len(self.last_board) > 0:
                self.hero_cards = []

        # 2. Otimização de Performance: Verifica se houve alteração na mesa
        last_pot = self.last_analysis.get("pot_size") if self.last_analysis else None
        last_bet = self.last_analysis.get("bet_to_call") if self.last_analysis else None
        last_hero = self.last_analysis.get("hero_cards") if self.last_analysis else None
        has_changed = (
            (current_board != self.last_board)
            or (current_state != self.last_state)
            or (pot_size != last_pot)
            or (bet_to_call != last_bet)
            or (self.hero_cards != last_hero)
        )

        hero_is_missing = (
            not isinstance(self.hero_cards, list)
            or len(self.hero_cards) < 2
            or any(c.lower() in ("desconhecida", "vazio", "none", "--") for c in self.hero_cards)
        )
        if hero_is_missing:
            has_changed = True

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

        if not has_valid_hero:
            analysis = {
                "state": current_state.value,
                "board": current_board,
                "hero_cards": [],
                "hero_hand": "Não identificadas",
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
                "pot_odds": 0.0,
                "equity": 0.0,
                "p_win": 0.0,
                "p_tie": 0.0,
                "p_lose": 0.0,
                "ev": 0.0,
                "action": "AGUARDANDO LEITURA DAS CARTAS",
                "recommended_amount": 0.0,
                "reason": "Cartas do Hero não identificadas com precisão na mesa. Simulação fictícia bloqueada.",
                "sim_time": 0.0,
                "recalculated": False,
            }
            self.last_analysis = analysis
            if show_dashboard:
                self.render_dashboard(analysis, cached=False)
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

            # Simulação de Monte Carlo para cálculo de Equity
            p_win, p_tie, p_lose, equity, elapsed = calculate_equity(
                hero_cards=self.hero_cards,
                board_cards=current_board,
                num_opponents=self.num_opponents,
                iterations=self.iterations
            )

            outs, draw_name = detect_draws(self.hero_cards, current_board)

            # Motor de Risco e Tomada de Decisão
            decision = make_decision(
                p_win=p_win,
                p_lose=p_lose,
                p_tie=p_tie,
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
                "hero_hand": hero_hand_desc,
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
                "pot_odds": decision["pot_odds"],
                "equity": decision["equity"],
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

        else:
            action = "CHECK" if bet_to_call == 0 else "FOLD"
            analysis = {
                "state": current_state.value,
                "board": current_board,
                "hero_cards": self.hero_cards,
                "hero_hand": "Aguardando cartas suficientes",
                "pot_size": pot_size,
                "bet_to_call": bet_to_call,
                "hero_stack": self.hero_stack,
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
        print(f"╠{sep_double}╣")
        print(f"║ {C_BOLD}DADOS DA RODADA:{C_RESET}")
        print(
            f"║   • Pote Atual:        {a['pot_size']:8.1f} fichas   │ "
            f"• Aposta a Pagar (Call): {a['bet_to_call']:8.1f} fichas"
        )
        print(
            f"║   • Stack do Hero:     {a['hero_stack']:8.1f} fichas   │ "
            f"• Pot Odds Exigidas:     {a['pot_odds']:8.2f}%"
        )
        print(f"╠{sep_double}╣")
        print(f"║ {C_BOLD}PROBABILIDADES (MONTE CARLO - {self.iterations:,} iterações):{C_RESET}")
        print(
            f"║   • Taxa de Vitória (P_win): {a['p_win']:6.2f}%  │ "
            f"• Equity Consolidada:  {a['equity']:8.2f}%"
        )
        print(
            f"║   • Taxa de Derrota (P_lose): {a['p_lose']:5.2f}%  │ "
            f"• Tempo de Simulação:  {a['sim_time']:8.3f}s"
        )
        print(f"╠{sep_double}╣")
        print(f"║ {C_BOLD}ANÁLISE DE RISCO & VALOR ESPERADO (EV):{C_RESET}")
        print(f"║   • Expectativa Matemática (EV): {ev_color}{ev_val:+8.2f} fichas{C_RESET}")
        print("║")
        print(f"║   >>> DECISÃO RECOMENDADA: {action_color}[ {a['action']} ]{C_RESET} ")
        if a["recommended_amount"] > 0:
            print(f"║   • Valor Recomendado:     {a['recommended_amount']:.1f} fichas")
        print("║")
        print(f"║   • Justificativa: {a['reason']}")
        print(f"╚{sep_double}╝\n")


def run_live(poll_interval: float = POLL_INTERVAL):
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

    try:
        while True:
            frame = grab_screen()
            turn_info = detect_turn_and_bet_to_call(frame, verbose=False)

            if not turn_info["is_hero_turn"]:
                sys.stdout.write(f"\r{C_YELLOW}[AGUARDANDO SUA VEZ DE JOGAR...]{C_RESET}   ")
                sys.stdout.flush()
                time.sleep(poll_interval)
                continue

            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()

            try:
                active_opponents = detect_active_opponents(frame)
                app.num_opponents = active_opponents
            except Exception:
                app.num_opponents = DEFAULT_NUM_OPPONENTS

            hero_detected = find_and_detect_hero_cards(frame, verbose=False)
            has_valid_detected = (
                hero_detected
                and len(hero_detected) == 2
                and all(c and c != "Desconhecida" for c in hero_detected)
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

            # Iron Lock Estrito: Só atualiza se ainda não tivermos uma mão sólida
            if has_valid_detected and not already_has_valid_hand:
                app.hero_cards = hero_detected
            elif not already_has_valid_hand:
                app.hero_cards = []

            detected_cards = detect_board(frame, verbose=False)

            pot_size = extract_pot(frame, verbose=False)
            if pot_size <= 0:
                pot_size = app.last_analysis.get("pot_size", 85.0) if app.last_analysis else 85.0

            app.hero_stack = extract_hero_stack(frame, last_stack=app.hero_stack)
            bet_to_call = turn_info["bet_to_call"]

            app.process_cards(
                detected_cards=detected_cards,
                pot_size=pot_size,
                bet_to_call=bet_to_call,
                show_dashboard=True
            )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        print("\n" + "=" * 83)
        print(f"{C_BOLD}{C_GREEN}[ENCERRADO]{C_RESET} Assistente ao vivo finalizado com sucesso pelo usuário (Ctrl + C).")
        print("MÉTRICAS DA SESSÃO AO VIVO:")
        print(f"  • Total de Ciclos Processados:       {app.total_frames_processed}")
        print(f"  • Recálculos Monte Carlo Executados:  {app.total_recalculations}")
        print(f"  • Ciclos Otimizados por Cache:       {app.total_frames_processed - app.total_recalculations}")
        print("=" * 83 + "\n")


def main():
    print("=" * 83)
    print("POKER ANALYTICS ENGINE (PAE) - SUÍTE INTEGRADA DE VALIDAÇÃO ARQUITETURAL")
    print("Simulação do ciclo de vida: WAITING_HAND -> FLOP -> TURN -> RIVER -> PRE_FLOP")
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
        app.process_cards(
            detected_cards=["Vazio", "Vazio", "Vazio", "Vazio", "Vazio"],
            show_dashboard=False
        )
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
        app.process_cards(
            detected_cards=["Vazio", "Vazio", "Vazio", "Vazio", "Vazio"],
            show_dashboard=False
        )
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
    print(f"  • Recálculos Monte Carlo Executados: {app.total_recalculations}")
    print(f"  • Ciclos Otimizados por Cache:      {app.total_frames_processed - app.total_recalculations}")
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
        run_live(poll_interval=poll)
    else:
        main()
