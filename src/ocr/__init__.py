"""
src/ocr
Subpacote de Leitura Óptica de Caracteres (OCR) do Poker Analytics Engine.
"""

from src.ocr.ocr_reader import (
    detect_turn_and_bet_to_call,
    extract_hero_stack,
    extract_hero_position,
    extract_pot,
    parse_action_button_text,
)

__all__ = [
    "extract_pot",
    "extract_hero_stack",
    "detect_turn_and_bet_to_call",
    "parse_action_button_text",
]
