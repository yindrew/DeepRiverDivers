from treys import Card, Evaluator, Deck
from typing import Tuple, List, Dict, Any

def convert_to_treys_format(card_str: str) -> str:
    """Convert our card format (e.g., 'Ah') to treys format (e.g., 'Ah')"""
    # Our format is already compatible with treys!
    return card_str

def evaluate_hands(hero_cards: list[str], villain_cards: list[str], board: list[str]) -> tuple[int, int]:
    """
    Evaluate two poker hands and return their relative strengths.
    Returns:
        tuple[int, int]: (hero_score, villain_score)
        Lower score is better in treys.
    """
    evaluator = Evaluator()
    
    # Convert cards to treys format
    hero_treys = [Card.new(convert_to_treys_format(card)) for card in hero_cards]
    villain_treys = [Card.new(convert_to_treys_format(card)) for card in villain_cards]
    board_treys = [Card.new(convert_to_treys_format(card)) for card in board]
    
    # Evaluate both hands
    hero_score = evaluator.evaluate(board_treys, hero_treys)
    villain_score = evaluator.evaluate(board_treys, villain_treys)
    
    return hero_score, villain_score
