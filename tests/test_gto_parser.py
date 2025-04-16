from pathlib import Path

from models.encoded_handhistory import EncodedHandHistory
from schemas.hand_history import Action, Actor, GameAction, HandHistory, Player, Street
from utils.gto_hand_parser import GTOHandParser, load_gto_hands


def test_preflop_parser():
    parser = GTOHandParser()
    parser.hero = Player.DEALER
    actions = parser.parse_preflop_action("BU 2.5 SB 12 BU c")
    assert len(actions) == 3
    assert actions[0].action == Action.RAISE
    assert actions[0].amount == 0
    assert actions[0].player == Player.DEALER
    assert actions[0].actor == Actor.HERO
    assert actions[0].street == Street.PREFLOP

    assert actions[1].action == Action.RAISE
    assert actions[1].amount == 0
    assert actions[1].player == Player.SMALL_BLIND
    assert actions[1].actor == Actor.VILLAIN

    assert actions[2].action == Action.CALL
    assert actions[2].amount == 0
    assert actions[2].player == Player.DEALER
    assert actions[2].actor == Actor.HERO

    assert parser.pot_size == 25
    assert parser.villain == Player.SMALL_BLIND

    parser.hero = Player.BIG_BLIND
    actions = parser.parse_preflop_action("BU 2.5 BB 12 BU c")
    assert parser.pot_size == 24.5

    parser.hero = Player.UTG_PLUS_2
    actions = parser.parse_preflop_action("CO 2.5 BU 12 CO c")
    assert parser.pot_size == 25.5


def test_flop_parser():
    parser = GTOHandParser()
    parser.hero = Player.DEALER
    parser.villain = Player.SMALL_BLIND
    parser.pot_size = 5.4
    board, actions = parser.parse_flop_action("Ks8h3d x 1.8 10.9 c")
    assert len(board) == 3
    assert len(actions) == 4
    assert actions[0].action == Action.CHECK
    assert actions[0].amount == 0
    assert actions[0].player == Player.SMALL_BLIND
    assert actions[0].actor == Actor.VILLAIN

    assert actions[1].action == Action.BET
    assert round(actions[1].amount, 2) == 0.33
    assert actions[1].player == Player.DEALER
    assert actions[1].actor == Actor.HERO

    assert actions[2].action == Action.RAISE
    assert round(actions[2].amount, 1) == 1
    assert actions[2].player == Player.SMALL_BLIND
    assert actions[2].actor == Actor.VILLAIN

    assert actions[3].action == Action.CALL
    assert round(actions[3].amount, 1) == 0.5
    assert actions[3].player == Player.DEALER
    assert actions[3].actor == Actor.HERO

    assert round(parser.pot_size, 1) == 27.2


def test_turn_parser():
    parser = GTOHandParser()
    parser.hero = Player.DEALER
    parser.villain = Player.SMALL_BLIND
    parser.pot_size = 21

    board, actions = parser.parse_turn_or_river_action("9h x 10.5 22.5 c", Street.TURN)
    assert board == "9h"
    assert len(actions) == 4
    assert actions[0].action == Action.CHECK
    assert actions[0].amount == 0
    assert actions[0].player == Player.SMALL_BLIND
    assert actions[0].actor == Actor.VILLAIN

    assert actions[1].action == Action.BET
    assert actions[1].amount == 0.5
    assert actions[1].player == Player.DEALER
    assert actions[1].actor == Actor.HERO

    assert actions[2].action == Action.RAISE
    assert round(actions[2].amount, 2) == 0.43
    assert actions[2].player == Player.SMALL_BLIND
    assert actions[2].actor == Actor.VILLAIN

    assert actions[3].action == Action.CALL
    assert round(actions[3].amount, 2) == 0.22
    assert actions[3].player == Player.DEALER
    assert actions[3].actor == Actor.HERO

    assert parser.pot_size == 66


def test_pot_size():
    parser = GTOHandParser()
    parser.pot_size = 10
    parser.hero = Player.DEALER
    parser.villain = Player.SMALL_BLIND
    board, actions = parser.parse_turn_or_river_action("9h x 10 20 50 c", Street.TURN)

    assert actions[0].action == Action.CHECK
    assert actions[0].amount == 0
    assert actions[0].player == Player.SMALL_BLIND
    assert actions[0].actor == Actor.VILLAIN

    assert actions[1].action == Action.BET
    assert actions[1].amount == 1
    assert actions[1].player == Player.DEALER
    assert actions[1].actor == Actor.HERO

    assert actions[2].action == Action.RAISE
    assert actions[2].amount == 0.5
    assert actions[2].player == Player.SMALL_BLIND
    assert actions[2].actor == Actor.VILLAIN

    assert parser.pot_size == 110


def test_whole_hand_parser():
    parser = GTOHandParser()
    hand_histories = parser.parse_hand_file("data/gto/bu_vs_bb_3bp.txt")
    encoded_hand, ev = (
        hand_histories[1]["encoded_hand_history"],
        hand_histories[1]["expected_ev"],
    )
    print(f"\nHand History (EV: {ev}):")
    print(EncodedHandHistory.decode_to_string(encoded_hand))


def test_all_hands():
    path = Path(__file__).parent.parent / "data" / "gto"
    hand_histories = load_gto_hands(path)
    encoded_hand, ev = (
        hand_histories[0]["encoded_hand_history"],
        hand_histories[0]["expected_ev"],
    )
    print(f"\nHand History (EV: {ev}):")
    print(EncodedHandHistory.decode_to_string(encoded_hand))
