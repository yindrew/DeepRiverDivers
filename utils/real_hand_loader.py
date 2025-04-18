import json
import pytest
from utils.real_hand_extractor import HandHistoryExtractor
from utils.poker_evaluator import evaluate_hands
from schemas.hand_history import Action, Actor, GameAction, HandHistory, Player, Street

def load_real_hands():
    # Initialize the extractor
    extractor = HandHistoryExtractor()
    
    # Read hands from river_hands.json
    with open('data/human/river_all_in_hands.json', 'r') as f:
        hands_data = json.load(f)
    

    all_hands = []
    for i, hand_data in enumerate(hands_data):
        # print(hand_data['content'])
        actions, board, player_cards, initial_stacks = extractor.extract_hand_history(hand_data['content'])
        # print("Board:", board)
        # print("Player cards:", player_cards)
        #print("Actions:", actions[:-1])


        if len(actions) == 0:
            continue
        if not player_cards:
            continue
        hero = actions[-1]['position']
        villain = actions[-2]['position']

        if hero not in player_cards or villain not in player_cards:
            continue

        # potsize for preflop
        pot_size = 0 # Initial pot (SB + BB)
        for idx, real_action in enumerate(actions):
            if 'Street' in real_action and real_action['Street'] != 'Preflop':
                pot_size += actions[idx-1]['total'] * 2
                if hero != 'Small Blind' and villain != 'Small Blind':
                    pot_size += 0.5
                if hero != 'Big Blind' and villain != 'Big Blind':
                    pot_size += 1

                if actions[idx-1]['position'] == 'Big Blind' and actions[idx-1]['action'] == 'Checks':
                    if actions[idx-2]['position'] == 'Small Blind':
                        pot_size = 2
                    if actions[idx-2]['position'] != 'Small Blind':
                        pot_size = 2.5
                break
        

        map_action_to_game_action = {
            'Checks': Action.CHECK,
            'Folds': Action.FOLD,
            'Calls': Action.CALL,
            'Bets': Action.BET,
            'Raises': Action.RAISE,
        }

        map_street_to_street = {
            'Preflop': Street.PREFLOP,
            'Flop': Street.FLOP,
            'Turn': Street.TURN,
            'River': Street.RIVER,
        }

        map_position_to_player = {
            "UTG": Player.UTG,
            "UTG+1": Player.UTG_PLUS_1,
            "UTG+2": Player.UTG_PLUS_2,
            "Dealer": Player.DEALER,
            "Small Blind": Player.SMALL_BLIND,
            "Big Blind": Player.BIG_BLIND,
        }

        game_action_list = []
        for idx, real_action in enumerate(actions[:-1]):
            if real_action['Street'] == 'Preflop':
                tempAction = GameAction(
                    action=map_action_to_game_action[real_action['action']],
                    amount=0,
                    player=map_position_to_player[real_action['position']],
                    street=map_street_to_street[real_action['Street']],
                    actor=Actor.HERO if real_action['position'] == hero else Actor.VILLAIN
                )
                game_action_list.append(tempAction)

            if real_action['Street'] != 'Preflop':
                if real_action['action'] == 'Checks' or real_action['action'] == 'Folds':
                    temp_amount = 0
                elif real_action['action'] == 'Calls' and actions[idx-1]['action'] == 'Raises':
                    temp_amount = (actions[idx]['total']) / pot_size
                    pot_size += actions[idx]['total']
                elif real_action['action'] == 'Calls' and actions[idx-1]['action'] == 'Bets':
                    temp_amount = actions[idx]['total'] / pot_size
                    pot_size += actions[idx]['total']
                elif real_action['action'] == 'Bets':
                    temp_amount = actions[idx]['total'] / pot_size
                    pot_size += actions[idx]['total']
                elif real_action['action'] == 'Raises':
                    temp_amount = actions[idx]['total'] / (pot_size + 2 * actions[idx-1]['total'])
                    pot_size += actions[idx]['total']
                elif real_action['action'] == 'All-in':
                    # Convert All-in to appropriate action type first
                    if actions[idx-1]['action'] == 'Bets' or actions[idx-1]['action'] == 'Raises':
                        actions[idx]['action'] = 'Raises'
                        temp_amount = actions[idx]['total'] / (pot_size + 2 * actions[idx-1]['total'])
                    else:  # After Checks
                        actions[idx]['action'] = 'Bets'
                        temp_amount = actions[idx]['total'] / pot_size
                    pot_size += actions[idx]['total']

                tempAction = GameAction(
                    action=map_action_to_game_action[real_action['action']],
                    amount=temp_amount,
                    player=map_position_to_player[real_action['position']],
                    street=map_street_to_street[real_action['Street']],
                    actor=Actor.HERO if real_action['position'] == hero else Actor.VILLAIN
                )
                game_action_list.append(tempAction)
        
        hero_cards =  player_cards[hero]
        hand_history = HandHistory(
            hand=hero_cards,
            board=board,
            gameLog=game_action_list
        )
        # print(player_cards)
        # print("This is the hero cards", hero_cards)
        # print("This is the villain cards", player_cards[villain])
        # print("This is the board", board)
        hero_score, villain_score = evaluate_hands(hero_cards, player_cards[villain], board)
        # # print("These are the scores", hero_score, villain_score)
        # print("hand winner", "hero" if hero_score < villain_score else "villain")
        # print("pot size", pot_size)


        if hero_score > villain_score: # if hero hand is worse than villain hand
            call_ev = -actions[-2]['total']
        elif hero_score < villain_score: # if hero hand is better than villain hand
            call_ev = pot_size
        else:
            call_ev = 0
        
        all_hands.append((hand_history, call_ev))

    return all_hands

if __name__ == "__main__":
    load_real_hands()
    pytest.main([__file__, '-v', '--capture=no'])