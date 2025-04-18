import re
from typing import List, Optional, Dict
from schemas.hand_history import HandHistory, GameAction, Action, Actor, Street, Player

class HandHistoryExtractor:
    """Extracts hand content into HandHistory objects"""
    
    def __init__(self):
        self.hole_card_tag = '*** HOLE CARDS ***'
        self.river_tag = "*** RIVER ***"
        self.flop_tag = "*** FLOP ***"
        self.turn_tag = "*** TURN ***"
        
        self.stakes = None
        self.initial_stacks = {}

    def post_flop_sequencing(self, hero: Player, villain: Player) -> bool:
            """
            Check if the hero goes before the villain postflop
            """
            if hero == Player.UTG:
                if (
                    villain == Player.UTG_PLUS_1
                    or villain == Player.UTG_PLUS_2
                    or villain == Player.DEALER
                ):
                    return True
                else:
                    return False
            elif hero == Player.UTG_PLUS_1:
                if villain == Player.UTG_PLUS_2 or villain == Player.DEALER:
                    return True
                else:
                    return False
            elif hero == Player.UTG_PLUS_2:
                if villain == Player.DEALER:
                    return True
                else:
                    return False
            elif hero == Player.DEALER:
                return False
            elif hero == Player.SMALL_BLIND:
                return True
            else:
                if villain == Player.SMALL_BLIND:
                    return False
                else:
                    return True

    def extract_initial_stacks(self, hand_raw: str) -> Dict[str, float]:
        """Extract initial stack sizes for all players"""
        # Pattern to match seat info with stack size
        stack_pattern = r'Seat \d+: ((?:UTG(?:\+[12])?|BB|SB|Dealer|Small Blind|Big Blind)(?:\s*\[ME\])?) \((\$[\d,]+\.?\d*) in chips\)'
        
        # Get all stacks
        stacks = {}
        for match in re.finditer(stack_pattern, hand_raw):
            position = match.group(1).strip()
            position = re.sub(r'\s*\[ME\]\s*', '', position).strip()
            # Remove $ and commas, then convert to float
            stack_size = float(match.group(2).replace('$', '').replace(',', ''))
            stacks[position] = stack_size / self.stakes
            
        self.initial_stacks = stacks
        return stacks

    def extract_board_cards(self, hand_raw: str) -> List[str]:
        """Extract board cards from the hand history"""
        board = []
        
        # Extract flop
        flop_match = re.search(r'\*\*\* FLOP \*\*\* \[(.*?)\]', hand_raw)
        if flop_match:
            board.extend(flop_match.group(1).split())
            
        # Extract turn
        turn_match = re.search(r'\*\*\* TURN \*\*\* \[.*?\] \[(.*?)\]', hand_raw)
        if turn_match:
            board.append(turn_match.group(1))
            
        # Extract river
        river_match = re.search(r'\*\*\* RIVER \*\*\* \[.*?\] \[(.*?)\]', hand_raw)
        if river_match:
            board.append(river_match.group(1))
            
        return board

    def extract_bb(self, hand_raw: str) -> float:
        """Extract the big blind from the hand history"""
        # First try to get stakes from the table info
        stakes_match = re.search(r'Stakes: \$(\d+(?:\.\d+)?)-\$(\d+(?:\.\d+)?)', hand_raw)
        if stakes_match:
            self.stakes = float(stakes_match.group(2))  # Use the big blind amount
            return self.stakes

        # If not found, try to match Big Blind action with optional [ME] tag and dollar sign
        bb_match = re.search(r'Big Blind\s*(?:\[ME\])?\s*:\s*Big [Bb]lind \$(\d+(?:\.\d+)?)', hand_raw)
        if not bb_match:
            raise ValueError("Could not find big blind amount in hand history")
        
        self.stakes = float(bb_match.group(1))
        if self.stakes == 0:
            raise ValueError("Big blind amount cannot be zero")
        return self.stakes
    

    def extract_preflop_actions(self, hand_raw: str) -> List[GameAction]:
        """Extract the preflop actions from the hand history"""
        # Find the section between HOLE CARDS and FLOP
        sections = hand_raw.split("*** HOLE CARDS ***")
        if len(sections) < 2:
            return []
            
        preflop_section = sections[1].split("*** FLOP ***")[0] if "*** FLOP ***" in sections[1] else sections[1]
        
        # Pattern to match betting actions including checks
        action_pattern = r'(UTG(?:\+[12])?|BB|SB|Dealer|Small Blind|Big Blind)\s*(?:\[ME\])?\s*:\s*(Raises|Calls|Folds|Checks)(?:\s+\$(\d+(?:\.\d+)?)\s+to\s+\$(\d+(?:\.\d+)?))?'
        
        actions = []
        for match in re.finditer(action_pattern, preflop_section):
            # Get position and clean it
            position = match.group(1).strip()
            position = re.sub(r'\s*\[ME\]\s*', '', position).strip()

            action_type = match.group(2)
            if action_type == 'Folds':
                continue

            action_data = {
                'position': position,
                'action': action_type,
                'Street': 'Preflop'
            }
            
            # Handle different action types
            if action_type == 'Raises' and match.group(3) and match.group(4):
                action_data['total'] = float(match.group(4)) / self.stakes
            elif action_type == 'Calls':
                if len(actions) >= 1:
                    action_data['total'] = actions[-1]['total']
                else:
                    action_data['total'] = 1.0
            elif action_type == 'Checks':
                action_data['total'] = 0.0

            actions.append(action_data)
        return actions

    def extract_postflop_actions(self, hand_raw: str) -> List[Dict]:
        """Extract the postflop actions from the hand history"""
        # Find the section between FLOP and TURN
        sections = hand_raw.split("*** FLOP ***")
        if len(sections) < 2:
            return []
    
        flop_section = sections[1].split("*** SUMMARY ***")[0]
    
        # Pattern to match betting actions including checks, bets, and all-ins
        action_pattern = r'(?:UTG(?:\+[12])?|BB|SB|Dealer|Small Blind|Big Blind)\s*(?:\[ME\])?\s*:\s*(Raises|Calls|Folds|Checks|Bets|All-in(?:\(raise\))?)(?:\s+\$(\d+\.?\d*)(?:\s+to\s+\$(\d+\.?\d*))?)?'
    
        actions = []
        numberOfCalls = 0
        for match in re.finditer(action_pattern, flop_section):
            # Get position and clean it
            position = match.group(0).split(':')[0].strip()
            position = re.sub(r'\s*\[ME\]\s*', '', position).strip()
    
            action_type = match.group(1)
            # Strip "(raise)" from All-in action type
            if '(raise)' in action_type:
                action_type = 'All-in'
            
            action_data = {
                'position': position,
                'action': action_type
            }

            if action_type == 'Raises':
                action_data['total'] = (float(match.group(3)) / self.stakes)
                if numberOfCalls == 0:
                    action_data['Street'] = 'Flop'
                elif numberOfCalls == 1:
                    action_data['Street'] = 'Turn'
                elif numberOfCalls == 2:
                    action_data['Street'] = 'River'
            # Add amounts for betting actions
            if action_type in ['Calls', 'Bets', 'All-in'] and match.group(2):
                if match.group(3):  # If there's a "to" amount (for raises)
                    action_data['total'] = float(match.group(3)) / self.stakes
                else:
                    action_data['total'] = float(match.group(2)) / self.stakes
                if numberOfCalls == 0:
                    action_data['Street'] = 'Flop'
                elif numberOfCalls == 1:
                    action_data['Street'] = 'Turn'
                elif numberOfCalls == 2:
                    action_data['Street'] = 'River'
                if action_type == 'Calls':
                    numberOfCalls += 1
    
            if action_type == 'Checks':
                action_data['total'] = 0
                if numberOfCalls == 0:
                    action_data['Street'] = 'Flop'
                elif numberOfCalls == 1:
                    action_data['Street'] = 'Turn'
                elif numberOfCalls == 2:
                    action_data['Street'] = 'River'
                # if in position and checks, then move onto next street 
                if len(actions) >= 1 and not self.post_flop_sequencing(action_data['position'], actions[-1]['position']):
                    numberOfCalls += 1

            actions.append(action_data)
        return actions

    def extract_all_player_cards(self, hand_raw: str) -> Optional[List[str]]:
        """Extract all player's cards from the hand history"""
        # Look for any line with hole cards and capture both position and cards
        # Make [ME] optional with (?:\s*\[ME\])?
        hero_pattern = r'(?:UTG(?:\+[12])?|BB|SB|Dealer|Small Blind|Big Blind)\s*(?:\[ME\])?\s*:\s*(?:Card dealt to a spot )?\[(.*?)\]'
        
        player_cards = {}
        # Find all matches in the hand history
        matches = re.finditer(hero_pattern, hand_raw)
        for match in matches:
            # Get the position and clean it up by removing [ME] and extra whitespace
            position = match.group(0).split(':')[0].strip()
            position = re.sub(r'\s*\[ME\]\s*', '', position).strip()
            cards = match.group(1).split()  # Get the cards from the capture group
            player_cards[position] = cards

        return player_cards

    def extract_hand_history(self, hand_raw: str) -> tuple[list, list, dict, dict]:
        """Extract a single hand history from raw content"""
        # Extract big blind amount
        self.extract_bb(hand_raw)
        if not hand_raw.strip():
            print("DEBUG: Empty hand content")
            return [], [], {}, {}
            
        if self.river_tag not in hand_raw:
            print("DEBUG: No river tag found")
            return [], [], {}, {}
            
        # Extract initial stack sizes
        initial_stacks = self.extract_initial_stacks(hand_raw)
            
        # Extract all components
        board = self.extract_board_cards(hand_raw)
        if not board or len(board) != 5:
            print("DEBUG: Invalid board cards:", board)
            return [], [], {}, {}
            
        # Extract player cards
        player_cards = self.extract_all_player_cards(hand_raw)
        if not player_cards:
            print("DEBUG: No player cards found")
            return [], [], {}, {}

        # Extract all actions
        preflop_actions = self.extract_preflop_actions(hand_raw)
        postflop_actions = self.extract_postflop_actions(hand_raw)
        
        # Combine all actions
        all_actions = preflop_actions + postflop_actions

        # Fix the total for the last two actions
        jammer = all_actions[-2]['position']
        caller = all_actions[-1]['position']

        if initial_stacks[caller] < initial_stacks[jammer]:
            total_bet_so_far = (initial_stacks[jammer] - all_actions[-2]['total'])
            real_bet = initial_stacks[caller] - total_bet_so_far
            all_actions[-2]['total'] = real_bet

        return all_actions, board, player_cards, initial_stacks


    def extract_hand_histories(self, content: str) -> List[HandHistory]:
        """Extract multiple hand histories from content"""
        hands_raw = content.strip().split("\n\n")
        parsed_hands = []
        
        for hand_raw in hands_raw:
            hand_history = self.extract_hand_history(hand_raw)
            if hand_history:
                parsed_hands.append(hand_history)
        
        return parsed_hands 