from pathlib import Path

from models.encoded_handhistory import EncodedHandHistory
from schemas.dataloader_datatypes import DatasetBaseType
from schemas.hand_history import Action, Actor, GameAction, HandHistory, Player, Street


class GTOHandParser:
    """
    Parser for GTO hand text files with structured format.
    Main function parse_hand_file parses a GTO txt file.
    """

    def __init__(self):

        # Mapping for player positions
        self.player_map = {
            "UTG": Player.UTG,
            "HJ": Player.UTG_PLUS_1,
            "CO": Player.UTG_PLUS_2,
            "BU": Player.DEALER,
            "SB": Player.SMALL_BLIND,
            "BB": Player.BIG_BLIND,
        }

        # Mapping for actions
        self.action_map = {
            "x": Action.CHECK,
            "c": Action.CALL,
            "f": Action.FOLD,
        }

        self.hero = None
        self.villain = None
        self.pot_size = 0

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

    def set_hero(self, line: str) -> Player:
        self.hero = self.player_map[line]
        return self.hero

    def parse_preflop_action(self, line: str) -> list[GameAction]:
        """
        Parse the preflop action line.
        Handles formats like:
        - "BU 2.5 SB 12 BU c"
        - "SB c BB 3 SB 14 BB c"
        - "CO 2.5 BU 7.5 CO c"
        - "UTG 2 SB 10 UTG c"
        """
        parts = line.strip().split()
        actions: list[GameAction] = []
        for i in range(0, len(parts), 2):
            temp_action = (
                self.action_map[parts[i + 1]]
                if parts[i + 1] in self.action_map
                else Action.RAISE
            )
            temp_player = self.player_map[parts[i]]
            temp_actor = Actor.HERO if temp_player == self.hero else Actor.VILLAIN
            if temp_player != self.hero:
                self.villain = temp_player

            action = GameAction(
                action=temp_action,
                amount=0,
                player=temp_player,
                street=Street.PREFLOP,
                actor=temp_actor,
            )
            actions.append(action)

        # given we have a call at the end no matter what, we can just get the last raise amount * 2.
        self.pot_size = float(parts[-3]) * 2
        if self.hero != Player.BIG_BLIND and self.villain != Player.BIG_BLIND:
            self.pot_size += 1
        if self.hero != Player.SMALL_BLIND and self.villain != Player.SMALL_BLIND:
            self.pot_size += 0.5

        return actions

    def parse_flop_action(self, line: str) -> tuple[list[str], list[GameAction]]:
        """Parse a flop action line.
        Examples:
        Ks8h3d x 1.8 10.9 c
        Ts5h3c 12.5 c
        QhTd4h x 1.5 9 c
        """
        parts = line.strip().split()
        actions = []

        # add the board cards to the board
        board = []
        for i in range(0, len(parts[0]), 2):
            if i + 1 < len(parts[0]):
                card = parts[0][i : i + 2]
                board.append(card)

        # add the actions to the actions list
        previous_action = None
        hero_is_oop = self.post_flop_sequencing(self.hero, self.villain)

        # Process actions starting from index 1
        temp_pot_size = self.pot_size
        for i in range(1, len(parts)):
            # Determine current player and actor
            if i % 2 == 1 and hero_is_oop:  # If OOP turn and hero is OOP
                temp_player = self.hero
                temp_actor = Actor.HERO
            elif i % 2 == 1 and not hero_is_oop:  # If OOP turn and hero is not OOP
                temp_player = self.villain
                temp_actor = Actor.VILLAIN
            elif i % 2 == 0 and hero_is_oop:  # If IP turn and hero is OOP
                temp_player = self.villain
                temp_actor = Actor.VILLAIN
            elif i % 2 == 0 and not hero_is_oop:  # If IP turn and hero is not OOP
                temp_player = self.hero
                temp_actor = Actor.HERO

            # Determine current action
            if parts[i] in self.action_map:
                temp_action = self.action_map[parts[i]]
            elif previous_action and previous_action == Action.BET:
                temp_action = Action.RAISE
            else:
                temp_action = Action.BET

            # Determine current amount
            if (
                temp_action == Action.CALL and previous_action == Action.RAISE
            ):  # if you are calling a raise, you need to factor in your original bet size.
                temp_amount = (
                    float(parts[i - 1]) - float(parts[i - 2])
                ) / self.pot_size
                self.pot_size += float(parts[i - 1]) - float(parts[i - 2])
            elif temp_action == Action.CALL and previous_action == Action.BET:
                temp_amount = float(parts[i - 1]) / self.pot_size
                self.pot_size += float(parts[i - 1])
            elif temp_action == Action.BET:
                temp_amount = float(parts[i]) / self.pot_size
                self.pot_size += float(parts[i])
            elif temp_action == Action.RAISE:
                temp_amount = float(parts[i]) / (
                    self.pot_size + 2 * float(parts[i - 1])
                )
                self.pot_size += float(parts[i])
            elif temp_action == Action.CHECK or temp_action == Action.FOLD:
                temp_amount = 0

            action = GameAction(
                action=temp_action,
                amount=temp_amount,
                player=temp_player,
                street=Street.FLOP,
                actor=temp_actor,
            )
            previous_action = temp_action
            actions.append(action)
        self.pot_size = (
            temp_pot_size + (float(parts[-2]) * 2)
            if previous_action != Action.CHECK
            else self.pot_size
        )
        return board, actions

    def parse_turn_or_river_action(
        self, line: str, current_street: Street
    ) -> tuple[str, list[GameAction]]:
        """Parse a turn or river action line.
        Examples:
        9h x 10.5 22.5 c
        Ts 12.5 c
        Ad x 22.5 c
        """
        parts = line.strip().split()
        actions: list[GameAction] = []
        # add the board cards to the board
        board = parts[0]

        # add the actions to the actions list
        previous_action = None
        hero_is_oop = self.post_flop_sequencing(self.hero, self.villain)
        temp_pot_size = self.pot_size

        # Process actions starting from index 1
        for i in range(1, len(parts)):
            # Determine current player and actor
            if i % 2 == 1 and hero_is_oop:  # If OOP turn and hero is OOP
                temp_player = self.hero
                temp_actor = Actor.HERO
            elif i % 2 == 1 and not hero_is_oop:  # If OOP turn and hero is not OOP
                temp_player = self.villain
                temp_actor = Actor.VILLAIN
            elif i % 2 == 0 and hero_is_oop:  # If IP turn and hero is OOP
                temp_player = self.villain
                temp_actor = Actor.VILLAIN
            elif i % 2 == 0 and not hero_is_oop:  # If IP turn and hero is not OOP
                temp_player = self.hero
                temp_actor = Actor.HERO

            # Determine current action
            if parts[i] in self.action_map:
                temp_action = self.action_map[parts[i]]
            elif previous_action and previous_action == Action.BET:
                temp_action = Action.RAISE
            else:
                temp_action = Action.BET

            # Determine current amount
            if (
                temp_action == Action.CALL and previous_action == Action.RAISE
            ):  # if you are calling a raise, you need to factor in your original bet size.
                temp_amount = (
                    float(parts[i - 1]) - float(parts[i - 2])
                ) / self.pot_size
                self.pot_size += float(parts[i - 1]) - float(parts[i - 2])
            elif temp_action == Action.CALL and previous_action == Action.BET:
                temp_amount = float(parts[i - 1]) / self.pot_size
                self.pot_size += float(parts[i - 1])
            elif temp_action == Action.BET:
                temp_amount = float(parts[i]) / self.pot_size
                self.pot_size += float(parts[i])
            elif temp_action == Action.RAISE:
                temp_amount = float(parts[i]) / (
                    self.pot_size + 2 * float(parts[i - 1])
                )
                self.pot_size += float(parts[i])
            elif temp_action == Action.CHECK or temp_action == Action.FOLD:
                temp_amount = 0

            action = GameAction(
                action=temp_action,
                amount=temp_amount,
                player=temp_player,
                street=current_street,
                actor=temp_actor,
            )
            previous_action = temp_action

            actions.append(action)
        self.pot_size = (
            temp_pot_size + (float(parts[-2]) * 2)
            if Street.TURN == current_street
            else self.pot_size
        )
        return board, actions

    def parse_hand_file(self, file_path: str) -> list[DatasetBaseType]:
        """Parse a GTO hand text file.
        Format:
        Line 1: Hero position (e.g., "BU")
        Line 2: Preflop action (e.g., "BU 2.5 SB 12 BU c")
        Line 3: Flop cards and action (e.g., "7h6h3d x 8.75 c")
        Line 4: Turn card and action (e.g., "Jc x 22 c")
        Line 5: River card and action (e.g., "Kc 56.25")
        Remaining lines: EV values for different hands (e.g., "AKsh 113.77")
        """
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        self.pot_size = 0
        self.hero = None
        self.villain = None

        # Parse hero position
        self.set_hero(lines[0])

        # Parse preflop action
        preflop_actions = self.parse_preflop_action(lines[1])

        # Parse flop
        flop_board, flop_actions = self.parse_flop_action(lines[3])

        # Parse turn
        turn_card, turn_actions = self.parse_turn_or_river_action(lines[4], Street.TURN)

        # Parse river
        river_card, river_actions = self.parse_turn_or_river_action(
            lines[5], Street.RIVER
        )

        # Combine all board cards
        board = flop_board + [turn_card] + [river_card]

        # Combine all actions
        game_log = preflop_actions + flop_actions + turn_actions + river_actions

        # Parse EV lines
        hand_histories: list[DatasetBaseType] = []
        for line in lines[6:]:
            if not line:
                continue

            parts = line.strip().split()

            hand_str = parts[0]

            ev = float(parts[1])
            # Split the hand string into two cards
            # Example: "AKsh" -> ["As", "Kh"]
            rank1, rank2 = hand_str[0], hand_str[1]
            suit1, suit2 = hand_str[2], hand_str[3]
            hand = [rank1 + suit1, rank2 + suit2]
            hand_history = HandHistory(hand=hand, board=board, gameLog=game_log)
            encoded_hand_history = EncodedHandHistory.encode_hand_history(hand_history)
            hand_histories.append(
                {"expected_ev": ev, "encoded_hand_history": encoded_hand_history}
            )

        return hand_histories


def load_gto_hands(path: Path) -> list[DatasetBaseType]:
    parser = GTOHandParser()
    hand_histories: list[DatasetBaseType] = []
    for txt_file in path.glob("*.txt"):
        hand_histories.extend(parser.parse_hand_file(str(path / txt_file)))
    return hand_histories
