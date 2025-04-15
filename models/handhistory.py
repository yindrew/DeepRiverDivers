from dataclasses import dataclass
from enum import Enum
from typing import override


class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    BET = 4


class Player(Enum):
    UTG = 0
    UTG_PLUS_1 = 1
    UTG_PLUS_2 = 2
    DEALER = 3
    SMALL_BLIND = 4
    BIG_BLIND = 5


class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class Actor(Enum):
    HERO = 0
    VILLAIN = 1


@dataclass
class GameAction:
    action: Action
    amount: float
    player: Player
    street: Street
    actor: Actor

    @override
    def __str__(self) -> str:
        """Return a string representation of the game action."""
        amount_str = f" {self.amount:.1f}" if self.amount > 0 else ""
        street_name = self.street.name
        player_name = self.player.name.replace("_", " ")
        action_name = self.action.name
        actor_name = "Hero" if self.actor == Actor.HERO else "Villain"
        return f"{street_name}: {player_name} {action_name}{amount_str} ({actor_name})"


@dataclass
class HandHistory:
    hand: list[str]
    board: list[str]
    gameLog: list[GameAction]

    @override
    def __str__(self) -> str:
        """Return a string representation of the hand history."""
        # Format the hand
        hand_str = "".join(self.hand)

        # Format the board
        board_str = " ".join(self.board) if self.board else "No board"

        # Format the game log
        game_log_str = "\n  ".join([str(action) for action in self.gameLog])

        return f"Hand: {hand_str}\nBoard: {board_str}\nGame Log:\n  {game_log_str}"

