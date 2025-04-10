from dataclasses import dataclass
from typing import List
from enum import Enum


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


@dataclass
class HandHistory:
    hand: List[str]
    board: List[str]
    gameLog: List[GameAction]