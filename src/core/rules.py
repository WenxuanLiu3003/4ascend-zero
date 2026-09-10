from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RulesConfig:
    board_size: int = 9
    win_k: int = 4                    
    allow_overlap_plants: bool = False 
    max_turns: int = 15 * 15        

    hp_max: int = 6

    # False: requeue survivors; True: survivors continue cancelling immediately.
    relay_cancellation: bool = True

    # expired variables, useless in current rules; kept for compatibility with old checkpoints
    ad_attacker_hp_delta_on_fail: int = -1
    ad_defender_hp_delta_on_fail: int = +1
    ad_attacker_hp_delta_on_success: int = +1
    ad_defender_hp_delta_on_success: int = 0

    def bounds(self) -> Tuple[int, int]:
        return (self.board_size, self.board_size)
