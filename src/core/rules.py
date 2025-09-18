from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RulesConfig:
    board_size: int = 9
    win_k: int = 5                     # 触发攻防的连子长度阈值（四连请设为 4）
    allow_overlap_plants: bool = False # 棋子与植物是否可同格（通常 False）
    max_turns: int = 15 * 15           # 达到后终局（也可改为判和等）

    # HP 相关（用于状态编码归一化）
    hp_max: int = 10

    # 攻防结算数值（示例，可按 4ascend 规则自行调参/替换）
    ad_attacker_hp_delta_on_fail: int = -1
    ad_defender_hp_delta_on_fail: int = +1
    ad_attacker_hp_delta_on_success: int = +1
    ad_defender_hp_delta_on_success: int = 0

    def bounds(self) -> Tuple[int, int]:
        return (self.board_size, self.board_size)
