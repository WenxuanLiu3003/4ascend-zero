# ──────────────────────────────────────────────────────────────────────────────
# File: src/core/engine.py （移除 AttackContext；不使用 ns.ad_ctx；按 a,b,c,d 规则结算伤害）
# 说明：
#   - 攻防结算：a=攻方有效stone，b=攻方有效plant，c=防守方有效stone，d=防守方有效plant，
#     伤害 = |(a+b) - (c+d)|
#   - a/b 的“被占据扣减”仅需检查“防守方当前 move 是否占据攻方 chain 中的某一格”，若是则各减 1。
#   - 攻防结束后清除参与格子的植物，并强制刷新植物。
#   - 默认每 10 手常规刷新 2 个植物（空位且计数<2）。
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Set, Tuple, Iterable
import numpy as np
import random

from .types import Player, Move, Phase
from .state import GameState
from .board import Board

# 四个方向：水平/垂直/两对角
_DIRS = [(0,1),(1,0),(1,1),(1,-1)]


# ──────────────────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────────────────
def _in_bounds(size: int, r: int, c: int) -> bool:
    return 0 <= r < size and 0 <= c < size

def _other(p: Player) -> Player:
    return Player.WHITE if p is Player.BLACK else Player.BLACK

def zsin(x: float, period: float) -> float:
    t = np.clip(np.asarray(x, dtype=float) / period, 0.0, 1.0)
    return np.sin(t * np.pi * 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# 引擎主体
# ──────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, win_k: int = 4):
        self.win_k = win_k
        self.over_fill = False
        self.just_unascend = False
        self.grow_count = 11
        self.unascend_charge = 25

    def step(self, s: GameState, move: Move) -> GameState:
        """
        推进一步并处理阶段切换/结算：
          - NORMAL：当前方落子；若触发四连 → 进入 ATTACK_DEFENSE（设置 attack_chain_mask 并移除攻方链棋子）；
                    否则轮转执手并按周期刷新植物。
          - ATTACK_DEFENSE：防守方落子 → 依据 a,b,c,d 规则结算 → 清除参与植物 → 回到 NORMAL 并强制刷新植物。
        """
        ns = s.copy()
        # record board state
        ns.last_moves.append(ns.board.grid.copy())

        if ns.phase is Phase.NORMAL:
            # 当前执手下子
            self._place_stone_or_raise(ns, move, ns.to_play)
            ns.last_move = move
            # ns.last_moves.append(move)

            # 计算以该落点为锚的所有四连（去重合集）
            atk_cells = self._collect_four_chains(ns.board, move, player_id=self._pid(ns.to_play))

            if atk_cells:
                # 进入ascend：先把攻方链上的棋子从棋盘上移除（按规则）
                self._remove_stones(ns.board, atk_cells)

                # 用 mask 表达攻方链（供 UI/编码/结算使用）
                mask = np.zeros_like(ns.board.grid, dtype=np.uint8)
                for (r, c) in atk_cells:
                    mask[r, c] = 1
                ns.attack_chain_mask = mask

                # 阶段切换到 ascend，轮到防守方应手
                ns.phase = Phase.ATTACK_DEFENSE
                ns.to_play = _other(ns.to_play)
            else:
                # 未触发攻防：正常轮转并按周期刷新
                ns.to_play = _other(ns.to_play)

            if self.unascend_charge > 0:
                self.unascend_charge -= 1
            self.just_unascend = False
            

        elif ns.phase is Phase.ATTACK_DEFENSE:
            # 记录本手的防守方（当前执手）
            defender = ns.to_play

            # 防守方下子
            self._place_stone_or_raise(ns, move, defender)
            ns.last_move = move
            # ns.last_moves.append(move)

            # 以防守方落点为锚，收集防守方四连（可能为空）
            def_cells = self._collect_four_chains(ns.board, move, player_id=self._pid(defender))

            # 如果防守方存在四连，则清除防守方棋子
            if def_cells:
                self._remove_stones(ns.board, def_cells)

            # 由 attack_chain_mask 还原攻方链坐标集合（避免维护额外上下文）
            atk_cells = self._cells_from_mask(ns.attack_chain_mask)

            # 进行 a,b,c,d 伤害结算并更新 HP
            self._resolve_attack_defense(ns, defender, move, atk_cells, def_cells)

            # 清除参与攻防的植物（攻/防链并集）
            if def_cells:
                cells = set().union(atk_cells, def_cells)
                for (r, c) in cells:
                    ns.board.plants[r, c] = 0
            else:
                # 防守方未触发ascend, 则仅消除攻击方的植物，但防守方此次move占据的攻击方植物不消除
                for (r, c) in atk_cells:
                    if (r, c) == move.to_tuple():
                        continue
                    ns.board.plants[r, c] = 0

            # 回到 NORMAL；清 mask；轮到进攻方对手（即当前 defender 的对手）
            ns.phase = Phase.NORMAL
            self.just_unascend = True
            self.unascend_charge = int(max(12.5, min(self.unascend_charge + (25 - self.unascend_charge) * 0.4, 25)))

            ns.attack_chain_mask = None
            ns.to_play = _other(defender)
            

        # 计入一手
        ns.turn += 1

        # 刷新植物
        self.grow_count -= 1
        if not ns.phase == Phase.ATTACK_DEFENSE:  # 攻防开始时不刷草
            if self.grow_count <= 0 or (s.phase == Phase.ATTACK_DEFENSE and not self.just_unascend):
                self.grow_count = max(7, int(11 - s.turn / 22 * 2))
                stone_count = np.sum(ns.board.grid > 0)
                if stone_count >= 44:
                    self.over_fill = True
                if self.over_fill and stone_count < 22:
                    self.over_fill = False
                flower_num = 3 if s.turn >= 65 else 2
                self._refresh_plants(ns, flower_num, just_ascend=(s.phase == Phase.ATTACK_DEFENSE))
                if self.over_fill:
                    self.grow_count /= 2
                if self.grow_count % 2 == 0:
                    self.grow_count -= 1
            
        else:
            raise RuntimeError(f"未知阶段: {ns.phase}")

        return ns

    # ──────────────────────────────────────────────────────────────────────────
    # NORMAL 阶段辅助
    # ──────────────────────────────────────────────────────────────────────────
    def _place_stone_or_raise(self, s: GameState, move: Move, who: Player) -> None:
        r, c = move.r, move.c
        assert s.board.is_empty(r, c), "cell not empty"
        s.board.place_stone(r, c, self._pid(who))

    # ──────────────────────────────────────────────────────────────────────────
    # 四连收集 / 棋子移除
    # ──────────────────────────────────────────────────────────────────────────
    def _collect_four_chains(self, board: Board, last_move: Move, player_id: int) -> Set[Tuple[int,int]]:
        """
        以 last_move 为锚，按四个方向收集连续同色段，长度>=win_k 的都纳入合集（支持“同时多条四连”）。
        """
        size = board.size
        r0, c0 = last_move.r, last_move.c
        all_cells: Set[Tuple[int,int]] = set()

        for dr, dc in _DIRS:
            cells = [(r0, c0)]
            # 正向延伸
            r, c = r0 + dr, c0 + dc
            while _in_bounds(size, r, c) and board.grid[r, c] == player_id:
                cells.append((r, c))
                r += dr; c += dc
            # 反向延伸
            r, c = r0 - dr, c0 - dc
            while _in_bounds(size, r, c) and board.grid[r, c] == player_id:
                cells.append((r, c))
                r -= dr; c -= dc

            if len(cells) >= self.win_k:
                for cell in cells:
                    all_cells.add(cell)

        return all_cells

    def _remove_stones(self, board: Board, cells: Iterable[Tuple[int,int]]) -> None:
        """把给定集合中的棋子从棋盘上消去（不影响植物计数）。"""
        for (r, c) in cells:
            board.grid[r, c] = 0

    # ──────────────────────────────────────────────────────────────────────────
    # 攻防结算（a,b,c,d）
    # ──────────────────────────────────────────────────────────────────────────
    def _cells_from_mask(self, mask: np.ndarray | None) -> Set[Tuple[int,int]]:
        if mask is None:
            return set()
        rs, cs = np.where(mask > 0)
        return set(zip(rs.tolist(), cs.tolist()))

    def _resolve_attack_defense(
        self,
        s: GameState,
        defender: Player,
        defense_move: Move,
        atk_cells: Set[Tuple[int,int]],
        def_cells: Set[Tuple[int,int]],
    ) -> None:
        """
        使用 power 消除规则结算伤害：
          - 每个参与格子的 power = plants[r, c] + 1
          - 若 atk/def 有重合格子，视为防守方占据攻方格子：先从 atk_cells 中移除重合格子
          - 循环比较双方最大 power，较大方用该 power 按从大到小消除对方 <=P 的项
          - 一方 power 列表为空时停止，另一方剩余项个数即对对手伤害
        """
        board = s.board
        attacker = defender.other()

        overlap = atk_cells.intersection(def_cells)
        if overlap:
            atk_cells = set(atk_cells)
            atk_cells.difference_update(overlap)

        atk_power = [int(board.plants[r, c]) + 1 for (r, c) in atk_cells]
        def_power = [int(board.plants[r, c]) + 1 for (r, c) in def_cells]

        while atk_power and def_power:
            atk_max = max(atk_power)
            def_max = max(def_power)
            atk_act = atk_max >= def_max

            if atk_act:
                idx = atk_power.index(atk_max)
                p = atk_power[idx]
                for p1 in sorted(def_power, reverse=True):
                    if p1 > p:
                        continue
                    p -= p1
                    def_power.remove(p1)
                    if p == 0:
                        break
                if p == 0:
                    del atk_power[idx]
                else:
                    atk_power[idx] = p
            else:
                idx = def_power.index(def_max)
                p = def_power[idx]
                for p1 in sorted(atk_power, reverse=True):
                    if p1 > p:
                        continue
                    p -= p1
                    atk_power.remove(p1)
                    if p == 0:
                        break
                if p == 0:
                    del def_power[idx]
                else:
                    def_power[idx] = p

        if atk_power:
            damage = len(atk_power)
            s.hp[defender.value] = max(0, s.hp[defender.value] - damage)
        elif def_power:
            damage = len(def_power)
            s.hp[attacker.value] = max(0, s.hp[attacker.value] - damage)

        # 旧版简化伤害逻辑（保留，不删除）
        # size = board.size
        # occupied_attk = (defense_move.r, defense_move.c) in atk_cells
        # occupy_penalty = 1 if occupied_attk else 0
        # a_raw = len(atk_cells)
        # a = max(0, a_raw - occupy_penalty)
        # b = 0
        # for (r, c) in atk_cells:
        #     if (r, c) == (defense_move.r, defense_move.c):
        #         continue
        #     b += int(board.plants[r, c])
        # c = len(def_cells)
        # d = sum(int(board.plants[r, c]) for (r, c) in def_cells)
        # if a >= c and b >= d:
        #     if b <= (c + d) / 2:
        #         damage = (a + b) - (c + d)
        #     else:
        #         damage = ceil(a - (c + d) / 2)
        #     s.hp[defender.value] = max(0, s.hp[defender.value] - damage)
        # elif a <= c and b <= d:
        #     if d <= (a + b) / 2:
        #         damage = (c + d) - (a + b)
        #     else:
        #         damage = ceil(c - (a + b) / 2)
        #     s.hp[attacker.value] = max(0, s.hp[attacker.value] - damage)
        # else:
        #     if (a + b) > (c + d):
        #         damage = (a + b) - (c + d)
        #         s.hp[defender.value] = max(0, s.hp[defender.value] - damage)
        #     elif (c + d) > (a + b):
        #         damage = (c + d) - (a + b)
        #         s.hp[attacker.value] = max(0, s.hp[attacker.value] - damage)

    # ──────────────────────────────────────────────────────────────────────────
    # 植物清除 / 刷新
    # ──────────────────────────────────────────────────────────────────────────
    def _calc_align_stats_for_candidate(
        self, board: Board, r0: int, c0: int, stone_id: int
    ) -> Tuple[int, int]:
        """
        计算某个空位若放置 stone_id 后的连子统计：
          - max_align: 四个轴线中最大连子数
          - max_align_total: 四个轴线连子数之和
        连子数按“该点+双向连续同色棋子”计算。
        """
        size = board.size
        axis_counts = []
        for dr, dc in _DIRS:
            count = 1
            r, c = r0 + dr, c0 + dc
            while _in_bounds(size, r, c) and board.grid[r, c] == stone_id:
                count += 1
                r += dr
                c += dc
            r, c = r0 - dr, c0 - dc
            while _in_bounds(size, r, c) and board.grid[r, c] == stone_id:
                count += 1
                r -= dr
                c -= dc
            axis_counts.append(count)
        return max(axis_counts), sum(axis_counts)

    def _refresh_plants(self, s: GameState, flower_num: int, just_ascend: bool) -> None:
        """
        刷新植物：
          - 候选点：所有空位；
          - 对每个候选点，计算黑方(0)/白方(1)的 max_align 与 max_align_total；
          - 每个候选点初始权重 500，并叠加均匀随机整数 [0, 20]；
          - 按权重从高到低取前 flower_num 个点，各 +1（封顶 2）。
        """
        board = s.board
        size = board.size

        candidates = [(r, c)
                      for r in range(size) for c in range(size)
                      if board.grid[r, c] == 0 and board.plants[r, c] < 2]
        if not candidates:
            return

        k = min(flower_num, len(candidates))
        weighted_candidates = []
        attacker_idx = s.to_play.value
        defender_idx = s.to_play.other().value
        for (r, c) in candidates:
            black_max_align, black_max_align_total = self._calc_align_stats_for_candidate(
                board, r, c, stone_id=1
            )
            white_max_align, white_max_align_total = self._calc_align_stats_for_candidate(
                board, r, c, stone_id=2
            )
            # 刷草位置逻辑
            weight = 500 + random.randint(0, 20)
            if max(black_max_align, white_max_align) >= 4:
                if self.over_fill:
                    weight += 450
                    if board.plants[r, c] > 0:
                        weight += 100
                else:
                    weight -= 450

            if just_ascend:
                max_align = [black_max_align, white_max_align]  # [黑方0, 白方1]
                max_align_total = [black_max_align_total, white_max_align_total]  # [黑方0, 白方1]
                if max_align_total[attacker_idx] > max_align_total[defender_idx]:
                    weight += (max_align_total[attacker_idx] - max_align_total[defender_idx]) * 3
                if max_align[attacker_idx] > max_align[defender_idx]:
                    weight += (max_align[attacker_idx] - max_align[defender_idx]) * 15
                if self.unascend_charge <= 0:
                    weight += int(zsin(25 - self.unascend_charge, 25.0) * 120.0)
            else:
                pass  # TODO: 这里有一段weight更新逻辑没有实现，对应源代码TTRPlant.cs的第229-236行
            if board.plants[r, c] > 0 and not (just_ascend and self.unascend_charge <= 0):
                weight -= 30
            elif True:
                pass  # TODO: 这里有一段更新逻辑没有实现，对应源代码TTRPlant.cs第239-240行
            

            weighted_candidates.append(
                {
                    "weight": weight,
                    "r": r,
                    "c": c,
                    "max_align": [black_max_align, white_max_align],  # [黑方0, 白方1]
                    "max_align_total": [black_max_align_total, white_max_align_total],  # [黑方0, 白方1]
                }
            )

        weighted_candidates.sort(key=lambda x: x["weight"], reverse=True)
        for item in weighted_candidates[:k]:
            board.plants[item["r"], item["c"]] = min(2, board.plants[item["r"], item["c"]] + 1)

    # ──────────────────────────────────────────────────────────────────────────
    # 杂项
    # ──────────────────────────────────────────────────────────────────────────
    def _pid(self, p: Player) -> int:
        return 1 if p is Player.BLACK else 2
