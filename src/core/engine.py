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
from math import ceil

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


# ──────────────────────────────────────────────────────────────────────────────
# 引擎主体
# ──────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, win_k: int = 4):
        self.win_k = win_k

    def step(self, s: GameState, move: Move) -> GameState:
        """
        推进一步并处理阶段切换/结算：
          - NORMAL：当前方落子；若触发四连 → 进入 ATTACK_DEFENSE（设置 attack_chain_mask 并移除攻方链棋子）；
                    否则轮转执手并按周期刷新植物。
          - ATTACK_DEFENSE：防守方落子 → 依据 a,b,c,d 规则结算 → 清除参与植物 → 回到 NORMAL 并强制刷新植物。
        """
        ns = s.copy()

        if ns.phase is Phase.NORMAL:
            # 当前执手下子
            self._place_stone_or_raise(ns, move, ns.to_play)
            ns.last_move = move
            ns.last_moves.append(move)

            # 计算以该落点为锚的所有四连（去重合集）
            atk_cells = self._collect_four_chains(ns.board, move, player_id=self._pid(ns.to_play))

            if atk_cells:
                # 进入攻防：先把攻方链上的棋子从棋盘上移除（按规则）
                self._remove_stones(ns.board, atk_cells)

                # 用 mask 表达攻方链（供 UI/编码/结算使用）
                mask = np.zeros_like(ns.board.grid, dtype=np.uint8)
                for (r, c) in atk_cells:
                    mask[r, c] = 1
                ns.attack_chain_mask = mask

                # 阶段切换到 ATTACK_DEFENSE，轮到防守方应手
                ns.phase = Phase.ATTACK_DEFENSE
                ns.to_play = _other(ns.to_play)
            else:
                # 未触发攻防：正常轮转并按周期刷新
                ns.to_play = _other(ns.to_play)
                self._post_normal_move_bookkeep_and_refresh(ns, force=False)

        elif ns.phase is Phase.ATTACK_DEFENSE:
            # 记录本手的防守方（当前执手）
            defender = ns.to_play

            # 防守方下子
            self._place_stone_or_raise(ns, move, defender)
            ns.last_move = move
            ns.last_moves.append(move)

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
            ns.attack_chain_mask = None
            ns.to_play = _other(defender)

            # 攻防后立刻刷新植物（强制），并计入一手
            self._refresh_plants(ns, force=True)
            ns.turn += 1

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

    def _post_normal_move_bookkeep_and_refresh(self, s: GameState, force: bool) -> None:
        s.turn += 1
        self._refresh_plants(s, force=force)

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
        依据规则计算：
          a = 攻方 chain 的棋子数 - 被防守方占据的棋子数（只需检查防守方当前 move 是否占据攻方 chain）
          b = 攻方 chain 的植物数   - 被防守方占据的棋子数（同上，仅按是否占据扣 1，而非扣相应格的植物计数）
          c = 防守方 chain 的棋子数
          d = 防守方 chain 的植物数
          伤害: 暂不考虑双草
        """
        board = s.board
        size = board.size
        attacker = defender.other()

        # 是否“本手占据了攻方链中的某格”
        occupied_attk = (defense_move.r, defense_move.c) in atk_cells
        occupy_penalty = 1 if occupied_attk else 0

        # a: 攻方有效 stone 数
        a_raw = len(atk_cells)
        a = max(0, a_raw - occupy_penalty)

        # b: 攻方有效 plant 数（按“位置是否被占据”扣 1，而不是扣该格的植物计数）
        b = 0
        for (r, c) in atk_cells:
            if (r, c) == (defense_move.r, defense_move.c): 
                continue
            b += int(board.plants[r, c])

        # c: 防守方有效 stone 数
        c = len(def_cells)

        # d: 防守方有效 plant 数
        d = sum(int(board.plants[r, c]) for (r, c) in def_cells)

        if a >= c and b >= d:
            if b <= (c + d) / 2:
                damage = (a + b) - (c + d)
            else:
                damage = ceil(a - (c + d) / 2)
            s.hp[defender.value] = max(0, s.hp[defender.value] - damage)
        elif a <= c and b <= d:
            if d <= (a + b) / 2:
                damage = (c + d) - (a + b)
            else:
                damage = ceil(c - (a + b) / 2)
            s.hp[attacker.value] = max(0, s.hp[attacker.value] - damage)
        else:
            if (a + b) > (c + d):
                damage = (a + b) - (c + d)
                s.hp[defender.value] = max(0, s.hp[defender.value] - damage)
            elif (c + d) > (a + b):
                damage = (c + d) - (a + b)
                s.hp[attacker.value] = max(0, s.hp[attacker.value] - damage)

        # 相等则不掉血

    # ──────────────────────────────────────────────────────────────────────────
    # 植物清除 / 刷新
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_plants(self, s: GameState, force: bool) -> None:
        """
        刷新两株植物：
          - force=False：仅当 (turn % 10 == 0) 时刷新；
          - force=True ：无视步数立即刷新；
          - 刷新位置：空位且 plants<2；随机等概率挑选不同位置至多 2 个，各 +1（封顶 2）。
        """
        flower_num = 0
        if not force:
            if s.turn in [11,22,31,40]:
                flower_num = 2
            elif s.turn >= 49 and (s.turn - 49) % 3 == 0:
                flower_num = 3
            else:
                return
        # if s.turn < 50 or s.turn % 20 != 0:
        #     return
        # else:
        #     flower_num = 2

        board = s.board
        size = board.size

        candidates = [(r, c)
                      for r in range(size) for c in range(size)
                      if board.grid[r, c] == 0]
        if not candidates:
            return

        k = min(flower_num, len(candidates))
        for (r, c) in random.sample(candidates, k=k):
            board.plants[r, c] = min(2, board.plants[r, c] + 1)

    # ──────────────────────────────────────────────────────────────────────────
    # 杂项
    # ──────────────────────────────────────────────────────────────────────────
    def _pid(self, p: Player) -> int:
        return 1 if p is Player.BLACK else 2
