# ──────────────────────────────────────────────────────────────────────────────
# File: src/core/engine.py 
# The game engine: the step function that takes a state and a move, and returns the next state after applying the game rules.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Set, Tuple, Iterable
import numpy as np
import random

from .types import Player, Move, Phase
from .state import GameState
from .board import Board

_DIRS = [(0,1),(1,0),(1,1),(1,-1)]


# ──────────────────────────────────────────────────────────────────────────────
# basic tools
# ──────────────────────────────────────────────────────────────────────────────
def _in_bounds(size: int, r: int, c: int) -> bool:
    return 0 <= r < size and 0 <= c < size

def _other(p: Player) -> Player:
    return Player.WHITE if p is Player.BLACK else Player.BLACK

def zsin(x: float, period: float) -> float:
    t = np.clip(np.asarray(x, dtype=float) / period, 0.0, 1.0)
    return np.sin(t * np.pi * 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# game engine
# ──────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, win_k: int = 4):
        self.win_k = win_k  # do not change since 4ASCEND is designed around 4-in-a-row mechanics

    def step(self, s: GameState, move: Move) -> GameState:
        """
        step the status s by applying move, return the next status.
        """
        ns = s.copy()
        # record board state
        ns.last_moves.append(ns.board.grid.copy())

        if ns.phase is Phase.NORMAL:
            """
            When the current phase is NORMAL, the current player simply places a stone. 
            If this creates a >=win_k chain, those stones are removed and the game transitions to ATTACK_DEFENSE with the opponent as the defender; 
            otherwise, it just switches to the opponent and potentially refreshes plants below.
            """

            # the current player places a stone
            self._place_stone_or_raise(ns, move, ns.to_play)
            ns.last_move = move
            # ns.last_moves.append(move)

            # calculate the attack chain (if any) triggered by this move: collect all >=win_k chains of the current player that include the move
            atk_cells = self._collect_four_chains(ns.board, move, player_id=self._pid(ns.to_play))

            if atk_cells:
                # enter ATTACK_DEFENSE phase: remove the attack chain stones from the board (but keep the plant counts, which will be used for damage calculation)
                self._remove_stones(ns.board, atk_cells)

                # establish the attack chain mask for the defense phase (to be used for damage calculation and observation encoding)
                mask = np.zeros_like(ns.board.grid, dtype=np.uint8)
                for (r, c) in atk_cells:
                    mask[r, c] = 1
                ns.attack_chain_mask = mask

                # transition to ATTACK_DEFENSE phase; the opponent becomes the defender
                ns.phase = Phase.ATTACK_DEFENSE
                ns.to_play = _other(ns.to_play)
            else:
                # If there is no ASCEND trigger, simply switch the player and potentially refresh plants below.
                ns.to_play = _other(ns.to_play)
                ns.just_unascend = False

            if ns.unascend_charge > 0:
                ns.unascend_charge -= 1

            for index in range(2):
                if ns.Aunascend_charge_fast[index] > 0:
                    ns.Aunascend_charge_fast[index] -= 1

        elif ns.phase is Phase.ATTACK_DEFENSE:
            """
            When the current phase is ATTACK_DEFENSE, the current player is the defender who just placed a stone to defend against the opponent's attack chain.
            """
            defender = ns.to_play

            # the defender places a stone
            self._place_stone_or_raise(ns, move, defender)
            ns.last_move = move
            # ns.last_moves.append(move)

            # collect the 4-in-a-row chains of the defender that include the defense move (if any), 
            # which will be used for damage calculation. 
            # Note that these chains may overlap with the attack chain, which will be handled in the damage calculation logic.
            def_cells = self._collect_four_chains(ns.board, move, player_id=self._pid(defender))

            # If the defender also creates >=win_k chains, those stones are removed before damage calculation
            if def_cells:
                self._remove_stones(ns.board, def_cells)

            atk_cells = self._cells_from_mask(ns.attack_chain_mask)
            # If the defender fails to create any >=win_k chain, 
            # they may also occupy the attacker's chain cells with their defense move, which deletes the occupied attack chain cell from the attacker's effective attack cells.;
            if not def_cells:
                atk_cells.discard(move.to_tuple())

            # computing the damage
            self._resolve_attack_defense(ns, defender, move, atk_cells, def_cells)

            # eliminate all plants that participate in the attack/defense (i.e. those in atk_cells or def_cells)
            if def_cells:
                cells = set().union(atk_cells, def_cells)
                for (r, c) in cells:
                    ns.board.plants[r, c] = 0
            else:
                # Note that if the defender fails to create any >=win_k chain, they can only occupy at most one attack chain cell (since one move), so DO NOT clear the plants in that cell.
                for (r, c) in atk_cells:
                    if (r, c) == move.to_tuple():
                        continue
                    ns.board.plants[r, c] = 0

            # the status return to NORMAL phase, and the defender becomes the next attacker
            ns.phase = Phase.NORMAL
            ns.attack_chain_mask = None
            ns.to_play = _other(defender)
            
        else:
            raise RuntimeError(f"未知阶段: {ns.phase}")
        
        ns.turn += 1

        # plant refreshing
        ns.grow_count -= 1
        if not ns.phase == Phase.ATTACK_DEFENSE:  # If we just enter the attack phase, do not create any plant
            if ns.grow_count <= 0 or (s.phase == Phase.ATTACK_DEFENSE and not ns.just_unascend):  # create plant only when grow_count runs out, or (leaving the ascend phase and we are not in a continuous ascending)
                ns.grow_count = max(7, int(11 - int(ns.turn / 22) * 2))
                stone_count = np.sum(ns.board.grid > 0)
                if stone_count >= 44:
                    ns.over_fill = True
                if ns.over_fill and stone_count < 22:
                    ns.over_fill = False
                flower_num = 3 if s.turn >= 65 else 2
                self._refresh_plants(ns, flower_num, just_ascend=(s.phase == Phase.ATTACK_DEFENSE))
                if ns.over_fill:
                    ns.grow_count = int(ns.grow_count / 2)
                if ns.grow_count % 2 == 0:
                    ns.grow_count -= 1
                
                if s.phase == Phase.ATTACK_DEFENSE:
                    ns.just_unascend = True
                    ns.unascend_charge = int(max(12.5, min(ns.unascend_charge + (25 - ns.unascend_charge) * 0.4, 25)))
                    pr = 0 if ns.to_play is Player.BLACK else 1
                    ns.Aunascend_charge_fast[pr] = max(
                        ns.Aunascend_charge_fast[pr],
                        4 if ns.over_fill else 9,
                    )


        return ns

    def _place_stone_or_raise(self, s: GameState, move: Move, who: Player) -> None:
        r, c = move.r, move.c
        assert s.board.is_empty(r, c), "cell not empty"
        s.board.place_stone(r, c, self._pid(who))

    def _collect_four_chains(self, board: Board, last_move: Move, player_id: int) -> Set[Tuple[int,int]]:
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
        Use the array-algorithm to resolve the damage of the attack and defense, and update the HP in the status accordingly.
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

        # old version of damage calculation
        
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
        return max(axis_counts), sum(axis_counts) - 3

    def _refresh_plants(self, s: GameState, flower_num: int, just_ascend: bool) -> None:
        """
        refreshing the plants
        """
        board = s.board
        size = board.size
        if s.over_fill:
            flower_num += 1

        candidates = [(r, c)
                      for r in range(size) for c in range(size)
                      if board.grid[r, c] == 0 and board.plants[r, c] < 2]
        if not candidates:
            return

        k = min(flower_num, len(candidates))
        weighted_candidates = []
        attacker_idx = s.to_play.value
        defender_idx = s.to_play.other().value
        index1 = 0 if s.to_play is Player.BLACK else 1
        flag1 = s.Aunascend_charge_fast[index1] > 0
        flag2 = s.Aunascend_charge_fast[1 - index1] > 0
        flag4 = just_ascend and s.unascend_charge <= 0
        ovr_buff = []
        num1 = [0, 0]
        for (r, c) in candidates:
            black_max_align, black_max_align_total = self._calc_align_stats_for_candidate(
                board, r, c, stone_id=1
            )
            white_max_align, white_max_align_total = self._calc_align_stats_for_candidate(
                board, r, c, stone_id=2
            )
            # determine the place to refresh the plants
            max_align = [black_max_align, white_max_align]  # black: 0, white: 1
            _ovr_bits = [0, 0]
            weight = 500 + random.randint(0, 20)
            flag5 = False
            if max(black_max_align, white_max_align) >= 4:
                if s.over_fill:
                    weight += 450
                    if board.plants[r, c] > 0:
                        weight += 100

                    if max_align[index1] < 4 and not flag2:
                        _ovr_bits[1] = 1
                    if max_align[1 - index1] < 4 and not flag1:
                        _ovr_bits[0] = 1
                else:
                    weight -= 450

            if just_ascend:
                max_align_total = [black_max_align_total, white_max_align_total]  # black: 0, white: 1
                if max_align_total[attacker_idx] > max_align_total[defender_idx]:
                    weight += (max_align_total[attacker_idx] - max_align_total[defender_idx]) * 3
                if max_align[attacker_idx] > max_align[defender_idx]:
                    weight += (max_align[attacker_idx] - max_align[defender_idx]) * 15
                if max_align[attacker_idx] == 3 and max_align[defender_idx] <= 1:
                    if flag4:
                        flag5 = True
                    else:
                        weight += int(zsin(25 - s.unascend_charge, 25.0) * 120.0)
            else:   
                pass  # TODO: there is a branch in the original logic that has no implementation, corresponding to TTRPlant.cs line 227-228
            if board.plants[r, c] > 0 and not flag4:
                weight -= 30
            elif True:
                pass  # TODO: athere is a branch in the original logic that has no implementation, corresponding to TTRPlant.cs line 230-231

            if _ovr_bits[0] or _ovr_bits[1]:
                ovr_buff.append(
                    {
                        "weight": weight,
                        "r": r,
                        "c": c,
                        "ovr_bits": _ovr_bits,
                        "max_align_total": [black_max_align_total, white_max_align_total],  # black: 0, white: 1
                        "max_align": [black_max_align, white_max_align],  # black: 0, white: 1
                    }
                )
                num1[0] = max(num1[0], _ovr_bits[0])
                num1[1] = max(num1[1], _ovr_bits[1])
            else:
                weighted_candidates.append(
                    {
                        "weight": weight,
                        "r": r,
                        "c": c,
                        "max_align": [black_max_align, white_max_align],  # black: 0, white: 1
                        "max_align_total": [black_max_align_total, white_max_align_total],  # black: 0, white: 1
                        "ovr_bits": [0, 0]
                    }
                )
                if flag5:
                    k += 1
                    weighted_candidates.append(
                        {
                            "weight": weight + 30 - random.randint(0, 89),
                            "r": r,
                            "c": c,
                            "max_align": [black_max_align, white_max_align],  # black: 0, white: 1
                            "max_align_total": [black_max_align_total, white_max_align_total],  # black: 0, white: 1
                            "ovr_bits": [0, 0]
                        }
                    )

        if ovr_buff:
            random.shuffle(ovr_buff)
            for ovr in ovr_buff:
                overBits = ovr["ovr_bits"]
                matched = False
                for i in range(2):
                    if num1[i] == 1 and overBits[i] == 1:
                        num1[i] = 0
                        matched = True
                if matched:
                    k += 1
                    weighted_candidates.append(
                        {
                            "weight": ovr["weight"],
                            "r": ovr["r"],
                            "c": ovr["c"],
                            "max_align": ovr["max_align"],  # black: 0, white: 1
                            "max_align_total": ovr["max_align_total"],  # black: 0, white: 1
                            "ovr_bits": ovr["ovr_bits"]
                        }
                    )
                    weighted_candidates.append(
                        {
                            "weight": ovr["weight"] + 40 - random.randint(0, 119),
                            "r": ovr["r"],
                            "c": ovr["c"],
                            "max_align": ovr["max_align"],  # black: 0, white: 1
                            "max_align_total": ovr["max_align_total"],  # black: 0, white: 1
                            "ovr_bits": ovr["ovr_bits"]
                        }
                    )
                

        k = min(k, 6)

        weighted_candidates.sort(key=lambda x: x["weight"], reverse=True)
        refreshed = 0
        for item in weighted_candidates:
            if refreshed >= k:
                break
            r, c = item["r"], item["c"]
            if board.plants[r, c] >= 2:
                continue
            board.plants[r, c] += 1
            refreshed += 1
            over_bits = item.get("ovr_bits", [0, 0])
            if over_bits[index1] == 1:
                s.Aunascend_charge_fast[index1] = max(s.Aunascend_charge_fast[index1], 4)
            other_idx = 1 - index1
            if over_bits[other_idx] == 1:
                s.Aunascend_charge_fast[other_idx] = max(s.Aunascend_charge_fast[other_idx], 4)


    def _pid(self, p: Player) -> int:
        return 1 if p is Player.BLACK else 2
