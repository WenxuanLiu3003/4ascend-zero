from __future__ import annotations
import sys
import pygame
import numpy as np
from typing import List

# 颜色与尺寸
BG = (245, 245, 245)
GRID = (180, 180, 180)
BLACK = (30, 30, 30)
WHITE = (235, 235, 235)
STONE_OUTLINE = (50, 50, 50)
GREEN = (60, 160, 75)
BLUE = (70, 120, 240)   # 植物刷新(+)
RED = (200, 60, 60)     # 植物消失(-)
ORANGE = (245, 150, 60)

CELL = 48
MARGIN = 32
INFO_H = 150
STONE_R = 18
PLANT_R = 6
GRID_EXT = 10


def _draw_text(screen, txt, pos, size=24, color=BLACK):
    font = pygame.font.SysFont(None, size)
    screen.blit(font.render(txt, True, color), pos)


def _compute_layout(win_w: int, win_h: int, size: int):
    margin = max(12, int(min(win_w, win_h) * 0.04))
    max_cell_w = (win_w - 2 * margin) / max(1, size)
    cell = max(18, int(max_cell_w))
    for _ in range(2):
        info_h = max(110, int(cell * 3.2))
        info_h = min(info_h, int(win_h * 0.46))
        max_cell_h = (win_h - info_h - 2 * margin) / max(1, size)
        cell = max(18, int(min(max_cell_w, max_cell_h)))

    info_h = max(110, int(cell * 3.2))
    info_h = min(info_h, int(win_h * 0.46))
    board_area = size * cell
    board_outer_x = (win_w - board_area) // 2
    board_outer_y = info_h + max(0, (win_h - info_h - board_area) // 2)
    board_pad = cell // 2
    return {
        "cell": cell,
        "margin": margin,
        "info_h": info_h,
        "board_pad": board_pad,
        "board_x0": board_outer_x + board_pad,
        "board_y0": board_outer_y + board_pad,
        "board_span": (size - 1) * cell,
        "grid_ext": max(4, cell // 5),
        "stone_r": max(8, int(cell * 0.38)),
        "plant_r": max(3, int(cell * 0.14)),
        "font_big": max(18, int(cell * 0.56)),
        "font_mid": max(14, int(cell * 0.42)),
        "font_small": max(12, int(cell * 0.34)),
        "line_gap": max(4, int(cell * 0.10)),
    }


def launch_replay(trace: List, board_size: int = 9):
    """基于 trace（见 selfplay_trace.TraceStep 列表）启动回放 GUI。
    键位：Left/Right 上一步/下一步；Home/End 跳到首/尾；R 重开本回放；Esc 退出。
    """
    pygame.init()
    board_span = (board_size - 1) * CELL
    board_pad = CELL // 2
    W = 2 * (MARGIN * 2 + board_span + board_pad * 2)
    H = 2 * (INFO_H + MARGIN * 2 + board_span + board_pad * 2)
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("4ascend - Self-play Replay Viewer")
    clock = pygame.time.Clock()

    idx = 0  # 当前帧索引

    def draw_frame():
        screen.fill(BG)
        step = trace[idx]
        size = board_size
        win_w, win_h = screen.get_size()
        ly = _compute_layout(win_w, win_h, size)
        over_fill = getattr(step, "over_fill", False)
        just_unascend = getattr(step, "just_unascend", False)
        grow_count = getattr(step, "grow_count", 0)
        unascend_charge = getattr(step, "unascend_charge", 0)
        # 顶栏信息：步号/HP/阶段
        x0 = ly["margin"]
        y0 = ly["margin"]
        _draw_text(screen, f"Frame {idx+1}/{len(trace)} | Turn {step.turn} | Phase: {step.phase}", (x0, y0), ly["font_big"])
        y0 += ly["font_big"] + ly["line_gap"]
        _draw_text(screen, f"BLACK HP: {step.hp[0]}   WHITE HP: {step.hp[1]}", (x0, y0), ly["font_mid"])
        y0 += ly["font_mid"] + ly["line_gap"]
        _draw_text(
            screen,
            f"over_fill={over_fill}  just_unascend={just_unascend}  "
            f"grow_count={grow_count}  unascend_charge={unascend_charge}",
            (x0, y0),
            ly["font_small"],
        )
        y0 += ly["font_small"] + ly["line_gap"]
        _draw_text(screen, "←/→: 上一手/下一手   Home/End: 首/尾   R: 重开   Esc: 退出", (x0, y0), ly["font_small"])

        # 棋盘与网格
        board_x0 = ly["board_x0"]
        board_y0 = ly["board_y0"]
        span = ly["board_span"]
        board_pad = ly["board_pad"]
        cell = ly["cell"]
        grid_ext = ly["grid_ext"]
        stone_r = ly["stone_r"]
        plant_r = ly["plant_r"]
        pygame.draw.rect(
            screen,
            (230, 230, 230),
            (board_x0 - board_pad, board_y0 - board_pad, span + board_pad * 2, span + board_pad * 2),
        )
        for i in range(size):
            x = board_x0 + i * cell
            y = board_y0 + i * cell
            pygame.draw.line(screen, GRID, (board_x0 - grid_ext, y), (board_x0 + span + grid_ext, y), 1)
            pygame.draw.line(screen, GRID, (x, board_y0 - grid_ext), (x, board_y0 + span + grid_ext), 1)

        # 被无效化掩码高亮（若存在）
        if step.attack_chain_mask is not None:
            mask = step.attack_chain_mask
            rs, cs = np.where(mask)
            for r, c in zip(rs, cs):
                cx = board_x0 + c * cell
                cy = board_y0 + r * cell
                pygame.draw.circle(screen, ORANGE, (cx, cy), stone_r + 4, 2)

        # 棋子
        for r in range(size):
            for c in range(size):
                v = step.grid[r, c]
                if v == 0:
                    continue
                cx = board_x0 + c * cell
                cy = board_y0 + r * cell
                color = BLACK if v == 1 else WHITE
                pygame.draw.circle(screen, color, (cx, cy), stone_r)
                pygame.draw.circle(screen, STONE_OUTLINE, (cx, cy), stone_r, 2)

        # 植物：以 0/1/2 株显示（两个小点表示 2 株）
        for r in range(size):
            for c in range(size):
                k = int(step.plants[r, c])
                if k <= 0:
                    continue
                cx = board_x0 + c * cell
                cy = board_y0 + r * cell
                if k == 1:
                    pygame.draw.circle(screen, GREEN, (cx + max(1, plant_r // 2), cy - max(1, plant_r // 2)), plant_r)
                else:
                    pygame.draw.circle(screen, GREEN, (cx - plant_r + 1, cy - 1), plant_r)
                    pygame.draw.circle(screen, GREEN, (cx + plant_r - 1, cy + 1), plant_r)

        # 上一步落点高亮
        if step.last_move is not None:
            r, c = step.last_move
            cx = board_x0 + c * cell
            cy = board_y0 + r * cell
            pygame.draw.circle(screen, BLUE, (cx, cy), stone_r + 6, 2)

        # 植物事件（文本提示）
        y0 = ly["info_h"] - (ly["font_small"] * 2 + ly["line_gap"] + 6)
        spawns = step.plant_events.get("spawn", [])
        clears = step.plant_events.get("clear", [])
        if spawns:
            _draw_text(
                screen,
                f"植物刷新(+): {[(r,c,d) for r,c,d in spawns][:5]}{' ...' if len(spawns)>5 else ''}",
                (x0, y0),
                ly["font_small"],
                BLUE,
            )
            y0 += ly["font_small"] + ly["line_gap"]
        if clears:
            _draw_text(
                screen,
                f"植物消失(-): {[(r,c,d) for r,c,d in clears][:5]}{' ...' if len(clears)>5 else ''}",
                (x0, y0),
                ly["font_small"],
                RED,
            )

        pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 1)
                elif event.key == pygame.K_RIGHT:
                    idx = min(len(trace) - 1, idx + 1)
                elif event.key == pygame.K_HOME:
                    idx = 0
                elif event.key == pygame.K_END:
                    idx = len(trace) - 1
                elif event.key == pygame.K_r:
                    idx = 0
            elif event.type == pygame.VIDEORESIZE:
                w = max(520, event.w)
                h = max(620, event.h)
                screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        draw_frame()
        clock.tick(60)

    pygame.quit()
    sys.exit()
