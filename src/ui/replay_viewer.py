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
INFO_H = 120
STONE_R = 18
PLANT_R = 6


def _draw_text(screen, txt, pos, size=24, color=BLACK):
    font = pygame.font.SysFont(None, size)
    screen.blit(font.render(txt, True, color), pos)


def launch_replay(trace: List, board_size: int = 9):
    """基于 trace（见 selfplay_trace.TraceStep 列表）启动回放 GUI。
    键位：Left/Right 上一步/下一步；Home/End 跳到首/尾；R 重开本回放；Esc 退出。
    """
    pygame.init()
    W = MARGIN*2 + board_size*CELL
    H = INFO_H + MARGIN*2 + board_size*CELL
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("4ascend - Self-play Replay Viewer")
    clock = pygame.time.Clock()

    idx = 0  # 当前帧索引

    def draw_frame():
        screen.fill(BG)
        step = trace[idx]
        size = board_size
        # 顶栏信息：步号/HP/阶段
        _draw_text(screen, f"Frame {idx+1}/{len(trace)}  |  Turn {step.turn}  |  Phase: {step.phase}", (MARGIN, MARGIN-8), 28)
        _draw_text(screen, f"BLACK HP: {step.hp[0]}   WHITE HP: {step.hp[1]}", (MARGIN, MARGIN+20), 22)
        _draw_text(screen, "←/→: 上一手/下一手   Home/End: 首/尾   R: 重开   Esc: 退出", (MARGIN, MARGIN+46), 20)

        # 棋盘与网格
        board_x0 = MARGIN
        board_y0 = INFO_H + MARGIN
        L = size * CELL
        pygame.draw.rect(screen, (230,230,230), (board_x0, board_y0, L, L))
        for i in range(size + 1):
            pygame.draw.line(screen, GRID, (board_x0, board_y0 + i*CELL), (board_x0 + L, board_y0 + i*CELL), 1)
            pygame.draw.line(screen, GRID, (board_x0 + i*CELL, board_y0), (board_x0 + i*CELL, board_y0 + L), 1)

        # 被无效化掩码高亮（若存在）
        if step.attack_chain_mask is not None:
            mask = step.attack_chain_mask
            rs, cs = np.where(mask)
            for r, c in zip(rs, cs):
                cx = board_x0 + c*CELL + CELL//2
                cy = board_y0 + r*CELL + CELL//2
                pygame.draw.circle(screen, ORANGE, (cx, cy), STONE_R + 4, 2)

        # 棋子
        for r in range(size):
            for c in range(size):
                v = step.grid[r, c]
                if v == 0:
                    continue
                cx = board_x0 + c*CELL + CELL//2
                cy = board_y0 + r*CELL + CELL//2
                color = BLACK if v == 1 else WHITE
                pygame.draw.circle(screen, color, (cx, cy), STONE_R)
                pygame.draw.circle(screen, STONE_OUTLINE, (cx, cy), STONE_R, 2)

        # 植物：以 0/1/2 株显示（两个小点表示 2 株）
        for r in range(size):
            for c in range(size):
                k = int(step.plants[r, c])
                if k <= 0:
                    continue
                cx = board_x0 + c*CELL + CELL//2
                cy = board_y0 + r*CELL + CELL//2
                if k == 1:
                    pygame.draw.circle(screen, GREEN, (cx, cy), PLANT_R)
                else:
                    pygame.draw.circle(screen, GREEN, (cx - PLANT_R, cy), PLANT_R)
                    pygame.draw.circle(screen, GREEN, (cx + PLANT_R, cy), PLANT_R)

        # 上一步落点高亮
        if step.last_move is not None:
            r, c = step.last_move
            cx = board_x0 + c*CELL + CELL//2
            cy = board_y0 + r*CELL + CELL//2
            pygame.draw.circle(screen, BLUE, (cx, cy), STONE_R + 6, 2)

        # 植物事件（文本提示）
        y0 = INFO_H - 16
        spawns = step.plant_events.get("spawn", [])
        clears = step.plant_events.get("clear", [])
        if spawns:
            _draw_text(screen, f"植物刷新(+): {[(r,c,d) for r,c,d in spawns][:5]}{' ...' if len(spawns)>5 else ''}", (MARGIN, y0), 20, BLUE)
            y0 += 22
        if clears:
            _draw_text(screen, f"植物消失(-): {[(r,c,d) for r,c,d in clears][:5]}{' ...' if len(clears)>5 else ''}", (MARGIN, y0), 20, RED)

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
        draw_frame()
        clock.tick(60)

    pygame.quit()
    sys.exit()
