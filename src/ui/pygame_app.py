# ──────────────────────────────────────────────────────────────────────────────
# File: src/ui/pygame_app.py  （极简可视化 UI：点击下子，显示HP/植物/阶段）
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
import pygame
import numpy as np

from ..core.rules import RulesConfig
from ..core.board import Board
from ..core.state import GameState
from ..core.types import Player, Phase, Move
from ..core.engine import Engine

# 颜色与UI参数
BG = (245, 245, 245)
GRID = (180, 180, 180)
BLACK = (30, 30, 30)
WHITE = (235, 235, 235)
STONE_OUTLINE = (50, 50, 50)
GREEN = (60, 160, 75)
RED = (200, 60, 60)
BLUE = (70, 100, 220)
ORANGE = (245, 150, 60)

CELL = 48            # 每格像素
MARGIN = 32          # 棋盘外边距
INFO_H = 100         # 顶部信息栏高度（HP、阶段等）
STONE_R = 18         # 棋子半径
PLANT_R = 6          # 植物小圆半径（最多画两个）


def rc_from_pos(pos, size):
    x, y = pos
    # 计算棋盘左上角
    board_x0 = MARGIN
    board_y0 = INFO_H + MARGIN
    col = int((x - board_x0) // CELL)
    row = int((y - board_y0) // CELL)
    if 0 <= row < size and 0 <= col < size:
        # 棋格中心位置
        cx = board_x0 + col * CELL + CELL // 2
        cy = board_y0 + row * CELL + CELL // 2
        # 鼠标需在格内一定范围才算点击
        if abs(x - cx) <= CELL // 2 and abs(y - cy) <= CELL // 2:
            return row, col
    return None


def draw_board(screen, state: GameState):
    screen.fill(BG)
    size = state.board.size

    # 顶部信息栏
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 34)

    # HP 条
    hp_w = 280
    hp_h = 16
    pad = 10

    # 黑方（idx=0）
    pygame.draw.rect(screen, (220,220,220), (MARGIN, MARGIN, hp_w, hp_h))
    ratio_b = max(0.0, min(1.0, state.hp[0] / state.cfg.hp_max))
    pygame.draw.rect(screen, BLACK, (MARGIN, MARGIN, int(hp_w * ratio_b), hp_h))
    screen.blit(font.render(f"BLACK HP: {state.hp[0]}/{state.cfg.hp_max}", True, BLACK), (MARGIN, MARGIN + hp_h + 4))

    # 白方（idx=1）
    x2 = MARGIN + hp_w + 120
    pygame.draw.rect(screen, (220,220,220), (x2, MARGIN, hp_w, hp_h))
    ratio_w = max(0.0, min(1.0, state.hp[1] / state.cfg.hp_max))
    pygame.draw.rect(screen, (180,180,180), (x2, MARGIN, int(hp_w * ratio_w), hp_h))
    screen.blit(font.render(f"WHITE HP: {state.hp[1]}/{state.cfg.hp_max}", True, BLACK), (x2, MARGIN + hp_h + 4))

    # 阶段/当前行动方
    phase_txt = "ATTACK_DEFENSE" if state.phase == Phase.ATTACK_DEFENSE else "NORMAL"
    turn_txt = "BLACK" if state.to_play == Player.BLACK else "WHITE"
    screen.blit(bigfont.render(f"Phase: {phase_txt}  |  To Play: {turn_txt}  |  Turn: {state.turn}", True, BLACK), (MARGIN, INFO_H - 40))
    screen.blit(font.render("[R] 重开  [Esc] 退出", True, BLACK), (MARGIN, INFO_H - 16))

    # 棋盘矩形
    board_x0 = MARGIN
    board_y0 = INFO_H + MARGIN
    L = size * CELL
    pygame.draw.rect(screen, (230,230,230), (board_x0, board_y0, L, L))

    # 网格
    for i in range(size + 1):
        x = board_x0 + i * CELL
        y = board_y0 + i * CELL
        pygame.draw.line(screen, GRID, (board_x0, board_y0 + i*CELL), (board_x0 + L, board_y0 + i*CELL), 1)
        pygame.draw.line(screen, GRID, (board_x0 + i*CELL, board_y0), (board_x0 + i*CELL, board_y0 + L), 1)

    # 画被无效化连子掩码（攻防阶段提示）
    if state.attack_chain_mask is not None:
        mask = state.attack_chain_mask
        for r, c in zip(*np.where(mask)):
            cx = board_x0 + c*CELL + CELL//2
            cy = board_y0 + r*CELL + CELL//2
            pygame.draw.circle(screen, ORANGE, (cx, cy), STONE_R + 4, 2)

    # 画棋子
    for r in range(size):
        for c in range(size):
            v = state.board.grid[r, c]
            if v == 0:
                continue
            cx = board_x0 + c*CELL + CELL//2
            cy = board_y0 + r*CELL + CELL//2
            color = BLACK if v == 1 else WHITE
            pygame.draw.circle(screen, color, (cx, cy), STONE_R)
            pygame.draw.circle(screen, STONE_OUTLINE, (cx, cy), STONE_R, 2)

    # 画植物（以 0/1/2 株的小圆表示，>2 会被引擎限制）
    plant_font = pygame.font.SysFont(None, 20)
    for r in range(size):
        for c in range(size):
            k = int(state.board.plants[r, c])
            if k <= 0:
                continue
            cx = board_x0 + c*CELL + CELL//2
            cy = board_y0 + r*CELL + CELL//2
            if k == 1:
                pygame.draw.circle(screen, GREEN, (cx, cy), PLANT_R)
            else:
                # 2株：画成两枚偏移的绿点
                pygame.draw.circle(screen, GREEN, (cx - PLANT_R, cy), PLANT_R)
                pygame.draw.circle(screen, GREEN, (cx + PLANT_R, cy), PLANT_R)

    pygame.display.flip()


def main():
    pygame.init()
    # 配置规则
    cfg = RulesConfig(board_size=9, win_k=4, hp_max=6)
    board = Board(cfg.board_size)
    state = GameState(cfg=cfg, board=board)
    engine = Engine(win_k=cfg.win_k)

    W = MARGIN*2 + cfg.board_size*CELL
    H = INFO_H + MARGIN*2 + cfg.board_size*CELL
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("4ascend - Minimal UI")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # 重新开始
                    board = Board(cfg.board_size)
                    state = GameState(cfg=cfg, board=board)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                rc = rc_from_pos(event.pos, cfg.board_size)
                if rc is not None and not state.is_terminal():
                    r, c = rc
                    # NORMAL 或 ATTACK_DEFENSE 都允许点空位
                    if state.board.grid[r, c] == 0:
                        try:
                            state = engine.step(state, Move(r, c))
                        except AssertionError:
                            # 若处于 ATTACK_DEFENSE 阶段但未传动作等断言，这里忽略
                            pass
        draw_board(screen, state)
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

