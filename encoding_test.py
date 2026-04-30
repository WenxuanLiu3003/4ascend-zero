from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pygame

from src.core.board import Board
from src.core.encoding import AlphaZeroStateEncoder
from src.core.engine import Engine
from src.core.rules import RulesConfig
from src.core.state import GameState
from src.core.types import Move, Phase, Player


BG = (245, 245, 245)
PANEL = (232, 232, 232)
GRID = (175, 175, 175)
BLACK = (25, 25, 25)
WHITE = (238, 238, 238)
OUTLINE = (55, 55, 55)
GREEN = (55, 155, 75)
ORANGE = (245, 150, 60)
BLUE = (65, 105, 220)
RED = (205, 65, 65)
TEXT = (28, 28, 28)

BOARD_SIZE = 9
CELL = 46
MARGIN = 24
TOP_H = 126
SIDE_W = 116
GAP = 34
STONE_R = 17
PLANT_R = 6
CLICK_TOL = 16


PLANE_NAMES = [
    "0 me stones",
    "1 opponent stones",
    "2 attack mask",
    "3 plant count",
    "4 me HP",
    "5 opponent HP",
    "6 attack-defense",
    "7 black to play",
    "8 grow parity",
    "9 me >=4 potential",
    "10 opp >=4 potential",
]
PLANE_NAMES.extend(
    f"{idx} history {idx - 10}"
    for idx in range(11, 25)
)


def _reset_state(cfg: RulesConfig) -> GameState:
    return GameState(cfg=cfg, board=Board(cfg.board_size))


def _board_origin(left: int, top: int) -> tuple[int, int]:
    return left + CELL // 2, top + CELL // 2


def _rc_from_pos(pos: tuple[int, int], origin: tuple[int, int], size: int) -> Optional[tuple[int, int]]:
    x, y = pos
    x0, y0 = origin
    col = int(round((x - x0) / CELL))
    row = int(round((y - y0) / CELL))
    if not (0 <= row < size and 0 <= col < size):
        return None
    cx = x0 + col * CELL
    cy = y0 + row * CELL
    if abs(x - cx) <= CLICK_TOL and abs(y - cy) <= CLICK_TOL:
        return row, col
    return None


def _draw_button(screen, rect: pygame.Rect, text: str, font: pygame.font.Font, active: bool = False) -> None:
    fill = (210, 226, 248) if active else (238, 238, 238)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), rect, width=1, border_radius=6)
    label = font.render(text, True, TEXT)
    screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))


def _draw_grid(screen, origin: tuple[int, int], size: int) -> None:
    x0, y0 = origin
    span = (size - 1) * CELL
    pad = CELL // 2
    pygame.draw.rect(screen, PANEL, (x0 - pad, y0 - pad, span + pad * 2, span + pad * 2))
    ext = 8
    for i in range(size):
        x = x0 + i * CELL
        y = y0 + i * CELL
        pygame.draw.line(screen, GRID, (x0 - ext, y), (x0 + span + ext, y), 1)
        pygame.draw.line(screen, GRID, (x, y0 - ext), (x, y0 + span + ext), 1)


def _draw_state_board(screen, state: GameState, origin: tuple[int, int]) -> None:
    _draw_grid(screen, origin, state.board.size)
    x0, y0 = origin

    if state.attack_chain_mask is not None:
        for r, c in zip(*np.where(state.attack_chain_mask > 0)):
            pygame.draw.circle(screen, ORANGE, (x0 + c * CELL, y0 + r * CELL), STONE_R + 5, 2)

    for r in range(state.board.size):
        for c in range(state.board.size):
            v = int(state.board.grid[r, c])
            if v:
                color = BLACK if v == 1 else WHITE
                cx = x0 + c * CELL
                cy = y0 + r * CELL
                pygame.draw.circle(screen, color, (cx, cy), STONE_R)
                pygame.draw.circle(screen, OUTLINE, (cx, cy), STONE_R, 2)

            plants = int(state.board.plants[r, c])
            if plants == 1:
                pygame.draw.circle(screen, GREEN, (x0 + c * CELL + 3, y0 + r * CELL - 3), PLANT_R)
            elif plants >= 2:
                pygame.draw.circle(screen, GREEN, (x0 + c * CELL - 5, y0 + r * CELL - 1), PLANT_R)
                pygame.draw.circle(screen, GREEN, (x0 + c * CELL + 5, y0 + r * CELL + 1), PLANT_R)


def _plane_color(value: float, min_v: float, max_v: float) -> tuple[int, int, int]:
    if abs(max_v - min_v) < 1e-8:
        if abs(value) < 1e-8:
            return (245, 245, 245)
        return (75, 135, 220)
    t = float((value - min_v) / (max_v - min_v))
    t = max(0.0, min(1.0, t))
    low = np.array([245, 245, 245], dtype=np.float32)
    high = np.array([45, 105, 205], dtype=np.float32)
    rgb = (low * (1.0 - t) + high * t).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _draw_plane(screen, tensor: Optional[np.ndarray], channel: int, origin: tuple[int, int], font: pygame.font.Font) -> None:
    _draw_grid(screen, origin, BOARD_SIZE)
    if tensor is None:
        label = font.render("Press Encode", True, TEXT)
        screen.blit(label, (origin[0] - label.get_width() // 2 + CELL * 4, origin[1] + CELL * 4))
        return

    plane = tensor[channel]
    min_v = float(np.min(plane))
    max_v = float(np.max(plane))
    x0, y0 = origin
    pad = CELL // 2 - 2

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            value = float(plane[r, c])
            cx = x0 + c * CELL
            cy = y0 + r * CELL
            rect = pygame.Rect(cx - pad, cy - pad, pad * 2, pad * 2)
            pygame.draw.rect(screen, _plane_color(value, min_v, max_v), rect)
            pygame.draw.rect(screen, GRID, rect, width=1)
            if abs(value) > 1e-8:
                text = f"{value:g}"
                label = font.render(text, True, BLACK if value < max_v * 0.65 else WHITE)
                screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))


def _state_lines(state: GameState, encoded: Optional[np.ndarray], channel: int) -> list[str]:
    phase = "ATTACK_DEFENSE" if state.phase is Phase.ATTACK_DEFENSE else "NORMAL"
    last = "None" if state.last_move is None else f"({state.last_move.r},{state.last_move.c})"
    encoded_txt = "None" if encoded is None else f"{encoded.shape}, viewing C{channel}"
    return [
        f"Phase: {phase} | To Play: {state.to_play.name} | Turn: {state.turn}",
        f"BLACK HP: {int(state.hp[0])}/{state.cfg.hp_max}    WHITE HP: {int(state.hp[1])}/{state.cfg.hp_max}",
        f"grow_count={state.grow_count}  unascend_charge={state.unascend_charge}",
        f"over_fill={state.over_fill}  just_unascend={state.just_unascend}  last_move={last}",
        f"encoded={encoded_txt}",
    ]


def _print_state_and_encoding(state: GameState, encoded: np.ndarray, as_player: Player) -> None:
    print("\n" + "=" * 78)
    print(f"Encoding as_player={as_player.name}, tensor shape={encoded.shape}, dtype={encoded.dtype}")
    for line in _state_lines(state, encoded, 0):
        print(line)
    print("grid (0 empty, 1 black, 2 white):")
    print(state.board.grid)
    print("plants:")
    print(state.board.plants)
    if state.attack_chain_mask is not None:
        print("attack_chain_mask:")
        print(state.attack_chain_mask)
    for idx, plane in enumerate(encoded):
        name = PLANE_NAMES[idx] if idx < len(PLANE_NAMES) else f"{idx}"
        print(f"\nPlane {name}:")
        print(plane)


def _layout() -> dict[str, object]:
    board_top = TOP_H + MARGIN
    left_panel_x = MARGIN
    board_left = left_panel_x + SIDE_W + MARGIN
    plane_left = board_left + CELL * BOARD_SIZE + GAP
    board_origin = _board_origin(board_left, board_top)
    plane_origin = _board_origin(plane_left, board_top)
    return {
        "left_panel_x": left_panel_x,
        "board_left": board_left,
        "plane_left": plane_left,
        "board_top": board_top,
        "board_origin": board_origin,
        "plane_origin": plane_origin,
    }


def _buttons(edit_mode: Optional[str]) -> dict[str, pygame.Rect | bool]:
    ly = _layout()
    x = int(ly["left_panel_x"])
    y = int(ly["board_top"])
    h = 28
    gap = 9
    rects: dict[str, pygame.Rect | bool] = {}
    for name in ("black", "white", "plant", "exit_edit", "encode", "prev", "next", "reset"):
        rects[name] = pygame.Rect(x, y, SIDE_W, h)
        y += h + gap
    rects["black_active"] = edit_mode == "black"
    rects["white_active"] = edit_mode == "white"
    rects["plant_active"] = edit_mode == "plant"
    hp_y = TOP_H - 34
    rects["bhp_minus"] = pygame.Rect(730, hp_y, 28, 24)
    rects["bhp_plus"] = pygame.Rect(764, hp_y, 28, 24)
    rects["whp_minus"] = pygame.Rect(875, hp_y, 28, 24)
    rects["whp_plus"] = pygame.Rect(909, hp_y, 28, 24)
    rects["grow_minus"] = pygame.Rect(1032, hp_y, 28, 24)
    rects["grow_plus"] = pygame.Rect(1066, hp_y, 28, 24)
    return rects


def draw(
    screen,
    state: GameState,
    edit_mode: Optional[str],
    encoded: Optional[np.ndarray],
    channel: int,
) -> None:
    screen.fill(BG)
    ly = _layout()
    font = pygame.font.SysFont(None, 22)
    small = pygame.font.SysFont(None, 18)
    big = pygame.font.SysFont(None, 28)

    y = MARGIN
    for idx, line in enumerate(_state_lines(state, encoded, channel)):
        f = big if idx == 0 else font
        screen.blit(f.render(line, True, TEXT), (MARGIN, y))
        y += f.get_height() + 5

    btns = _buttons(edit_mode)
    _draw_button(screen, btns["black"], "Black", font, bool(btns["black_active"]))
    _draw_button(screen, btns["white"], "White", font, bool(btns["white_active"]))
    _draw_button(screen, btns["plant"], "Plant", font, bool(btns["plant_active"]))
    _draw_button(screen, btns["exit_edit"], "Exit Edit", font)
    _draw_button(screen, btns["encode"], "Encode", font)
    _draw_button(screen, btns["prev"], "< Plane", font)
    _draw_button(screen, btns["next"], "Plane >", font)
    _draw_button(screen, btns["reset"], "Reset", font)
    _draw_button(screen, btns["bhp_minus"], "-", small)
    _draw_button(screen, btns["bhp_plus"], "+", small)
    _draw_button(screen, btns["whp_minus"], "-", small)
    _draw_button(screen, btns["whp_plus"], "+", small)
    _draw_button(screen, btns["grow_minus"], "-", small)
    _draw_button(screen, btns["grow_plus"], "+", small)

    board_title = big.render("State", True, TEXT)
    plane_title = big.render(PLANE_NAMES[channel], True, TEXT)
    screen.blit(board_title, (int(ly["board_left"]), TOP_H))
    screen.blit(plane_title, (int(ly["plane_left"]), TOP_H))
    if encoded is not None:
        plane = encoded[channel]
        stats = f"min={np.min(plane):g} max={np.max(plane):g} sum={np.sum(plane):g}"
        screen.blit(small.render(stats, True, TEXT), (int(ly["plane_left"]), TOP_H + 28))

    _draw_state_board(screen, state, ly["board_origin"])
    _draw_plane(screen, encoded, channel, ly["plane_origin"], small)

    footer_y = TOP_H + MARGIN + CELL * BOARD_SIZE + 16
    hints = "[click] alternate move  [Left/Right] channel  [R] reset  [Esc] quit"
    if edit_mode:
        hints = f"Edit {edit_mode}: click board to toggle. " + hints
    screen.blit(font.render(hints, True, TEXT), (MARGIN, footer_y))
    pygame.display.flip()


def _handle_board_click(state: GameState, engine: Engine, rc: tuple[int, int], edit_mode: Optional[str]) -> GameState:
    r, c = rc
    if edit_mode == "plant":
        state.board.plants[r, c] = (int(state.board.plants[r, c]) + 1) % 3
        return state
    if edit_mode in ("black", "white"):
        target = 1 if edit_mode == "black" else 2
        state.board.grid[r, c] = 0 if int(state.board.grid[r, c]) == target else target
        return state
    if state.is_terminal() or int(state.board.grid[r, c]) != 0:
        return state
    try:
        return engine.step(state, Move(r, c))
    except AssertionError:
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the 25 encoding planes for a custom 4ascend state")
    parser.add_argument("--NoAutoRefresh", action="store_true", help="Keep plants unchanged after rule-engine moves")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("4ascend Encoding Test")
    screen = pygame.display.set_mode((1130, 640))
    clock = pygame.time.Clock()

    cfg = RulesConfig(board_size=BOARD_SIZE, win_k=4, hp_max=6)
    state = _reset_state(cfg)
    engine = Engine(win_k=cfg.win_k)
    encoder = AlphaZeroStateEncoder(last_k=8)
    encoded: Optional[np.ndarray] = None
    channel = 0
    edit_mode: Optional[str] = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = _reset_state(cfg)
                    encoded = None
                    channel = 0
                    edit_mode = None
                elif event.key == pygame.K_LEFT:
                    channel = (channel - 1) % encoder.num_planes
                elif event.key == pygame.K_RIGHT:
                    channel = (channel + 1) % encoder.num_planes
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    encoded = encoder.encode(state, as_player=state.to_play)
                    _print_state_and_encoding(state, encoded, state.to_play)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                ly = _layout()
                btns = _buttons(edit_mode)
                if btns["black"].collidepoint(event.pos):
                    edit_mode = "black"
                elif btns["white"].collidepoint(event.pos):
                    edit_mode = "white"
                elif btns["plant"].collidepoint(event.pos):
                    edit_mode = "plant"
                elif btns["exit_edit"].collidepoint(event.pos):
                    edit_mode = None
                elif btns["encode"].collidepoint(event.pos):
                    encoded = encoder.encode(state, as_player=state.to_play)
                    _print_state_and_encoding(state, encoded, state.to_play)
                elif btns["prev"].collidepoint(event.pos):
                    channel = (channel - 1) % encoder.num_planes
                elif btns["next"].collidepoint(event.pos):
                    channel = (channel + 1) % encoder.num_planes
                elif btns["reset"].collidepoint(event.pos):
                    state = _reset_state(cfg)
                    encoded = None
                    channel = 0
                    edit_mode = None
                elif btns["bhp_minus"].collidepoint(event.pos):
                    state.hp[0] = max(0, int(state.hp[0]) - 1)
                elif btns["bhp_plus"].collidepoint(event.pos):
                    state.hp[0] = min(cfg.hp_max, int(state.hp[0]) + 1)
                elif btns["whp_minus"].collidepoint(event.pos):
                    state.hp[1] = max(0, int(state.hp[1]) - 1)
                elif btns["whp_plus"].collidepoint(event.pos):
                    state.hp[1] = min(cfg.hp_max, int(state.hp[1]) + 1)
                elif btns["grow_minus"].collidepoint(event.pos):
                    state.grow_count = max(1, int(state.grow_count) - 1)
                elif btns["grow_plus"].collidepoint(event.pos):
                    state.grow_count += 1
                else:
                    rc = _rc_from_pos(event.pos, ly["board_origin"], cfg.board_size)
                    if rc is not None:
                        prev_plants = state.board.plants.copy()
                        state = _handle_board_click(state, engine, rc, edit_mode)
                        if args.NoAutoRefresh:
                            state.board.plants[:, :] = prev_plants

        draw(screen, state, edit_mode, encoded, channel)
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
