# ──────────────────────────────────────────────────────────────────────────────
# File: src/ui/pygame_app.py  （极简可视化 UI：点击下子，显示HP/植物/阶段）
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
import argparse
import pygame
import numpy as np

from ..core.rules import RulesConfig
from ..core.board import Board
from ..core.state import GameState
from ..core.types import Player, Phase, Move
from ..core.engine import Engine
from ..utils.checkpoint import latest_checkpoint_path, load_checkpoint

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
RUN_RED = (220, 60, 60)

CELL = 48            # 每格像素
MARGIN = 32          # 棋盘外边距
INFO_H = 130         # 顶部信息栏高度（HP、阶段等）
STONE_R = 18         # 棋子半径
PLANT_R = 6          # 植物小圆半径（最多画两个）
GRID_EXT = 10        # 网格线向外延伸像素
CLICK_TOL = 16       # 点击吸附到交叉点的容差（像素）
RUN_MCTS_SIMS = 1600


class MCTSRunner:
    def __init__(self, cfg: RulesConfig, engine: Engine):
        # 延迟导入 AI 依赖，避免窗口创建前卡在 torch/cuda 初始化
        import torch
        from ..core.encoding import AlphaZeroStateEncoder
        from ..ai.model import PolicyValueNet
        from ..ai.mcts import MCTS

        self.cfg = cfg
        self.engine = engine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = AlphaZeroStateEncoder(last_k=8)
        self.model = PolicyValueNet(in_planes=self.encoder.num_planes, board_size=cfg.board_size).to(self.device)
        self.mcts = MCTS(
            self.model,
            self.encoder,
            self.engine,
            board_size=cfg.board_size,
            c_puct=2.0,
            sims=RUN_MCTS_SIMS,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.0,  # 推理建议关闭根噪声，保持稳定
            device=self.device,
        )
        self.loaded = False
        self.load_msg = "Model not loaded"

    def ensure_loaded(self):
        if self.loaded:
            return
        ckpt = latest_checkpoint_path("checkpoints")
        if ckpt is not None:
            load_checkpoint(ckpt, self.model, optimizer=None, map_location=self.device)
            self.load_msg = f"Loaded: {ckpt}"
        else:
            self.load_msg = "No checkpoint found, using random model"
        self.model.eval()
        self.loaded = True

    def best_move(self, state: GameState):
        self.ensure_loaded()
        pi, root = self.mcts.run(state, turn_related_sim=-1)
        legal_mask = (state.board.grid == 0).astype(np.float32).reshape(-1)
        pi = pi * legal_mask
        if pi.sum() <= 1e-8:
            return None, None
        a = int(np.argmax(pi))
        # 用该动作对应子节点 Q 近似“当前执手若下此处的胜率”
        q = root.children[a].Q if a in root.children else 0.0
        win_rate = float(np.clip((q + 1.0) * 0.5, 0.0, 1.0))
        return divmod(a, self.cfg.board_size), win_rate


def _makes_four(grid: np.ndarray, r: int, c: int, stone: int, need: int = 4) -> bool:
    """检查在 (r, c) 放置 stone 后，是否形成 need 连（含 need 及以上）。"""
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    n = grid.shape[0]
    for dr, dc in dirs:
        cnt = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < n and 0 <= cc < n and grid[rr, cc] == stone:
            cnt += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < n and 0 <= cc < n and grid[rr, cc] == stone:
            cnt += 1
            rr -= dr
            cc -= dc
        if cnt >= need:
            return True
    return False


def _compute_layout(win_w: int, win_h: int, size: int):
    margin = max(12, int(min(win_w, win_h) * 0.04))
    cell = max(18, int((win_w - 3 * margin) / max(1, size + 2)))
    side_w = max(92, int(cell * 2.3))
    for _ in range(2):
        side_w = max(92, int(cell * 2.3))
        max_cell_w = (win_w - 3 * margin - side_w) / max(1, size)
        info_h = max(96, int(cell * 2.8))
        info_h = min(info_h, int(win_h * 0.42))
        max_cell_h = (win_h - info_h - 2 * margin) / max(1, size)
        cell = max(18, int(min(max_cell_w, max_cell_h)))

    side_w = max(92, int(cell * 2.3))
    info_h = max(96, int(cell * 2.8))
    info_h = min(info_h, int(win_h * 0.42))
    board_area = size * cell
    content_w = side_w + margin + board_area
    content_x = max(margin, (win_w - content_w) // 2)
    board_outer_x = content_x + side_w + margin
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
        "click_tol": max(8, int(cell * 0.34)),
        "font_big": max(18, int(cell * 0.56)),
        "font_mid": max(14, int(cell * 0.42)),
        "font_small": max(12, int(cell * 0.34)),
        "line_gap": max(4, int(cell * 0.10)),
        "hp_h": max(10, int(cell * 0.22)),
        "side_x": content_x,
        "side_w": side_w,
        "board_outer_y": board_outer_y,
    }


def _draw_button(screen, rect, text, font, active=False):
    fill = (210, 225, 245) if active else (235, 235, 235)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), rect, width=1, border_radius=6)
    label = font.render(text, True, BLACK)
    tx = rect.x + (rect.w - label.get_width()) // 2
    ty = rect.y + (rect.h - label.get_height()) // 2
    screen.blit(label, (tx, ty))


def _build_buttons(layout, edit_mode):
    x0 = layout["side_x"]
    btn_h = max(24, int(layout["cell"] * 0.58))
    btn_w = max(70, layout["side_w"])
    gap = max(8, int(layout["cell"] * 0.25))
    y0 = max(layout["info_h"] + gap, layout["board_outer_y"])
    black_rect = pygame.Rect(x0, y0, btn_w, btn_h)
    white_rect = pygame.Rect(x0, y0 + btn_h + gap, btn_w, btn_h)
    plant_rect = pygame.Rect(x0, y0 + (btn_h + gap) * 2, btn_w, btn_h)
    run_rect = pygame.Rect(x0, y0 + (btn_h + gap) * 3, btn_w, btn_h)
    back_rect = pygame.Rect(x0, y0 + (btn_h + gap) * 4, btn_w, btn_h)
    return {
        "black": black_rect,
        "white": white_rect,
        "plant": plant_rect,
        "run": run_rect,
        "back": back_rect,
        "active_black": edit_mode == "black",
        "active_white": edit_mode == "white",
        "active_plant": edit_mode == "plant",
    }


def _build_grow_buttons(layout, win_w):
    """构建 grow_count +/- 按钮，位置贴近 grow_count 信息行。"""
    font = pygame.font.SysFont(None, layout["font_mid"])
    bigfont = pygame.font.SysFont(None, layout["font_big"])
    smallfont = pygame.font.SysFont(None, layout["font_small"])

    y = layout["margin"]
    y += bigfont.get_height() + layout["line_gap"]
    y += layout["hp_h"] + font.get_height() + layout["line_gap"]
    y += layout["hp_h"] + font.get_height() + layout["line_gap"]
    # 到达 grow_count 所在行
    line_h = smallfont.get_height()

    btn = max(18, int(layout["font_small"] * 1.2))
    gap = max(4, int(layout["line_gap"]))
    x = min(
        win_w - layout["margin"] - (btn * 2 + gap),
        layout["margin"] + int(win_w * 0.62),
    )
    y_btn = y + (line_h - btn) // 2
    plus_rect = pygame.Rect(x, y_btn, btn, btn)
    minus_rect = pygame.Rect(x + btn + gap, y_btn, btn, btn)
    return {"plus": plus_rect, "minus": minus_rect}


def rc_from_pos(pos, size, layout):
    x, y = pos
    board_x0 = layout["board_x0"]
    board_y0 = layout["board_y0"]
    cell = layout["cell"]
    col = int(round((x - board_x0) / cell))
    row = int(round((y - board_y0) / cell))
    if 0 <= row < size and 0 <= col < size:
        # 交叉点吸附
        cx = board_x0 + col * cell
        cy = board_y0 + row * cell
        if abs(x - cx) <= layout["click_tol"] and abs(y - cy) <= layout["click_tol"]:
            return row, col
    return None


def draw_board(
    screen,
    state: GameState,
    edit_mode: str | None,
    run_best_rc,
    run_best_winrate,
    run_busy: bool,
    run_msg: str,
    show_refresh_notice: bool,
):
    screen.fill(BG)
    size = state.board.size
    win_w, win_h = screen.get_size()
    ly = _compute_layout(win_w, win_h, size)

    font = pygame.font.SysFont(None, ly["font_mid"])
    bigfont = pygame.font.SysFont(None, ly["font_big"])
    smallfont = pygame.font.SysFont(None, ly["font_small"])

    hp_h = ly["hp_h"]
    text_h = font.get_height()
    small_h = smallfont.get_height()
    x0 = ly["margin"]
    y = ly["margin"]

    phase_txt = "ATTACK_DEFENSE" if state.phase == Phase.ATTACK_DEFENSE else "NORMAL"
    turn_txt = "BLACK" if state.to_play == Player.BLACK else "WHITE"
    mode_txt = f"EDIT: {edit_mode.upper()}" if edit_mode else "EDIT: OFF"
    screen.blit(
        bigfont.render(f"Phase: {phase_txt} | To Play: {turn_txt} | Turn: {state.turn} | {mode_txt}", True, BLACK),
        (x0, y),
    )
    y += bigfont.get_height() + ly["line_gap"]

    hp_w = max(160, min(win_w - 2 * x0, int(win_w * 0.40)))
    pygame.draw.rect(screen, (220, 220, 220), (x0, y, hp_w, hp_h))
    ratio_b = max(0.0, min(1.0, state.hp[0] / state.cfg.hp_max))
    pygame.draw.rect(screen, BLACK, (x0, y, int(hp_w * ratio_b), hp_h))
    screen.blit(font.render(f"BLACK HP: {state.hp[0]}/{state.cfg.hp_max}", True, BLACK), (x0, y + hp_h + 2))
    y += hp_h + text_h + ly["line_gap"]

    pygame.draw.rect(screen, (220, 220, 220), (x0, y, hp_w, hp_h))
    ratio_w = max(0.0, min(1.0, state.hp[1] / state.cfg.hp_max))
    pygame.draw.rect(screen, (180, 180, 180), (x0, y, int(hp_w * ratio_w), hp_h))
    screen.blit(font.render(f"WHITE HP: {state.hp[1]}/{state.cfg.hp_max}", True, BLACK), (x0, y + hp_h + 2))
    y += hp_h + text_h + ly["line_gap"]

    screen.blit(
        smallfont.render(
            f"over_fill={state.over_fill}  just_unascend={state.just_unascend}  "
            f"grow_count={state.grow_count}  unascend_charge={state.unascend_charge}",
            True,
            BLACK,
        ),
        (x0, y),
    )
    grow_btns = _build_grow_buttons(ly, win_w)
    _draw_button(screen, grow_btns["plus"], "+", smallfont, active=False)
    _draw_button(screen, grow_btns["minus"], "-", smallfont, active=False)
    y += small_h + ly["line_gap"]
    screen.blit(smallfont.render("[R] 重开  [Esc] 退出", True, BLACK), (x0, y))
    y += small_h + ly["line_gap"]
    if run_msg:
        screen.blit(smallfont.render(run_msg, True, RUN_RED if run_busy else BLACK), (x0, y))
    elif show_refresh_notice:
        refresh_font = pygame.font.SysFont(None, max(28, int(ly["font_small"] * 1.8)))
        screen.blit(refresh_font.render("Refresh", True, RUN_RED), (x0, y))

    buttons = _build_buttons(ly, edit_mode)
    _draw_button(screen, buttons["black"], "Black", smallfont, buttons["active_black"])
    _draw_button(screen, buttons["white"], "White", smallfont, buttons["active_white"])
    _draw_button(screen, buttons["plant"], "Plant", smallfont, buttons["active_plant"])
    _draw_button(screen, buttons["run"], "Run", smallfont, active=run_busy)
    _draw_button(screen, buttons["back"], "Back", smallfont, active=False)

    board_x0 = ly["board_x0"]
    board_y0 = ly["board_y0"]
    span = ly["board_span"]
    board_pad = ly["board_pad"]
    cell = ly["cell"]
    stone_r = ly["stone_r"]
    plant_r = ly["plant_r"]
    grid_ext = ly["grid_ext"]
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

    if state.attack_chain_mask is not None:
        mask = state.attack_chain_mask
        for r, c in zip(*np.where(mask)):
            cx = board_x0 + c * cell
            cy = board_y0 + r * cell
            pygame.draw.circle(screen, ORANGE, (cx, cy), stone_r + 4, 2)

    for r in range(size):
        for c in range(size):
            v = state.board.grid[r, c]
            if v == 0:
                continue
            cx = board_x0 + c * cell
            cy = board_y0 + r * cell
            color = BLACK if v == 1 else WHITE
            pygame.draw.circle(screen, color, (cx, cy), stone_r)
            pygame.draw.circle(screen, STONE_OUTLINE, (cx, cy), stone_r, 2)

    for r in range(size):
        for c in range(size):
            k = int(state.board.plants[r, c])
            if k <= 0:
                continue
            cx = board_x0 + c * cell
            cy = board_y0 + r * cell
            if k == 1:
                pygame.draw.circle(screen, GREEN, (cx + max(1, plant_r // 2), cy - max(1, plant_r // 2)), plant_r)
            else:
                pygame.draw.circle(screen, GREEN, (cx - plant_r + 1, cy - 1), plant_r)
                pygame.draw.circle(screen, GREEN, (cx + plant_r - 1, cy + 1), plant_r)

    if run_best_rc is not None:
        rr, cc = run_best_rc
        cx = board_x0 + cc * cell
        cy = board_y0 + rr * cell
        pygame.draw.circle(screen, RUN_RED, (cx, cy), stone_r + 8, 3)
        if run_best_winrate is not None:
            txt = f"{int(round(run_best_winrate * 100))}%"
            rate_font = pygame.font.SysFont(None, max(14, int(stone_r * 1.2)))
            label = rate_font.render(txt, True, RUN_RED)
            lx = cx - label.get_width() // 2
            ly_txt = cy - label.get_height() // 2
            screen.blit(label, (lx, ly_txt))

    pygame.display.flip()
    return ly, buttons, grow_btns


def main():
    parser = argparse.ArgumentParser(description="4ascend pygame app")
    parser.add_argument("--NoAutoRefresh", action="store_true", help="Disable plant auto-refresh on real moves")
    args = parser.parse_args()

    pygame.init()
    # 配置规则
    cfg = RulesConfig(board_size=9, win_k=4, hp_max=6)
    board = Board(cfg.board_size)
    state = GameState(cfg=cfg, board=board)
    engine = Engine(win_k=cfg.win_k)
    mcts_runner = None

    board_span = (cfg.board_size - 1) * CELL
    board_pad = CELL // 2
    W = 2 * (MARGIN * 2 + board_span + board_pad * 2)
    H = 2 * (INFO_H + MARGIN * 2 + board_span + board_pad * 2)
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("4ascend - Minimal UI")
    clock = pygame.time.Clock()

    running = True
    edit_mode: str | None = None  # None | "black" | "white" | "plant"
    run_busy = False
    run_best_rc = None
    run_best_winrate = None
    run_msg = ""
    refresh_notice_turn = -1
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
                    edit_mode = None
                    run_best_rc = None
                    run_best_winrate = None
                    run_busy = False
                    run_msg = ""
                    refresh_notice_turn = -1
            elif event.type == pygame.VIDEORESIZE:
                w = max(520, event.w)
                h = max(620, event.h)
                screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                layout = _compute_layout(*screen.get_size(), cfg.board_size)
                grow_btns = _build_grow_buttons(layout, screen.get_width())
                if grow_btns["plus"].collidepoint(event.pos):
                    state.grow_count += 1
                    continue
                if grow_btns["minus"].collidepoint(event.pos):
                    if state.grow_count > 1:
                        state.grow_count -= 1
                    continue
                buttons = _build_buttons(layout, edit_mode)
                if buttons["black"].collidepoint(event.pos):
                    if run_busy:
                        continue
                    edit_mode = "black"
                    continue
                if buttons["white"].collidepoint(event.pos):
                    if run_busy:
                        continue
                    edit_mode = "white"
                    continue
                if buttons["plant"].collidepoint(event.pos):
                    if run_busy:
                        continue
                    edit_mode = "plant"
                    continue
                if buttons["run"].collidepoint(event.pos):
                    if edit_mode is not None or run_busy or state.is_terminal():
                        continue
                    run_busy = True
                    run_msg = "Initializing MCTS..."
                    draw_board(
                        screen, state, edit_mode, run_best_rc, run_best_winrate, run_busy, run_msg,
                        show_refresh_notice=(refresh_notice_turn == state.turn),
                    )
                    try:
                        if mcts_runner is None:
                            mcts_runner = MCTSRunner(cfg, engine)
                        run_msg = "Running MCTS..."
                        draw_board(
                            screen, state, edit_mode, run_best_rc, run_best_winrate, run_busy, run_msg,
                            show_refresh_notice=(refresh_notice_turn == state.turn),
                        )
                        run_best_rc, run_best_winrate = mcts_runner.best_move(state)
                        run_msg = "MCTS done" if run_best_rc is not None else "No legal move"
                    except Exception as exc:
                        run_best_rc = None
                        run_best_winrate = None
                        run_msg = f"Run failed: {exc}"
                    run_busy = False
                    continue
                if buttons["back"].collidepoint(event.pos):
                    if run_busy:
                        continue
                    edit_mode = None
                    continue

                if run_busy:
                    continue
                rc = rc_from_pos(event.pos, cfg.board_size, layout)
                if rc is None:
                    continue
                r, c = rc

                if edit_mode is not None:
                    if edit_mode == "plant":
                        curp = int(state.board.plants[r, c])
                        state.board.plants[r, c] = (curp + 1) % 3
                    else:
                        target = 1 if edit_mode == "black" else 2
                        other = 2 if target == 1 else 1
                        cur = int(state.board.grid[r, c])
                        if cur == target:
                            state.board.grid[r, c] = 0
                        elif cur == other:
                            pass
                        else:
                            state.board.grid[r, c] = target
                            if _makes_four(state.board.grid, r, c, target, need=4):
                                state.board.grid[r, c] = 0
                elif not state.is_terminal():
                    if state.board.grid[r, c] == 0:
                        try:
                            prev_plants = state.board.plants.copy()
                            state = engine.step(state, Move(r, c))
                            if args.NoAutoRefresh and not np.array_equal(state.board.plants, prev_plants):
                                state.board.plants[:, :] = prev_plants
                                refresh_notice_turn = state.turn
                                edit_mode = "plant"
                            run_best_rc = None
                            run_best_winrate = None
                            run_msg = ""
                        except AssertionError:
                            pass
        draw_board(
            screen, state, edit_mode, run_best_rc, run_best_winrate, run_busy, run_msg,
            show_refresh_notice=(refresh_notice_turn == state.turn),
        )
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
