import cv2
import numpy as np

from datetime import datetime

from .dashboard_utils import (
    create_output_dirs,
    calculate_percent_change,
    get_severity,
    evaluate_mr,
    save_violation_image,
    save_results_csv,
    update_summary
)

from .control_panel import DashboardConfig
from .graph_panel import draw_summary_panel


# ─────────────────────────────────────────────────────────────────────────────
#  Design tokens  —  edit here to retheme the entire dashboard
# ─────────────────────────────────────────────────────────────────────────────

class Theme:
    # Backgrounds
    BG_DEEP       = (18,  20,  26)    # near-black canvas
    BG_PANEL      = (26,  30,  40)    # tile / panel fill
    BG_CARD       = (32,  37,  52)    # card fill inside panels
    BG_HEADER     = (16,  18,  24)    # top header strip (slightly darker)

    # Accents (BGR)
    ACCENT_CYAN   = (210, 220,   0)   # primary — vivid cyan
    ACCENT_GREEN  = ( 80, 210, 100)   # pass / ok
    ACCENT_RED    = ( 60,  60, 230)   # fail / alert
    ACCENT_AMBER  = ( 30, 180, 230)   # warning / epsilon

    # Text
    TEXT_PRIMARY   = (230, 232, 238)
    TEXT_SECONDARY = (130, 135, 155)
    TEXT_DIM       = ( 60,  65,  82)

    # Structure
    BORDER         = ( 45,  50,  68)
    SEPARATOR      = ( 36,  40,  56)

T = Theme   # shorthand throughout


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def _fr(img, x1, y1, x2, y2, color, alpha=1.0):
    """Filled rectangle, optional alpha blend."""
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _br(img, x1, y1, x2, y2, color, t=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)


def _txt(img, s, x, y, color=T.TEXT_PRIMARY, scale=0.50, bold=False,
         font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, s, (x, y), font, scale, color,
                2 if bold else 1, cv2.LINE_AA)


def _hline(img, y, x1, x2, color=T.SEPARATOR, t=1):
    cv2.line(img, (x1, y), (x2, y), color, t, cv2.LINE_AA)


def _vline(img, x, y1, y2, color=T.SEPARATOR, t=1):
    cv2.line(img, (x, y1), (x, y2), color, t, cv2.LINE_AA)


def _tw(s, scale=0.50, bold=False):
    """Return pixel width of a string at given scale."""
    (w, _), _ = cv2.getTextSize(
        s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 if bold else 1)
    return w


def _pill(img, label, x, y, bg, fg=T.BG_DEEP, scale=0.38, pad=5):
    """
    Compact pill tag.  Returns the x coordinate of the right edge.
    Uses only ASCII — OpenCV built-in fonts don't support Unicode.
    """
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x2 = x + tw + pad * 2
    y1 = y - th - pad + 1
    y2 = y + pad
    _fr(img, x, y1, x2, y2, bg)
    cv2.putText(img, label, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, 1, cv2.LINE_AA)
    return x2 + 8   # right edge + gap for next pill


def _bar(img, x, y, w, h, value, max_value, fill, track=T.BG_CARD):
    _fr(img, x, y, x + w, y + h, track)
    _br(img, x, y, x + w, y + h, T.BORDER)
    if max_value > 0 and value > 0:
        fw = int(w * min(value / max_value, 1.0))
        if fw > 0:
            _fr(img, x, y, x + fw, y + h, fill)


# ─────────────────────────────────────────────────────────────────────────────
#  Hold overlay  —  drawn on cached result tiles while TTL counts down
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hold_overlay(tile, ttl, ttl_max):
    """
    Top-right corner badge showing "HOLD" + a draining countdown bar.
    The bar empties left-to-right as frames tick down so the user always
    knows how long the result will stay on screen.
    """
    h, w = tile.shape[:2]

    BADGE_W = 72
    BADGE_H = 20
    BAR_H   = 4
    PAD     = 6

    bx = w - BADGE_W - PAD
    by = PAD

    # Badge background
    _fr(tile, bx, by, bx + BADGE_W, by + BADGE_H, T.BG_DEEP, alpha=0.82)

    # "HOLD" label
    _txt(tile, "HOLD", bx + 5, by + 14, T.ACCENT_AMBER, scale=0.38)

    # Remaining time as mm:ss-style fraction text
    frac_str = f"{ttl}/{ttl_max}"
    fw = _tw(frac_str, scale=0.32)
    _txt(tile, frac_str, bx + BADGE_W - fw - 4, by + 14,
         T.TEXT_DIM, scale=0.32)

    # Countdown bar (drains as ttl decreases)
    bar_y = by + BADGE_H + 2
    _fr(tile, bx, bar_y, bx + BADGE_W, bar_y + BAR_H, T.BG_CARD)
    filled = int(BADGE_W * (ttl / ttl_max)) if ttl_max > 0 else 0
    if filled > 0:
        # Colour shifts amber -> dim as time runs out
        frac = ttl / ttl_max
        bar_color = T.ACCENT_AMBER if frac > 0.4 else T.TEXT_DIM
        _fr(tile, bx, bar_y, bx + filled, bar_y + BAR_H, bar_color)


# ─────────────────────────────────────────────────────────────────────────────
#  Tile overlay
# ─────────────────────────────────────────────────────────────────────────────

def _style_tile(tile, label, pred_value, is_original=False, status=None):
    """
    Draws a styled bottom-bar overlay onto a video tile in-place.
    Colour-codes by role: green = original, cyan = pass, red = fail.
    """
    h, w = tile.shape[:2]
    BAR_H = 36

    accent = (
        T.ACCENT_GREEN if is_original else
        T.ACCENT_RED   if status == "FAIL" else
        T.ACCENT_CYAN
    )

    # Semi-transparent bottom strip
    _fr(tile, 0, h - BAR_H, w, h, T.BG_DEEP, alpha=0.75)

    # Top edge of strip
    cv2.line(tile, (0, h - BAR_H), (w, h - BAR_H), accent, 2, cv2.LINE_AA)

    # Role pill
    role_label = "ORIGINAL" if is_original else label.upper()[:10]
    _pill(tile, role_label, 8, h - BAR_H + 18, accent)

    # Prediction value — right-aligned
    pred_str = f"{pred_value:.4f}"
    px = w - _tw(pred_str, scale=0.55, bold=True) - 8
    _txt(tile, pred_str, px, h - BAR_H + 20, T.TEXT_PRIMARY, scale=0.55, bold=True)

    # Corner brackets (signature element — 4 corners of bottom strip only)
    BL = 10
    BT = 2
    corners = [
        ((0,       h - BAR_H),  ( 1,  1)),
        ((w - 1,   h - BAR_H),  (-1,  1)),
        ((0,       h - 1),      ( 1, -1)),
        ((w - 1,   h - 1),      (-1, -1)),
    ]
    for (cx, cy), (dx, dy) in corners:
        cv2.line(tile, (cx, cy), (cx + dx * BL, cy), accent, BT, cv2.LINE_AA)
        cv2.line(tile, (cx, cy), (cx, cy + dy * BL), accent, BT, cv2.LINE_AA)

    return tile


# ─────────────────────────────────────────────────────────────────────────────
#  Header bar  —  two rows: [brand | pills | stats]  then  [hint]
# ─────────────────────────────────────────────────────────────────────────────

HEADER_H = 64   # enough for two comfortable rows

def _draw_header(canvas, width, frame_id, active_count,
                 epsilon, current_mr, total_tests, total_failures):
    """
    Row 1 (y=0..46):  AutoMR brand  |  status pills  |  right-aligned stats
    Row 2 (y=46..64): dim hint text spanning full width
    All strings are pure ASCII — OpenCV built-in fonts do not support Unicode.
    """
    _fr(canvas, 0, 0, width, HEADER_H, T.BG_HEADER)

    # Subtle separator between row 1 and row 2
    _hline(canvas, 46, 0, width, T.BORDER, t=1)
    # Bottom border
    _hline(canvas, HEADER_H - 1, 0, width, T.BORDER, t=1)

    # ── ROW 1 ────────────────────────────────────────────────────────────────

    # Brand  (vertically centred in row 1: baseline ~32)
    _txt(canvas, "AutoMR", 14, 32, T.ACCENT_CYAN, scale=0.72, bold=True)

    # Divider after brand
    _vline(canvas, 98, 6, 40, T.BORDER)

    # Status pills — single row, baseline y=32
    x = 110
    mr_label = current_mr.upper()[:14]
    x = _pill(canvas, f"MR: {mr_label}",    x, 32, T.BG_CARD, T.ACCENT_CYAN,    scale=0.40)
    x = _pill(canvas, f"ON: {active_count}", x, 32, T.BG_CARD, T.ACCENT_GREEN,   scale=0.40)
    x = _pill(canvas, f"eps:{epsilon:.3f}",  x, 32, T.BG_CARD, T.ACCENT_AMBER,   scale=0.40)
    x = _pill(canvas, f"F:{frame_id}",       x, 32, T.BG_CARD, T.TEXT_SECONDARY, scale=0.38)

    # ── Right-aligned stats (row 1) ───────────────────────────────────────────
    fail_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0.0
    rate_color = (
        T.ACCENT_RED   if fail_rate > 20 else
        T.ACCENT_AMBER if fail_rate > 5  else
        T.ACCENT_GREEN
    )
    stats = [
        (f"Tests:{total_tests}",     T.TEXT_SECONDARY),
        (f"Fails:{total_failures}",  T.ACCENT_RED if total_failures > 0 else T.TEXT_SECONDARY),
        (f"Rate:{fail_rate:.1f}%",   rate_color),
    ]
    rx = width - 14
    for s, col in reversed(stats):
        rx -= _tw(s, scale=0.44) + 18
        _txt(canvas, s, rx, 30, col, scale=0.44, bold=True)

    # ── ROW 2: hint ───────────────────────────────────────────────────────────
    _txt(canvas,
         "1-9 / V / D  toggle MRs      R  run benchmark      ESC  quit",
         14, HEADER_H - 10,
         T.TEXT_DIM, scale=0.34)


# ─────────────────────────────────────────────────────────────────────────────
#  Main class  —  structure and logic unchanged
# ─────────────────────────────────────────────────────────────────────────────

class LiveDashboard:

    def __init__(
        self,
        automr,
        model,
        frame_skip=30,
        save_results=True,
        save_violations=True,
        output_dir="results/live_dashboard"
    ):
        self.pending_test = False
        self.testing = False
        self.progress = 0
        self.total_progress = 0

        self.automr = automr
        self.model = model

        self.config = DashboardConfig(
            automr
        )

        self.frame_skip = frame_skip
        self.save_results = save_results
        self.save_violations = save_violations
        self.output_dir = output_dir

        self.results = []

        self.total_tests = 0
        self.total_failures = 0

        self.CELL_W = 320
        self.CELL_H = 240
        self.current_epsilon = 0.05

        # ── Result hold cache ─────────────────────────────────────────────────
        # Keyed by mr_name -> {tile, status, pred, ttl, ttl_max}
        # Holds the worst-case result tile visible for HOLD_FRAMES frames
        # after each test run, so the user can actually see what happened.
        self.HOLD_FRAMES = 90       # ~3 s at 30 fps — tune freely
        self._tile_cache = {}

        create_output_dirs(
            self.output_dir
        )

    def get_test_parameters(
        self,
        mr_name
    ):

        settings = (
            self.config
            .mr_ranges[mr_name]
        )

        return np.linspace(
            settings["start"],
            settings["end"],
            settings["tests"]
        )

    def update_controls(self):

        tests = cv2.getTrackbarPos(
            "Tests",
            "AutoMR Live Dashboard"
        )

        epsilon = cv2.getTrackbarPos(
            "Epsilon x1000",
            "AutoMR Live Dashboard"
        )

        epsilon = max(epsilon, 1) / 1000.0

        frame_skip = cv2.getTrackbarPos(
            "FrameSkip",
            "AutoMR Live Dashboard"
        )

        range_scale = cv2.getTrackbarPos(
            "Range %",
            "AutoMR Live Dashboard"
        )

        mr_index = cv2.getTrackbarPos(
            "MR Index",
            "AutoMR Live Dashboard"
        )

        enabled = cv2.getTrackbarPos(
            "Enable",
            "AutoMR Live Dashboard"
        )

        if abs(epsilon - self.current_epsilon) > 1e-6:

            self.current_epsilon = epsilon

            self.automr.set_epsilon(epsilon)

        mrs = [
            mr
            for mr in self.automr.list_transforms()
            if mr != "temporal"
        ]

        self.config.current_mr = mrs[mr_index]

        current = self.config.current_mr

        if enabled == 1:

            if current not in self.config.selected_mrs:
                self.config.selected_mrs.append(current)

        else:

            if current in self.config.selected_mrs:
                self.config.selected_mrs.remove(current)

        self.frame_skip = max(
            frame_skip,
            1
        )

        current = (
            self.config.current_mr
        )

        start, end = (
            self.automr.mr_ranges[current]
        )

        scaled_end = (
            start +
            ((end - start) *
            range_scale / 100.0)
        )

        self.config.mr_ranges[
            current
        ]["start"] = start

        self.config.mr_ranges[
            current
        ]["end"] = scaled_end

        self.config.mr_ranges[
            current
        ]["tests"] = max(
            tests,
            1
        )

    def handle_keys(self, key):

        mapping = {
            ord("1"): "brightness",
            ord("2"): "rotation",
            ord("3"): "translation",
            ord("4"): "noise",
            ord("5"): "blur",
            ord("6"): "contrast",
            ord("7"): "rain",
            ord("8"): "snow",
            ord("9"): "fog",
            ord("v"): "visibility",
            ord("d"): "darkness"
        }

        if key in mapping:

            mr = mapping[key]

            self.config.current_mr = mr

            if mr in self.config.selected_mrs:

                self.config.selected_mrs.remove(
                    mr
                )

            else:

                self.config.selected_mrs.append(
                    mr
                )

    def process_frame(
        self,
        frame,
        frame_id
    ):

        tiles = []

        original_pred = float(
            self.model.predict(frame)
        )

        # ── Original tile ─────────────────────────────────────────────────────
        original_tile = frame.copy()
        _style_tile(
            original_tile,
            "original",
            original_pred,
            is_original=True
        )
        tiles.append(original_tile)

        # ── Decide whether to run tests this frame ────────────────────────────
        # Only the *current* MR is tested each cycle.
        # All other selected MRs are served from their cached result tiles.
        # This keeps the loop responsive and lets the user focus on one MR
        # at a time via the MR Index trackbar or 1-9/V/D keys.
        run_tests = (frame_id % self.frame_skip == 0)

        for mr_name in self.config.selected_mrs:

            if run_tests and mr_name == self.config.current_mr:

                # ── Run the full MR test sweep ────────────────────────────────
                try:

                    transform = (
                        self.automr
                        .transform_registry
                        .get(mr_name)
                    )

                    parameters = (
                        self.get_test_parameters(
                            mr_name
                        )
                    )

                    best_tile   = None
                    best_status = None
                    best_pred   = None
                    worst_diff  = -1

                    for param in parameters:

                        transformed = transform(
                            frame.copy(),
                            float(param)
                        )

                        transformed_pred = float(
                            self.model.predict(
                                transformed
                            )
                        )

                        diff, pct = (
                            calculate_percent_change(
                                original_pred,
                                transformed_pred
                            )
                        )

                        status = evaluate_mr(
                            self.automr,
                            mr_name,
                            original_pred,
                            transformed_pred
                        )

                        severity = get_severity(diff)

                        self.total_tests += 1

                        if status == "FAIL":

                            self.total_failures += 1

                            if self.save_violations:

                                save_violation_image(
                                    self.output_dir,
                                    mr_name,
                                    frame_id,
                                    transformed
                                )

                        self.results.append({
                            "timestamp":             datetime.now(),
                            "frame_id":              frame_id,
                            "mr":                    mr_name,
                            "parameter":             float(param),
                            "original_prediction":   original_pred,
                            "transformed_prediction": transformed_pred,
                            "difference":            diff,
                            "percent_change":        pct,
                            "status":                status,
                            "severity":              severity,
                            "epsilon":               self.current_epsilon,
                        })

                        if diff > worst_diff:
                            worst_diff  = diff
                            best_tile   = transformed
                            best_status = status
                            best_pred   = transformed_pred

                    # ── Store result in hold cache ─────────────────────────────
                    if best_tile is not None:
                        styled = best_tile.copy()
                        _style_tile(
                            styled,
                            mr_name,
                            best_pred,
                            is_original=False,
                            status=best_status
                        )
                        self._tile_cache[mr_name] = {
                            "tile":    styled,
                            "status":  best_status,
                            "pred":    best_pred,
                            "ttl":     self.HOLD_FRAMES,
                            "ttl_max": self.HOLD_FRAMES,
                        }

                except Exception as e:

                    print(f"{mr_name}: {e}")

            # ── Serve cached tile (decrement TTL each frame) ──────────────────
            entry = self._tile_cache.get(mr_name)

            if entry is not None and entry["ttl"] > 0:

                # Draw fresh hold overlay onto a copy so we don't
                # mutate the stored tile (TTL text would compound)
                display_tile = entry["tile"].copy()
                _draw_hold_overlay(
                    display_tile,
                    entry["ttl"],
                    entry["ttl_max"]
                )
                tiles.append(display_tile)

                # Tick down only on non-test frames so the hold
                # duration is purely real-time, not test-time
                if not run_tests:
                    entry["ttl"] -= 1

        return tiles

    def run_selected_benchmark(
            self,
            frame,
            frame_id
        ):

            self.testing = True

            for mr_name in self.config.selected_mrs:

                parameters = self.get_test_parameters(
                    mr_name
                )

                self.total_progress += len(parameters)

                for param in parameters:

                    # existing testing code
                    # prediction
                    # diff
                    # save result

                    self.progress += 1

            self.testing = False
            self.pending_test = False

    def build_dashboard(
        self,
        tiles
    ):

        resized = []

        for tile in tiles:

            resized.append(
                cv2.resize(
                    tile,
                    (self.CELL_W, self.CELL_H)
                )
            )

        rows = []

        for i in range(0, len(resized), 3):

            row = resized[i:i + 3]

            while len(row) < 3:

                # ── Empty cell placeholder ────────────────────────────────────
                ph = np.full(
                    (self.CELL_H, self.CELL_W, 3),
                    T.BG_DEEP,
                    dtype=np.uint8
                )
                _br(ph, 1, 1, self.CELL_W - 2, self.CELL_H - 2, T.BORDER)

                # Centred "NO SIGNAL" label
                label = "NO SIGNAL"
                lw = _tw(label, scale=0.42)
                _txt(
                    ph, label,
                    (self.CELL_W - lw) // 2,
                    self.CELL_H // 2 + 5,
                    T.TEXT_DIM, scale=0.42
                )
                row.append(ph)

            rows.append(np.hstack(row))

        if len(rows) == 0:
            return np.full(
                (self.CELL_H, self.CELL_W, 3),
                T.BG_DEEP,
                dtype=np.uint8
            )

        # ── 1-px separator between tile rows ─────────────────────────────────
        grid = rows[0]
        for row in rows[1:]:
            sep = np.full((1, row.shape[1], 3), T.SEPARATOR, dtype=np.uint8)
            grid = np.vstack([grid, sep, row])

        return grid

    def run(
        self,
        video_source=0
    ):

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {video_source}"
            )

        frame_id = 0

        cv2.namedWindow("AutoMR Live Dashboard")

        cv2.createTrackbar(
            "MR Index",
            "AutoMR Live Dashboard",
            0,
            len([
                mr for mr in self.automr.list_transforms()
                if mr != "temporal"
            ]) - 1,
            lambda x: None
        )

        cv2.createTrackbar(
            "Enable",
            "AutoMR Live Dashboard",
            1, 1,
            lambda x: None
        )

        cv2.createTrackbar(
            "Tests",
            "AutoMR Live Dashboard",
            5, 100,
            lambda x: None
        )

        cv2.createTrackbar(
            "FrameSkip",
            "AutoMR Live Dashboard",
            30, 100,
            lambda x: None
        )

        cv2.createTrackbar(
            "Range %",
            "AutoMR Live Dashboard",
            50, 100,
            lambda x: None
        )

        cv2.createTrackbar(
            "Epsilon x1000",
            "AutoMR Live Dashboard",
            int(self.current_epsilon * 1000), 500,
            lambda x: None
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            self.update_controls()

            if self.pending_test and not self.testing:
                self.run_selected_benchmark(frame, frame_id)

            frame_id += 1

            tiles     = self.process_frame(frame, frame_id)
            grid      = self.build_dashboard(tiles)

            failure_rate = (
                (self.total_failures / self.total_tests) * 100
                if self.total_tests > 0 else 0
            )

            # ── Summary panel ─────────────────────────────────────────────────
            summary_panel = draw_summary_panel(
                280,                     # narrower — gives tiles more space
                grid.shape[0],
                self.total_tests,
                self.total_failures,
                failure_rate,
                progress=self.progress,
                total=self.total_progress
            )

            content = np.hstack([grid, summary_panel])

            # ── Header bar ────────────────────────────────────────────────────
            W = content.shape[1]
            header = np.full((HEADER_H, W, 3), T.BG_HEADER, dtype=np.uint8)
            _draw_header(
                header, W,
                frame_id,
                len(self.config.selected_mrs),
                self.current_epsilon,
                self.config.current_mr,
                self.total_tests,
                self.total_failures
            )

            # ── Benchmark progress strip (4 px, cyan) ─────────────────────────
            if self.testing and self.total_progress > 0:
                strip = np.full((4, W, 3), T.BG_DEEP, dtype=np.uint8)
                _bar(strip, 0, 0, W, 4,
                     self.progress, self.total_progress,
                     fill=T.ACCENT_CYAN, track=T.BG_PANEL)
                dashboard = np.vstack([header, strip, content])
            else:
                dashboard = np.vstack([header, content])

            cv2.imshow("AutoMR Live Dashboard", dashboard)

            if frame_id % self.frame_skip == 0:
                save_results_csv(self.results, self.output_dir)
                update_summary(self.results, self.output_dir)

            key = cv2.waitKey(1)

            if key == ord("r"):
                self.pending_test = True

            #self.handle_keys(key)

            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        save_results_csv(self.results, self.output_dir)
        update_summary(self.results, self.output_dir)