import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from .dashboard_utils import (
    create_output_dirs,
    calculate_percent_change,
    get_severity,
    evaluate_mr,
    save_violation_image,
    save_results_csv,
    update_summary,
)
from .control_panel import DashboardConfig
from .graph_panel import draw_sidebar
from .camera_source import CameraSource


# ═════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class T:
    BG_BASE    = ( 10,  12,  18)
    BG_PANEL   = ( 16,  19,  28)
    BG_CARD    = ( 22,  26,  38)
    BG_CTRL    = ( 19,  22,  34)   # control panel background
    BG_TILE    = ( 14,  16,  24)

    CYAN       = (220, 230,   0)
    GREEN      = ( 60, 200,  90)
    RED        = ( 48,  52, 220)
    AMBER      = ( 20, 160, 215)
    PURPLE     = (200,  90, 160)
    BLUE_SOFT  = (180, 140,  60)

    TXT_HI     = (235, 237, 245)
    TXT_MID    = (140, 145, 168)
    TXT_LO     = ( 55,  60,  80)

    BORDER     = ( 38,  44,  64)
    DIVIDER    = ( 28,  32,  48)
    GLOW_CYAN  = (180, 200,   0)
    TRACK_BG   = ( 28,  32,  46)   # slider track background
    TRACK_FILL = ( 45,  50,  72)   # inactive slider fill
    THUMB      = ( 80,  85, 110)   # slider thumb base
    THUMB_ACT  = (220, 230,   0)   # active thumb (cyan)

    HEADER_H   = 60
    STATUS_H   = 28
    SIDEBAR_W  = 340
    CTRL_H     = 340               # height of custom control panel
    CELL_W     = 520               # ↑ from 320 — sharper tiles
    CELL_H     = 390               # ↑ from 240 — sharper tiles


# ═════════════════════════════════════════════════════════════════════════════
#  PRIMITIVE DRAWING
# ═════════════════════════════════════════════════════════════════════════════

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONTS = cv2.FONT_HERSHEY_SIMPLEX


def _fr(img, x1, y1, x2, y2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _br(img, x1, y1, x2, y2, color, t=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)


def _line(img, x1, y1, x2, y2, color, t=1):
    cv2.line(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)


def _hline(img, y, x1, x2, color=T.DIVIDER, t=1):
    cv2.line(img, (x1, y), (x2, y), color, t, cv2.LINE_AA)


def _vline(img, x, y1, y2, color=T.DIVIDER, t=1):
    cv2.line(img, (x, y1), (x, y2), color, t, cv2.LINE_AA)


def _txt(img, s, x, y, color=T.TXT_HI, scale=0.44, thick=1):
    cv2.putText(img, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _txt_bold(img, s, x, y, color=T.TXT_HI, scale=0.50):
    cv2.putText(img, s, (x, y), _FONT, scale, color, 2, cv2.LINE_AA)


def _tw(s, scale=0.44, thick=1):
    (w, _), _ = cv2.getTextSize(s, _FONT, scale, thick)
    return w


def _pill(img, label, x, y, bg, fg=T.BG_BASE, scale=0.34, pad=4):
    (tw, th), _ = cv2.getTextSize(label, _FONT, scale, 1)
    x2 = x + tw + pad * 2
    y1 = y - th - pad + 1
    y2 = y + pad
    _fr(img, x, y1, x2, y2, bg)
    cv2.putText(img, label, (x + pad, y), _FONT, scale, fg, 1, cv2.LINE_AA)
    return x2 + 6


def _fill_bar(img, x, y, w, h, frac, fill, track=T.BG_CARD):
    _fr(img, x, y, x + w, y + h, track)
    _br(img, x, y, x + w, y + h, T.BORDER)
    if frac > 0:
        fw = max(1, int(w * min(frac, 1.0)))
        _fr(img, x, y, x + fw, y + h, fill)


def _corner_marks(img, x1, y1, x2, y2, color, length=14, t=1):
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        _line(img, cx, cy, cx + dx * length, cy, color, t)
        _line(img, cx, cy, cx, cy + dy * length, color, t)


def _glow_rect(img, x1, y1, x2, y2, color, t=1):
    _br(img, x1, y1, x2, y2, color, t)
    inner = tuple(int(c * 0.22) for c in color)
    _br(img, x1+1, y1+1, x2-1, y2-1, inner, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  LOGO LOADER
# ═════════════════════════════════════════════════════════════════════════════

_LOGO_CACHE: dict = {}

def _load_logo(target_h: int = 40) -> "np.ndarray | None":
    if target_h in _LOGO_CACHE:
        return _LOGO_CACHE[target_h]
    candidates = [
        Path(__file__).parent / "automrlogo.png",
        Path(__file__).parent.parent / "automrlogo.png",
        Path("automrlogo.png"),
        Path(r"D:\FYP 78SEm\AutoMR\AutoMR-Framework\automr\dashboard\automrlogo.png"),
    ]
    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(w * target_h / h), target_h),
                                 interpolation=cv2.INTER_AREA)
                if img.ndim == 3 and img.shape[2] == 4:
                    alpha = img[:, :, 3:] / 255.0
                    bgr   = img[:, :, :3].astype(float)
                    bg    = np.full_like(bgr, T.BG_PANEL, dtype=float)
                    img   = (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)
                _LOGO_CACHE[target_h] = img
                return img
    _LOGO_CACHE[target_h] = None
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  CUSTOM CONTROL PANEL  (replaces OpenCV trackbars)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Drawn into the lower section of the sidebar.
#  Mouse events on the main window are forwarded here via _ControlPanel.on_mouse().
#
#  Layout (inside the sidebar, below the MR list):
#
#  ┌──────────────────────────────────────┐
#  │  CONTROLS                            │
#  │  MR  ◄ BRIGHTNESS ►  [toggle]        │
#  │  ────────────────────────────────    │
#  │  Intensity   ████░░░░░░  50%         │
#  │  Range       ██████░░░░  50%         │
#  │  Tests       ██░░░░░░░░   5          │
#  │  FrameSkip   ██████░░░░  30          │
#  │  Epsilon     ████░░░░░░  0.050       │
#  │  ────────────────────────────────    │
#  │  [ ENABLE ]  [ RUN BENCH ]           │
#  └──────────────────────────────────────┘

class _Slider:
    """Single horizontal slider — fully custom drawn."""
    def __init__(self, label, vmin, vmax, value, color=T.CYAN, fmt=None):
        self.label  = label
        self.vmin   = vmin
        self.vmax   = vmax
        self.value  = value
        self.color  = color
        self.fmt    = fmt          # callable: value -> str
        self.rect   = (0,0,0,0)   # (x,y,w,h) of the track, set on draw
        self.active = False

    @property
    def frac(self):
        span = self.vmax - self.vmin
        return (self.value - self.vmin) / span if span else 0

    def set_frac(self, f):
        f = max(0.0, min(1.0, f))
        self.value = self.vmin + f * (self.vmax - self.vmin)

    def val_str(self):
        if self.fmt:
            return self.fmt(self.value)
        if isinstance(self.vmax, float) or isinstance(self.vmin, float):
            return f"{self.value:.3f}"
        return str(int(round(self.value)))

    def draw(self, canvas, x, y, w, label_w=90, h=8):
        """Draw onto canvas. Returns bottom y used."""
        # Label
        lc = self.color if self.active else T.TXT_MID
        _txt(canvas, self.label, x, y + 12, lc, scale=0.36)

        # Track
        tx = x + label_w
        tw = w - label_w - 54
        ty = y + 5
        _fr(canvas, tx, ty, tx + tw, ty + h, T.TRACK_BG)
        _br(canvas, tx, ty, tx + tw, ty + h, T.BORDER)

        # Fill
        fw = int(tw * self.frac)
        if fw > 0:
            fill = self.color if self.active else T.TRACK_FILL
            _fr(canvas, tx, ty, tx + fw, ty + h, fill)

        # Thumb
        thumb_x = tx + fw
        thumb_c = self.color if self.active else T.THUMB
        cv2.circle(canvas, (thumb_x, ty + h // 2), 7, T.BG_BASE, -1, cv2.LINE_AA)
        cv2.circle(canvas, (thumb_x, ty + h // 2), 5, thumb_c, -1, cv2.LINE_AA)

        # Value text
        vs = self.val_str()
        _txt_bold(canvas, vs, tx + tw + 8, y + 13, self.color, scale=0.38)

        self.rect = (tx, ty - 4, tw, h + 8)
        return y + 26

    def hit(self, mx, my, oy):
        """Check if (mx, my) is on this slider's track (oy = y-offset of panel)."""
        tx, ty, tw, th = self.rect
        return tx <= mx <= tx + tw and ty + oy <= my <= ty + oy + th

    def update_from_mouse(self, mx, oy):
        tx, _, tw, _ = self.rect
        f = (mx - tx) / tw
        self.set_frac(f)


class _ControlPanel:
    """
    Custom control panel drawn in the lower portion of the sidebar.
    Handles its own mouse interactions via on_mouse().
    """

    def __init__(self, mrs, n_mrs, epsilon_init=0.05):
        self.mrs    = mrs      # list of MR name strings
        self.n_mrs  = n_mrs
        self.mr_idx = 0
        self.enabled = True    # current MR enabled toggle

        self._active_slider = None   # which slider is being dragged

        # ── Sliders ───────────────────────────────────────────────────────────
        self.s_intensity  = _Slider("Intensity",  0,   100,  50,  T.CYAN,
                                    fmt=lambda v: f"{int(v)}%")
        self.s_range      = _Slider("Range",      0,   100,  50,  T.PURPLE,
                                    fmt=lambda v: f"{int(v)}%")
        self.s_tests      = _Slider("Tests",      1,   100,  5,   T.GREEN,
                                    fmt=lambda v: str(int(v)))
        self.s_frameskip  = _Slider("FrameSkip",  1,   120,  30,  T.AMBER,
                                    fmt=lambda v: str(int(v)))
        self.s_epsilon    = _Slider("Epsilon",    0.001, 0.5, epsilon_init, T.BLUE_SOFT,
                                    fmt=lambda v: f"{v:.3f}")

        self._sliders = [
            self.s_intensity,
            self.s_range,
            self.s_tests,
            self.s_frameskip,
            self.s_epsilon,
        ]

        # Bounding rects for buttons (set on draw)
        self._btn_toggle = (0, 0, 0, 0)
        self._btn_bench  = (0, 0, 0, 0)

        # Panel origin within the full window (set externally before on_mouse)
        self.origin_x = 0
        self.origin_y = 0

    @property
    def current_mr(self):
        if self.mrs:
            return self.mrs[self.mr_idx % len(self.mrs)]
        return "none"

    def prev_mr(self): self.mr_idx = (self.mr_idx - 1) % max(len(self.mrs), 1)
    def next_mr(self): self.mr_idx = (self.mr_idx + 1) % max(len(self.mrs), 1)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, canvas, x, y, w, h):
        """
        Draw the full control panel into canvas at (x, y) with size (w, h).
        Returns nothing; modifies canvas in-place.
        """
        PAD = 12

        # Panel background
        _fr(canvas, x, y, x + w, y + h, T.BG_CTRL)
        _hline(canvas, y, x, x + w, T.BORDER, t=1)
        # Top accent
        _line(canvas, x, y, x + w, y, T.CYAN, 2)

        cy = y + 10

        # ── Section header ────────────────────────────────────────────────────
        _txt(canvas, "CONTROLS", x + PAD, cy + 10, T.TXT_LO, scale=0.34)
        cy += 18
        _hline(canvas, cy, x + PAD, x + w - PAD, T.BORDER)
        cy += 10

        # ── MR selector ───────────────────────────────────────────────────────
        _txt(canvas, "MR", x + PAD, cy + 13, T.TXT_MID, scale=0.36)

        # ◄ arrow button
        arr_w = 20
        ax = x + PAD + 40
        _fr(canvas, ax, cy, ax + arr_w, cy + 20, T.BG_CARD)
        _br(canvas, ax, cy, ax + arr_w, cy + 20, T.BORDER)
        _txt(canvas, "<", ax + 5, cy + 14, T.TXT_MID, scale=0.38)

        # MR name chip
        chip_x = ax + arr_w + 4
        chip_w = w - PAD * 2 - 40 - arr_w * 2 - 12
        mr_name = self.current_mr.upper()[:14]
        chip_c = T.CYAN if self.enabled else T.TXT_LO
        _fr(canvas, chip_x, cy, chip_x + chip_w, cy + 20, T.BG_CARD)
        _br(canvas, chip_x, cy, chip_x + chip_w, cy + 20, chip_c)
        nw = _tw(mr_name, scale=0.38)
        _txt_bold(canvas, mr_name,
                  chip_x + (chip_w - nw) // 2, cy + 14,
                  chip_c, scale=0.38)

        # ► arrow button
        bx = chip_x + chip_w + 4
        _fr(canvas, bx, cy, bx + arr_w, cy + 20, T.BG_CARD)
        _br(canvas, bx, cy, bx + arr_w, cy + 20, T.BORDER)
        _txt(canvas, ">", bx + 5, cy + 14, T.TXT_MID, scale=0.38)

        # Store arrow hit areas (canvas-absolute)
        self._arr_left  = (ax, y + (cy - y), arr_w, 20)
        self._arr_right = (bx, y + (cy - y), arr_w, 20)

        cy += 28
        _hline(canvas, cy, x + PAD, x + w - PAD, T.DIVIDER)
        cy += 8

        # ── Sliders ───────────────────────────────────────────────────────────
        IW = w - PAD * 2
        for sl in self._sliders:
            cy = sl.draw(canvas, x + PAD, cy, IW, label_w=74, h=7)
            cy += 2

        cy += 6
        _hline(canvas, cy, x + PAD, x + w - PAD, T.DIVIDER)
        cy += 10

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_w = (w - PAD * 2 - 8) // 2
        # Enable / Disable toggle
        tog_c  = T.GREEN if self.enabled else T.TXT_LO
        tog_bg = T.BG_CARD
        _fr(canvas, x + PAD, cy, x + PAD + btn_w, cy + 24, tog_bg)
        _br(canvas, x + PAD, cy, x + PAD + btn_w, cy + 24, tog_c)
        tog_lbl = "ENABLED" if self.enabled else "DISABLED"
        tw2 = _tw(tog_lbl, scale=0.38, thick=2)
        _txt_bold(canvas, tog_lbl,
                  x + PAD + (btn_w - tw2) // 2, cy + 16,
                  tog_c, scale=0.38)
        self._btn_toggle = (x + PAD, cy, btn_w, 24)

        # Run benchmark button
        bx2 = x + PAD + btn_w + 8
        _fr(canvas, bx2, cy, bx2 + btn_w, cy + 24, T.BG_CARD)
        _br(canvas, bx2, cy, bx2 + btn_w, cy + 24, T.AMBER)
        bl = "RUN  BENCH"
        bw2 = _tw(bl, scale=0.38, thick=2)
        _txt_bold(canvas, bl, bx2 + (btn_w - bw2) // 2, cy + 16, T.AMBER, scale=0.38)
        self._btn_bench = (bx2, cy, btn_w, 24)

    # ── Mouse handling ────────────────────────────────────────────────────────

    def on_mouse(self, event, mx, my, flags):
        """
        Call from the OpenCV mouse callback.
        mx, my are window-absolute coordinates.
        Returns "bench" if run-bench was clicked, else None.
        """
        # Translate to panel-local coordinates
        lx = mx - self.origin_x
        ly = my - self.origin_y

        if event == cv2.EVENT_LBUTTONDOWN:
            # Arrow buttons (MR nav)
            ax, ay, aw, ah = self._arr_left
            if ax <= lx <= ax + aw and ay <= ly <= ay + ah:
                self.prev_mr()
                return

            ax, ay, aw, ah = self._arr_right
            if ax <= lx <= ax + aw and ay <= ly <= ay + ah:
                self.next_mr()
                return

            # Toggle button
            bx, by, bw, bh = self._btn_toggle
            if bx <= lx <= bx + bw and by <= ly <= by + bh:
                self.enabled = not self.enabled
                return

            # Bench button
            bx, by, bw, bh = self._btn_bench
            if bx <= lx <= bx + bw and by <= ly <= by + bh:
                return "bench"

            # Sliders
            for sl in self._sliders:
                tx, ty, tw, th = sl.rect
                if tx <= lx <= tx + tw and ty <= ly <= ty + th:
                    sl.active = True
                    sl.update_from_mouse(lx, 0)
                    self._active_slider = sl
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._active_slider and (flags & cv2.EVENT_FLAG_LBUTTON):
                self._active_slider.update_from_mouse(lx, 0)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._active_slider:
                self._active_slider.active = False
                self._active_slider = None

    # ── Value accessors (called by update_controls) ───────────────────────────

    @property
    def intensity(self):  return int(self.s_intensity.value)
    @property
    def range_pct(self):  return int(self.s_range.value)
    @property
    def tests(self):      return max(1, int(round(self.s_tests.value)))
    @property
    def frame_skip(self): return max(1, int(round(self.s_frameskip.value)))
    @property
    def epsilon(self):    return float(self.s_epsilon.value)


# ═════════════════════════════════════════════════════════════════════════════
#  HOLD OVERLAY
# ═════════════════════════════════════════════════════════════════════════════

def _hold_overlay(tile, ttl, ttl_max):
    h, w = tile.shape[:2]
    BW, BH, PAD = 80, 22, 6
    bx = w - BW - PAD
    by = PAD
    _fr(tile, bx, by, bx + BW, by + BH, T.BG_BASE, alpha=0.85)
    _br(tile, bx, by, bx + BW, by + BH, T.AMBER)
    _txt(tile, "HOLD", bx + 5, by + 15, T.AMBER, scale=0.38)
    frac_str = f"{ttl}/{ttl_max}"
    fw2 = _tw(frac_str, scale=0.30)
    _txt(tile, frac_str, bx + BW - fw2 - 4, by + 15, T.TXT_LO, scale=0.30)
    bar_y = by + BH + 2
    frac  = ttl / ttl_max if ttl_max > 0 else 0
    bar_c = T.AMBER if frac > 0.4 else T.TXT_LO
    _fill_bar(tile, bx, bar_y, BW, 3, frac, bar_c, T.BG_CARD)


# ═════════════════════════════════════════════════════════════════════════════
#  VIDEO TILE OVERLAY
# ═════════════════════════════════════════════════════════════════════════════

def _style_tile(tile, label, pred_value, is_original=False, status=None):
    h, w = tile.shape[:2]
    BAR_H = 44

    accent = (
        T.GREEN if is_original else
        T.RED   if status == "FAIL" else
        T.CYAN
    )

    _fr(tile, 0, h - BAR_H, w, h, T.BG_BASE, alpha=0.82)
    _line(tile, 0, h - BAR_H, w, h - BAR_H, accent, 2)

    role = "ORIGINAL" if is_original else label.upper()[:12]
    _pill(tile, role, 8, h - BAR_H + 22, accent, T.BG_BASE, scale=0.38)

    if not is_original and status is not None:
        dot_c  = T.GREEN if status == "PASS" else T.RED
        dot_x  = w - 68
        cv2.circle(tile, (dot_x, h - BAR_H + 14), 5, dot_c, -1, cv2.LINE_AA)
        _txt(tile, status, dot_x + 12, h - BAR_H + 18, dot_c, scale=0.36)

    pred_str = f"{pred_value:+.4f}"
    px = w - _tw(pred_str, scale=0.58, thick=2) - 10
    _txt_bold(tile, pred_str, px, h - 12, T.TXT_HI, scale=0.58)

    _corner_marks(tile, 0, 0, w - 1, h - 1, accent, length=18, t=1)
    return tile


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER BAR
# ═════════════════════════════════════════════════════════════════════════════

def _draw_header(canvas, width, frame_id, active_count,
                 epsilon, current_mr, total_tests, total_failures,
                 intensity_pct=50):
    H = T.HEADER_H
    _fr(canvas, 0, 0, width, H, T.BG_PANEL)
    _hline(canvas, H - 1, 0, width, T.BORDER)
    _line(canvas, 0, 0, width, 0, T.CYAN, 2)

    logo = _load_logo(target_h=42)
    lw = 0
    if logo is not None:
        lh, lw = logo.shape[:2]
        y0 = (H - lh) // 2
        canvas[y0: y0 + lh, 8: 8 + lw] = logo

    bx = 8 + lw + (10 if lw else 0)
    _txt_bold(canvas, "AutoMR", bx, 36, T.CYAN, scale=0.75)
    _txt(canvas, "LIVE DASHBOARD", bx + 80, 26, T.TXT_LO, scale=0.32)
    _txt(canvas, "Metamorphic Testing", bx + 80, 40, T.TXT_LO, scale=0.30)

    div_x = bx + 210
    _vline(canvas, div_x, 8, H - 8, T.BORDER)

    x = div_x + 14
    x = _pill(canvas, f"MR: {current_mr.upper()[:14]}", x, 34, T.BG_CARD, T.CYAN,    scale=0.38)
    x = _pill(canvas, f"ON: {active_count}",             x, 34, T.BG_CARD, T.GREEN,   scale=0.38)
    x = _pill(canvas, f"eps:{epsilon:.3f}",              x, 34, T.BG_CARD, T.AMBER,   scale=0.38)
    x = _pill(canvas, f"INT:{intensity_pct}%",           x, 34, T.BG_CARD, T.PURPLE,  scale=0.38)
    x = _pill(canvas, f"F:{frame_id}",                   x, 34, T.BG_CARD, T.TXT_MID, scale=0.34)

    fail_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0.0
    rc = T.RED if fail_rate > 20 else (T.AMBER if fail_rate > 5 else T.GREEN)
    stats = [
        (f"Tests:{total_tests}",    T.TXT_MID),
        (f"Fails:{total_failures}", T.RED if total_failures > 0 else T.TXT_MID),
        (f"Rate:{fail_rate:.1f}%",  rc),
    ]
    rx = width - 14
    for s, col in reversed(stats):
        rx -= _tw(s, scale=0.44, thick=2) + 18
        _txt_bold(canvas, s, rx, 34, col, scale=0.44)

    _txt(canvas,
         "1-9/V/D toggle MRs      R benchmark      ESC quit",
         div_x + 14, H - 8, T.TXT_LO, scale=0.30)


# ═════════════════════════════════════════════════════════════════════════════
#  STATUS BAR
# ═════════════════════════════════════════════════════════════════════════════

def _draw_statusbar(canvas, width, current_mr, status_text,
                    total_tests, total_failures, is_testing=False):
    H  = T.STATUS_H
    h0 = canvas.shape[0] - H
    _fr(canvas, 0, h0, width, h0 + H, T.BG_PANEL)
    _hline(canvas, h0, 0, width, T.BORDER)

    _txt(canvas, f"MR: {current_mr.upper()}", 12, h0 + 18, T.CYAN, scale=0.36)
    _vline(canvas, 130, h0 + 5, h0 + H - 5, T.BORDER)

    state_c = T.AMBER if is_testing else T.GREEN
    state_s = "BENCHMARKING..." if is_testing else "LIVE"
    _txt(canvas, state_s, 140, h0 + 18, state_c, scale=0.36)

    sw = _tw(status_text, scale=0.34)
    _txt(canvas, status_text, (width - sw) // 2, h0 + 18, T.TXT_MID, scale=0.34)

    pf = f"Pass: {total_tests - total_failures}   Fail: {total_failures}"
    pw = _tw(pf, scale=0.34)
    _txt(canvas, pf, width - pw - 12, h0 + 18, T.TXT_MID, scale=0.34)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  (metrics + MR list + custom control panel)
# ═════════════════════════════════════════════════════════════════════════════

def _draw_sidebar(panel, width, height, tests, failures, rate,
                  current_mr, selected_mrs, mr_cache,
                  ctrl_panel: "_ControlPanel",
                  progress=0, total=0):
    _fr(panel, 0, 0, width, height, T.BG_PANEL)
    _vline(panel, width - 1, 0, height, T.BORDER)

    PAD   = 12
    CW    = width - PAD * 2
    y     = 0

    # ── Metrics header ────────────────────────────────────────────────────────
    _fr(panel, 0, 0, width, 28, T.BG_CARD)
    _hline(panel, 28, 0, width, T.BORDER)
    _txt(panel, "LIVE  METRICS", PAD, 19, T.TXT_LO, scale=0.34)
    y = 34

    # ── Metric cards ──────────────────────────────────────────────────────────
    def _mcard(px, py, pw, ph, label, val_str, accent, bar_frac=None):
        _fr(panel, px, py, px + pw, py + ph, T.BG_CARD)
        _glow_rect(panel, px, py, px + pw, py + ph, T.BORDER)
        _fr(panel, px, py, px + 3, py + ph, accent)
        _txt(panel, label, px + 10, py + 15, T.TXT_LO, scale=0.32)
        _txt_bold(panel, val_str, px + 10, py + ph - 10, accent, scale=0.56)
        if bar_frac is not None:
            bx2, by2 = px + 10, py + ph - 5
            bw2 = pw - 20
            _fill_bar(panel, bx2, by2, bw2, 3, bar_frac, accent, T.BORDER)

    CARDH = 54
    _mcard(PAD, y, CW, CARDH, "TOTAL TESTS", str(tests), T.CYAN)
    y += CARDH + 5
    fc = T.RED if failures > 0 else T.GREEN
    _mcard(PAD, y, CW, CARDH, "FAILURES", str(failures), fc)
    y += CARDH + 5
    rc = T.RED if rate > 20 else (T.AMBER if rate > 5 else T.GREEN)
    _mcard(PAD, y, CW, CARDH, "FAILURE RATE", f"{rate:.2f}%", rc,
           bar_frac=rate / 100.0)
    y += CARDH + 10

    # ── Active MR list ────────────────────────────────────────────────────────
    _hline(panel, y, PAD, width - PAD, T.BORDER)
    y += 6
    _txt(panel, "ACTIVE  RELATIONS", PAD, y + 10, T.TXT_LO, scale=0.32)
    y += 20

    ROW_H = 28
    for mr in selected_mrs:
        is_cur  = (mr == current_mr)
        entry   = mr_cache.get(mr)
        row_bg  = T.BG_BASE if is_cur else T.BG_CARD
        _fr(panel, PAD, y, PAD + CW, y + ROW_H, row_bg)
        if is_cur:
            _fr(panel, PAD, y, PAD + 3, y + ROW_H, T.CYAN)
            _glow_rect(panel, PAD, y, PAD + CW, y + ROW_H, T.GLOW_CYAN)
        else:
            _br(panel, PAD, y, PAD + CW, y + ROW_H, T.BORDER)
        name_c = T.CYAN if is_cur else T.TXT_MID
        _txt(panel, mr.upper()[:16], PAD + 8, y + 18, name_c, scale=0.34)
        if entry is not None:
            dot_c = T.GREEN if entry["status"] == "PASS" else T.RED
            cv2.circle(panel, (PAD + CW - 16, y + ROW_H // 2), 5, dot_c, -1, cv2.LINE_AA)
            ttl_f = entry["ttl"] / entry["ttl_max"] if entry["ttl_max"] > 0 else 0
            _fill_bar(panel, PAD + 8, y + ROW_H - 5, CW - 26, 3, ttl_f, T.AMBER, T.BORDER)
        else:
            cv2.circle(panel, (PAD + CW - 16, y + ROW_H // 2), 5, T.TXT_LO, -1, cv2.LINE_AA)
        y += ROW_H + 2
        if y > height - T.CTRL_H - 20:
            break

    # ── Custom control panel (pinned to bottom of sidebar) ────────────────────
    ctrl_y = height - T.CTRL_H
    ctrl_panel.draw(panel, 0, ctrl_y, width, T.CTRL_H)
    # Store the panel's origin so mouse events can be translated
    ctrl_panel.origin_y = ctrl_y
    ctrl_panel.origin_x = 0


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS  —  operational code completely unchanged
# ═════════════════════════════════════════════════════════════════════════════

class LiveDashboard:

    def __init__(
        self,
        automr,
        model,
        frame_skip=30,
        save_results=True,
        save_violations=True,
        output_dir="results/live_dashboard",
    ):
        self.pending_test   = False
        self.testing        = False
        self.progress       = 0
        self.total_progress = 0

        self.automr = automr
        self.model  = model

        self.config = DashboardConfig(automr)

        self.frame_skip      = frame_skip
        self.save_results    = save_results
        self.save_violations = save_violations
        self.output_dir      = output_dir

        self.results        = []
        self.total_tests    = 0
        self.total_failures = 0

        self.CELL_W = T.CELL_W
        self.CELL_H = T.CELL_H
        self.current_epsilon = 0.05

        self.HOLD_FRAMES  = 90
        self._tile_cache  = {}
        self.live_parameter = 0.5
        self._status_msg  = "Ready"

        create_output_dirs(self.output_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_test_parameters(self, mr_name):
        s = self.config.mr_ranges[mr_name]
        return np.linspace(s["start"], s["end"], s["tests"])

    def update_controls(self, ctrl: _ControlPanel):
        """Read values from the custom control panel instead of trackbars."""
        mrs = [mr for mr in self.automr.list_transforms() if mr != "temporal"]

        # Sync MR selection
        if mrs:
            self.config.current_mr = ctrl.current_mr
            if ctrl.enabled:
                if ctrl.current_mr not in self.config.selected_mrs:
                    self.config.selected_mrs.append(ctrl.current_mr)
            else:
                if ctrl.current_mr in self.config.selected_mrs:
                    self.config.selected_mrs.remove(ctrl.current_mr)

        # Sync epsilon
        new_eps = ctrl.epsilon
        if abs(new_eps - self.current_epsilon) > 1e-6:
            self.current_epsilon = new_eps
            self.automr.set_epsilon(new_eps)

        # Sync frame skip
        self.frame_skip = ctrl.frame_skip
        self.config.live_intensity = ctrl.intensity

        # Sync range + tests for current MR
        current = self.config.current_mr

        if current in self.config.mr_ranges:

            cfg = self.automr.mr_ranges[current]

            start = cfg["start"]
            end = cfg["end"]

            scaled_end = start + (end - start) * ctrl.range_pct / 100.0

            self.live_parameter = (
                start +
                (scaled_end - start) * ctrl.intensity / 100.0
            )

            self.config.mr_ranges[current]["start"] = start
            self.config.mr_ranges[current]["end"] = scaled_end
            self.config.mr_ranges[current]["tests"] = ctrl.tests

    def handle_keys(self, key):
        mapping = {
            ord("1"): "brightness", ord("2"): "rotation",
            ord("3"): "translation", ord("4"): "noise",
            ord("5"): "blur",  ord("6"): "contrast",
            ord("7"): "rain",  ord("8"): "snow",
            ord("9"): "fog",   ord("v"): "visibility",
            ord("d"): "darkness",
        }
        if key in mapping:
            mr = mapping[key]
            self.config.current_mr = mr
            if mr in self.config.selected_mrs:
                self.config.selected_mrs.remove(mr)
            else:
                self.config.selected_mrs.append(mr)

    # ── Frame processing (unchanged) ──────────────────────────────────────────

    def process_frame(self, frame, frame_id):
        tiles = []
        original_pred = float(self.model.predict(frame))

        orig_tile = frame.copy()
        _style_tile(orig_tile, "original", original_pred, is_original=True)
        tiles.append(orig_tile)

        run_tests = (frame_id % self.frame_skip == 0)

        for mr_name in self.config.selected_mrs:
            if run_tests and mr_name == self.config.current_mr:
                try:
                    transform        = self.automr.transform_registry.get(mr_name)
                    param            = self.live_parameter
                    transformed      = transform(frame.copy(), float(param))
                    transformed_pred = float(self.model.predict(transformed))
                    diff, pct        = calculate_percent_change(original_pred, transformed_pred)
                    status           = evaluate_mr(self.automr, mr_name, original_pred, transformed_pred)
                    severity         = get_severity(diff)

                    self.total_tests += 1
                    if status == "FAIL":
                        self.total_failures += 1
                        self._status_msg = f"FAIL  {mr_name}  diff={diff:.4f}"
                        if self.save_violations:
                            save_violation_image(self.output_dir, mr_name, frame_id, transformed)
                    else:
                        self._status_msg = f"PASS  {mr_name}  diff={diff:.4f}"

                    tile = transformed.copy()
                    _style_tile(tile, mr_name, transformed_pred,
                                is_original=False, status=status)
                    self._tile_cache[mr_name] = {
                        "tile": tile, "status": status,
                        "pred": transformed_pred,
                        "ttl": self.HOLD_FRAMES, "ttl_max": self.HOLD_FRAMES,
                    }
                    self.results.append({
                        "timestamp": datetime.now(), "frame_id": frame_id,
                        "mr": mr_name, "parameter": float(param),
                        "original_prediction": original_pred,
                        "transformed_prediction": transformed_pred,
                        "difference": diff, "percent_change": pct,
                        "status": status, "severity": severity,
                        "epsilon": self.current_epsilon,
                    })
                except Exception as e:
                    print(f"{mr_name}: {e}")

            entry = self._tile_cache.get(mr_name)
            if entry is not None and entry["ttl"] > 0:
                display_tile = entry["tile"].copy()
                _hold_overlay(display_tile, entry["ttl"], entry["ttl_max"])
                tiles.append(display_tile)
                if not run_tests:
                    entry["ttl"] -= 1

        return tiles

    def run_selected_benchmark(self, frame, frame_id):
        self.testing = True
        for mr_name in self.config.selected_mrs:
            parameters = self.get_test_parameters(mr_name)
            self.total_progress += len(parameters)
            for param in parameters:
                self.progress += 1
        self.testing      = False
        self.pending_test = False

    # ── Grid builder ──────────────────────────────────────────────────────────

    def build_dashboard(self, tiles):
        resized = [cv2.resize(t, (self.CELL_W, self.CELL_H),
                              interpolation=cv2.INTER_LINEAR) for t in tiles]
        rows = []
        for i in range(0, len(resized), 3):
            row = resized[i:i + 3]
            while len(row) < 3:
                ph = np.full((self.CELL_H, self.CELL_W, 3), T.BG_TILE, dtype=np.uint8)
                _br(ph, 1, 1, self.CELL_W - 2, self.CELL_H - 2, T.BORDER)
                lbl = "NO  SIGNAL"
                lw  = _tw(lbl, scale=0.46)
                _txt(ph, lbl, (self.CELL_W - lw) // 2, self.CELL_H // 2 + 8,
                     T.TXT_LO, scale=0.46)
                cx, cy = self.CELL_W // 2, self.CELL_H // 2 - 24
                _line(ph, cx - 12, cy, cx + 12, cy, T.TXT_LO)
                _line(ph, cx, cy - 12, cx, cy + 12, T.TXT_LO)
                row.append(ph)
            rows.append(np.hstack(row))
        if not rows:
            return np.full((self.CELL_H, self.CELL_W, 3), T.BG_TILE, dtype=np.uint8)
        grid = rows[0]
        for row in rows[1:]:
            sep  = np.full((2, row.shape[1], 3), T.DIVIDER, dtype=np.uint8)
            grid = np.vstack([grid, sep, row])
        return grid

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, video_source=0):
        cap = CameraSource.open(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")

        frame_id = 0
        WIN = "AutoMR Live Dashboard"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        mrs = [mr for mr in self.automr.list_transforms() if mr != "temporal"]

        # ── Custom control panel ──────────────────────────────────────────────
        ctrl = _ControlPanel(mrs, len(mrs), epsilon_init=self.current_epsilon)

        # Mouse callback — forward to control panel
        _bench_flag = [False]

        def _mouse_cb(event, mx, my, flags, param):
            result = ctrl.on_mouse(event, mx, my, flags)
            if result == "bench":
                _bench_flag[0] = True

        cv2.setMouseCallback(WIN, _mouse_cb)

        # NOTE: We keep ONE invisible trackbar so OpenCV doesn't complain
        # about an empty window — it stays hidden behind the custom UI.
        cv2.createTrackbar("_", WIN, 0, 1, lambda x: None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.update_controls(ctrl)

            if _bench_flag[0] and not self.testing:
                self.run_selected_benchmark(frame, frame_id)
                _bench_flag[0] = False

            if self.pending_test and not self.testing:
                self.run_selected_benchmark(frame, frame_id)

            frame_id += 1

            tiles = self.process_frame(frame, frame_id)
            grid  = self.build_dashboard(tiles)

            failure_rate = (
                (self.total_failures / self.total_tests) * 100
                if self.total_tests > 0 else 0.0
            )

            # ── Sidebar ───────────────────────────────────────────────────────
            sb_h    = grid.shape[0]
            sidebar = np.full((sb_h, T.SIDEBAR_W, 3), T.BG_PANEL, dtype=np.uint8)
            _draw_sidebar(
                sidebar, T.SIDEBAR_W, sb_h,
                self.total_tests, self.total_failures, failure_rate,
                self.config.current_mr, self.config.selected_mrs,
                self._tile_cache, ctrl,
                progress=self.progress, total=self.total_progress,
            )

            content = np.hstack([sidebar, grid])
            W       = content.shape[1]

            # ── Header ────────────────────────────────────────────────────────
            header = np.full((T.HEADER_H, W, 3), T.BG_PANEL, dtype=np.uint8)
            _draw_header(
                header, W, frame_id,
                len(self.config.selected_mrs),
                self.current_epsilon,
                self.config.current_mr,
                self.total_tests, self.total_failures,
                intensity_pct=ctrl.intensity,
            )

            # Benchmark progress strip
            if self.testing and self.total_progress > 0:
                strip = np.full((4, W, 3), T.BG_BASE, dtype=np.uint8)
                _fill_bar(strip, 0, 0, W, 4,
                          self.progress / self.total_progress,
                          T.CYAN, T.BG_PANEL)
                body = np.vstack([header, strip, content])
            else:
                body = np.vstack([header, content])

            # ── Status bar ────────────────────────────────────────────────────
            total_h   = body.shape[0] + T.STATUS_H
            dashboard = np.full((total_h, W, 3), T.BG_BASE, dtype=np.uint8)
            dashboard[: body.shape[0]] = body
            _draw_statusbar(
                dashboard, W,
                self.config.current_mr, self._status_msg,
                self.total_tests, self.total_failures,
                is_testing=self.testing,
            )

            cv2.imshow(WIN, dashboard)

            if frame_id % self.frame_skip == 0:
                save_results_csv(self.results, self.output_dir)
                update_summary(self.results, self.output_dir)

            key = cv2.waitKey(1)
            if key == ord("r"):
                self.pending_test = True
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        save_results_csv(self.results, self.output_dir)
        update_summary(self.results, self.output_dir)