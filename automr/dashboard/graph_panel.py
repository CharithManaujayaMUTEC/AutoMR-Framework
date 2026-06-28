import cv2
import numpy as np


# ── Shared theme (must stay in sync with live_dashboard.py) ──────────────────

class _T:
    BG_DEEP      = (18,  20,  26)
    BG_PANEL     = (26,  30,  40)
    BG_CARD      = (32,  37,  52)
    ACCENT_CYAN  = (210, 220,  0)
    ACCENT_GREEN = (80,  210, 100)
    ACCENT_RED   = (60,   60, 230)
    ACCENT_AMBER = (30,  180, 230)
    TEXT_PRIMARY = (230, 232, 238)
    TEXT_SECONDARY = (130, 135, 155)
    TEXT_DIM     = (70,   75,  90)
    BORDER       = (45,   50,  68)
    SEPARATOR    = (40,   44,  60)


def _fr(img, x1, y1, x2, y2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _br(img, x1, y1, x2, y2, color, t=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)


def _txt(img, s, x, y, color=_T.TEXT_PRIMARY, scale=0.50, bold=False):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


def _hline(img, y, x1, x2, color=_T.SEPARATOR):
    cv2.line(img, (x1, y), (x2, y), color, 1, cv2.LINE_AA)


def _bar(img, x, y, w, h, value, max_value, fill, track=_T.BG_CARD):
    _fr(img, x, y, x + w, y + h, track)
    _br(img, x, y, x + w, y + h, _T.BORDER)
    if max_value > 0 and value > 0:
        fw = int(w * min(value / max_value, 1.0))
        if fw > 0:
            _fr(img, x, y, x + fw, y + h, fill)


# ── Stat card ─────────────────────────────────────────────────────────────────

def _stat_card(panel, x, y, w, h, label, value_str, value_color, bar_frac=None):
    """Draws a compact metric card with optional mini bar."""
    _fr(panel, x, y, x + w, y + h, _T.BG_CARD)
    _br(panel, x, y, x + w, y + h, _T.BORDER)

    # Left accent stripe
    cv2.rectangle(panel, (x, y), (x + 3, y + h), value_color, -1)

    # Label
    _txt(panel, label, x + 12, y + 18, _T.TEXT_SECONDARY, scale=0.38)

    # Value
    _txt(panel, value_str, x + 12, y + h - 14, value_color, scale=0.62, bold=True)

    # Mini bar
    if bar_frac is not None:
        bx, by = x + 12, y + h - 8
        bw, bh = w - 24, 3
        _fr(panel, bx, by, bx + bw, by + bh, _T.BORDER)
        fw = int(bw * min(bar_frac, 1.0))
        if fw > 0:
            _fr(panel, bx, by, bx + fw, by + bh, value_color)


# ── Main entry ────────────────────────────────────────────────────────────────

def draw_summary_panel(
    width,
    height,
    tests,
    failures,
    rate,
    progress=0,
    total=0
):
    panel = np.full((height, width, 3), _T.BG_PANEL, dtype=np.uint8)

    # Left border separator
    cv2.line(panel, (0, 0), (0, height), _T.BORDER, 2, cv2.LINE_AA)

    PAD    = 14
    CARDW  = width - PAD * 2
    FOOTER = 36          # reserved at bottom for hint
    BENCH  = 52          # reserved above footer for benchmark bar
    TOP    = 38          # reserved at top for section label

    # Available height for the 3 stat cards
    avail  = height - TOP - BENCH - FOOTER - 8
    CARDH  = max(54, avail // 3 - 6)   # never smaller than 54 px

    # ── Section label ─────────────────────────────────────────────────────────
    _txt(panel, "LIVE  METRICS", PAD, 24, _T.TEXT_DIM, scale=0.38)
    _hline(panel, 32, PAD, width - PAD, _T.BORDER)

    # ── Stat cards ────────────────────────────────────────────────────────────
    y = TOP

    _stat_card(
        panel, PAD, y, CARDW, CARDH,
        "TOTAL TESTS", str(tests),
        _T.ACCENT_CYAN
    )

    y += CARDH + 6

    fail_color = _T.ACCENT_RED if failures > 0 else _T.ACCENT_GREEN
    _stat_card(
        panel, PAD, y, CARDW, CARDH,
        "FAILURES", str(failures),
        fail_color
    )

    y += CARDH + 6

    rate_color = (
        _T.ACCENT_RED   if rate > 20 else
        _T.ACCENT_AMBER if rate > 5  else
        _T.ACCENT_GREEN
    )
    _stat_card(
        panel, PAD, y, CARDW, CARDH,
        "FAILURE RATE", f"{rate:.2f}%",
        rate_color,
        bar_frac=rate / 100.0
    )

    # ── Benchmark progress (pinned above footer) ──────────────────────────────
    bench_y = height - FOOTER - BENCH
    _hline(panel, bench_y, PAD, width - PAD, _T.BORDER)
    _txt(panel, "BENCHMARK", PAD, bench_y + 14, _T.TEXT_DIM, scale=0.36)

    if total > 0:
        pct = progress / total
        _bar(panel, PAD, bench_y + 20, CARDW, 7,
             progress, total, fill=_T.ACCENT_CYAN)
        _txt(panel, f"{progress}/{total}  {pct*100:.0f}%",
             PAD, bench_y + 42, _T.TEXT_SECONDARY, scale=0.38)
    else:
        _bar(panel, PAD, bench_y + 20, CARDW, 7, 0, 1, _T.BG_CARD)
        _txt(panel, "idle", PAD, bench_y + 42, _T.TEXT_DIM, scale=0.38)

    # ── Footer (pinned to bottom) ─────────────────────────────────────────────
    _hline(panel, height - FOOTER, PAD, width - PAD, _T.BORDER)
    _txt(panel, "R run   ESC quit",
         PAD, height - 14, _T.TEXT_DIM, scale=0.34)

    return panel