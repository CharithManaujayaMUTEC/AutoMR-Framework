"""
graph_panel.py
──────────────
Kept for backwards compatibility.
The sidebar rendering has moved into live_dashboard.py as _draw_sidebar().
draw_summary_panel() is preserved so external callers don't break.
"""

import cv2
import numpy as np


class _T:
    BG_BASE    = ( 10,  12,  18)
    BG_PANEL   = ( 16,  19,  28)
    BG_CARD    = ( 22,  26,  38)
    CYAN       = (220, 230,   0)
    GREEN      = ( 60, 200,  90)
    RED        = ( 48,  52, 220)
    AMBER      = ( 20, 160, 215)
    TXT_HI     = (235, 237, 245)
    TXT_MID    = (140, 145, 168)
    TXT_LO     = ( 55,  60,  80)
    BORDER     = ( 38,  44,  64)
    DIVIDER    = ( 28,  32,  48)


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fr(img, x1, y1, x2, y2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _br(img, x1, y1, x2, y2, color, t=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)


def _txt(img, s, x, y, color=_T.TXT_HI, scale=0.46, bold=False):
    cv2.putText(img, s, (x, y), _FONT, scale, color,
                2 if bold else 1, cv2.LINE_AA)


def _hline(img, y, x1, x2, color=_T.DIVIDER):
    cv2.line(img, (x1, y), (x2, y), color, 1, cv2.LINE_AA)


def _fill_bar(img, x, y, w, h, frac, fill, track=_T.BG_CARD):
    _fr(img, x, y, x + w, y + h, track)
    _br(img, x, y, x + w, y + h, _T.BORDER)
    if frac > 0:
        fw = max(1, int(w * min(frac, 1.0)))
        _fr(img, x, y, x + fw, y + h, fill)


def _stat_card(panel, x, y, w, h, label, value_str, accent, bar_frac=None):
    _fr(panel, x, y, x + w, y + h, _T.BG_CARD)
    _br(panel, x, y, x + w, y + h, _T.BORDER)
    cv2.rectangle(panel, (x, y), (x + 3, y + h), accent, -1)
    _txt(panel, label,     x + 10, y + 17, _T.TXT_LO,  scale=0.34)
    _txt(panel, value_str, x + 10, y + h - 12, accent, scale=0.60, bold=True)
    if bar_frac is not None:
        bx, by = x + 10, y + h - 6
        bw, bh = w - 20, 3
        _fr(panel, bx, by, bx + bw, by + bh, _T.BORDER)
        fw = int(bw * min(bar_frac, 1.0))
        if fw > 0:
            _fr(panel, bx, by, bx + fw, by + bh, accent)


# Public API — called by live_dashboard.py run()
def draw_sidebar(panel, width, height, tests, failures, rate,
                 current_mr, selected_mrs, mr_cache,
                 progress=0, total=0):
    """
    Thin shim — delegates to live_dashboard._draw_sidebar().
    Kept so __init__.py imports stay clean.
    """
    from .live_dashboard import _draw_sidebar
    _draw_sidebar(panel, width, height, tests, failures, rate,
                  current_mr, selected_mrs, mr_cache,
                  progress=progress, total=total)


# Legacy compat — external code that calls draw_summary_panel() still works
def draw_summary_panel(width, height, tests, failures, rate,
                       progress=0, total=0):
    panel = np.full((height, width, 3), _T.BG_PANEL, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height), _T.BORDER, 2, cv2.LINE_AA)

    PAD   = 12
    CW    = width - PAD * 2
    TOP   = 36
    CARDH = max(54, (height - TOP - 90) // 3 - 6)
    FOOT  = 36

    _txt(panel, "LIVE  METRICS", PAD, 22, _T.TXT_LO, scale=0.36)
    _hline(panel, 30, PAD, width - PAD, _T.BORDER)

    y = TOP
    _stat_card(panel, PAD, y, CW, CARDH, "TOTAL TESTS", str(tests), _T.CYAN)
    y += CARDH + 6
    fc = _T.RED if failures > 0 else _T.GREEN
    _stat_card(panel, PAD, y, CW, CARDH, "FAILURES", str(failures), fc)
    y += CARDH + 6
    rc = _T.RED if rate > 20 else (_T.AMBER if rate > 5 else _T.GREEN)
    _stat_card(panel, PAD, y, CW, CARDH, "FAILURE RATE", f"{rate:.2f}%",
               rc, bar_frac=rate / 100.0)

    bench_y = height - FOOT - 54
    _hline(panel, bench_y, PAD, width - PAD, _T.BORDER)
    _txt(panel, "BENCHMARK", PAD, bench_y + 14, _T.TXT_LO, scale=0.34)
    frac = (progress / total) if total > 0 else 0
    _fill_bar(panel, PAD, bench_y + 20, CW, 7, frac, _T.CYAN, _T.BG_BASE)
    label = f"{progress}/{total}  {frac*100:.0f}%" if total > 0 else "idle"
    _txt(panel, label, PAD, bench_y + 42, _T.TXT_MID, scale=0.36)

    _hline(panel, height - FOOT, PAD, width - PAD, _T.BORDER)
    _txt(panel, "R run   ESC quit", PAD, height - 12, _T.TXT_LO, scale=0.32)
    return panel