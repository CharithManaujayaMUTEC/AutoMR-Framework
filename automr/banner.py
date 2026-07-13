# ------------------------------------------------------------------
# AutoMR Startup Banner
# ------------------------------------------------------------------
from pathlib import Path
import tomllib
import sys


def _get_version() -> str:
    try:
        pyproject = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        )
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return "unknown"


def _get_engine_label() -> str:
    """Returns HPC or Standard based on what has been imported."""
    try:
        from automr.hpc import HighPerformanceAutoMR   # noqa: F401
        return "HPC + Standard"
    except Exception:
        return "Standard"


def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ANSI colour helpers — fall back to plain strings on non-TTY
def _c(code: str, text: str) -> str:
    if _supports_colour():
        return f"\033[{code}m{text}\033[0m"
    return text


CYAN    = lambda t: _c("96",   t)
GREEN   = lambda t: _c("92",   t)
YELLOW  = lambda t: _c("93",   t)
BLUE    = lambda t: _c("94",   t)
MAGENTA = lambda t: _c("95",   t)
DIM     = lambda t: _c("2",    t)
BOLD    = lambda t: _c("1",    t)
WHITE   = lambda t: _c("97",   t)


def print_banner() -> None:
    version = _get_version()
    engine  = _get_engine_label()

    # ── Logo block (pure ASCII, stays within 96 chars) ────────────────────────
    logo_lines = [
        r"   █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗██████╗ ",
        r"  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║██╔══██╗",
        r"  ███████║██║   ██║   ██║   ██║   ██║██╔████╔██║██████╔╝",
        r"  ██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██╔══██╗",
        r"  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║██║  ██║",
        r"  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝",
    ]

    # ── Feature columns ───────────────────────────────────────────────────────
    col_a = [
        "  Core",
        "  ✦ Model-agnostic",
        "  ✦ Input-agnostic",
        "  ✦ Output-agnostic",
        "  ✦ Plugin architecture",
        "  ✦ 16 metamorphic relations",
        "  ✦ 17 built-in transforms",
    ]
    col_b = [
        "  Performance",
        "  ✦ HPC parallel engine",
        "  ✦ Batch inference",
        "  ✦ Prediction caching",
        "  ✦ CPU/GPU backend",
        "  ✦ Multi-threaded workers",
        "  ✦ Prefetch data loading",
    ]
    col_c = [
        "  Analysis",
        "  ✦ Epsilon sensitivity",
        "  ✦ Auto epsilon recommend",
        "  ✦ Failure localization",
        "  ✦ Severity scoring",
        "  ✦ Worst-case detection",
        "  ✦ Live dashboard",
    ]

    # ── Colourize ─────────────────────────────────────────────────────────────
    def fmt_col(lines, hdr_fn, bullet_fn):
        out = []
        for i, line in enumerate(lines):
            if i == 0:
                out.append(BOLD(hdr_fn(line)))
            else:
                out.append(bullet_fn(line))
        return out

    col_a = fmt_col(col_a, CYAN,    WHITE)
    col_b = fmt_col(col_b, GREEN,   WHITE)
    col_c = fmt_col(col_c, MAGENTA, WHITE)

    # ── Box geometry ──────────────────────────────────────────────────────────
    W       = 96          # inner width (between ║ chars)
    FULL_W  = W + 2       # including ║ on both sides
    COL_W   = 30          # visible chars per feature column (no ANSI)

    def pad(s: str, width: int) -> str:
        """Left-pad a string to `width` visible characters (strips ANSI for counting)."""
        import re
        visible = re.sub(r"\033\[[0-9;]*m", "", s)
        return s + " " * max(0, width - len(visible))

    def box_row(content: str = "") -> str:
        """Wrap content in ║ … ║, padding to full width."""
        import re
        visible = re.sub(r"\033\[[0-9;]*m", "", content)
        padding = W - len(visible)
        return "║" + content + " " * max(0, padding) + "║"

    top    = "╔" + "═" * W + "╗"
    div    = "╠" + "═" * W + "╣"
    bot    = "╚" + "═" * W + "╝"
    empty  = box_row()

    # ── Build output ──────────────────────────────────────────────────────────
    lines = []
    lines.append(BLUE(top))
    lines.append(BLUE(empty))

    # Logo rows — centred within the box
    logo_w = max(len(l) for l in logo_lines)
    left_pad = (W - logo_w) // 2
    for logo_line in logo_lines:
        padded = " " * left_pad + logo_line
        lines.append(BLUE("║") + CYAN(padded.ljust(W)) + BLUE("║"))

    lines.append(BLUE(empty))

    # Tagline
    tag = "  Automated Metamorphic Testing Framework for Autonomous Moving Regression-Based AI/ML Models"
    lines.append(BLUE(box_row(WHITE(BOLD(tag)))))

    lines.append(BLUE(empty))
    lines.append(BLUE(DIM("╠" + "─" * W + "╣")))
    lines.append(BLUE(empty))

    # Feature columns
    for i in range(len(col_a)):
        a = pad(col_a[i], COL_W)
        b = pad(col_b[i], COL_W)
        c = pad(col_c[i], COL_W)
        row_content = "  " + a + "  " + b + "  " + c
        lines.append(BLUE(box_row(row_content)))

    lines.append(BLUE(empty))
    lines.append(BLUE(DIM("╠" + "─" * W + "╣")))
    lines.append(BLUE(empty))

    # Metadata row
    ver_label    = CYAN(BOLD(f"v{version}"))
    engine_label = GREEN(BOLD(engine))
    py_label     = DIM(f"Python {sys.version.split()[0]}")

    meta = f"  Version: {ver_label}   Engine: {engine_label}   {py_label}"
    lines.append(BLUE(box_row(meta)))

    lines.append(BLUE(empty))
    lines.append(BLUE(bot))
    lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    print_banner()