import argparse
from pathlib import Path

from automr import __version__


DOCS_DIR = Path(__file__).parent / "docs"


def print_banner():
    print("=" * 80)
    print(f"AutoMR Framework v{__version__}")
    print("Automated Metamorphic Testing Framework")
    print("for Regression-based Autonomous Driving AI/ML Models")
    print("=" * 80)


def show_info():
    print_banner()

    print("\nFramework Features")
    print("-" * 80)
    print("✓ Model Agnostic")
    print("✓ Input Agnostic")
    print("✓ Output Agnostic")
    print("✓ HPC Parallel Validation")
    print("✓ Prediction Caching")
    print("✓ Batch Inference")
    print("✓ Epsilon Sensitivity Analysis")
    print("✓ Automatic Epsilon Recommendation")
    print("✓ Failure Analysis")
    print("✓ Severity Analysis")
    print("✓ Worst Case Detection")
    print("✓ Range Analysis")
    print("✓ Graph Generation")
    print("✓ Report Generation")
    print("✓ Transformation Visualization")
    print("✓ Plugin Architecture")

    print("\nSupported Backends")
    print("-" * 80)
    print("• TensorFlow")
    print("• PyTorch")
    print("• ONNX Runtime")

    print("\nVersion")
    print("-" * 80)
    print(f"AutoMR : {__version__}")


def list_docs():
    print_banner()

    print("\nDocumentation")
    print("-" * 80)

    if not DOCS_DIR.exists():
        print("Documentation folder not found.")
        return

    files = sorted(DOCS_DIR.rglob("*"))

    count = 0

    for file in files:
        if file.is_file():
            count += 1
            rel = file.relative_to(DOCS_DIR)
            print(f"{count:02d}. {rel}")

    if count == 0:
        print("No documentation files found.")


def main():

    parser = argparse.ArgumentParser(
        prog="automr",
        description="Automated Metamorphic Testing Framework",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples
--------
automr --info
automr --version
automr --docs
"""
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show framework version"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Display framework information"
    )

    parser.add_argument(
        "--docs",
        action="store_true",
        help="List available documentation"
    )

    args = parser.parse_args()

    if args.version:
        print(f"AutoMR v{__version__}")
        return

    if args.info:
        show_info()
        return

    if args.docs:
        list_docs()
        return

    print_banner()
    parser.print_help()


if __name__ == "__main__":
    main()