import argparse
import sys
from pathlib import Path

from automr import __version__


ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"


# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------

def banner():
    print(
r"""
================================================================================
      █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗██████╗
     ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║██╔══██╗
     ███████║██║   ██║   ██║   ██║   ██║██╔████╔██║██████╔╝
     ██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██╔══██╗
     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║██║  ██║
     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝

        Automated Metamorphic Testing Framework
================================================================================
""")

# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def cmd_version(args):
    print(f"AutoMR Framework v{__version__}")


def cmd_info(args):

    banner()

    print(f"Version              : {__version__}")
    print("Framework            : AutoMR")
    print("Architecture         : Plugin-based")
    print("Validation           : Parallel")
    print()

    print("Features")
    print("-------------------------------------------")
    print("✓ Model Agnostic")
    print("✓ Input Agnostic")
    print("✓ Output Agnostic")
    print("✓ Regression Model Support")
    print("✓ TensorFlow")
    print("✓ PyTorch")
    print("✓ ONNX Runtime")
    print("✓ HPC Parallel Validation")
    print("✓ Prediction Caching")
    print("✓ Batch Inference")
    print("✓ Failure Analysis")
    print("✓ Severity Analysis")
    print("✓ Worst-case Detection")
    print("✓ Parameter Range Analysis")
    print("✓ Automatic Epsilon Search")
    print("✓ Report Generation")
    print("✓ Graph Generation")
    print("✓ Plugin Architecture")
    print("✓ Transformation Registry")
    print("✓ Relation Registry")


def cmd_docs(args):

    if not DOCS_DIR.exists():
        print("Documentation folder not found.")
        return

    if args.name:

        matches = []

        for f in DOCS_DIR.rglob("*"):

            if f.is_file():

                if args.name.lower() in f.stem.lower():
                    matches.append(f)

        if not matches:
            print("Documentation not found.")
            return

        doc = matches[0]

        print("=" * 80)
        print(doc.name)
        print("=" * 80)
        print(doc.read_text(encoding="utf8"))

        return

    banner()

    print("Available Documentation\n")

    files = sorted(DOCS_DIR.rglob("*"))

    for f in files:

        if f.is_file():
            print(f" • {f.relative_to(DOCS_DIR)}")

    print()
    print("Example:")
    print("   automr docs api")
    print("   automr docs transformations")


def cmd_examples(args):

    print("Example Commands\n")

    print("Run validation")
    print("  automr validate")

    print()

    print("Run benchmark")
    print("  automr benchmark")

    print()

    print("List documentation")
    print("  automr docs")

    print()

    print("Show API documentation")
    print("  automr docs api")


def cmd_license(args):

    print("""
MIT License

Copyright (c) AutoMR

Permission is hereby granted, free of charge,
to any person obtaining a copy...
""")


def cmd_citation(args):

    print("""
If you use AutoMR in your research please cite:

AutoMR Framework
Automated Metamorphic Testing Framework
for Regression-based AI/ML Models

Version: {}
""".format(__version__))


def cmd_list_mrs(args):

    try:
        from automr.relations.registry import relation_registry

        print("\nRegistered Metamorphic Relations\n")

        for i, name in enumerate(sorted(relation_registry.list()), 1):
            print(f"{i:2d}. {name}")

    except Exception:

        print("Unable to load relation registry.")


def cmd_list_transforms(args):

    try:
        from automr.transformations.registry import transformation_registry

        print("\nRegistered Transformations\n")

        for i, name in enumerate(sorted(transformation_registry.list()), 1):
            print(f"{i:2d}. {name}")

    except Exception:

        print("Unable to load transformation registry.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        prog="automr",
        description="Automated Metamorphic Testing Framework",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Show framework version").set_defaults(func=cmd_version)

    sub.add_parser("info", help="Framework information").set_defaults(func=cmd_info)

    p = sub.add_parser("docs", help="Browse documentation")
    p.add_argument("name", nargs="?", help="Documentation name")
    p.set_defaults(func=cmd_docs)

    sub.add_parser(
        "list-mrs",
        help="List registered metamorphic relations"
    ).set_defaults(func=cmd_list_mrs)

    sub.add_parser(
        "list-transforms",
        help="List registered transformations"
    ).set_defaults(func=cmd_list_transforms)

    sub.add_parser(
        "examples",
        help="Show example commands"
    ).set_defaults(func=cmd_examples)

    sub.add_parser(
        "license",
        help="Show license"
    ).set_defaults(func=cmd_license)

    sub.add_parser(
        "citation",
        help="Show citation information"
    ).set_defaults(func=cmd_citation)

    # Reserved for future implementation
    sub.add_parser("benchmark", help="Run benchmark")
    sub.add_parser("validate", help="Run validation")
    sub.add_parser("graphs", help="Generate graphs")
    sub.add_parser("report", help="Generate reports")

    args = parser.parse_args()

    if not hasattr(args, "func"):
        banner()
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()