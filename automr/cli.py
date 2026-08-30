import argparse
import sys
from pathlib import Path

from automr import __version__

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"


# =============================================================================
# Banner
# =============================================================================

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
          Autonomous Regression-based AI/ML Models
================================================================================
"""
    )


# =============================================================================
# Commands
# =============================================================================

def cmd_version(args):
    print(f"AutoMR Framework v{__version__}")


def cmd_info(args):

    banner()

    print(f"Version              : {__version__}")
    print("Framework            : AutoMR")
    print("Architecture         : Plugin-based")
    print("Execution            : Parallel")
    print("Documentation        : Built-in")
    print()

    print("Core Features")
    print("-" * 60)

    features = [
        "Model Agnostic",
        "Input Agnostic",
        "Output Agnostic",
        "Regression Model Support",
        "TensorFlow",
        "PyTorch",
        "Scikit-Learn",
        "XGBoost",
        "ONNX Runtime",
        "Remote Models",
        "Plugin Architecture",
        "Transformation Registry",
        "Relation Registry",
        "Automatic Epsilon Search",
        "Parallel Validation",
        "Prediction Caching",
        "Failure Analysis",
        "Severity Analysis",
        "Worst-case Detection",
        "Range Analysis",
        "Graph Generation",
        "Report Generation",
        "Live Dashboard",
        "HPC Engine",
    ]

    for feature in features:
        print(f"✓ {feature}")


def cmd_docs(args):

    if not DOCS_DIR.exists():
        print("Documentation folder not found.")
        return

    if args.name is None:

        banner()

        print("Documentation\n")

        print("User Guide")
        print("--------------------------------")
        print("  getting-started")
        print("  configuration")
        print("  running-tests")
        print("  transformations")
        print("  metamorphic-relations")
        print("  supported-models")
        print("  reports")
        print("  faq")
        print()

        print("Tutorials")
        print("--------------------------------")
        print("  regression")
        print("  image-classification")
        print("  object-detection")
        print("  lane-detection")
        print("  custom-plugin")
        print()

        print("Developer Guide")
        print("--------------------------------")
        print("  architecture")
        print("  project-structure")
        print("  registry-system")
        print("  model-wrappers")
        print("  input-handlers")
        print("  adding-transformations")
        print("  adding-relations")
        print("  extending-automr")
        print("  comparators")
        print("  epsilon-analysis")
        print("  hpc-engine")
        print("  report-generator")
        print("  contributing")
        print()

        print("API")
        print("--------------------------------")
        print("  automr")
        print("  registry")
        print("  relations")
        print("  transforms")
        print("  wrappers")
        print("  dashboard")
        print("  hpc")
        print("  utils")
        print()

        print("Examples")
        print("--------------------------------")
        print("automr docs architecture")
        print("automr docs regression")
        print("automr docs registry")
        print("automr docs automr")

        return

    matches = sorted(
        DOCS_DIR.rglob(f"*{args.name}*.md"),
        key=lambda p: len(str(p))
    )

    if not matches:
        print(f"No documentation found for '{args.name}'.")
        return

    doc = matches[0]

    print("=" * 80)
    print(doc.relative_to(DOCS_DIR))
    print("=" * 80)
    print(doc.read_text(encoding="utf8"))


def cmd_examples(args):

    banner()

    print("Example Commands\n")

    examples = [
        "automr info",
        "automr version",
        "automr docs",
        "automr docs architecture",
        "automr docs regression",
        "automr docs automr",
        "automr list-mrs",
        "automr list-transforms",
        "automr benchmark",
        "automr validate",
        "automr report",
        "automr graphs",
    ]

    for cmd in examples:
        print(f"  {cmd}")


def cmd_license(args):

    print("""
MIT License

Copyright (c) AutoMR

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files to deal in
the Software without restriction.
""")


def cmd_citation(args):

    print(f"""
If you use AutoMR in your research, please cite:

AutoMR Framework
Automated Metamorphic Testing Framework
for Autonomous Regression-based AI/ML Models

Version: {__version__}
""")


def cmd_list_mrs(args):

    try:

        from automr.registry.relation_registry import relation_registry

        print("\nRegistered Metamorphic Relations\n")

        relations = sorted(relation_registry.list())

        for i, relation in enumerate(relations, start=1):
            print(f"{i:2d}. {relation}")

        print(f"\nTotal: {len(relations)}")

    except Exception as e:
        print(f"Unable to load relation registry.\n{e}")


def cmd_list_transforms(args):

    try:

        from automr.registry.transformation_registry import transformation_registry

        print("\nRegistered Transformations\n")

        transforms = sorted(transformation_registry.list())

        for i, transform in enumerate(transforms, start=1):
            print(f"{i:2d}. {transform}")

        print(f"\nTotal: {len(transforms)}")

    except Exception as e:
        print(f"Unable to load transformation registry.\n{e}")

# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        prog="automr",
        description="Automated Metamorphic Testing Framework",
        epilog="""
Examples:
  automr info
  automr version
  automr docs
  automr docs architecture
  automr docs regression
  automr docs automr
  automr list-mrs
  automr list-transforms
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(
        dest="command",
        metavar="<command>",
    )

    # -------------------------------------------------------------------------
    # Core Commands
    # -------------------------------------------------------------------------

    sub.add_parser(
        "version",
        help="Show framework version",
    ).set_defaults(func=cmd_version)

    sub.add_parser(
        "info",
        help="Show framework information",
    ).set_defaults(func=cmd_info)

    # -------------------------------------------------------------------------
    # Documentation
    # -------------------------------------------------------------------------

    docs = sub.add_parser(
        "docs",
        help="Browse framework documentation",
    )

    docs.add_argument(
        "name",
        nargs="?",
        help="Documentation page (e.g. architecture, registry, regression)",
    )

    docs.set_defaults(func=cmd_docs)

    # -------------------------------------------------------------------------
    # Registries
    # -------------------------------------------------------------------------

    sub.add_parser(
        "list-mrs",
        help="List registered metamorphic relations",
    ).set_defaults(func=cmd_list_mrs)

    sub.add_parser(
        "list-transforms",
        help="List registered transformations",
    ).set_defaults(func=cmd_list_transforms)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    sub.add_parser(
        "examples",
        help="Show example commands",
    ).set_defaults(func=cmd_examples)

    sub.add_parser(
        "license",
        help="Show framework license",
    ).set_defaults(func=cmd_license)

    sub.add_parser(
        "citation",
        help="Show citation information",
    ).set_defaults(func=cmd_citation)

    # -------------------------------------------------------------------------
    # Reserved Commands
    # -------------------------------------------------------------------------

    benchmark = sub.add_parser(
        "benchmark",
        help="Run benchmarking suite",
    )

    benchmark.set_defaults(
        func=lambda args: print(
            "Benchmark module will be available in a future release."
        )
    )

    validate = sub.add_parser(
        "validate",
        help="Run metamorphic validation",
    )

    validate.set_defaults(
        func=lambda args: print(
            "Validation CLI will be available in a future release."
        )
    )

    graphs = sub.add_parser(
        "graphs",
        help="Generate result graphs",
    )

    graphs.set_defaults(
        func=lambda args: print(
            "Graph generation CLI will be available in a future release."
        )
    )

    report = sub.add_parser(
        "report",
        help="Generate analysis reports",
    )

    report.set_defaults(
        func=lambda args: print(
            "Report generation CLI will be available in a future release."
        )
    )

    # -------------------------------------------------------------------------
    # Parse Arguments
    # -------------------------------------------------------------------------

    args = parser.parse_args()

    if not hasattr(args, "func"):

        banner()

        parser.print_help()

        print("\nUseful commands:")
        print("  automr info")
        print("  automr docs")
        print("  automr docs architecture")
        print("  automr list-mrs")
        print("  automr list-transforms")
        print("  automr examples")

        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()