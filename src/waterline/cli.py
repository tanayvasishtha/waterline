"""Command line entry point. Subcommands are added as their modules land."""

import argparse

from waterline import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterline", description=__doc__)
    parser.add_argument("--version", action="version", version=f"waterline {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
