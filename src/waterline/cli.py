"""Waterline command line. `uv run waterline --help` lists the subcommands."""

import argparse
from pathlib import Path

from waterline import __version__, fixtures, screen
from waterline import eval as ev

FIXTURES_DIR = Path("tests/fixtures")


def _cmd_screen(args: argparse.Namespace) -> int:
    scores = screen.run_screen(Path("results/site_screen.json"), Path(args.sheet))
    for s in scores[:12]:
        print(
            f"{s.score:5.2f}  {s.cam_id:<52} range {s.range_ft:6.2f}ft  "
            f"events {s.events:<3} every {s.interval_min:.0f}min {s.note}"
        )
    print(f"screened {len(scores)} staff cameras -> results/site_screen.json + {args.sheet}")
    return 0


def _cmd_replay_record(args: argparse.Namespace) -> int:
    entries = fixtures.record(args.site, args.cam, args.nwis, fixtures.HANCOCK_WANTED, FIXTURES_DIR)
    for e in entries:
        print(f"{e.filename}  gage={e.gage_ft}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    ev.run_calibrate(args.site, args.start, args.end, args.every)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    ev.run_eval(args.site, args.start, args.end, args.every)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterline", description=__doc__)
    parser.add_argument("--version", action="version", version=f"waterline {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_screen = sub.add_parser("screen", help="rank USGS staff-gauge cameras")
    p_screen.add_argument("--sheet", default="docs/media/site_screen.jpg")
    p_screen.set_defaults(func=_cmd_screen)

    p_rec = sub.add_parser("replay-record", help="download fixture frames + gage truth")
    p_rec.add_argument("--site", default="hancock")
    p_rec.add_argument("--cam", default="NY_East_Branch_Delaware_River_at_Hancock_Staff")
    p_rec.add_argument("--nwis", default="01421500")
    p_rec.set_defaults(func=_cmd_replay_record)

    p_cal = sub.add_parser("calibrate", help="fit the datum offset on a calibration window")
    p_cal.add_argument("--site", required=True)
    p_cal.add_argument("--start", required=True)
    p_cal.add_argument("--end", required=True)
    p_cal.add_argument("--every", type=int, default=6)
    p_cal.set_defaults(func=_cmd_calibrate)

    p_eval = sub.add_parser("eval", help="evaluate on a held-out window")
    p_eval.add_argument("--site", required=True)
    p_eval.add_argument("--start", required=True)
    p_eval.add_argument("--end", required=True)
    p_eval.add_argument("--every", type=int, default=1)
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
