from __future__ import annotations

import argparse

from cue.core.cue_runner import add_common_args, run_cue


def parse_args() -> argparse.Namespace:
    """
    Args:
        None

    Returns:
        argparse.Namespace
    """
    p = argparse.ArgumentParser()
    add_common_args(p, default_modalities=["fp", "img"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_cue(args)


if __name__ == "__main__":
    main()
