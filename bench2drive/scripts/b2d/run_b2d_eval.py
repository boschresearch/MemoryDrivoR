# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.b2d.common import ensure_external_paths


def _ensure_carla_paths() -> Path:
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        raise EnvironmentError("CARLA_ROOT must be set for B2D evaluation.")

    carla_root_path = Path(carla_root).expanduser()
    agents_root = carla_root_path / "PythonAPI" / "carla"

    try:
        import carla  # noqa: F401

        agents_root_str = str(agents_root)
        if agents_root.exists() and agents_root_str not in sys.path:
            sys.path.insert(0, agents_root_str)

        return carla_root_path
    except ImportError:
        pass

    dist_dir = carla_root_path / "PythonAPI" / "carla" / "dist"
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    py_egg_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
    dist_candidates = []
    fallback_candidates = []
    if dist_dir.exists():
        for path in dist_dir.iterdir():
            if py_tag in path.name or py_egg_tag in path.name:
                dist_candidates.append(path)
            elif sys.version_info.major == 3 and ("-cp3" in path.name or "-py3" in path.name):
                fallback_candidates.append(path)
        dist_candidates = sorted(dist_candidates, key=lambda path: (path.suffix != ".egg", path.name))
        fallback_candidates = sorted(fallback_candidates, key=lambda path: (path.suffix != ".egg", path.name))
    used_fallback = False
    if not dist_candidates and fallback_candidates:
        dist_candidates = fallback_candidates
        used_fallback = True
    if not dist_candidates:
        available = ", ".join(sorted(path.name for path in dist_dir.iterdir())) if dist_dir.exists() else "<missing dist dir>"
        raise EnvironmentError(
            f"Could not import carla from the active Python environment, and no CARLA Python package compatible with Python {sys.version_info.major}.{sys.version_info.minor} was found under {dist_dir}. "
            f"Available files: {available}. Install carla into the active environment, use a Python 3.7 environment for this CARLA build, or provide a compatible CARLA wheel."
        )

    if used_fallback:
        print(
            f"Warning: using fallback CARLA Python package {dist_candidates[0].name} with Python {sys.version_info.major}.{sys.version_info.minor}.",
            file=sys.stderr,
        )

    for path in [
        agents_root,
        dist_candidates[0],
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return carla_root_path


def main():
    repo_root, bench2drive_root, _ = ensure_external_paths()
    _ensure_carla_paths()

    parser = argparse.ArgumentParser(description="Run DrivoR closed-loop Bench2Drive evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--routes",
        type=Path,
        default=bench2drive_root / "leaderboard" / "data" / "bench2drive220.xml",
    )
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument(
        "--agent-path",
        type=Path,
        default=repo_root / "scripts" / "b2d" / "drivor_b2d_agent.py",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=repo_root / "scripts" / "b2d" / "drivor_b2d_config.py",
    )
    parser.add_argument(
        "--drivor-agent-config",
        type=Path,
        default=repo_root / "navsim" / "planning" / "script" / "config" / "common" / "agent" / "drivoR_b2d.yaml",
    )
    parser.add_argument("--checkpoint-endpoint", type=Path, default=None)
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--traffic-manager-port", type=int, default=20500)
    parser.add_argument("--gpu-rank", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--track", type=str, default="SENSORS")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument(
        "--save-camera-frames",
        action="store_true",
        help="Save per-frame camera PNGs under SAVE_PATH.",
    )
    parser.add_argument(
        "--save-trajectory-overlays",
        action="store_true",
        help="Save per-frame camera PNGs with the predicted trajectory overlaid.",
    )
    parser.add_argument(
        "--save-front-camera-only",
        action="store_true",
        help="Restrict saved PNGs to CAM_FRONT only and imply --save-camera-frames.",
    )
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.save_front_camera_only:
        args.save_camera_frames = True

    checkpoint_endpoint = args.checkpoint_endpoint
    if checkpoint_endpoint is None:
        checkpoint_endpoint = args.save_path / f"eval_{args.routes.stem}.json"

    args.save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_endpoint.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and checkpoint_endpoint.exists():
        checkpoint_endpoint.unlink()

    os.environ["Bench2Drive_ROOT"] = str(bench2drive_root)
    os.environ["LEADERBOARD_ROOT"] = str(bench2drive_root / "leaderboard")
    os.environ["SCENARIO_RUNNER_ROOT"] = str(bench2drive_root / "scenario_runner")
    os.environ["IS_BENCH2DRIVE"] = "True"
    os.environ["SAVE_PATH"] = str(args.save_path.resolve())
    os.environ["ROUTES"] = str(args.routes.resolve())
    os.environ["TEAM_AGENT"] = str(args.agent_path.resolve())
    os.environ["TEAM_CONFIG"] = f"{args.agent_config.resolve()}+{args.checkpoint.resolve()}"
    os.environ["CHECKPOINT_ENDPOINT"] = str(checkpoint_endpoint.resolve())
    os.environ["DRIVOR_B2D_AGENT_CONFIG"] = str(args.drivor_agent_config.resolve())
    os.environ["DRIVOR_B2D_SAVE_CAMERA_FRAMES"] = "1" if args.save_camera_frames else "0"
    os.environ["DRIVOR_B2D_SAVE_TRAJECTORY_OVERLAYS"] = "1" if args.save_trajectory_overlays else "0"
    os.environ["DRIVOR_B2D_SAVE_FRONT_CAMERA_ONLY"] = "1" if args.save_front_camera_only else "0"

    cli_args = [
        "leaderboard_evaluator.py",
        f"--routes={args.routes.resolve()}",
        f"--repetitions={args.repetitions}",
        f"--track={args.track}",
        f"--checkpoint={checkpoint_endpoint.resolve()}",
        f"--agent={args.agent_path.resolve()}",
        f"--agent-config={args.agent_config.resolve()}+{args.checkpoint.resolve()}",
        f"--debug={args.debug}",
        "--resume=True",
        f"--port={args.port}",
        f"--traffic-manager-port={args.traffic_manager_port}",
        f"--gpu-rank={args.gpu_rank}",
        f"--timeout={args.timeout}",
    ]
    if args.record is not None:
        cli_args.append(f"--record={args.record.resolve()}")

    # The route parser requires Element.getchildren(), which lxml provides on
    # Python 3.9 and later.
    from lxml import etree
    from leaderboard.utils import route_parser as leaderboard_route_parser

    leaderboard_route_parser.ET = etree

    from leaderboard.leaderboard_evaluator import main as leaderboard_main

    old_cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(bench2drive_root)
        sys.argv = cli_args
        leaderboard_main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
