# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import ctypes
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


def _preload_libjpeg() -> None:
    candidates = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libjpeg.so.8")

    conda_pkg_root = Path.home() / ".conda" / "pkgs"
    if conda_pkg_root.exists():
        candidates.extend(sorted(conda_pkg_root.glob("libjpeg-turbo-*/lib/libjpeg.so.8"), reverse=True))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            return
        except OSError:
            continue


def _ensure_carla_paths():
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        raise EnvironmentError("CARLA_ROOT must be set to generate B2D map info.")
    carla_root = Path(carla_root)
    #dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    #py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    #py_egg_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
    #dist_candidates = []
    #if dist_dir.exists():
    #    dist_candidates = sorted(
    #        [
    #            path
    #            for path in dist_dir.iterdir()
    #            if py_tag in path.name or py_egg_tag in path.name
    #        ],
    #        key=lambda path: (path.suffix != ".egg", path.name),
    #    )
    #if not dist_candidates:
    #    available = ", ".join(sorted(path.name for path in dist_dir.iterdir())) if dist_dir.exists() else "<missing dist dir>"
    #    raise EnvironmentError(
    #        f"No CARLA Python package compatible with Python {sys.version_info.major}.{sys.version_info.minor} was found under {dist_dir}. "
    #        f"Available files: {available}. Use a Python 3.7 environment for map generation with this CARLA build, or provide a compatible CARLA wheel."
    #    )
    #
    #for path in [
    #    carla_root / "PythonAPI",
    #    carla_root / "PythonAPI" / "carla",
    #    dist_candidates[0],
    #]:
    #    path_str = str(path)
    #    if path_str not in sys.path:
    #        sys.path.insert(0, path_str)

    _preload_libjpeg()
    return carla_root


def _resolve_carla_command(carla_root: Path, port: int, fps: int, graphics_adapter: Optional[int], use_wrapper: bool) -> List[str]:
    direct_binary = carla_root / "CarlaUE4" / "Binaries" / "Linux" / "CarlaUE4-Linux-Shipping"
    wrapper = carla_root / "CarlaUE4.sh"

    if use_wrapper or not direct_binary.exists():
        executable = wrapper
    else:
        executable = direct_binary

    cmd = [
        str(executable),
        "-RenderOffScreen",
        "-nosound",
        f"-fps={fps}",
        f"-carla-rpc-port={port}",
    ]
    if graphics_adapter is not None:
        cmd.append(f"-graphicsadapter={graphics_adapter}")
    return cmd


def _wait_for_carla_server(server: subprocess.Popen, carla_module, port: int, timeout: float):
    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"CARLA server exited before accepting connections (exit code {server.returncode}).")
        try:
            client = carla_module.Client("localhost", port)
            client.set_timeout(5.0)
            client.get_world()
            client.set_timeout(300.0)
            return client
        except RuntimeError as exc:
            last_error = exc
            time.sleep(2)

    raise TimeoutError(f"Timed out waiting {timeout:.0f}s for CARLA on port {port}: {last_error}")


def get_lane_markings(carla_map, precision=0.05):
    topology = [x[0] for x in carla_map.get_topology()]
    topology = sorted(topology, key=lambda waypoint: waypoint.road_id)
    map_list = []

    for waypoint in topology:
        waypoints = [waypoint]
        nxt = waypoint.next(precision)
        if nxt:
            nxt = nxt[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)
                if nxt:
                    nxt = nxt[0]
                else:
                    break

        maps = []
        for wp in waypoints:
            transform = wp.transform
            road_lane_id = wp.road_id + wp.lane_id * 0.001
            maps.append((transform.location.x, -transform.location.y, wp.lane_width * 0.5, road_lane_id))

        maps = np.array(maps, dtype=np.float32)[::20]
        if len(maps) > 1:
            way_dist = np.linalg.norm(maps[1:, :2] - maps[:-1, :2], axis=-1)
            width = np.maximum(maps[1:, 2], maps[:-1, 2])
            maps[:-1, 2] = np.sqrt(way_dist * way_dist / 4 + width * width)
        map_list.append(maps)

    return np.concatenate(map_list)


def _get_loaded_town_name(world) -> str:
    return world.get_map().name.split("/")[-1]


def _load_world_and_wait(client, carla_town: str, timeout: float):
    deadline = time.monotonic() + timeout
    last_loaded_town = None

    client.load_world(carla_town)
    while time.monotonic() < deadline:
        world = client.get_world()
        loaded_town = _get_loaded_town_name(world)
        last_loaded_town = loaded_town
        if loaded_town[:6] == carla_town[:6]:
            return world
        time.sleep(2)

    raise RuntimeError(
        f"Requested CARLA town {carla_town}, but server loaded {last_loaded_town}. "
        "This would generate an invalid map.pkl. Check the CARLA build/assets before retrying."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate map.pkl for DrivoR B2D scoring.")
    parser.add_argument("--port", type=int, default=20001)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--graphics-adapter", type=int, default=int(os.environ.get("B2D_CARLA_GRAPHICS_ADAPTER", 0)))
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--use-wrapper", action="store_true")
    default_output = None
    navsim_exp_root = os.environ.get("NAVSIM_EXP_ROOT")
    if navsim_exp_root:
        default_output = Path(navsim_exp_root) / "map.pkl"
    parser.add_argument("--output-path", type=Path, default=default_output)
    args = parser.parse_args()

    carla_root = _ensure_carla_paths()
    if args.output_path is None:
        raise EnvironmentError("Set NAVSIM_EXP_ROOT or pass --output-path explicitly.")

    import carla

    cmd = _resolve_carla_command(
        carla_root=carla_root,
        port=args.port,
        fps=args.fps,
        graphics_adapter=args.graphics_adapter,
        use_wrapper=args.use_wrapper,
    )
    server_env = os.environ.copy()
    server_env.setdefault("DISPLAY", "")
    print(f"Launching CARLA: {' '.join(cmd)}", flush=True)
    server = subprocess.Popen(cmd, cwd=carla_root, env=server_env, preexec_fn=os.setsid)

    try:
        client = _wait_for_carla_server(server, carla, args.port, args.startup_timeout)
        map_dict = {}
        for town_id in ["11", "01", "02", "03", "04", "05", "06", "07", "10HD", "12", "13", "15"]:
            carla_town = "Town" + town_id
            world = _load_world_and_wait(client, carla_town, args.startup_timeout)

            lane_markings = get_lane_markings(world.get_map())
            duplicate_town = next(
                (
                    existing_town
                    for existing_town, existing_markings in map_dict.items()
                    if existing_markings.shape == lane_markings.shape and np.array_equal(existing_markings, lane_markings)
                ),
                None,
            )
            if duplicate_town is not None:
                raise RuntimeError(
                    f"Generated identical lane-marking arrays for {carla_town} and {duplicate_town}. "
                    "This is unexpected and likely indicates CARLA loaded the wrong map or reused stale assets."
                )

            map_dict[carla_town[:6]] = lane_markings
            print(f"Extracted map info for {carla_town}", flush=True)

        with open(args.output_path, "wb") as f:
            pickle.dump(map_dict, f)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
