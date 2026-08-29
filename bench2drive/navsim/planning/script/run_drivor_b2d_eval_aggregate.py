#!/usr/bin/env python3
# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import argparse
import contextlib
import io
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


B2D_WEATHER_BUCKETS = {
    "easy": frozenset({0, 1, 7, 26}),
    "okay": frozenset({2, 3, 5, 6, 15, 18}),
    "medium": frozenset({8, 14}),
    "hard": frozenset({9, 10, 11, 12, 13, 19, 22, 23}),
    "extreme": frozenset({20, 21, 25}),
}
DEFAULT_BAD_WEATHER_LEVELS = ("hard", "extreme")


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def _normalize_weather_level(weather_level: str) -> str:
    normalized_weather_level = str(weather_level).strip().lower()
    if normalized_weather_level not in B2D_WEATHER_BUCKETS:
        valid_levels = ", ".join(sorted(B2D_WEATHER_BUCKETS.keys()))
        raise ValueError(
            f"Unknown weather level '{weather_level}'. Valid levels: {valid_levels}."
        )
    return normalized_weather_level


def _resolve_requested_weather_levels(
    weather_levels: Optional[Sequence[str]],
    bad_weather_only: bool,
) -> List[str]:
    requested_levels: List[str] = []
    if bad_weather_only:
        requested_levels.extend(list(DEFAULT_BAD_WEATHER_LEVELS))
    if weather_levels is not None:
        requested_levels.extend([str(weather_level) for weather_level in weather_levels])

    normalized_levels: List[str] = []
    seen_levels = set()
    for weather_level in requested_levels:
        normalized_weather_level = _normalize_weather_level(weather_level)
        if normalized_weather_level in seen_levels:
            continue
        normalized_levels.append(normalized_weather_level)
        seen_levels.add(normalized_weather_level)
    return normalized_levels


def add_weather_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bad-weather-only",
        action="store_true",
        help="Aggregate only the hard/extreme Bench2Drive weather buckets.",
    )
    parser.add_argument(
        "--weather-levels",
        nargs="+",
        default=None,
        help="Aggregate only Bench2Drive weather buckets: easy, okay, medium, hard, extreme.",
    )
    parser.add_argument(
        "--weather-indices",
        nargs="+",
        type=int,
        default=None,
        help="Aggregate only exact Bench2Drive weather ids.",
    )


def append_weather_filter_args(
    command: List[str],
    *,
    bad_weather_only: bool,
    weather_levels: Optional[Sequence[str]],
    weather_indices: Optional[Sequence[int]],
) -> None:
    if bad_weather_only:
        command.append("--bad-weather-only")
    if weather_levels:
        command.append("--weather-levels")
        command.extend(str(weather_level) for weather_level in weather_levels)
    if weather_indices:
        command.append("--weather-indices")
        command.extend(str(weather_index) for weather_index in weather_indices)


def resolve_allowed_weather_ids(
    weather_levels: Optional[Sequence[str]],
    weather_indices: Optional[Sequence[int]],
    bad_weather_only: bool = False,
) -> Optional[set[int]]:
    allowed_weather_ids: set[int] = set()
    for weather_level in _resolve_requested_weather_levels(weather_levels, bad_weather_only):
        allowed_weather_ids.update(B2D_WEATHER_BUCKETS[weather_level])

    if weather_indices is not None:
        allowed_weather_ids.update(int(weather_index) for weather_index in weather_indices)

    if len(allowed_weather_ids) == 0:
        return None
    return allowed_weather_ids


def resolve_weather_filter_name(
    *,
    weather_levels: Optional[Sequence[str]],
    weather_indices: Optional[Sequence[int]],
    bad_weather_only: bool,
) -> Optional[str]:
    if not bad_weather_only and not weather_levels and not weather_indices:
        return None

    if bad_weather_only and not weather_levels and not weather_indices:
        return "weather_bad"

    parts: List[str] = []
    if bad_weather_only:
        parts.append("bad")

    specific_levels = _resolve_requested_weather_levels(weather_levels, bad_weather_only=False)
    if specific_levels:
        parts.append(f"levels-{'-'.join(specific_levels)}")

    if weather_indices:
        unique_indices = sorted({int(weather_index) for weather_index in weather_indices})
        parts.append(f"ids-{'-'.join(str(weather_index) for weather_index in unique_indices)}")

    return f"weather_{'__'.join(parts)}"


def resolve_aggregate_output_dir(
    base_output_dir: Path,
    *,
    weather_levels: Optional[Sequence[str]],
    weather_indices: Optional[Sequence[int]],
    bad_weather_only: bool,
) -> Path:
    filter_name = resolve_weather_filter_name(
        weather_levels=weather_levels,
        weather_indices=weather_indices,
        bad_weather_only=bad_weather_only,
    )
    if filter_name is None:
        return base_output_dir
    return base_output_dir / filter_name


def _parse_record_weather_id(record: Dict[str, object]) -> Optional[int]:
    weather_id = record.get("weather_id")
    if weather_id is None:
        return None

    try:
        return int(str(weather_id).strip())
    except (TypeError, ValueError):
        return None


def _extract_benchmark_route_id_from_record(record: Dict[str, object]) -> Optional[str]:
    route_id = str(record.get("route_id", "")).strip()
    if not route_id:
        return None

    match = re.search(r"_(\d+)_rep\d+$", route_id)
    if match is not None:
        return match.group(1)
    if route_id.isdigit():
        return route_id
    return None


def _route_succeeded_without_infractions(record: Dict[str, object]) -> bool:
    if record.get("status") not in {"Completed", "Perfect"}:
        return False

    infractions = record.get("infractions") or {}
    if not isinstance(infractions, dict):
        return False

    for infraction_name, values in infractions.items():
        if infraction_name == "min_speed_infractions":
            continue
        if len(values) > 0:
            return False
    return True


def _build_merged_data(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    merged_records = sorted(records, key=lambda record: str(record.get("route_id", "")), reverse=True)
    eval_num = len(merged_records)
    total_driving_score = 0.0
    success_num = 0

    for record in merged_records:
        scores = record.get("scores") or {}
        if isinstance(scores, dict):
            total_driving_score += float(scores.get("score_composed", 0.0))
        if _route_succeeded_without_infractions(record):
            success_num += 1

    return {
        "_checkpoint": {"records": merged_records},
        "driving score": (total_driving_score / eval_num) if eval_num > 0 else None,
        "success rate": (success_num / eval_num) if eval_num > 0 else None,
        "eval num": eval_num,
    }


def _summarize_weather_ids(records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    weather_counts: Dict[int, int] = {}
    unknown_count = 0

    for record in records:
        weather_id = _parse_record_weather_id(record)
        if weather_id is None:
            unknown_count += 1
            continue
        weather_counts[weather_id] = weather_counts.get(weather_id, 0) + 1

    summary = {str(weather_id): weather_counts[weather_id] for weather_id in sorted(weather_counts)}
    if unknown_count > 0:
        summary["unknown"] = unknown_count
    return summary


def _filter_merged_data_by_weather(
    merged_data: Dict[str, object],
    allowed_weather_ids: set[int],
) -> Dict[str, object]:
    records = merged_data.get("_checkpoint", {}).get("records", [])
    filtered_records: List[Dict[str, object]] = []
    available_weather_ids: set[int] = set()

    for record in records:
        if not isinstance(record, dict):
            continue
        weather_id = _parse_record_weather_id(record)
        if weather_id is not None:
            available_weather_ids.add(weather_id)
        if weather_id in allowed_weather_ids:
            filtered_records.append(record)

    if not filtered_records:
        requested = ", ".join(str(weather_id) for weather_id in sorted(allowed_weather_ids))
        available = ", ".join(str(weather_id) for weather_id in sorted(available_weather_ids)) or "none"
        raise ValueError(
            "Weather filter selected zero routes. "
            f"Requested weather ids: {requested}. Available weather ids in results: {available}."
        )

    return _build_merged_data(filtered_records)


def _merge_route_results(results_dir: Path) -> Dict[str, object]:
    from Bench2Drive.tools.merge_route_json import merge_route_json

    file_paths = sorted(path for path in results_dir.glob("*.json") if "merged" not in path.name)
    if not file_paths:
        raise FileNotFoundError(f"No route result JSON files found in {results_dir}")

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        merge_route_json(str(results_dir))
    for line in stdout.getvalue().splitlines():
        if "Warning:" in line:
            print(line)

    merged_file = results_dir / "merged.json"
    with merged_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_local_port(start: int, end: int, step: int) -> int:
    for port in range(start, end, step):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            sock.close()
            continue
        sock.close()
        return port
    raise RuntimeError(f"Unable to find a free local port in range [{start}, {end}) with step {step}")


def _compute_ability_metrics(
    merged_data: Dict[str, object],
    routes_file: Path,
    carla_root: Path,
    host: str,
    port: int,
    startup_sleep: int,
) -> Dict[str, object]:
    os.environ["CARLA_ROOT"] = str(carla_root)
    from Bench2Drive.tools import ability_benchmark as ability_tool
    import carla

    server = subprocess.Popen(
        f"{_quote(carla_root / 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={port}",
        shell=True,
        preexec_fn=os.setsid,
    )
    try:
        time.sleep(startup_sleep)
        client = carla.Client(host, port)
        client.set_timeout(300)

        records = merged_data.get("_checkpoint", {}).get("records", [])
        if not records:
            raise RuntimeError("No merged route records are available for ability aggregation.")

        selected_route_ids = {
            route_id
            for record in records
            if (route_id := _extract_benchmark_route_id_from_record(record)) is not None
        }
        if not selected_route_ids:
            raise RuntimeError(
                "Could not map merged route records back to benchmark route ids for ability aggregation."
            )

        root = ET.parse(routes_file).getroot()
        routes = sorted(
            (route for route in root.findall("route") if route.get("id") in selected_route_ids),
            key=lambda item: item.get("town"),
        )
        if not routes:
            raise RuntimeError(
                f"No routes from {routes_file} matched the merged route results selected for aggregation."
            )

        ability_stats = {key: [0, 0.0] for key in ability_tool.Ability}
        crash_route_list: List[Tuple[str, str]] = []

        current_town = routes[0].get("town")
        world = client.load_world(current_town)
        carla_map = world.get_map()
        grp = ability_tool.GlobalRoutePlanner(carla_map, 1.0)

        for route in routes:
            scenarios = route.find("scenarios")
            scenario_name = scenarios.find("scenario").get("type")
            route_id = route.get("id")
            route_record = ability_tool.get_route_result(records, route_id)
            if route_record is None:
                crash_route_list.append((scenario_name, route_id))
                continue

            if route_record.get("status") in {"Completed", "Perfect"}:
                record_success_status = not ability_tool.get_infraction_status(route_record)
            else:
                record_success_status = False

            ability_tool.update_Ability(scenario_name, ability_stats, record_success_status)

            if scenario_name in ability_tool.Ability["Traffic_Signs"]:
                if route.get("town") != current_town:
                    current_town = route.get("town")
                    world = client.load_world(current_town)
                carla_map = world.get_map()
                grp = ability_tool.GlobalRoutePlanner(carla_map, 1.0)
                location_list = ability_tool.get_position(route)
                waypoint_route = ability_tool.get_waypoint_route(location_list, grp)
                count = 0
                for waypoint in waypoint_route:
                    count += 1
                    if waypoint.is_junction:
                        break
                junction_completion = float(count + 8) / float(len(waypoint_route))
                record_completion = route_record["scores"]["score_route"] / 100.0
                stop_infraction = route_record["infractions"]["stop_infraction"]
                red_light_infraction = route_record["infractions"]["red_light"]
                if record_completion > junction_completion and not stop_infraction and not red_light_infraction:
                    ability_stats["Traffic_Signs"][0] += 1
                ability_stats["Traffic_Signs"][1] += 1

        ability_res: Dict[str, object] = {}
        for ability_name, stats in ability_stats.items():
            if stats[1] == 0:
                ability_res[ability_name] = None
            else:
                ability_res[ability_name] = float(stats[0]) / float(stats[1])

        numeric_scores = [score for score in ability_res.values() if isinstance(score, (int, float))]
        ability_res["mean"] = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        ability_res["crashed"] = crash_route_list
        ability_res["routes_evaluated"] = len(routes)
        return ability_res
    finally:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _find_metric_run_dir(metric_dir: Path, save_name: str) -> Optional[Path]:
    direct_candidates = [
        metric_dir / save_name,
        metric_dir / f"shuffle_{save_name}",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir() and (candidate / "metric_info.json").is_file():
            return candidate

    matches: Dict[str, Path] = {}
    for pattern in (f"*/{save_name}", f"*/*{save_name}"):
        for candidate in metric_dir.glob(pattern):
            if candidate.is_dir() and (candidate / "metric_info.json").is_file():
                matches[str(candidate.resolve())] = candidate

    if not matches:
        return None

    if len(matches) > 1:
        matched_paths = ", ".join(sorted(matches))
        raise RuntimeError(
            f"More than one metric run directory matched save_name {save_name} under {metric_dir}: {matched_paths}"
        )

    return next(iter(matches.values()))


def _ensure_b2d_metric_layout(merged_file: Path, metric_dir: Path) -> None:
    with merged_file.open("r", encoding="utf-8") as handle:
        merged_data = json.load(handle)

    for record in merged_data.get("_checkpoint", {}).get("records", []):
        save_name = record.get("save_name")
        if not save_name:
            continue

        metric_run_dir = _find_metric_run_dir(metric_dir, save_name)
        if metric_run_dir is None or metric_run_dir.name == save_name:
            continue

        alias_dir = metric_run_dir.parent / save_name
        if alias_dir.exists() or alias_dir.is_symlink():
            continue

        alias_dir.symlink_to(metric_run_dir, target_is_directory=True)


def _compute_efficiency_and_smoothness(
    merged_file: Path,
    metric_dir: Path,
) -> Dict[str, object]:
    from Bench2Drive.tools.efficiency_smoothness_benchmark import read_from_json, seg_compute_comfort_metric

    _ensure_b2d_metric_layout(merged_file=merged_file, metric_dir=metric_dir)

    all_data, driving_efficiency_scores = read_from_json(str(merged_file), str(metric_dir))

    comfort_scores = [seg_compute_comfort_metric(**record) for record in all_data]

    return {
        "driving_efficiency": (
            sum(driving_efficiency_scores) / len(driving_efficiency_scores)
            if driving_efficiency_scores
            else None
        ),
        "driving_smoothness": sum(comfort_scores) / len(comfort_scores) if comfort_scores else None,
    }


def _run_aggregate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_data = _merge_route_results(args.results_dir)
    allowed_weather_ids = resolve_allowed_weather_ids(
        weather_levels=args.weather_levels,
        weather_indices=args.weather_indices,
        bad_weather_only=args.bad_weather_only,
    )
    if allowed_weather_ids is not None:
        merged_data = _filter_merged_data_by_weather(merged_data, allowed_weather_ids)

    merged_file = args.output_dir / "merged.json"
    with merged_file.open("w", encoding="utf-8") as handle:
        json.dump(merged_data, handle, indent=2)

    summary: Dict[str, object] = {
        "merged_json": str(merged_file),
        "driving_score": merged_data.get("driving score"),
        "success_rate": merged_data.get("success rate"),
        "eval_num": merged_data.get("eval num"),
    }
    if allowed_weather_ids is not None:
        summary["weather_filter"] = {
            "bad_weather_only": bool(args.bad_weather_only),
            "requested_levels": list(args.weather_levels or []),
            "requested_indices": list(args.weather_indices or []),
            "resolved_weather_ids": sorted(allowed_weather_ids),
            "matched_weather_id_counts": _summarize_weather_ids(
                merged_data.get("_checkpoint", {}).get("records", [])
            ),
        }

    if not args.skip_ability and args.routes_file is not None and args.carla_root is not None:
        ability_port = _pick_local_port(args.ability_port_start, args.ability_port_end, args.ability_port_step)
        ability_data = _compute_ability_metrics(
            merged_data=merged_data,
            routes_file=args.routes_file,
            carla_root=args.carla_root,
            host=args.host,
            port=ability_port,
            startup_sleep=args.ability_startup_sleep,
        )
        ability_file = args.output_dir / "ability.json"
        with ability_file.open("w", encoding="utf-8") as handle:
            json.dump(ability_data, handle, indent=2)
        summary["ability_file"] = str(ability_file)
        summary["ability_mean"] = ability_data.get("mean")
    elif not args.skip_ability:
        summary["ability_skipped_reason"] = "routes_file or carla_root missing"

    if not args.skip_efficiency and args.metric_dir is not None:
        efficiency_data = _compute_efficiency_and_smoothness(merged_file=merged_file, metric_dir=args.metric_dir)
        efficiency_file = args.output_dir / "efficiency_smoothness.json"
        with efficiency_file.open("w", encoding="utf-8") as handle:
            json.dump(efficiency_data, handle, indent=2)
        summary["efficiency_file"] = str(efficiency_file)
        summary["driving_efficiency"] = efficiency_data.get("driving_efficiency")
        summary["driving_smoothness"] = efficiency_data.get("driving_smoothness")
    elif not args.skip_efficiency:
        summary["efficiency_skipped_reason"] = "metric_dir missing"

    summary_file = args.output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate route-level DrivoR Bench2Drive results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-dir", type=Path, default=None)
    parser.add_argument("--routes-file", type=Path, default=None)
    parser.add_argument(
        "--carla-root",
        type=Path,
        default=Path(os.environ["CARLA_ROOT"]) if os.environ.get("CARLA_ROOT") else None,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--skip-ability", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--ability-port-start", type=int, default=20002)
    parser.add_argument("--ability-port-end", type=int, default=29999)
    parser.add_argument("--ability-port-step", type=int, default=17)
    parser.add_argument("--ability-startup-sleep", type=int, default=15)
    add_weather_filter_args(parser)
    args = parser.parse_args(list(argv))
    args.results_dir = args.results_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.metric_dir is not None:
        args.metric_dir = args.metric_dir.expanduser().resolve()
    if args.routes_file is not None:
        args.routes_file = args.routes_file.expanduser().resolve()
    if args.carla_root is not None:
        args.carla_root = args.carla_root.expanduser().resolve()
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _run_aggregate(args)


if __name__ == "__main__":
    main()