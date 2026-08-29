# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import IntersectionType, SemanticMapLayer, StopLineType
from nuplan.common.maps.nuplan_map.utils import get_distance_between_map_object_and_point
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    prune_route_by_connectivity,
)

from navsim.common.dataclasses import Scene
from navsim.agents.drivoR.hd_map_schema import (
    HD_MAP_DEFAULT_POLYGON_LAYER_NAMES,
    HD_MAP_ELEMENT_TYPE_TO_ID,
    HD_MAP_GEOMETRY_TYPE_TO_ID,
    HD_MAP_ON_ROUTE_OFF,
    HD_MAP_ON_ROUTE_ON,
    HD_MAP_ON_ROUTE_UNKNOWN,
    HD_MAP_POLYGON_LAYER_TO_ELEMENT_TYPE,
    HD_MAP_STOP_LINE_SUBTYPE_UNKNOWN,
    HD_MAP_SUPPORTED_POLYGON_LAYER_NAMES,
)


@lru_cache(maxsize=16)
def get_map_api(map_name: str) -> AbstractMap:
    """Loads and caches the nuPlan map API for repeated feature extraction."""
    return Scene._build_map_api(map_name)


def _empty_hd_map_tensors(max_polylines: int, points_per_polyline: int) -> Dict[str, torch.Tensor]:
    return {
        "hd_map_coords": torch.zeros((max_polylines, points_per_polyline, 2), dtype=torch.float32),
        "hd_map_vxvy": torch.zeros((max_polylines, points_per_polyline, 2), dtype=torch.float32),
        "hd_map_valid_mask": torch.zeros((max_polylines,), dtype=torch.bool),
        "hd_map_geometry_type_ids": torch.zeros((max_polylines,), dtype=torch.long),
        "hd_map_element_type_ids": torch.zeros((max_polylines,), dtype=torch.long),
        "hd_map_on_route_ids": torch.zeros((max_polylines,), dtype=torch.long),
        "hd_map_has_traffic_light_ids": torch.zeros((max_polylines,), dtype=torch.long),
        "hd_map_stop_line_subtype_ids": torch.full(
            (max_polylines,),
            fill_value=HD_MAP_STOP_LINE_SUBTYPE_UNKNOWN,
            dtype=torch.long,
        ),
        "hd_map_speed_limit_mps": torch.full((max_polylines,), fill_value=-1.0, dtype=torch.float32),
    }


def _sample_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """Uniformly samples a polyline in local ego coordinates."""
    if len(polyline) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(polyline) == 1:
        return np.repeat(polyline.astype(np.float32), num_points, axis=0)

    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64)))
    total_length = float(cumulative[-1])

    if total_length <= 1e-6:
        return np.repeat(polyline[:1].astype(np.float32), num_points, axis=0)

    sample_positions = np.linspace(0.0, total_length, num_points, dtype=np.float64)
    sampled = np.zeros((num_points, 2), dtype=np.float32)

    for sample_idx, position in enumerate(sample_positions):
        seg_idx = min(int(np.searchsorted(cumulative, position, side="right") - 1), len(segment_lengths) - 1)
        seg_start = cumulative[seg_idx]
        seg_length = max(float(segment_lengths[seg_idx]), 1e-6)
        alpha = float((position - seg_start) / seg_length)
        sampled[sample_idx] = (
            (1.0 - alpha) * polyline[seg_idx].astype(np.float32)
            + alpha * polyline[seg_idx + 1].astype(np.float32)
        )

    return sampled


def _global_to_local_points(points: np.ndarray, ego_global_pose: np.ndarray) -> np.ndarray:
    dx = points[:, 0] - float(ego_global_pose[0])
    dy = points[:, 1] - float(ego_global_pose[1])
    yaw = float(ego_global_pose[2])
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    local_x = (dx * cos_yaw) + (dy * sin_yaw)
    local_y = (-dx * sin_yaw) + (dy * cos_yaw)
    return np.stack([local_x, local_y], axis=-1).astype(np.float32)


def _sample_direction_vectors(
    global_points: np.ndarray,
    headings: np.ndarray,
    num_points: int,
    ego_global_pose: np.ndarray,
) -> np.ndarray:
    """Uniformly samples baseline driving directions as local-frame unit (vx, vy) vectors."""
    if len(global_points) == 0 or len(headings) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    local_headings = headings.astype(np.float32) - float(ego_global_pose[2])
    direction_vectors = np.stack([np.cos(local_headings), np.sin(local_headings)], axis=-1).astype(np.float32)
    if len(direction_vectors) == 1:
        return np.repeat(direction_vectors[:1], num_points, axis=0)

    segment_lengths = np.linalg.norm(np.diff(global_points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(direction_vectors[:1], num_points, axis=0)

    sample_positions = np.linspace(0.0, total_length, num_points, dtype=np.float64)
    sampled = np.zeros((num_points, 2), dtype=np.float32)

    for sample_idx, position in enumerate(sample_positions):
        seg_idx = min(int(np.searchsorted(cumulative, position, side="right") - 1), len(segment_lengths) - 1)
        seg_start = cumulative[seg_idx]
        seg_length = max(float(segment_lengths[seg_idx]), 1e-6)
        alpha = float((position - seg_start) / seg_length)
        interp_direction = (
            (1.0 - alpha) * direction_vectors[seg_idx].astype(np.float32)
            + alpha * direction_vectors[seg_idx + 1].astype(np.float32)
        )
        interp_norm = float(np.linalg.norm(interp_direction))
        if interp_norm <= 1e-6:
            sampled[sample_idx] = direction_vectors[seg_idx]
        else:
            sampled[sample_idx] = interp_direction / interp_norm

    return sampled


def _normalize_polygon_layers(config) -> List[SemanticMapLayer]:
    configured_layers = config.get("polygon_layers", HD_MAP_DEFAULT_POLYGON_LAYER_NAMES)
    polygon_layers: List[SemanticMapLayer] = []
    for layer_name in configured_layers:
        if not isinstance(layer_name, str):
            raise ValueError(
                "`hd_map.feature_builder.polygon_layers` must contain SemanticMapLayer names as strings, "
                f"got {layer_name!r}."
            )
        try:
            layer = SemanticMapLayer[layer_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported polygon layer {layer_name!r}. Expected one of {HD_MAP_SUPPORTED_POLYGON_LAYER_NAMES}."
            ) from exc
        if layer not in HD_MAP_POLYGON_LAYER_TO_ELEMENT_TYPE:
            raise ValueError(
                f"Unsupported polygon layer {layer_name!r}. Expected one of {HD_MAP_SUPPORTED_POLYGON_LAYER_NAMES}."
            )
        polygon_layers.append(layer)
    return polygon_layers


def _discrete_path_to_global_points(discrete_path) -> np.ndarray:
    return np.array([[point.x, point.y] for point in discrete_path], dtype=np.float32)


def _polygon_to_global_points(map_obj) -> np.ndarray:
    polygon = map_obj.polygon
    exterior = np.array(polygon.exterior.coords, dtype=np.float32)
    if exterior.ndim != 2 or exterior.shape[-1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return exterior[:, :2]


def _safe_speed_limit_mps(map_obj) -> float:
    try:
        speed_limit_mps = getattr(map_obj, "speed_limit_mps", None)
    except NotImplementedError:
        return -1.0
    if speed_limit_mps is None:
        return -1.0
    speed_limit_mps = float(speed_limit_mps)
    if not np.isfinite(speed_limit_mps):
        return -1.0
    return speed_limit_mps


def _stop_line_subtype_id(map_obj, layer_name: SemanticMapLayer) -> int:
    if layer_name != SemanticMapLayer.STOP_LINE:
        return HD_MAP_STOP_LINE_SUBTYPE_UNKNOWN

    stop_line_type = getattr(map_obj, "stop_line_type", StopLineType.UNKNOWN)
    return int(stop_line_type)


def _has_traffic_lights(map_obj, layer_name: SemanticMapLayer, stop_line_subtype_id: int) -> int:
    if layer_name == SemanticMapLayer.STOP_LINE:
        return int(stop_line_subtype_id == int(StopLineType.TRAFFIC_LIGHT))

    intersection_type = getattr(map_obj, "intersection_type", None)
    if intersection_type is not None:
        return int(int(intersection_type) == int(IntersectionType.TRAFFIC_LIGHT))

    try:
        return int(bool(map_obj.has_traffic_lights()))
    except (AttributeError, NotImplementedError):
        return 0


def _roadblock_id(map_obj, layer_name: SemanticMapLayer) -> Optional[str]:
    if layer_name in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        return str(map_obj.get_roadblock_id())
    if layer_name in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
        return str(map_obj.id)
    return None


def build_hd_map_feature_tensors(
    map_name: str,
    ego_global_pose: np.ndarray,
    route_roadblock_ids: Sequence[str],
    config,
) -> Dict[str, torch.Tensor]:
    """Builds fixed-shape HD-map tensors for caching and model consumption."""
    query_radius_m = float(config.get("query_radius_m", 35.0))
    points_per_polyline = int(config.get("points_per_polyline", 20))
    max_polylines = int(config.get("max_polylines", 96))
    include_baseline = bool(config.get("include_baseline", config.get("include_center", True)))
    include_left = bool(config.get("include_left_boundary", True))
    include_right = bool(config.get("include_right_boundary", True))
    polygon_layers = _normalize_polygon_layers(config)

    if max_polylines <= 0:
        raise ValueError(f"`hd_map.feature_builder.max_polylines` must be > 0, got {max_polylines}.")
    if points_per_polyline <= 1:
        raise ValueError(
            f"`hd_map.feature_builder.points_per_polyline` must be > 1, got {points_per_polyline}."
        )

    tensors = _empty_hd_map_tensors(max_polylines=max_polylines, points_per_polyline=points_per_polyline)
    if not map_name:
        return tensors

    map_api = get_map_api(map_name)
    query_point = Point2D(float(ego_global_pose[0]), float(ego_global_pose[1]))
    line_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layer_names = line_layers + polygon_layers
    proximal_objects = map_api.get_proximal_map_objects(query_point, query_radius_m, layer_names)

    nearby_objects: List[Dict[str, object]] = []
    extracted_roadblock_ids = set()
    for layer_name in layer_names:
        for map_obj in proximal_objects[layer_name]:
            roadblock_id = _roadblock_id(map_obj, layer_name)
            if roadblock_id is not None:
                extracted_roadblock_ids.add(roadblock_id)

            stop_line_subtype_id = _stop_line_subtype_id(map_obj, layer_name)
            item = {
                "distance": float(get_distance_between_map_object_and_point(query_point, map_obj)),
                "map_obj": map_obj,
                "layer_name": layer_name,
                "roadblock_id": roadblock_id,
                "geometry_type_id": (
                    HD_MAP_GEOMETRY_TYPE_TO_ID["POLYLINE"]
                    if layer_name in line_layers
                    else HD_MAP_GEOMETRY_TYPE_TO_ID["POLYGON"]
                ),
                "has_traffic_lights": _has_traffic_lights(map_obj, layer_name, stop_line_subtype_id),
                "stop_line_subtype_id": stop_line_subtype_id,
                "speed_limit_mps": _safe_speed_limit_mps(map_obj),
            }

            if layer_name in line_layers:
                baseline_discrete_path = map_obj.baseline_path.discrete_path
                baseline_global_points = _discrete_path_to_global_points(baseline_discrete_path)
                baseline_headings = np.array([point.heading for point in baseline_discrete_path], dtype=np.float32)
                item["sampled_vxvy"] = _sample_direction_vectors(
                    global_points=baseline_global_points,
                    headings=baseline_headings,
                    num_points=points_per_polyline,
                    ego_global_pose=ego_global_pose,
                )
            else:
                item["sampled_vxvy"] = np.zeros((points_per_polyline, 2), dtype=np.float32)

            nearby_objects.append(item)

    nearby_objects.sort(key=lambda item: float(item["distance"]))
    if route_roadblock_ids:
        on_route_roadblocks = set(prune_route_by_connectivity(list(route_roadblock_ids), extracted_roadblock_ids))
        use_unknown_route_label = False
    else:
        on_route_roadblocks = set()
        use_unknown_route_label = True

    polyline_entries: List[Dict[str, object]] = []
    for item in nearby_objects:
        map_obj = item["map_obj"]
        layer_name = item["layer_name"]
        if layer_name == SemanticMapLayer.LANE:
            if include_baseline:
                polyline_entries.append(
                    {
                        **item,
                        "element_type_id": HD_MAP_ELEMENT_TYPE_TO_ID["LANE_BASELINE"],
                        "global_points": _discrete_path_to_global_points(map_obj.baseline_path.discrete_path),
                    }
                )
            if include_left:
                polyline_entries.append(
                    {
                        **item,
                        "element_type_id": HD_MAP_ELEMENT_TYPE_TO_ID["LANE_LEFT_BOUNDARY"],
                        "global_points": _discrete_path_to_global_points(map_obj.left_boundary.discrete_path),
                    }
                )
            if include_right:
                polyline_entries.append(
                    {
                        **item,
                        "element_type_id": HD_MAP_ELEMENT_TYPE_TO_ID["LANE_RIGHT_BOUNDARY"],
                        "global_points": _discrete_path_to_global_points(map_obj.right_boundary.discrete_path),
                    }
                )
            continue

        if layer_name == SemanticMapLayer.LANE_CONNECTOR:
            if include_baseline:
                polyline_entries.append(
                    {
                        **item,
                        "element_type_id": HD_MAP_ELEMENT_TYPE_TO_ID["LANE_CONNECTOR_BASELINE"],
                        "global_points": _discrete_path_to_global_points(map_obj.baseline_path.discrete_path),
                    }
                )
            if include_left:
                polyline_entries.append(
                    {
                        **item,
                        "element_type_id": HD_MAP_ELEMENT_TYPE_TO_ID["LANE_CONNECTOR_LEFT_BOUNDARY"],
                        "global_points": _discrete_path_to_global_points(map_obj.left_boundary.discrete_path),
                    }
                )
            if include_right:
                polyline_entries.append(
                    {
                        **item,
                        "element_type_id": HD_MAP_ELEMENT_TYPE_TO_ID["LANE_CONNECTOR_RIGHT_BOUNDARY"],
                        "global_points": _discrete_path_to_global_points(map_obj.right_boundary.discrete_path),
                    }
                )
            continue

        element_type_id = HD_MAP_POLYGON_LAYER_TO_ELEMENT_TYPE.get(layer_name)
        if element_type_id is None:
            continue
        polyline_entries.append(
            {
                **item,
                "element_type_id": element_type_id,
                "global_points": _polygon_to_global_points(map_obj),
            }
        )

    for polyline_idx, item in enumerate(polyline_entries[:max_polylines]):
        global_points = item["global_points"]
        if len(global_points) == 0:
            continue

        local_points = _global_to_local_points(global_points, ego_global_pose)
        sampled_points = _sample_polyline(local_points, num_points=points_per_polyline)

        if use_unknown_route_label:
            on_route_id = HD_MAP_ON_ROUTE_UNKNOWN
        elif item["roadblock_id"] is None:
            on_route_id = HD_MAP_ON_ROUTE_UNKNOWN
        else:
            on_route_id = (
                HD_MAP_ON_ROUTE_ON
                if item["roadblock_id"] in on_route_roadblocks
                else HD_MAP_ON_ROUTE_OFF
            )

        tensors["hd_map_coords"][polyline_idx] = torch.from_numpy(sampled_points)
        tensors["hd_map_vxvy"][polyline_idx] = torch.from_numpy(item["sampled_vxvy"])
        tensors["hd_map_valid_mask"][polyline_idx] = True
        tensors["hd_map_geometry_type_ids"][polyline_idx] = int(item["geometry_type_id"])
        tensors["hd_map_element_type_ids"][polyline_idx] = int(item["element_type_id"])
        tensors["hd_map_on_route_ids"][polyline_idx] = on_route_id
        tensors["hd_map_has_traffic_light_ids"][polyline_idx] = int(item["has_traffic_lights"])
        tensors["hd_map_stop_line_subtype_ids"][polyline_idx] = int(item["stop_line_subtype_id"])
        tensors["hd_map_speed_limit_mps"][polyline_idx] = float(item["speed_limit_mps"])

    return tensors
