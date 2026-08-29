# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch
from shapely import Point
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree
import warnings

from .score_aggregation import (
    aggregate_multiplicative_weighted_score,
    apply_multiplicative_metric_weight,
)


def _safe_project(line: LineString, point: Point) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in line_locate_point", category=RuntimeWarning
        )
        return float(line.project(point))


def compute_corners_torch(proposals):
    headings = proposals[..., 2]
    cos_yaw = torch.cos(headings)
    sin_yaw = torch.sin(headings)

    x = proposals[..., 0] + 0.39 * cos_yaw
    y = proposals[..., 1] + 0.39 * sin_yaw
    half_length = 2.042 + torch.zeros_like(headings)
    half_width = 0.925 + torch.zeros_like(headings)

    cos_yaw = cos_yaw[..., None]
    sin_yaw = sin_yaw[..., None]

    corners_x = torch.stack([half_length, -half_length, -half_length, half_length], dim=-1)
    corners_y = torch.stack([half_width, half_width, -half_width, -half_width], dim=-1)

    rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y
    corners = torch.stack((rot_corners_x + x[..., None], rot_corners_y + y[..., None]), dim=-1)
    return corners


def evaluate_coll(fut_box_corners, ego_coords):
    n_future = ego_coords.shape[1]
    num_proposals = ego_coords.shape[0]
    fut_mask = fut_box_corners.any(-1).any(-1)

    ego_polygons_all = np.empty(ego_coords.shape[:3], dtype=object)
    for proposal_idx in range(ego_coords.shape[0]):
        for time_idx in range(ego_coords.shape[1]):
            for ttc_idx in range(ego_coords.shape[2]):
                ego_polygons_all[proposal_idx, time_idx, ttc_idx] = Polygon(ego_coords[proposal_idx, time_idx, ttc_idx])

    proposal_fault_collided_track_ids = {proposal_idx: [] for proposal_idx in range(num_proposals)}
    ttc_collided_track_ids = {proposal_idx: [] for proposal_idx in range(num_proposals)}

    key_agent_corners = np.zeros([num_proposals, 6, 2, 4, 2], dtype=np.float32)
    key_agent_labels = np.zeros([num_proposals, 6, 2], dtype=bool)
    collision_all = np.zeros([num_proposals, n_future], dtype=bool)
    ttc_collision_all = np.zeros([num_proposals, n_future], dtype=bool)

    for time_idx in range(n_future):
        geometries = fut_box_corners[:, time_idx][fut_mask[:, time_idx]]
        polygons = [Polygon(geometry) for geometry in geometries]
        if not polygons:
            continue
        str_tree = STRtree(polygons)
        token_list = np.arange(len(fut_box_corners))[fut_mask[:, time_idx]]

        ego_polygons = ego_polygons_all[:, time_idx, 0]
        intersecting = str_tree.query(ego_polygons, predicate="intersects")
        for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
            token = token_list[geometry_idx]
            if token in proposal_fault_collided_track_ids[proposal_idx]:
                continue
            proposal_fault_collided_track_ids[proposal_idx].append(token)
            collision_all[proposal_idx, time_idx] = True
            key_agent_labels[proposal_idx, : time_idx + 1, 0] = fut_mask[token][: time_idx + 1]
            key_agent_corners[proposal_idx, : time_idx + 1, 0] = fut_box_corners[token][: time_idx + 1]

        for ttc_idx in [1, 2]:
            ego_polygons = ego_polygons_all[:, time_idx, ttc_idx]
            intersecting = str_tree.query(ego_polygons, predicate="intersects")
            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = token_list[geometry_idx]
                if token in ttc_collided_track_ids[proposal_idx]:
                    continue
                ttc_collided_track_ids[proposal_idx].append(token)
                ttc_collision_all[proposal_idx, time_idx] = True
                key_agent_labels[proposal_idx, : time_idx + 1, 1] = fut_mask[token][: time_idx + 1]
                key_agent_corners[proposal_idx, : time_idx + 1, 1] = fut_box_corners[token][: time_idx + 1]

    return collision_all, ttc_collision_all, key_agent_corners[:-1], key_agent_labels[:-1]


def get_scores(args):
    return [
        get_sub_score(
            a["fut_box_corners"],
            a["_ego_coords"],
            a["proposal"],
            a["target_traj"],
            a["comfort"],
            a["ego_areas"],
            a.get("noc_weight", 1.0),
            a.get("dac_weight", 1.0),
            a.get("ttc_weight", 5.0),
            a.get("ep_weight", 5.0),
            a.get("comfort_weight", 2.0),
            a.get("enable_collision_gate_for_ep", True),
            a.get("enable_drivable_gate_for_ep", True),
        )
        for a in args
    ]


def get_sub_score(
    fut_box_corners,
    ego_coords,
    proposals,
    target_traj,
    comfort,
    ego_areas,
    noc_weight=1.0,
    dac_weight=1.0,
    ttc_weight=5.0,
    ep_weight=5.0,
    comfort_weight=2.0,
    enable_collision_gate_for_ep=True,
    enable_drivable_gate_for_ep=True,
):
    collisions, ttc_collision, key_agent_corners, key_agent_labels = evaluate_coll(
        fut_box_corners, ego_coords
    )

    collisions = collisions[:-1] & (~collisions[-1:])
    ttc_collision = ttc_collision[:-1] & (~ttc_collision[-1:])

    collision = (1 - collisions.any(-1)).astype(np.float32)
    ttc = (1 - ttc_collision.any(-1)).astype(np.float32)

    on_road_all = ego_areas[:-1, :, 1]
    on_route_all = ego_areas[:-1, :, 2]
    drivable_area_compliance = (on_road_all.all(-1) & on_route_all.any(-1)).astype(np.float32)
    ego_areas = np.stack([on_road_all, on_route_all], axis=-1)
    ep_collision_gate = collision if enable_collision_gate_for_ep else np.ones_like(collision)
    ep_drivable_gate = (
        drivable_area_compliance
        if enable_drivable_gate_for_ep
        else np.ones_like(drivable_area_compliance)
    )

    target_line = np.concatenate([np.zeros([1, 2], dtype=np.float32), target_traj[..., :2]])
    valid_target_line = np.isfinite(target_line).all() and np.unique(target_line, axis=0).shape[0] > 1

    raw_progress = np.ones([len(proposals)], dtype=np.float32)
    if valid_target_line:
        centerline = LineString(target_line)
        target_progress = _safe_project(centerline, Point(target_line[-1]))
        valid_target_line = np.isfinite(target_progress) and centerline.length > 0.0

    if valid_target_line:
        for proposal_idx, proposal in enumerate(proposals[..., :2]):
            end_point = Point(proposal[-1])
            proj_progress = _safe_project(centerline, end_point)
            if not np.isfinite(proj_progress):
                proj_progress = float(np.linalg.norm(proposal[-1] - target_line[0]))
            if proj_progress == target_progress:
                proj_progress = proj_progress + np.linalg.norm(proposal[-1] - target_traj[-1][:2])
            raw_progress[proposal_idx] = proj_progress
    else:
        target_progress = float(np.linalg.norm(target_traj[-1][:2]))
        for proposal_idx, proposal in enumerate(proposals[..., :2]):
            raw_progress[proposal_idx] = float(np.linalg.norm(proposal[-1]))

    raw_progress = np.clip(raw_progress, a_min=0, a_max=None)
    multiplicative_metric_scores = ep_collision_gate * ep_drivable_gate
    max_raw_progress = np.maximum(raw_progress, target_progress) + 0.01
    min_raw_progress = np.minimum(raw_progress, target_progress) + 0.01
    progress_ratio = min_raw_progress / max_raw_progress
    progress = multiplicative_metric_scores * progress_ratio
    comfort = comfort.astype(np.float32, copy=False)

    final_scores = aggregate_multiplicative_weighted_score(
        multiplicative_metrics=[
            apply_multiplicative_metric_weight(collision, noc_weight),
            apply_multiplicative_metric_weight(drivable_area_compliance, dac_weight),
        ],
        weighted_metrics=[ttc, progress, comfort],
        weighted_metric_weights=[ttc_weight, ep_weight, comfort_weight],
    )
    target_scores = np.stack(
        [collision, drivable_area_compliance, progress, ttc, comfort, final_scores], axis=-1
    ).astype(np.float32)

    return target_scores, key_agent_corners, key_agent_labels, ego_areas
