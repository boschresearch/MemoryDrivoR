# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterable, Optional

import numpy as np

EARTH_RADIUS_EQUATOR_M = 6378137.0


class RoutePlanner:
    def __init__(
        self,
        min_distance: float,
        max_distance: float,
        debug_size: int = 256,
        lat_ref: float = 42.0,
        lon_ref: float = 2.0,
    ) -> None:
        del debug_size
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.lat_ref = float(lat_ref)
        self.lon_ref = float(lon_ref)
        self.route: Deque[tuple] = deque()

    def _to_local_xy(self, position, gps: bool) -> np.ndarray:
        if gps:
            return self.gps_to_location(np.asarray([position["lat"], position["lon"]], dtype=np.float64))
        return np.asarray([position.location.x, position.location.y], dtype=np.float64)

    def set_route(self, global_plan, gps: bool = False, global_plan_world: Optional[Iterable] = None) -> None:
        self.route.clear()
        if global_plan_world is None:
            for position, command in global_plan:
                self.route.append((self._to_local_xy(position, gps), command))
            return

        for (position, command), (world_position, _) in zip(global_plan, global_plan_world):
            self.route.append((self._to_local_xy(position, gps), command, world_position))

    def run_step(self, ego_xy):
        if not self.route:
            raise RuntimeError("RoutePlanner.run_step called before set_route.")
        if len(self.route) == 1:
            return self.route[0]

        ego_xy = np.asarray(ego_xy, dtype=np.float64)
        pop_until = 0
        farthest_close_distance = -math.inf
        traversed = 0.0

        for idx in range(1, len(self.route)):
            if traversed > self.max_distance:
                break

            prev_xy = self.route[idx - 1][0]
            curr_xy = self.route[idx][0]
            traversed += float(np.linalg.norm(curr_xy - prev_xy))
            distance_to_ego = float(np.linalg.norm(curr_xy - ego_xy))
            if distance_to_ego <= self.min_distance and distance_to_ego > farthest_close_distance:
                farthest_close_distance = distance_to_ego
                pop_until = idx

        for _ in range(pop_until):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]

    def gps_to_location(self, gps):
        lat, lon = np.asarray(gps, dtype=np.float64)[:2]
        lat_ref_rad = math.radians(self.lat_ref)
        scale = math.cos(lat_ref_rad)

        lon_m = math.radians(lon) * EARTH_RADIUS_EQUATOR_M * scale
        ref_lon_m = math.radians(self.lon_ref) * EARTH_RADIUS_EQUATOR_M * scale
        mercator_y = math.log(math.tan(math.radians(lat + 90.0) / 2.0)) * EARTH_RADIUS_EQUATOR_M * scale
        ref_y = math.log(math.tan(math.radians(self.lat_ref + 90.0) / 2.0)) * EARTH_RADIUS_EQUATOR_M * scale

        return np.asarray([lon_m - ref_lon_m, ref_y - mercator_y], dtype=np.float64)
