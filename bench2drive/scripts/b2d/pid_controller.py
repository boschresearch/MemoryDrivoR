# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Tuple

import numpy as np


def _as_float(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


class _RollingPID:
    def __init__(self, kp: float, ki: float, kd: float, window: int) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        window = max(1, int(window))
        self.errors: Deque[float] = deque([0.0] * window, maxlen=window)

    def step(self, error: float) -> float:
        error = float(error)
        prev_error = self.errors[-1] if self.errors else error
        self.errors.append(error)
        integral = float(np.mean(self.errors)) if self.errors else 0.0
        derivative = error - prev_error
        return self.kp * error + self.ki * integral + self.kd * derivative


class PIDController:
    def __init__(
        self,
        turn_KP: float = 0.75,
        turn_KI: float = 0.75,
        turn_KD: float = 0.3,
        turn_n: int = 40,
        speed_KP: float = 5.0,
        speed_KI: float = 0.5,
        speed_KD: float = 1.0,
        speed_n: int = 40,
        max_throttle: float = 0.75,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta: float = 0.25,
        aim_dist: float = 4.0,
        angle_thresh: float = 0.3,
        dist_thresh: float = 10.0,
    ) -> None:
        self.turn = _RollingPID(turn_KP, turn_KI, turn_KD, turn_n)
        self.speed = _RollingPID(speed_KP, speed_KI, speed_KD, speed_n)
        self.max_throttle = float(max_throttle)
        self.brake_speed = float(brake_speed)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.aim_dist = float(aim_dist)
        self.angle_thresh = float(angle_thresh)
        self.dist_thresh = float(dist_thresh)

    @staticmethod
    def _steering_angle(point: Iterable[float]) -> float:
        point = np.asarray(point, dtype=np.float64)
        if point.shape[0] < 2:
            return 0.0
        return float(np.arctan2(point[0], point[1]) / (np.pi / 2.0))

    def _select_aim_point(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(waypoints) == 0:
            return np.zeros(2, dtype=np.float64), 0.0
        if len(waypoints) == 1:
            return waypoints[0], 0.0

        segment_midpoints = 0.5 * (waypoints[:-1] + waypoints[1:])
        midpoint_distances = np.linalg.norm(segment_midpoints, axis=1)
        aim_index = int(np.argmin(np.abs(midpoint_distances - self.aim_dist)))
        segment_lengths = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
        desired_speed = float(segment_lengths.mean() * 2.0)
        return waypoints[aim_index], desired_speed

    def control_pid(self, waypoints, speed, target):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] < 2:
            raise ValueError(f"Expected waypoints with shape [N, 2+], got {waypoints.shape}.")
        waypoints = waypoints[:, :2]
        target = np.asarray(target, dtype=np.float64)[:2]
        speed_value = _as_float(speed)

        aim, desired_speed = self._select_aim_point(waypoints)
        last_segment = waypoints[-1] - waypoints[-2] if len(waypoints) >= 2 else aim

        aim_angle = self._steering_angle(aim)
        last_angle = self._steering_angle(last_segment)
        target_angle = self._steering_angle(target)

        prefer_target = abs(target_angle) < abs(aim_angle)
        prefer_target = prefer_target or (
            abs(target_angle - last_angle) > self.angle_thresh and target[1] < self.dist_thresh
        )
        steering_error = target_angle if prefer_target else aim_angle

        steer = float(np.clip(self.turn.step(steering_error), -1.0, 1.0))
        should_brake = desired_speed < self.brake_speed
        if desired_speed > 1e-6:
            should_brake = should_brake or speed_value > desired_speed * self.brake_ratio

        speed_error = float(np.clip(desired_speed - speed_value, 0.0, self.clip_delta))
        throttle = float(np.clip(self.speed.step(speed_error), 0.0, self.max_throttle))
        if should_brake:
            throttle = 0.0

        metadata = {
            "speed": speed_value,
            "steer": steer,
            "throttle": throttle,
            "brake": float(should_brake),
            "desired_speed": desired_speed,
            "angle": aim_angle,
            "angle_last": last_angle,
            "angle_target": target_angle,
            "angle_final": steering_error,
            "delta": speed_error,
            "aim": tuple(float(v) for v in aim),
            "target": tuple(float(v) for v in target),
        }
        for idx in range(min(4, len(waypoints))):
            metadata[f"wp_{idx + 1}"] = tuple(float(v) for v in waypoints[idx])

        return steer, throttle, bool(should_brake), metadata
