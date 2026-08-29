# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import datetime
import json
import logging
import math
import os
import pathlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.b2d.common import ensure_external_paths
from scripts.b2d.drivor_b2d_config import (
    B2D_CAMERA_ORDER,
    build_b2d_ego_status_vector,
    preprocess_b2d_camera_images,
)

ensure_external_paths()

import carla
from leaderboard.autoagents import autonomous_agent
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from omegaconf import OmegaConf
from scripts.b2d.pid_controller import PIDController
from scripts.b2d.route_planner import RoutePlanner

SAVE_PATH = os.environ.get("SAVE_PATH")
IS_BENCH2DRIVE = os.environ.get("IS_BENCH2DRIVE")
TEAM_AGENT = os.environ.get("TEAM_AGENT", "")
DRIVOR_B2D_SAVE_CAMERA_FRAMES = os.environ.get("DRIVOR_B2D_SAVE_CAMERA_FRAMES", "0")
DRIVOR_B2D_SAVE_TRAJECTORY_OVERLAYS = os.environ.get("DRIVOR_B2D_SAVE_TRAJECTORY_OVERLAYS", "0")
DRIVOR_B2D_SAVE_FRONT_CAMERA_ONLY = os.environ.get("DRIVOR_B2D_SAVE_FRONT_CAMERA_ONLY", "0")

logger = logging.getLogger(__name__)


def _env_flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_entry_point():
    return "drivorAgent"


class drivorAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController()

        conf_parts = path_to_conf_file.split("+")
        self.config_path = conf_parts[0]
        self.ckpt_path = conf_parts[1] if len(conf_parts) > 1 else ""
        if not self.ckpt_path:
            raise ValueError("DrivoR B2D agent requires a checkpoint path in TEAM_CONFIG.")

        if IS_BENCH2DRIVE and len(conf_parts) > 2:
            self.save_name = conf_parts[-1]
        else:
            now = datetime.datetime.now()
            self.save_name = "_".join(map(lambda x: f"{x:02d}", (now.month, now.day, now.hour, now.minute, now.second)))

        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent_cfg_path = Path(
            os.environ.get(
                "DRIVOR_B2D_AGENT_CONFIG",
                REPO_ROOT / "navsim/planning/script/config/common/agent/drivoR_b2d.yaml",
            )
        )
        agent_cfg = OmegaConf.load(agent_cfg_path)
        self.camera_names = B2D_CAMERA_ORDER
        self.model_image_width = int(agent_cfg.config.image_size[0])
        self.model_image_height = int(agent_cfg.config.image_size[1])
        self.sensor_width = self.model_image_width
        self.sensor_height = self.model_image_height
        self.model = DrivoRAgent(
            config=agent_cfg.config,
            lr_args=agent_cfg.lr_args,
            checkpoint_path=self.ckpt_path,
            scheduler_args=None,
            batch_size=1,
            num_gpus=1,
            progress_bar=False,
        )
        self.model.initialize()
        self.model.to(self.device)
        self.model.eval()

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self._warned_invalid_imu = False
        self.scenario_town_name = getattr(self, "scenario_town_name", None)
        self.save_path = None
        self.save_camera_frames = _env_flag(DRIVOR_B2D_SAVE_CAMERA_FRAMES)
        self.save_trajectory_overlays = _env_flag(DRIVOR_B2D_SAVE_TRAJECTORY_OVERLAYS)
        self.save_front_camera_only = _env_flag(DRIVOR_B2D_SAVE_FRONT_CAMERA_ONLY)
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        self.prev_control = control
        self.prev_control_cache = []
        if SAVE_PATH is not None:
            self.save_path = pathlib.Path(SAVE_PATH) / self.save_name
            self.save_path.mkdir(parents=True, exist_ok=True)
            (self.save_path / "meta").mkdir(exist_ok=True)
            if self.save_camera_frames:
                for camera_dir in self._camera_output_dirs().values():
                    (self.save_path / camera_dir).mkdir(exist_ok=True)
            if self.save_trajectory_overlays:
                for overlay_dir in self._overlay_output_dirs().values():
                    (self.save_path / overlay_dir).mkdir(exist_ok=True)

        self.lidar2img = {
            "CAM_FRONT": np.array(
                [
                    [1.14251841e03, 8.00000000e02, 0.00000000e00, -9.52000000e02],
                    [0.00000000e00, 4.50000000e02, -1.14251841e03, -8.09704417e02],
                    [0.00000000e00, 1.00000000e00, 0.00000000e00, -1.19000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_FRONT_LEFT": np.array(
                [
                    [6.03961325e-14, 1.39475744e03, 0.00000000e00, -9.20539908e02],
                    [-3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                    [-8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_FRONT_RIGHT": np.array(
                [
                    [1.31064327e03, -4.77035138e02, 0.00000000e00, -4.06010608e02],
                    [3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                    [8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_BACK": np.array(
                [
                    [-5.60166031e02, -8.00000000e02, 0.00000000e00, -1.28800000e03],
                    [5.51091060e-14, -4.50000000e02, -5.60166031e02, -8.58939847e02],
                    [1.22464680e-16, -1.00000000e00, 0.00000000e00, -1.61000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
        }
        self.lidar2cam = {
            "CAM_FRONT": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, 1.0, 0.0, -1.19], [0.0, 0.0, 0.0, 1.0]]),
            "CAM_FRONT_LEFT": np.array(
                [[0.57357644, 0.81915204, 0.0, -0.22517331], [0.0, 0.0, -1.0, -0.24], [-0.81915204, 0.57357644, 0.0, -0.82909407], [0.0, 0.0, 0.0, 1.0]]
            ),
            "CAM_FRONT_RIGHT": np.array(
                [[0.57357644, -0.81915204, 0.0, 0.22517331], [0.0, 0.0, -1.0, -0.24], [0.81915204, 0.57357644, 0.0, -0.82909407], [0.0, 0.0, 0.0, 1.0]]
            ),
            "CAM_BACK": np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, -1.0, 0.0, -1.61], [0.0, 0.0, 0.0, 1.0]]),
        }
        self.lidar2ego = np.array([[0.0, 1.0, 0.0, -0.39], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0.0, 0.0, 0.0, 1.0]])

        scale_factor = np.eye(4)
        scale_factor[0, 0] *= self.sensor_width / 1600.0
        scale_factor[1, 1] *= self.sensor_height / 900.0
        for key, lidar2img in self.lidar2img.items():
            self.lidar2img[key] = scale_factor @ lidar2img

    def _init(self):
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]["lon"]
            lat = self._global_plan[0][0]["lat"]
            earth_radius = 6378137.0

            def equations(values):
                x, y = values
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * 180) / (math.pi * earth_radius) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * earth_radius * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * earth_radius * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]

            from scipy.optimize import fsolve

            self.lat_ref, self.lon_ref = fsolve(equations, [0, 0])
        except Exception as error:
            print(error, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def _resolve_scenario_town_name(self):
        if self.scenario_town_name is not None:
            return
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

            carla_map = CarlaDataProvider.get_map()
            if carla_map is not None:
                self.scenario_town_name = carla_map.name.split("/")[-1]
        except Exception:
            logger.debug("Could not resolve scenario town from CARLA world.", exc_info=True)

    def sensors(self):
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": self.sensor_width,
                "height": self.sensor_height,
                "fov": 70,
                "id": "CAM_FRONT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.27,
                "y": -0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -55.0,
                "width": self.sensor_width,
                "height": self.sensor_height,
                "fov": 70,
                "id": "CAM_FRONT_LEFT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.27,
                "y": 0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 55.0,
                "width": self.sensor_width,
                "height": self.sensor_height,
                "fov": 70,
                "id": "CAM_FRONT_RIGHT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": -2.0,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": self.sensor_width,
                "height": self.sensor_height,
                "fov": 110,
                "id": "CAM_BACK",
            },
            {
                "type": "sensor.other.imu",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "IMU",
            },
            {
                "type": "sensor.other.gnss",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "GPS",
            },
            ### Debug sensor, not used by the model
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 50.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'TOP_DOWN'
                },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        # match training distribution by applying JPEG compression
        for cam in [*self.camera_names, "TOP_DOWN"]:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode(".jpg", img, encode_param)
            imgs[cam] = cv2.imdecode(img, cv2.IMREAD_COLOR)

        gps = input_data["GPS"][1][:2]
        speed = input_data["SPEED"][1]["speed"]
        imu_values = np.asarray(input_data["IMU"][1], dtype=np.float32)
        compass = float(imu_values[-1])
        acceleration = imu_values[:3].copy()
        angular_velocity = imu_values[3:6].copy()

        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)

        if np.isnan(imu_values).any():
            if not self._warned_invalid_imu:
                logger.warning(
                    "Received NaN values in the B2D IMU packet; replacing NaNs with zero fallback values."
                )
                self._warned_invalid_imu = True
            compass = float(np.nan_to_num(compass, nan=0.0))
            acceleration = np.nan_to_num(acceleration, nan=0.0)
            angular_velocity = np.nan_to_num(angular_velocity, nan=0.0)

        return {
            "imgs": imgs,
            "gps": gps,
            "pos": pos,
            "speed": speed,
            "compass": compass,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "command_near": near_command,
            "command_near_xy": near_node,
        }

    def process_input(self, images, speed, ego_accel, local_command_xy, ego_fut_cmd, ego_global_pose=None):
        ego_status = build_b2d_ego_status_vector(
            speed,
            ego_accel[:2],
            local_command_xy,
            ego_fut_cmd,
            warning_context="Closed-loop B2D ego_status",
        )[None, None]
        camera_feature = preprocess_b2d_camera_images(
            [images[cam] for cam in self.camera_names],
            image_size=(self.model_image_width, self.model_image_height),
        )[None]
        features = {
            "ego_status": ego_status.to(self.device),
            "camera_feature": camera_feature.to(self.device),
        }
        if ego_global_pose is not None:
            features["ego_global_pose"] = torch.as_tensor(
                ego_global_pose,
                dtype=torch.float64,
                device=self.device,
            )[None]
        if self.scenario_town_name is not None:
            features["scenario_town_name"] = self.scenario_town_name
        return features

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        del timestamp
        if not self.initialized:
            self._init()
        self._resolve_scenario_town_name()

        tick_data = self.tick(input_data)
        raw_theta = tick_data["compass"] if not np.isnan(tick_data["compass"]) else 0
        ego_theta = -raw_theta + np.pi / 2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data["pos"][0]
        can_bus[1] = -tick_data["pos"][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data["speed"]
        can_bus[10:13] = tick_data["acceleration"]
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data["angular_velocity"]
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180

        command = tick_data["command_near"]
        if command < 0:
            command = 4
        command -= 1
        command_onehot = np.zeros(6)
        command_onehot[command] = 1

        command_near_xy = np.array(
            [tick_data["command_near_xy"][0] - can_bus[0], -tick_data["command_near_xy"][1] - can_bus[1]]
        )
        rotation_matrix = np.array(
            [[np.cos(raw_theta), -np.sin(raw_theta)], [np.sin(raw_theta), np.cos(raw_theta)]]
        )
        local_command_xy = rotation_matrix @ command_near_xy
        ego_global_pose = np.array([can_bus[0], can_bus[1], ego_theta], dtype=np.float64)

        features = self.process_input(
            tick_data["imgs"],
            tick_data["speed"],
            tick_data["acceleration"],
            local_command_xy,
            command_onehot,
            ego_global_pose=ego_global_pose,
        )
        output_data_batch = self.model(features)

        out_traj = output_data_batch["trajectory"].detach().cpu().numpy()[0][:, :2]
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
            out_traj, tick_data["speed"], local_command_xy
        )

        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata["agent"] = "only_traj"
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)
        self.pid_metadata["steer"] = control.steer
        self.pid_metadata["throttle"] = control.throttle
        self.pid_metadata["brake"] = control.brake
        self.pid_metadata["steer_traj"] = float(steer_traj)
        self.pid_metadata["throttle_traj"] = float(throttle_traj)
        self.pid_metadata["brake_traj"] = float(brake_traj)
        self.pid_metadata["plan"] = out_traj.tolist()
        self.pid_metadata["command"] = command

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if self.save_path is not None:
            self.save(tick_data, out_traj)

        self.prev_control = control
        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def save(self, tick_data, ego_traj):
        save_path = self.save_path
        if save_path is None:
            return

        frame = self.step // 10
        self.pid_metadata["plan"] = ego_traj.tolist()

        if self.step % 10 == 0:
            if self.save_camera_frames:
                self._save_camera_frames(tick_data, frame)
            if self.save_trajectory_overlays:
                self._save_trajectory_overlays(tick_data, ego_traj, frame)

            with open(save_path / "meta" / f"{frame:04d}.json", "w") as outfile:
                json.dump(self.pid_metadata, outfile, indent=4)
        with open(save_path / "metric_info.json", "w") as outfile:
            json.dump(self.metric_info, outfile, indent=4)

    def _camera_output_dirs(self):
        camera_output_dirs = {
            "CAM_FRONT": "rgb_front",
            "CAM_FRONT_LEFT": "rgb_front_left",
            "CAM_FRONT_RIGHT": "rgb_front_right",
            "CAM_BACK": "rgb_back",
            "TOP_DOWN": "rgb_top_down",
        }
        if self.save_front_camera_only:
            return {"CAM_FRONT": camera_output_dirs["CAM_FRONT"]}
        return camera_output_dirs

    def _overlay_output_dirs(self):
        overlay_output_dirs = {
            "CAM_FRONT": "overlay_front",
            "CAM_FRONT_LEFT": "overlay_front_left",
            "CAM_FRONT_RIGHT": "overlay_front_right",
            "CAM_BACK": "overlay_back",
        }
        if self.save_front_camera_only:
            return {"CAM_FRONT": overlay_output_dirs["CAM_FRONT"]}
        return overlay_output_dirs

    def _save_camera_frames(self, tick_data, frame):
        save_path = self.save_path
        if save_path is None:
            return

        for camera_name, directory_name in self._camera_output_dirs().items():
            Image.fromarray(tick_data["imgs"][camera_name]).save(save_path / directory_name / f"{frame:04d}.png")

    def _save_trajectory_overlays(self, tick_data, ego_traj, frame):
        save_path = self.save_path
        if save_path is None:
            return

        for camera_name, directory_name in self._overlay_output_dirs().items():
            overlay = self._draw_trajectory_on_camera(
                ego_traj,
                tick_data["imgs"][camera_name],
                self.lidar2img[camera_name],
            )
            Image.fromarray(overlay).save(save_path / directory_name / f"{frame:04d}.png")

    def _draw_trajectory_on_camera(
        self,
        trajectory,
        raw_img,
        lidar2img_rt,
        canvas_size=None,
        color=(222, 112, 97),
        thickness=3,
    ):
        if canvas_size is None:
            canvas_size = (self.sensor_height, self.sensor_width)

        image = raw_img.copy()
        line = np.concatenate([np.zeros((1, 2), dtype=trajectory.dtype), trajectory], axis=0)
        pts_4d = np.stack(
            [line[:, 0], line[:, 1], np.full(line.shape[0], -1.84), np.ones(line.shape[0])],
            axis=0,
        )
        pts_2d = (lidar2img_rt @ pts_4d).T
        positive_depth = pts_2d[:, 2] > 1e-5
        if not positive_depth.any():
            return image

        pts_2d = pts_2d[positive_depth]
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        in_view = (
            (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 0] < canvas_size[1])
            & (pts_2d[:, 1] >= 0)
            & (pts_2d[:, 1] < canvas_size[0])
        )
        points = pts_2d[in_view, :2]
        if len(points) < 2:
            return image

        smoothed_pts = points.astype(int)
        for start, end in zip(smoothed_pts[:-1], smoothed_pts[1:]):
            cv2.line(image, tuple(start), tuple(end), color=color, thickness=thickness)
            cv2.circle(image, tuple(end), thickness + 1, color, -1)
            if thickness == 3:
                cv2.circle(image, tuple(end), thickness + 2, (0, 0, 0), 0)
        return image

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        earth_radius = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (earth_radius * scale)
        mx = (lon * (math.pi * earth_radius * scale)) / 180.0
        y = scale * earth_radius * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * earth_radius / 180.0
        return np.array([x, y])
