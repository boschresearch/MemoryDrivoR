# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Repo-local Bench2Drive constants for the DrivoR B2D integration."""

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import logging
import numpy as np
import torch
from PIL import Image

point_cloud_range = [-64.0, -64.0, -2.0, 64.0, 64.0, 2.0]

class_names = [
	"car",
	"van",
	"truck",
	"bicycle",
	"traffic_sign",
	"traffic_cone",
	"traffic_light",
	"pedestrian",
	"others",
]

eval_cfg = {
	"dist_ths": [0.5, 1.0, 2.0, 4.0],
	"dist_th_tp": 2.0,
	"min_recall": 0.1,
	"min_precision": 0.1,
	"mean_ap_weight": 5,
	"class_names": [
		"car",
		"van",
		"truck",
		"bicycle",
		"traffic_sign",
		"traffic_cone",
		"traffic_light",
		"pedestrian",
	],
	"tp_metrics": ["trans_err", "scale_err", "orient_err", "vel_err"],
	"err_name_maping": {
		"trans_err": "mATE",
		"scale_err": "mASE",
		"orient_err": "mAOE",
		"vel_err": "mAVE",
		"attr_err": "mAAE",
	},
	"class_range": {
		"car": (50, 50),
		"van": (50, 50),
		"truck": (50, 50),
		"bicycle": (40, 40),
		"traffic_sign": (30, 30),
		"traffic_cone": (30, 30),
		"traffic_light": (30, 30),
		"pedestrian": (40, 40),
	},
}

B2D_CAMERA_ORDER = (
	"CAM_FRONT",
	"CAM_FRONT_LEFT",
	"CAM_FRONT_RIGHT",
	"CAM_BACK",
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_VALID_B2D_ACCELERATION_ABS = 50.0

logger = logging.getLogger(__name__)

NameMapping = {
	"vehicle.bh.crossbike": "bicycle",
	"vehicle.diamondback.century": "bicycle",
	"vehicle.gazelle.omafiets": "bicycle",
	"vehicle.audi.etron": "car",
	"vehicle.chevrolet.impala": "car",
	"vehicle.dodge.charger_2020": "car",
	"vehicle.dodge.charger_police": "car",
	"vehicle.dodge.charger_police_2020": "car",
	"vehicle.lincoln.mkz_2017": "car",
	"vehicle.lincoln.mkz_2020": "car",
	"vehicle.mini.cooper_s_2021": "car",
	"vehicle.mercedes.coupe_2020": "car",
	"vehicle.ford.mustang": "car",
	"vehicle.nissan.patrol_2021": "car",
	"vehicle.audi.tt": "car",
	"vehicle.ford.crown": "car",
	"vehicle.tesla.model3": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": "car",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": "car",
	"vehicle.ford.ambulance": "van",
	"/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
	"vehicle.carlamotors.firetruck": "truck",
	"traffic.speed_limit.30": "traffic_sign",
	"traffic.speed_limit.40": "traffic_sign",
	"traffic.speed_limit.50": "traffic_sign",
	"traffic.speed_limit.60": "traffic_sign",
	"traffic.speed_limit.90": "traffic_sign",
	"traffic.speed_limit.120": "traffic_sign",
	"traffic.stop": "traffic_sign",
	"traffic.yield": "traffic_sign",
	"traffic.traffic_light": "traffic_light",
	"static.prop.warningconstruction": "traffic_cone",
	"static.prop.warningaccident": "traffic_cone",
	"static.prop.trafficwarning": "traffic_cone",
	"static.prop.constructioncone": "traffic_cone",
	"walker.pedestrian.0001": "pedestrian",
	"walker.pedestrian.0003": "pedestrian",
	"walker.pedestrian.0004": "pedestrian",
	"walker.pedestrian.0005": "pedestrian",
	"walker.pedestrian.0007": "pedestrian",
	"walker.pedestrian.0010": "pedestrian",
	"walker.pedestrian.0013": "pedestrian",
	"walker.pedestrian.0014": "pedestrian",
	"walker.pedestrian.0015": "pedestrian",
	"walker.pedestrian.0016": "pedestrian",
	"walker.pedestrian.0017": "pedestrian",
	"walker.pedestrian.0018": "pedestrian",
	"walker.pedestrian.0019": "pedestrian",
	"walker.pedestrian.0020": "pedestrian",
	"walker.pedestrian.0021": "pedestrian",
	"walker.pedestrian.0022": "pedestrian",
	"walker.pedestrian.0025": "pedestrian",
	"walker.pedestrian.0027": "pedestrian",
	"walker.pedestrian.0030": "pedestrian",
	"walker.pedestrian.0031": "pedestrian",
	"walker.pedestrian.0032": "pedestrian",
	"walker.pedestrian.0034": "pedestrian",
	"walker.pedestrian.0035": "pedestrian",
	"walker.pedestrian.0041": "pedestrian",
	"walker.pedestrian.0042": "pedestrian",
	"walker.pedestrian.0046": "pedestrian",
	"walker.pedestrian.0047": "pedestrian",
	"static.prop.dirtdebris01": "others",
	"static.prop.dirtdebris02": "others",
}

modality = dict(
	use_camera=True,
	use_lidar=False,
	use_radar=False,
	use_map=False,
	use_external=False,
)


def sanitize_b2d_ego_status_array(
	values: Union[Sequence[float], np.ndarray],
	warning_context: Optional[str] = None,
) -> np.ndarray:
	array = np.asarray(values, dtype=np.float32).copy()
	non_finite_mask = ~np.isfinite(array)
	accel_mask = np.zeros_like(array, dtype=bool)
	if array.shape[-1] >= 3:
		accel_mask[..., 1:3] = np.abs(array[..., 1:3]) > MAX_VALID_B2D_ACCELERATION_ABS

	if (non_finite_mask | accel_mask).any():
		context = warning_context or "B2D ego_status"
		finite_values = np.abs(array[np.isfinite(array)])
		max_abs_value = float(finite_values.max()) if finite_values.size > 0 else float("nan")
		logger.warning(
			"%s contains invalid B2D ego_status values; replacing non-finite values and "
			"acceleration magnitudes above %.1f with zeros (max_abs=%.3f).",
			context,
			MAX_VALID_B2D_ACCELERATION_ABS,
			max_abs_value,
		)

	array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
	if array.shape[-1] >= 3:
		array[..., 1:3] = np.where(
			np.abs(array[..., 1:3]) > MAX_VALID_B2D_ACCELERATION_ABS,
			0.0,
			array[..., 1:3],
		)
	return array


def build_b2d_ego_status_vector(
	speed: float,
	ego_accel_xy: Sequence[float],
	local_command_xy: Sequence[float],
	ego_fut_cmd: Sequence[float],
	warning_context: Optional[str] = None,
) -> torch.Tensor:
	values = np.asarray(
		[
			float(speed),
			float(ego_accel_xy[0]),
			float(ego_accel_xy[1]),
			float(local_command_xy[0]),
			float(local_command_xy[1]),
			*[float(value) for value in ego_fut_cmd],
		],
		dtype=np.float32,
	)
	values = sanitize_b2d_ego_status_array(values, warning_context=warning_context)
	return torch.from_numpy(values)


def preprocess_b2d_camera_images(
	images: Sequence[np.ndarray],
	image_size: Sequence[int],
) -> torch.Tensor:
	target_size = (int(image_size[0]), int(image_size[1]))
	processed_images = []
	for image in images:
		pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
		pil_image = pil_image.resize(target_size)
		image_array = np.asarray(pil_image, dtype=np.float32) / 255.0
		image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
		processed_images.append(torch.from_numpy(image_array).permute(2, 0, 1))
	return torch.stack(processed_images, dim=0)


def load_b2d_camera_images(
	data_root: Path,
	sensors: Mapping[str, Mapping[str, object]],
	camera_order: Sequence[str] = B2D_CAMERA_ORDER,
) -> list[np.ndarray]:
	images = []
	for camera_name in camera_order:
		if camera_name not in sensors:
			raise KeyError(f"Missing camera {camera_name} in B2D sensor dictionary.")

		image_path = Path(data_root) / Path(str(sensors[camera_name]["data_path"]))
		with Image.open(image_path) as image_file:
			images.append(np.asarray(image_file.convert("RGB"), dtype=np.uint8))
	return images
