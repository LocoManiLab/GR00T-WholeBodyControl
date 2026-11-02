"""ZED (Stereolabs) camera driver.

Requires the ZED SDK and pyzed Python bindings:
    https://www.stereolabs.com/developers/release/

Install pyzed into the active venv with::

    python /usr/local/zed/get_python_api.py
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyzed.sl as sl

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
)


class ZEDConfig:
    """Configuration for ZED stereo cameras (ZED Mini, ZED 2/2i, ZED X)."""

    color_image_dim: tuple[int, int] = (640, 480)  # (W, H) of final ndarray
    fps: int = 30
    resolution: str = "HD720"  # native ZED grab res: VGA/HD720/HD1080/HD2K
    use_left_eye: bool = True
    enable_stereo: bool = False  # also publish the other eye as a second RGB


_RES_MAP = {
    "VGA": sl.RESOLUTION.VGA,
    "HD720": sl.RESOLUTION.HD720,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD2K": sl.RESOLUTION.HD2K,
}


class ZEDSensor(Sensor):
    """Sensor for ZED stereo cameras using the Stereolabs SDK.

    Output schema matches the OAK driver: a single RGB ndarray keyed by
    ``mount_position``. With ``enable_stereo=True`` a second RGB image
    keyed ``{mount_position}_stereo`` is added.
    """

    def __init__(
        self,
        config: ZEDConfig = ZEDConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_id: str | None = None,
    ):
        self.config = config
        self.mount_position = mount_position

        init = sl.InitParameters()
        init.camera_resolution = _RES_MAP.get(config.resolution, sl.RESOLUTION.HD720)
        init.camera_fps = config.fps
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        if device_id is not None and str(device_id).strip() != "":
            init.set_from_serial_number(int(device_id))

        self.cam = sl.Camera()
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        info = self.cam.get_camera_information()
        cfg = info.camera_configuration
        print(
            f"[{mount_position}] ZED opened: model={info.camera_model} "
            f"sn={info.serial_number} fw={cfg.firmware_version} "
            f"native={cfg.resolution.width}x{cfg.resolution.height}@{cfg.fps:.0f}fps "
            f"-> output={config.color_image_dim[0]}x{config.color_image_dim[1]}"
        )

        self._runtime = sl.RuntimeParameters()
        self._mat_primary = sl.Mat()
        self._mat_secondary = sl.Mat() if config.enable_stereo else None

    def _retrieve_rgb(self, mat: sl.Mat, view: "sl.VIEW") -> np.ndarray:
        self.cam.retrieve_image(mat, view)
        bgra = mat.get_data()
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        target_wh = self.config.color_image_dim
        if (rgb.shape[1], rgb.shape[0]) != target_wh:
            rgb = cv2.resize(rgb, target_wh, interpolation=cv2.INTER_AREA)
        return rgb

    def read(self) -> dict[str, Any] | None:
        if self.cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        primary_view = sl.VIEW.LEFT if self.config.use_left_eye else sl.VIEW.RIGHT
        rgb = self._retrieve_rgb(self._mat_primary, primary_view)
        capture_time = time.time()

        timestamps = {self.mount_position: capture_time}
        images = {self.mount_position: rgb}

        if self.config.enable_stereo:
            secondary_view = sl.VIEW.RIGHT if self.config.use_left_eye else sl.VIEW.LEFT
            rgb_other = self._retrieve_rgb(self._mat_secondary, secondary_view)
            other_key = f"{self.mount_position}_stereo"
            timestamps[other_key] = capture_time
            images[other_key] = rgb_other

        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        h, w = self.config.color_image_dim[1], self.config.color_image_dim[0]
        spaces = {
            "color_image": gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
        }
        if self.config.enable_stereo:
            spaces["color_image_stereo"] = gym.spaces.Box(
                low=0, high=255, shape=(h, w, 3), dtype=np.uint8
            )
        return gym.spaces.Dict(spaces)

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass
