# Modified by LocoManiLab (Gamma Robotics), 2026
"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (protocol v4 motion token, or protocol v3 SMPL)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> blend smoothly to initial pose (or snap if no prior token) and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  <  -> rotate robot yaw left  (default 5°, or type "< 10" for custom)
  >  -> rotate robot yaw right (default 5°, or type "> 10" for custom)
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import math
import queue
import threading
import time

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    action_space: str = "motion_token"
    """Policy action type: motion_token sends protocol v4; smpl sends protocol v3."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Anti-jitter options
    max_latency_skip: int = 3
    """Cap on latency compensation skip steps (default 3 = 60ms at 50Hz).
    Prevents large position jumps when a new chunk arrives after slow inference."""

    token_ema_alpha: float = 0.4
    """EMA blend weight for motion_token/hand_joints at chunk boundaries (0=freeze, 1=snap).
    Smooths the token discontinuity between consecutive action chunks."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Yaw alignment
    yaw_step_deg: float = 5.0
    """Degrees to rotate per '<' / '>' keypress for pre-inference yaw alignment."""

    # Initial pose
    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing
# ---------------------------------------------------------------------------

SMPL_JOINTS_DIM = 72
SMPL_NUM_JOINTS = 24
SMPL_POSE_DIM = 63
SMPL_NUM_POSES = 21
SONIC_BODY_JOINT_DIM = 29
MIN_SMPL_CHUNK_FRAMES = 4


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def _squeeze_action_batch(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    while arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _as_smpl_joints_chunk(smpl_joints: np.ndarray) -> np.ndarray:
    arr = _squeeze_action_batch(np.asarray(smpl_joints, dtype=np.float32))
    if arr.ndim == 1:
        if arr.size != SMPL_JOINTS_DIM:
            raise ValueError(f"smpl_joints must be 72D, got {arr.shape}")
        arr = arr.reshape(1, SMPL_NUM_JOINTS, 3)
    elif arr.ndim == 2:
        if arr.shape == (SMPL_NUM_JOINTS, 3):
            arr = arr.reshape(1, SMPL_NUM_JOINTS, 3)
        elif arr.shape[1] == SMPL_JOINTS_DIM:
            arr = arr.reshape(arr.shape[0], SMPL_NUM_JOINTS, 3)
        else:
            raise ValueError(f"smpl_joints must be [T,72] or [24,3], got {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[1:] != (SMPL_NUM_JOINTS, 3):
            raise ValueError(f"smpl_joints must be [T,24,3], got {arr.shape}")
    else:
        raise ValueError(f"smpl_joints must have 1-3 dims, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_repeated_matrix(
    value: np.ndarray | None,
    frames: int,
    dim: int,
    name: str,
    *,
    default: np.ndarray | None = None,
) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be vector or matrix, got scalar")
    if arr.ndim == 1:
        if arr.size != dim:
            raise ValueError(f"{name} must be {dim}D, got {arr.shape}")
        arr = arr.reshape(1, dim)
    elif arr.ndim == 2:
        if arr.shape[1] != dim:
            raise ValueError(f"{name} must have trailing dim {dim}, got {arr.shape}")
    elif arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == dim:
        arr = arr[:, 0, :]
    else:
        raise ValueError(f"{name} must be [{dim}] or [T,{dim}], got {arr.shape}")

    if arr.shape[0] == 1 and frames > 1:
        arr = np.repeat(arr, frames, axis=0)
    elif arr.shape[0] < frames:
        pad = np.repeat(arr[-1:], frames - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > frames:
        arr = arr[:frames]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_smpl_pose_chunk(smpl_pose: np.ndarray | None, frames: int) -> np.ndarray:
    if smpl_pose is None:
        return np.zeros((frames, SMPL_NUM_POSES, 3), dtype=np.float32)
    arr = _squeeze_action_batch(np.asarray(smpl_pose, dtype=np.float32))
    if arr.ndim == 1:
        if arr.size != SMPL_POSE_DIM:
            raise ValueError(f"smpl_pose must be 63D, got {arr.shape}")
        arr = arr.reshape(1, SMPL_NUM_POSES, 3)
    elif arr.ndim == 2:
        if arr.shape == (SMPL_NUM_POSES, 3):
            arr = arr.reshape(1, SMPL_NUM_POSES, 3)
        elif arr.shape[1] == SMPL_POSE_DIM:
            arr = arr.reshape(arr.shape[0], SMPL_NUM_POSES, 3)
        else:
            raise ValueError(f"smpl_pose must be [T,63] or [21,3], got {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[1:] != (SMPL_NUM_POSES, 3):
            raise ValueError(f"smpl_pose must be [T,21,3], got {arr.shape}")
    else:
        raise ValueError(f"smpl_pose must have 1-3 dims, got {arr.shape}")

    if arr.shape[0] == 1 and frames > 1:
        arr = np.repeat(arr, frames, axis=0)
    elif arr.shape[0] < frames:
        arr = np.concatenate([arr, np.repeat(arr[-1:], frames - arr.shape[0], axis=0)], axis=0)
    elif arr.shape[0] > frames:
        arr = arr[:frames]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _pad_smpl_chunk(smpl_joints: np.ndarray, min_frames: int = MIN_SMPL_CHUNK_FRAMES) -> np.ndarray:
    chunk = _as_smpl_joints_chunk(smpl_joints)
    if chunk.shape[0] >= min_frames:
        return chunk
    pad = np.repeat(chunk[-1:], min_frames - chunk.shape[0], axis=0)
    return np.ascontiguousarray(np.concatenate([chunk, pad], axis=0), dtype=np.float32)


def pack_smpl_action_message(
    smpl_joints: np.ndarray,
    frame_index: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_quat_w: np.ndarray,
    smpl_pose: np.ndarray | None = None,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    """Pack an SMPL action chunk into protocol v3 for the C++ SONIC encoder."""
    smpl_joints = _as_smpl_joints_chunk(smpl_joints)
    frames = smpl_joints.shape[0]
    frame_index = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    if frame_index.size == 1 and frames > 1:
        frame_index = np.arange(frame_index[0], frame_index[0] + frames, dtype=np.int64)
    elif frame_index.size != frames:
        raise ValueError(f"frame_index length {frame_index.size} != smpl frames {frames}")

    pose_data = {
        "smpl_joints": smpl_joints,
        "smpl_pose": _as_smpl_pose_chunk(smpl_pose, frames),
        "body_quat_w": _as_repeated_matrix(body_quat_w, frames, 4, "body_quat_w"),
        "frame_index": frame_index,
        "joint_pos": _as_repeated_matrix(joint_pos, frames, SONIC_BODY_JOINT_DIM, "joint_pos"),
        "joint_vel": _as_repeated_matrix(joint_vel, frames, SONIC_BODY_JOINT_DIM, "joint_vel"),
    }

    if left_hand_joints is not None:
        pose_data["left_hand_joints"] = _as_repeated_matrix(
            left_hand_joints, 1, 7, "left_hand_joints"
        )
    if right_hand_joints is not None:
        pose_data["right_hand_joints"] = _as_repeated_matrix(
            right_hand_joints, 1, 7, "right_hand_joints"
        )

    return pack_pose_message(pose_data, topic="pose", version=3)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


def get_optional_action_field(action_dict: dict, key: str, default):
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    return default


def select_action_step(value: np.ndarray, index: int, dim: int, name: str) -> np.ndarray:
    arr = _squeeze_action_batch(np.asarray(value, dtype=np.float32))
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"{name} must be {dim}D, got {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] != dim:
            raise ValueError(f"{name} must have trailing dim {dim}, got {arr.shape}")
        return np.ascontiguousarray(arr[min(index, arr.shape[0] - 1)], dtype=np.float32)
    raise ValueError(f"{name} must be [D] or [T,D], got {arr.shape}")


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]
    observation["_soma_raw_state"] = {
        "body_q": np.asarray(state_msg.get("body_q", np.zeros(SONIC_BODY_JOINT_DIM)), dtype=np.float32).copy(),
        "body_dq": np.asarray(state_msg.get("body_dq", np.zeros(SONIC_BODY_JOINT_DIM)), dtype=np.float32).copy(),
        "base_quat": np.asarray(state_msg.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).copy(),
    }

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        raw_state = observation.get("_soma_raw_state", {})
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = None
        if "motion_token" in action:
            motion_key = "motion_token"
        elif "action.motion_token" in action:
            motion_key = "action.motion_token"

        if motion_key is not None:
            if np.abs(action[motion_key]).max() > 1.25:
                print(
                    f"[Warning] action['{motion_key}'] max "
                    f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                    "Exceeds action bound, skipping."
                )
                return None
        elif "smpl_joints" not in action and "action.smpl_joints" not in action:
            raise AssertionError(
                "Policy returned neither motion_token nor smpl_joints. "
                f"Available keys: {list(action.keys())}"
            )

        processed_action = concat_action(robot_model, action)
        for key in ("body_q", "body_dq", "base_quat"):
            if key in raw_state:
                processed_action[f"_soma_{key}"] = np.asarray(raw_state[key], dtype=np.float32)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
    stage_ref: dict[str, str] | None = None,
):
    """Persistent worker thread for async inference."""
    def set_stage(stage: str):
        if stage_ref is not None:
            stage_ref["stage"] = stage

    set_stage("idle")
    while not stop_event.is_set():
        try:
            try:
                set_stage("waiting_for_request")
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                set_stage("request_received")
                print("[DEBUG] Worker thread: inference task received", flush=True)
                set_stage("prepare_observation")
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                print("[DEBUG] Worker thread: observation ready, calling PolicyServer", flush=True)
                set_stage("policy_get_action")
                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    print("[DEBUG] Worker thread: PolicyServer returned action, caching chunk", flush=True)
                    set_stage("queue_result")
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
                else:
                    print("[DEBUG] Worker thread: PolicyServer returned no usable action", flush=True)
            finally:
                busy_event.clear()
                set_stage("idle")
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    config.action_space = config.action_space.strip().lower()
    if config.action_space not in {"motion_token", "smpl"}:
        raise ValueError("--action-space must be 'motion_token' or 'smpl'")

    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")
    print_green(f"Using action space: {config.action_space}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        if config.action_space == "smpl":
            print(
                "SMPL action space: skipping latent initial pose to keep the "
                "C++ ZMQ stream on protocol v3. The first SMPL policy chunk will "
                "initialize the stream."
            )
            time.sleep(0.2)
            print("Initial pose skipped for SMPL mode.")
            return

        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(
                np.float32
            )
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    def send_yaw_command(delta_deg: float):
        """Rotate the robot yaw by delta_deg degrees (negative = left, positive = right).

        Uses build_planner_message with mode=IDLE and movement=[0,0,0] so the
        robot turns in place to face the new direction. Requires PLANNER mode
        (press 'k' first). The facing direction is accumulated across keypresses,
        starting from the WBC default [1, 0, 0].
        """
        nonlocal yaw_facing_rad
        if not cpp_loop_running:
            print("Warning: C++ loop not running — press 'k' first to enable yaw control")
            return
        if cpp_mode != "PLANNER":
            print("Warning: yaw control requires PLANNER mode — press 'k' to start")
            return
        try:
            yaw_facing_rad += math.radians(delta_deg)
            facing = [math.cos(yaw_facing_rad), math.sin(yaw_facing_rad), 0.0]
            msg = build_planner_message(
                mode=0,                      # IDLE: stand still, just change facing
                movement=[0.0, 0.0, 0.0],
                facing=facing,
            )
            zmq_socket.send(msg)
            direction = "left" if delta_deg < 0 else "right"
            print_green(
                f"Yaw {direction} {abs(delta_deg):.1f}° → "
                f"facing=[{facing[0]:.3f}, {facing[1]:.3f}] "
                f"(total {math.degrees(yaw_facing_rad):.1f}°)"
            )
        except Exception as e:
            print(f"Warning: Failed to send yaw command: {e}")

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    # Yaw alignment state: accumulated facing direction (world frame unit vector)
    # Starts at [1, 0, 0] matching the WBC's default initial facing direction.
    yaw_facing_rad = 0.0

    # EMA smoothing state (chunk-boundary anti-jitter)
    prev_published_token: np.ndarray | None = None
    prev_left_hand: np.ndarray | None = None
    prev_right_hand: np.ndarray | None = None
    new_chunk_arrived = False

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter
        nonlocal prev_published_token, prev_left_hand, prev_right_hand
        nonlocal yaw_facing_rad

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            print("Moving to initial pose")
            publish_initial_pose()
            # Drain any stale inference results so they don't immediately
            # refill cached_action_chunk with a pre-reset observation.
            drained = 0
            while True:
                try:
                    result_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            cached_action_chunk = None
            action_chunk_index = 0
            last_inference_time = 0.0   # trigger fresh inference immediately
            prev_published_token = None  # reset EMA state
            prev_left_hand = None
            prev_right_hand = None
            yaw_facing_rad = 0.0        # reset accumulated yaw to WBC default facing
            # In latent mode the initial pose already used frame_index=0. In SMPL
            # mode no latent initial pose is sent, so the first policy chunk starts at 0.
            zmq_frame_counter = 1 if config.action_space == "motion_token" else 0
            print(f"Reset: drained {drained} stale result(s), cleared chunk cache, yaw reset")
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            zmq_frame_counter = 0
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk, reset frame counter")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )
        elif key.startswith("<") or key.startswith(">"):
            # Yaw alignment: '<' = left, '>' = right
            # Supports optional angle: "< 10" rotates 10° left
            direction = -1 if key.startswith("<") else 1
            parts = key.split()
            try:
                step = float(parts[1]) if len(parts) > 1 else config.yaw_step_deg
            except (ValueError, IndexError):
                step = config.yaw_step_deg
            send_yaw_command(direction * step)

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()
    worker_stage = {"stage": "idle"}

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
            worker_stage,
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()
            new_chunk_arrived = False

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                raw_idx = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                action_chunk_index = min(raw_idx, config.max_latency_skip)
                new_chunk_arrived = True
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s, "
                    f"skip: {raw_idx}→{action_chunk_index})"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                    if cached_action_chunk is None:
                        print(
                            "[DEBUG] Main loop: queued inference request "
                            f"(worker_busy={worker_is_busy}, "
                            f"worker_stage={worker_stage['stage']})",
                            flush=True,
                        )
                except queue.Full:
                    if cached_action_chunk is None:
                        print(
                            "[DEBUG] Main loop: inference request already queued "
                            f"(worker_busy={worker_is_busy}, "
                            f"worker_stage={worker_stage['stage']})",
                            flush=True,
                        )
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                published_horizon = config.action_horizon
                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                elif config.action_space == "motion_token":
                    motion_token_arr = _squeeze_action_batch(
                        get_action_field(processed_action, "motion_token"),
                    )
                    left_hand_arr = _squeeze_action_batch(
                        get_action_field(processed_action, "left_hand_joints"),
                    )
                    right_hand_arr = _squeeze_action_batch(
                        get_action_field(processed_action, "right_hand_joints"),
                    )

                    horizon = motion_token_arr.shape[0] if motion_token_arr.ndim == 2 else 1
                    published_horizon = horizon
                    current_idx = min(action_chunk_index, horizon - 1)

                    motion_token = select_action_step(motion_token_arr, current_idx, 64, "motion_token")
                    left_hand_joints = select_action_step(
                        left_hand_arr, current_idx, 7, "left_hand_joints"
                    )
                    right_hand_joints = select_action_step(
                        right_hand_arr, current_idx, 7, "right_hand_joints"
                    )

                    # EMA smoothing at chunk boundaries: blends new token with the
                    # last published token to eliminate hard discontinuities.
                    if new_chunk_arrived and prev_published_token is not None:
                        alpha = config.token_ema_alpha
                        motion_token = alpha * motion_token + (1 - alpha) * prev_published_token
                        if prev_left_hand is not None:
                            left_hand_joints = alpha * left_hand_joints + (1 - alpha) * prev_left_hand
                        if prev_right_hand is not None:
                            right_hand_joints = alpha * right_hand_joints + (1 - alpha) * prev_right_hand
                    prev_published_token = motion_token.copy()
                    prev_left_hand = left_hand_joints.copy()
                    prev_right_hand = right_hand_joints.copy()

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    last_sent_motion_token = motion_token.copy()
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )
                elif config.action_space == "smpl":
                    smpl_arr = _squeeze_action_batch(
                        get_action_field(processed_action, "smpl_joints")
                    )
                    smpl_arr = np.asarray(smpl_arr, dtype=np.float32)
                    if smpl_arr.ndim == 1 or (
                        smpl_arr.ndim == 2 and smpl_arr.shape == (SMPL_NUM_JOINTS, 3)
                    ):
                        horizon = 1
                        current_idx = 0
                        smpl_chunk = smpl_arr
                    elif smpl_arr.ndim in (2, 3):
                        horizon = smpl_arr.shape[0]
                        current_idx = min(action_chunk_index, horizon - 1)
                        smpl_chunk = smpl_arr[current_idx:]
                    else:
                        raise ValueError(f"smpl_joints has unsupported shape {smpl_arr.shape}")

                    published_horizon = horizon
                    smpl_chunk = _pad_smpl_chunk(smpl_chunk)

                    left_hand_joints = select_action_step(
                        get_optional_action_field(
                            processed_action,
                            "left_hand_joints",
                            np.zeros(7, dtype=np.float32),
                        ),
                        current_idx,
                        7,
                        "left_hand_joints",
                    )
                    right_hand_joints = select_action_step(
                        get_optional_action_field(
                            processed_action,
                            "right_hand_joints",
                            np.zeros(7, dtype=np.float32),
                        ),
                        current_idx,
                        7,
                        "right_hand_joints",
                    )

                    if new_chunk_arrived and prev_published_token is not None:
                        alpha = config.token_ema_alpha
                        smpl_chunk[0] = alpha * smpl_chunk[0] + (1 - alpha) * prev_published_token
                        if prev_left_hand is not None:
                            left_hand_joints = alpha * left_hand_joints + (1 - alpha) * prev_left_hand
                        if prev_right_hand is not None:
                            right_hand_joints = alpha * right_hand_joints + (1 - alpha) * prev_right_hand
                    prev_published_token = smpl_chunk[0].copy()
                    prev_left_hand = left_hand_joints.copy()
                    prev_right_hand = right_hand_joints.copy()

                    body_q = np.asarray(
                        processed_action.get(
                            "_soma_body_q", np.zeros(SONIC_BODY_JOINT_DIM, dtype=np.float32)
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    body_dq = np.asarray(
                        processed_action.get(
                            "_soma_body_dq", np.zeros(SONIC_BODY_JOINT_DIM, dtype=np.float32)
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    base_quat = np.asarray(
                        processed_action.get(
                            "_soma_base_quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    if body_q.size != SONIC_BODY_JOINT_DIM:
                        raise ValueError(f"_soma_body_q must be {SONIC_BODY_JOINT_DIM}D, got {body_q.shape}")
                    if body_dq.size != SONIC_BODY_JOINT_DIM:
                        raise ValueError(f"_soma_body_dq must be {SONIC_BODY_JOINT_DIM}D, got {body_dq.shape}")
                    if base_quat.size != 4:
                        raise ValueError(f"_soma_base_quat must be 4D, got {base_quat.shape}")

                    frame_indices = np.arange(
                        zmq_frame_counter,
                        zmq_frame_counter + smpl_chunk.shape[0],
                        dtype=np.int64,
                    )
                    zmq_frame_counter += 1

                    zmq_message = pack_smpl_action_message(
                        smpl_joints=smpl_chunk,
                        frame_index=frame_indices,
                        joint_pos=body_q,
                        joint_vel=body_dq,
                        body_quat_w=base_quat,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent SMPL action - "
                            f"frame: {frame_indices[0]}, "
                            f"chunk shape: {smpl_chunk.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, published_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
