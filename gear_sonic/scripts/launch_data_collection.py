# Modified by LocoManiLab (Gamma Robotics), 2026
"""
All-in-one tmux launcher for SONIC data collection.

Starts the full data collection stack in a single tmux session:

    Window 0 — data_collection (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 2: Data Exporter │
    │ (gear_sonic_deploy)   │ (.venv_data_collection)│
    ├───────────────────────┼───────────────────────┤
    │ Pane 1: Teleop        │ Pane 3: Camera Viewer │
    │ (.venv_teleop)        │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — camera  (when --camera-viewer):
    ┌─────────────────────────────────────────────────┐
    │ Camera Viewer (run_camera_viewer.py)            │
    │ (.venv_data_collection)                         │
    └─────────────────────────────────────────────────┘

    Window 2 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_pico.sh            -> .venv_teleop
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - For sim: .venv_sim must exist (see install instructions)

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_data_collection.py                          # real robot (default)
    python gear_sonic/scripts/launch_data_collection.py --sim                    # MuJoCo sim
    python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer       # skip viewer
    python gear_sonic/scripts/launch_data_collection.py --pico-input-source isaac-teleop  # in-process CloudXR / DeviceIO
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def _bootstrap_venv():
    """Re-exec with the .venv_data_collection Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP for the PICO to connect to.

    Prefers a real Wi-Fi / Ethernet LAN address and skips loopback and
    VPN/proxy interfaces (Tailscale, WireGuard, a Clash/Mihomo TUN whose
    default route hijacks the 8.8.8.8 trick and hands back 198.18.x.x, etc).
    Set SOMA_PC_IP to force a specific address.
    """
    import ipaddress
    import subprocess

    override = os.environ.get("SOMA_PC_IP", "").strip()
    if override:
        return override

    bad_nets = [
        ipaddress.ip_network(n)
        for n in ("127.0.0.0/8", "169.254.0.0/16", "198.18.0.0/15", "100.64.0.0/10")
    ]
    bad_prefixes = (
        "lo", "docker", "br-", "veth", "virbr", "tun", "tap",
        "tailscale", "mihomo", "clash", "zt", "wg",
    )
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            text=True, timeout=2.0,
        )
        candidates = []  # (priority, ifname, ip)  -- lower priority first
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            ifname, addr = parts[1], parts[3].split("/")[0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if any(ip in net for net in bad_nets):
                continue
            if ifname.lower().startswith(bad_prefixes):
                continue
            is_wifi = ifname.startswith(("wl", "wlan", "wlp", "wlx"))
            candidates.append((0 if is_wifi else 1, ifname, addr))
        if candidates:
            candidates.sort()
            return candidates[0][2]
    except Exception:
        pass

    # Fallback: default-route source IP (may be a proxy/VPN address).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def _missing_g1_robot_assets(repo_root: Path) -> list[Path]:
    """Return URDF/mesh files required by the Pinocchio G1 model that are absent."""
    model_dir = repo_root / "gear_sonic/data/robot_model/model_data/g1"
    urdf_path = model_dir / "g1_29dof_with_hand.urdf"
    if not urdf_path.is_file():
        return [urdf_path]
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError):
        return [urdf_path]
    return [
        model_dir / mesh.attrib["filename"]
        for mesh in root.iter("mesh")
        if mesh.attrib.get("filename")
        and not (model_dir / mesh.attrib["filename"]).is_file()
    ]


@dataclass
class DataCollectionLaunchConfig:
    """CLI config for the all-in-one data collection tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    # C++ deploy options
    low_latency: bool = False
    """Use the new official low_latency policy. Shortcut that fills in
    --deploy-checkpoint policy/low_latency/model and
    --deploy-obs-config policy/low_latency/observation_config.yaml
    (explicit --deploy-checkpoint / --deploy-obs-config still take precedence)."""

    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy (zmq_manager, keyboard, etc.)."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # Teleop streamer options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    pico_input_source: str = "xrt"
    """Teleop input source for pico_manager_thread_server.py (xrt or isaac-teleop)."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    # Data exporter options
    task_prompt: str = "demo"
    """Language task prompt for the data exporter."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate from timestamp."""

    root_output_dir: str = "outputs"
    """Parent directory in which the data exporter creates the dataset directory."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

    text_to_speech: bool = True
    """Enable voice feedback via espeak (data exporter)."""

    # Camera viewer
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host (shared by data exporter and viewer)."""

    camera_port: int = 5555
    """Camera server port (shared by data exporter and viewer)."""

    # Keyboard control
    keyboard_control: bool = True
    """Start keyboard_control.py in a separate window for ZMQ recording control."""


SESSION_NAME = "sonic_data_collection"


def _check_prerequisites(config: DataCollectionLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(
            ".venv_teleop not found. Run: bash install_scripts/install_pico.sh"
        )

    if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
        errors.append(
            ".venv_data_collection not found. Run: "
            "bash install_scripts/install_data_collection.sh"
        )

    missing_robot_assets = _missing_g1_robot_assets(repo_root)
    if missing_robot_assets:
        errors.append(
            "G1 robot model assets are missing:\n"
            + "\n".join(f"    {path}" for path in missing_robot_assets)
        )

    if config.pico_input_source == "xrt":
        xrt_root = (
            repo_root
            / "external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
        )
        if not list(xrt_root.glob("xrobotoolkit_sdk*.so")):
            errors.append(f"XRoboToolkit Python extension is missing under: {xrt_root}")
        xrt_runtime = xrt_root / "lib/libPXREARobotSDK.so"
        if not xrt_runtime.is_file():
            errors.append(f"XRoboToolkit runtime library is missing: {xrt_runtime}")

    crc_library = (
        "crc_aarch64.so"
        if os.uname().machine in {"aarch64", "arm64"}
        else "crc_amd64.so"
    )
    crc_runtime = (
        repo_root
        / "external_dependencies/unitree_sdk2_python/unitree_sdk2py/utils/lib"
        / crc_library
    )
    if not crc_runtime.is_file():
        errors.append(f"Unitree CRC runtime library is missing: {crc_runtime}")

    deploy_dir = repo_root / "gear_sonic_deploy"
    deploy_wrapper = deploy_dir / "run_deploy_cu12.sh"
    deploy_binary = deploy_dir / "target/release/g1_deploy_onnx_ref"
    if not deploy_wrapper.exists():
        errors.append(
            f"gear_sonic_deploy/run_deploy_cu12.sh not found at {deploy_dir}. "
            "Ensure the CUDA12 deploy wrapper is installed."
        )
    elif not os.access(deploy_wrapper, os.X_OK):
        errors.append(f"CUDA12 deploy wrapper is not executable: {deploy_wrapper}")

    if not deploy_binary.exists():
        errors.append(
            f"Compiled deploy binary not found: {deploy_binary}. "
            "Build it from gear_sonic_deploy before collecting data."
        )

    if config.low_latency:
        low_latency_files = [
            deploy_dir / "policy/low_latency/model_decoder.onnx",
            deploy_dir / "policy/low_latency/model_encoder.onnx",
            deploy_dir / "policy/low_latency/observation_config.yaml",
            deploy_dir / "planner/target_vel/V2/planner_sonic.onnx",
        ]
        missing = [str(path) for path in low_latency_files if not path.exists()]
        if missing:
            errors.append(
                "low_latency deploy files are missing:\n"
                + "\n".join(f"    {path}" for path in missing)
            )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first "
            "(see install instructions)."
        )

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be one of: xrt, isaac-teleop")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session():
    """Create a 4-pane tmux layout."""
    # Create detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )

    # Enable mouse support (click panes, scroll, resize)
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )

    # Rename default window
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "data_collection"],
    )

    # Split into 4 panes:
    #   0 | 1
    #   -----
    #   2 | 3

    # Split horizontally: pane 0 (left) and pane 1 (right)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )

    # Split left pane vertically: pane 0 (top-left) and pane 2 (bottom-left)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )

    # Split right pane vertically: pane 1 becomes top-right, new pane 3 bottom-right
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{SESSION_NAME}:0.{pane_index}"

    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Apply the low_latency preset (explicit flags still take precedence).
    if config.low_latency:
        if not config.deploy_checkpoint:
            config.deploy_checkpoint = "policy/low_latency/model"
        if not config.deploy_obs_config:
            config.deploy_obs_config = "policy/low_latency/observation_config.yaml"

    _check_prerequisites(config)
    _kill_existing_session()

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Output root:     {config.root_output_dir}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    print(f"  Teleop input:    {config.pico_input_source}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  DC frequency:    {config.data_exporter_frequency} Hz")
    print(f"  Camera viewer:   {'Yes' if config.camera_viewer else 'No'}")
    print(f"  Wrist cameras:   {'Yes' if config.record_wrist_cameras else 'No'}")
    print(f"  Text-to-speech:  {'Yes' if config.text_to_speech else 'No'}")
    print(f"  PICO vis:        vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
    print(f"  PC IP (for PICO): {_get_local_ip()}")
    print(f"  Teleop vis:      vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
    print("=" * 60)

    _create_tmux_session()
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        # Switch back to the data_collection window for the remaining panes
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        # deploy.sh may select the system CUDA 13 runtime, which is incompatible
        # with this machine's CUDA 12/TensorRT deployment environment.
        f"./run_deploy_cu12.sh "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 2 (bottom-left): Teleop Streamer ---
    pico_cmd = (
        f"cd {repo_root} && "
        f"source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/pico_manager_thread_server.py "
        f"--input-source {config.pico_input_source}"
    )
    if config.pico_manager:
        pico_cmd += " --manager"
    if config.pico_vis_vr3pt:
        pico_cmd += " --vis_vr3pt"
    if config.pico_vis_smpl:
        pico_cmd += " --vis_smpl"
    if config.pico_waist_tracking:
        pico_cmd += " --waist_tracking"

    print("Starting teleop streamer (pane 2)...")
    _send_to_pane(1, pico_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Keyboard Control ---
    if config.keyboard_control:
        keyboard_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/keyboard_control.py"
        )
        print("Starting keyboard control (pane 3)...")
        _send_to_pane(3, keyboard_cmd, wait=2.0)

    # --- Window 'camera': Camera Viewer ---
    if config.camera_viewer:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "camera"],
        )
        viewer_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_camera_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port}"
        )
        camera_target = f"{SESSION_NAME}:camera"
        subprocess.run(
            ["tmux", "send-keys", "-t", camera_target, viewer_cmd, "C-m"],
        )
        print("Starting camera viewer (window: camera)...")
        time.sleep(1.0)
        # Switch back to data_collection window
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 1 (top-right): Data Exporter ---
    exporter_cmd = (
        f"cd {_shell_quote(repo_root)} && "
        f"source .venv_data_collection/bin/activate && "
        f"python gear_sonic/scripts/run_data_exporter.py "
        f"--task-prompt {_shell_quote(config.task_prompt)} "
        f"--root-output-dir {_shell_quote(config.root_output_dir)} "
        f"--data-collection-frequency {_shell_quote(config.data_exporter_frequency)} "
        f"--camera-host {_shell_quote(config.camera_host)} "
        f"--camera-port {_shell_quote(config.camera_port)}"
    )
    if config.dataset_name:
        exporter_cmd += f" --dataset-name {_shell_quote(config.dataset_name)}"
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"

    print("Starting data exporter (pane 1)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Select the data exporter pane so the user lands there for interactive input
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
    print("  Window 'data_collection':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  Teleop Streamer")
    print("    Pane 2 (top-right):    Data Exporter  <-- you are here")
    if config.keyboard_control:
        print("    Pane 3 (bottom-right): Keyboard Control")
    print()
    if config.camera_viewer:
        print("  Window 'camera':")
        print("    Camera Viewer (.venv_data_collection)")
        print()
    print("  ** run_deploy_cu12.sh (pane 0) is waiting for confirmation —")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim or config.camera_viewer:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    # Attach to the session
    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    # After detach/exit, offer cleanup
    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(DataCollectionLaunchConfig)
    main(config)
