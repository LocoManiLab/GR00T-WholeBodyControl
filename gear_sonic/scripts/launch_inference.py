# Modified by LocoManiLab (Gamma Robotics), 2026
"""
All-in-one tmux launcher for SONIC VLA inference.

Starts the inference stack in a single tmux session:

    Window 0 — inference (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: VLA Inference │
    │ (gear_sonic_deploy)   │ (.venv_inference)     │
    ├───────────────────────┼───────────────────────┤
    │ Pane 2: Keyboard Pub  │ Pane 3: Data Exporter │
    │ (.venv_inference)     │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_inference.sh     -> .venv_inference
        bash install_scripts/install_data_collection.sh -> .venv_data_collection (optional, for recording)
    - gear_sonic_deploy built (see docs)
    - Isaac-GR00T PolicyServer running separately

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_inference.py                        # real robot
    python gear_sonic/scripts/launch_inference.py --sim                  # MuJoCo sim
    python gear_sonic/scripts/launch_inference.py --no-data-exporter     # no recording pane
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import base64
import subprocess
import sys
import textwrap
import time
import xml.etree.ElementTree as ET


def _bootstrap_venv():
    """Re-exec with the .venv_inference Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_inference" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_inference not found.\n"
            "  Run: bash install_scripts/install_inference.sh"
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
class InferenceLaunchConfig:
    """CLI config for the all-in-one VLA inference tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim instead of real robot."""

    # C++ deploy options
    low_latency: bool = False
    """Use the official low_latency deploy policy. Shortcut that fills in
    --deploy-checkpoint policy/low_latency/model and
    --deploy-obs-config policy/low_latency/observation_config.yaml
    (explicit --deploy-checkpoint / --deploy-obs-config still take precedence)."""

    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh. Leave empty for default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # VLA inference options
    policy_host: str = "localhost"
    """Isaac-GR00T PolicyServer host."""

    policy_port: int = 5550
    """Isaac-GR00T PolicyServer port."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    prompt: str = "demo"
    """Language prompt for inference."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 16
    """Action horizon of the VLA policy."""

    action_space: str = "motion_token"
    """Policy action type passed to run_vla_inference.py: motion_token or smpl."""

    max_latency_skip: int = 3
    """Cap on latency compensation skip steps passed to run_vla_inference.py."""

    token_ema_alpha: float = 0.4
    """EMA blend weight at chunk boundaries passed to run_vla_inference.py."""

    inference_rate: float = 2.5
    """VLA policy forward-pass frequency (Hz). Use 10.0 when TRT is enabled (~50ms inference)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # Data exporter (optional recording during inference)
    data_exporter: bool = False
    """Start the data exporter pane for recording during inference."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    task_prompt: str = ""
    """Task prompt for the data exporter. Defaults to the inference prompt if empty."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate."""

    yaw_step_deg: float = 5.0
    """Degrees to rotate per '<' / '>' keypress for pre-inference yaw alignment."""


SESSION_NAME = "sonic_inference"


def _check_prerequisites(config: InferenceLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_inference" / "bin" / "activate").exists():
        errors.append(
            ".venv_inference not found. Run: bash install_scripts/install_inference.sh"
        )

    missing_robot_assets = _missing_g1_robot_assets(repo_root)
    if missing_robot_assets:
        errors.append(
            "G1 robot model assets are missing:\n"
            + "\n".join(f"    {path}" for path in missing_robot_assets)
        )

    deploy_dir = repo_root / "gear_sonic_deploy"
    deploy_wrapper = deploy_dir / "run_deploy_cu12.sh"
    deploy_binary = deploy_dir / "target/release/g1_deploy_onnx_ref"
    if not deploy_wrapper.exists():
        errors.append(
            f"gear_sonic_deploy/run_deploy_cu12.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )
    elif not os.access(deploy_wrapper, os.X_OK):
        errors.append(f"CUDA12 deploy wrapper is not executable: {deploy_wrapper}")

    if not deploy_binary.exists():
        errors.append(
            f"Compiled deploy binary not found: {deploy_binary}. "
            "Build it from gear_sonic_deploy before launching inference."
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
                "low_latency policy files are missing:\n"
                + "\n".join(f"    {path}" for path in missing)
            )

    if config.data_exporter:
        if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
            errors.append(
                ".venv_data_collection not found (needed for data exporter). Run: "
                "bash install_scripts/install_data_collection.sh"
            )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first."
        )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session():
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "inference"],
    )

    # Split into 4 panes: 0|1 / 2|3
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    target = f"{SESSION_NAME}:0.{pane_index}"
    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: InferenceLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Apply the low_latency preset (explicit flags still take precedence).
    if config.low_latency:
        if not config.deploy_checkpoint:
            config.deploy_checkpoint = "policy/low_latency/model"
        if not config.deploy_obs_config:
            config.deploy_obs_config = "policy/low_latency/observation_config.yaml"

    _check_prerequisites(config)
    _kill_existing_session()

    exporter_prompt = config.task_prompt if config.task_prompt else config.prompt

    print("=" * 60)
    print("  SONIC VLA Inference Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  PolicyServer:    {config.policy_host}:{config.policy_port}")
    print(f"  Embodiment:      {config.embodiment_tag}")
    print(f"  Prompt:          {config.prompt}")
    print(f"  Low latency:     {'Yes' if config.low_latency else 'No'}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    if config.deploy_obs_config:
        print(f"  Obs config:      {config.deploy_obs_config}")
    print(f"  Action rate:     {config.action_publish_rate} Hz")
    print(f"  Action horizon:  {config.action_horizon}")
    print(f"  Action space:    {config.action_space}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Data exporter:   {'Yes' if config.data_exporter else 'No'}")
    if config.data_exporter:
        print(f"    DC frequency:  {config.data_exporter_frequency} Hz")
        print(f"    Task prompt:   {exporter_prompt}")
    print(f"  PC IP:           {_get_local_ip()}")
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

        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {_shell_quote(repo_root / 'gear_sonic_deploy')} && "
        # Use the CUDA 12 wrapper. deploy.sh can pick up /usr/local/cuda
        # (CUDA 13.2 on this host), which the 570 driver cannot run.
        f"./run_deploy_cu12.sh "
        f"--input-type {_shell_quote(config.deploy_input_type)} "
        f"--zmq-host {_shell_quote(config.deploy_zmq_host)} "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {_shell_quote(config.deploy_checkpoint)} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {_shell_quote(config.deploy_obs_config)} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {_shell_quote(config.deploy_planner)} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {_shell_quote(config.deploy_motion_data)} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {_shell_quote(config.deploy_output_type)} "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 2 (bottom-left): Keyboard Publisher ---
    keyboard_script = textwrap.dedent("""\
        import zmq, time
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind('tcp://localhost:5580')
        time.sleep(0.5)
        print('Keyboard publisher ready. Keys: p=pause, k=start/stop, i=init pose, [/]=toggle hands, t=prompt, </>=yaw left/right')
        while True:
            key = input()
            if key.startswith('t '):
                pub.send_string('prompt:' + key[2:])
                print('Sent prompt: ' + key[2:])
            else:
                pub.send_string(key)
                print('Sent: ' + key)
    """)
    encoded = base64.b64encode(keyboard_script.encode()).decode()
    keyboard_cmd = (
        f"cd {_shell_quote(repo_root)} && "
        f"source .venv_inference/bin/activate && "
        f"python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    )

    print("Starting keyboard publisher (pane 2)...")
    _send_to_pane(1, keyboard_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Data Exporter (optional) ---
    if config.data_exporter:
        exporter_cmd = (
            f"cd {_shell_quote(repo_root)} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_data_exporter.py "
            f"--task-prompt {_shell_quote(exporter_prompt)} "
            f"--data-collection-frequency {_shell_quote(config.data_exporter_frequency)} "
            f"--camera-host {_shell_quote(config.camera_host)} "
            f"--camera-port {_shell_quote(config.camera_port)}"
        )
        if config.dataset_name:
            exporter_cmd += f" --dataset-name {_shell_quote(config.dataset_name)}"

        print("Starting data exporter (pane 3)...")
        _send_to_pane(3, exporter_cmd, wait=2.0)

    # --- Pane 1 (top-right): VLA Inference ---
    inference_cmd = (
        f"cd {_shell_quote(repo_root)} && "
        f"source .venv_inference/bin/activate && "
        f"python gear_sonic/scripts/run_vla_inference.py "
        f"--host {_shell_quote(config.policy_host)} "
        f"--port {_shell_quote(config.policy_port)} "
        f"--embodiment-tag {_shell_quote(config.embodiment_tag)} "
        f"--prompt {_shell_quote(config.prompt)} "
        f"--action-publish-rate {_shell_quote(config.action_publish_rate)} "
        f"--action-horizon {_shell_quote(config.action_horizon)} "
        f"--action-space {_shell_quote(config.action_space)} "
        f"--max-latency-skip {_shell_quote(config.max_latency_skip)} "
        f"--token-ema-alpha {_shell_quote(config.token_ema_alpha)} "
        f"--rate {_shell_quote(config.inference_rate)} "
        f"--yaw-step-deg {_shell_quote(config.yaw_step_deg)} "
        f"--camera-host {_shell_quote(config.camera_host)} "
        f"--camera-port {_shell_quote(config.camera_port)}"
    )

    print("Starting VLA inference (pane 1)...")
    _send_to_pane(2, inference_cmd, wait=1.0)

    # Select the VLA inference pane
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
    print("  Window 'inference':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  Keyboard Publisher")
    print("    Pane 2 (top-right):    VLA Inference  <-- you are here")
    if config.data_exporter:
        print("    Pane 3 (bottom-right): Data Exporter")
    print()
    print("  ** run_deploy_cu12.sh (pane 0) is waiting for confirmation --")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Keyboard controls (type in pane 1):")
    print("    p        - Pause / resume inference")
    print("    k        - Start / stop C++ control loop")
    print("    i        - Send initial pose")
    print("    [        - Toggle left hand open/closed (initial pose)")
    print("    ]        - Toggle right hand open/closed (initial pose)")
    print("    t <text> - Change inference prompt")
    if config.data_exporter:
        print("    c        - Start recording episode")
        print("    s        - Stop recording (success)")
        print("    f        - Stop recording (failure)")
    print()
    print("  Navigation:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig, _frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(InferenceLaunchConfig)
    main(config)
