"""Keyboard publisher for data collection recording control.

Publishes single-key commands to the ZMQ topic that ``run_data_exporter.py``
already listens on (port 5580, see ``ZMQKeyboardSubscriber``). Useful when you
cannot tell whether the Pico trigger was missed or just delayed — press the
key locally and you'll see the ``>>> sent`` line in this terminal and the
sticky status bar flip in the exporter terminal instantly.

The keyboard path is independent from the Pico ``manager_state`` path, so you
can keep using the Pico triggers at the same time.

Usage
-----
    # Terminal 1: data exporter (as usual)
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up cup"

    # Terminal 2: keyboard controller
    python gear_sonic/scripts/keyboard_control.py

Keys
----
    c  or  ENTER   Toggle recording   (start episode  |  stop & save)
    x              Discard the current episode without saving
    q  or  Ctrl-C  Quit this controller (does not affect the exporter)
"""

import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import tyro
import zmq

from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
)


@dataclass
class KeyboardControlConfig:
    port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ PUB port (must match the exporter's subscriber)."""

    bind_host: str = "*"
    """Bind interface. '*' = any, 'localhost' to restrict to this machine."""


HELP_BANNER = """\
Keyboard control for run_data_exporter
--------------------------------------
  c / ENTER   Toggle recording  (start episode  |  stop & save)
  x           Discard current episode (only while RECORDING)
  q / Ctrl-C  Quit this controller
"""


def _getch(timeout: float = 0.1) -> str | None:
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    return sys.stdin.read(1)


def main(config: KeyboardControlConfig) -> None:
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{config.bind_host}:{config.port}")
    # Give ZMQ a moment to complete the PUB/SUB handshake with any already-running subscriber.
    time.sleep(0.3)

    print(f"[keyboard_control] Publishing to tcp://{config.bind_host}:{config.port}")
    print(HELP_BANNER)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = _getch(0.1)
            if ch is None:
                continue
            if ch in ("q", "\x03"):
                print("[keyboard_control] quit")
                break
            if ch in ("\n", "\r"):
                ch = "c"
            if ch == "c":
                pub.send_string("c")
                print(">>> sent 'c' (toggle record)")
            elif ch == "x":
                pub.send_string("x")
                print(">>> sent 'x' (discard)")
            else:
                printable = ch if ch.isprintable() else repr(ch)
                print(f"(ignored key: {printable})")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main(tyro.cli(KeyboardControlConfig))
