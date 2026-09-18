"""
immich_static.service — Built-in systemd service installation and management.
"""

import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional


def is_systemd_available() -> bool:
    """Checks if systemd systemctl command is present on the system."""
    return shutil.which("systemctl") is not None


def get_service_paths(is_user_mode: bool) -> Tuple_Path_Cmd:
    if is_user_mode:
        user_dir = Path.home() / ".config" / "systemd" / "user"
        unit_file = user_dir / "immich-static.service"
        cmd_prefix = ["systemctl", "--user"]
    else:
        system_dir = Path("/etc/systemd/system")
        unit_file = system_dir / "immich-static.service"
        cmd_prefix = ["sudo", "systemctl"] if os.geteuid() != 0 else ["systemctl"]
    return unit_file, cmd_prefix


Tuple_Path_Cmd = Any


def generate_unit_content(
    python_exec: str,
    working_dir: Path,
    env_file: Optional[Path],
    serve_args: List[str],
    is_user_mode: bool,
    user: str,
) -> str:
    args_str = " ".join(serve_args) if serve_args else ""
    exec_start = f"{python_exec} -m immich_static.cli serve {args_str}".strip()
    env_line = f"EnvironmentFile={env_file.resolve()}" if env_file and env_file.is_file() else ""
    target = "default.target" if is_user_mode else "multi-user.target"
    user_line = "" if is_user_mode else f"User={user}\n"

    content = f"""[Unit]
Description=Immich Static Video Webhook Daemon
After=network.target

[Service]
Type=simple
{user_line}WorkingDirectory={working_dir.resolve()}
ExecStart={exec_start}
Restart=always
RestartSec=5
{env_line}
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy={target}
"""
    return content.strip() + "\n"


def install_service(
    port: int = 8080,
    secret: Optional[str] = None,
    system_wide: bool = False,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
):
    """Installs and starts the immich-static systemd service."""
    if not is_systemd_available():
        print("❌ Error: systemctl not found. systemd is not available on this system.")
        return 1

    is_user_mode = not system_wide and os.geteuid() != 0
    unit_file, cmd_prefix = get_service_paths(is_user_mode)

    python_exec = sys.executable
    working_dir = Path.cwd()
    env_file = working_dir / ".env"
    if not env_file.exists():
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / ".env").exists():
            working_dir = repo_root
            env_file = repo_root / ".env"

    serve_args = [f"--port {port}"]
    if secret:
        serve_args.append(f"--secret {secret}")
    if extra_args:
        serve_args.extend(extra_args)

    user = getpass.getuser()
    unit_content = generate_unit_content(
        python_exec=python_exec,
        working_dir=working_dir,
        env_file=env_file if env_file.exists() else None,
        serve_args=serve_args,
        is_user_mode=is_user_mode,
        user=user,
    )

    mode_label = "User service (~/.config/systemd/user)" if is_user_mode else "System service (/etc/systemd/system)"
    print("=" * 65)
    print("🔧 Installing Immich-Static systemd Service")
    print(f"   • Mode             : {mode_label}")
    print(f"   • Service Unit     : {unit_file}")
    print(f"   • Python Binary    : {python_exec}")
    print(f"   • Working Dir      : {working_dir}")
    print(f"   • Environment File : {env_file if env_file.exists() else 'None'}")
    print("=" * 65 + "\n")

    if dry_run:
        print("🔍 [DRY-RUN] Generated Unit File Content:")
        print("-" * 50)
        print(unit_content)
        print("-" * 50)
        print("Commands that would run:")
        print(f"   {' '.join(cmd_prefix)} daemon-reload")
        print(f"   {' '.join(cmd_prefix)} enable --now immich-static")
        if is_user_mode:
            print(f"   loginctl enable-linger {user}")
        return 0

    # Write unit file
    if is_user_mode:
        unit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(unit_file, "w", encoding="utf-8") as f:
            f.write(unit_content)
    else:
        # Use sudo tee if not root
        if os.geteuid() != 0:
            proc = subprocess.Popen(["sudo", "tee", str(unit_file)], stdin=subprocess.PIPE, text=True)
            proc.communicate(input=unit_content)
            if proc.returncode != 0:
                print("❌ Failed to write systemd unit file.")
                return 1
        else:
            unit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(unit_file, "w", encoding="utf-8") as f:
                f.write(unit_content)

    print(f"✅ Service unit written to {unit_file}")

    # Reload systemd
    subprocess.run(cmd_prefix + ["daemon-reload"], check=True)
    print("✅ systemd daemon reloaded.")

    # Enable and start
    subprocess.run(cmd_prefix + ["enable", "--now", "immich-static"], check=True)
    print("🚀 Service enabled and started successfully!")

    # Enable lingering for user mode so it runs on boot without login
    if is_user_mode:
        try:
            subprocess.run(["loginctl", "enable-linger", user], check=False)
            print(f"✅ Enabled systemd linger for user '{user}' (starts on boot without login).")
        except Exception:
            pass

    print("\n💡 Service Management Commands:")
    if is_user_mode:
        print("   • Status  : immich-static service status (or systemctl --user status immich-static)")
        print("   • Logs    : immich-static service logs   (or journalctl --user -u immich-static -f)")
        print("   • Stop    : immich-static service stop   (or systemctl --user stop immich-static)")
        print("   • Restart : immich-static service restart (or systemctl --user restart immich-static)")
    else:
        print("   • Status  : sudo systemctl status immich-static")
        print("   • Logs    : sudo journalctl -u immich-static -f")
        print("   • Stop    : sudo systemctl stop immich-static")
        print("   • Restart : sudo systemctl restart immich-static")

    return 0


def uninstall_service(system_wide: bool = False):
    """Stops, disables, and removes the immich-static systemd service."""
    is_user_mode = not system_wide and os.geteuid() != 0
    unit_file, cmd_prefix = get_service_paths(is_user_mode)

    print(f"🛑 Stopping and disabling immich-static service...")
    try:
        subprocess.run(cmd_prefix + ["stop", "immich-static"], check=False)
        subprocess.run(cmd_prefix + ["disable", "immich-static"], check=False)
    except Exception:
        pass

    if unit_file.exists():
        if is_user_mode:
            unit_file.unlink()
        else:
            subprocess.run(["sudo", "rm", "-f", str(unit_file)], check=False)
        print(f"🗑️  Removed unit file: {unit_file}")

    subprocess.run(cmd_prefix + ["daemon-reload"], check=False)
    print("✅ Service uninstalled successfully.")
    return 0


def run_service_cmd(action: str, system_wide: bool = False):
    """Controls service state (status, start, stop, restart, logs)."""
    is_user_mode = not system_wide and os.geteuid() != 0
    _, cmd_prefix = get_service_paths(is_user_mode)

    if action == "logs":
        log_cmd = ["journalctl"]
        if is_user_mode:
            log_cmd.append("--user")
        log_cmd.extend(["-u", "immich-static", "-f"])
        try:
            subprocess.run(log_cmd)
        except KeyboardInterrupt:
            pass
        return 0

    if action in ("status", "start", "stop", "restart"):
        subprocess.run(cmd_prefix + [action, "immich-static"])
        return 0

    print(f"❌ Unknown action: {action}")
    return 1


def run_service(args: Any):
    """CLI dispatcher for the service subcommand."""
    action = getattr(args, "action", "status")
    system_wide = getattr(args, "system", False)
    dry_run = getattr(args, "dry_run", False)

    if action == "install":
        port = int(getattr(args, "port", 8080) or 8080)
        secret = getattr(args, "secret", None)
        install_service(port=port, secret=secret, system_wide=system_wide, dry_run=dry_run)
    elif action == "uninstall":
        uninstall_service(system_wide=system_wide)
    else:
        run_service_cmd(action, system_wide=system_wide)
