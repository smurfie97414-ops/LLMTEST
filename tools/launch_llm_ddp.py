from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tail(path: Path, *, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _normalize_train_args(raw: list[str], gloo_interface: str | None) -> list[str]:
    args = list(raw)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        args = ["smoke", "--out-dir", "runs/llm-ddp-smoke", "--steps", "8"]
    if "--distributed" not in args:
        args.append("--distributed")
    if gloo_interface and "--gloo-interface" not in args:
        args.extend(["--gloo-interface", gloo_interface])
    return args


def _default_gloo_interface(value: str | None) -> str | None:
    if value:
        return value
    if platform.system() == "Windows":
        return "Ethernet"
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch tools/train_llm.py under a local multi-process DDP environment.")
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--gloo-interface", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--log-dir", default="runs/llm-ddp-worker-logs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.nproc < 2:
        raise SystemExit("--nproc must be >= 2 for DDP validation")
    if args.master_port < 1 or args.master_port > 65535:
        raise SystemExit("--master-port must be in 1..65535")

    gloo_interface = _default_gloo_interface(args.gloo_interface)
    train_args = _normalize_train_args(args.train_args, gloo_interface)
    log_dir = (ROOT / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [str(Path(args.python).resolve()), str(ROOT / "tools" / "train_llm.py"), *train_args]
    print("Launching DDP workers:")
    print("  command:", " ".join(command))
    print("  world_size:", args.nproc)
    print("  master:", f"{args.master_addr}:{args.master_port}")
    if gloo_interface:
        print("  gloo_interface:", gloo_interface)
    print("  logs:", log_dir)

    processes: list[tuple[int, subprocess.Popen[bytes], object, object, Path, Path]] = []
    for rank in range(args.nproc):
        env = os.environ.copy()
        env.update(
            {
                "WORLD_SIZE": str(args.nproc),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": args.master_addr,
                "MASTER_PORT": str(args.master_port),
                "CORTEX3_TCPSTORE_USE_LIBUV": "0",
                "CORTEX3_DISTRIBUTED_TIMEOUT_SECONDS": str(max(30, int(args.timeout))),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if gloo_interface:
            env["GLOO_SOCKET_IFNAME"] = gloo_interface
            env["CORTEX3_GLOO_IFNAME"] = gloo_interface
        stdout_path = log_dir / f"rank{rank}.stdout.log"
        stderr_path = log_dir / f"rank{rank}.stderr.log"
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        processes.append((rank, process, stdout, stderr, stdout_path, stderr_path))

    deadline = time.monotonic() + args.timeout
    timed_out = False
    while True:
        if all(process.poll() is not None for _, process, *_ in processes):
            break
        if time.monotonic() > deadline:
            timed_out = True
            for _, process, *_ in processes:
                if process.poll() is None:
                    process.kill()
            break
        time.sleep(0.2)

    failures: list[str] = []
    for rank, process, stdout, stderr, stdout_path, stderr_path in processes:
        process.wait(timeout=10)
        stdout.close()
        stderr.close()
        code = process.returncode
        if code != 0:
            failures.append(f"rank {rank} exited {code}")
        print(f"rank {rank}: exit={code} stdout={stdout_path} stderr={stderr_path}")

    if timed_out:
        failures.append(f"timeout after {args.timeout:.1f}s")
    if failures:
        print("DDP launch failed:", "; ".join(failures), file=sys.stderr)
        for rank, _, _, _, stdout_path, stderr_path in processes:
            print(f"\n--- rank {rank} stdout tail ---", file=sys.stderr)
            print(_tail(stdout_path), file=sys.stderr)
            print(f"\n--- rank {rank} stderr tail ---", file=sys.stderr)
            print(_tail(stderr_path), file=sys.stderr)
        raise SystemExit(124 if timed_out else 1)
    print("DDP launch completed successfully.")


if __name__ == "__main__":
    main()
