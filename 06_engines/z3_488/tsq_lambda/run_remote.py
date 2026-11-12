# -*- coding: utf-8 -*-
"""run_remote.py - Server-only driver. The connection password is supplied
by the operator through the environment variable MOF_PSC_SSH_PASSWORD; no
credential is stored in this repository.

Helper for executing jobs on the internal server (paramiko).
Usage:
  python run_remote.py exec "command"
  python run_remote.py put <local file> <remote path>
  python run_remote.py get <remote path> <local file>
"""
import os
import sys
import paramiko

HOST = os.environ.get("MOF_PSC_SSH_HOST")
USER = os.environ.get("MOF_PSC_SSH_USER")
if not (HOST and USER):
    sys.exit("set MOF_PSC_SSH_HOST and MOF_PSC_SSH_USER before running run_remote.py")
PWD = os.environ.get("MOF_PSC_SSH_PASSWORD")
if PWD is None:
    sys.exit("set MOF_PSC_SSH_PASSWORD before running run_remote.py")


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=15)
    return c


def main():
    mode = sys.argv[1]
    c = client()
    if mode == "exec":
        cmd = sys.argv[2]
        stdin, stdout, stderr = c.exec_command(cmd, timeout=600)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        print(out)
        if err.strip():
            print("[stderr]", err[-3000:], file=sys.stderr)
        sys.exit(rc)
    elif mode == "put":
        with open(sys.argv[2], "rb") as f:
            data = f.read()
        stdin, stdout, stderr = c.exec_command(
            f"cat > {sys.argv[3]} && wc -c {sys.argv[3]}", timeout=120)
        stdin.write(data)
        stdin.channel.shutdown_write()
        rc = stdout.channel.recv_exit_status()
        print(stdout.read().decode(), "rc =", rc)
        sys.exit(rc)
    elif mode == "get":
        s = c.open_sftp()
        s.get(sys.argv[2], sys.argv[3])
        print(f"get {sys.argv[2]} -> {sys.argv[3]} ok")


if __name__ == "__main__":
    main()
