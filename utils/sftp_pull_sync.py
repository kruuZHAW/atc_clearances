#!/usr/bin/env python3
"""
sftp_pull_sync.py — Pull-sync a directory from an SFTP server to a local directory,
organizing files into sector/year/month/day based on filenames like:

    delta_high_132_083_2025-08-21_13-56-18.mp3
 -> delta_high/2025/08/21/delta_high_132_083_2025-08-21_13-56-18.mp3

Designed to run non-interactively on HPC/SLURM. Uses Paramiko (SFTP over SSH).

Configuration:
- CLI flags (see --help), or environment variables (used as defaults):
    SFTP_HOST, SFTP_PORT, SFTP_USER, SFTP_KEY, SFTP_REMOTE, SFTP_LOCAL,
    SFTP_DELETE, SFTP_INCLUDE_HIDDEN, SFTP_DRY_RUN

Example (no flags, rely on env defaults below):
    python sftp_pull_sync.py

Example (explicit flags):
    python sftp_pull_sync.py \
      --host junzis.com --port 2222 --user filedrop \
      --key ~/.ssh/id_ed25519_tudelft \
      --remote atco_audio/audio_sdrplay \
      --local  /store/kruu/atc_muac/audio_sdrplay \
      --dry-run
"""

import argparse
import os
import sys
import time
import stat
import re
from pathlib import Path
import paramiko

# ---------- Helpers ----------

def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def connect_sftp(host: str, port: int, username: str, key_filename: str):
    ssh = paramiko.SSHClient()
    try:
        ssh.load_system_host_keys()
    except Exception:
        pass
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=host,
        port=port,
        username=username,
        key_filename=os.path.expanduser(key_filename),
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
    )
    return ssh, ssh.open_sftp()

def is_dir_attr(attr: paramiko.SFTPAttributes) -> bool:
    return stat.S_ISDIR(attr.st_mode)

def is_file_attr(attr: paramiko.SFTPAttributes) -> bool:
    return stat.S_ISREG(attr.st_mode)

def remote_listdir_attr(sftp: paramiko.SFTPClient, path: str):
    try:
        return sftp.listdir_attr(path)
    except FileNotFoundError:
        raise SystemExit(f"[fatal] Remote path not found: {path!r}")
    except PermissionError:
        raise SystemExit(f"[fatal] Permission denied listing remote path: {path!r}")

def remote_walk(sftp: paramiko.SFTPClient, top: str, include_hidden: bool=False):
    """Yield (dirpath, dirnames, files) where files are tuples (name, full, size, mtime)."""
    items = remote_listdir_attr(sftp, top)
    dirnames = []
    files = []
    for attr in items:
        name = attr.filename
        if not include_hidden and name.startswith('.'):
            continue
        full = f"{top.rstrip('/')}/{name}"
        if is_dir_attr(attr):
            dirnames.append((name, full))
        elif is_file_attr(attr):
            files.append((name, full, attr.st_size, attr.st_mtime))
        # ignore symlinks/devices
    yield top, [d for d,_ in dirnames], files
    for name, full in dirnames:
        yield from remote_walk(sftp, full, include_hidden=include_hidden)

def ensure_local_dir(path: Path, dry_run: bool):
    if path.exists():
        return
    print(f"[mkdir] {path}")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)

def download_file(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path, remote_mtime: int, dry_run: bool):
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    print(f"[get]   {remote_path}  ->  {local_path}")
    if dry_run:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(tmp_path))
    os.replace(tmp_path, local_path)
    os.utime(local_path, (time.time(), remote_mtime))  # preserve mtime

def file_needs_update(local_path: Path, remote_size: int, remote_mtime: int) -> bool:
    if not local_path.exists():
        return True
    try:
        st = local_path.stat()
    except FileNotFoundError:
        return True
    if st.st_size != remote_size:
        return True
    if int(remote_mtime) > int(st.st_mtime) + 2:  # tolerance
        return True
    return False

# ---------- Filename mapping: sector/year/month/day ----------

# Accepts delta_high|delta_low|delta_mid; two numeric groups (frequency); YYYY-MM-DD; then underscore, then time.
FILENAME_RE = re.compile(
    r'^(?P<sector>delta_(?:high|low|mid))_\d+_\d+_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_',
    re.ASCII,
)

def map_destination(rel_file: str) -> str:
    """
    Map a remote relative file path to a local relative path organized as:
        sector / YYYY / MM / DD / original_filename
    If the name doesn't match, return the original rel_file.
    """
    base = os.path.basename(rel_file)
    m = FILENAME_RE.match(base)
    if not m:
        return rel_file
    sector = m.group('sector')
    year = m.group('year')
    month = m.group('month')
    day = m.group('day')
    return f"{sector}/{year}/{month}/{day}/{base}"

# ---------- Core sync ----------

def build_remote_set_and_sync(sftp, remote_root: str, local_root: Path, include_hidden: bool, dry_run: bool):
    """Walk remote and ensure local is up to date. Returns set of mapped relative file paths."""
    remote_files_set = set()
    for dirpath, _dirnames, files in remote_walk(sftp, remote_root, include_hidden=include_hidden):
        rel_dir = os.path.relpath(dirpath, remote_root).replace('\\','/')
        rel_dir = "" if rel_dir == "." else rel_dir

        for (name, full, size, mtime) in files:
            src_rel = f"{rel_dir}/{name}" if rel_dir else name
            dst_rel = map_destination(src_rel)
            remote_files_set.add(dst_rel)

            local_path = local_root / dst_rel
            ensure_local_dir(local_path.parent, dry_run=dry_run)

            if file_needs_update(local_path, size, mtime):
                download_file(sftp, full, local_path, mtime, dry_run=dry_run)
            else:
                print(f"[skip]  up-to-date: {dst_rel}")
    return remote_files_set

def delete_extraneous_local(local_root: Path, remote_files_set: set, include_hidden: bool, dry_run: bool):
    """Optionally remove local files that are not present remotely (after mapping)."""
    for root, _, files in os.walk(local_root):
        for fname in files:
            if not include_hidden and fname.startswith('.'):
                continue
            local_path = Path(root) / fname
            rel = str(local_path.relative_to(local_root)).replace('\\','/')
            if rel not in remote_files_set:
                print(f"[delete] {rel}")
                if not dry_run:
                    try:
                        local_path.unlink()
                    except Exception as e:
                        print(f"[warn] failed to delete {rel}: {e}")

# ---------- Main ----------

def main():
    # Defaults pulled from environment so SLURM wrapper can keep CLI short/empty
    default_host = os.getenv("SFTP_HOST", "junzis.com")
    default_port = int(os.getenv("SFTP_PORT", "2222"))
    default_user = os.getenv("SFTP_USER", "filedrop")
    default_key  = os.getenv("SFTP_KEY",  "~/.ssh/id_ed25519_tudelft")
    default_remote = os.getenv("SFTP_REMOTE", "atco_audio/audio_sdrplay")
    default_local  = os.path.expanduser(os.getenv("SFTP_LOCAL",  "/store/kruu/atc_muac/audio_sdrplay"))
    default_delete = getenv_bool("SFTP_DELETE", False)
    default_hidden = getenv_bool("SFTP_INCLUDE_HIDDEN", False)
    default_dryrun = getenv_bool("SFTP_DRY_RUN", False)

    p = argparse.ArgumentParser(
        description="Pull-sync a remote SFTP directory to a local directory; organize by sector/year/month/day."
    )
    p.add_argument("--host", default=default_host, help=f"SFTP host (default: {default_host})")
    p.add_argument("--port", type=int, default=default_port, help=f"SFTP port (default: {default_port})")
    p.add_argument("--user", default=default_user, help=f"SFTP username (default: {default_user})")
    p.add_argument("--key",  default=default_key,  help=f"Path to private key (default: {default_key})")
    p.add_argument("--remote", default=default_remote, help=f"Remote dir (default: {default_remote})")
    p.add_argument("--local",  default=default_local,  help=f"Local dir  (default: {default_local})")
    p.add_argument("--delete", action="store_true", default=default_delete, help="Delete locals not present remotely")
    p.add_argument("--include-hidden", action="store_true", default=default_hidden, help="Include dotfiles (.*)")
    p.add_argument("--dry-run", action="store_true", default=default_dryrun, help="Show actions only")
    args = p.parse_args()

    local_root = Path(os.path.expanduser(args.local)).resolve()
    remote_root = args.remote.rstrip('/')

    print(f"[info] host={args.host} port={args.port} user={args.user}")
    print(f"[info] key={os.path.expanduser(args.key)}")
    print(f"[info] remote={remote_root}")
    print(f"[info] local={local_root}")
    print(f"[info] delete={args.delete} include_hidden={args.include_hidden} dry_run={args.dry_run}")

    try:
        ssh, sftp = connect_sftp(args.host, args.port, args.user, args.key)
    except Exception as e:
        print(f"[fatal] Failed to connect: {e}")
        sys.exit(2)

    try:
        remote_files = build_remote_set_and_sync(
            sftp, remote_root=remote_root, local_root=local_root,
            include_hidden=args.include_hidden, dry_run=args.dry_run
        )
        if args.delete:
            delete_extraneous_local(local_root, remote_files, include_hidden=args.include_hidden, dry_run=args.dry_run)
        print("[done]")
    finally:
        try:
            sftp.close()
        except Exception:
            pass
        try:
            ssh.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
