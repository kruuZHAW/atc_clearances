#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import io
import sys
import zipfile
import pathlib
import argparse
from dotenv import load_dotenv
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

import requests

# --------- Config via env vars ----------
load_dotenv()
SURFDRIVE_URL  = os.environ.get("SURFDRIVE_URL")
SURFDRIVE_PASS = os.environ.get("SURFDRIVE_PASS")

if not SURFDRIVE_URL or not SURFDRIVE_PASS:
    print("ERROR: Please set SURFDRIVE_URL and SURFDRIVE_PASS in your environment.", file=sys.stderr)
    sys.exit(1)

# ---------- URL helpers ----------
def extract_base_and_token(share_url: str) -> tuple[str, str]:
    """
    From e.g. 'https://surfdrive.surf.nl/files/index.php/s/<token>?path=...'
    return:
      base  -> 'https://surfdrive.surf.nl/files' (preserves '/files' if present)
      token -> '<token>'
    """
    u = urlsplit(share_url)
    prefix = "/index.php/s/"
    idx = u.path.find(prefix)
    if idx == -1:
        raise ValueError("Not a Nextcloud public share URL (missing /index.php/s/).")
    base_path = u.path[:idx]  # keep any '/files' prefix
    token = u.path[idx + len(prefix):].split("/", 1)[0]
    base = urlunsplit((u.scheme, u.netloc, base_path, "", ""))
    return base, token


BASE, TOKEN = extract_base_and_token(SURFDRIVE_URL)

def share_page_url() -> str:
    # Public share landing page
    return f"{BASE}/index.php/s/{TOKEN}"

def day_zip_endpoint() -> str:
    # ZIP endpoint (folder given via ?path=/YYYY/MM/DD)
    return f"{BASE}/index.php/s/{TOKEN}/download"

# ---------- Session / auth ----------

def open_share_session(password: str) -> requests.Session:
    """
    Establish a session by visiting the share page and posting the password.
    Keeps auth cookies (NCSESSID, etc.) so /download works.
    """
    s = requests.Session()
    # 1) GET share page (get cookies & maybe requesttoken)
    r = s.get(share_page_url(), timeout=30)
    r.raise_for_status()

    # Some Nextcloud require a requesttoken header
    m = re.search(r'name="requesttoken"\s+content="([^"]+)"', r.text)
    headers = {"requesttoken": m.group(1)} if m else {}

    # 2) POST password to same URL
    r2 = s.post(share_page_url(), data={"password": password},
                headers=headers, allow_redirects=True, timeout=30)
    r2.raise_for_status()

    # Basic sanity check: make sure we’re past the password form
    if 'name="password"' in r2.text.lower():
        raise RuntimeError("Password submission appears to have failed (still seeing password form).")

    return s

# ---------- ZIP download / extract ----------
def fetch_day_zip(session: requests.Session, year: str, month: str, day: str, timeout: int = 900) -> bytes:
    url = day_zip_endpoint()
    params = {"path": f"/{year}/{month}/{day}"}  # let requests encode
    r = session.get(url, params=params, stream=True, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "zip" not in ctype.lower():
        # If we got HTML, we likely aren’t authenticated for ZIP
        body_peek = r.text[:500] if "text/html" in ctype.lower() else ctype
        raise RuntimeError(f"Expected ZIP but got {ctype}. Body starts: {body_peek!r}")
    # Stream into memory
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1 << 20):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()

def extract_zip(zip_bytes: bytes, out_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Extract the ZIP to out_dir. Returns list of extracted file paths.
    (If you want to filter by sector prefix, we can add that here.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[pathlib.Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename.rsplit("/", 1)[-1]  # flatten any paths
            target = out_dir / name
            if target.exists() and target.stat().st_size > 0:
                extracted.append(target)
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Download MUAC daily audio from SURFdrive as ZIP and extract locally (no WebDAV fallback)."
    )
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("outdir", help="Local output directory")
    ap.add_argument("--timeout", type=int, default=900, help="ZIP HTTP timeout in seconds")
    args = ap.parse_args()

    # Parse date
    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("ERROR: date must be in YYYY-MM-DD format.", file=sys.stderr)
        sys.exit(2)

    year, month, day = f"{dt.year:04d}", f"{dt.month:02d}", f"{dt.day:02d}"
    out_dir = pathlib.Path(args.outdir) / year / month / day

    print("[info] Opening session (password auth)…")
    try:
        session = open_share_session(SURFDRIVE_PASS)
    except Exception as e:
        print(f"[error] Login failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Downloading ZIP for {year}-{month}-{day} …")
    try:
        zip_bytes = fetch_day_zip(session, year, month, day, timeout=args.timeout)
    except Exception as e:
        print(f"[error] ZIP download failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Extracting to {out_dir} …")
    files = extract_zip(zip_bytes, out_dir)
    print(f"[done] Extracted {len(files)} files to {out_dir}")

if __name__ == "__main__":
    main()