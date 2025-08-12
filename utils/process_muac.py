import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Iterable
from pathlib import Path
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- filename parsing ----------
FILENAME_RE = re.compile(
    r"""^
    (?P<sector>[a-z0-9_]+)      _      # e.g. muac_delta_low
    (?P<date>\d{8})             _      # YYYYMMDD
    (?P<time>\d{6})             _      # HHMMSS
    (?P<freq_hz>\d+)                   # e.g. 135958300 (Hz)
    \.(?P<ext>mp3|wav)$
    """,
    re.X | re.I
)

AUDIO_EXT = {".mp3", ".wav", ".flac"}

# ----- DATA CLASS TO STORE AUDIO FILE INFO -----

@dataclass(frozen=True)
class ATCFileMeta:
    sector: str           # e.g., muac_delta_low / muac_delta_middle
    dt: datetime          # timezone-aware (UTC)
    freq_hz: int
    freq_mhz: float
    ext: str              # mp3/wav/flac
    original_name: str
    path: str              # file path

def parse_atc_filename(name: str, path: str, tz=timezone.utc) -> Optional[ATCFileMeta]:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    sector = m.group("sector").lower()
    dt = datetime.strptime(
        f"{m.group('date')} {m.group('time')}", "%Y%m%d %H%M%S"
    ).replace(tzinfo=tz)
    freq_hz = int(m.group("freq_hz"))
    return ATCFileMeta(
        sector=sector,
        dt=dt,
        freq_hz=freq_hz,
        freq_mhz=freq_hz / 1_000_000.0,
        ext=m.group("ext").lower(),
        original_name=name,
        path=path,
    )
    

# ----- AUDIO FILE HELPERS -----

def list_years(base_dir: str | Path) -> List[str]:
    p = Path(base_dir)
    return sorted([d.name for d in p.iterdir() if d.is_dir() and d.name.isdigit()])

def list_months(base_dir: str | Path, year: str) -> List[str]:
    p = Path(base_dir) / year
    return sorted([d.name for d in p.iterdir() if d.is_dir()])

def list_days(base_dir: str | Path, year: str, month: str) -> List[str]:
    p = Path(base_dir) / year / month
    return sorted([d.name for d in p.iterdir() if d.is_dir()])

def list_audio_one_day(base_dir: str | Path, year: str, month: str, day: str) -> List[dict]:
    """
    Return list of dicts {name, path} for audio files in <base_dir>/<Y>/<M>/<D>.
    """
    folder = Path(base_dir) / year / month / day
    out = []
    if not folder.exists():
        return out
    for f in folder.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in AUDIO_EXT:
            out.append({"name": f.name, "path": str(f)})
    return sorted(out, key=lambda x: x["name"])

# ----- CATALOG BUILDER -----

AudioCatalog = List[Tuple[ATCFileMeta, AudioSegment]]

def build_audio_catalog(
    base_dir: str | Path,
    date_ymd: tuple[str, str, str],
    *,
    sectors: Optional[Iterable[str]] = None,
    start: Optional[datetime] = None,
    stop: Optional[datetime] = None,
    min_mhz: Optional[float] = None,
    max_mhz: Optional[float] = None,
    max_workers: int = 0,            # 0/1 = no parallel load
    limit: Optional[int] = None,
    show_progress: bool = False,
    target_sr: Optional[int] = 16000, # resample to 16k by default (None = keep native)
    mono: bool = True,                # force mono by default
) -> AudioCatalog:
    """
    Scan local day folder, parse+filter, then load audio with pydub.AudioSegment.
    Returns a list of (ATCFileMeta, AudioSegment).
    """
    Y, M, D = date_ymd
    sectors_set = set(s.lower() for s in sectors) if sectors else None

    files = list_audio_one_day(base_dir, Y, M, D)

    # Parse & filter
    metas: List[ATCFileMeta] = []
    for f in files:
        meta = parse_atc_filename(f["name"], f["path"])
        if not meta:
            continue
        if sectors_set and meta.sector not in sectors_set:
            continue
        if start and meta.dt < start:
            continue
        if stop and meta.dt > stop:
            continue
        if min_mhz and meta.freq_mhz < min_mhz:
            continue
        if max_mhz and meta.freq_mhz > max_mhz:
            continue
        metas.append(meta)

    if limit:
        metas = metas[:limit]

    def _load(meta: ATCFileMeta) -> Tuple[ATCFileMeta, AudioSegment]:
        seg = AudioSegment.from_file(meta.path)
        if mono:
            seg = seg.set_channels(1)
        if target_sr:
            seg = seg.set_frame_rate(target_sr)
        return meta, seg

    catalog: AudioCatalog = []
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(metas), desc="Loading audio", unit="file")
        except Exception:
            pbar = None

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_load, m): m for m in metas}
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    catalog.append(fut.result())
                except Exception as e:
                    print(f"[warn] audio load failed: {m.original_name} -> {e}")
                if pbar:
                    pbar.update(1)
    else:
        for m in metas:
            try:
                catalog.append(_load(m))
            except Exception as e:
                print(f"[warn] audio load failed: {m.original_name} -> {e}")
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    return catalog