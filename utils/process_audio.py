import re
import numpy as np
from datetime import datetime, timezone, timedelta, date
from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterable
from pathlib import Path
from pydub import AudioSegment, effects
from pyannote.audio import Pipeline
import webrtcvad
import torch
from pydub.utils import mediainfo
import math

AUDIO_EXT = {".mp3", ".wav", ".flac"}

# New layout example:
#   delta_mid_135_508_2025-08-21_03-00-00.mp3
NEW_FILENAME_RE = re.compile(
    r"""^
    (?P<sector>delta_(?:low|mid|high)) _
    (?P<fM>\d{3}) _ (?P<fK>\d{3}) _              # freq MHz: fM_fK  -> fM.fK (e.g., 135_508 -> 135.508)
    (?P<Y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2}) _     # date YYYY-MM-DD
    (?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})       # time HH-MM-SS
    \.(?P<ext>mp3|wav|flac)$
    """,
    re.X | re.I,
)

@dataclass(frozen=True)
class ATCFileMeta:
    sector: str
    dt: datetime          # tz-aware
    freq_hz: int
    freq_mhz: float
    ext: str
    original_name: str
    path: str

def parse_atc_filename(name: str, path: str, tz=timezone.utc) -> Optional[ATCFileMeta]:
    
    m = NEW_FILENAME_RE.match(name)
    if m:
        sector = m.group("sector").lower()
        # Build datetime
        dt = datetime(
            int(m.group("Y")), int(m.group("m")), int(m.group("d")),
            int(m.group("H")), int(m.group("M")), int(m.group("S")),
            tzinfo=tz
        )
        # Frequency: fM_fK -> fM.fK MHz -> Hz (exact, no float rounding)
        fM = int(m.group("fM"))   # e.g. 135
        fK = int(m.group("fK"))   # e.g. 508
        freq_hz  = fM * 1_000_000 + fK * 1_000     # 135_508 -> 135,508,000 Hz
        freq_mhz = fM + fK / 1000.0                # 135.508 MHz
        ext = m.group("ext").lower()
        return ATCFileMeta(sector, dt, freq_hz, freq_mhz, ext, name, path)

    return None

# ----- AUDIO FILE HELPERS -----
def list_audio_one_day_sector(base_dir: str | Path, sector: str, Y: str, M: str, D: str) -> List[dict]:
    folder = Path(base_dir) / sector / Y / M / D
    out: List[dict] = []
    if not folder.exists():
        return out
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".flac"}:
            out.append({"name": f.name, "path": str(f)})
    return sorted(out, key=lambda x: x["name"])

def _normalize_sectors_arg(sectors: str | Iterable[str]) -> List[str]:
    if isinstance(sectors, str):
        sectors = [sectors]
    out = [s.strip().lower() for s in sectors if str(s).strip()]
    if not out:
        raise ValueError("`sectors` must be a non-empty string or iterable of strings.")
    return sorted(set(out))

def _validate_sectors_exist(base_dir: str | Path, sectors: List[str]) -> None:
    existing = {d.name for d in Path(base_dir).iterdir() if d.is_dir()}
    missing = [s for s in sectors if s not in existing]
    if missing:
        raise FileNotFoundError(
            f"Sector folder(s) not found under {Path(base_dir).resolve()}: {missing}. "
            f"Available: {sorted(existing)}"
        )

def _dates_span(start: datetime, stop: datetime) -> List[date]:
    """All calendar dates touched by [start, stop)."""
    # ensure half-open; if stop == start, empty
    if not start < stop:
        return []
    cur = start.date()
    last = (stop - timedelta(microseconds=1)).date()
    days = []
    while cur <= last:
        days.append(cur)
        cur = cur + timedelta(days=1)
    return days

def file_interval(meta) -> tuple[datetime, datetime]:
    """Return [fs, fe) using probed duration; clamp to <= 1h + 10s tolerance."""
    fs = meta.dt
    try:
        dur_s = float(mediainfo(meta.path).get("duration", "0"))
    except Exception:
        dur_s = 0.0
    if not (dur_s and dur_s > 0):
        dur_s = 3600.0
    fe_probe = fs + timedelta(seconds=dur_s)
    fe_clamped = min(fs + timedelta(hours=1), fe_probe + timedelta(seconds=10))
    return fs, fe_clamped

def intervals_intersect(a0, a1, b0, b1) -> bool:
    """Half-open intersection for [a0,a1) and [b0,b1). None means unbounded."""
    if b0 is None: b0 = a0  # treat None as minimal bound
    if b1 is None: b1 = a1
    return a0 < b1 and b0 < a1

# ----- CATALOG BUILDER -----
AudioCatalog = List[Tuple[ATCFileMeta, AudioSegment]]

def build_audio_catalog(
    base_dir: str | Path,
    *,
    sectors: str | Iterable[str],          # REQUIRED
    start: datetime,                        # REQUIRED (tz-aware)
    stop: datetime,                         # REQUIRED (tz-aware), half-open
    min_mhz: Optional[float] = None,
    max_mhz: Optional[float] = None,
    max_workers: int = 0,
    limit: Optional[int] = None,
    show_progress: bool = False,
    target_sr: Optional[int] = 16000,
    mono: bool = True,
) -> AudioCatalog:
    """
    Scan base_dir/<sector>/<YYYY>/<MM>/<DD>/ across all dates in [start, stop),
    parse + filter by time/frequency, then load AudioSegment. Returns list of (ATCFileMeta, AudioSegment).
    """
    
    # validate time window
    if start.tzinfo is None or stop.tzinfo is None:
        raise ValueError("start/stop must be timezone-aware (UTC).")
    if not start < stop:
        raise ValueError("Expected start < stop for half-open interval [start, stop).")

    # validate sectors
    sectors_list = _normalize_sectors_arg(sectors)
    _validate_sectors_exist(base_dir, sectors_list)

    # gather candidate files across all touched dates
    files: List[dict] = []
    for sec in sectors_list:
        for d in _dates_span(start, stop):
            files.extend(
                list_audio_one_day_sector(base_dir, sec, f"{d:%Y}", f"{d:%m}", f"{d:%d}")
            )

    # parse & filter
    metas: List["ATCFileMeta"] = []
    for f in files:
        meta = parse_atc_filename(f["name"], f["path"])
        if not meta:
            continue
        if meta.sector not in sectors_list:
            continue

        # time window intersection
        # if start = 13:00, stop = 15:00 -> picks files 13:00 and 14:00 only 
        # if start = 13:00, stop = 13:45 -> still includes 13:00 (overlap)
        fs, fe = file_interval(meta)
        if not intervals_intersect(fs, fe, start, stop):
            continue

        # frequency filters
        if (min_mhz is not None) and (meta.freq_mhz < min_mhz):
            continue
        if (max_mhz is not None) and (meta.freq_mhz > max_mhz):
            continue

        metas.append(meta)

    if not metas:
        return []

    if limit:
        metas = metas[:limit]

    # loader
    def _load(meta: "ATCFileMeta") -> Tuple["ATCFileMeta", AudioSegment]:
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
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_load, m): m for m in metas}
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    catalog.append(fut.result())
                except Exception as e:
                    print(f"[warn] audio load failed: {m.original_name} -> {e}")
                if pbar: pbar.update(1)
    else:
        for m in metas:
            try:
                catalog.append(_load(m))
            except Exception as e:
                print(f"[warn] audio load failed: {m.original_name} -> {e}")
            if pbar: pbar.update(1)

    if pbar:
        pbar.close()

    catalog.sort(key=lambda item: item[0].dt)
    return catalog

# ----- SPEECH SEGMENT RECOGNITION -----

Segment = Tuple[float, float, AudioSegment]  # (start_s, end_s, audio)

def default_segmenter(audio: AudioSegment) -> List[Segment]:
    """Return the whole audio as one segment, same format as VAD output."""
    # match your VAD preprocessing for consistency
    seg = audio.set_channels(1).set_frame_rate(16000)
    duration_s = len(seg) / 1000.0  # pydub duration in ms
    return [(0.0, duration_s, seg)]

def vad_split_segments(
    audio: AudioSegment,
    *,
    target_sr=16000,
    frame_ms=20,               # 10/20/30ms only for WebRTC
    aggressiveness=0,          # 0=least strict (more speech), 3=most strict
    min_speech_ms=300,         # discard segments shorter than this
    min_silence_ms=200,        # merge gaps shorter than this
    pre_pad_ms=120,            # add a bit of context at start
    post_pad_ms=150,           # and at end
    normalize_dbfs=-20.0,      # loudness target before VAD
    use_compressor=True,       # dynamic range compression
) -> list[tuple[float, float, AudioSegment]]:
    
    # 1: Prepare audio: mono, target SR, band-limit, normalize, (optional) compress
    a = audio.set_channels(1).set_frame_rate(target_sr)
    a = a.high_pass_filter(150).low_pass_filter(3800) # voice band
    # normalize to target dBFS
    if a.dBFS != float("-inf"):
        gain = normalize_dbfs - a.dBFS
        a = a.apply_gain(gain)
    if use_compressor:
        a = effects.compress_dynamic_range(a, threshold=-18.0, ratio=3.0, attack=5, release=50)

    samples = np.array(a.get_array_of_samples()).astype(np.int16)
    sr = a.frame_rate
    assert frame_ms in (10, 20, 30)
    frame_size = int(sr * frame_ms / 1000)  # samples per frame

    vad = webrtcvad.Vad(aggressiveness)

    # 2: Frame-level decisions
    n_frames = max(0, (len(samples) // frame_size))
    voiced = []
    for i in range(n_frames):
        frm = samples[i*frame_size:(i+1)*frame_size]
        voiced.append(vad.is_speech(frm.tobytes(), sr))

    # 3: Hysteresis smoothing (start/end guards)
    # require k_on voiced frames to start, k_off unvoiced to end
    k_on  = max(1, int(60 / frame_ms))   # ~60ms to start
    k_off = max(1, int(120 / frame_ms))  # ~120ms to stop

    segments = []
    in_seg = False
    seg_start_f = 0
    unvoiced_run = 0
    voiced_run = 0

    for i, v in enumerate(voiced):
        if v:
            voiced_run += 1
            unvoiced_run = 0
            if not in_seg and voiced_run >= k_on:
                in_seg = True
                seg_start_f = i - k_on + 1
        else:
            voiced_run = 0
            if in_seg:
                unvoiced_run += 1
                if unvoiced_run >= k_off:
                    in_seg = False
                    seg_end_f = i - k_off + 1
                    segments.append((seg_start_f, seg_end_f))

    if in_seg:
        segments.append((seg_start_f, n_frames))

    # 4: Merge short gaps & apply padding
    def f2ms(f): return f * frame_ms
    merged = []
    for s, e in segments:
        if not merged:
            merged.append([s, e])
            continue
        prev_s, prev_e = merged[-1]
        gap_ms = f2ms(s - prev_e)
        if gap_ms < min_silence_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # padding in frames
    pre_pad_f  = int(pre_pad_ms  / frame_ms)
    post_pad_f = int(post_pad_ms / frame_ms)

    out = []
    for s, e in merged:
        # enforce minimum duration
        if f2ms(e - s) < min_speech_ms:
            continue
        s_pad = max(0, s - pre_pad_f)
        e_pad = min(n_frames, e + post_pad_f)
        t0_ms = s_pad * frame_ms
        t1_ms = e_pad * frame_ms
        seg_audio = a[t0_ms:t1_ms]
        out.append((t0_ms/1000.0, t1_ms/1000.0, seg_audio))

    return out

def pyannote_vad_segments(
    audio: AudioSegment,
    *,
    model_id: str = "pyannote/voice-activity-detection",
    hf_token: str | None = None,
    target_sr: int = 16000,
    min_speech_ms: int = 300,
    min_silence_ms: int = 400,
    pre_pad_ms: int = 200,
    post_pad_ms: int = 150,
    normalize_dbfs: float = -20.0,
    use_compressor: bool = True,
    device: str | None = None,
) -> List[Segment]:
    # --- Preprocess (same as WebRTC path) ---
    a = audio.set_channels(1).set_frame_rate(target_sr)
    a = a.high_pass_filter(150).low_pass_filter(3800)
    if a.dBFS != float("-inf"):
        a = a.apply_gain(normalize_dbfs - a.dBFS)
    if use_compressor:
        a = effects.compress_dynamic_range(a, threshold=-18.0, ratio=3.0, attack=5, release=50)

    # float32 [-1, 1]
    samples = np.array(a.get_array_of_samples()).astype(np.float32)
    if a.sample_width == 2:
        samples /= 32768.0
    else:
        peak = np.max(np.abs(samples)) or 1.0
        samples /= peak
    wav = torch.from_numpy(samples).unsqueeze(0)  # [1, T]
    sr = a.frame_rate

    # --- Load pipeline (reuse your existing HF token) ---
    if hf_token:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    else:
        pipeline = Pipeline.from_pretrained(model_id)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(torch.device(device))

    # --- Run VAD ---
    vad = pipeline({"waveform": wav, "sample_rate": sr})
    # Get speech regions as a Timeline, then simplify to support (merged speech)
    timeline = vad.get_timeline().support()
    segments = [(float(s.start), float(s.end)) for s in timeline]

    # Merge short gaps, enforce min duration, pad
    merged: List[List[float]] = []
    for s, e in segments:
        if not merged or s - merged[-1][1] >= (min_silence_ms / 1000.0):
            merged.append([s, e])
        else:
            merged[-1][1] = e

    out: List[Segment] = []
    dur_s = len(a) / 1000.0
    for s, e in merged:
        if (e - s) * 1000 < min_speech_ms:
            continue
        s_pad = max(0.0, s - pre_pad_ms / 1000.0)
        e_pad = min(dur_s, e + post_pad_ms / 1000.0)
        seg = a[int(s_pad * 1000): int(e_pad * 1000)]
        out.append((s_pad, e_pad, seg))
    return out

def diarize_audio(
    audio: AudioSegment,
    diarization_pipeline,            # pyannote.audio.Pipeline instance
    *,
    target_sr: int = 16000,          # pyannote prefers 16 kHz
    mono: bool = True,               # mono recommended
    min_turn_ms: int = 200,          # drop ultra-short turns
    include_speaker: bool = False    # return speaker labels if True
) -> List[Tuple[float, float, AudioSegment] | Tuple[float, float, AudioSegment, str]]:
    """
    Run speaker diarization on the *entire* AudioSegment.

    Returns:
      - if include_speaker=False (default): List[(start_sec, end_sec, chunk_audio)]
      - if include_speaker=True:           List[(start_sec, end_sec, chunk_audio, speaker_label)]

    Notes:
      - No files written/read.
      - Chunks are cut from the input audio (resampled if requested).
    """
    # 1) ensure model-friendly format
    a = audio
    if mono:
        a = a.set_channels(1)
    if target_sr is not None:
        a = a.set_frame_rate(target_sr)
    a = a.high_pass_filter(300).low_pass_filter(3500) # filter voice band only
    sample_rate = a.frame_rate

    # 2) build waveform tensor for pyannote (float32 in [-1, 1], shape (1, T))
    samples = np.array(a.get_array_of_samples()).astype(np.float32)
    # convert from int PCM range to [-1, 1]
    # pydub sample width is 2 bytes here => int16
    if a.sample_width == 2:
        samples = samples / 32768.0
    else:
        # general fallback: normalize by max(|x|)
        peak = np.max(np.abs(samples)) or 1.0
        samples = samples / peak
    waveform = torch.from_numpy(samples).unsqueeze(0)

    # 3) run diarization
    diar = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # 4) slice turns back into AudioSegments
    results = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        t0, t1 = float(turn.start), float(turn.end)
        if (t1 - t0) * 1000 < min_turn_ms:
            continue
        seg = a[int(t0 * 1000): int(t1 * 1000)]
        if include_speaker:
            results.append((t0, t1, seg, speaker))
        else:
            results.append((t0, t1, seg))

    # If nothing detected (rare), return single full segment
    if not results:
        if include_speaker:
            results.append((0.0, len(a) / 1000.0, a, "UNK"))
        else:
            results.append((0.0, len(a) / 1000.0, a))

    return results