#### DEPRECIATED: WORKS WITH THE OLD RECORDING WAY ####

import os
import re
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Iterable, Callable
from pathlib import Path
from pydub import AudioSegment
import webrtcvad
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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

def vad_split_segments(audio, frame_duration_ms=30, aggressiveness=2, min_speech_ms=300) -> List[Segment]:
    # Load and convert audio - ensure 16 kHz mono for VAD
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = audio.high_pass_filter(300).low_pass_filter(3500) # Voice band only to reduce noise
    samples = np.array(audio.get_array_of_samples()).astype(np.int16)
    sample_rate = 16000
    bytes_per_sample = 2

    # Frame configuration
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    vad = webrtcvad.Vad(aggressiveness)

    speech_segments = []
    current_chunk = b''
    start_time = None
    times = []

    for i in range(0, len(samples) - frame_size, frame_size):
        frame = samples[i:i+frame_size]
        frame_bytes = frame.tobytes()

        is_speech = vad.is_speech(frame_bytes, sample_rate)
        timestamp = i / sample_rate

        if is_speech:
            if start_time is None:
                start_time = timestamp
            current_chunk += frame_bytes
        else:
            if current_chunk:
                duration = len(current_chunk) / (sample_rate * bytes_per_sample)
                if duration * 1000 >= min_speech_ms:
                    end_time = timestamp
                    chunk_audio = AudioSegment(
                        data=current_chunk,
                        sample_width=bytes_per_sample,
                        frame_rate=sample_rate,
                        channels=1,
                    )
                    speech_segments.append((start_time, end_time, chunk_audio))
                    times.append((start_time, end_time))
                current_chunk = b''
                start_time = None

    # Handle last chunk
    if current_chunk and start_time is not None:
        end_time = len(samples) / sample_rate
        chunk_audio = AudioSegment(
            data=current_chunk,
            sample_width=bytes_per_sample,
            frame_rate=sample_rate,
            channels=1,
        )
        speech_segments.append((start_time, end_time, chunk_audio))
        times.append((start_time, end_time))

    return speech_segments

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

# ----- TRANSCRIPTION FUNCTIONS (OLD) -----

def audiosegment_to_tensor(audio):
    """Convert pydub.AudioSegment to torch.Tensor and sample rate."""
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0  # normalize to [-1, 1]
    waveform = torch.from_numpy(samples).unsqueeze(0)  # shape [1, num_samples]
    return waveform, audio.frame_rate

MIN_SAMPLES = 160  # ~10 ms at 16k; 

def transcribe_segments(
    speech_segments,
    base_timestamp,
    model_name="jacktol/whisper-medium.en-fine-tuned-for-ATC",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = WhisperProcessor.from_pretrained(model_name)
    model.eval()

    starts = []
    waves = []
    srs = []
    # Prepare batches
    for (start, end, audio_segment) in speech_segments:
        # returns torch.Tensor [1, T] and sr
        waveform, sr = audiosegment_to_tensor(audio_segment)
        # ensure 1-D float32 numpy
        if hasattr(waveform, "detach"):
            arr = waveform.detach().cpu().squeeze().numpy()
        else:
            arr = np.asarray(waveform).squeeze()
        arr = arr.astype(np.float32, copy=False)

        # skip empty or too-short segments
        if arr.ndim != 1 or arr.size < MIN_SAMPLES:
            # print(f"Skipping segment at {start:.2f}s: size={arr.size}")
            continue

        starts.append(float(start))
        waves.append(arr)
        srs.append(int(sr))

    if not waves:
        return []

    # Ensure all SRs identical (Whisper expects a single sampling_rate value)
    sr0 = srs[0]
    if any(sr != sr0 for sr in srs):
        raise ValueError(f"Mixed sampling rates in batch: {set(srs)}")

    # Build inputs with padding + attention mask
    inputs = processor(
        waves,
        sampling_rate=sr0,
        return_tensors="pt",
        padding=True,                # pad to the longest
        return_attention_mask=True,  # <-- important
    )

    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
        )

    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    transcripts = []
    for start, text in zip(starts, texts):
        text = (text or "").strip()
        if not text:
            continue
        time = (base_timestamp + timedelta(seconds=start)).replace(microsecond=0)
        transcripts.append((time, text))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}")

    return transcripts

def transcribe_catalog(
    audio_catalog: Iterable[Tuple[object, AudioSegment]],
    segmentation_method: Optional[Callable[[AudioSegment], List[Segment]]] = None,
) -> List[List[str]]:
    """
    Apply audio segmentation to audio catalog and transcribe to text.
    Returns list of lists: each inner list is the transcripts for one input file.
    """
    transcription: List[List[str]] = []
    segmenter = segmentation_method or default_segmenter

    for (meta, audio) in audio_catalog:
        speech_segments = segmenter(audio)
        transcription.append(transcribe_segments(speech_segments, base_timestamp=meta.dt))

    return transcription