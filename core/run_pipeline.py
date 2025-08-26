
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import torch
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon

from utils.process_audio import (
    build_audio_catalog,
    vad_split_segments,
    pyannote_vad_segments,
)
from utils.transcript_audio import (
    WhisperTranscriber,
    transcribe_catalog,
    extract_adsb,
    extract_callsign_communications,
)
from utils.cs_matching import (
    build_timestamp_range,
    closest_callsign_at_time,
    merge_callsign_entities,
)

# ---------- HELPERS ----------

def save_transcripts_json(transcription, path: str):
    # transcription: List[List[(datetime, str)]]
    serializable = [
        [(ts.isoformat(), text) for ts, text in transcripts]
        for transcripts in transcription
    ]
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

def save_callsign_comms_json(callsign_communications, path: str):
    serializable = {
        cs: [
            {
                "timestamp": entry["timestamp"].isoformat(),
                "sentence": entry["sentence"],
                "detected": entry["detected"],
                "score": entry["score"],
            }
            for entry in entries
        ]
        for cs, entries in callsign_communications.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

def save_catalog_json(catalog, path: str):
    """
    catalog: List[(ATCFileMeta, AudioSegment)]
    We only store metadata to keep the file small.
    """
    rows = []
    for meta, audio in catalog:
        rows.append({
            "sector": meta.sector,
            "dt": meta.dt.isoformat(),
            "freq_hz": meta.freq_hz,
            "freq_mhz": meta.freq_mhz,
            "ext": meta.ext,
            "original_name": meta.original_name,
            "path": meta.path,
            "duration_s": len(audio) / 1000.0,
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

# ---------- MAIN ----------

def main():
    p = argparse.ArgumentParser(description="ATC transcription + ADS-B matching pipeline")
    p.add_argument("--base-dir", required=True, help="Root folder with sector/<Y>/<M>/<D>/ audio files")
    p.add_argument("--sectors", required=True, nargs="+", help="Sectors to include (e.g. delta_low delta_mid delta_high)")
    p.add_argument("--start-utc", required=True, help="Start ISO or 'YYYY-MM-DDTHH:MM:SS' (UTC)")
    p.add_argument("--stop-utc",  required=True, help="Stop  ISO or 'YYYY-MM-DDTHH:MM:SS' (UTC), half-open")
    p.add_argument("--out-dir",   required=True, help="Output directory")

    # VAD choice
    p.add_argument("--segmenter", choices=["webrtc", "pyannote"], default="webrtc")
    p.add_argument("--hf-token", default=None, help="HF token (needed for pyannote VAD)")

    # Performance knobs
    p.add_argument("--workers", type=int, default=64, help="Parallel file loads for catalog")
    p.add_argument("--show-progress", action="store_true", help="Show tqdm progress bars")

    # Whisper model config
    p.add_argument("--whisper-model", default="jacktol/whisper-medium.en-fine-tuned-for-ATC")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")

    # ADS-B options
    p.add_argument("--min-fl-m", type=float, default=10000.0, help="Minimum baro altitude in meters (~FL330)")
    p.add_argument("--chunk-min", type=int, default=60, help="Chunk size in minutes for OpenSky queries")

    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse times
    def parse_iso(s: str) -> datetime:
        # Allow plain "YYYY-MM-DDTHH:MM:SS"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    start = parse_iso(args.start_utc)
    stop  = parse_iso(args.stop_utc)

    # ---------- 1. BUILD CATALOG ----------
    print("[info] building catalog ...")
    catalog = build_audio_catalog(
        base_dir=args.base_dir,
        sectors=args.sectors,
        start=start,
        stop=stop,
        max_workers=args.workers,
        show_progress=args.show_progress,
        target_sr=16000,
        mono=True,
    )
    print(f"[info] {len(catalog)} files in catalog")

    # Save catalog metadata
    save_catalog_json(catalog, str(out_dir / "catalog.json"))

    # ---------- 2. CHOSE SEGMENTER ----------
    if args.segmenter == "webrtc":
        segmenter = vad_split_segments
        segmenter_kwargs = {}  # tweak if needed
        print("[info] using WebRTC VAD")
    else:
        # pyannote VAD needs token if the model is gated; pass via kwargs
        segmenter = pyannote_vad_segments
        segmenter_kwargs = {"hf_token": args.hf_token}
        print("[info] using pyannote VAD")

    # ---------- 3. TRANSCRIBER ----------
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    transcriber = WhisperTranscriber(
        model_name=args.whisper_model,
        device=None,   # auto cuda/cpu
        dtype=dtype,
    )

    print("[info] transcribing ...")
    transcripts = transcribe_catalog(
        catalog,
        segmentation_method=segmenter,
        transcriber=transcriber,
        verbose=False,
        **segmenter_kwargs,
    )
    # Save transcripts
    save_transcripts_json(transcripts, str(out_dir / "transcripts.json"))

    # ---------- 4. ADS-B EXTRACTION ----------
    print("[info] fetching ADS-B ...")
    # MUAC delta polygon
    delta_coords = [
        (53.454167, 3.606111),
        (52.733333, 5.583333),
        (52.663333, 7.168889),
        (51.193611, 5.521389),
        (51.607778, 3.171944),
        (51.480556, 3.171944),
        (51.636944, 2.500000),
        (51.455556, 2.500000),
        (51.500000, 2.000000),
        (51.950556, 2.356389),
    ]
    delta_geom = ShapelyPolygon([(lon, lat) for lat, lon in delta_coords])
    bbox = delta_geom.bounds  # (W, S, E, N)

    adsb_traf = extract_adsb(
        start=start,
        stop=stop,
        bbox=bbox,
        chunk_minutes=args.chunk_min,
        min_baroalt_m=args.min_fl_m,
        save_parquet=str(out_dir / "adsb.parquet"),
        show_progress=args.show_progress,
    )

    if adsb_traf is None:
        print("[warn] No ADS-B data returned; skipping matching.")
        return

    # ---------- 5. CALLSIGN MATCHING ----------
    print("[info] building ADS-B time ranges ...")
    adsb_ranges = build_timestamp_range(adsb_traf)

    print("[info] extracting callsign communications ...")
    callsign_communications, df = extract_callsign_communications(
        transcripts,
        adsb_traf=adsb_traf,               # matches your current function’s signature
        adsb_ranges=adsb_ranges,
        closest_callsign_at_time=closest_callsign_at_time,
        merge_callsign_entities=merge_callsign_entities,
        batch_size=64,
        match_threshold=0.7,
        time_tolerance_s=60,
        include_unmatched=True,
        progress=args.show_progress,
        return_df=True,
    )

    # Save callsign results
    save_callsign_comms_json(callsign_communications, str(out_dir / "callsign_comms.json"))
    if df is not None and not df.empty:
        df.to_parquet(out_dir / "callsign_comms.parquet", index=False)

    print("[done] outputs written to:", out_dir)


if __name__ == "__main__":
    main()
