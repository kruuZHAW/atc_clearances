
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import math

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
    
    if not start < stop:
        raise ValueError("Expected start < stop (half-open interval [start, stop)).")
    
    # ---------- 0A. PICK VAD / SEGMENTER ----------
     
    if args.segmenter == "webrtc":
        segmenter = vad_split_segments
        segmenter_kwargs = {}
        print("[info] using WebRTC VAD")
    else:
        segmenter = pyannote_vad_segments
        segmenter_kwargs = {"hf_token": args.hf_token}
        print("[info] using pyannote VAD")
    
    # ---------- 0B. LOAD TRANSCRIBER ----------
    
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    transcriber = WhisperTranscriber(
        model_name=args.whisper_model,
        device=None,   # auto cuda/cpu
        dtype=dtype,
    )
    
    # ---------- 0C. DAY ITERATOR OVER [start, stop) ----------
    
    def day_iter(s: datetime, e: datetime):
        """Yield (day_start, day_end, tag_YYYYMMDD) per calendar day overlapping [s,e)."""
        cur = (s.replace(hour=0, minute=0, second=0, microsecond=0))
        end_anchor = (e.replace(hour=0, minute=0, second=0, microsecond=0))
        # ensure tz-aware midnight
        if cur.tzinfo is None:
            cur = cur.replace(tzinfo=timezone.utc)
        if end_anchor.tzinfo is None:
            end_anchor = end_anchor.replace(tzinfo=timezone.utc)
        while cur <= end_anchor:
            next_midnight = cur + timedelta(days=1)
            win_start = max(s, cur)
            win_stop  = min(e, next_midnight)
            if win_start < win_stop:
                yield win_start, win_stop, cur.strftime("%Y%m%d")
            cur = next_midnight
    
    # ---------- 0D. MUAC POLYGON ----------
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
    W, S, E, N = delta_geom.bounds  
    
    # Add a buffer layer around delta sector
    # 1° lat ≈ 111.32 km; 1° lon ≈ 111.32 km * cos(latitude)
    buffer_km = 50.0 # Roughly 4min at 400kts
    lat_c = delta_geom.centroid.y
    deg_lat = buffer_km / 111.32
    deg_lon = buffer_km / (111.32 * math.cos(math.radians(lat_c)))

    bbox_buffered = (W - deg_lon, S - deg_lat, E + deg_lon, N + deg_lat)  # (west, south, east, north)
    
    # ---------- 1. PER DAY LOOP ----------
    for day_start, day_stop, day_tag in day_iter(start, stop):
        day_dir = (out_dir / day_tag).resolve()
        day_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] === {day_tag} :: {day_start} → {day_stop} ===")

        # ---------- 2. BUILD CATALOG ----------
        print("[info] building catalog ...")
        catalog = build_audio_catalog(
            base_dir=args.base_dir,
            sectors=args.sectors,
            start=day_start,
            stop=day_stop,
            max_workers=args.workers,
            show_progress=args.show_progress,
            target_sr=16000,
            mono=True,
        )
        print(f"[info] {len(catalog)} files in catalog")

        # Save catalog metadata
        save_catalog_json(catalog, str(day_dir / "catalog.json"))
        
        # ---------- 3. ADS-B EXTRACTION ----------
        print("[info] fetching ADS-B ...")

        # If needed: add a clipping to avoid selecting trajectories that only stay in the buffer layer without entering the delta sector
        adsb_traf = extract_adsb(
            start=day_start,
            stop=day_stop,
            bbox=bbox_buffered,
            chunk_minutes=args.chunk_min,
            min_baroalt_m=args.min_fl_m,
            save_parquet=str(day_dir / "adsb.parquet"),
        )

        if adsb_traf is None:
            print(f"[warn] No ADS-B data for {day_tag}; skipping matching.")
            continue

        # ---------- 4. TRANSCRIBER ----------

        print(f"[info] transcribing {day_tag} ...")
        transcripts = transcribe_catalog(
            catalog,
            segmentation_method=segmenter,
            transcriber=transcriber,
            verbose=False,
            **segmenter_kwargs,
        )
        # Save transcripts
        save_transcripts_json(transcripts, str(day_dir / "transcripts.json"))

        # ---------- 5. CALLSIGN MATCHING ----------
        print("[info] building ADS-B time ranges ...")
        
        adsb_ranges = build_timestamp_range(adsb_traf)

        print(f"[info] extracting callsign communications for {day_tag} ...")
        callsign_communications, df = extract_callsign_communications(
            transcripts,
            adsb_traf=adsb_traf,               
            adsb_ranges=adsb_ranges,
            closest_callsign_at_time=closest_callsign_at_time,
            merge_callsign_entities=merge_callsign_entities,
            batch_size=64,
            match_threshold=0.7,
            time_tolerance_s=5*60,
            include_unmatched=True,
            progress=args.show_progress,
            return_df=True,
        )

        # Save callsign results
        save_callsign_comms_json(callsign_communications, str(day_dir / "callsign_comms.json"))
        if df is not None and not df.empty:
            df.to_parquet(day_dir / "callsign_comms.parquet", index=False)

        print(f"[done] Wrote outputs to {day_dir}")


if __name__ == "__main__":
    main()
