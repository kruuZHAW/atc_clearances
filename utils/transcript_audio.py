import re
import os
from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Callable, Iterable, List, Tuple, Optional, Dict, Any
import time

import torch
from utils.process_audio import default_segmenter, AudioCatalog
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForTokenClassification, pipeline

from shapely.geometry import Polygon as ShapelyPolygon

from traffic.data import opensky
from traffic.core import Traffic, Flight
from pyopensky.schema import StateVectorsData4

# TODO: Make the transcription everytime I load a new file, and not everything at the same time

MIN_SAMPLES = 160

def audiosegment_to_tensor(audio):
    """Convert pydub.AudioSegment to torch.Tensor and sample rate."""
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0  # normalize to [-1, 1]
    waveform = torch.from_numpy(samples).unsqueeze(0)  # shape [1, num_samples]
    return waveform, audio.frame_rate

class WhisperTranscriber:
    def __init__(
        self,
        model_name="jacktol/whisper-medium.en-fine-tuned-for-ATC",
        device=None,
        dtype=torch.float16,  # fp16 is fine on most GPUs
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model.eval()
        self.dtype = dtype

    def transcribe_segments(self, speech_segments, base_timestamp):
        t0 = time.time()

        starts, waves, srs = [], [], []
        for (start, end, audio_segment) in speech_segments:
            waveform, sr = audiosegment_to_tensor(audio_segment)
            arr = waveform.detach().cpu().squeeze().numpy().astype(np.float32)
            if arr.size < MIN_SAMPLES:
                continue
            starts.append(float(start))
            waves.append(arr)
            srs.append(int(sr))
        if not waves:
            return []

        # print(f"[DEBUG] preprocessing took {time.time() - t0:.2f}s")
        t1 = time.time()

        # Preprocess with WhisperProcessor
        inputs = self.processor(
            waves,
            sampling_rate=srs[0],
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        # print(f"[DEBUG] feature extraction took {time.time() - t1:.2f}s")
        t2 = time.time()

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
            )

        # print(f"[DEBUG] model.generate() took {time.time() - t2:.2f}s")
        t3 = time.time()

        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print(f"[DEBUG] decoding took {time.time() - t3:.2f}s")

        transcripts = [
            ((base_timestamp + timedelta(seconds=start)).replace(microsecond=0), text.strip())
            for start, text in zip(starts, texts) if text.strip()
        ]

        # print(f"[DEBUG] total time per file: {time.time() - t0:.2f}s")
        return transcripts


def transcribe_catalog(
    audio_catalog: AudioCatalog,
    segmentation_method: Optional[Callable] = None,
    transcriber=None,
    verbose: bool = False,
    segmentation_kwargs: Optional[dict] = None,
) -> List[List[str]]:
    
    """
    Apply audio segmentation to audio catalog and transcribe to text.
    Returns list of lists: each inner list is the transcripts for one input file.
    """
    segmenter = segmentation_method or default_segmenter
    if transcriber is None:
        raise ValueError("Please provide a `transcriber` instance.")
    segmentation_kwargs = segmentation_kwargs or {}

    all_transcripts: List[List[str]] = []
    for (meta, audio) in tqdm(audio_catalog, total=len(audio_catalog),
                              desc="Transcribing", unit="audio chunk"):
        speech_segments = segmenter(audio, **segmentation_kwargs)
        trs = transcriber.transcribe_segments(speech_segments, base_timestamp=meta.dt)
        all_transcripts.append(trs)
        if verbose:
            for (t, txt) in trs:
                print(f"[{t:%Y-%m-%d %H:%M:%S}] {txt}")
    return all_transcripts

def extract_adsb(
    *,
    start,
    stop,
    bbox: Optional[Tuple[float, float, float, float]] = None, #bbox=(W,S,E,N)
    polygon: Optional[ShapelyPolygon] = None,
    chunk_minutes: int = 30, # Time window chunking
    selected_columns: Optional[List] = None,
    min_baroalt_m: Optional[float] = None, # Altitude filtering
    callsign_regex: Optional[str] = None, # Filtering bad callsigns ex: r"^[A-Z]{3}[0-9]+[A-Z]*$"
    save_parquet: Optional[str] = None,  # path like "data/extract.parquet"
) -> Optional[Traffic]:
    """
    Fetch ADS-B state vectors from OpenSky via `traffic` in time chunks.
    Returns a `Traffic` object or None if no data. Optionally writes Parquet.
    """

    if bbox is None:
        if polygon is None:
            raise ValueError("Provide either `bbox=(W,S,E,N)` or `polygon=ShapelyPolygon`.")
        bbox = polygon.bounds  # (W, S, E, N)

    if selected_columns is None:
        selected_columns = [
            StateVectorsData4.time,       
            StateVectorsData4.lat,
            StateVectorsData4.lon,
            StateVectorsData4.baroaltitude,
            StateVectorsData4.velocity,
            StateVectorsData4.heading,
            StateVectorsData4.icao24,
            StateVectorsData4.callsign,
        ]

    # optional callsign filter
    cs_pattern = re.compile(callsign_regex) if callsign_regex else None

    def _keep_callsign(f: Flight) -> Optional[Flight]:
        if not isinstance(f, Flight):
            return None
        if cs_pattern is None:
            return f
        cs = (f.callsign or "").strip().upper()
        return f if cs_pattern.match(cs) else None

    # iterate chunks
    chunks = pd.date_range(start, stop, freq=f"{chunk_minutes}min", inclusive="left")
    collected: List[Traffic] = []

    for t0 in chunks:
        t1 = min(t0 + timedelta(minutes=chunk_minutes), stop)

        try:
            if min_baroalt_m: 
                trf = opensky.history(
                    t0,
                    t1,
                    StateVectorsData4.baroaltitude >= min_baroalt_m,
                    bounds=bbox,                      
                    selected_columns=selected_columns, 
                )
            else:
                trf = opensky.history(
                    start=t0,
                    stop=t1,
                    bounds=bbox,                      
                    selected_columns=selected_columns, 
                )
        except Exception as e:
            print(f"[warn] OpenSky query failed for {t0}–{t1}: {e}")
            continue

        if trf is None:
            continue
        
        if trf is not None:
            collected.append(trf)
            
    if not collected:
        print("No data returned for the requested period/filters.")
        return None

    adsb_traf = collected[0]
    for tr in collected[1:]:
        adsb_traf = adsb_traf + tr
    adsb_traf = adsb_traf.iterate_lazy().pipe(_keep_callsign).assign_id().eval() # To avoid splitting flights across 2 consecutive hours

    if save_parquet:
        df = adsb_traf.data
        os.makedirs(os.path.dirname(save_parquet), exist_ok=True)
        df.to_parquet(save_parquet)

    return adsb_traf

def extract_callsign_communications(
    transcripts: List[List[Tuple[Any, str]]],
    *,
    adsb_traf,                        
    adsb_ranges,                     # precomputed ranges for callsigns
    closest_callsign_at_time: Callable[..., Optional[Dict[str, Any]]],
    merge_callsign_entities: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    model_name: str = "Jzuluaga/bert-base-ner-atc-en-atco2-1h",
    batch_size: int = 32,
    device: Optional[str] = None,    # "cuda" / "cpu" / None->auto
    aggregation_strategy: str = "first",
    match_threshold: float = 0.7,
    time_tolerance_s: int = 60,
    include_unmatched: bool = False, # optionally collect unmatched detections
    progress: bool = True,
    return_df: bool = False,         # also return a flattened DataFrame
):
    """
    Runs NER on transcripts and associates callsigns to ADS-B context.

    Returns:
      - callsign_communications: Dict[str, List[dict]] mapping matched callsign -> list of comm dicts
      - (optional) df: pandas DataFrame with rows of matched (and optionally unmatched) items
    """
    # 0) Prepare pipeline
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_device = 0 if device == "cuda" else -1

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy=aggregation_strategy, device=hf_device)

    # 1) Flatten transcripts into a list of (timestamp, sentence)
    flat: List[Tuple[Any, str]] = [pair for group in transcripts for pair in group]
    # Keep index to map back after batch inference
    indices = [i for i, (_ts, txt) in enumerate(flat) if isinstance(txt, str) and txt.strip()]
    inputs  = [flat[i][1] for i in indices]

    callsign_communications: Dict[str, List[dict]] = defaultdict(list)
    rows = []

    # 2) Run NER in batches
    def _maybe_merge(ents):
        return merge_callsign_entities(ents) if merge_callsign_entities is not None else ents

    rng = range(0, len(inputs), batch_size)
    iterator = tqdm(rng, desc="NER (callsigns)", unit="batch") if progress else rng

    for start in iterator:
        batch_texts = inputs[start:start + batch_size]
        batch_out   = nlp(batch_texts)

        for j, ents in enumerate(batch_out):
            idx = indices[start + j]
            timestamp, sentence = flat[idx]

            # merge callsign sub-entities if helper is provided
            ents = _maybe_merge(ents)
            # select callsign entities (robust to case)
            callsigns = [e for e in ents if str(e.get("entity_group", "")).lower() == "callsign"]

            result = None
            if callsigns:
                # Call your matcher (expects list of entities)
                result = closest_callsign_at_time(
                    ner_callsigns=callsigns,
                    traffic=adsb_traf,
                    adsb_ranges=adsb_ranges,
                    comm_time=timestamp,
                    match_threshold=match_threshold,
                    time_tolerance_s=time_tolerance_s,
                )

            if result:
                cs = result["best_context_match"]
                entry = {
                    "callsign": cs,
                    "timestamp": timestamp,
                    "sentence": sentence,
                    "detected": result.get("ner_detected_callsign"),
                    "score": result.get("match_score"),
                }
                callsign_communications[cs].append(entry)
                rows.append({**entry, "matched": True})
            else:
                if include_unmatched:
                    rows.append({
                        "callsign": None,
                        "timestamp": timestamp,
                        "sentence": sentence,
                        "detected": callsigns[0]["word"] if callsigns else None,
                        "score": None,
                        "matched": False,
                    })

    if return_df:
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["callsign", "timestamp", "sentence", "detected", "score", "matched"]
        )
        return callsign_communications, df

    return callsign_communications