import os
from typing import Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

from traffic.core.structure import Airport
from traffic.core.airspace import Airspace

from pyannote.audio import Pipeline
diarization_pipline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

import re
import requests
from bs4 import BeautifulSoup

from traffic.data import opensky, airports, eurofirs, faa
from geopy.distance import distance
from geopy.point import Point

import json
import wave
import webrtcvad
from pydub import AudioSegment
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Core data model to represent each audio + ads-b + transcript data
@dataclass
class RecordingSession:
    channel: str
    start_time: datetime
    end_time: datetime
    audio_path: Path
    airport: Optional[Union[str, Airport]] = None
    artcc: Optional[Union[str, Airspace]] = None
    transcript_path: Optional[Path] = None
    transcription: Optional[str] = None
    adsb_data: Optional[pd.DataFrame] = None
    adsb_path: Optional[Path] = None
    
    def __post_init__(self):
        if (self.airport is None and self.artcc is None) or (self.airport and self.artcc):
            raise ValueError("You must provide exactly one of 'airport' or 'artcc', but not both.")
    
def download_archive_audio(session: RecordingSession, save_path="./audio", use_cache=True) -> RecordingSession:
    """
    Downloads audio from LiveATC and saves path to session.audio_path.
    If use_cache is True, skips download if file already exists.
    """
    import requests
    from bs4 import BeautifulSoup

    date = session.start_time.strftime("%b-%d-%Y")
    time = session.start_time.strftime("%H%M") + "Z"

    # Look up archive ID from LiveATC
    page = requests.get(f'https://www.liveatc.net/archive.php?m={session.channel}')
    soup = BeautifulSoup(page.content, 'html.parser')
    archive_identifier = soup.find('option', selected=True).attrs['value']
    filename = f'{archive_identifier}-{date}-{time}.mp3'

    local_path = Path(save_path) / filename
    session.audio_path = local_path

    if use_cache and local_path.exists():
        print(f"[audio] Using cached file: {local_path}")
        return session

    # Build full download URL
    code = re.sub(r'\d+', '', session.channel.split("_")[0])
    url = f'https://archive.liveatc.net/{code}/{filename}'
    print(f"[audio] Downloading from {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}

    os.makedirs(save_path, exist_ok=True)

    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[audio] Download successful: {local_path}")
        else:
            print(f"[audio] Download failed: {response.status_code} {response.reason}")
            session.audio_path = None
    except Exception as e:
        print(f"[audio] Error during download: {e}")
        session.audio_path = None

    return session

def download_adsb(session: RecordingSession,
                  time_buffer: int = 10,
                  half_side_nm: Optional[float] = 50,
                  save_path: str = "./adsb",
                  use_cache: bool = True) -> RecordingSession:
    """
    Downloads ADS-B data using OpenSky and saves it as a Parquet file.

    - If airport is given: clips to a square box around the airport (requires half_side_nm).
    - If ARTCC is given: uses the airspace's pre-defined bounds directly (airspace.bounds).
    - Filters by arrivals/departures only when airport is used.
    """
    if session.airport:
        if isinstance(session.airport, str):
            session.airport = airports[session.airport]
        
        if half_side_nm is None:
            raise ValueError("half_side_nm must be provided when using an airport.")

        label = session.airport.icao
        center = Point(session.airport.latitude, session.airport.longitude)

        # Bounding box around the airport
        north = distance(nautical=half_side_nm).destination(center, bearing=0)
        south = distance(nautical=half_side_nm).destination(center, bearing=180)
        east = distance(nautical=half_side_nm).destination(center, bearing=90)
        west = distance(nautical=half_side_nm).destination(center, bearing=270)

        bounds = (west.longitude, south.latitude, east.longitude, north.latitude)

    elif session.artcc:
        if isinstance(session.artcc, str):
            if session.artcc in eurofirs.data.designator.tolist():
                session.artcc = eurofirs[session.artcc]
            elif session.artcc in faa.airspace_boundary.data.designator.tolist():
                session.artcc = faa.airspace_boundary[session.artcc]
            else:
                raise ValueError("Airspace not found in Eurocontrol and FAA datasets.")
            
        label = session.artcc.designator  
        bounds = session.artcc.bounds 

        if half_side_nm is not None:
            print("[adsb] Ignoring 'half_side_nm' since ARTCC bounds are defined by airspace.bounds")

    else:
        raise ValueError("RecordingSession must have either 'airport' or 'artcc' defined.")

    # Generate file name and output path
    date_str = session.start_time.strftime("%b-%d-%Y")
    time_str = session.start_time.strftime("%H%M") + "Z"
    filename = f"{label}-{session.channel}-{date_str}-{time_str}.parquet"

    os.makedirs(save_path, exist_ok=True)
    adsb_file = Path(save_path) / filename

    if use_cache and adsb_file.exists():
        session.adsb_data = pd.read_parquet(adsb_file)
        session.adsb_path = adsb_file
        print(f"[adsb] Loaded from cache: {adsb_file}")
        return session

    try:
        if session.airport:
            adsb_df = opensky.history(
                start=session.start_time - timedelta(minutes=time_buffer),
                stop=session.end_time + timedelta(minutes=time_buffer),
                bounds=bounds,
                airport=label
            )
        else:
            adsb_df = opensky.history(
                start=session.start_time - timedelta(minutes=time_buffer),
                stop=session.end_time + timedelta(minutes=time_buffer),
                bounds=bounds
            )

        if adsb_df is None or len(adsb_df) == 0:
            print(f"[adsb] No data found for {label} from {session.start_time} to {session.end_time}")
            session.adsb_data = None
            session.adsb_path = None
        else:
            adsb_df.to_parquet(adsb_file, index=False)
            print(f"[adsb] Saved {len(adsb_df)} records to {adsb_file}")
            session.adsb_data = adsb_df
            session.adsb_path = adsb_file

    except Exception as e:
        print(f"[adsb] Failed to download ADS-B data: {e}")
        session.adsb_data = None
        session.adsb_path = None

    return session

## DEPRECIATED
# def split_audio_by_vad(session: RecordingSession, chunk_dir="./chunks", use_cache=True) -> list[Path]:
#     """
#     Splits session.audio_path into voice activity chunks using WebRTC VAD.
#     Returns list of saved .wav paths.
#     """
#     if session.audio_path is None or not session.audio_path.exists():
#         print("[vad] No audio found, skipping VAD splitting.")
#         return []

#     # Format consistent with other outputs
#     date_str = session.start_time.strftime("%b-%d-%Y")
#     time_str = session.start_time.strftime("%H%M") + "Z"
#     if session.airport:
#         base_name = f"{session.airport.icao}-{session.channel}-{date_str}-{time_str}"
#     elif session.artcc:
#         base_name = f"{session.artcc.designator}-{session.channel}-{date_str}-{time_str}"
    
#     chunk_output_dir = Path(chunk_dir) / base_name
#     meta_path = chunk_output_dir / "vad_chunks_metadata.json"
#     os.makedirs(chunk_output_dir, exist_ok=True)

#     # Use cache: skip if already split
#     if use_cache and any(chunk_output_dir.glob("speech_*.wav")):
#         print(f"[vad] Using cached audio chunks in {chunk_output_dir}")
#         return sorted(chunk_output_dir.glob("speech_*.wav"))
    
#     if use_cache and meta_path.exists():
#         with open(meta_path) as f:
#             chunk_meta = json.load(f)
#         return [(chunk_output_dir / entry["path"], entry["offset"]) for entry in chunk_meta]

#     # Step 1: Convert MP3 to WAV (16kHz mono)
#     temp_wav = chunk_output_dir / "temp_audio.wav"
#     audio = AudioSegment.from_file(session.audio_path).set_channels(1).set_frame_rate(16000)
#     audio.export(temp_wav, format="wav")

#     # Step 2: Run VAD to chunk the audio
#     voiced_chunks = []
#     chunk_meta = []
#     with wave.open(str(temp_wav), 'rb') as wf:
#         vad = webrtcvad.Vad(2)
#         frame_duration = 30  # ms
#         sample_rate = wf.getframerate()
#         frame_bytes = int(sample_rate * frame_duration / 1000) * 2  # 16-bit
#         frames = wf.readframes(wf.getnframes())

#         i = 0
#         voiced = False
#         chunk = b''
#         min_samples = 16000  # 1s

#         for offset in range(0, len(frames), frame_bytes):
#             frame = frames[offset:offset + frame_bytes]
#             if len(frame) < frame_bytes:
#                 break

#             is_speech = vad.is_speech(frame, sample_rate)
#             if is_speech:
#                 if not voiced:
#                     voiced = True
#                     chunk = b''
#                 chunk += frame
#             else:
#                 if voiced:
#                     voiced = False
#                     if len(chunk) >= min_samples * 2:
#                         start_sec = offset / (sample_rate * 2)
#                         fname = chunk_output_dir / f"speech_{i:03d}.wav"
#                         segment = AudioSegment(
#                             data=chunk,
#                             sample_width=2,
#                             frame_rate=16000,
#                             channels=1
#                         )
#                         segment.export(fname, format="wav")
#                         chunk_meta.append({"path": str(fname.name), "offset": start_sec})
#                         voiced_chunks.append((fname, start_sec))
#                         print(f"[vad] Saved chunk {i+1}: {fname}")
#                         i += 1

#     # Medata data that stores the offset from start in seconds.
#     with open(meta_path, "w") as f:
#         json.dump(chunk_meta, f, indent=2)

#     # Clean up temp
#     if temp_wav.exists():
#         temp_wav.unlink()

#     return voiced_chunks # (chunk_path, offset_in_seconds)

def split_audio_by_vad(session: RecordingSession, chunk_dir="./chunks", use_cache=True) -> list[tuple[Path, float]]:
    """
    Splits session.audio_path into voice activity chunks using WebRTC VAD,
    enhanced with speaker diarization to refine segmentation.

    Returns list of (chunk_path, offset_in_seconds)
    """
    if session.audio_path is None or not session.audio_path.exists():
        print("[vad] No audio found, skipping VAD splitting.")
        return []

    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    import wave, os, json
    from pathlib import Path

    # Format output path
    date_str = session.start_time.strftime("%b-%d-%Y")
    time_str = session.start_time.strftime("%H%M") + "Z"
    if session.airport:
        base_name = f"{session.airport.icao}-{session.channel}-{date_str}-{time_str}"
    elif session.artcc:
        base_name = f"{session.artcc.designator}-{session.channel}-{date_str}-{time_str}"
    else:
        base_name = f"unknown-{date_str}-{time_str}"

    chunk_output_dir = Path(chunk_dir) / base_name
    meta_path = chunk_output_dir / "vad_chunks_metadata.json"
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Use cache
    if use_cache and any(chunk_output_dir.glob("speech_*.wav")):
        print(f"[vad] Using cached audio chunks in {chunk_output_dir}")
        return sorted((p, 0.0) for p in chunk_output_dir.glob("speech_*.wav"))
    
    if use_cache and meta_path.exists():
        with open(meta_path) as f:
            chunk_meta = json.load(f)
        return [(chunk_output_dir / entry["path"], entry["offset"]) for entry in chunk_meta]

    # Step 1: Convert MP3 to WAV (16kHz mono)
    temp_wav = chunk_output_dir / "temp_audio.wav"
    audio = AudioSegment.from_file(session.audio_path).set_channels(1).set_frame_rate(16000)
    audio.export(temp_wav, format="wav")

    # Step 2: VAD
    import webrtcvad
    vad = webrtcvad.Vad(2)
    voiced_chunks = []
    chunk_meta = []
    with wave.open(str(temp_wav), 'rb') as wf:
        frame_duration = 30  # ms
        sample_rate = wf.getframerate()
        frame_bytes = int(sample_rate * frame_duration / 1000) * 2  # 16-bit PCM
        frames = wf.readframes(wf.getnframes())

        i = 0
        voiced = False
        chunk = b''
        min_samples = 8000  # 0.5s

        for offset in range(0, len(frames), frame_bytes):
            frame = frames[offset:offset + frame_bytes]
            if len(frame) < frame_bytes:
                break

            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                if not voiced:
                    voiced = True
                    chunk = b''
                chunk += frame
            else:
                if voiced:
                    voiced = False
                    if len(chunk) >= min_samples * 2:
                        start_sec = offset / (sample_rate * 2)
                        full_segment = AudioSegment(
                            data=chunk,
                            sample_width=2,
                            frame_rate=16000,
                            channels=1
                        )
                        temp_chunk_path = chunk_output_dir / f"temp_chunk_{i:03d}.wav"
                        full_segment.export(temp_chunk_path, format="wav")

                        # --- Speaker diarization here ---
                        try:
                            diarization = diarization_pipline(str(temp_chunk_path))
                            for j, (turn, _, _) in enumerate(diarization.itertracks(yield_label=True)):
                                segment_audio = full_segment[turn.start * 1000: turn.end * 1000]
                                out_path = chunk_output_dir / f"speech_{i:03d}_{j:02d}.wav"
                                segment_audio.export(out_path, format="wav")
                                chunk_meta.append({"path": str(out_path.name), "offset": start_sec + float(turn.start)})
                                voiced_chunks.append((out_path, start_sec + float(turn.start)))
                                print(f"[vad+spk] Saved chunk {i}_{j}: {out_path}")
                        except Exception as e:
                            print(f"[vad+spk] Diarization failed on chunk {i}: {e}")
                            # fallback: save whole chunk
                            out_path = chunk_output_dir / f"speech_{i:03d}.wav"
                            full_segment.export(out_path, format="wav")
                            chunk_meta.append({"path": str(out_path.name), "offset": start_sec})
                            voiced_chunks.append((out_path, start_sec))
                        i += 1

    with open(meta_path, "w") as f:
        json.dump(chunk_meta, f, indent=2)

    temp_wav.unlink(missing_ok=True)

    return voiced_chunks

def split_audio_by_diarization(session: RecordingSession, chunk_dir="./chunks", use_cache=True) -> list[tuple[Path, float]]:
    """
    Splits session.audio_path into speaker-specific chunks using pyannote speaker diarization.
    Returns list of (chunk_path, offset_in_seconds)
    """
    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    from pathlib import Path
    import os, json

    # Prepare output paths
    date_str = session.start_time.strftime("%b-%d-%Y")
    time_str = session.start_time.strftime("%H%M") + "Z"
    base_name = (
        f"{session.airport.icao}-{session.channel}-{date_str}-{time_str}"
        if session.airport else
        f"{session.artcc.designator}-{session.channel}-{date_str}-{time_str}"
    )

    chunk_output_dir = Path(chunk_dir) / base_name
    meta_path = chunk_output_dir / "vad_chunks_metadata.json"
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Use cache
    if use_cache and meta_path.exists():
        with open(meta_path) as f:
            chunk_meta = json.load(f)
        return [(chunk_output_dir / entry["path"], entry["offset"]) for entry in chunk_meta]

    # Convert audio to 16kHz mono WAV
    audio = AudioSegment.from_file(session.audio_path).set_channels(1).set_frame_rate(16000)
    temp_wav = chunk_output_dir / "full_audio.wav"
    audio.export(temp_wav, format="wav")

    # Run diarization on full audio
    print("[diarization] Running full-audio speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
    diarization = pipeline(str(temp_wav))

    # Export speaker segments
    chunk_meta = []
    voiced_chunks = []
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        segment = audio[turn.start * 1000: turn.end * 1000]
        out_path = chunk_output_dir / f"speech_{i:03d}.wav"
        segment.export(out_path, format="wav")
        offset_sec = float(turn.start)
        chunk_meta.append({"path": str(out_path.name), "offset": offset_sec})
        voiced_chunks.append((out_path, offset_sec))
        print(f"[diarization] Saved {out_path} ({speaker})")

    with open(meta_path, "w") as f:
        json.dump(chunk_meta, f, indent=2)

    temp_wav.unlink(missing_ok=True)

    return voiced_chunks

def transcribe_audio(session: RecordingSession, save_path="./transcripts", use_cache=True) -> RecordingSession:
    """
    Transcribes audio from session.audio_path using fine-tuned Whisper model.
    Automatically splits audio via VAD and transcribes each chunk.
    Saves combined transcript to disk and updates session.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained("jacktol/whisper-medium.en-fine-tuned-for-ATC")
    processor = WhisperProcessor.from_pretrained("jacktol/whisper-medium.en-fine-tuned-for-ATC")

    if session.audio_path is None or not session.audio_path.exists():
        print("[transcribe] No audio file found, skipping.")
        return session

    date_str = session.start_time.strftime("%b-%d-%Y")
    time_str = session.start_time.strftime("%H%M") + "Z"
    if session.airport:
        filename = f"{session.airport.icao}-{session.channel}-{date_str}-{time_str}.txt"
    elif session.artcc:
        filename = f"{session.artcc.designator}-{session.channel}-{date_str}-{time_str}.txt"

    os.makedirs(save_path, exist_ok=True)
    transcript_file = Path(save_path) / filename

    if use_cache and transcript_file.exists():
        session.transcription = transcript_file.read_text()
        session.transcript_path = transcript_file
        print(f"[transcribe] Loaded transcript from cache: {transcript_file}")
        return session

    try:
        # Split audio by VAD
        chunk_paths = split_audio_by_vad(session, use_cache=use_cache)
        if not chunk_paths:
            raise RuntimeError("No voiced chunks found for transcription.")

        results = []
        for i, (path, offset_sec) in enumerate(sorted(chunk_paths)):
            # Tracking communication time
            chunk_time = session.start_time + timedelta(seconds=offset_sec)
            timestamp_str = chunk_time.strftime("[%H:%M:%S]") # Not the day as the audio are only 30min long
            
            audio_input, _ = torchaudio.load(path)
            inputs = processor(audio_input.squeeze(), sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features
            # Specify padding tokens
            attention_mask = inputs.get("attention_mask", None)
            generated_ids = model.generate(input_features, attention_mask=attention_mask)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            results.append(f"{timestamp_str} {transcription}")
            print(f"[transcribe] Chunk {i+1}/{len(chunk_paths)}: {timestamp_str} {transcription}")

        final_transcript = "\n".join(results)
        transcript_file.write_text(final_transcript)
        session.transcription = final_transcript
        session.transcript_path = transcript_file
        print(f"[transcribe] Saved transcript to {transcript_file}")

    except Exception as e:
        print(f"[transcribe] Failed to transcribe audio: {e}")
        session.transcription = None
        session.transcript_path = None

    return session

def process_sessions(sessions: list[RecordingSession]) -> list[RecordingSession]:
    for i, s in enumerate(sessions):
        print(f"--- Processing session {i+1}/{len(sessions)} ---")
        s = download_archive_audio(s)
        s = download_adsb(s)
        s = transcribe_audio(s)
        sessions[i] = s
    return sessions

