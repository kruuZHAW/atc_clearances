import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

import requests
from bs4 import BeautifulSoup

from traffic.data import opensky

import wave
import webrtcvad
from pydub import AudioSegment
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Core data model to represent each audio + ads-b + transcript data
@dataclass
class RecordingSession:
    airport: str
    channel: str
    start_time: datetime
    end_time: datetime
    audio_path: Path
    transcript_path: Optional[Path] = None
    transcription: Optional[str] = None
    adsb_data: Optional[pd.DataFrame] = None
    adsb_path: Optional[Path] = None
    
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
    url = f'https://archive.liveatc.net/{session.channel.split("_")[0]}/{filename}'
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

def download_adsb(session: RecordingSession, save_path="./adsb", use_cache=True) -> RecordingSession:
    """
    Downloads ADS-B data using OpenSky and saves as Parquet.
    Uses cache if file already exists.
    Note: Only handles arrival aircraft
    """
    date_str = session.start_time.strftime("%b-%d-%Y")
    time_str = session.start_time.strftime("%H%M") + "Z"
    filename = f"{session.airport}-{session.channel}-{date_str}-{time_str}.parquet"

    os.makedirs(save_path, exist_ok=True)
    adsb_file = Path(save_path) / filename

    if use_cache and adsb_file.exists():
        session.adsb_data = pd.read_parquet(adsb_file)
        session.adsb_path = adsb_file
        print(f"[adsb] Loaded from cache: {adsb_file}")
        return session

    try:
        adsb_df = opensky.history(
            start=session.start_time,
            stop=session.end_time,
            arrival_airport=session.airport
        )

        if adsb_df is None or len(adsb_df) == 0:
            print(f"[adsb] No data found for {session.airport} from {session.start_time} to {session.end_time}")
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

def split_audio_by_vad(session: RecordingSession, chunk_dir="./chunks", use_cache=True) -> list[Path]:
    """
    Splits session.audio_path into voice activity chunks using WebRTC VAD.
    Returns list of saved .wav paths.
    """
    if session.audio_path is None or not session.audio_path.exists():
        print("[vad] No audio found, skipping VAD splitting.")
        return []

    # Format consistent with other outputs
    date_str = session.start_time.strftime("%b-%d-%Y")
    time_str = session.start_time.strftime("%H%M") + "Z"
    base_name = f"{session.airport}-{session.channel}-{date_str}-{time_str}"
    
    chunk_output_dir = Path(chunk_dir) / base_name
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Use cache: skip if already split
    if use_cache and any(chunk_output_dir.glob("speech_*.wav")):
        print(f"[vad] Using cached audio chunks in {chunk_output_dir}")
        return sorted(chunk_output_dir.glob("speech_*.wav"))

    # Step 1: Convert MP3 to WAV (16kHz mono)
    temp_wav = chunk_output_dir / "temp_audio.wav"
    audio = AudioSegment.from_file(session.audio_path).set_channels(1).set_frame_rate(16000)
    audio.export(temp_wav, format="wav")

    # Step 2: Run VAD to chunk the audio
    voiced_chunks = []
    with wave.open(str(temp_wav), 'rb') as wf:
        vad = webrtcvad.Vad(2)
        frame_duration = 30  # ms
        sample_rate = wf.getframerate()
        frame_bytes = int(sample_rate * frame_duration / 1000) * 2  # 16-bit
        frames = wf.readframes(wf.getnframes())

        i = 0
        voiced = False
        chunk = b''
        min_samples = 16000  # 1s

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
                        fname = chunk_output_dir / f"speech_{i:03d}.wav"
                        segment = AudioSegment(
                            data=chunk,
                            sample_width=2,
                            frame_rate=16000,
                            channels=1
                        )
                        segment.export(fname, format="wav")
                        voiced_chunks.append(fname)
                        print(f"[vad] Saved chunk {i+1}: {fname}")
                        i += 1

    # Clean up temp
    if temp_wav.exists():
        temp_wav.unlink()

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
    filename = f"{session.airport}-{session.channel}-{date_str}-{time_str}.txt"

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
        for i, path in enumerate(sorted(chunk_paths)):
            audio_input, _ = torchaudio.load(path)
            inputs = processor(audio_input.squeeze(), sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features
            # Specify padding tokens
            attention_mask = inputs.get("attention_mask", None)
            generated_ids = model.generate(input_features, attention_mask=attention_mask)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append(transcription)
            print(f"[transcribe] Chunk {i+1}/{len(chunk_paths)}: {transcription}")

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
        s = download_audio(s)
        s = download_adsb(s)
        s = transcribe_audio(s)
        sessions[i] = s
    return sessions

