import re
from datetime import datetime
from typing import List
import pandas as pd

from traffic.core import Traffic
from traffic.data import airports

# NATO alphabet and known prefixes
NATO = {
    'a': 'alfa', 'b': 'bravo', 'c': 'charlie', 'd': 'delta', 'e': 'echo',
    'f': 'foxtrot', 'g': 'golf', 'h': 'hotel', 'i': 'india', 'j': 'juliett',
    'k': 'kilo', 'l': 'lima', 'm': 'mike', 'n': 'november', 'o': 'oscar',
    'p': 'papa', 'q': 'quebec', 'r': 'romeo', 's': 'sierra', 't': 'tango',
    'u': 'uniform', 'v': 'victor', 'w': 'whiskey', 'x': 'xray', 'y': 'yankee', 'z': 'zulu'
}

CALLSIGN_PREFIXES = {
    "AAL": "american",
    "AAY": "allegiant",
    "ACA": "air canada",
    "AEE": "aegean",
    "AFR": "air france",
    "AMF": "ameriflight",
    "AMX": "aeromexico",
    "ANA": "all nippon",
    "APZ": "amazon prime",
    "ASA": "alaska",
    "ASL": "aeroflot",
    "AUA": "austrian",
    "BAW": "speedbird",
    "CAL": "china airlines",
    "CAT": "telephony",
    "CFG": "condor",
    "CTN": "croatia",
    "CSW": "silkitalia",
    "CXK": "cathay",
    "DAL": "delta",
    "DLH": "lufthansa",
    "DWW": "don juan",
    "EDW": "edelweiss",
    "EIN": "shamrock",
    "EJA": "executive jet",
    "EJM": "ejm",
    "ELY": "elal",
    "ENY": "envoy",
    "EVA": "eva",
    "FDX": "fedex",
    "FFT": "frontier flight",
    "FIN": "finnair",
    "GTI": "giant",
    "HAL": "hawaiian",
    "IBE": "iberia",
    "JBU": "jetblue",
    "JTL": "jetlinx",
    "KAL": "korean air",
    "LOT": "lot",
    "LXJ": "flexjet",
    "MGH": "mavi",
    "MOC": "monarch cargo",
    "MSC": "air cairo",
    "MXY": "mercy flight",
    "NCA": "nippon cargo",
    "NJE": "fraction",
    "NKS": "spirit wings",
    "OAW": "helvetic",
    "PCM": "pacific coast",
    "PGT": "sunturk",
    "QFA": "qantas",
    "QTR": "qatar",
    "QXE": "horizon",
    "RYR": "ryan air",
    "SAS": "scandinavian",
    "SDR": "swedestar",
    "SIA": "singapore",
    "SWA": "southwest",
    "SWR": "swiss",
    "TAP": "air portugal",
    "TAY": "quality",
    "THY": "turkish",
    "TKJ": "anatolian",
    "TVP": "jet travel",
    "TVS": "skytravel",
    "UAL": "united",
    "UAE": "emirates",
    "UPS": "ups",
    "VAR": "varig",
    "VIV": "vivaaerobus",
    "VJA": "volaris",
    "VOI": "volaris",
    "VXP": "avelo",
    "WJA": "westjet",
    "WZZ": "wizz air"
}

num2words = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
}

# ------------- Core Helpers ---------------

def callsign_to_words(callsign: str) -> str:
    callsign = callsign.strip().upper()
    
    # If the callsign is all letters, return full NATO spelling
    if callsign.isalpha():
        return " ".join(NATO[char.lower()] for char in callsign)
    
    # airline identifier
    for i in [3, 2]:
        if callsign[:i] in CALLSIGN_PREFIXES:
            prefix = CALLSIGN_PREFIXES[callsign[:i]]
            rest = callsign[i:]
            break
    else:
        prefix = callsign[:3].lower()
        rest = callsign[3:]
    
    # rest of callsign
    parts = []
    for char in rest:
        if char.isdigit():
            parts.append(num2words[int(char)])
        elif char.isalpha():
            parts.append(NATO[char.lower()])
    return prefix + " " + " ".join(parts)

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())

# ------------- Matching ---------------

def callsign_match_score(words_line: List[str], callsign_words: List[str]) -> int:
    best_score = 0
    for start in range(len(callsign_words)):
        trimmed = callsign_words[start:]
        i = j = score = 0
        while i < len(words_line) and j < len(trimmed):
            if words_line[i] == trimmed[j]:
                score += 1
                j += 1
            i += 1
        best_score = max(best_score, score)
    return best_score

def identify_callsigns_hard(transcript_lines: List[str], known_callsigns: List[str]):
    callsign_variants = {cs: callsign_to_words(cs) for cs in known_callsigns}
    results = []

    for line in transcript_lines:
        norm_line = normalize(line)
        candidates = [(cs, variant) for cs, variant in callsign_variants.items() if variant in norm_line]

        if candidates:
            max_len = max(len(v.split()) for _, v in candidates)
            matched = [cs for cs, v in candidates if len(v.split()) == max_len]
        else:
            matched = []

        results.append({"line": line, "callsigns": matched})

    return results

def identify_callsigns_soft(
    transcript_lines: List[str],
    known_callsigns: List[str],
    timestamp_range: pd.DataFrame,
    session_start: datetime,
    threshold: int = 3
):
    callsign_word_forms = {cs: callsign_to_words(cs).split() for cs in known_callsigns}
    results = []

    for line in transcript_lines:
        time_match = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
        if not time_match:
            continue

        t = datetime.strptime(time_match.group(1), "%H:%M:%S").time()
        timestamp = session_start.replace(hour=t.hour, minute=t.minute, second=t.second)

        norm_line = normalize(line)
        words_line = norm_line.split()[1:]  # skip timestamp

        scored_matches = []
        for cs, cs_words in callsign_word_forms.items():
            if cs not in timestamp_range.index:
                continue
            t_min = timestamp_range.loc[cs, "min"]
            t_max = timestamp_range.loc[cs, "max"]
            if not (t_min <= timestamp <= t_max):
                continue

            score = callsign_match_score(words_line, cs_words)
            if score >= threshold:
                scored_matches.append((cs, score))

        scored_matches.sort(key=lambda x: -x[1])
        results.append({
            "line": line,
            "callsigns": [scored_matches[0][0] if scored_matches else None],
            "scores": scored_matches
        })

    return results

# ------------- ADS-B timestamp loader ---------------

def build_timestamp_range(adsb_path: str) -> pd.DataFrame:
    adsb = Traffic.from_file(adsb_path)

    min_ts = adsb.data[["callsign", "timestamp"]].groupby("callsign").min()
    max_ts = adsb.data[["callsign", "timestamp"]].groupby("callsign").max()
    ts_range = min_ts.merge(max_ts, on="callsign")
    ts_range.columns = ["min", "max"]
    return ts_range