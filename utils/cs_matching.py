import re
from typing import Optional, Tuple, List
from datetime import datetime
import pandas as pd

from traffic.core import Traffic, Flight

from shapely.geometry import Point

from sentence_transformers import SentenceTransformer, util
model_embedding = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")


# ------------- NATO ALPHABET AND CALLSIGNS TELEPHONY ---------------
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
    "AIC": "air india",
    "AMF": "ameriflight",
    "AMX": "aeromexico",
    "ANA": "all nippon",
    "APZ": "amazon prime",
    "ASA": "alaska",
    "ASL": "aeroflot",
    "AUA": "austrian",
    "BAW": "speedbird",
    "BEL": "beeline",
    "BGA": "belouga",
    "CAL": "china airlines",
    "CAT": "aircat",
    "CCA": "air china",
    "CFE": "flyer",
    "CFG": "condor",
    "CHX": "christoph",
    "CMB": "camber",
    "CTN": "croatia",
    "CSA": "csa",
    "CSN": "china southern",
    "CSW": "silkitalia",
    "CXK": "cathay",
    "DAL": "delta",
    "DLH": "lufthansa",
    "DNA": "dan air",
    "DWW": "don juan",
    "EDW": "edelweiss",
    "EFW": "griffin",
    "EJA": "executive jet",
    "EJM": "ejm",
    "EIN": "shamrock",
    "ELY": "elal",
    "ENY": "envoy",
    "ETD": "etihad",
    "ETH": "ethiopian",
    "EVA": "eva",
    "EXS": "channex",
    "EZY": "easy",
    "FDX": "fedex",
    "FFT": "frontier flight",
    "FIA": "fia airlines",
    "FIN": "finnair",
    "GTI": "giant",
    "HAL": "hawaiian",
    "HYS": "sky europe",
    "IBE": "iberia",
    "ICE": "ice air",
    "JAF": "beauty",
    "JBU": "jetblue",
    "JTL": "jetlinx",
    "KAL": "korean air",
    "KLM": "klm",
    "LAV": "albastar",
    "LHX": "city air",
    "LOT": "lot",
    "LXJ": "flexjet",
    "MGH": "mavi",
    "MMD": "mermaid",
    "MOC": "monarch cargo",
    "MSC": "air cairo",
    "MSR": "egypt air",
    "MXY": "mercy flight",
    "NCA": "nippon cargo",
    "NJE": "fraction",
    "NKS": "spirit wings",
    "NOZ": "nordic",
    "NSZ": "rednose",
    "OAW": "helvetic",
    "PAV": "brilliant",
    "PCM": "pacific coast",
    "PGT": "sunturk",
    "QFA": "qantas",
    "QTR": "qatar",
    "QXE": "horizon",
    "RJA": "jordanian",
    "RUK": "blue max",
    "RYR": "ryan air",
    "SAS": "scandinavian",
    "SDR": "swedestar",
    "SIA": "singapore",
    "SVA": "saudia",
    "SWA": "southwest",
    "SWR": "swiss",
    "SXS": "sunexpress",
    "TAP": "air portugal",
    "TAY": "quality",
    "TFL": "orange",
    "THY": "turkish",
    "TKJ": "anatolian",
    "TOM": "tom jet",
    "TRA": "transavia",
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
    "WMT": "wizz air malta",
    "WUK": "wizz go",
    "WZZ": "wizz air"
}

num2words = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
}

# ------------- CORE HELPERS ---------------

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

def merge_callsign_entities(entities):
    """Merge two consecutive detected callsigns by the NER

    Args:
        entities (dict): NER output

    Returns:
        dict: NER output with merged consecutive callsigns
    """
    merged = []
    i = 0
    while i < len(entities):
        entity = entities[i]

        # If current entity is not a callsign, just append
        if entity["entity_group"] != "callsign":
            merged.append(entity)
            i += 1
            continue

        # Start merging if it's a callsign
        merged_word = entity["word"]
        start_pos = entity["start"]
        end_pos = entity["end"]
        score_sum = entity["score"]
        count = 1

        # Look ahead for consecutive callsigns
        while i + 1 < len(entities) and entities[i + 1]["entity_group"] == "callsign":
            next_entity = entities[i + 1]
            merged_word += " " + next_entity["word"]
            end_pos = next_entity["end"]
            score_sum += next_entity["score"]
            count += 1
            i += 1

        # Append merged callsign entity
        merged.append({
            "entity_group": "callsign",
            "word": merged_word,
            "start": start_pos,
            "end": end_pos,
            "score": score_sum / count  # average score
        })
        i += 1

    return merged

# ------------- ADSB HELPERS ---------------

def build_timestamp_range(adsb: Traffic) -> pd.DataFrame:
    min_ts = adsb.data[["callsign", "timestamp"]].groupby("callsign").min()
    max_ts = adsb.data[["callsign", "timestamp"]].groupby("callsign").max()
    ts_range = min_ts.merge(max_ts, on="callsign")
    ts_range.columns = ["min", "max"]
    return ts_range

# ------------- MATCHNING FUNCTIONS ---------------

def active_callsigns_at(
    adsb_ranges: pd.DataFrame,
    comm_time: pd.Timestamp,
    time_tolerance_s: int = 5*60,
) -> List[str]:
    """
    adsb_ranges must have columns: ['callsign', 'min', 'max'] (timestamps)
    Return callsigns whose active window contains comm_time (± tolerance). Aircraft contact radio several minutes before entering the airspace. 
    """
    # Ensure timestamps
    t0 = pd.to_datetime(comm_time) - pd.Timedelta(seconds=time_tolerance_s)
    t1 = pd.to_datetime(comm_time) + pd.Timedelta(seconds=time_tolerance_s)

    mask = (adsb_ranges["min"] <= t1) & (adsb_ranges["max"] >= t0)
    return mask[mask == True].index.tolist()

def flights_at_time(
    traffic: Traffic,
    callsigns: List[str],
    comm_time: pd.Timestamp,
) -> List[Flight]:
    """
    Return the Flight objects (might be several) that actually have
    data covering comm_time.
    Made to prevent callsigns that flew multiple legs. Only pick the segment that exists at comm_time. 
    """
    results: List[Flight] = []
    for cs in callsigns:
        slice_cs = traffic[[cs]] # might be multiple legs ([cs] to keep the traffic structure)
        if slice_cs is None:
            continue
        found = None
        for fl in slice_cs:
            if fl.at(comm_time) is not None: #Select the leg that has an observation at comm_time
                found = fl
                break
        if found is not None:
            results.append(found)
    return results

def in_sector_at_time(
    flight: Flight,
    comm_time: pd.Timestamp,
    sector_geom,
) -> bool:
    """
    Return True if the flight is in the sector at comm_time.
    """
    rec = flight.at(comm_time)
    if rec is None:
        return False
    lat = rec.get("latitude")
    lon = rec.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        return False
    return sector_geom.contains(Point(float(lon), float(lat)))

def closest_callsign_at_time(
    ner_callsigns: List[str],                # detected callsigns by ner
    traffic: Traffic,
    adsb_ranges: pd.DataFrame,
    comm_time: pd.Timestamp,
    match_threshold: int = 0.6,
    sector_geom=None,                 # optional Shapely polygon to keep only flights inside the sector at comm_time
    time_tolerance_s: int = 60,      
) -> Optional[Tuple[str, int]]:
    """
    Returns (best_callsign, similarity_score) between detected cs and the one present in adsb data.
    None if no candidate.
    Similarity is cosine simularity between embeddings.
    """

    # Selecting only the callsigns active at the communication time
    candidates_cs = active_callsigns_at(adsb_ranges, comm_time, time_tolerance_s)
    if not candidates_cs:
        return None
    
    active_flights = [fl for fl in traffic[candidates_cs]]

    # # 2) fetch the specific flight instances that actually exist at comm_time
    # # Get rid of flight segments that might end before the communication but are still in the tolerence_s
    # active_flights = flights_at_time(traffic, candidates_cs, comm_time)
    # if not active_flights:
    #     return None

    # # 3) Check if flight is in sector at comm_time (as the opensky bbox is larger)
    # if sector_geom is not None:
    #     active_flights = [fl for fl in active_flights if in_sector_at_time(fl, comm_time, sector_geom)]
    #     if not active_flights:
    #         return None
    
    # 4) Find thebest match
    best_match = None
    best_score = -1
    best_ner_callsign = None
    
    active_cs = {fl.callsign: callsign_to_words(fl.callsign) for fl in active_flights}
    context_cs = list(active_cs.values())
    context_cs_embeddings = model_embedding.encode(context_cs, convert_to_tensor=True)
    
    for identified_cs in ner_callsigns:
        identified_cs_word = identified_cs["word"]
        identified_cs_embedding = model_embedding.encode(identified_cs_word, convert_to_tensor=True)
        match_scores = util.cos_sim(identified_cs_embedding, context_cs_embeddings)[0]
        
        top_index = match_scores.argmax()
        top_score = float(match_scores[top_index])
        top_context_cs = list(active_cs.keys())[top_index]

        if top_score > best_score and top_score > match_threshold:
            best_score = top_score
            best_match = top_context_cs
            best_ner_callsign = identified_cs_word
            
    if best_match:
        return {
            "best_context_match": best_match,
            "ner_detected_callsign": best_ner_callsign,
            "match_score": round(best_score, 2)
        }
    else: 
        return None

# ------------- MATCHNING FUNCTIONS (DEPRECIATED) ---------------

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

