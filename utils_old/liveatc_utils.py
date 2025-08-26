import requests
from bs4 import BeautifulSoup
import re

from datetime import datetime
from pathlib import Path
import pandas as pd

def get_stations(icao):
  page = requests.get(f'https://www.liveatc.net/search/?icao={icao}')
  soup = BeautifulSoup(page.content, 'html.parser')

  stations = soup.find_all('table', class_='body', border='0', padding=lambda x: x != '0')
  freqs = soup.find_all('table', class_='freqTable', colspan='2')

  for table, freqs in zip(stations, freqs):
    title = table.find('strong').text
    up = table.find('font').text == 'UP'
    href = table.find('a', href=lambda x: x and x.startswith('/archive.php')).attrs['href']

    identifier = re.findall(r'/archive.php\?m=([a-zA-Z0-9_]+)', href)[0]

    frequencies = []
    rows = freqs.find_all('tr')[1:]
    for row in rows:
      cols = row.find_all('td')
      freq_title = cols[0].text
      freq_frequency = cols[1].text

      frequencies.append({'title': freq_title, 'frequency': freq_frequency})
      
    yield {'identifier': identifier, 'title': title, 'frequencies': frequencies, 'up': up}
  
def is_enroute_frequency(freq_entry: dict) -> bool:
  """
  Returns True if the frequency title indicates an ARTCC/Center sector.
  """
  title = freq_entry['title'].lower()
  return (
      "center" in title and "sector" in title
      and not any(kw in title for kw in ["approach", "departure", "tower", "ground", "clearance", "Twr", "App", "Gnd", "Del"])
  )
  
def extract_timestamp_from_path(transcript_path: str) -> datetime:
  filename = Path(transcript_path).stem  # e.g., "EPWA-epwa_app-Jun-27-2025-1230Z"

  # Extract the date and time string
  match = re.search(r"(\w+)-(\d{2})-(\d{4})-(\d{4})Z", filename)
  if not match:
      raise ValueError("Could not extract timestamp from path")

  month_str, day, year, time_str = match.groups()
  date_str = f"{day} {month_str} {year} {time_str}"

  # Parse the datetime (assumes Zulu time / UTC)
  return pd.Timestamp(datetime.strptime(date_str, "%d %b %Y %H%M"), tz="UTC")