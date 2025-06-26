import requests
from bs4 import BeautifulSoup
import re

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