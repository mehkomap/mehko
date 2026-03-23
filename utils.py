"""
mehko/utils.py
Utility functions for geocoding, web search, Claude classification,
social-media extraction, and map generation.
All expensive operations are cached by their callers in the notebook.
"""

import io, json, re, time
import requests
import pandas as pd
import numpy as np
from ddgs import DDGS


# ── Census geocoding ──────────────────────────────────────────────────────────

def geocode_census_batch(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Geocode a DataFrame of addresses using the US Census batch geocoder.
    Returns a DataFrame with columns: id, match, matched_addr, lat, lon.
    """
    url = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
    results = []

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        batch_csv = "\n".join(
            f'{i},"{r.address}","{r.city}","CA","{r.postal_code}"'
            for i, r in chunk.iterrows()
        )
        resp = requests.post(url, files={
            "addressFile": ("addresses.csv", batch_csv, "text/csv"),
            "benchmark":   (None, "Public_AR_Current"),
        })
        resp.raise_for_status()
        chunk_df = pd.read_csv(
            io.StringIO(resp.text), header=None,
            names=["id", "input_addr", "match", "match_type",
                   "matched_addr", "coords", "tiger_line_id", "side"],
            dtype=str,
        )
        results.append(chunk_df)
        print(f"  Census: rows {start}–{start + len(chunk) - 1}")
        time.sleep(0.5)

    out = pd.concat(results, ignore_index=True)
    out["id"] = out["id"].astype(int)
    out[["lat", "lon"]] = pd.DataFrame(
        out["coords"].apply(_parse_coords).tolist(), index=out.index
    )
    return out[["id", "match", "matched_addr", "lat", "lon"]]


def _parse_coords(coord_str: str):
    if pd.isna(coord_str) or not str(coord_str).strip():
        return None, None
    try:
        lon, lat = coord_str.strip().split(",")
        return float(lat), float(lon)
    except Exception:
        return None, None


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Return distance in metres between two WGS84 coordinate pairs."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── DuckDuckGo snippet search ─────────────────────────────────────────────────

def search_business(name: str, city: str, max_results: int = 2) -> str:
    """Return concatenated web snippets for a business, or '' if none found."""
    try:
        results = list(DDGS().text(f'"{name}" {city} food menu', max_results=max_results))
        if results:
            return " | ".join(r.get("body", "")[:150] for r in results if r.get("body"))
    except Exception:
        pass
    return ""


# ── Claude classification ─────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """You classify LA County MEHKO (home kitchen) businesses by food type.
For each business return a JSON object with exactly these fields:
- "business_type": one of [Bakery, BBQ, Beverage/Coffee/Tea, Breakfast/Brunch, Caribbean,
  Catering/Events, Charcuterie/Boards, Chinese, Comfort Food/Soul Food, Desserts/Sweets,
  Filipino, Indian/South Asian, Japanese/Korean, Latin American,
  Mediterranean/Middle Eastern, Mexican, Nigerian/West African, Pizza/Italian,
  Salvadoran/Guatemalan, Sandwiches/Burgers, Seafood, Vegan/Vegetarian, Vietnamese, Other]
- "cuisine_tags": list of 1-3 short lowercase keyword tags (e.g. ["mexican", "birria", "tacos"])
- "description": one sentence (max 15 words) describing what the business sells

Use the web snippet if relevant; otherwise infer from the business name alone."""


def classify_batch(batch_rows: pd.DataFrame, snippets: dict, client) -> list:
    """
    Send a batch of businesses to Claude and return a list of classification dicts.
    Each dict has keys: business_type, cuisine_tags, description.
    """
    items = [
        f"- Name: {row['business_name']}\n"
        f"  City: {row['city']}\n"
        f"  Web snippet: {snippets.get(row['business_name'], '') or '(none)'}"
        for _, row in batch_rows.iterrows()
    ]
    user_msg = (
        "Classify each business below. "
        "Return a JSON array in the same order, one object per business.\n\n"
        + "\n".join(items)
    )
    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=_CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    start, end = text.find("["), text.rfind("]") + 1
    return json.loads(text[start:end])


# ── Social media extraction ───────────────────────────────────────────────────

_SOCIAL_SYSTEM = """You extract social media and website info for small food businesses
from web search snippets.

Given a business name, city, and raw search result snippets, return a JSON object with:
- "instagram": the Instagram handle (e.g. "@handle"), or "" if not found or uncertain
- "website": the business's own website URL, or "" if not found or uncertain
- "confidence": "high", "medium", or "low"

Rules:
- Only return a handle if you are confident it belongs to THIS specific business
- Ignore aggregator sites (Yelp, DoorDash, Google, TripAdvisor, Grubhub, UberEats)
- If results seem to be for a different business, return empty strings with low confidence
- Return valid JSON only, no explanation"""


def search_and_extract(name: str, city: str, client) -> tuple[str, str, str]:
    """
    Search DuckDuckGo then use Claude to extract Instagram handle, website URL,
    and confidence level for a business. Returns (instagram, website, confidence).
    """
    snippets = []
    try:
        for query in [f"{name} instagram", f'"{name}" {city} website']:
            results = list(DDGS().text(query, max_results=3))
            for r in results:
                snippets.append(
                    f"URL: {r.get('href', '')}\n"
                    f"Title: {r.get('title', '')}\n"
                    f"Text: {r.get('body', '')[:200]}"
                )
            time.sleep(0.2)
    except Exception:
        pass

    if not snippets:
        return "", "", "low"

    user_msg = (
        f"Business: {name}\nCity: {city}\n\n"
        "Search results:\n" + "\n---\n".join(snippets)
    )
    try:
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            system=_SOCIAL_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        data = json.loads(text[start:end])
        return data.get("instagram", ""), data.get("website", ""), data.get("confidence", "low")
    except Exception:
        return "", "", "low"


# ── Map generation ────────────────────────────────────────────────────────────

def build_map(businesses: list, cuisine_types: list, output_path: str = "index.html"):
    """
    Write a self-contained Leaflet HTML map with sidebar + profile panel.
    `businesses` is a list of dicts with keys:
        name, type, tags, description, address, instagram, website, lat, lon
    """
    html = _MAP_TEMPLATE
    html = html.replace("__DATA_JSON__",  json.dumps(businesses))
    html = html.replace("__TYPES_JSON__", json.dumps(cuisine_types))
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved → {output_path}  ({len(businesses)} businesses)")


_MAP_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LA County MEHKO Businesses</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { display: flex; height: 100vh; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
#sidebar {
  width: 320px; min-width: 320px; height: 100vh; display: flex; flex-direction: column;
  background: #fff; border-right: 1px solid #e0e0e0;
  box-shadow: 2px 0 8px rgba(0,0,0,.08); z-index: 1000;
}
#sidebar-header { padding: 16px; border-bottom: 1px solid #eee; }
#sidebar-header h1 { font-size: 14px; font-weight: 700; color: #222; margin-bottom: 4px; }
#mehko-link { font-size: 11px; color: #4a90d9; text-decoration: none; display: block; margin-bottom: 10px; }
#mehko-link:hover { text-decoration: underline; }
#search {
  width: 100%; padding: 8px 12px; border: 1px solid #ddd;
  border-radius: 20px; font-size: 13px; outline: none;
}
#search:focus { border-color: #4a90d9; }
#search-row { display: flex; gap: 6px; align-items: center; }
#search-row #search { flex: 1; }
#reset-btn {
  display: none; padding: 7px 11px; border-radius: 20px; font-size: 11px;
  font-weight: 600; cursor: pointer; white-space: nowrap;
  background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5;
}
#reset-btn:hover { background: #fecaca; }
#reset-btn.visible { display: block; }
#filter-chips {
  padding: 10px 16px 4px; display: flex; flex-wrap: wrap; gap: 6px;
  border-bottom: 1px solid #eee; max-height: 130px; overflow-y: auto;
}
.chip {
  padding: 3px 10px; border-radius: 12px; font-size: 11px; cursor: pointer;
  border: 1px solid #ddd; background: #f5f5f5; color: #444;
  transition: all .15s; white-space: nowrap;
}
.chip:hover  { background: #e8e8e8; }
.chip.active { background: #222; color: #fff; border-color: #222; }
#biz-list { flex: 1; overflow-y: auto; padding: 8px 0; }
.biz-item {
  padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #f0f0f0;
  transition: background .1s;
}
.biz-item:hover    { background: #f7f7f7; }
.biz-item.selected { background: #eef4ff; }
.biz-name { font-size: 13px; font-weight: 600; color: #222; }
.biz-type { font-size: 11px; color: #888; margin-top: 1px; }
#biz-count { padding: 6px 16px; font-size: 11px; color: #999; border-bottom: 1px solid #eee; }
#sidebar-footer { padding: 10px 16px; font-size: 11px; color: #aaa; border-top: 1px solid #eee; }
#sidebar-footer a { color: #4a90d9; text-decoration: none; }
#sidebar-footer a:hover { text-decoration: underline; }
#disclaimer-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.5); z-index: 9000;
  align-items: center; justify-content: center;
}
#disclaimer-overlay.open { display: flex; }
#disclaimer-box {
  background: #fff; border-radius: 12px; padding: 28px 28px 22px;
  max-width: 420px; width: 90%; box-shadow: 0 8px 32px rgba(0,0,0,.2);
}
#disclaimer-box h2 { font-size: 15px; font-weight: 700; color: #111; margin-bottom: 12px; }
#disclaimer-box p  { font-size: 13px; color: #444; line-height: 1.6; margin-bottom: 10px; }
#disclaimer-box a  { color: #4a90d9; text-decoration: none; }
#disclaimer-box a:hover { text-decoration: underline; }
#disclaimer-close {
  display: block; width: 100%; margin-top: 18px; padding: 10px;
  background: #222; color: #fff; border: none; border-radius: 8px;
  font-size: 13px; font-weight: 600; cursor: pointer;
}
#disclaimer-close:hover { background: #444; }
#profile {
  display: none; position: absolute; bottom: 0; left: 320px; width: 280px;
  background: #fff; border: 1px solid #ddd; border-radius: 12px 12px 0 0;
  box-shadow: 0 -4px 20px rgba(0,0,0,.12); z-index: 2000;
  padding: 16px; max-height: 55vh; overflow-y: auto;
}
#profile.open { display: block; }
#profile-close {
  position: absolute; top: 10px; right: 12px;
  font-size: 18px; cursor: pointer; color: #aaa; line-height: 1;
}
#profile-close:hover { color: #333; }
#profile-name  { font-size: 15px; font-weight: 700; color: #111; padding-right: 20px; }
#profile-type  { font-size: 12px; color: #888; margin: 3px 0 8px; }
#profile-desc  { font-size: 12px; color: #444; line-height: 1.5; margin-bottom: 10px; }
#profile-addr  { font-size: 11px; color: #777; margin-bottom: 10px; }
#profile-links { display: flex; gap: 8px; flex-wrap: wrap; }
.profile-link {
  display: inline-block; padding: 5px 12px; border-radius: 16px;
  font-size: 11px; font-weight: 600; text-decoration: none;
  border: 1px solid #ddd; color: #333;
}
.profile-link:hover { background: #f0f0f0; }
.profile-link.ig  { background: #f9f3ff; border-color: #c084fc; color: #7e22ce; }
.profile-link.web { background: #eff6ff; border-color: #60a5fa; color: #1d4ed8; }
#map { flex: 1; height: 100vh; }

/* ── Cluster count colors: light → dark green ── */
.marker-cluster-small div        { background-color: #a5d6a7; }
.marker-cluster-small            { background-color: #c8e6c9; }
.marker-cluster-medium div       { background-color: #4caf50; }
.marker-cluster-medium           { background-color: #a5d6a7; }
.marker-cluster-large div        { background-color: #1b5e20; }
.marker-cluster-large            { background-color: #388e3c; }
.marker-cluster-small div span,
.marker-cluster-medium div span,
.marker-cluster-large div span   { color: #fff; font-weight: 600; }

/* ── Mobile ──────────────────────────────────────────────── */
#list-toggle {
  display: none;
  position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
  z-index: 3000; padding: 10px 22px; border-radius: 24px;
  background: #222; color: #fff; font-size: 13px; font-weight: 600;
  border: none; cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,.3);
}

@media (max-width: 600px) {
  body { display: block; }

  #sidebar {
    position: fixed; bottom: 0; left: 0; right: 0;
    width: 100%; min-width: unset; height: 65vh;
    border-right: none; border-radius: 16px 16px 0 0;
    border-top: 1px solid #ddd;
    transform: translateY(100%);
    transition: transform .3s ease;
    z-index: 2000;
  }
  #sidebar.open { transform: translateY(0); }

  #map { width: 100vw; height: 100vh; }

  #profile { left: 0; width: 100%; border-radius: 12px 12px 0 0; }

  #list-toggle { display: block; }
}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <h1>LA County MEHKO Businesses</h1>
    <a id="mehko-link" href="http://www.publichealth.lacounty.gov/EH/business/microenterprise-home-kitchen-operation.htm" target="_blank">What is a MEHKO?</a>
    <div id="search-row">
      <input id="search" type="text" placeholder="Search by name or cuisine…" autocomplete="off">
      <button id="reset-btn" onclick="resetAll()">&#x2715; Reset</button>
    </div>
  </div>
  <div id="filter-chips"></div>
  <div id="biz-count"></div>
  <div id="biz-list"></div>
  <div id="sidebar-footer">
    Error, question, or comment? <a href="mailto:swietek@usc.edu">swietek@usc.edu</a>
  </div>
</div>
<div id="map"></div>
<button id="list-toggle" onclick="toggleSidebar()">☰ List</button>
<div id="disclaimer-overlay">
  <div id="disclaimer-box">
    <h2>About This Map</h2>
    <p>This directory is based on active MEHKO (Micro-Enterprise Home Kitchen Operation) licenses issued by LA County. Not all listed businesses may currently be in operation, and some details — including addresses, contact info, and cuisine type — may be incomplete or out of date.</p>
    <p>Questions or corrections? Contact <a href="mailto:swietek@usc.edu">swietek@usc.edu</a>.</p>
    <button id="disclaimer-close" onclick="closeDisclaimer()">Got it</button>
  </div>
</div>
<div id="profile">
  <span id="profile-close">&#x2715;</span>
  <div id="profile-name"></div>
  <div id="profile-type"></div>
  <div id="profile-desc"></div>
  <div id="profile-addr"></div>
  <div id="profile-links"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script>
const BUSINESSES = __DATA_JSON__;
const TYPES      = __TYPES_JSON__;

const map = L.map('map').setView([34.05, -118.25], 10);
L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png', {
  minZoom: 0, maxZoom: 20,
  attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

// --- Icon mapping ---
const TYPE_ICON = {
  'Bakery':                       'assets/bakery.png',
  'BBQ':                          'assets/bbq.png',
  'Beverage/Coffee/Tea':          'assets/cafe.png',
  'Breakfast/Brunch':             'assets/comfort_food.png',
  'Caribbean':                    'assets/bbq.png',
  'Catering/Events':              'assets/comfort_food.png',
  'Charcuterie/Boards':           'assets/comfort_food.png',
  'Chinese':                      'assets/chinese_asian.png',
  'Comfort Food/Soul Food':       'assets/comfort_food.png',
  'Desserts/Sweets':              'assets/desserts.png',
  'Filipino':                     'assets/filipino.png',
  'Indian/South Asian':           'assets/indian.png',
  'Japanese/Korean':              'assets/chinese_asian.png',
  'Latin American':               'assets/latin.png',
  'Mediterranean/Middle Eastern': 'assets/indian.png',
  'Mexican':                      'assets/mexican.png',
  'Nigerian/West African':        'assets/latin.png',
  'Pizza/Italian':                'assets/pizza.png',
  'Salvadoran/Guatemalan':        'assets/mexican.png',
  'Sandwiches/Burgers':           'assets/burgers.png',
  'Seafood':                      'assets/filipino.png',
  'Vegan/Vegetarian':             'assets/comfort_food.png',
  'Vietnamese':                   'assets/chinese_asian.png',
  'Other':                        'assets/comfort_food.png',
};

// --- State ---
const activeFilters = { search: '', type: 'ALL' };
let selectedBiz = null;

// --- Map ---
const cluster = L.markerClusterGroup({ maxClusterRadius: 40, disableClusteringAtZoom: 14 });
map.addLayer(cluster);

function dotIcon(type) {
  const iconUrl = TYPE_ICON[type] || 'assets/comfort_food.png';
  return L.divIcon({
    className: '',
    html: `<img src="${iconUrl}" style="width:52px;height:52px;object-fit:contain;filter:drop-shadow(0 0 5px white) drop-shadow(0 0 5px white) drop-shadow(0 2px 6px rgba(0,0,0,.35))">`,
    iconSize:    [52, 52],
    iconAnchor:  [26, 52],
    popupAnchor: [0, -52],
  });
}

// --- Chips ---
function renderChips() {
  const container = document.getElementById('filter-chips');
  const types = ['ALL', ...TYPES];
  container.innerHTML = types.map(t => `
    <span class="chip ${t === activeFilters.type ? 'active' : ''}"
          onclick="filterByType('${t}')">${t === 'ALL' ? 'All' : t}</span>
  `).join('');
}

// --- Core update ---
function updateDisplay() {
  const q = activeFilters.search.toLowerCase();
  const filtered = BUSINESSES.filter(b => {
    const typeOk = activeFilters.type === 'ALL' || b.type === activeFilters.type;
    const textOk = !q || b.name.toLowerCase().includes(q) ||
                   b.tags.toLowerCase().includes(q) || b.type.toLowerCase().includes(q);
    return typeOk && textOk;
  });

  // List
  const list = document.getElementById('biz-list');
  list.innerHTML = filtered.map(b => `
    <div class="biz-item ${selectedBiz === b.name ? 'selected' : ''}"
         onclick="focusBusiness('${b.name.replace(/'/g, "\\'")}')">
      <div class="biz-name">${b.name}</div>
      <div class="biz-type">${b.type}${b.tags ? ' · ' + b.tags : ''}</div>
    </div>
  `).join('');

  // Markers
  cluster.clearLayers();
  filtered.forEach(b => {
    const marker = L.marker([b.lat, b.lon], { icon: dotIcon(b.type), title: b.name });
    marker.on('click', () => showProfile(b));
    cluster.addLayer(marker);
  });

  // Count + reset button
  const n = filtered.length;
  document.getElementById('biz-count').textContent = `${n} business${n !== 1 ? 'es' : ''}`;
  document.getElementById('reset-btn').classList.toggle(
    'visible', !!activeFilters.search || activeFilters.type !== 'ALL'
  );
}

// --- Interactions ---
function filterByType(type) {
  activeFilters.type = type;
  renderChips();
  updateDisplay();
}

function focusBusiness(name) {
  const biz = BUSINESSES.find(b => b.name === name);
  if (!biz) return;
  map.flyTo([biz.lat, biz.lon], 15, { animate: true, duration: 0.8 });
  // Close sidebar drawer on mobile so the map is visible
  if (window.innerWidth <= 600) {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('list-toggle').textContent = '☰ List';
  }
  showProfile(biz);
}

function showProfile(biz) {
  selectedBiz = biz.name;
  document.getElementById('profile-name').textContent = biz.name;
  document.getElementById('profile-type').textContent = biz.type;
  document.getElementById('profile-desc').textContent = biz.description || '';
  document.getElementById('profile-addr').textContent = biz.address;
  const links = document.getElementById('profile-links');
  links.innerHTML = '';
  if (biz.instagram) {
    links.innerHTML += `<a href="https://instagram.com/${biz.instagram.replace('@','')}" target="_blank" class="profile-link ig">${biz.instagram}</a>`;
  }
  if (biz.website) {
    links.innerHTML += `<a href="${biz.website}" target="_blank" class="profile-link web">Website</a>`;
  }
  document.getElementById('profile').classList.add('open');
  updateDisplay();
}

function closeDisclaimer() {
  document.getElementById('disclaimer-overlay').classList.remove('open');
  localStorage.setItem('mehko-disclaimer-seen', '1');
}

function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const btn = document.getElementById('list-toggle');
  const open = sb.classList.toggle('open');
  btn.textContent = open ? '✕ Close' : '☰ List';
}

function resetAll() {
  activeFilters.search = '';
  activeFilters.type = 'ALL';
  document.getElementById('search').value = '';
  map.flyTo([34.0522, -118.2437], 10, { animate: true, duration: 0.8 });
  renderChips();
  updateDisplay();
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  renderChips();
  updateDisplay();

  if (!localStorage.getItem('mehko-disclaimer-seen')) {
    document.getElementById('disclaimer-overlay').classList.add('open');
  }

  document.getElementById('search').addEventListener('input', e => {
    activeFilters.search = e.target.value;
    updateDisplay();
  });

  document.getElementById('profile-close').onclick = () => {
    document.getElementById('profile').classList.remove('open');
    selectedBiz = null;
    updateDisplay();
  };
});
</script>
</body>
</html>"""
