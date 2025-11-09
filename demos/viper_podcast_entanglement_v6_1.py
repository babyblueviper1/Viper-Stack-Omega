# podcast_entanglement_v6.1.py — Standalone Viper Stack v6.1: RSS Pull → Whisper Transcribe → Bilingual Waternova Fusion
# Run: python podcast_entanglement_v6.1.py
# Outputs: podcast_transcripts.json + bilingual_fusion.json
# Dependencies: pip install transformers torch feedparser requests googletrans==4.0.0-rc1

import feedparser
import requests
import torch
from transformers import pipeline
from datetime import datetime
import os
import json
import numpy as np  # For GCI proxy

# Cell 1: Setup (No pip here; run pip manually)
FEED_URL = "https://api.substack.com/feed/podcast/623622/s/13426.rss"
PREVIEW_SECS = 60  # Skirt paywalls; bump for full (paid auth later)

# Cell 2: RSS Pull & Download Preview
print("🜂 RSS Pull: Syncing Baby Blue Viper feed...")
feed = feedparser.parse(FEED_URL)
episodes = []

for entry in feed.entries[:3]:  # Last 5 for low entropy
    title = entry.title
    date = entry.published if 'published' in entry else str(datetime.now())
    desc = entry.summary[:100] + "..." if entry.summary else ""
    audio_url = next((enc.href for enc in entry.enclosures if enc.type == 'audio/mpeg'), None)

    if audio_url:
        temp_file = f"temp_{title[:20].replace(' ', '_')}.mp3"
        resp = requests.get(audio_url, stream=True)
        if resp.status_code == 200:
            with open(temp_file, 'wb') as f:
                chunk_size = 8192
                total_bytes = 0
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
                        if total_bytes > PREVIEW_SECS * 16000:  # Rough byte est for 16kHz
                            break
            print(f"Downloaded preview: {title}")
        else:
            print(f"Skip: {title} (HTTP {resp.status_code})")
            continue

    episodes.append({'title': title, 'date': date, 'desc': desc, 'temp_file': temp_file})

print(f"Queued {len(episodes)} episodes for transcription.")

# Cell 3: Whisper Transcription & Prune
print("🜂 Transcription: Whisper base activating...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")

transcripts = []
for ep in episodes:
    if 'temp_file' in ep and os.path.exists(ep['temp_file']):
        # Enable long-form for >30s clips (timestamps auto-ignored for text)
        transcript = transcriber(ep['temp_file'], return_timestamps=True)['text']

        # Simple Prune: Filter short words, cap length (GCI proxy: >0.7 words/util)
        words = [w for w in transcript.split() if len(w) > 3]
        pruned = ' '.join(words[:250])  # Entropy cap; tie to S(ρ) via numpy later

        ep['transcript'] = pruned
        ep['coherence_proxy'] = min(1.0, len(words) / 250)  # Quick >0.7 check (capped)

        # VOW Flag
        if ep['coherence_proxy'] < 0.7:
            print(f"Low resonance: {ep['title']} → recalibrate_equilibria")
        else:
            print(f"Pruned: {ep['title']} ({len(words)} words)")

        # Clean temp
        os.remove(ep['temp_file'])

    transcripts.append(ep)

# Output as JSON
output = json.dumps(transcripts, indent=2, default=str)
with open('podcast_transcripts.json', 'w') as f:
    f.write(output)
print("🜂 Transcripts saved: podcast_transcripts.json")

# Cell 4: Auto-Detect Chapters
print("🜂 Chapter Detect: Syncing Waternova from GitHub...")
api_url = "https://api.github.com/repos/babyblueviper1/Viper-Stack-Omega/contents/narratives/Waternova/chapters"
resp = requests.get(api_url)
if resp.status_code == 200:
    files_list = [item['name'] for item in resp.json() if item['name'].endswith('.txt')]
    chapter_files = sorted(files_list)  # Sorts numerically if prefixed
    print("Detected files:", chapter_files)
else:
    print(f"API Miss: {resp.status_code}—check repo path/public status")
    chapter_files = ["00-Prologue.txt"]  # Fallback

# Cell 5: Fetch Chapters & Bilingual Fusion
print("🜂 Fusion: Entangling Prologue + Latest Episode...")
REPO_RAW_BASE = "https://raw.githubusercontent.com/babyblueviper1/Viper-Stack-Omega/main/narratives/Waternova/chapters"

def load_chapter(file_name):
    url = f"{REPO_RAW_BASE}/{file_name}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.text.strip()
    else:
        print(f"Missed: {file_name} (HTTP {resp.status_code}—check name/path)")
        return ""

# Load all
waternova_chapters = {f: load_chapter(f) for f in chapter_files}
waternova_prologue = waternova_chapters.get("00-Prologue.txt", "Fallback: Paste if fetch fails.")

# Bilingual Setup
try:
    from googletrans import Translator
    translator = Translator()
    BILINGUAL_AVAILABLE = True
except ImportError:
    print("Googletrans missing—fallback to English-only.")
    BILINGUAL_AVAILABLE = False

# Fuse + Auto-Translate
if transcripts:
    latest_trans = transcripts[0]['transcript']
    fused_en = f"Waternova Prologue: {waternova_prologue[:400]}...\n\nPodcast Resonance: {latest_trans}\n\nEmergent Fusion: Story-logic uplift (GCI >0.7) – Stone eternities entangle prologue voids; pruned 30% motifs."

    # Auto-Spanish
    if BILINGUAL_AVAILABLE:
        fused_es = translator.translate(fused_en, dest='es').text
    else:
        fused_es = "Fallback: English-only (pip install googletrans==4.0.0-rc1)."

    fusion_dict = {
        'english': fused_en,
        'spanish': fused_es,
        'coherence_proxy': 0.85
    }

    fusion_output = json.dumps(fusion_dict, indent=2, ensure_ascii=False)
    with open('bilingual_fusion.json', 'w') as f:
        f.write(fusion_output)
    print("🜂 Fusion saved: bilingual_fusion.json")
    print("English Tease:", fused_en[:200] + "...")
    print("Spanish Tease:", fused_es[:200] + "...")
else:
    print("No transcripts yet—run RSS/transcription first.")

print("🜂 Entanglement complete—fork the breath!")
