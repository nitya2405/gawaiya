"""
backend/raga_meta.py

Hardcoded metadata for all 63 ragas in vocabulary.
Keys must exactly match the strings in runs/hindustani_small/vocabs/raga.json.
"""

from __future__ import annotations

RAGA_META: dict[str, dict] = {
    "Āhīr Bhairav": {"thaat": "Bhairav",   "time": "Morning",   "mood": "Serene"},
    "Āsāvarī":      {"thaat": "Asavari",   "time": "Morning",   "mood": "Serious"},
    "Bageshri":     {"thaat": "Kafi",      "time": "Night",     "mood": "Romantic"},
    "Bageshrī":     {"thaat": "Kafi",      "time": "Night",     "mood": "Romantic"},
    "Bahār":        {"thaat": "Kafi",      "time": "Any",       "mood": "Joyful"},
    "Bairāgī":      {"thaat": "Bhairav",   "time": "Morning",   "mood": "Devotional"},
    "Bairāgī Bhairav": {"thaat": "Bhairav","time": "Morning",   "mood": "Devotional"},
    "Bhairav":      {"thaat": "Bhairav",   "time": "Morning",   "mood": "Devotional"},
    "Bhairavī":     {"thaat": "Bhairavi",  "time": "Morning",   "mood": "Melancholic"},
    "Bhimpalāsī":   {"thaat": "Kafi",      "time": "Afternoon", "mood": "Serene"},
    "Bhupālī":      {"thaat": "Kalyan",    "time": "Evening",   "mood": "Joyful"},
    "Bilāskhānī Toḍī": {"thaat": "Todi",  "time": "Morning",   "mood": "Serious"},
    "Bilāval":      {"thaat": "Bilaval",   "time": "Morning",   "mood": "Peaceful"},
    "Bihāg":        {"thaat": "Bilaval",   "time": "Night",     "mood": "Romantic"},
    "Chandrakaus":  {"thaat": "Bhairav",   "time": "Night",     "mood": "Serene"},
    "Chandrakauns": {"thaat": "Bhairav",   "time": "Night",     "mood": "Serene"},
    "Charukeshī":   {"thaat": "Bilaval",   "time": "Afternoon", "mood": "Melancholic"},
    "Darbārī Kānaḍā": {"thaat": "Asavari","time": "Night",     "mood": "Serious"},
    "Des":          {"thaat": "Khamaj",    "time": "Evening",   "mood": "Romantic"},
    "Desh":         {"thaat": "Khamaj",    "time": "Evening",   "mood": "Romantic"},
    "Deśkār":       {"thaat": "Kalyan",    "time": "Morning",   "mood": "Joyful"},
    "Durgā":        {"thaat": "Bilaval",   "time": "Evening",   "mood": "Devotional"},
    "Gauḍ Malhār":  {"thaat": "Kafi",      "time": "Monsoon",   "mood": "Joyful"},
    "Gauḍ Sāraṅg":  {"thaat": "Bilaval",   "time": "Noon",      "mood": "Romantic"},
    "Gujrī Toḍī":   {"thaat": "Todi",      "time": "Morning",   "mood": "Melancholic"},
    "Hamīr":        {"thaat": "Kalyan",    "time": "Evening",   "mood": "Serious"},
    "Haṃsadhvani":  {"thaat": "Bilaval",   "time": "Evening",   "mood": "Joyful"},
    "Hindol":       {"thaat": "Kalyan",    "time": "Morning",   "mood": "Joyful"},
    "Jaunpurī":     {"thaat": "Asavari",   "time": "Morning",   "mood": "Melancholic"},
    "Jayjayvantī":  {"thaat": "Khamaj",    "time": "Night",     "mood": "Romantic"},
    "Jhinjhoṭī":    {"thaat": "Khamaj",    "time": "Night",     "mood": "Romantic"},
    "Jogiyā":       {"thaat": "Bhairav",   "time": "Morning",   "mood": "Devotional"},
    "Kāfī":         {"thaat": "Kafi",      "time": "Night",     "mood": "Romantic"},
    "Kalyāṇ":       {"thaat": "Kalyan",    "time": "Evening",   "mood": "Serene"},
    "Kāmbojī":      {"thaat": "Khamaj",    "time": "Afternoon", "mood": "Romantic"},
    "Kedar":        {"thaat": "Kalyan",    "time": "Evening",   "mood": "Devotional"},
    "Kedār":        {"thaat": "Kalyan",    "time": "Evening",   "mood": "Devotional"},
    "Khambāvatī":   {"thaat": "Khamaj",    "time": "Evening",   "mood": "Romantic"},
    "Khamāj":       {"thaat": "Khamaj",    "time": "Evening",   "mood": "Romantic"},
    "Kīrwānī":      {"thaat": "Asavari",   "time": "Night",     "mood": "Melancholic"},
    "Lālanī":       {"thaat": "Asavari",   "time": "Evening",   "mood": "Melancholic"},
    "Lalit":        {"thaat": "Marwa",     "time": "Dawn",      "mood": "Serious"},
    "Lālat":        {"thaat": "Marwa",     "time": "Dawn",      "mood": "Serious"},
    "Mādrī":        {"thaat": "Bhairav",   "time": "Morning",   "mood": "Devotional"},
    "Māhūr":        {"thaat": "Bhairav",   "time": "Morning",   "mood": "Devotional"},
    "Mālkauns":     {"thaat": "Bhairavi",  "time": "Night",     "mood": "Serious"},
    "Mānd":         {"thaat": "Bilaval",   "time": "Any",       "mood": "Joyful"},
    "Mārwā":        {"thaat": "Marwa",     "time": "Evening",   "mood": "Serious"},
    "Mianki Malhār": {"thaat": "Kafi",     "time": "Monsoon",   "mood": "Romantic"},
    "Miyāṃ Malhār": {"thaat": "Kafi",      "time": "Monsoon",   "mood": "Romantic"},
    "Multānī":      {"thaat": "Todi",      "time": "Afternoon", "mood": "Melancholic"},
    "Nand":         {"thaat": "Kalyan",    "time": "Night",     "mood": "Serene"},
    "Nandakalyāṇ":  {"thaat": "Kalyan",    "time": "Night",     "mood": "Serene"},
    "Pāhāḍī":       {"thaat": "Bilaval",   "time": "Any",       "mood": "Joyful"},
    "Pūriyā":       {"thaat": "Marwa",     "time": "Evening",   "mood": "Serious"},
    "Pūriyā Dhanāśrī": {"thaat": "Marwa", "time": "Evening",   "mood": "Serious"},
    "Pūriyā Kalyāṇ": {"thaat": "Marwa",   "time": "Evening",   "mood": "Romantic"},
    "Rāgeshvarī":   {"thaat": "Kafi",      "time": "Night",     "mood": "Romantic"},
    "Sāraṅg":       {"thaat": "Kafi",      "time": "Noon",      "mood": "Serene"},
    "Śivarañjanī":  {"thaat": "Kafi",      "time": "Any",       "mood": "Devotional"},
    "Śrī":          {"thaat": "Purvi",     "time": "Evening",   "mood": "Serious"},
    "Toḍī":         {"thaat": "Todi",      "time": "Morning",   "mood": "Melancholic"},
    "Yaman":        {"thaat": "Kalyan",    "time": "Evening",   "mood": "Serene"},
    "Yaman Kalyāṇ": {"thaat": "Kalyan",    "time": "Evening",   "mood": "Serene"},
}

TALA_META: dict[str, dict] = {
    # beats: total beat count
    # sam: 1-based beat number of sam (always 1 except Rūpak)
    # khali: list of 1-based beat numbers that are khali (empty/open hand)
    # vibhag: list of vibhag sizes (must sum to beats)
    # character: tempo feel
    "Ādi Tāl":  {"beats": 8,  "sam": 1, "khali": [5],     "vibhag": [2,2,2,2], "character": "Medium"},
    "Dādrā":    {"beats": 6,  "sam": 1, "khali": [4],     "vibhag": [3,3],     "character": "Medium"},
    "Dhamar":   {"beats": 14, "sam": 1, "khali": [9],     "vibhag": [5,2,3,4], "character": "Slow"},
    "Ēktāl":    {"beats": 12, "sam": 1, "khali": [7,9],   "vibhag": [2,2,2,2,2,2], "character": "Slow"},
    "Jhaptāl":  {"beats": 10, "sam": 1, "khali": [6],     "vibhag": [2,3,2,3], "character": "Medium"},
    "Jhūmrā":   {"beats": 14, "sam": 1, "khali": [9],     "vibhag": [3,4,3,4], "character": "Slow"},
    "Keharwā":  {"beats": 8,  "sam": 1, "khali": [5],     "vibhag": [4,4],     "character": "Fast"},
    "Rūpak":    {"beats": 7,  "sam": 4, "khali": [1],     "vibhag": [3,2,2],   "character": "Medium"},
    "Tīntāl":   {"beats": 16, "sam": 1, "khali": [9],     "vibhag": [4,4,4,4], "character": "Medium"},
    "Tilwāḍā":  {"beats": 16, "sam": 1, "khali": [9],     "vibhag": [4,4,4,4], "character": "Slow"},
}


def get_raga_list() -> list[dict]:
    return [
        {"name": name, **meta}
        for name, meta in sorted(RAGA_META.items())
    ]


def get_tala_list() -> list[dict]:
    return [
        {"name": name, **meta}
        for name, meta in sorted(TALA_META.items())
    ]


def raga_names() -> set[str]:
    return set(RAGA_META.keys())


def tala_names() -> set[str]:
    return set(TALA_META.keys())
