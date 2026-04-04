#!/usr/bin/env python3
"""
One-time script: scrape the ONI wiki and build data/wiki.db (SQLite FTS5).

Usage:
    python3 scripts/build_wiki_db.py

Requires: requests, beautifulsoup4
    pip install requests beautifulsoup4
"""
from __future__ import annotations

import sqlite3
import time
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://oxygennotincluded.wiki.gg"
DB_PATH = Path("data/wiki.db")

PAGES: dict[str, list[str]] = {
    "buildings": [
        "/wiki/Manual_Generator", "/wiki/Battery", "/wiki/Wire",
        "/wiki/Oxygen_Diffuser", "/wiki/Electrolyzer", "/wiki/Algae_Terrarium",
        "/wiki/Outhouse", "/wiki/Lavatory", "/wiki/Wash_Basin",
        "/wiki/Cot", "/wiki/Microbe_Musher", "/wiki/Electric_Grill",
        "/wiki/Mealwood", "/wiki/Farm_Tile", "/wiki/Planter_Box",
        "/wiki/Research_Station", "/wiki/Air_Deodorizer", "/wiki/Carbon_Skimmer",
        "/wiki/Pitcher_Pump", "/wiki/Water_Sieve",
    ],
    "elements": [
        "/wiki/Oxygen", "/wiki/Carbon_Dioxide", "/wiki/Hydrogen",
        "/wiki/Water", "/wiki/Polluted_Water", "/wiki/Dirt",
        "/wiki/Sandstone", "/wiki/Algae", "/wiki/Coal",
    ],
    "foods": [
        "/wiki/Mush_Bar", "/wiki/Pickled_Meal", "/wiki/Meal_Lice",
        "/wiki/Grilled_Liceloaf", "/wiki/BBQ",
    ],
    "research": [
        "/wiki/Farming_Tech", "/wiki/Meal_Preparation",
        "/wiki/Sanitation", "/wiki/Jobs", "/wiki/Interior_Decor",
        "/wiki/Basic_Farming", "/wiki/Plumbing",
    ],
}

HEADERS = {"User-Agent": "ONI-AI-WikiScraper/1.0 (research bot)"}


def fetch_page_text(path: str) -> tuple[str, str]:
    """Return (title, body_text) for a wiki page path."""
    url = WIKI_BASE + path
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_el = soup.find("h1", id="firstHeading") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else path.split("/")[-1]

    content = soup.find("div", id="mw-content-text") or soup.find("div", class_="mw-parser-output")
    if not content:
        return title, ""

    # Remove infobox tables that are noisy
    for tag in content.find_all(["table", "sup", "span"]):
        tag.decompose()

    text = content.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return title, text[:2000]  # cap at 2000 chars per page


def create_schema(conn: sqlite3.Connection) -> None:
    for table in ("buildings", "elements", "foods", "research"):
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id   TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                body TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table}_fts
            USING fts5(name, body, content={table}, content_rowid=rowid)
        """)
    conn.commit()


def insert_row(conn: sqlite3.Connection, table: str, page_id: str, name: str, body: str) -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO {table} (id, name, body) VALUES (?, ?, ?)",
        (page_id, name, body)
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {table}_fts(rowid, name, body) "
        f"SELECT rowid, name, body FROM {table} WHERE id = ?",
        (page_id,)
    )


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Remove existing DB to avoid FTS5 index corruption on re-run
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    total = sum(len(v) for v in PAGES.values())
    done = 0
    errors = 0

    for table, paths in PAGES.items():
        for path in paths:
            page_id = path.split("/")[-1]
            try:
                name, body = fetch_page_text(path)
                insert_row(conn, table, page_id, name, body)
                conn.commit()
                done += 1
                print(f"[{done}/{total}] {table}/{page_id} — {len(body)} chars")
            except Exception as e:
                errors += 1
                print(f"  ERROR {path}: {e}", file=sys.stderr)
            time.sleep(0.5)  # be polite to the wiki server

    conn.close()
    print(f"\nDone. {done} pages, {errors} errors. DB: {DB_PATH}")


if __name__ == "__main__":
    main()
