"""
Anonymize the raw Twitter/X exports before publishing them (GDPR).

- Tweet IDs (id, in_reply_to) are replaced by a salted SHA-256 hash, consistently
  across all files, so that joins (customer tweet <-> Free reply) still work.
- User-level columns (handle, name, user id, profile picture, URL, nested
  retweet/quote objects, media) are dropped.
- @mentions of private individuals are replaced by @user; brand, media and
  institutional accounts (Free, Freebox, Orange, Arcep...) are kept.
- E-mail addresses and phone numbers are masked.

Usage: python scripts/anonymize_data.py <input_dir> <output_dir>
"""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

SALT = "free-customer-care-llm"
PUBLIC_ACCOUNTS = {
    "free", "freebox", "free_1337", "freemobile", "universfreebox", "oqeebyfree",
    "groupeiliad", "xavier75", "bouyguestelecom", "orange_france", "orange",
    "orange_conseil", "sosh_fr", "sfr", "arcep", "60millions", "ufcquechoisir",
    "zoneadsl_panne", "canalplus", "infoabonnecanal", "eurosport_fr", "psg_inside",
    "fingapp", "outagedetect", "apple", "bfmtv", "canal", "cnil", "dgccrf", "elysee",
    "fnac", "lci", "ldlc", "mtvfr", "nperf", "tplink", "tpmp", "ups", "yahoo", "zimbra",
}
DROP_COLS = [
    "screen_name", "name", "profile_image_url", "user_id", "url", "media",
    "media_tags", "retweeted_status", "quoted_status",
]
ID_COLS = ["id", "in_reply_to"]
TEXT_COLS = ["full_text", "text_raw", "text_clean_llm", "text_clean"]
FILES = [
    "free_tweet_export.csv",
    "free_tweet_export.cleaned.llm.csv",
    "free_tweet_classified_clean.csv",
    "reponses_free.csv",
    "echantillion_5_tweets.csv",
    "resultat_classification_echantillion_5_tweets.csv",
]

MENTION = re.compile(r"@(\w+)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}")


def hash_id(x):
    if pd.isna(x) or str(x).strip() in ("", "nan", "None"):
        return x
    return hashlib.sha256((SALT + str(x).strip()).encode()).hexdigest()[:16]


PRIVATE_HANDLES = re.compile(r"$^")  # filled in main()


def clean_text(t):
    if not isinstance(t, str):
        return t
    t = EMAIL.sub("[email]", t)
    t = PHONE.sub("[phone]", t)
    t = MENTION.sub(
        lambda m: m.group(0) if m.group(1).lower() in PUBLIC_ACCOUNTS else "@user", t
    )
    # cleaned columns drop the "@": mask the bare handles as well
    return PRIVATE_HANDLES.sub("user", t)


def collect_private_handles(src: Path):
    handles = set()
    for name in FILES:
        f = src / name
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        for col in ("full_text", "screen_name"):
            if col in df.columns:
                for t in df[col].dropna():
                    found = MENTION.findall(t) if col == "full_text" else [t]
                    handles.update(h.lower() for h in found)
    handles -= PUBLIC_ACCOUNTS
    handles = {h for h in handles if len(h) >= 3}
    alt = "|".join(sorted(map(re.escape, handles), key=len, reverse=True))
    return re.compile(rf"(?<![\w@])(?:{alt})(?!\w)", re.IGNORECASE)


def main(src: Path, dst: Path):
    global PRIVATE_HANDLES
    PRIVATE_HANDLES = collect_private_handles(src)
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        f = src / name
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        for c in ID_COLS:
            if c in df.columns:
                df[c] = df[c].map(hash_id)
        for c in TEXT_COLS:
            if c in df.columns:
                df[c] = df[c].map(clean_text)
        df.to_csv(dst / name, index=False, encoding="utf-8")
        print(f"{name}: {len(df)} rows -> {dst / name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
