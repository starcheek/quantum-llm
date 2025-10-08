#!/usr/bin/env python3
"""
arXiv Data Fetcher

A Python script to fetch research papers from arXiv API with tiered category approach.
Based on the new Colab script with sophisticated quota planning.
"""

import time
import re
import math
import feedparser
import pandas as pd
from urllib.parse import urlencode, quote_plus
from pathlib import Path
from collections import defaultdict
import json

# API Configuration
ARXIV_API = "https://export.arxiv.org/api/query"
PER_REQ = 200        # results per API call (<=2000 max_results, but small is safer)
SLEEP_SEC = 3.0      # be polite to arXiv
TOTAL_TARGET = 5000  # final target

def load_categories(config_path: str = "config.json"):
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cats = cfg.get("categories", {})
        tier1 = cats.get("tier1", [])
        tier2 = cats.get("tier2", [])
        tier3 = cats.get("tier3", [])
        tier4 = cats.get("tier4", [])
        return tier1, tier2, tier3, tier4
    except Exception:
        # fallback to defaults if config missing
        tier1 = ["quant-ph","math-ph","cs.ET","cond-mat.mes-hall","cond-mat.supr-con","physics.optics","physics.comp-ph"]
        tier2 = ["cs.AR","cs.CE","cs.SE","cs.LG","cs.AI","stat.ML","stat.ME","stat.CO","stat.TH","stat.AP"]
        tier3 = ["physics.chem-ph","physics.data-an","physics.gen-ph","nucl-th","math.NT","cs.CC","physics.ed-ph"]
        tier4 = ["q-fin.CP","q-fin.GN","q-fin.PM","q-fin.PR","q-fin.RM","q-fin.ST","q-fin.TR","cs.CR","cs.CY"]
        return tier1, tier2, tier3, tier4

TIER1, TIER2, TIER3, TIER4 = load_categories()
ALL_CATS = list(TIER1) + list(TIER2) + list(TIER3) + list(TIER4)

# --- Tier weights: edit these to change representation ---
TIER_WEIGHTS = {1:0.60, 2:0.25, 3:0.10, 4:0.05}  # sums to 1.0

# Per-category minimum to guarantee presence from every category
MIN_PER_CAT = 80   # adjust if needed


def parse_entry(e):
    m = re.search(r'/abs/([^v]+)(v\d+)?$', e.id)
    arxiv_id = m.group(1) if m else e.id
    abstract = getattr(e, "summary", "").strip().replace("\n"," ")
    cats = [t["term"] for t in getattr(e, "tags", [])]
    return {
        "arxiv_id": arxiv_id,
        "title": e.title.strip().replace("\n"," "),
        "summary": abstract,            # (optional alias)
        "authors": [a.name for a in getattr(e, "authors", [])],
        "published": e.published,
        "updated": getattr(e, "updated", e.published),
        "primary_cat": cats[0] if cats else None,
        "all_cats": cats,
        "link_abs": e.link,
        "link_pdf": next((l["href"] for l in e.links if l.get("type")=="application/pdf"), None),
        "journal_ref": getattr(e, "arxiv_journal_ref", None),
        "doi": getattr(e, "arxiv_doi", None),
        "comment": getattr(e, "arxiv_comment", None),
    }


def fetch_cat(cat, goal_n, per_req=PER_REQ, sortBy="submittedDate", sortOrder="descending"):
    rows, got = [], 0
    while got < goal_n:
        n = min(per_req, goal_n - got)
        params = {
            "search_query": f"cat:{cat}",
            "start": got,
            "max_results": n,
            "sortBy": sortBy,
            "sortOrder": sortOrder,
        }
        url = f"{ARXIV_API}?{urlencode(params, quote_via=quote_plus)}"
        feed = feedparser.parse(url)
        if not feed.entries: break
        rows.extend(parse_entry(e) for e in feed.entries)
        got += len(feed.entries)
        time.sleep(SLEEP_SEC)
        if len(feed.entries) < n:  # no more pages
            break
    return rows


def plan_quotas(total=TOTAL_TARGET, min_per_cat=MIN_PER_CAT):
    tiers = {1:TIER1, 2:TIER2, 3:TIER3, 4:TIER4}
    # Guarantee a floor for every category first
    base_need = min_per_cat * len(ALL_CATS)
    if base_need > total:
        raise ValueError(f"MIN_PER_CAT={min_per_cat} too high for TOTAL_TARGET={total}")
    quotas = {c: min_per_cat for c in ALL_CATS}
    remaining = total - base_need

    # Distribute remaining by tier weights, then evenly within tier
    for t, cats in tiers.items():
        tier_share = int(round(remaining * TIER_WEIGHTS[t]))
        if tier_share <= 0: continue
        per_cat_add = tier_share // len(cats)
        leftover = tier_share - per_cat_add*len(cats)
        for c in cats:
            quotas[c] += per_cat_add
        # round-robin the leftover
        for i in range(leftover):
            quotas[cats[i % len(cats)]] += 1

    # Small rounding drift fix
    drift = total - sum(quotas.values())
    if drift != 0:
        # adjust on Tier1 first
        cats = TIER1 if drift > 0 else list(reversed(TIER1))
        idx = 0
        while drift != 0:
            quotas[cats[idx % len(cats)]] += 1 if drift>0 else -1
            drift += -1 if drift>0 else 1
            idx += 1
    assert sum(quotas.values()) == total
    return quotas

def tier_of(cat):
    if cat in TIER1: return 1
    if cat in TIER2: return 2
    if cat in TIER3: return 3
    return 4

def topup_across_tiers(df, want_total=TOTAL_TARGET, per_burst=60):
    if len(df) >= want_total: return df
    tiers_order = [1,2,3,4]
    seen = set(df["arxiv_id"])
    need = want_total - len(df)
    acc = []
    round_count = 0
    while need > 0:
        round_count += 1
        print(f"Top-up round {round_count}, need: {need} more papers")
        for t in tiers_order:
            cats = [c for c in (TIER1 if t==1 else TIER2 if t==2 else TIER3 if t==3 else TIER4)]
            per_cat = max(20, per_burst // max(1,len(cats)))
            for c in cats:
                extra = fetch_cat(c, goal_n=per_cat)
                if not extra: continue
                dfx = pd.DataFrame(extra)
                dfx = dfx[~dfx["arxiv_id"].isin(seen)]
                if len(dfx):
                    acc.append(dfx)
                    for aid in dfx["arxiv_id"]: seen.add(aid)
                    need = want_total - (len(df) + sum(len(a) for a in acc))
                    print(f"  Added {len(dfx)} new papers from {c}, need: {need}")
                    if need <= 0: break
            if need <= 0: break
        if not acc: break
        if round_count >= 3:  # limit rounds to prevent infinite loops
            print("Reached maximum top-up rounds")
            break
    if acc:
        df = pd.concat([df] + acc, ignore_index=True)\
               .drop_duplicates(subset=["arxiv_id"]).reset_index(drop=True)
    return df

def main():
    # Create output directory
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean papers.csv at start
    output_file = out_dir / "papers.csv"
    if output_file.exists():
        output_file.unlink()
        print(f"Cleaned existing {output_file}")
    
    # Plan quotas
    QUOTAS = plan_quotas()
    print(f"Planned quotas: {sum(QUOTAS.values())} total, {len(QUOTAS)} categories")
    print(f"Sample quotas: {list(QUOTAS.items())[:5]}")
    
    # 1) initial pass by plan
    all_rows = []
    per_cat_got = defaultdict(int)
    for cat, q in QUOTAS.items():
        print(f"Fetching {q} from {cat} ...")
        rows = fetch_cat(cat, goal_n=q)
        per_cat_got[cat] = len(rows)
        print(f"  got {len(rows)}")
        all_rows.extend(rows)
        
        # Save progress incrementally to papers.csv
        if rows:
            df_temp = pd.DataFrame(rows)
            df_temp_simple = df_temp[["arxiv_id", "title", "published", "summary"]].copy()
            df_temp_simple.columns = ["arxiv_id", "title", "date", "abstract"]
            
            # Append to file (first write includes header)
            output_file = out_dir / "papers.csv"
            if output_file.exists():
                df_temp_simple.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_temp_simple.to_csv(output_file, index=False)
            print(f"  Appended {len(rows)} papers to {output_file}")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["arxiv_id"]).reset_index(drop=True)
    print(f"Initial unique: {len(df)}")

    # 2) If under target, top-up in tier round-robin (keeps representation)
    df = topup_across_tiers(df, want_total=TOTAL_TARGET)

    print(f"Total unique: {len(df)}")
    print("By tier after dedup:")
    print(df["primary_cat"].map(tier_of).value_counts().sort_index())

    # Final save to CSV with only essential columns (overwrite with deduplicated data)
    output_file = out_dir / "papers.csv"
    essential_columns = ["arxiv_id", "title", "date", "abstract"]
    
    # Create a simplified dataframe with only essential columns
    df_simple = df[["arxiv_id", "title", "published", "summary"]].copy()
    df_simple.columns = ["arxiv_id", "title", "date", "abstract"]
    df_simple.to_csv(output_file, index=False)
    print(f"Final data saved to: {output_file} (deduplicated)")

    print("Per-category counts (by primary_cat):")
    print(df["primary_cat"].value_counts().head(20))
    print("Columns:", df.columns.tolist())
    
    return df


if __name__ == "__main__":
    df = main()
