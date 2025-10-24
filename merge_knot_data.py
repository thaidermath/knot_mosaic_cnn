"""
merge_knot_data.py

Join knotinfo.csv with mosaics/index.csv using a sequence of strategies:
  1) exact filename match
  2) extracted id match (e.g. 10_001 from 10_001-mt.png)
  3) normalized name exact match
  4) fuzzy name match (difflib)

Outputs:
- mosaics/merged_knotinfo.csv  (merged rows - left join of knotinfo + image info)
- prints a short report and writes unmatched lists to mosaics/
"""

from __future__ import annotations
import csv
import difflib
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(".")
KNOTINFO = ROOT / "knotinfo.csv"
MOSAIC_INDEX = ROOT / "mosaics" / "index.csv"
OUT = ROOT / "mosaics" / "merged_knotinfo.csv"
UNMATCHED_REPORT = ROOT / "mosaics" / "unmatched_report.txt"

ID_RE = re.compile(r"(\d{1,4}_\d{1,4})")  # matches patterns like 10_001
ALT_ID_RE = re.compile(r"(\d{1,6})")  # fallback: any integer token


def normalize_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    # remove punctuation except underscore/dash
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_id_from_filename(fname: str) -> Optional[str]:
    if not fname:
        return None
    m = ID_RE.search(fname)
    if m:
        return m.group(1)
    m2 = ALT_ID_RE.search(fname)
    if m2:
        return m2.group(1)
    return None


def normalize_mid(mid: Optional[str]) -> Optional[str]:
    """Normalize an id like '10_001' or '10_1' to a canonical '10_1' (no leading zeros).

    Returns None for empty inputs.
    """
    if not mid:
        return None
    s = str(mid)
    # match m_t pattern
    m = re.match(r"^(\d+)_(\d+)$", s)
    if m:
        a, b = m.group(1), m.group(2)
        try:
            return f"{int(a)}_{int(b)}"
        except Exception:
            return f"{a}_{b}"
    # if it's a single integer, return as-is
    m2 = re.match(r"^(\d+)$", s)
    if m2:
        return m2.group(1)
    return None


def parse_mosaic_tile_number(s: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """Parse a string like '{ 6 ; 22 }' or '6 ; 22' and return (m, t) as ints when possible.

    Returns (None, None) when parsing fails.
    """
    if not s:
        return None, None
    ss = str(s)
    # strip braces and whitespace
    ss = ss.replace("{", "").replace("}", "").strip()
    # try split by semicolon or comma
    parts = re.split(r"[;,]", ss)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return None, None
    try:
        if len(parts) >= 2:
            m = int(re.sub(r"[^0-9]", "", parts[0]))
            t = int(re.sub(r"[^0-9]", "", parts[1]))
            return m, t
        # single number
        m = int(re.sub(r"[^0-9]", "", parts[0]))
        return m, None
    except Exception:
        return None, None


def guess_filename_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["filename", "file", "image", "img", "img_file", "image_filename"]
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy detect
    for col in df.columns:
        low = col.lower()
        if "file" in low or "image" in low or "img" in low:
            return col
    return None


def guess_name_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["name", "label", "title", "knot", "knot_name"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        low = col.lower()
        if "name" in low or "label" in low or "title" in low:
            return col
    return None


def main():
    if not KNOTINFO.exists():
        print("knotinfo.csv not found at", KNOTINFO)
        return
    if not MOSAIC_INDEX.exists():
        print("mosaics/index.csv not found at", MOSAIC_INDEX)
        return

    ki = pd.read_csv(KNOTINFO, dtype=str).fillna("")
    mi = pd.read_csv(MOSAIC_INDEX, dtype=str).fillna("")

    print("knotinfo columns:", ki.columns.tolist())
    print("mosaic index columns:", mi.columns.tolist())

    # detect columns
    ki_fname_col = guess_filename_column(ki)
    ki_name_col = guess_name_column(ki)
    mi_fname_col = guess_filename_column(mi) or "filename"
    mi_name_col = guess_name_column(mi) or "name"

    print("Detected columns -> knotinfo filename:", ki_fname_col, "name:", ki_name_col)
    print("Detected columns -> mosaic filename:", mi_fname_col, "name:", mi_name_col)

    # Prepare working copies
    ki_work = ki.copy()
    mi_work = mi.copy()

    # Normalize name columns
    if ki_name_col:
        ki_work["__name_norm"] = ki_work[ki_name_col].apply(normalize_name)
    else:
        ki_work["__name_norm"] = ""

    mi_work["__name_norm"] = mi_work.get(mi_name_col, "").apply(normalize_name)

    # Add filename lower and extracted id
    if ki_fname_col:
        ki_work["__fname"] = ki_work[ki_fname_col].astype(str)
        ki_work["__fname_norm"] = ki_work["__fname"].str.lower()
        ki_work["__mid"] = ki_work["__fname"].apply(extract_id_from_filename)
        # normalized mid (strip leading zeros etc)
        ki_work["__mid_norm"] = ki_work["__mid"].apply(normalize_mid)
    else:
        ki_work["__fname"] = ""
        ki_work["__fname_norm"] = ""
        # try to extract an id from the Name column (e.g. '10_1') so we can still match by id
        if ki_name_col:
            ki_work["__mid"] = ki_work[ki_name_col].astype(str).apply(lambda s: extract_id_from_filename(s))
            ki_work["__mid_norm"] = ki_work["__mid"].apply(normalize_mid)
        else:
            ki_work["__mid"] = None
            ki_work["__mid_norm"] = None

    mi_work["__fname"] = mi_work[mi_fname_col].astype(str)
    mi_work["__fname_norm"] = mi_work["__fname"].str.lower()
    mi_work["__mid"] = mi_work["__fname"].apply(extract_id_from_filename)
    mi_work["__mid_norm"] = mi_work["__mid"].apply(normalize_mid)

    # parse mosaic tile-number in knotinfo (if present)
    if "Mosaic/Tile-Number" in ki_work.columns:
        ki_work[["mosaic_m", "mosaic_t"]] = ki_work["Mosaic/Tile-Number"].apply(lambda s: pd.Series(parse_mosaic_tile_number(s)))
    else:
        ki_work["mosaic_m"] = None
        ki_work["mosaic_t"] = None

    # also try to infer m,t from mosaic filenames (mi_work)
    def infer_from_fname(fname: str):
        # filenames like '10_001-mt.png' imply m=10, t=1 (or 001 -> 1). We parse the '10_001' part.
        mid = extract_id_from_filename(fname)
        if not mid:
            return None, None
        parts = mid.split("_")
        try:
            m = int(parts[0])
            t = int(parts[1]) if len(parts) > 1 else None
            return m, t
        except Exception:
            return None, None

    mi_work[["mosaic_m", "mosaic_t"]] = mi_work["__fname"].apply(lambda s: pd.Series(infer_from_fname(s)))

    # 1) Exact filename join (case-insensitive)
    merged = pd.merge(ki_work, mi_work, left_on="__fname_norm", right_on="__fname_norm", how="left", suffixes=("_ki", "_mi"))
    # Ensure match-flag columns exist so later logic and summary prints won't KeyError
    merged["__matched_via_id"] = False
    merged["__matched_via_name_exact"] = False
    merged["__matched_via_name_fuzzy"] = False

    matched_mask = merged["__fname_mi"].notna() if "__fname_mi" in merged.columns else merged["__fname"].notna()
    matched_exact_fname = merged[matched_mask].index.tolist()
    print(f"Exact filename matches: {len(matched_exact_fname)}")

    # For those not matched, try id-based matching
    unmatched = merged[~matched_mask].copy()
    # We'll look for a normalized mid column produced from either filename or Name
    if "__mid_norm_ki" in merged.columns or "__mid_norm" in merged.columns:
        # build map from mi mid_norm -> first row
        mi_mid_map = mi_work.dropna(subset=["__mid_norm"]).set_index("__mid_norm", drop=False)
        id_matches = []
        for idx, row in unmatched.iterrows():
            # merged may have the ki-side mid under '__mid_norm_ki' or '__mid_norm'
            mid = None
            if "__mid_norm_ki" in merged.columns:
                mid = row.get("__mid_norm_ki")
            else:
                mid = row.get("__mid_norm")
            if mid and mid in mi_mid_map.index:
                id_matches.append((idx, mi_mid_map.loc[mid].to_dict()))
        # apply id matches
        for idx, mi_row in id_matches:
            for k, v in mi_row.items():
                merged.at[idx, k + "_mi_idmatch"] = v
            merged.at[idx, "__matched_via_id"] = True

        # summary for id matches
        id_matched = int(merged["__matched_via_id"].fillna(False).sum())
        print(f"ID-based matches: {id_matched}")
    else:
        print("No ID column found for id-based matching.")

    # 3) Name exact match on normalized name for rows still unmatched
    still_unmatched_mask = merged.apply(lambda r: pd.isna(r.get("__fname_mi")) and not r.get("__matched_via_id", False), axis=1)
    still_unmatched = merged[still_unmatched_mask]
    name_map = mi_work.set_index("__name_norm", drop=False)
    name_exact_matches = []
    for idx, row in still_unmatched.iterrows():
        nm = row.get("__name_norm")
        if nm and nm in name_map.index:
            # assign
            for k, v in name_map.loc[nm].to_dict().items():
                merged.at[idx, k + "_mi_name"] = v
            merged.at[idx, "__matched_via_name_exact"] = True
            name_exact_matches.append(idx)
    print(f"Name-exact matches: {len(name_exact_matches)}")

    # 4) Fuzzy matching for remaining (difflib)
    still_unmatched_mask = merged.apply(
        lambda r: pd.isna(r.get("__fname_mi")) and not r.get("__matched_via_id", False) and not r.get("__matched_via_name_exact", False),
        axis=1,
    )
    still_unmatched = merged[still_unmatched_mask]
    mi_names = list(mi_work["__name_norm"].unique())
    fuzzy_matches = []
    for idx, row in still_unmatched.iterrows():
        name = row.get("__name_norm") or ""
        if not name:
            continue
        # use get_close_matches
        choices = difflib.get_close_matches(name, mi_names, n=1, cutoff=0.80)  # threshold: 0.8
        if choices:
            best = choices[0]
            # get the first matching row from mi_work
            mi_row = mi_work[mi_work["__name_norm"] == best].iloc[0].to_dict()
            for k, v in mi_row.items():
                merged.at[idx, k + "_mi_fuzzy"] = v
            merged.at[idx, "__matched_via_name_fuzzy"] = True
            merged.at[idx, "__fuzzy_match_score"] = 1.0  # difflib doesn't give score here; set 1.0 to mark matched
            fuzzy_matches.append((idx, best))
    print(f"Fuzzy name matches: {len(fuzzy_matches)}")

    # Build final merged table: prefer exact filename then id then exact name then fuzzy name
    # We'll pick columns from matching sources if present
    # For simplicity, add columns: image_filename_matched, image_url_matched
    def pick_image_filename(row):
        for suffix in ["_mi", "_mi_idmatch", "_mi_name", "_mi_fuzzy"]:
            key = "__fname" + suffix if "__fname" + suffix in row.index else "__fname"  # fallback
            # check several possible keys
            k1 = "__fname" + suffix
            k2 = "filename" + suffix
            # try original MI filename fields
            if k1 in row.index and pd.notna(row[k1]) and row[k1] != "":
                return row[k1]
            if k2 in row.index and pd.notna(row[k2]) and row[k2] != "":
                return row[k2]
        # fallback: use any mi filename column present
        for c in merged.columns:
            if c.startswith("__fname") and pd.notna(row[c]) and row[c] != "":
                return row[c]
        return ""

    def pick_image_url(row):
        # mosaic index often has 'url' column
        for suffix in ["_mi", "_mi_idmatch", "_mi_name", "_mi_fuzzy"]:
            c = "url" + suffix
            if c in row.index and pd.notna(row[c]) and row[c] != "":
                return row[c]
            if "url" in row.index and pd.notna(row["url"]) and row["url"] != "":
                return row["url"]
        # try 'url_mi' style
        for c in merged.columns:
            if c.startswith("url") and pd.notna(row[c]) and row[c] != "":
                return row[c]
        return ""

    merged["image_filename_matched"] = merged.apply(pick_image_filename, axis=1)
    merged["image_url_matched"] = merged.apply(pick_image_url, axis=1)

    # Pick mosaic number and tile number: prefer knotinfo parsed values, otherwise use matched mosaic info
    def pick_mosaic_number(row):
        # possible keys in merged from ki side
        for k in ("mosaic_m", "mosaic_m_ki"):
            if k in row.index and pd.notna(row[k]) and row[k] not in ("", None):
                return row[k]
        # from matched mi sources
        for suffix in ("_mi", "_mi_idmatch", "_mi_name", "_mi_fuzzy"):
            k = "mosaic_m" + suffix
            if k in row.index and pd.notna(row[k]) and row[k] not in ("", None):
                return row[k]
        # try mi_work generic column names
        for c in merged.columns:
            if c.startswith("mosaic_m") and pd.notna(row[c]) and row[c] not in ("", None):
                return row[c]
        return None

    def pick_tile_number(row):
        for k in ("mosaic_t", "mosaic_t_ki"):
            if k in row.index and pd.notna(row[k]) and row[k] not in ("", None):
                return row[k]
        for suffix in ("_mi", "_mi_idmatch", "_mi_name", "_mi_fuzzy"):
            k = "mosaic_t" + suffix
            if k in row.index and pd.notna(row[k]) and row[k] not in ("", None):
                return row[k]
        for c in merged.columns:
            if c.startswith("mosaic_t") and pd.notna(row[c]) and row[c] not in ("", None):
                return row[c]
        return None

    merged["mosaic number"] = merged.apply(pick_mosaic_number, axis=1)
    merged["Tile Number"] = merged.apply(pick_tile_number, axis=1)

    # Export merged (select original knotinfo columns plus the matched image info)
    # include original knotinfo columns (except the original Mosaic/Tile-Number), the two new mosaic columns, and matched image info
    output_cols = [c for c in ki.columns if c != "Mosaic/Tile-Number"] + ["mosaic number", "Tile Number", "image_filename_matched", "image_url_matched"]
    merged_out = merged.reindex(columns=output_cols)
    # rename columns to requested names
    rename_map = {
        "Crossing Number": "crossing_number",
        "Jones": "jones_polynomial",
        "Volume": "hyperbolic_volume",
        "Meridian Length": "meridian_length",
        "mosaic number": "mosaic_num",
        "Tile Number": "tile_num",
    }
    merged_out = merged_out.rename(columns=rename_map)
    merged_out.to_csv(OUT, index=False)
    print("Wrote merged output to", OUT)

    # Build unmatched report
    unmatched_rows = merged_out[merged_out["image_filename_matched"].fillna("") == ""]
    with open(UNMATCHED_REPORT, "w", encoding="utf-8") as f:
        f.write(f"Total knotinfo rows: {len(ki)}\n")
        f.write(f"Matched rows: {len(ki) - len(unmatched_rows)}\n")
        f.write(f"Unmatched rows: {len(unmatched_rows)}\n\n")
        for _, r in unmatched_rows.iterrows():
            f.write(str(r.to_dict()) + "\n")
    print(f"Unmatched: {len(unmatched_rows)} (details -> {UNMATCHED_REPORT})")

    # Print summary
    print("Summary:")
    print("  total knotinfo rows:", len(ki))
    print("  exact filename matches:", len(matched_exact_fname))
    # use .get to be defensive if the flag columns were never created for some reason
    id_matches_count = int(merged.get("__matched_via_id", pd.Series(False, index=merged.index)).fillna(False).sum())
    name_exact_count = int(merged.get("__matched_via_name_exact", pd.Series(False, index=merged.index)).fillna(False).sum())
    name_fuzzy_count = int(merged.get("__matched_via_name_fuzzy", pd.Series(False, index=merged.index)).fillna(False).sum())
    print("  id matches:", id_matches_count)
    print("  name exact matches:", name_exact_count)
    print("  fuzzy name matches:", name_fuzzy_count)
    print("  unmatched:", len(unmatched_rows))
    print("All done.")


if __name__ == "__main__":
    main()