from __future__ import annotations
from difflib import SequenceMatcher
import textwrap
import json
import re
from typing import Mapping
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple,
                    Union, Mapping, ClassVar, Any, Set)
import io
import pandas as pd
from src.mappers.antibiotic_to_grams import ABX_TARGET_MAP, CATALOG

def pick_abx_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_Tested")]

def merge_antibiotic_data(
    existing_file: str,
    who_file: str,
    output_file: str,
    class_column: str = "Class",
    broad_class_column: str = "Broad Class",
):
    """
    DYNAMIC merge with fuzzy matching for antibiotic names.
    
    Features:
    - Case-insensitive + normalization (remove spaces, dashes, slashes)
    - Fuzzy string matching (difflib) for near-misses
    - Comprehensive synonym mapping (1000+ aliases)
    - Manual brand name cross-reference
    - Collision detection and reporting
    - Preserves all rows from existing_file
    - WHO data filled with 'Not Set' where missing
    
    Args:
        existing_file: Path to existing antibiotic classification CSV
        who_file: Path to WHO AWaRe classification CSV
        output_file: Path to save merged result
        class_column: Column name for antibiotic class (default: "Class")
        broad_class_column: Column name for broad class output (default: "Broad Class")
    
    Returns:
        Merged DataFrame with all records
    """
    
    # Load datasets
    existing_df = pd.read_csv(existing_file)
    who_df = pd.read_csv(who_file)
    
    print(f"\n{'='*70}")
    print(f"ANTIBIOTIC MERGE: Dynamic Matching with Fuzzy Logic (EXPANDED)")
    print(f"{'='*70}")
    
    print(f"\nExisting file: {len(existing_df)} records")
    print(f"WHO file: {len(who_df)} records")
    
    # ===== STEP 1: NORMALIZATION FUNCTION =====
    def normalize_name(name):
        """Normalize antibiotic names for matching."""
        if pd.isna(name):
            return ""
        
        name = str(name).strip().lower()
        # Remove common separators and spaces
        name = name.replace("/", "").replace("-", "").replace(" ", "")
        # Remove leading/trailing underscores
        name = name.replace("_iv", "").replace("_oral", "")
        return name
    
    # ===== STEP 2: COMPREHENSIVE SYNONYM MAPPING (1000+ entries) =====
    
    from src.mappers.synonym_mapping import synonym_mapping
    # Create normalized → original mapping from WHO file
    who_normalized = {}
    for idx, row in who_df.iterrows():
        original_name = row["Antibiotic"]
        normalized = normalize_name(original_name)
        who_normalized[normalized] = original_name
    
    print(f"\nWHO file normalized entries: {len(who_normalized)}")
    print(f"Synonym mapping size: {len(synonym_mapping)} aliases")
    
    # ===== FUZZY MATCHING FUNCTION =====
    def fuzzy_match(existing_name, who_dict, threshold=0.80):
        """Find best match using similarity ratio."""
        normalized = normalize_name(existing_name)
        
        # Direct match
        if normalized in who_dict:
            return who_dict[normalized], 1.0
        
        # Synonym mapping
        if normalized in synonym_mapping:
            syn = normalize_name(synonym_mapping[normalized])
            if syn in who_dict:
                return who_dict[syn], 1.0
        
        # Fuzzy matching fallback
        best_match = None
        best_score = 0
        
        for who_normalized_name, who_original_name in who_dict.items():
            score = SequenceMatcher(None, normalized, who_normalized_name).ratio()
            if score > best_score:
                best_score = score
                best_match = who_original_name
        
        if best_score >= threshold:
            return best_match, best_score
        
        return None, best_score
    
    # ===== PERFORM MERGE =====
    print(f"\nPerforming fuzzy merge with {len(existing_df)} existing antibiotics...")
    
    merge_results = []
    matched_count = 0
    perfect_matches = 0
    fuzzy_matches = 0
    unmatched = []
    
    for idx, row in existing_df.iterrows():
        existing_name = row["Antibiotic Name"]
        matched_who_name, similarity = fuzzy_match(existing_name, who_normalized, threshold=0.80)
        
        if matched_who_name:
            matched_count += 1
            
            if similarity == 1.0:
                perfect_matches += 1
                match_type = "PERFECT"
            else:
                fuzzy_matches += 1
                match_type = f"FUZZY({similarity:.2f})"
            
            # Get WHO row
            who_row = who_df[who_df["Antibiotic"] == matched_who_name].iloc[0]
            
            # Build merged row
            merged_row = row.copy()
            merged_row["WHO_Match"] = matched_who_name
            merged_row["Match_Type"] = match_type
            merged_row["WHO_Class"] = who_row.get("WHO_Class", "Not Set")
            merged_row["WHO_ATC_code"] = who_row.get("WHO_ATC_code", "Not Set")
            merged_row["Category"] = who_row.get("Category", "Not Set")
            merged_row["Listed_on_EML_2019"] = who_row.get("Listed_on_EML_2019", "Not Set")
            
            merge_results.append(merged_row)
        else:
            unmatched.append(existing_name)
            merged_row = row.copy()
            merged_row["WHO_Match"] = "NO MATCH"
            merged_row["Match_Type"] = "UNMATCHED"
            merged_row["WHO_Class"] = "Not Set"
            merged_row["WHO_ATC_code"] = "Not Set"
            merged_row["Category"] = "Not Set"
            merged_row["Listed_on_EML_2019"] = "Not Set"
            merge_results.append(merged_row)
    
    merged_df = pd.DataFrame(merge_results)
    
    # ===== ADD BROAD CLASS =====
    def compute_broad_class(row):
        """Collapse all β-lactams into one broad category."""
        if class_column not in row or pd.isna(row[class_column]):
            return "Not Set"
        
        cls = str(row[class_column]).lower()
        if "β-lactam" in str(row[class_column]) or "beta-lactam" in cls:
            return "β-lactam"
        return row[class_column]
    
    if class_column in merged_df.columns:
        merged_df[broad_class_column] = merged_df.apply(compute_broad_class, axis=1)
    
    # ===== SAVE AND REPORT =====
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"MERGE RESULTS")
    print(f"{'='*70}")
    print(f"\nExisting antibiotics matched: {matched_count}/{len(existing_df)} ({100*matched_count/len(existing_df):.1f}%)")
    print(f"  - Perfect matches: {perfect_matches}")
    print(f"  - Fuzzy matches: {fuzzy_matches}")
    print(f"  - Unmatched: {len(unmatched)} ({100*len(unmatched)/len(existing_df):.1f}%)")
    
    if unmatched:
        print(f"\n⚠ UNMATCHED ANTIBIOTICS ({len(unmatched)}):")
        for ab in sorted(unmatched)[:100]:
            print(f"  - {ab}")
        if len(unmatched) > 100:
            print(f"  ... and {len(unmatched) - 100} more")
    
    matched_who_names = set(merged_df[merged_df["WHO_Match"] != "NO MATCH"]["WHO_Match"])
    unmatched_who = set(who_df["Antibiotic"]) - matched_who_names
    
    if unmatched_who:
        print(f"\n⚠ WHO ANTIBIOTICS NOT IN EXISTING FILE ({len(unmatched_who)}):")
        for ab in sorted(unmatched_who)[:100]:
            print(f"  - {ab}")
        if len(unmatched_who) > 100:
            print(f"  ... and {len(unmatched_who) - 100} more")
    
    print(f"\n✓ Merged dataset saved to: {output_file}")
    print(f"  Total records: {len(merged_df)}")
    print(f"  New columns: WHO_Match, Match_Type, WHO_Class, WHO_ATC_code, Category, Listed_on_EML_2019, {broad_class_column}")
    
    return merged_df

def compute_row_features(row, antibiotics, class_map, who_class_map):
    """
    Uses the *_Tested flags for each antibiotic to avoid mutating raw R/I/S values.
    """
    tested_abx = [abx for abx in antibiotics if row.get(f"{abx}_Tested", 0) == 1]
    classes = set(class_map.get(abx) for abx in tested_abx if abx in class_map)
    who_flags = [who_class_map.get(abx) for abx in tested_abx if abx in who_class_map]

    num_classes = len(classes)
    is_critical = int(any(cls in ['Watch', 'Reserve'] for cls in who_flags))
    is_reserve  = int(any(cls == 'Reserve' for cls in who_flags))

    return pd.Series([num_classes, is_critical, is_reserve])


def prepare_feature_inputs(ars_data: pd.DataFrame, who_data: pd.DataFrame):
    """
    - Returns antibiotic names and mappings.
    - Creates <abx>_Tested flags WITHOUT touching the raw antibiotic result columns (R/I/S/NaN).
    """
    antibiotics = [col for col in ars_data.columns if col in who_data['Full Name'].values]
    who_filtered = who_data[who_data['Full Name'].isin(antibiotics)]

    class_map = dict(zip(who_filtered['Full Name'], who_filtered['Class']))
    who_class_map = dict(zip(who_filtered['Full Name'], who_filtered['Category']))

    # Create _Tested flags in a non-destructive way
    for abx in antibiotics:
        tested_col = f"{abx}_Tested"
        if tested_col not in ars_data.columns:
            ars_data[tested_col] = ars_data[abx].notna().astype("Int8")

    return antibiotics, class_map, who_class_map

def format_antibiotic_label(abbr, name, style='abbr'):
    if style == 'abbr':
        return abbr
    elif style == 'name':
        return name
    elif style == 'both':
        return f"{abbr} ({name})"
    else:
        raise ValueError(
            "Invalid style. Choose from 'abbr', 'name', or 'both'.")


def build_mappings(class_dict, catalog_str):
    # Parse CATALOG into {abbr: full_name}
    abbr2full = {}
    for item in catalog_str.split(","):
        parts = item.strip().split(" - ")
        if len(parts) == 2:
            abbr, fullname = parts
            abbr2full[abbr.strip()] = fullname.strip()

    # Build {abbr: class_name} from existing dict
    class_mapping = {}
    for class_name, abx_list in class_dict.items():
        for abx in abx_list:
            abbr = abx.split(" - ")[0].replace("_Tested", "").strip()
            class_mapping[abbr] = class_name

    return class_mapping, abbr2full


def decorate_label_dynamic(label, abbr2full, class_mapping, target_map, format_type="full", include_class=False):
    """
    Decorates label based on format_type ('full' or 'abbr').
    Example outputs:
      - full: 'Amoxicillin/clavulanic acid [Penicillin + Beta-lactamase inhibitor] (mixed)'
      - abbr: 'AMC [Penicillin + Beta-lactamase inhibitor] (mixed)'
    """
    label_clean = label.replace("_Tested", "").strip()

    if format_type == "abbr":
        abbr = label_clean
        full = abbr2full.get(abbr, None)
    else:
        # Try to find abbreviation from full name
        reverse_map = {v: k for k, v in abbr2full.items()}
        abbr = reverse_map.get(label_clean, None)
        full = label_clean

    # If we fail to find abbreviation, try partial match
    if not abbr:
        match = next((a for a, f in abbr2full.items()
                      if label_clean.lower() in f.lower()), None)
        if match:
            abbr = match
            full = abbr2full[abbr]
        else:
            return label  # No match

    abx_class = class_mapping.get(abbr, "Unknown")
    tag = target_map.get(abbr, "null")
    if include_class:
        return f"{full if format_type == 'full' else abbr} [{abx_class}] ({tag})"
    else:
        return f"{full if format_type == 'full' else abbr} ({tag})"


def get_label(abx_cols=[], antibiotic_class_map="", format_type="abbr", enrich=False, include_class=False):
    label_map = create_label_mapping(
        abx_cols=abx_cols, format_type=format_type)

    if enrich:
        CLASS_MAPPING, ABBR2FULL = build_mappings(
            antibiotic_class_map, CATALOG)

        label_map = {
            abbr: decorate_label_dynamic(
                full_name, ABBR2FULL,
                CLASS_MAPPING, ABX_TARGET_MAP,
                format_type=format_type,
                include_class=include_class
            )
            for abbr, full_name in label_map.items()
        }

    return label_map


def create_label_mapping(abx_cols=[], format_type: str = 'combined', remove_suffix: str = "_Tested") -> dict:
    """
    Creates a dictionary for cleaner labels in various formats.

    Args:
        format_type (str): The desired format for the labels. 
                        Options: 'abbr', 'full', 'combined'.
        remove_suffix (str): The suffix to remove from column names.

    Returns:
        dict: A mapping from old labels to new, cleaner labels.
    """
    label_map = {}
    for col in abx_cols:
        # First, remove the suffix to clean up the string
        cleaned_col = col.replace(remove_suffix, "")

        # Try to split the string into abbreviation and full name
        parts = cleaned_col.split(' - ', 1)

        if len(parts) == 2:
            abbr, full_name = parts
            if format_type == 'abbr':
                label_map[col] = abbr
            elif format_type == 'full':
                label_map[col] = full_name
            elif format_type == 'combined':
                # Keep the "Abbr - Full Name" format
                label_map[col] = cleaned_col
            else:
                # Default to combined if format is invalid
                label_map[col] = cleaned_col
        else:
            # If the label doesn't contain " - ", just use the cleaned version
            label_map[col] = cleaned_col

    return label_map


def filter_antibiotic_group_items(df, groups):
    """
    Filter out antibiotics not found in our dataset but correctly place in the class
    this is needed to avoid errors
    """
    present = {g: [a for a in ab if a in df.columns]
               for g, ab in groups.items()}
    missing = {g: [a for a in ab if a not in df.columns]
               for g, ab in groups.items()}
    return {g: v for g, v in present.items() if v}, missing




def antibiotic_aggregate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    tested_cols = [c for c in df.columns if c.endswith("_Tested")]
    lab_testing_summary = (
        df.groupby(group_col)[tested_cols]
        .sum()
    )
    lab_testing_summary["Total_Tests"] = lab_testing_summary.sum(axis=1)
    lab_testing_summary_sorted = lab_testing_summary.sort_values(
        by="Total_Tests",
        ascending=False
    )
    return lab_testing_summary_sorted.head(50)

def learn_panel_implied_map(
    txn_df: pd.DataFrame,
    *,
    threshold: float = 0.98,
    min_count: int = 200,
    max_global_test_rate: Optional[float] = None,
    tested_suffix: str = "_T",
) -> Dict[str, Set[str]]:
    """
    Learn a panel implication map from the transaction matrix tested-layer.

    Returns
    -------
    panel_map : dict
        Maps source drug code -> set of drug codes that are (near-)deterministically co-tested.
        Example: {"MCL": {"AMC","AMP","AMX",...}}

    Definition
    ----------
    b is "panel-implied" by a if:
        P(b_T = 1 | a_T = 1) >= threshold
    computed over episodes (rows) in txn_df.

    Guardrails
    ----------
    - Only compute implications for a if count(a_T=1) >= min_count (stability).
    - Optionally ignore b that are tested in almost all episodes (max_global_test_rate),
      because they are not informative escalation targets anyway.
    """

    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in (0, 1].")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")

    t_cols = [c for c in txn_df.columns if c.endswith(tested_suffix)]
    if not t_cols:
        raise ValueError(f"No tested columns found with suffix '{tested_suffix}'.")

    # Convert to a dense boolean array for fast ops
    T = txn_df[t_cols].astype(bool)

    # global testing rates P(x_T=1)
    global_rate = T.mean(axis=0)  # Series indexed by t_cols

    # codes for each tested column
    codes = {col: col[: -len(tested_suffix)] for col in t_cols}

    panel_map: Dict[str, Set[str]] = {}

    # Precompute column arrays to speed things up
    T_np = T.to_numpy(dtype=bool)
    col_index = {col: i for i, col in enumerate(t_cols)}

    for a_col in t_cols:
        a_idx = col_index[a_col]
        a_mask = T_np[:, a_idx]
        a_count = int(a_mask.sum())

        a_code = codes[a_col]
        panel_map[a_code] = set()

        if a_count < min_count:
            continue  # too rare to learn stable panel structure

        # Compute P(b_T=1 | a_T=1) for all b at once:
        # among rows where a_T=1, what fraction have b_T=1
        sub = T_np[a_mask, :]  # rows conditioned on a_T=1
        cond_rate = sub.mean(axis=0)  # vector over b columns

        for b_col in t_cols:
            b_idx = col_index[b_col]
            b_code = codes[b_col]

            if b_code == a_code:
                continue

            # Optionally remove "always tested" drugs as targets
            if max_global_test_rate is not None:
                if float(global_rate[b_col]) >= float(max_global_test_rate):
                    continue

            if float(cond_rate[b_idx]) >= threshold:
                panel_map[a_code].add(b_code)

        # remove empties for cleanliness (optional)
        if not panel_map[a_code]:
            panel_map.pop(a_code, None)

    return panel_map
