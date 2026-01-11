import textwrap
import json
import re
from typing import Mapping
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple,
                    Union, Mapping, ClassVar, Any)
import io
import pandas as pd
from src.mappers.antibiotic_to_grams import ABX_TARGET_MAP, CATALOG

def pick_abx_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_Tested")]

def merge_antibiotic_data(
    existing_file: str,
    who_file: str,
    output_file: str,
    class_column: str = "Class",          # <-- this was "Antibiotic Class"
    broad_class_column: str = "Broad Class",
):
    """
    Merges an existing antibiotic classification dataset with WHO classification data,
    and adds a broad pharmacology class that collapses all β-lactams into one label.

    - Case-insensitive merge on antibiotic name
    - WHO columns filled with 'Not Set' if missing
    - Adds a 'broad class' layer where all β-lactam subclasses are grouped as 'β-lactam'
    """

    # 1) Load the datasets
    existing_df = pd.read_csv(existing_file)
    who_df = pd.read_csv(who_file)

    # Keep original name for later restore
    original_antibiotic_names = existing_df["Antibiotic Name"].copy()

    # 2) Case-insensitive merge
    existing_df["Antibiotic Name"] = existing_df["Antibiotic Name"].str.lower()
    who_df["Antibiotic"] = who_df["Antibiotic"].str.lower()

    merged_df = pd.merge(
        existing_df,
        who_df,
        left_on="Antibiotic Name",
        right_on="Antibiotic",
        how="left",
    )

    # Drop redundant merge key from WHO side
    merged_df.drop(columns=["Antibiotic"], inplace=True)

    # 3) Fill WHO columns with "Not Set" only
    who_columns = [col for col in who_df.columns if col != "Antibiotic"]
    existing_columns = set(merged_df.columns)
    who_columns = [col for col in who_columns if col in existing_columns]

    merged_df[who_columns] = merged_df[who_columns].fillna("Not Set")

    # Restore original casing of 'Antibiotic Name'
    merged_df["Antibiotic Name"] = original_antibiotic_names

    # 4) Add broad pharmacology class (collapse all β-lactams)

    def compute_broad_class(row):
        """
        Returns a broad pharmacology class.
        All β-lactam subclasses are grouped into 'β-lactam'.
        Non-β-lactams keep their original fine-grained class.
        """
        if class_column not in row or pd.isna(row[class_column]):
            return "Not Set"

        cls = str(row[class_column])

        # Any indicator of β-lactam membership?
        # Handles:
        #  - "Penicillin (β-lactam)"
        #  - "Third-gen cephalosporin (β-lactam)"
        #  - "Carbapenem (β-lactam)"
        #  - "β-lactam/β-lactamase inhibitor"
        #  - "Monobactam (β-lactam)"
        if "β-lactam" in cls or "beta-lactam" in cls.lower():
            return "β-lactam"

        # Otherwise just keep original class as its broad label
        return cls

    if class_column in merged_df.columns:
        merged_df[broad_class_column] = merged_df.apply(compute_broad_class, axis=1)
    else:
        merged_df[broad_class_column] = "Not Set"
        print(
            f"⚠ Warning: class_column '{class_column}' not found. "
            f"'{broad_class_column}' set to 'Not Set' for all rows."
        )

    # 5) Save result
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset with broad class saved to {output_file}")


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