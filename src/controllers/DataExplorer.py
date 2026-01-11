"""
DataExplorer_v3.py (Corrected + Robust + Publication-ready + Cohesive Styling)

WHAT’S NEW / FIXED (per your latest requests)
1) FilterConfig integration retained (imports your existing FilterConfig)
2) Robust boolean handling (bool / 0/1 / "true"/"false")
3) Robust outcome handling (strip/upper, handles "R ", "r", etc.)
4) Stage-by-stage audit logging
5) Prevents analysis/plots when cohort is empty
6) Fixes plot axis-title copy/paste bugs
7) Larger text everywhere (titles, axis labels, ticks, legend, bar text)
8) Color policy updated for publication consistency:
   - If <= 3 categories -> distinct colors
   - If > 3 categories -> cohesive color scheme:
       * Bar charts: single consistent fill color (all bars same)
       * Pie charts: monochrome shades of the same base color
9) Age bars sorted by true age order (0 -> max, handles "≥95 years", "80-84 years", "0 years")
10) Multi-format export (HTML, PNG, SVG, PDF) supported (requires kaleido for images)

Requirements:
    pip install pandas numpy plotly kaleido

Usage:
    explorer = DataExplorer(data_path="./data.parquet", filter_config=config)
    explorer.run_analysis()
    explorer.print_summary_report()
    explorer.generate_plots(output_dir="./publication_figures", formats=["png","svg","html"])
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re
import warnings
import colorsys

from src.controllers.filters.FilteringStrategy import FilterConfig

warnings.filterwarnings("ignore")


class DataExplorer:
    """
    Exploratory Data Analysis for ARS Surveillance Data.
    Robust cohort selection + publication-ready Plotly figures + CSV export for reporting.
    """

    # Distinct colors for small-category plots (<=3 categories)
    SMALL_N_PALETTE = [
        "#0072B2",  # Blue
        "#E69F00",  # Orange
        "#009E73",  # Green
    ]

    # Primary publication color used for cohesive plots (>3 categories)
    PRIMARY = "#0072B2"  # consistent publication blue

    def __init__(
        self,
        data_path: str,
        filter_config: Optional[FilterConfig] = None,
        palette_name: str = "publication_blue",
        required_covariates: Optional[List[str]] = None,
        drop_missing_covariates: bool = True,
    ):
        self.data_path = Path(data_path)
        self.raw_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, Any] = {}
        self.antibiotic_columns: Dict[str, Dict[str, str]] = {}
        self.output_dir: Optional[Path] = None
        self.export_formats = ["html", "png", "svg"]

        self.filter_config = filter_config
        self.palette_name = palette_name

        # Required covariates (rows missing any of these will be dropped)
        self.required_covariates = required_covariates or [
            "CareType",
            "ARS_WardType",
            "Sex",
            "AgeRange",
            "AgeGroup",
        ]
        self.drop_missing_covariates = drop_missing_covariates

        # PUBLICATION TEXT SIZES
        self.font_settings = dict(family="Arial, sans-serif", size=18, color="#1a1a1a")
        self.title_font = dict(family="Arial, sans-serif", size=28, color="#000000")
        self.label_font = dict(family="Arial, sans-serif", size=22, color="#000000")
        self.tick_font = dict(family="Arial, sans-serif", size=18, color="#1a1a1a")
        self.legend_font = dict(family="Arial, sans-serif", size=18, color="#1a1a1a")

        self.bar_text_size = 18
        self.pie_text_size = 18

        # Bigger canvas for publication export
        self.fig_width = 1900
        self.fig_height = 1100

        self.template = "plotly_white"
        self.paper_bg = "white"
        self.plot_bg = "white"

        # If True, generate_plots will also save CSV tables
        self.save_tables_default = True

    # =========================================================================
    # Robust coercion helpers
    # =========================================================================
    def _coerce_bool_series(self, s: pd.Series) -> pd.Series:
        """Coerce bool-like series: bool, 0/1, 'true'/'false' -> boolean; unknown -> NaN."""
        if s.dtype == bool:
            return s

        if pd.api.types.is_numeric_dtype(s):
            return s.map({1: True, 0: False})

        x = s.astype(str).str.strip().str.lower()
        mapping = {
            "true": True, "t": True, "yes": True, "y": True, "1": True,
            "false": False, "f": False, "no": False, "n": False, "0": False,
        }
        return x.map(mapping)

    def _clean_outcome_frame(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Standardize outcomes to uppercase stripped strings; NaNs become empty."""
        out = df[cols].copy()
        for c in cols:
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"NAN": "", "NONE": "", "NULL": ""})
            )
        return out

    def _stage_report(self, stage: str, df: pd.DataFrame, base_n: int) -> None:
        pct = (len(df) / base_n * 100) if base_n else 0
        print(f"{stage}: {len(df):,} records ({pct:.1f}%)")

    # =========================================================================
    # Cohesive color system (your requested rule)
    # =========================================================================
    def _hex_to_rgb01(self, hex_color: str) -> Tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return r, g, b

    def _rgb01_to_hex(self, r: float, g: float, b: float) -> str:
        return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

    def _mono_shades(self, base_hex: str, n: int, l_min: float = 0.35, l_max: float = 0.75) -> List[str]:
        """Generate n monochrome shades by varying lightness in HLS."""
        if n <= 0:
            return []
        r, g, b = self._hex_to_rgb01(base_hex)
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        if n == 1:
            rr, gg, bb = colorsys.hls_to_rgb(h, 0.55, s)
            return [self._rgb01_to_hex(rr, gg, bb)]

        shades = []
        for i in range(n):
            li = l_min + (l_max - l_min) * (i / (n - 1))
            rr, gg, bb = colorsys.hls_to_rgb(h, li, s)
            shades.append(self._rgb01_to_hex(rr, gg, bb))
        return shades

    def _colors_for_plot(self, n: int, kind: str = "bar") -> List[str]:
        """
        Rule:
        - if <= 3 categories -> distinct colors
        - if > 3 categories -> cohesive scheme:
            bar -> same fill color for all bars
            pie -> monochrome shades of same base color
        """
        if n <= 0:
            return []
        if n <= 3:
            return self.SMALL_N_PALETTE[:n]
        if kind == "pie":
            return self._mono_shades(self.PRIMARY, n)
        return [self.PRIMARY] * n

    # =========================================================================
    # Age sorting helpers
    # =========================================================================
    def _age_sort_key(self, label: Any) -> int:
        """
        Sort labels like:
          '0 years', '80-84 years', '≥95 years'
        Returns the lower-bound numeric age (>=95 -> 95).
        """
        if pd.isna(label):
            return 10**9
        s = str(label).strip()

        m = re.search(r"≥\s*(\d+)", s)
        if m:
            return int(m.group(1))

        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
        if m:
            return int(m.group(1))

        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))

        return 10**9

    # =========================================================================
    # Antibiotic column detection
    # =========================================================================
    def _detect_antibiotic_columns(self) -> Dict[str, Dict]:
        antibiotics: Dict[str, Dict[str, Optional[str]]] = {}
        pattern = r"^([A-Z]{2,3})\s+-\s+(.+?)_(Tested|Outcome)$"

        for col in self.raw_df.columns:
            match = re.match(pattern, col)
            if not match:
                continue
            code, name, col_type = match.group(1), match.group(2), match.group(3)

            if code not in antibiotics:
                antibiotics[code] = {"name": name, "tested": None, "outcome": None}

            if col_type == "Tested":
                antibiotics[code]["tested"] = col
            else:
                antibiotics[code]["outcome"] = col

        return {
            code: info for code, info in antibiotics.items()
            if info["tested"] is not None and info["outcome"] is not None
        }

    # =========================================================================
    # Missing covariates handling
    # =========================================================================
    def _drop_missing_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where ANY required covariate is missing.
        Excludes antibiotic Tested/Outcome columns from the missingness rule.
        Treats empty strings / whitespace as missing for string columns.
        """
        if not self.drop_missing_covariates:
            return df

        antibiotic_cols = set()
        if self.antibiotic_columns:
            for info in self.antibiotic_columns.values():
                antibiotic_cols.add(info["tested"])
                antibiotic_cols.add(info["outcome"])

        req = [c for c in self.required_covariates if c in df.columns and c not in antibiotic_cols]
        if not req:
            print("[COVARIATES] No required covariates present in dataset (or all excluded). Skipping drop.")
            return df

        before = len(df)

        for c in req:
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].astype(str).str.strip()
                df.loc[df[c].isin(["", "nan", "None", "NULL", "null"]), c] = np.nan

        df2 = df.dropna(subset=req)
        after = len(df2)

        removed = before - after
        pct_retained = (after / before * 100) if before else 0.0
        print(
            f"[COVARIATES] Dropped rows with missing required covariates ({', '.join(req)}): "
            f"{before:,} → {after:,} (removed {removed:,}, retained {pct_retained:.1f}%)"
        )
        return df2

    # =========================================================================
    # Load + filter
    # =========================================================================
    def load_and_filter(self) -> pd.DataFrame:
        print("=" * 80)
        print("LOADING AND FILTERING DATA")
        print("=" * 80)

        if self.data_path.is_dir():
            print("\n✓ Detected parquet directory (partitioned format)")
            print(f"  Path: {self.data_path}")
            self.raw_df = pd.read_parquet(self.data_path)
        elif str(self.data_path).endswith(".parquet"):
            print("\n✓ Detected parquet file")
            print(f"  Path: {self.data_path}")
            self.raw_df = pd.read_parquet(self.data_path)
        elif str(self.data_path).endswith(".csv"):
            print("\n✓ Detected CSV file")
            print(f"  Path: {self.data_path}")
            self.raw_df = pd.read_csv(self.data_path, low_memory=False)
        else:
            print("\n✓ Attempting auto-detection...")
            try:
                self.raw_df = pd.read_parquet(self.data_path)
                print("  ✓ Successfully loaded as parquet")
            except Exception:
                try:
                    self.raw_df = pd.read_csv(self.data_path, low_memory=False)
                    print("  ✓ Successfully loaded as CSV")
                except Exception:
                    raise ValueError("Unable to load file as parquet or CSV")

        print(f"\n✓ Loaded {len(self.raw_df):,} records")
        print(f"✓ Shape: {self.raw_df.shape}")

        self.antibiotic_columns = self._detect_antibiotic_columns()
        print(f"\n✓ Detected {len(self.antibiotic_columns)} antibiotics dynamically")
        if self.antibiotic_columns:
            print(f"  Examples: {list(self.antibiotic_columns.keys())[:5]}")

        base_n = len(self.raw_df)
        df = self.raw_df.copy()

        if self.filter_config is not None:
            df, _ = self.filter_config.apply(df)
            self._stage_report("[AFTER FilterConfig]", df, base_n)

        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        for bcol in ["IsSpecificlyExcluded_Screening", "IsSpecificlyExcluded_Pathogen"]:
            if bcol in df.columns:
                df[bcol] = self._coerce_bool_series(df[bcol])

        # Drop missing covariates early
        df = self._drop_missing_covariates(df)

        # STAGE 1: Pathogen
        if "Pathogen" in df.columns:
            df = df[df["Pathogen"].astype(str).str.contains("Escherichia coli", case=False, na=False)]
        self._stage_report("[STAGE 1] Pathogen = E. coli", df, base_n)

        # STAGE 2: Year window
        if "Year" in df.columns:
            df = df[df["Year"].between(2019, 2023)]
        self._stage_report("[STAGE 2] Year 2019–2023", df, base_n)

        # STAGE 3: Exclude screening
        if "IsSpecificlyExcluded_Screening" in df.columns:
            df = df[df["IsSpecificlyExcluded_Screening"].fillna(False) == False]
        self._stage_report("[STAGE 3] Not screening-excluded", df, base_n)

        # STAGE 4: Exclude pathogen QC
        if "IsSpecificlyExcluded_Pathogen" in df.columns:
            df = df[df["IsSpecificlyExcluded_Pathogen"].fillna(False) == False]
        self._stage_report("[STAGE 4] Not pathogen-excluded", df, base_n)

        # STAGE 5: >=3 antibiotics tested
        if self.antibiotic_columns:
            tested_cols = [info["tested"] for info in self.antibiotic_columns.values()]
            for col in tested_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df["n_tested"] = df[tested_cols].sum(axis=1)
            df = df[df["n_tested"] >= 3]
        self._stage_report("[STAGE 5] ≥ 3 antibiotics tested", df, base_n)

        # STAGE 6: >=1 resistant
        if self.antibiotic_columns:
            outcome_cols = [info["outcome"] for info in self.antibiotic_columns.values()]
            outcomes = self._clean_outcome_frame(df, outcome_cols)
            has_resistance = (outcomes == "R").any(axis=1)
            df = df[has_resistance]
        self._stage_report("[STAGE 6] ≥ 1 resistant outcome", df, base_n)

        self.filtered_df = df.copy()
        self.metrics["initial_count"] = base_n
        self.metrics["final_count"] = len(self.filtered_df)
        self.metrics["retention_pct"] = (len(self.filtered_df) / base_n * 100) if base_n else 0

        print("\n" + "=" * 80)
        print(f"ANALYTICAL COHORT: {self.metrics['final_count']:,} isolates")
        print(f"Overall retention: {self.metrics['retention_pct']:.1f}%")
        print("=" * 80 + "\n")

        if self.filtered_df.empty:
            print("⚠️ Filtered cohort is EMPTY. No analysis/plots will be generated.\n")

        return self.filtered_df

    # =========================================================================
    # Analyses
    # =========================================================================
    def analyze_temporal(self) -> Optional[pd.DataFrame]:
        if self.filtered_df is None or self.filtered_df.empty or "Year" not in self.filtered_df.columns:
            return None

        year = pd.to_numeric(self.filtered_df["Year"], errors="coerce").dropna()
        if year.empty:
            return None

        counts = year.value_counts().sort_index()
        total = len(self.filtered_df)

        temporal_df = pd.DataFrame({
            "Year": counts.index.astype(int),
            "Count": counts.values,
            "Percentage": (counts.values / total * 100).round(1),
        })
        self.metrics["temporal"] = temporal_df
        return temporal_df

    def analyze_specimen(self) -> Optional[pd.DataFrame]:
        if self.filtered_df is None or self.filtered_df.empty:
            return None
        if "TextMaterialgroupRkiL0" not in self.filtered_df.columns:
            return None

        counts = self.filtered_df["TextMaterialgroupRkiL0"].value_counts()
        total = len(self.filtered_df)

        specimen_df = pd.DataFrame({
            "Specimen Source": counts.index,
            "Count": counts.values,
            "Percentage": (counts.values / total * 100).round(1),
        }).sort_values("Count", ascending=False)

        self.metrics["specimen"] = specimen_df
        return specimen_df

    def analyze_ward_type(self) -> Optional[pd.DataFrame]:
        if self.filtered_df is None or self.filtered_df.empty:
            return None
        if "ARS_WardType" not in self.filtered_df.columns:
            return None

        counts = self.filtered_df["ARS_WardType"].value_counts()
        total = len(self.filtered_df)

        ward_df = pd.DataFrame({
            "Ward Type": counts.index,
            "Count": counts.values,
            "Percentage": (counts.values / total * 100).round(1),
        }).sort_values("Count", ascending=False)

        self.metrics["ward"] = ward_df
        return ward_df

    def analyze_care_type(self) -> Optional[pd.DataFrame]:
        if self.filtered_df is None or self.filtered_df.empty:
            return None
        if "CareType" not in self.filtered_df.columns:
            return None

        counts = self.filtered_df["CareType"].value_counts()
        total = len(self.filtered_df)

        care_df = pd.DataFrame({
            "Care Type": counts.index,
            "Count": counts.values,
            "Percentage": (counts.values / total * 100).round(1),
        }).sort_values("Count", ascending=False)

        self.metrics["care"] = care_df
        return care_df

    def analyze_demographics(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        if self.filtered_df is None or self.filtered_df.empty:
            return results

        total = len(self.filtered_df)

        if "AgeRange" in self.filtered_df.columns:
            counts = self.filtered_df["AgeRange"].value_counts(dropna=False)
            age_df = pd.DataFrame({
                "Age Range": counts.index.astype(str),
                "Count": counts.values,
                "Percentage": (counts.values / total * 100).round(1),
            })
            age_df["__sort"] = age_df["Age Range"].apply(self._age_sort_key)
            age_df = age_df.sort_values("__sort", ascending=True).drop(columns="__sort")

            results["age"] = age_df
            self.metrics["age"] = age_df

        if "Sex" in self.filtered_df.columns:
            counts = self.filtered_df["Sex"].value_counts(dropna=False)
            sex_df = pd.DataFrame({
                "Sex": counts.index.astype(str),
                "Count": counts.values,
                "Percentage": (counts.values / total * 100).round(1),
            }).sort_values("Count", ascending=False)

            results["sex"] = sex_df
            self.metrics["sex"] = sex_df

        return results

    def analyze_testing_frequency(self, top_n: Optional[int] = None) -> Optional[pd.DataFrame]:
        if self.filtered_df is None or self.filtered_df.empty or not self.antibiotic_columns:
            return None

        total = len(self.filtered_df)
        testing_stats = []
        for code, info in sorted(self.antibiotic_columns.items()):
            tested_col = info["tested"]
            n_tested = pd.to_numeric(self.filtered_df[tested_col], errors="coerce").sum()
            pct_tested = (n_tested / total * 100) if total else 0
            testing_stats.append({
                "Code": code,
                "Name": info["name"],
                "Count Tested": int(n_tested) if not pd.isna(n_tested) else 0,
                "Percentage Tested": round(pct_tested, 1),
            })

        testing_df = pd.DataFrame(testing_stats)

        # Sort highest -> lowest
        testing_df = testing_df.sort_values(
            ["Percentage Tested", "Count Tested", "Code"],
            ascending=[False, False, True],
        )

        if top_n is not None:
            testing_df = testing_df.head(int(top_n))

        self.metrics["testing"] = testing_df
        return testing_df


    def run_analysis(self) -> None:
        print("\n" + "=" * 80)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("=" * 80 + "\n")

        self.load_and_filter()
        if self.filtered_df is None or self.filtered_df.empty:
            return

        print("Analyzing dimensions...")
        self.analyze_temporal()
        self.analyze_specimen()
        self.analyze_ward_type()
        self.analyze_care_type()
        self.analyze_demographics()
        self.analyze_testing_frequency(top_n=40)
        self.analyze_resistance_top(top_n=40, sort_by="Tested (n)")
        self.analyze_yearly_ris_top(top_n=20, sort_by="R (%)")

        print("✓ Analysis complete!\n")

    def print_summary_report(self) -> None:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80 + "\n")

        print(f"Analytical Cohort Size: {self.metrics.get('final_count', 0):,} isolates")
        print(f"Overall Retention: {self.metrics.get('retention_pct', 0):.1f}%\n")

        for key, title in [
            ("temporal", "Temporal Distribution"),
            ("specimen", "Specimen Sources"),
            ("ward", "Ward Types"),
            ("care", "Care Types"),
            ("age", "Age Distribution (sorted by age)"),
            ("sex", "Sex Distribution"),
            ("testing", "Testing Frequency"),
        ]:
            if key in self.metrics and isinstance(self.metrics[key], pd.DataFrame) and not self.metrics[key].empty:
                print(title + ":")
                print(self.metrics[key].to_string(index=False))
                print()

    # =========================================================================
    # CSV table saving
    # =========================================================================
    def _save_table(self, df: pd.DataFrame, filename_base: str) -> None:
        """Save a DataFrame as CSV alongside figures."""
        if self.output_dir is None:
            raise ValueError("output_dir is not set. Call generate_plots(output_dir=...) first.")
        if df is None or df.empty:
            return

        outpath = self.output_dir / f"{filename_base}.csv"
        try:
            df.to_csv(outpath, index=False, encoding="utf-8")
            print(f"  ✓ Saved {outpath.name}")
        except Exception as e:
            print(f"  ⚠ Failed to save {outpath.name}: {e}")

    # =========================================================================
    # Plot export
    # =========================================================================
    def _save_figure(self, fig, filename_base: str, formats: Optional[List[str]] = None) -> None:
        if self.output_dir is None:
            raise ValueError("output_dir is not set. Call generate_plots(output_dir=...) first.")

        formats = formats or self.export_formats
        width = self.fig_width
        height = self.fig_height

        for fmt in formats:
            try:
                fmt_l = fmt.lower()
                outpath = self.output_dir / f"{filename_base}.{fmt_l}"
                if fmt_l == "html":
                    fig.write_html(outpath)
                else:
                    fig.write_image(outpath, width=width, height=height)  # kaleido required
                print(f"  ✓ Saved {outpath.name}")
            except Exception as e:
                print(f"  ⚠ Failed to save {filename_base}.{fmt}: {e}")

    def generate_plots(
        self,
        output_dir: str = "./figures",
        formats: Optional[List[str]] = None,
        save_tables: Optional[bool] = None,
    ) -> None:
        if self.filtered_df is None or self.filtered_df.empty:
            print("⚠ No plots generated: filtered cohort is empty.")
            return

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_formats = formats if formats else ["html", "png", "svg"]

        if save_tables is None:
            save_tables = self.save_tables_default

        print("\nGenerating publication-ready visualizations...\n")
        print(f"Palette mode: {self.palette_name}")
        print(f"Export formats: {', '.join(self.export_formats).upper()}")
        print(f"CSV tables: {'ON' if save_tables else 'OFF'}\n")

        self._plot_temporal(save_tables=save_tables)
        self._plot_specimen(save_tables=save_tables)
        self._plot_ward_type(save_tables=save_tables)
        self._plot_care_type(save_tables=save_tables)
        self._plot_demographics(save_tables=save_tables)
        self._plot_testing_frequency(top_n=40, save_tables=save_tables)
        self._plot_resistance_top(top_n=40, metric="%R", save_tables=save_tables)
        self._plot_yearly_ris_grid(top_n=20, sort_by="R (%)", save_tables=save_tables)


        print(f"\n✓ All outputs saved to: {self.output_dir}\n")

    # =========================================================================
    # Plot styling helper
    # =========================================================================
    def _apply_pub_layout(self, fig: go.Figure, title_html: str, margin: dict) -> None:
        fig.update_layout(
            title=dict(text=title_html, x=0.5, xanchor="center", font=self.title_font),
            template=self.template,
            font=self.font_settings,
            paper_bgcolor=self.paper_bg,
            plot_bgcolor=self.plot_bg,
            margin=margin,
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

    # =========================================================================
    # Plotting (now also writes CSV tables)
    # =========================================================================
    def _plot_temporal(self, save_tables: bool = True) -> None:
        df = self.metrics.get("temporal")
        if df is None or df.empty:
            return

        if save_tables:
            self._save_table(df, "01_temporal_distribution")

        colors = self._colors_for_plot(len(df), kind="bar")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Year"],
            y=df["Count"],
            text=df["Percentage"].astype(str) + "%",
            textposition="outside",
            textfont=dict(size=self.bar_text_size),
            marker=dict(color=colors, line=dict(color="#111111", width=1.5)),
            hovertemplate="<b>Year: %{x}</b><br>Count: %{y:,}<br>Percentage: %{text}<extra></extra>",
        ))

        self._apply_pub_layout(
            fig,
            "<b>Temporal Distribution of <i>E. coli</i> Isolates (2019–2023)</b>",
            margin=dict(l=120, r=60, t=140, b=120),
        )

        fig.update_xaxes(
            title_text="Year", title_font=self.label_font, 
            tickfont=self.tick_font,
            tickmode="linear",
            dtick=1,
            tickformat="d",
            tick0=int(df["Year"].min()),
        )
        fig.update_yaxes(title_text="Number of Isolates", title_font=self.label_font, tickfont=self.tick_font)

        self._save_figure(fig, "01_temporal_distribution")

    def _plot_specimen(self, save_tables: bool = True) -> None:
        df = self.metrics.get("specimen")
        if df is None or df.empty:
            return

        if save_tables:
            self._save_table(df, "02_specimen_distribution")

        colors = self._colors_for_plot(len(df), kind="bar")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df["Specimen Source"],
            x=df["Count"],
            orientation="h",
            text=df["Percentage"].astype(str) + "%",
            textposition="outside",
            textfont=dict(size=self.bar_text_size),
            marker=dict(color=colors, line=dict(color="#111111", width=1.5)),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        ))

        fig.update_yaxes(categoryorder="array", categoryarray=list(df["Specimen Source"]))
        fig.update_yaxes(autorange="reversed")

        self._apply_pub_layout(
            fig,
            "<b>Specimen Sources</b>",
            margin=dict(l=340, r=80, t=140, b=120),
        )

        fig.update_xaxes(title_text="Number of Isolates", title_font=self.label_font, tickfont=self.tick_font)
        fig.update_yaxes(title_text="Specimen Type", title_font=self.label_font, tickfont=self.tick_font)

        self._save_figure(fig, "02_specimen_distribution")

    def _plot_ward_type(self, save_tables: bool = True) -> None:
        df = self.metrics.get("ward")
        if df is None or df.empty:
            return

        if save_tables:
            self._save_table(df, "03_ward_distribution")

        colors = self._colors_for_plot(len(df), kind="pie")

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=df["Ward Type"],
            values=df["Count"],
            textposition="inside",
            textinfo="label+percent",
            textfont=dict(size=self.pie_text_size, family="Arial", color="white"),
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="<b>Ward Type Distribution</b>", x=0.5, xanchor="center", font=self.title_font),
            template=self.template,
            font=self.font_settings,
            paper_bgcolor=self.paper_bg,
            height=self.fig_height,
            width=self.fig_width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=self.legend_font,
            ),
            margin=dict(l=60, r=60, t=140, b=220),
        )

        self._save_figure(fig, "03_ward_distribution")

    def _plot_care_type(self, save_tables: bool = True) -> None:
        df = self.metrics.get("care")
        if df is None or df.empty:
            return

        if save_tables:
            self._save_table(df, "04_care_distribution")

        colors = self._colors_for_plot(len(df), kind="pie")

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=df["Care Type"],
            values=df["Count"],
            textposition="inside",
            textinfo="label+percent",
            textfont=dict(size=self.pie_text_size, family="Arial", color="white"),
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="<b>Care Type Distribution</b>", x=0.5, xanchor="center", font=self.title_font),
            template=self.template,
            font=self.font_settings,
            paper_bgcolor=self.paper_bg,
            height=self.fig_height,
            width=self.fig_width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=self.legend_font,
            ),
            margin=dict(l=60, r=60, t=140, b=220),
        )

        self._save_figure(fig, "04_care_distribution")

    def _plot_demographics(self, save_tables: bool = True) -> None:
        age_df = self.metrics.get("age")
        sex_df = self.metrics.get("sex")
        if age_df is None or age_df.empty or sex_df is None or sex_df.empty:
            return

        if save_tables:
            self._save_table(age_df, "05a_age_distribution")
            self._save_table(sex_df, "05b_sex_distribution")

        age_colors = self._colors_for_plot(len(age_df), kind="bar")
        sex_colors = self._colors_for_plot(len(sex_df), kind="pie")

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.18
        )

        fig.add_trace(
            go.Bar(
                y=age_df["Age Range"],
                x=age_df["Count"],
                orientation="h",
                text=age_df["Percentage"].astype(str) + "%",
                textposition="outside",
                textfont=dict(size=self.bar_text_size),
                marker=dict(color=age_colors, line=dict(color="#111111", width=1.5)),
                hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1
        )

        fig.update_yaxes(categoryorder="array", categoryarray=list(age_df["Age Range"]), row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=1)

        fig.add_trace(
            go.Pie(
                labels=sex_df["Sex"],
                values=sex_df["Count"],
                textposition="inside",
                textinfo="label+percent",
                textfont=dict(size=self.pie_text_size, family="Arial", color="white"),
                marker=dict(colors=sex_colors, line=dict(color="white", width=2)),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Count", title_font=self.label_font, tickfont=self.tick_font, row=1, col=1)
        fig.update_yaxes(title_text="Age Range", title_font=self.label_font, tickfont=self.tick_font, row=1, col=1)

        fig.update_layout(
            title=dict(text="<b>Patient Demographics</b>", x=0.5, xanchor="center", font=self.title_font),
            template=self.template,
            font=self.font_settings,
            paper_bgcolor=self.paper_bg,
            plot_bgcolor=self.plot_bg,
            height=self.fig_height,
            width=self.fig_width,
            showlegend=False,
            margin=dict(l=320, r=60, t=140, b=120),
        )

        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False, row=1, col=1)

        self._save_figure(fig, "05_demographics")

    
    def _dynamic_height(self, n_bars: int, min_h=900, px_per_bar=44, max_h=3200):
        """Compute figure height so all y-axis labels + annotations are visible."""
        return int(min(max(min_h, n_bars * px_per_bar), max_h))




    def _plot_testing_frequency(self, top_n: int = 20, save_tables: bool = True) -> None:
        df = self.metrics.get("testing")
        if df is None or df.empty:
            return

        df = df.head(int(top_n)).copy()

        # Make sure it's sorted high -> low for plotting too
        df = df.sort_values(
            ["Percentage Tested", "Count Tested", "Code"],
            ascending=[False, False, True],
        )

        if save_tables:
            self._save_table(df, f"06_testing_frequency_top{top_n}")

        # Short labels to prevent chart shifting
        df["Label"] = df["Code"]

        colors = self._colors_for_plot(len(df), kind="bar")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df["Label"],
            x=df["Percentage Tested"],
            orientation="h",
            text=df["Percentage Tested"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=self.bar_text_size),
            cliponaxis=False,  # <-- IMPORTANT: prevent outside text clipping
            marker=dict(color=colors, line=dict(color="#111111", width=1.5)),
            customdata=np.stack([df["Name"], df["Count Tested"]], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Antibiotic: %{customdata[0]}<br>"
                "Tested (n): %{customdata[1]:,}<br>"
                "Tested: %{x:.1f}%<extra></extra>"
            ),
        ))

        # Force all y tick labels to show (prevents Plotly thinning at high N)
        fig.update_yaxes(
            autorange="reversed",
            automargin=True,
            tickmode="array",
            tickvals=df["Label"].tolist(),
            ticktext=df["Label"].tolist(),
        )
        fig.update_xaxes(automargin=True)

        self._apply_pub_layout(
            fig,
            f"<b>Top {top_n} Antibiotics by Testing Frequency</b>",
            margin=dict(l=260, r=140, t=140, b=120),  # <-- r increased for outside text
        )

        fig.update_layout(height=self._dynamic_height(len(df)))

        fig.update_xaxes(
            title_text="Percentage Tested (%)",
            title_font=self.label_font,
            tickfont=self.tick_font,
        )
        fig.update_yaxes(
            title_text="Antibiotic",
            title_font=self.label_font,
            tickfont=self.tick_font,
        )

        self._save_figure(fig, f"06_testing_frequency_top{top_n}")



    
    def analyze_resistance_top(
        self,
        top_n: int = 40,
        sort_by: str = "Tested (n)",  # or "%R" or "%Non-susceptible (R+I)"
    ) -> Optional[pd.DataFrame]:
        if self.filtered_df is None or self.filtered_df.empty or not self.antibiotic_columns:
            return None

        rows = []
        for code, info in sorted(self.antibiotic_columns.items()):
            tested_col = info["tested"]
            outcome_col = info["outcome"]

            tested = pd.to_numeric(self.filtered_df[tested_col], errors="coerce").fillna(0) > 0
            n_tested = int(tested.sum())
            if n_tested == 0:
                continue

            outcomes = (
                self.filtered_df.loc[tested, outcome_col]
                .astype(str).str.strip().str.upper()
                .replace({"NAN": "", "NONE": "", "NULL": ""})
            )

            n_R = int((outcomes == "R").sum())
            n_I = int((outcomes == "I").sum())
            n_S = int((outcomes == "S").sum())

            rows.append({
                "Code": code,
                "Name": info["name"],
                "Tested (n)": n_tested,
                "R (n)": n_R,
                "I (n)": n_I,
                "S (n)": n_S,
                "%R": round(100 * n_R / n_tested, 1),
                "%Non-susceptible (R+I)": round(100 * (n_R + n_I) / n_tested, 1),
            })

        if not rows:
            return None

        res_df = pd.DataFrame(rows)

        # Validate sort_by
        allowed = {"Tested (n)", "%R", "%Non-susceptible (R+I)"}
        if sort_by not in allowed:
            raise ValueError(f"sort_by must be one of {sorted(allowed)}")

        # Sort highest -> lowest, with stable tie-breakers
        res_df = res_df.sort_values(
            [sort_by, "Tested (n)", "Code"],
            ascending=[False, False, True],
        ).head(int(top_n))

        self.metrics["resistance_top"] = res_df
        self.metrics["resistance_top_sort_by"] = sort_by
        self.metrics["resistance_top_n"] = int(top_n)
        return res_df

    
    
    def _plot_resistance_top(
        self,
        top_n: int = 40,
        metric: str = "%R",  # or "%Non-susceptible (R+I)"
        save_tables: bool = True,
    ) -> None:
        df = self.metrics.get("resistance_top")
        if df is None or df.empty:
            return

        if metric not in df.columns:
            raise ValueError(f"metric must be one of {list(df.columns)}")

        df = (
            df.sort_values([metric, "Tested (n)", "Code"], ascending=[False, False, True])
            .head(int(top_n))
            .copy()
        )

        if save_tables:
            self._save_table(
                df,
                f"07_resistance_top{top_n}_{metric.replace('%','pct').replace(' ','_').replace('+','plus')}"
            )

        # Keep axis label short to avoid huge left margin / squeezed plot
        df["Label"] = df["Code"]

        colors = self._colors_for_plot(len(df), kind="bar")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df["Label"],
            x=df[metric],
            orientation="h",
            text=df[metric].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=self.bar_text_size),
            cliponaxis=False,  # <-- IMPORTANT: prevents outside text from being clipped
            marker=dict(color=colors, line=dict(color="#111111", width=1.5)),
            customdata=np.stack(
                [df["Name"], df["Tested (n)"], df["R (n)"], df["I (n)"], df["S (n)"]],
                axis=1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Antibiotic: %{customdata[0]}<br>"
                "Tested: %{customdata[1]:,}<br>"
                "R/I/S: %{customdata[2]:,} / %{customdata[3]:,} / %{customdata[4]:,}<br>"
                f"{metric}: " + "%{x:.1f}%<extra></extra>"
            ),
        ))

        # Force all labels to be shown (prevents Plotly dropping ticks at high N)
        fig.update_yaxes(
            autorange="reversed",
            automargin=True,
            tickmode="array",
            tickvals=df["Label"].tolist(),
            ticktext=df["Label"].tolist(),
        )
        fig.update_xaxes(automargin=True)

        title_metric = "Resistant (%R)" if metric == "%R" else "Non-susceptible (R+I)"

        # Give a bit more right margin for outside text; dynamic height for 40 bars
        self._apply_pub_layout(
            fig,
            f"<b><i>E. coli</i> {title_metric} — Top {top_n} Antibiotics</b>",
            margin=dict(l=260, r=140, t=140, b=120),  # <-- r increased
        )

        fig.update_layout(height=self._dynamic_height(len(df)))

        fig.update_xaxes(
            title_text=f"{title_metric} among tested (%)",
            title_font=self.label_font,
            tickfont=self.tick_font
        )
        fig.update_yaxes(
            title_text="Antibiotic",
            title_font=self.label_font,
            tickfont=self.tick_font
        )

        safe_metric = (
            metric.replace("%", "pct")
                .replace(" ", "_")
                .replace("+", "plus")
                .replace("(", "")
                .replace(")", "")
        )
        self._save_figure(fig, f"07_ecoli_{safe_metric}_top{top_n}")



    
    def analyze_yearly_ris_top(
        self,
        top_n: int = 10,
        years: Optional[List[int]] = None,
        selection: str = "global_top",   # "global_top" | "top_per_year" | "custom"
        sort_by: str = "Tested (n)",     # "Tested (n)" | "R (%)" | "Non-susceptible (R+I) (%)"
        custom_codes: Optional[List[str]] = None,
        min_tested: int = 1,
    ) -> Optional[pd.DataFrame]:
        """
        Build a long table with counts and percentages of R/I/S by Year and Antibiotic,
        then select antibiotics to keep using a clear selection strategy.

        selection:
        - "global_top": pick ONE fixed set of top_n antibiotics across all years (best for year comparisons)
        - "top_per_year": pick top_n antibiotics independently per year (panels may differ; exploratory)
        - "custom": use custom_codes as the fixed set

        sort_by:
        - "Tested (n)" (recommended default; stable)
        - "R (%)"
        - "Non-susceptible (R+I) (%)"
        """
        if self.filtered_df is None or self.filtered_df.empty or not self.antibiotic_columns:
            return None
        if "Year" not in self.filtered_df.columns:
            return None

        df = self.filtered_df.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Year"])

        if years is not None:
            df = df[df["Year"].isin([int(y) for y in years])]

        if df.empty:
            return None

        rows = []
        for year_val in sorted(df["Year"].unique().tolist()):
            dyy = df[df["Year"] == year_val]
            for code, info in sorted(self.antibiotic_columns.items()):
                tested_col = info["tested"]
                outcome_col = info["outcome"]

                tested_mask = pd.to_numeric(dyy[tested_col], errors="coerce").fillna(0) > 0
                n_tested = int(tested_mask.sum())
                if n_tested < int(min_tested):
                    continue

                outcomes = (
                    dyy.loc[tested_mask, outcome_col]
                    .astype(str).str.strip().str.upper()
                    .replace({"NAN": "", "NONE": "", "NULL": ""})
                )

                n_R = int((outcomes == "R").sum())
                n_I = int((outcomes == "I").sum())
                n_S = int((outcomes == "S").sum())

                rows.append({
                    "Year": int(year_val),
                    "Code": code,
                    "Name": info["name"],
                    "Tested (n)": n_tested,
                    "R (n)": n_R,
                    "I (n)": n_I,
                    "S (n)": n_S,
                    "R (%)": round(100 * n_R / n_tested, 1),
                    "I (%)": round(100 * n_I / n_tested, 1),
                    "S (%)": round(100 * n_S / n_tested, 1),
                    "Non-susceptible (R+I) (%)": round(100 * (n_R + n_I) / n_tested, 1),
                })

        if not rows:
            return None

        out = pd.DataFrame(rows)

        allowed_sort = {"Tested (n)", "R (%)", "Non-susceptible (R+I) (%)"}
        if sort_by not in allowed_sort:
            raise ValueError(f"sort_by must be one of {sorted(allowed_sort)}")

        # -------------------------------------------
        # Selection strategy: decide which antibiotics to keep
        # -------------------------------------------
        if selection == "custom":
            if not custom_codes:
                raise ValueError("custom_codes must be provided when selection='custom'")
            keep_codes = list(dict.fromkeys([str(c).strip().upper() for c in custom_codes]))  # preserve order, unique
            out = out[out["Code"].isin(keep_codes)].copy()

            # enforce custom order
            order_map = {code: i for i, code in enumerate(keep_codes)}
            out["__order"] = out["Code"].map(order_map)
            out = out.sort_values(["Year", "__order"]).drop(columns="__order")

        elif selection == "global_top":
            # Weighted ranking across all years (reviewer-friendly)
            agg = (
                out.groupby(["Code", "Name"], as_index=False)
                .agg({
                    "Tested (n)": "sum",
                    "R (n)": "sum",
                    "I (n)": "sum",
                    "S (n)": "sum",
                })
            )
            agg["R (%)"] = (100 * agg["R (n)"] / agg["Tested (n)"]).round(1)
            agg["Non-susceptible (R+I) (%)"] = (100 * (agg["R (n)"] + agg["I (n)"]) / agg["Tested (n)"]).round(1)

            agg = agg.sort_values([sort_by, "Tested (n)", "Code"], ascending=[False, False, True]).head(int(top_n))
            keep_codes = agg["Code"].tolist()

            out = out[out["Code"].isin(keep_codes)].copy()

            # enforce global order across all years
            order_map = {code: i for i, code in enumerate(keep_codes)}
            out["__order"] = out["Code"].map(order_map)
            out = out.sort_values(["Year", "__order"]).drop(columns="__order")


        elif selection == "top_per_year":
            # select top_n per year (may differ by year)
            out = (
                out.sort_values(["Year", sort_by, "Tested (n)", "Code"], ascending=[True, False, False, True])
                .groupby("Year", as_index=False, group_keys=False)
                .head(int(top_n))
            )
        else:
            raise ValueError("selection must be one of: 'global_top', 'top_per_year', 'custom'")

        self.metrics["yearly_ris_top"] = out
        self.metrics["yearly_ris_top_n"] = int(top_n)
        self.metrics["yearly_ris_selection"] = selection
        self.metrics["yearly_ris_sort_by"] = sort_by
        self.metrics["yearly_ris_years"] = years
        return out


    def _plot_yearly_ris_grid(
        self,
        top_n: int = 10,
        selection: str = "global_top",
        sort_by: str = "Tested (n)",
        metric_stack: str = "RIS",  # "RIS" | "NS_vs_S"
        years: Optional[List[int]] = None,
        custom_codes: Optional[List[str]] = None,
        save_tables: bool = True,
        label_mode: str = "pct_counts",   # "counts" | "pct_counts" | "none"
        label_min_pct: float = 6.0,       # only label segments >= this % to avoid clutter
    ) -> None:
        df = self.metrics.get("yearly_ris_top")

        need_rebuild = (
            df is None or df.empty or
            self.metrics.get("yearly_ris_top_n") != int(top_n) or
            self.metrics.get("yearly_ris_selection") != selection or
            self.metrics.get("yearly_ris_sort_by") != sort_by or
            (years is not None and self.metrics.get("yearly_ris_years") != years)
        )
        if need_rebuild:
            df = self.analyze_yearly_ris_top(
                top_n=top_n,
                years=years,
                selection=selection,
                sort_by=sort_by,
                custom_codes=custom_codes,
            )

        if df is None or df.empty:
            return

        df = df.copy()

        if save_tables:
            safe_sel = selection.replace(" ", "_")
            safe_sort = sort_by.replace("%", "pct").replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            self._save_table(df, f"08_yearly_RIS_{safe_sel}_top{top_n}_sort_{safe_sort}")

        years_list = sorted(df["Year"].unique().tolist())
        if not years_list:
            return

        n_years = len(years_list)
        ncols = 2 if n_years > 1 else 1
        nrows = int(np.ceil(n_years / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[str(y) for y in years_list],
            horizontal_spacing=0.10,
            vertical_spacing=0.10,
        )

        if metric_stack == "RIS":
            stack_cols = ["S (%)", "I (%)", "R (%)"]
            count_cols = {"S (%)": "S (n)", "I (%)": "I (n)", "R (%)": "R (n)"}
            legend_labels = {
                "S (%)": "Susceptible (S)",
                "I (%)": "Intermediate (I)",
                "R (%)": "Resistant (R)",
            }
            colors = {"S (%)": "#009E73", "I (%)": "#E69F00", "R (%)": "#D55E00"}
            title_suffix = "S/I/R (%) Among Tested"
        elif metric_stack == "NS_vs_S":
            stack_cols = ["S (%)", "Non-susceptible (R+I) (%)"]
            # build NS counts from R+I
            legend_labels = {
                "S (%)": "Susceptible (S)",
                "Non-susceptible (R+I) (%)": "Non-susceptible (R+I)",
            }
            colors = {"S (%)": "#009E73", "Non-susceptible (R+I) (%)": "#D55E00"}
            title_suffix = "Susceptible vs Non-susceptible (%) Among Tested"
        else:
            raise ValueError("metric_stack must be 'RIS' or 'NS_vs_S'")

        # Consistent order across years if selection is global_top/custom
        if selection in {"global_top", "custom"}:
            first_year = years_list[0]
            order_codes = df[df["Year"] == first_year]["Code"].tolist()
            order_map = {code: i for i, code in enumerate(order_codes)}
        else:
            order_map = None

        def _make_text(pct, n, denom):
            # pct is numeric, n & denom ints
            if label_mode == "none":
                return ""
            if pct < float(label_min_pct):
                return ""
            if label_mode == "counts":
                return f"{n}/{denom}"
            # pct_counts
            return f"{pct:.1f}%\n({n}/{denom})"

        for idx, y in enumerate(years_list):
            rr = idx // ncols + 1
            cc = idx % ncols + 1

            dyy = df[df["Year"] == y].copy()

            # Create labels FIRST (fixes KeyError)
            dyy["Label"] = dyy["Code"]

            # Apply ordering BEFORE annotation so annotations align with displayed order
            if order_map is not None:
                dyy["__order"] = dyy["Code"].map(order_map).fillna(10**9)
                dyy = dyy.sort_values("__order").drop(columns="__order")
            else:
                dyy = dyy.sort_values(["Tested (n)", "Code"], ascending=[False, True])

            # Add denominator annotations (n=...) AFTER ordering
            # Use subplot row/col refs (more robust than idx+1)
            x_ref = "x" if (rr == 1 and cc == 1) else f"x{idx+1}"
            y_ref = "y" if (rr == 1 and cc == 1) else f"y{idx+1}"

            for lab, denom in zip(dyy["Label"].tolist(), dyy["Tested (n)"].astype(int).tolist()):
                fig.add_annotation(
                    x=102, y=lab,
                    xref=x_ref, yref=y_ref,
                    text=f"n={denom:,}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=self.tick_font["size"], color="#111111"),
                    align="left",
                )


            if order_map is not None:
                dyy["__order"] = dyy["Code"].map(order_map).fillna(10**9)
                dyy = dyy.sort_values("__order").drop(columns="__order")
            else:
                dyy = dyy.sort_values(["Tested (n)", "Code"], ascending=[False, True])

            # For NS_vs_S we need NS counts per row
            if metric_stack == "NS_vs_S":
                dyy["Non-susceptible (R+I) (n)"] = dyy["R (n)"] + dyy["I (n)"]

            for col in stack_cols:
                # Determine numerator counts for this segment
                if metric_stack == "RIS":
                    num = dyy[count_cols[col]].astype(int).tolist()
                else:
                    # NS_vs_S
                    if col == "S (%)":
                        num = dyy["S (n)"].astype(int).tolist()
                    else:
                        num = dyy["Non-susceptible (R+I) (n)"].astype(int).tolist()

                denom = dyy["Tested (n)"].astype(int).tolist()
                pct_vals = dyy[col].astype(float).tolist()

                # Build in-bar text labels (with denom)
                text_labels = [_make_text(p, n, d) for p, n, d in zip(pct_vals, num, denom)]

                fig.add_trace(
                    go.Bar(
                        y=dyy["Label"],
                        x=dyy[col],
                        orientation="h",
                        name=legend_labels[col],
                        marker=dict(color=colors[col], line=dict(color="#111111", width=0.5)),
                        cliponaxis=False,
                        text=text_labels,
                        textposition="inside",
                        insidetextanchor="middle",
                        textfont=dict(size=max(10, self.bar_text_size - 6), color="white"),
                        hovertemplate=(
                            "<b>Year:</b> %{customdata[0]}<br>"
                            "<b>Drug:</b> %{y} (%{customdata[1]})<br>"
                            "<b>Tested (denom):</b> %{customdata[2]:,}<br>"
                            "<b>S/I/R (n):</b> %{customdata[3]:,} / %{customdata[4]:,} / %{customdata[5]:,}<br>"
                            f"<b>{legend_labels[col]}:</b> " + "%{x:.1f}%<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                np.repeat(y, len(dyy)),
                                dyy["Name"],
                                dyy["Tested (n)"],
                                dyy["S (n)"],
                                dyy["I (n)"],
                                dyy["R (n)"],
                            ],
                            axis=1,
                        ),
                        showlegend=(idx == 0),
                    ),
                    row=rr, col=cc
                )

            fig.update_yaxes(
                autorange="reversed",
                tickmode="array",
                tickvals=dyy["Label"].tolist(),
                ticktext=dyy["Label"].tolist(),
                automargin=True,
                row=rr, col=cc
            )
            # fig.update_xaxes(range=[0, 100], ticksuffix="%", automargin=True, row=rr, col=cc)
            fig.update_xaxes(range=[0, 115], ticksuffix="%", automargin=True, row=rr, col=cc)

        fig.update_layout(
            barmode="stack",
            template=self.template,
            font=self.font_settings,
            paper_bgcolor=self.paper_bg,
            plot_bgcolor=self.plot_bg,
            title=dict(
                text=f"<b><i>E. coli</i> — {title_suffix}, by Year ({selection}, top {top_n})</b>",
                x=0.5, xanchor="center", font=self.title_font
            ),
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.08, yanchor="top",
                font=self.legend_font
            ),
            # margin=dict(l=140, r=60, t=120, b=160),
            margin=dict(l=160, r=260, t=140, b=180),
            height=max(1000, self._dynamic_height(top_n) * nrows),
            width=max(self.fig_width, 1800),
        )

        self._save_figure(fig, f"08_yearly_RIS_grid_{selection}_top{top_n}")







# ============================================================================
# MAIN (Example)
# ============================================================================
if __name__ == "__main__":
    config = FilterConfig.from_dict({
        "name": "E. coli Inpatient Analysis",
        "description": "E. coli, first isolates, inpatient only, 2020-2023",
        "filters": [
            {"column": "Pathogen", "operator": "equals", "value": "Escherichia coli"},
            {"column": "CSQ", "operator": "equals", "value": "Erstisolat"},
            {"column": "ARS_WardType", "operator": "in", "values": ["Normal Ward", "Intensive Care Unit"]},
            {"column": "CareType", "operator": "in", "values": ["In-Patient", "Out-Patient"]},
            {"column": "Year", "operator": "range", "min": 2020, "max": 2023},
            {"column": "TotalAntibioticsTested", "operator": "gte", "value": 3},
            {"column": "IsSpecificlyExcluded_Screening", "operator": "is_false"},
            {"column": "IsSpecificlyExcluded_Pathogen", "operator": "is_false"},
        ],
        "verbose": True,
    })

    explorer = DataExplorer(
        data_path="./data.parquet",
        filter_config=config,
        required_covariates=["CareType", "ARS_WardType", "Sex", "AgeRange", "AgeGroup"],
        drop_missing_covariates=True,
    )

    explorer.run_analysis()
    explorer.print_summary_report()
    explorer.generate_plots(
        output_dir="./publication_figures",
        formats=["html", "png", "svg"],
        save_tables=True,  # <- NEW
    )
