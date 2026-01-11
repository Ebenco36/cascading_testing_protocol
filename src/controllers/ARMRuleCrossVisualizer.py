# ============================
# Publication-ready Cross-ARM Visualization
# Sankey, Heatmap (Plotly) + Chord (pyCirclize) + Circos radar + Network (bipartite R->T)
# Exports: HTML + PNG + SVG + PDF
#
# Layout fix for chord:
# - Chord axes explicitly positioned in upper region
# - Key axes is a separate bottom panel with opaque white background
# - Key rendered as two-column text, never overlapping the chord
# ============================

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================
# Rule splitting by year
# ============================

def split_rules_by_year(
    df: pd.DataFrame,
    antecedent_column: str = "Antecedents",
    save_csv: bool = False,
    output_prefix: str = "rules",
    verbose: bool = True,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame, List[int]]:
    """
    Split association rules DataFrame into separate DataFrames by year.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with association rules.
    antecedent_column : str, default="Antecedents"
        Column name containing year information (e.g., "CIP_R|Year=2022").
    save_csv : bool, default=False
        Whether to save each year's DataFrame to CSV.
    output_prefix : str, default="rules"
        Prefix for output CSV filenames (e.g., "rules_2022.csv").
    verbose : bool, default=True
        Whether to print summary statistics.

    Returns
    -------
    dfs_by_year : dict
        Dictionary with year (int) as key and DataFrame as value.
    df_pooled : pd.DataFrame
        DataFrame containing rules without year information.
    available_years : list
        Sorted list of years found in the data.
    """

    def extract_year(antecedents_str: Any) -> Optional[int]:
        """Extract year from antecedent string like 'CIP_R|MER_R|Year=2022'."""
        if pd.isna(antecedents_str):
            return None

        s = str(antecedents_str)

        # Pattern Year=YYYY
        m = re.search(r"Year=(\d{4})", s)
        if m:
            return int(m.group(1))

        # Alternative: YearYYYY
        m = re.search(r"Year(\d{4})", s)
        if m:
            return int(m.group(1))

        return None

    df_work = df.copy()
    df_work["Year"] = df_work[antecedent_column].apply(extract_year)

    available_years = sorted([int(y) for y in df_work["Year"].dropna().unique()])

    if verbose:
        print(f"Found {len(available_years)} years in dataset: {available_years}")
        print(f"Total rules: {len(df_work)}")
        print(f"Rules with year info: {df_work['Year'].notna().sum()}")
        print(f"Rules without year info (pooled): {df_work['Year'].isna().sum()}")

    dfs_by_year: Dict[int, pd.DataFrame] = {}
    for year in available_years:
        df_year = df_work[df_work["Year"] == year].copy()
        dfs_by_year[year] = df_year

        if verbose:
            print(f"\nYear {year}: {len(df_year)} rules")

        if save_csv:
            filename = f"{output_prefix}_{year}.csv"
            df_year.to_csv(filename, index=False)
            if verbose:
                print(f"  Saved: {filename}")

    df_pooled = df_work[df_work["Year"].isna()].copy()
    if verbose:
        print(f"\nPooled (no year): {len(df_pooled)} rules")

    if save_csv and not df_pooled.empty:
        filename = f"{output_prefix}_pooled.csv"
        df_pooled.to_csv(filename, index=False)
        if verbose:
            print(f"  Saved: {filename}")

    # Optional summary of cross-informative rules
    if verbose and "Is_cross_informative" in df_work.columns:
        print("\n" + "=" * 70)
        print("CROSS-INFORMATIVE RULES (R → T patterns) BY YEAR")
        print("=" * 70)

        for year in available_years:
            df_year = dfs_by_year[year]
            df_year_cross = df_year[df_year["Is_cross_informative"] == True]
            print(f"\n{year}: {len(df_year_cross)} cross-informative rules")

            if len(df_year_cross) > 0 and "Support" in df_year_cross.columns:
                top3 = df_year_cross.nlargest(3, "Support")
                for _, row in top3.iterrows():
                    res_item = row.get("ResistanceItem", "N/A")
                    test_item = row.get("TestItem", "N/A")
                    supp = float(row.get("Support", 0))
                    conf = float(row.get("Confidence", 0))
                    print(
                        f"  {res_item} → {test_item} "
                        f"(Supp={supp:.4f}, Conf={conf:.2f})"
                    )

    return dfs_by_year, df_pooled, available_years


# ============================
# Utilities
# ============================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except Exception:
        return default


def _split_pipe(s: Any) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [t.strip() for t in s.split("|") if t and t.strip()]


def _uniq_preserve(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _uniq_sorted(items: List[str]) -> List[str]:
    return sorted({x for x in items if x})


def _join_pipe(items: List[str]) -> str:
    return "|".join(_uniq_sorted(items))


def _drug_from_token(tok: str) -> str:
    return tok.split("_", 1)[0].strip().upper() if "_" in tok else tok.strip().upper()


def _suffix_from_token(tok: str) -> str:
    return tok.split("_", 1)[1].strip().upper() if "_" in tok else ""


def _is_antibiotic_event(tok: str) -> bool:
    if "_" not in tok:
        return False
    return _suffix_from_token(tok) in {"R", "T", "S", "I"}


def _token_to_code(token: Any) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    if not t or "=" in t:
        return None
    return _drug_from_token(t)


def _shorten_class_name(cls: str) -> str:
    if not cls or cls == "Unknown":
        return "Unknown"
    s = str(cls)
    s = s.replace("(β-lactam)", "").replace("β-lactam", "β-lac").strip()
    s = " ".join(s.split())
    if len(s) > 28:
        s = s[:25].rstrip() + "…"
    return s


# ============================
# Visualization config
# ============================

@dataclass
class CrossVizConfig:
    # Data
    weight_col: str = "Support"
    cross_flag_col: str = "Is_cross_informative"
    ant_r_col: str = "Antecedent_R_only"
    con_t_cross_col: str = "Consequent_T_cross_only"

    min_weight: float = 0.0
    top_edges: Optional[int] = 250

    # Plotly export quality
    export_scale: int = 3

    # pyCirclize chord settings
    chord_space_deg: int = 9
    chord_start_deg: float = -265.0
    chord_end_deg: float = 95.0
    chord_r_lim: Tuple[float, float] = (90, 100)

    chord_label_size: int = 10
    chord_title_size: int = 16
    chord_group_label_size: int = 14

    chord_link_ec: str = "black"
    chord_link_lw: float = 0.30
    chord_link_alpha: float = 0.78
    chord_direction: int = 1

    chord_max_links: Optional[int] = 140
    chord_figsize: Tuple[float, float] = (12.5, 12.5)
    chord_dpi: int = 500

    chord_key_top_n_sectors: Optional[int] = 18

    # Layout for bottom key panel
    chord_panel_bottom: float = -0.2
    chord_panel_height: float = 0.1
    chord_panel_gap: float = 0.0
    chord_key_font_size: int = 9

    # Circos radar settings
    radar_figsize: Tuple[float, float] = (10.5, 10.5)
    radar_dpi: int = 450


# ============================
# Cross visualizer
# ============================

class ARMRuleCrossVisualizer:
    def __init__(
        self,
        rules_df: pd.DataFrame,
        drugclass_json: Dict,
        aware_json: Optional[Dict] = None,
        cfg: Optional[CrossVizConfig] = None,
    ):
        self.rules_df = rules_df.copy()
        self.cfg = cfg or CrossVizConfig()

        self.class_map, self.code_to_tested_name = self._parse_group_json(drugclass_json)
        self.aware_map = self._parse_aware_json(aware_json) if aware_json else {}

        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None

    # -------- mappings --------
    @staticmethod
    def _parse_group_json(group_json: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
        class_map: Dict[str, str] = {}
        code_to_name: Dict[str, str] = {}
        for group_name, items in group_json.items():
            for item in items:
                s = str(item)
                code = s.split(" - ", 1)[0].strip().upper()
                if code:
                    class_map[code] = str(group_name)
                    code_to_name[code] = s
        return class_map, code_to_name

    @staticmethod
    def _parse_aware_json(aware_json: Dict) -> Dict[str, str]:
        def normalize_bucket(k: str) -> str:
            k = str(k).strip().lower()
            if "watch" in k:
                return "Watch"
            if "access" in k:
                return "Access"
            if "reserve" in k:
                return "Reserve"
            return "Not set"

        m: Dict[str, str] = {}
        if not aware_json:
            return m
        for bucket, items in aware_json.items():
            bucket_norm = normalize_bucket(bucket)
            for item in items:
                code = str(item).split(" - ", 1)[0].strip().upper()
                if code:
                    m[code] = bucket_norm
        return m

    def code_class(self, code: str) -> str:
        return self.class_map.get(code, "Unknown")

    def code_aware(self, code: str) -> str:
        return self.aware_map.get(code, "Not set")

    def code_label_rich(self, code: str) -> str:
        aware = self.code_aware(code)
        cls = self.code_class(code)
        return f"{code} | AWaRe: {aware} | Class: {cls}"

    # palettes
    @staticmethod
    def _aware_palette_hex() -> Dict[str, str]:
        return {
            "Access": "#2E7D32",
            "Watch": "#EF6C00",
            "Reserve": "#C62828",
            "Not set": "#616161",
        }

    def _palette_for_groups(self, groups: List[str], color_by: str) -> Dict[str, str]:
        groups = [str(g) for g in groups if str(g).strip()]
        uniq = _uniq_preserve(groups)

        if color_by == "aware":
            base = self._aware_palette_hex()
            uniq_set = set(uniq)
            pal = {k: v for k, v in base.items() if k in uniq_set}
            for g in uniq:
                if g not in pal:
                    pal[g] = "#777777"
            return pal

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        cmap = plt.get_cmap("tab20")
        return {g: mcolors.to_hex(cmap(i % 20)) for i, g in enumerate(uniq)}

    # -------- edges/nodes --------
    def build_cross_edges(self) -> pd.DataFrame:
        df = self.rules_df.copy()
        if self.cfg.cross_flag_col in df.columns:
            df = df[df[self.cfg.cross_flag_col] == True].copy()

        if df.empty:
            self.edges_df = pd.DataFrame(columns=["src", "dst", "weight", "rule_count"])
            self.nodes_df = pd.DataFrame(columns=["code", "aware", "class"])
            return self.edges_df

        wcol = self.cfg.weight_col if self.cfg.weight_col in df.columns else "Support"
        df["_w"] = df[wcol].apply(_safe_float)

        rows: List[Tuple[str, str, float]] = []
        for _, r in df.iterrows():
            r_items = _split_pipe(r.get(self.cfg.ant_r_col, ""))
            t_items = _split_pipe(r.get(self.cfg.con_t_cross_col, ""))

            r_codes = [c for c in (_token_to_code(x) for x in r_items) if c]
            t_codes = [c for c in (_token_to_code(x) for x in t_items) if c]
            if not r_codes or not t_codes:
                continue

            w = _safe_float(r.get("_w", 0.0), 0.0)
            for src in r_codes:
                for dst in t_codes:
                    if src != dst:
                        rows.append((src, dst, w))

        if not rows:
            self.edges_df = pd.DataFrame(columns=["src", "dst", "weight", "rule_count"])
            self.nodes_df = pd.DataFrame(columns=["code", "aware", "class"])
            return self.edges_df

        e = pd.DataFrame(rows, columns=["src", "dst", "weight"])
        e = e.groupby(["src", "dst"], as_index=False).agg(
            weight=("weight", "sum"),
            rule_count=("weight", "size"),
        )
        e = e[e["weight"] >= float(self.cfg.min_weight)].copy()
        e = e.sort_values(["weight", "rule_count"], ascending=[False, False])
        if self.cfg.top_edges is not None:
            e = e.head(int(self.cfg.top_edges)).copy()

        e["src_aware"] = e["src"].map(self.code_aware)
        e["dst_aware"] = e["dst"].map(self.code_aware)

        self.edges_df = e
        self.nodes_df = pd.DataFrame({"code": sorted(set(e["src"]).union(set(e["dst"])))})
        self.nodes_df["aware"] = self.nodes_df["code"].map(self.code_aware)
        self.nodes_df["class"] = self.nodes_df["code"].map(self.code_class)
        return e

    # ----------------------------
    # Plotly Sankey
    # ----------------------------
    def plot_sankey(self, title: str) -> go.Figure:
        if self.edges_df is None:
            self.build_cross_edges()
        e = self.edges_df.copy()
        if e.empty:
            return go.Figure().update_layout(title="No cross-informative edges to plot.")

        nodes = sorted(set(e["src"]).union(set(e["dst"])))
        idx = {n: i for i, n in enumerate(nodes)}

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=16,
                        thickness=20,
                        line=dict(width=0.8),
                        label=[self.code_label_rich(c) for c in nodes],
                    ),
                    link=dict(
                        source=e["src"].map(idx).tolist(),
                        target=e["dst"].map(idx).tolist(),
                        value=e["weight"].tolist(),
                        hovertemplate=(
                            "<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                            "Weight: %{value:.4f}<br>"
                            "<extra></extra>"
                        ),
                    ),
                )
            ]
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=900,
            width=1600,
            font=dict(size=16),
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    # ----------------------------
    # Plotly Heatmap
    # ----------------------------
    def plot_heatmap(self, title: str) -> go.Figure:
        if self.edges_df is None:
            self.build_cross_edges()
        e = self.edges_df.copy()
        if e.empty:
            return go.Figure().update_layout(title="No cross-informative edges to plot.")

        pivot = e.pivot_table(
            index="src", columns="dst", values="weight", aggfunc="sum", fill_value=0.0
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[self.code_label_rich(c) for c in pivot.columns.tolist()],
                y=[self.code_label_rich(c) for c in pivot.index.tolist()],
                hovertemplate="R=%{y}<br>T=%{x}<br>Weight=%{z:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=1200,
            width=1600,
            font=dict(size=14),
            margin=dict(l=60, r=60, t=80, b=80),
        )
        return fig

    def save_plotly_all(self, fig: go.Figure, out_dir: str, base_name: str) -> Dict[str, str]:
        _ensure_dir(out_dir)
        html = os.path.join(out_dir, f"{base_name}.html")
        png = os.path.join(out_dir, f"{base_name}.png")
        svg = os.path.join(out_dir, f"{base_name}.svg")
        pdf = os.path.join(out_dir, f"{base_name}.pdf")

        fig.write_html(html, include_plotlyjs="cdn")
        try:
            fig.write_image(png, scale=self.cfg.export_scale)
            fig.write_image(svg)
            fig.write_image(pdf)
        except Exception as ex:
            print(f"[WARN] Plotly static export failed. Install/upgrade kaleido. Error: {ex}")

        return {"html": html, "png": png, "svg": svg, "pdf": pdf}

    # ----------------------------
    # Chord diagram (pyCirclize) with fixed key layout
    # ----------------------------
    def chord_pycirclize_bipartite(
        self,
        out_dir: str,
        base_name: str,
        title: str,
        color_by: str = "aware",
        legend_loc: str = "upper right",
    ) -> Dict[str, str]:
        """
        Publication chord (pyCirclize) with:
        - bipartite R/T rings
        - colors by AWaRe or class
        - separate bottom key panel that does not overlap chord
        """
        if self.edges_df is None:
            self.build_cross_edges()
        e = self.edges_df.copy()
        if e.empty:
            raise ValueError("No cross-informative edges to plot.")
        if color_by not in {"aware", "class"}:
            raise ValueError("color_by must be 'aware' or 'class'.")

        _ensure_dir(out_dir)

        if self.cfg.chord_max_links is not None:
            e = (
                e.sort_values(["weight", "rule_count"], ascending=[False, False])
                .head(int(self.cfg.chord_max_links))
                .copy()
            )

        r_codes = sorted(set(e["src"].tolist()))
        t_codes = sorted(set(e["dst"].tolist()))
        r_labels = [f"R:{c}" for c in r_codes]
        t_labels = [f"T:{c}" for c in t_codes]

        mat = pd.DataFrame(0.0, index=r_labels, columns=t_labels)
        for _, row in e.iterrows():
            mat.loc[f"R:{row['src']}", f"T:{row['dst']}"] += float(row["weight"])

        mat = mat.loc[(mat.sum(axis=1) > 0), (mat.sum(axis=0) > 0)]
        r_labels = mat.index.tolist()
        t_labels = mat.columns.tolist()
        order = r_labels + t_labels

        def lbl_to_code(lbl: str) -> str:
            return lbl.split(":", 1)[1].strip().upper()

        sector_groups: Dict[str, str] = {}
        for lbl in order:
            code = lbl_to_code(lbl)
            sector_groups[lbl] = (
                self.code_aware(code) if color_by == "aware" else self.code_class(code)
            )

        present_groups = _uniq_preserve(list(sector_groups.values()))
        pal = self._palette_for_groups(present_groups, color_by=color_by)
        cmap_sector = {lbl: pal.get(sector_groups[lbl], "#777777") for lbl in sector_groups}

        from pycirclize import Circos
        import matplotlib.patches as mpatches

        start = float(self.cfg.chord_start_deg)
        end = float(self.cfg.chord_end_deg)
        if not (-360 <= start < end <= 360):
            raise ValueError(
                f"Invalid chord start/end: start={start}, end={end}. "
                f"Must satisfy -360 <= start < end <= 360."
            )

        circos = Circos.chord_diagram(
            mat,
            start=start,
            end=end,
            space=int(self.cfg.chord_space_deg),
            r_lim=self.cfg.chord_r_lim,
            cmap=cmap_sector,
            order=order,
            label_kws=dict(size=int(self.cfg.chord_label_size)),
            link_kws=dict(
                ec=str(self.cfg.chord_link_ec),
                lw=float(self.cfg.chord_link_lw),
                alpha=float(self.cfg.chord_link_alpha),
                direction=int(self.cfg.chord_direction),
            ),
        )

        fig = circos.plotfig(figsize=self.cfg.chord_figsize)
        ax = circos.ax
        ax.set_title(title, fontsize=int(self.cfg.chord_title_size), fontweight="bold", pad=18)

        panel_bottom = float(self.cfg.chord_panel_bottom)
        panel_h = float(self.cfg.chord_panel_height)
        gap = float(self.cfg.chord_panel_gap)

        chord_y0 = panel_bottom + panel_h + gap
        chord_h = 1.0 - chord_y0 - 0.06
        ax.set_position([0.06, chord_y0, 0.88, chord_h])

        handles = [mpatches.Patch(color=pal[g], label=str(g)) for g in present_groups]
        if handles:
            ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=10)

        out_w = e.groupby("src")["weight"].sum().to_dict()
        in_w = e.groupby("dst")["weight"].sum().to_dict()

        sector_scores: Dict[str, float] = {}
        for lbl in r_labels:
            sector_scores[lbl] = out_w.get(lbl_to_code(lbl), 0.0)
        for lbl in t_labels:
            sector_scores[lbl] = in_w.get(lbl_to_code(lbl), 0.0)

        show_labels = order
        if self.cfg.chord_key_top_n_sectors is not None:
            n = int(self.cfg.chord_key_top_n_sectors)
            show_labels = [
                k for k, _ in sorted(
                    sector_scores.items(), key=lambda kv: kv[1], reverse=True
                )[:n]
            ]

        lines: List[str] = []
        for lbl in show_labels:
            code = lbl_to_code(lbl)
            aware = self.code_aware(code)
            cls = _shorten_class_name(self.code_class(code))
            side = lbl.split(":", 1)[0]
            lines.append(f"{side}:{code}  |  AWaRe={aware}  |  Class={cls}")

        mid = (len(lines) + 1) // 2
        left_lines = lines[:mid]
        right_lines = lines[mid:]

        key_ax = fig.add_axes([0.06, panel_bottom, 0.88, panel_h], zorder=50)
        key_ax.set_facecolor("white")
        key_ax.patch.set_alpha(1.0)
        key_ax.axis("off")

        key_ax.text(
            0.0,
            0.98,
            "Key (top sectors):",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        key_font = int(self.cfg.chord_key_font_size)
        key_ax.text(
            0.00,
            0.90,
            "\n".join(left_lines) if left_lines else "",
            ha="left",
            va="top",
            fontsize=key_font,
            family="monospace",
        )
        key_ax.text(
            0.52,
            0.90,
            "\n".join(right_lines) if right_lines else "",
            ha="left",
            va="top",
            fontsize=key_font,
            family="monospace",
        )

        out_png = os.path.join(out_dir, f"{base_name}.png")
        out_svg = os.path.join(out_dir, f"{base_name}.svg")
        out_pdf = os.path.join(out_dir, f"{base_name}.pdf")

        fig.savefig(out_png, dpi=int(self.cfg.chord_dpi))
        fig.savefig(out_svg)
        fig.savefig(out_pdf)

        out_html = os.path.join(out_dir, f"{base_name}.html")
        try:
            with open(out_svg, "r", encoding="utf-8") as f_svg, open(
                out_html, "w", encoding="utf-8"
            ) as f_html:
                svg_text = f_svg.read()
                f_html.write("<html><body style='margin:0; padding:0;'>\n")
                f_html.write(svg_text)
                f_html.write("\n</body></html>")
        except Exception:
            out_html = ""

        return {"html": out_html, "png": out_png, "svg": out_svg, "pdf": out_pdf}

    # ----------------------------
    # Circos radar: R outflow vs T inflow
    # ----------------------------
    def circos_radar_rt(
        self,
        out_dir: str,
        base_name: str,
        title: str,
        color_by: str = "aware",
        top_n: int = 18,
        legend_loc: str = "upper right",
    ) -> Dict[str, str]:
        if self.edges_df is None:
            self.build_cross_edges()
        e = self.edges_df.copy()
        if e.empty:
            raise ValueError("No edges to plot.")
        if color_by not in {"aware", "class"}:
            raise ValueError("color_by must be 'aware' or 'class'.")

        _ensure_dir(out_dir)

        out_w = e.groupby("src")["weight"].sum()
        in_w = e.groupby("dst")["weight"].sum()

        all_codes = sorted(set(out_w.index).union(set(in_w.index)))
        total = {c: float(out_w.get(c, 0.0)) + float(in_w.get(c, 0.0)) for c in all_codes}

        if top_n >= len(all_codes):
            top_codes = all_codes
        else:
            top_codes = [
                c
                for c, _ in sorted(
                    total.items(), key=lambda kv: kv[1], reverse=True
                )[: int(top_n)]
            ]

        df = pd.DataFrame(
            {c: [float(out_w.get(c, 0.0)), float(in_w.get(c, 0.0))] for c in top_codes},
            index=["R_outflow", "T_inflow"],
        )

        vmax = float(df.to_numpy().max()) if df.size else 1.0
        vmax = max(vmax, 1e-9)

        from pycirclize import Circos
        import matplotlib.patches as mpatches

        circos = Circos.radar_chart(df, vmax=vmax, marker_size=4, grid_interval_ratio=0.2)
        fig = circos.plotfig(figsize=self.cfg.radar_figsize)
        circos.ax.set_title(title, fontsize=16, fontweight="bold", pad=18)

        def grp(code: str) -> str:
            return self.code_aware(code) if color_by == "aware" else self.code_class(code)

        groups = _uniq_preserve([grp(c) for c in top_codes])
        pal = self._palette_for_groups(groups, color_by=color_by)

        handles = [
            mpatches.Patch(color=pal[g], label=str(g)) for g in groups if g in pal
        ]
        if handles:
            circos.ax.legend(handles=handles, loc=legend_loc, fontsize=10, frameon=False)

        out_png = os.path.join(out_dir, f"{base_name}.png")
        out_svg = os.path.join(out_dir, f"{base_name}.svg")
        out_pdf = os.path.join(out_dir, f"{base_name}.pdf")

        fig.savefig(out_png, dpi=int(self.cfg.radar_dpi), bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")

        return {"png": out_png, "svg": out_svg, "pdf": out_pdf}

    # ----------------------------
    # Network graphs (bipartite R -> T)
    # ----------------------------
    def network_publication(
        self,
        out_dir: str,
        base_name: str,
        color_by: str = "aware",
        bipartite_rt: bool = True,
        r_label: str = "(R)",
        t_label: str = "(T)",
    ) -> Dict[str, str]:
        if self.edges_df is None:
            self.build_cross_edges()
        e = self.edges_df.copy()
        if e.empty:
            raise ValueError("No edges for network graph.")

        _ensure_dir(out_dir)

        import networkx as nx
        import matplotlib.pyplot as plt

        pal_aware = self._aware_palette_hex()

        G = nx.DiGraph()
        r_nodes = sorted(set(e["src"].tolist()))
        t_nodes = sorted(set(e["dst"].tolist()))

        for code in r_nodes:
            G.add_node(
                f"{code}__R",
                aware=self.code_aware(code),
                cls=self.code_class(code),
                label=f"{code} {r_label}",
            )
        for code in t_nodes:
            G.add_node(
                f"{code}__T",
                aware=self.code_aware(code),
                cls=self.code_class(code),
                label=f"{code} {t_label}",
            )

        for _, row in e.iterrows():
            G.add_edge(
                f"{row['src']}__R",
                f"{row['dst']}__T",
                weight=float(row["weight"]),
                rules=int(row["rule_count"]),
            )

        out_w = e.groupby("src")["weight"].sum().to_dict()
        in_w = e.groupby("dst")["weight"].sum().to_dict()
        r_sorted = sorted(r_nodes, key=lambda c: out_w.get(c, 0.0), reverse=True)
        t_sorted = sorted(t_nodes, key=lambda c: in_w.get(c, 0.0), reverse=True)

        pos: Dict[str, Tuple[float, float]] = {}
        for i, code in enumerate(r_sorted):
            pos[f"{code}__R"] = (0.0, -i)
        for i, code in enumerate(t_sorted):
            pos[f"{code}__T"] = (1.0, -i)

        node_colors = [
            pal_aware.get(G.nodes[n].get("aware", "Not set"), "#616161") for n in G.nodes()
        ]
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}

        plt.figure(figsize=(16, 10))
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=980,
            node_color=node_colors,
            linewidths=1.2,
            edgecolors="#222222",
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=11)

        wmax = max((d["weight"] for _, _, d in G.edges(data=True)), default=1.0)
        widths = [0.8 + 6.0 * (d["weight"] / wmax) for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            arrowsize=24,
            width=widths,
            alpha=0.55,
            edge_color="#333333",
        )

        plt.title(base_name, fontsize=18, fontweight="bold")
        plt.axis("off")

        png_path = os.path.join(out_dir, f"{base_name}.png")
        svg_path = os.path.join(out_dir, f"{base_name}.svg")
        pdf_path = os.path.join(out_dir, f"{base_name}.pdf")

        plt.savefig(png_path, dpi=350, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()

        return {"png": png_path, "svg": svg_path, "pdf": pdf_path}



# if __name__ == "__main__":
#     rules = pd.read_csv("testRules_normalized.csv")
#     # rules = pd.read_csv("rules_2019.0.csv")

#     with open("./datasets/antibiotic_class_grouping.json", "r", encoding="utf-8") as f:
#         drugclass_json = json.load(f)

#     with open("./datasets/antibiotic_class.json", "r", encoding="utf-8") as f:
#         aware_json = json.load(f)

#     cfg = CrossVizConfig(
#         weight_col="Support",
#         min_weight=0.0, 
#         top_edges=250,
#         chord_key_top_n_sectors=18,
#         chord_panel_bottom=-0.25,  # increase to -0.25 if you want more room
#         chord_panel_height=0.28,  # increase to 0.32 if you want even more room
#         chord_figsize=(12.5, 12.5),
#     )

#     viz = ARMRuleCrossVisualizer(rules, drugclass_json, aware_json=aware_json, cfg=cfg)
#     viz.build_cross_edges()

#     out_dir = "./outputs_crossviz_publication"
#     _ensure_dir(out_dir)

#     sankey = viz.plot_sankey("Cross-informative flow: Resistance → Cross-testing (Sankey)")
#     viz.save_plotly_all(sankey, out_dir, "01_sankey_cross")

#     heat = viz.plot_heatmap("Cross-informative matrix: Resistance → Cross-testing (Heatmap)")
#     viz.save_plotly_all(heat, out_dir, "02_heatmap_cross")

#     viz.chord_pycirclize_bipartite(
#         out_dir=out_dir,
#         base_name="03_chord_bipartite_AWaRe_pycirclize_pub_FIXED",
#         title="Chord (bipartite): Cross-informative flow (R → T) — colored by AWaRe",
#         color_by="aware",
#         legend_loc="upper right",
#     )

#     viz.circos_radar_rt(
#         out_dir=out_dir,
#         base_name="04_circos_radar_R_outflow_vs_T_inflow_all",
#         title="Circos radar: R outflow vs T inflow (all antibiotics) — colored by AWaRe",
#         color_by="aware",
#         top_n=100000,
#         legend_loc="upper right",
#     )

#     viz.network_publication(
#         out_dir=out_dir,
#         base_name="05_network_cross_colorAware_bipartite",
#         color_by="aware",
#         bipartite_rt=True,
#         r_label="(R)",
#         t_label="(T)",
#     )

#     print(f"Done. Outputs saved to: {os.path.abspath(out_dir)}")
