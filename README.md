
# escalation-causal

**Estimating causal effects of antimicrobial resistance on diagnostic escalation: a targeted learning approach using surveillance data**

This package implements a complete causal inference pipeline for estimating whether resistance to a trigger antibiotic *causes* increased probability of testing a target antibiotic outside routine practice (diagnostic escalation). The pipeline uses state‑of‑the‑art methods:

- **Targeted Maximum Likelihood Estimation (TMLE)** with cross‑fitting for doubly robust, efficient estimation of the risk difference.
- **Joint selection model** (bivariate probit) to account for correlation between testing and resistance due to unmeasured common causes (e.g., severity).
- **Routine policy learning** directly from data to define counterfactual escalation outcomes.
- **Causal forests** to explore heterogeneity in treatment effects.
- **Cinelli‑Hazlett sensitivity analysis** to bound the impact of unmeasured confounding.
- **Bayesian hierarchical shrinkage** to handle multiple comparisons.

The code is modular, well‑documented, and designed to work with data from antimicrobial resistance surveillance systems (e.g., the German ARS). It can be adapted to other pathogens, antibiotics, and healthcare settings.

---

## Features

- **Data preparation**: Load and filter surveillance data, construct antibiotic tested/resistance flags.
- **Phase 1 screening**: Efficiently select promising trigger–target pairs using crude odds ratios with FDR correction (performed on a separate discovery set to avoid winner’s curse).
- **Routine policy learning**: Estimate context‑specific routine testing probabilities (empirical or ML‑based) and compute continuous escalation scores.
- **Selection bias adjustment**: Inverse probability weighting via standard ML models or a joint bivariate probit model that accounts for correlation between testing and resistance.
- **Doubly robust causal estimation**: TMLE for risk difference with cross‑fitted nuisance models (propensity score and outcome regressions).
- **Heterogeneity analysis**: Causal forests to discover effect modification by clinical context (ward type, age, year).
- **Sensitivity analysis**: Cinelli‑Hazlett robustness values and contour plots for unmeasured confounding.
- **Multiple comparison adjustment**: Bayesian hierarchical shrinkage of estimates across pairs.
- **Export**: Filtered cascade dependencies in JSON/CSV for downstream use.
- **Publication‑ready visualisations**: Forest plots, network graphs, heatmaps, scatter plots, calibration curves, time trends, subgroup forests, and combined figures.

---

## Installation

### From source

```bash
git clone https://github.com/yourusername/escalation-causal.git
cd escalation-causal
pip install -e .
```
### Dependencies

Core dependencies are listed in `requirements.txt`:


Optional dependencies (for specific features):

- `econml` – causal forests (`pip install econml`)
- `pymc` and `arviz` – Bayesian shrinkage (`pip install pymc arviz`)
- `kaleido` – static image exports from Plotly (`pip install kaleido`)

---

## Quick start

1. **Prepare your data** in a format compatible with the German ARS (or adapt the `DataLoader`).  
   The minimal required columns are:
   - Patient demographics: `AgeGroup`, `Sex`
   - Specimen: `Pathogen`, `TextMaterialgroupRkiL0`
   - Healthcare setting: `ARS_WardType`, `CareType`, `Anonymized_Lab`
   - Time: `Year`, `Month`
   - Antibiotic columns: for each antibiotic, two columns named `"{code} - {name}_Tested"` and `"{code} - {name}_Outcome"` (see example data).

2. **Create a filter configuration** JSON file specifying inclusion/exclusion criteria (see `config_all_klebsiella.json` for an example).

3. **Run the main analysis**:

```bash
python run_analysis.py
```

This will:
- Load and filter data
- Split into discovery (50% of labs) and estimation (50% of labs) sets
- Perform Phase 1 screening on discovery set → select 100 pairs
- Run the causal pipeline on estimation set
- Save results, diagnostics, and all publication plots in `./output/`

4. **Compare standard vs. joint selection models**:

```bash
python compare_joint_selection.py
```

This runs the pipeline twice (with and without joint selection) on the estimation set and produces a comparison plot.

---

## Detailed usage

### Configuration

All pipeline settings are defined in Pydantic models in `config/settings.py`. You can modify them directly in the scripts or create your own `RunConfig` instance.

Key configuration blocks:

- **SplitConfig**: train/test split (proportion, grouping column, random state).
- **CovariateConfig**: which columns to use as covariates, minimum count per level, etc.
- **PolicyConfig**: context columns for routine policy, method (`empirical` or `ml`), minimum context size, ML model type, calibration settings.
- **NuisanceConfig**: model types for testing, propensity, and outcome; whether to calibrate; use of joint selection.
- **TMLEConfig**: number of cross‑fitting folds, probability clipping, weight capping, minimum sample sizes, etc.

### Main scripts

#### `run_analysis.py`

Performs the complete end‑to‑end analysis:

1. **Load and filter** data using `DataLoader` and `FilterConfig`.
2. **Split** data into discovery (screening) and estimation sets (grouped by laboratory).
3. **Phase 1 screening** on discovery set → selects top 100 pairs.
4. **Configure** the causal pipeline.
5. **Run pipeline** on estimation set → obtains risk differences, confidence intervals, diagnostics.
6. **Save** results and generate failure summary.
7. **Export** filtered cascade dependencies (JSON/CSV).
8. **Bayesian shrinkage** across all pairs → produces posterior probabilities and shrinkage forest plot.
9. **Heterogeneity analysis** (causal forest) for the pair with the largest absolute RD – if pipeline stored test data (see note below).
10. **Generate publication plots** (forest, sensitivity contour, network, heatmap, scatter, calibration, time trend, subgroup forests, combined figure placeholder).
11. **Optional yearly time‑trend** analysis (runs pipeline separately for each year).

**Note on heterogeneity analysis**: To enable causal forest, you need to add three lines to `pipeline.py` after the test set is created:

```python
self.df_test = df_test
self.flags_test = flags_test
self.esc_scores = esc_scores
```

Without these, the script will skip this step.

#### `compare_joint_selection.py`

Compares estimates obtained with and without the joint selection model:

- Splits data into discovery/estimation (same as above).
- Phase 1 screening on discovery set.
- Runs pipeline on estimation set twice: once with `use_joint_selection=False`, once with `True`.
- Merges results and produces a two‑panel comparison plot (point estimates with CIs + differences).

### Outputs

All outputs are saved under `./output/` (or a custom directory). Key files:

- `results.csv` – one row per pair with estimates, CIs, p‑values, diagnostics.
- `manifest.json` – metadata (config, cohort info, runtime).
- `failed_pairs_summary.csv` – reasons for failures and group sizes.
- `bayesian_shrinkage_summary.csv` – shrunken estimates and posterior probabilities.
- `yearly_results.csv` – estimates per year (if yearly analysis run).
- `figures/` – all publication‑ready plots (PNG, PDF, HTML).

---

## Interpretation of the escalation score

The continuous escalation score is defined as:

\[
Y^*_D = \frac{T_D}{\hat{P}(T_D=1|C)}
\]

where \(T_D\) indicates testing of target drug \(D\), and \(\hat{P}(T_D=1|C)\) is the estimated probability that \(D\) is tested routinely given context \(C\) (laboratory, pathogen group, year). For tested isolates, a score > 1 means testing was more likely than average in that context; a score of 2 means testing occurred in a context where only 50% of isolates are tested, etc. The risk difference \(\psi = \mathbb{E}[Y^*|A=1] - \mathbb{E}[Y^*|A=0]\) measures how much resistance increases this escalation score.

---

## Customising for your own data

If your surveillance data have a different structure, you may need to adapt:

- `DataLoader` – modify `_detect_antibiotic_columns` to match your column naming convention.
- `FilterConfig` – adjust filtering rules.
- `CovariateConfig` – change the list of covariate columns.
- `PolicyConfig` – adapt context columns to those available in your dataset.

The core causal inference modules (`estimator`, `nuisance`, `heterogeneity`, `sensitivity`) are dataset‑agnostic as long as the input DataFrames are formatted correctly.

---

## Citation

If you use this package in your research, please cite:

```
Awotoro E, et al. (2026). escalation-causal: Estimating causal effects of antimicrobial resistance on diagnostic escalation. 
https://github.com/yourusername/escalation-causal
```

*(A full paper is forthcoming.)*

---

## Contact

For questions, bug reports, or suggestions, please open an issue on GitHub or contact:

Ebenezer Awotoro  
[ebenco94@gmail.com](mailto:ebenco94@gmail.com)  
Robert Koch Institute, Nordufer, Berlin, Germany.