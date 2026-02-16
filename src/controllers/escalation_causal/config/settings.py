"""
Configuration Models using Pydantic
====================================

Centralized, validated configuration for the entire causal pipeline.
All settings are defined as Pydantic models with sensible defaults.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator


class SplitConfig(BaseModel):
    """Configuration for train/test splitting."""
    test_size: float = Field(0.3, ge=0.0, le=1.0)
    split_group_col: Optional[str] = Field("Anonymized_Lab", description="Column to use for group shuffle (e.g., lab). If None, random split.")
    random_state: int = 42

    @validator("test_size")
    def test_size_not_zero(cls, v):
        if v == 0:
            raise ValueError("test_size cannot be 0")
        return v


class CovariateConfig(BaseModel):
    """Configuration for building covariates."""
    covariate_cols: List[str] = Field(..., description="List of column names to use as covariates.")
    min_count: int = Field(200, ge=1, description="Minimum count for a categorical level to be kept.")
    max_levels: int = Field(25, ge=1, description="Maximum number of levels per categorical variable.")
    drop_first: bool = Field(True, description="Drop first level in one‑hot encoding to avoid collinearity.")


class PolicyConfig(BaseModel):
    """Configuration for routine policy learning."""
    context_cols: List[str] = Field(..., description="Columns defining context C (e.g., lab, pathogen group, year).")
    method: Literal["empirical", "ml"] = Field("empirical", description="Method to estimate P(T=1|C).")
    min_context_n: int = Field(100, ge=1, description="Minimum number of observations in a context (for empirical method).")
    model_type: Literal["xgb", "rf", "logit"] = Field("xgb", description="Model type for ML method.")
    calibrate: bool = Field(True, description="Whether to calibrate the ML model.")
    calibration_method: Literal["isotonic", "sigmoid"] = Field("isotonic", description="Calibration method.")
    calibration_cv: int = Field(5, ge=2, description="Number of folds for calibration.")


class NuisanceConfig(BaseModel):
    """Configuration for nuisance models used in TMLE."""
    testing_model: Literal["xgb", "rf", "logit"] = Field("xgb", description="Model for P(T_trigger=1|X).")
    propensity_model: Literal["xgb", "rf", "logit"] = Field("xgb", description="Model for P(A=1|X, tested).")
    outcome_model: Literal["xgb", "rf", "logit"] = Field("xgb", description="Model for E[Y*|A, X, tested].")
    calibrate_testing: bool = Field(True, description="Calibrate testing model.")
    calibrate_propensity: bool = Field(False, description="Calibrate propensity model (not usually needed).")
    calibrate_outcome: bool = Field(False, description="Calibrate outcome model (for regression, not typical).")
    testing_cv_folds: int = Field(5, ge=1, description="Number of folds for cross‑fitting the testing model (1 = no cross‑fit).")
    random_state: int = 42
    use_joint_selection: bool = False


class TMLEConfig(BaseModel):
    """Configuration for TMLE estimation (formerly AIPWConfig)."""
    n_folds: int = Field(5, ge=2, description="Number of folds for cross‑fitting nuisance models.")
    min_prob: float = Field(0.01, ge=0.0, le=0.5, description="Minimum probability for clipping (avoids extreme weights).")
    weight_cap_percentile: float = Field(99.0, ge=0.0, le=100.0, description="Cap IPW weights at this percentile.")
    min_tested: int = Field(200, ge=1, description="Minimum tested isolates for a trigger.")
    min_group: int = Field(50, ge=1, description="Minimum size of resistant/susceptible groups among tested.")
    stabilize_weights: bool = Field(True, description="Whether to cap escalation score weights.")
    n_bootstrap: Optional[int] = Field(None, description="If set, use bootstrap for CIs instead of influence curve.")
    alpha: float = Field(0.05, ge=0.0, le=1.0, description="Significance level for confidence intervals.")


class RunConfig(BaseModel):
    """Top‑level configuration for a full pipeline run."""
    split: SplitConfig
    covariates: CovariateConfig
    policy: PolicyConfig
    nuisance: NuisanceConfig
    tmle: TMLEConfig   # renamed from aipw

    class Config:
        extra = "forbid"   # disallow extra fields