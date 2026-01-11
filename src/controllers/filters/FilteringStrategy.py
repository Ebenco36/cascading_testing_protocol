"""
FilteringStrategy: Dynamic, JSON-driven filtering framework for AMR surveillance
COMPLETE VERSION - Handles string/numeric type conversion and string booleans
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import json
import re


class DynamicColumnFilter:
    """Advanced filter for ANY column with ANY operator - 15 operators total"""
    
    def __init__(
        self,
        column: str,
        operator: str,
        value: Any = None,
        values: List[Any] = None,
        min_val: Any = None,
        max_val: Any = None,
        case_sensitive: bool = True
    ):
        self.column = column
        self.operator = operator.lower()
        self.value = value
        self.values = values or []
        self.min_val = min_val
        self.max_val = max_val
        self.case_sensitive = case_sensitive
        
        self.n_before = 0
        self.n_after = 0
        self.n_removed = 0
        self.pct_retained = 0.0
    
    def apply(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Apply filter and return filtered dataframe + report"""
        self.n_before = len(df)
        
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in dataframe")
        
        col_data = df[self.column]
        
        # 15 Operators
        if self.operator == "equals":
            mask = self._equals(df)
        elif self.operator == "not_equals":
            mask = ~self._equals(df)
        elif self.operator == "in":
            mask = self._in(df)
        elif self.operator == "not_in":
            mask = ~self._in(df)
        elif self.operator == "contains":
            mask = self._contains(df)
        elif self.operator == "regex":
            mask = self._regex(df)
        elif self.operator == "gte":
            mask = self._numeric_compare(col_data, ">=", self.value)
        elif self.operator == "lte":
            mask = self._numeric_compare(col_data, "<=", self.value)
        elif self.operator == "gt":
            mask = self._numeric_compare(col_data, ">", self.value)
        elif self.operator == "lt":
            mask = self._numeric_compare(col_data, "<", self.value)
        elif self.operator == "range":
            mask = self._range(col_data)
        elif self.operator == "is_true":
            mask = self._is_true(col_data)
        elif self.operator == "is_false":
            mask = self._is_false(col_data)
        elif self.operator == "is_null":
            mask = col_data.isna()
        elif self.operator == "not_null":
            mask = ~col_data.isna()
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
        
        df_filtered = df[mask].copy()
        
        self.n_after = len(df_filtered)
        self.n_removed = self.n_before - self.n_after
        self.pct_retained = (self.n_after / self.n_before * 100) if self.n_before > 0 else 0
        
        report = {
            "column": self.column,
            "operator": self.operator,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "n_removed": self.n_removed,
            "pct_retained": self.pct_retained,
        }
        
        if verbose:
            print(
                f"[{self.column}:{self.operator}] {self.n_before:,} → {self.n_after:,} "
                f"({self.pct_retained:.1f}% retained, {self.n_removed:,} dropped)"
            )
        
        return df_filtered, report
    
    def _equals(self, df: pd.DataFrame) -> pd.Series:
        """Equals operator - string comparison"""
        col_data = df[self.column].astype(str)
        value = str(self.value)
        
        if not self.case_sensitive:
            return col_data.str.lower() == value.lower()
        return col_data == value
    
    def _in(self, df: pd.DataFrame) -> pd.Series:
        """In operator - check if value is in list"""
        col_data = df[self.column].astype(str)
        values = [str(v) for v in self.values]
        
        if not self.case_sensitive:
            values_lower = [str(v).lower() for v in values]
            return col_data.str.lower().isin(values_lower)
        return col_data.isin(values)
    
    def _contains(self, df: pd.DataFrame) -> pd.Series:
        """Contains operator - substring match"""
        col_data = df[self.column].astype(str)
        
        if not self.case_sensitive:
            return col_data.str.lower().str.contains(str(self.value).lower(), na=False, regex=False)
        return col_data.str.contains(str(self.value), na=False, regex=False)
    
    def _regex(self, df: pd.DataFrame) -> pd.Series:
        """Regex operator - regex pattern match"""
        col_data = df[self.column].astype(str)
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return col_data.str.match(self.value, flags=flags, na=False)
    
    def _numeric_compare(self, col_data: pd.Series, op: str, value: Any) -> pd.Series:
        """Handle numeric comparisons with type conversion (>=, <=, >, <)"""
        try:
            # Convert to numeric
            col_numeric = pd.to_numeric(col_data, errors='coerce')
            value_numeric = float(value)
            
            if op == ">=":
                return col_numeric >= value_numeric
            elif op == "<=":
                return col_numeric <= value_numeric
            elif op == ">":
                return col_numeric > value_numeric
            elif op == "<":
                return col_numeric < value_numeric
        except:
            pass
        
        return col_data.astype(str) == str(value)
    
    def _range(self, col_data: pd.Series) -> pd.Series:
        """Range operator - values between min and max (inclusive)"""
        try:
            # Convert to numeric
            col_numeric = pd.to_numeric(col_data, errors='coerce')
            min_numeric = float(self.min_val)
            max_numeric = float(self.max_val)
            
            return (col_numeric >= min_numeric) & (col_numeric <= max_numeric)
        except:
            # Fallback to string comparison
            col_str = col_data.astype(str)
            return (col_str >= str(self.min_val)) & (col_str <= str(self.max_val))
    
    def _is_true(self, col_data: pd.Series) -> pd.Series:
        """Is True operator - works with bool, string 'True', int 1"""
        # Check for actual boolean True
        mask = col_data == True
        
        # Also check for string "True" (case insensitive)
        col_str = col_data.astype(str)
        mask = mask | (col_str.str.lower() == "true")
        
        # Also check for int 1
        try:
            col_numeric = pd.to_numeric(col_data, errors='coerce')
            mask = mask | (col_numeric == 1)
        except:
            pass
        
        return mask
    
    def _is_false(self, col_data: pd.Series) -> pd.Series:
        """Is False operator - works with bool, string 'False', int 0"""
        # Check for actual boolean False
        mask = col_data == False
        
        # Also check for string "False" (case insensitive)
        col_str = col_data.astype(str)
        mask = mask | (col_str.str.lower() == "false")
        
        # Also check for int 0
        try:
            col_numeric = pd.to_numeric(col_data, errors='coerce')
            mask = mask | (col_numeric == 0)
        except:
            pass
        
        return mask


class DynamicFilterPipeline:
    """Chain multiple filters with AND logic"""
    
    def __init__(self, filters: List[DynamicColumnFilter] = None):
        self.filters = filters or []
        self.reports = []
    
    def apply(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        stop_on_empty: bool = True
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply all filters sequentially with AND logic
        
        Args:
            df: Input dataframe
            verbose: Print filter reports
            stop_on_empty: Stop if dataframe becomes empty
            
        Returns:
            Tuple of (filtered_dataframe, list_of_reports)
        """
        self.reports = []
        df_current = df.copy()
        
        if verbose:
            print("\n" + "=" * 80)
            print("APPLYING FILTERS")
            print("=" * 80)
        
        for filter_obj in self.filters:
            df_current, report = filter_obj.apply(df_current, verbose=verbose)
            self.reports.append(report)
            
            if len(df_current) == 0 and stop_on_empty:
                print(f"\n DataFrame empty after {filter_obj.column}. Stopping.")
                break
        
        if verbose:
            self._print_summary(df)
        
        return df_current, self.reports
    
    def _print_summary(self, df_original: pd.DataFrame) -> None:
        """Print filtering summary statistics"""
        print("\n" + "=" * 80)
        print(" FILTERING SUMMARY")
        print("=" * 80)
        
        if self.reports:
            df_final_rows = self.reports[-1]["n_after"]
            pct_retained = (df_final_rows / len(df_original) * 100) if len(df_original) > 0 else 0
            
            print(f"Initial rows:     {len(df_original):,}")
            print(f"Final rows:       {df_final_rows:,}")
            print(f"Total removed:    {len(df_original) - df_final_rows:,}")
            print(f"Overall retained: {pct_retained:.1f}%")
            print("=" * 80 + "\n")


class FilterConfig:
    """JSON-based configuration management for filtering"""
    
    def __init__(self, config_dict: Dict):
        """
        Initialize from configuration dictionary
        
        Args:
            config_dict: Dictionary with keys:
                - name: Config name
                - description: Config description
                - filters: List of filter configs
                - verbose: Print messages (optional, default True)
        """
        self.name = config_dict.get("name", "Default")
        self.description = config_dict.get("description", "")
        self.filters_config = config_dict.get("filters", [])
        self.verbose = config_dict.get("verbose", True)
    
    def build_pipeline(self) -> DynamicFilterPipeline:
        """
        Build DynamicFilterPipeline from config
        
        Returns:
            DynamicFilterPipeline object
        """
        filters = []
        
        for filter_config in self.filters_config:
            f = DynamicColumnFilter(
                column=filter_config["column"],
                operator=filter_config["operator"],
                value=filter_config.get("value"),
                values=filter_config.get("values"),
                min_val=filter_config.get("min"),
                max_val=filter_config.get("max"),
                case_sensitive=filter_config.get("case_sensitive", True)
            )
            filters.append(f)
        
        return DynamicFilterPipeline(filters)
    
    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply configuration to dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (filtered_dataframe, list_of_reports)
        """
        pipeline = self.build_pipeline()
        return pipeline.apply(df, verbose=self.verbose)
    
    @classmethod
    def from_json(cls, path: str) -> "FilterConfig":
        """
        Load configuration from JSON file
        
        Args:
            path: Path to JSON config file
            
        Returns:
            FilterConfig object
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FilterConfig":
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            FilterConfig object
        """
        return cls(config_dict)
    
    def to_json(self, path: str = None) -> str:
        """
        Convert to JSON string or save to file
        
        Args:
            path: Optional file path to save to
            
        Returns:
            JSON string
        """
        config_dict = {
            "name": self.name,
            "description": self.description,
            "filters": self.filters_config,
            "verbose": self.verbose,
        }
        json_str = json.dumps(config_dict, indent=2)
        
        if path:
            with open(path, "w") as f:
                f.write(json_str)
            print(f"✓ Config saved to {path}")
        
        return json_str


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    # Example 1: Load from JSON
    from FilteringStrategy import FilterConfig
    import pandas as pd
    
    df = pd.read_parquet("raw_data.parquet")
    config = FilterConfig.from_json("config.json")
    filtered_df, reports = config.apply(df)
    filtered_df.to_parquet("filtered_data.parquet")
    
    # Example 2: Create from dictionary
    config_dict = {
        "name": "E. coli Filter",
        "description": "Filter for E. coli",
        "filters": [
            {
                "column": "Pathogen",
                "operator": "equals",
                "value": "Escherichia coli"
            },
            {
                "column": "Year",
                "operator": "range",
                "min": 2020,
                "max": 2023
            },
            {
                "column": "IsExcluded",
                "operator": "is_false"
            }
        ]
    }
    
    config = FilterConfig.from_dict(config_dict)
    filtered_df, reports = config.apply(df)
    """
    pass
