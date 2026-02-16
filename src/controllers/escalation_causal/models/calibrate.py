"""
Model calibration utilities.
"""

from typing import Any
from sklearn.calibration import CalibratedClassifierCV


def maybe_calibrate(model: Any, calibrate: bool, method: str = "isotonic", cv: int = 3) -> Any:
    """
    If calibrate is True, return a calibrated version of the model using CalibratedClassifierCV.
    Otherwise, return the model unchanged.
    """
    if not calibrate:
        return model
    return CalibratedClassifierCV(model, method=method, cv=cv)