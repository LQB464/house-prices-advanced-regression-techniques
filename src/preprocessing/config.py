"""
preprocessing/config.py

Canonical configuration mappings used for preprocessing ordinal features.

Extended Description
--------------------
This module defines the canonical ordinal mapping used across the entire
preprocessing system. The mapping specifies the ordered levels for various
categorical features in the Ames Housing dataset (e.g. ExterQual, KitchenQual).
These lists define the semantic order of categories, which is crucial for the
OrdinalMapper transformer.

The mapping is referenced by multiple modules, including:
- transformers.py (OrdinalMapper)
- preprocessor.py (default ordinal mapping)
- pipeline.py (pipeline construction)

Main Components
---------------
- ORDINAL_MAP_CANONICAL : A dictionary mapping feature names to an ordered list
  of categorical levels, used to build numeric ordinal encodings.

Usage Example
-------------
>>> from preprocessing.config import ORDINAL_MAP_CANONICAL
>>> ORDINAL_MAP_CANONICAL["ExterQual"]
['Po', 'Fa', 'TA', 'Gd', 'Ex']

Notes
-----
These category orders are based on domain knowledge from the Ames Housing dataset.
If new categories appear at inference time, they will be mapped to NaN and handled
later by imputers in the preprocessing pipeline.
"""

from __future__ import annotations

from typing import Dict, List

ORDINAL_MAP_CANONICAL: Dict[str, List[str]] = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "Street": ["Grvl", "Pave"],
    "Alley": ["NA", "Grvl", "Pave"],
    "CentralAir": ["N", "Y"],
}

__all__ = ["ORDINAL_MAP_CANONICAL"]