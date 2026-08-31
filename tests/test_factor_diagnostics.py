from __future__ import annotations

import numpy as np
import pandas as pd

from csi500_alpha.research.diagnostics import compute_factor_diagnostics


def test_quintiles_are_oriented_by_economic_direction() -> None:
    instruments = [f"S{position}" for position in range(10)]
    score = np.arange(10, dtype=float)
    features = pd.DataFrame(
        {
            "decision_date": "20250102",
            "instrument": instruments,
            "inverse__z": score,
        }
    )
    labels = pd.DataFrame(
        {
            "decision_date": "20250102",
            "instrument": instruments,
            "forward_active_return": -score / 100.0,
        }
    )
    quality = pd.DataFrame(
        {
            "decision_date": ["20250102"],
            "factor": ["inverse"],
            "coverage": [1.0],
            "active": [True],
            "clipped_fraction": [0.0],
            "industry_neutralized": [True],
        }
    )

    diagnostics = compute_factor_diagnostics(
        features=features,
        labels=labels,
        feature_quality=quality,
        factor_names=("inverse",),
        directions={"inverse": -1},
    )
    means = diagnostics.quintile_returns.set_index("quintile")[
        "mean_active_return"
    ]

    assert means.loc[5] > means.loc[1]
    assert diagnostics.summary.iloc[0]["mean_directed_rank_ic"] > 0
    assert diagnostics.summary.iloc[0]["quintile_monotonicity"] > 0
