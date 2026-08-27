from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reversal_5d(
    market_panel: pd.DataFrame,
    open_dates: list[str],
    *,
    window: int = 5,
) -> pd.DataFrame:
    """Compute a close-time reversal score using only prices dated at or before t."""
    close = market_panel.pivot(index="trade_date", columns="instrument", values="adjusted_close")
    close = close.reindex(open_dates).sort_index().ffill()
    score = -np.log(close / close.shift(window))
    result = score.stack(future_stack=True).rename("score").reset_index()
    result = result[np.isfinite(result["score"])]
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)

