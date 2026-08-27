from __future__ import annotations

from dataclasses import dataclass

_PRICE_AVAILABILITY = "daily bar available after close; tradable from next open"
_BASIC_AVAILABILITY = "same-date daily_basic available after close; tradable from next open"
_ENDPOINT_RULE = (
    "forward-fill only from a prior observed close; require positive filled endpoints"
)
_RETURN_20_RULE = (
    "require 20 open-calendar returns after prior-close fill; pre-history remains missing"
)
_RETURN_60_RULE = (
    "require 60 open-calendar returns after prior-close fill; pre-history remains missing"
)


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    direction: int
    lookback: int
    formula_version: str
    input_fields: tuple[str, ...]
    availability: str
    family: str
    missing_rule: str
    formula: str
    hypothesis: str


FACTOR_CATALOG: tuple[FactorDefinition, ...] = (
    FactorDefinition(
        "reversal_1d", 1, 1, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "reversal", _ENDPOINT_RULE,
        "-log(adjusted_close_t / adjusted_close_t-1)",
        "One-day price pressure may reverse.",
    ),
    FactorDefinition(
        "reversal_5d", 1, 5, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "reversal", _ENDPOINT_RULE,
        "-log(adjusted_close_t / adjusted_close_t-5)",
        "Short-horizon price pressure may reverse.",
    ),
    FactorDefinition(
        "reversal_10d", 1, 10, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "reversal", _ENDPOINT_RULE,
        "-log(adjusted_close_t / adjusted_close_t-10)",
        "Two-week price pressure may reverse.",
    ),
    FactorDefinition(
        "reversal_20d", 1, 20, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "reversal", _ENDPOINT_RULE,
        "-log(adjusted_close_t / adjusted_close_t-20)",
        "One-month price pressure may reverse.",
    ),
    FactorDefinition(
        "momentum_20_5", 1, 20, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "momentum", _ENDPOINT_RULE,
        "log(adjusted_close_t-5 / adjusted_close_t-20)",
        "Returns may persist after skipping the most recent week.",
    ),
    FactorDefinition(
        "momentum_60_20", 1, 60, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "momentum", _ENDPOINT_RULE,
        "log(adjusted_close_t-20 / adjusted_close_t-60)",
        "Quarter-horizon trends may persist after a one-month skip.",
    ),
    FactorDefinition(
        "momentum_120_20", 1, 120, "v1", ("adjusted_close",),
        _PRICE_AVAILABILITY, "momentum", _ENDPOINT_RULE,
        "log(adjusted_close_t-20 / adjusted_close_t-120)",
        "Medium-horizon trends may persist.",
    ),
    FactorDefinition(
        "momentum_250_20", 1, 250, "v1", ("adjusted_close",),
        _PRICE_AVAILABILITY, "momentum", _ENDPOINT_RULE,
        "log(adjusted_close_t-20 / adjusted_close_t-250)",
        "Long-horizon trends may persist after skipping the recent month.",
    ),
    FactorDefinition(
        "low_vol_20", 1, 20, "v1", ("adjusted_close",), _PRICE_AVAILABILITY,
        "risk", _RETURN_20_RULE,
        "-std(log adjusted-close return, 20)",
        "Lower realized volatility may earn better risk-adjusted returns.",
    ),
    FactorDefinition(
        "low_downside_vol_60", 1, 60, "v1", ("adjusted_close",),
        _PRICE_AVAILABILITY, "risk", _RETURN_60_RULE,
        "-sqrt(mean(min(log return, 0)^2, 60))",
        "Lower downside variation may be rewarded.",
    ),
    FactorDefinition(
        "low_range_vol_20", 1, 20, "v1", ("high", "low"),
        _PRICE_AVAILABILITY, "risk",
        "after price history starts, missing or zero daily range maps to zero",
        "-sqrt(mean(log(high / low)^2, 20) / (4*log(2)))",
        "Intraday range supplies an alternative volatility estimate.",
    ),
    FactorDefinition(
        "beta_60", -1, 60, "v1", ("adjusted_close", "index_close"),
        _PRICE_AVAILABILITY, "risk",
        "require 60 aligned returns after prior-close fill and positive benchmark variance",
        "cov(stock_return, index_return, 60) / var(index_return, 60)",
        "Lower market beta may earn better risk-adjusted returns.",
    ),
    FactorDefinition(
        "low_idio_vol_60", 1, 60, "v1", ("adjusted_close", "index_close"),
        _PRICE_AVAILABILITY, "risk",
        "require 60 aligned returns after prior-close fill and non-negative residual variance",
        "-sqrt(var(stock_return,60) - cov(stock,index,60)^2/var(index,60))",
        "Lower market-residual volatility may be rewarded.",
    ),
    FactorDefinition(
        "skewness_60", -1, 60, "v1", ("adjusted_close",),
        _PRICE_AVAILABILITY, "risk", _RETURN_60_RULE,
        "skew(log adjusted-close return, 60)",
        "Lottery-like positive skew may be overpriced.",
    ),
    FactorDefinition(
        "max_return_20", -1, 20, "v1", ("adjusted_close",),
        _PRICE_AVAILABILITY, "risk", _RETURN_20_RULE,
        "max(log adjusted-close return, 20)",
        "Stocks with extreme recent gains may attract lottery demand.",
    ),
    FactorDefinition(
        "amihud_20", 1, 20, "v1", ("adjusted_close", "amount_cny"),
        _PRICE_AVAILABILITY, "liquidity",
        "require 20 returns with strictly positive amount",
        "log(mean(abs(return) / amount_cny, 20))",
        "Illiquidity may command a premium before implementation costs.",
    ),
    FactorDefinition(
        "free_turnover_20", -1, 20, "v1", ("turnover_rate_f",),
        _BASIC_AVAILABILITY, "liquidity",
        "missing on a valid suspended session is treated as zero turnover",
        "log(mean(free-float turnover_rate, 20))",
        "Very high turnover may indicate crowding and short-lived demand.",
    ),
    FactorDefinition(
        "turnover_shock_5_60", -1, 60, "v1", ("turnover_rate",),
        _BASIC_AVAILABILITY, "liquidity",
        "missing on a valid suspended session is treated as zero turnover",
        "log(mean(turnover_rate, 5) / mean(turnover_rate, 60))",
        "Abnormally high turnover may indicate crowding or overreaction.",
    ),
    FactorDefinition(
        "zero_return_20", 1, 20, "v1", ("adjusted_close",),
        _PRICE_AVAILABILITY, "liquidity",
        _RETURN_20_RULE,
        "mean(abs(log adjusted-close return) <= 1e-12, 20)",
        "A zero-return premium may proxy for neglected or illiquid names.",
    ),
    FactorDefinition(
        "close_location_20", 1, 20, "v1", ("high", "low", "close"),
        _PRICE_AVAILABILITY, "liquidity",
        "zero-range sessions map to neutral; otherwise require 20 observed ranges",
        "mean((2*close-high-low)/(high-low), 20)",
        "Persistent closes near the daily high may reflect buying pressure.",
    ),
    FactorDefinition(
        "size", 1, 0, "v1", ("circ_mv_cny",), _BASIC_AVAILABILITY,
        "size", "forward-fill only from prior observations; require positive value",
        "-log(circulating_market_cap_cny)",
        "Smaller free-float companies may earn a size premium.",
    ),
    FactorDefinition(
        "total_size", 1, 0, "v1", ("total_mv_cny",), _BASIC_AVAILABILITY,
        "size", "forward-fill only from prior observations; require positive value",
        "-log(total_market_cap_cny)",
        "Smaller companies may earn a size premium.",
    ),
    FactorDefinition(
        "free_float_ratio", 1, 0, "v1", ("circ_mv_cny", "total_mv_cny"),
        _BASIC_AVAILABILITY, "size",
        "require positive total cap and circulating cap no greater than total cap",
        "log(circulating_market_cap_cny / total_market_cap_cny)",
        "A larger tradable share base may reduce locked-share and crowding risk.",
    ),
    FactorDefinition(
        "book_to_price", 1, 0, "v1", ("pb",), _BASIC_AVAILABILITY,
        "value", "forward-fill only from prior observations; require positive PB",
        "log(1 / price_to_book)",
        "Cheaper book valuations may earn a value premium.",
    ),
    FactorDefinition(
        "abnormal_turnover_reversal", 1, 60, "v1",
        ("adjusted_close", "turnover_rate"),
        "both price and daily_basic inputs available after close; tradable from next open",
        "interaction", "require valid reversal_5d and positive turnover shock",
        "reversal_5d * max(turnover_shock_5_60, 0)",
        "Short-term reversal may be stronger after abnormal trading activity.",
    ),
)


FACTOR_NAMES: tuple[str, ...] = tuple(definition.name for definition in FACTOR_CATALOG)
DIRECTIONS: dict[str, int] = {
    definition.name: definition.direction for definition in FACTOR_CATALOG
}
FAMILIES: dict[str, str] = {
    definition.name: definition.family for definition in FACTOR_CATALOG
}
