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


# A2 candidates are deliberately kept outside ``FACTOR_CATALOG``.  The latter
# is the frozen 25-factor daily catalog used by A0/A1; exposing A2 through a
# separate provider prevents an old config with ``workflow.factors: []`` from
# silently changing its research universe.
A2_DAILY_FACTOR_CATALOG: tuple[FactorDefinition, ...] = (
    FactorDefinition(
        "market_residual_reversal_20",
        1,
        81,
        "a2-v1",
        ("adjusted_close", "benchmark_close"),
        _PRICE_AVAILABILITY,
        "reversal",
        "require 60 prior aligned returns for lagged beta and 20 residual returns",
        "-sum(r_t-j - beta_60_t-j-1 * benchmark_return_t-j, j=0..19)",
        "Short-horizon market-residual price pressure may reverse.",
    ),
    FactorDefinition(
        "market_residual_momentum_120_20",
        1,
        180,
        "a2-v1",
        ("adjusted_close", "benchmark_close"),
        _PRICE_AVAILABILITY,
        "momentum",
        "require lagged 60-day beta and 100 residual returns ending 20 sessions ago",
        "sum(r_t-j - beta_60_t-j-1 * benchmark_return_t-j, j=20..119)",
        "Medium-horizon momentum may be clearer after removing market exposure.",
    ),
    FactorDefinition(
        "turnover_volatility_20",
        -1,
        20,
        "a2-v1",
        ("turnover_rate",),
        _BASIC_AVAILABILITY,
        "liquidity",
        "valid suspended sessions are zero turnover; require 20 open-calendar values",
        "std(log(1 + turnover_rate), 20)",
        "Unstable trading activity may proxy for speculative demand and crowding.",
    ),
    FactorDefinition(
        "high_turnover_return_20",
        -1,
        79,
        "a2-v2",
        ("adjusted_close", "turnover_rate"),
        "price and daily_basic inputs are available after close; tradable next open",
        "trading",
        "zero turnover contributes zero; require a positive trailing 60-day mean",
        "sum(return * max(log(turnover_rate / mean_60(turnover_rate)), 0), 20)",
        "High-turnover price pressure may reverse after speculative overreaction.",
    ),
    FactorDefinition(
        "intraday_strength_20",
        -1,
        20,
        "a2-v2",
        ("adjusted_open", "adjusted_close"),
        _PRICE_AVAILABILITY,
        "trading",
        "require 20 sessions with positive adjusted open and close",
        "mean(log(adjusted_close / adjusted_open), 20)",
        "Persistent intraday demand may reverse when it reflects retail overreaction.",
    ),
    FactorDefinition(
        "limit_up_close_rate_20",
        -1,
        20,
        "a2-v2",
        ("high", "close", "up_limit"),
        _PRICE_AVAILABILITY,
        "price_limit",
        "require 20 sessions with observed high, close and upper price limit",
        "mean(1(close >= up_limit), 20)",
        "Repeated upper-limit closes may identify lottery demand and subsequent overpricing.",
    ),
    FactorDefinition(
        "failed_limit_up_rate_20",
        -1,
        20,
        "a2-v1",
        ("high", "close", "up_limit"),
        _PRICE_AVAILABILITY,
        "price_limit",
        "require 20 sessions with observed high, close and upper price limit",
        "mean(1(high >= up_limit and close < up_limit), 20)",
        "Repeated failed upper-limit attempts may indicate exhausted speculative demand.",
    ),
)


# A3 remains opt-in for the same reason as A2: adding research candidates must
# never change an older experiment whose factor list was left empty.
A3_DAILY_FACTOR_CATALOG: tuple[FactorDefinition, ...] = (
    FactorDefinition(
        "limit_adjusted_momentum_120_20",
        1,
        121,
        "a3-v1",
        ("adjusted_close", "close", "up_limit"),
        _PRICE_AVAILABILITY,
        "momentum",
        "exclude returns on upper-limit closes and the following open session",
        "sum(return * eligible, 100 sessions ending 20 sessions ago)",
        "Medium-horizon momentum may be clearer after removing price-limit delays.",
    ),
    FactorDefinition(
        "alpha006_open_volume_corr_10",
        1,
        10,
        "a3-v1",
        ("open", "volume_shares"),
        _PRICE_AVAILABILITY,
        "price_volume",
        "require ten sessions with positive open and observed share volume",
        "-corr(open, volume_shares, 10)",
        "The public Alpha101 price-volume relation is retained as a baseline seed.",
    ),
    FactorDefinition(
        "high_price_momentum_250_20",
        1,
        250,
        "a3-v1",
        ("adjusted_close", "close"),
        _PRICE_AVAILABILITY,
        "momentum",
        "rank the unadjusted price within the point-in-time benchmark universe",
        "momentum_250_20 * cross_section_rank(close)",
        "China momentum evidence is stronger among relatively high-priced stocks.",
    ),
    FactorDefinition(
        "overnight_intraday_divergence_20",
        1,
        21,
        "a3-v2",
        ("adjusted_open", "adjusted_close"),
        _PRICE_AVAILABILITY,
        "trading",
        "require adjusted prior close, open and close for twenty sessions",
        "mean(log(open / close_lag1) - log(close / open), 20)",
        "Persistent overnight demand relative to intraday demand may continue in China.",
    ),
    FactorDefinition(
        "limit_down_close_rate_20",
        1,
        20,
        "a3-v1",
        ("low", "close", "down_limit"),
        _PRICE_AVAILABILITY,
        "price_limit",
        "require twenty sessions with observed low, close and lower price limit",
        "mean(1(close <= down_limit), 20)",
        "Repeated lower-limit closes may subsequently reverse after forced selling.",
    ),
    FactorDefinition(
        "failed_limit_down_rate_20",
        1,
        20,
        "a3-v1",
        ("low", "close", "down_limit"),
        _PRICE_AVAILABILITY,
        "price_limit",
        "require twenty sessions with observed low, close and lower price limit",
        "mean(1(low <= down_limit and close > down_limit), 20)",
        "Recovery from an intraday lower-limit hit may reveal buying absorption.",
    ),
)


FACTOR_NAMES: tuple[str, ...] = tuple(definition.name for definition in FACTOR_CATALOG)
DIRECTIONS: dict[str, int] = {
    definition.name: definition.direction for definition in FACTOR_CATALOG
}
FAMILIES: dict[str, str] = {
    definition.name: definition.family for definition in FACTOR_CATALOG
}

A2_DAILY_FACTOR_NAMES: tuple[str, ...] = tuple(
    definition.name for definition in A2_DAILY_FACTOR_CATALOG
)
A2_DAILY_DIRECTIONS: dict[str, int] = {
    definition.name: definition.direction for definition in A2_DAILY_FACTOR_CATALOG
}
A2_DAILY_FAMILIES: dict[str, str] = {
    definition.name: definition.family for definition in A2_DAILY_FACTOR_CATALOG
}
ALL_DAILY_FACTOR_CATALOG: tuple[FactorDefinition, ...] = (
    *FACTOR_CATALOG,
    *A2_DAILY_FACTOR_CATALOG,
)
ALL_DAILY_FACTOR_NAMES: tuple[str, ...] = (
    *FACTOR_NAMES,
    *A2_DAILY_FACTOR_NAMES,
)
ALL_DAILY_DIRECTIONS: dict[str, int] = {
    **DIRECTIONS,
    **A2_DAILY_DIRECTIONS,
}
ALL_DAILY_FAMILIES: dict[str, str] = {
    **FAMILIES,
    **A2_DAILY_FAMILIES,
}

A3_DAILY_FACTOR_NAMES: tuple[str, ...] = tuple(
    definition.name for definition in A3_DAILY_FACTOR_CATALOG
)
A3_DAILY_DIRECTIONS: dict[str, int] = {
    definition.name: definition.direction for definition in A3_DAILY_FACTOR_CATALOG
}
A3_DAILY_FAMILIES: dict[str, str] = {
    definition.name: definition.family for definition in A3_DAILY_FACTOR_CATALOG
}
A3_ALL_DAILY_FACTOR_CATALOG: tuple[FactorDefinition, ...] = (
    *ALL_DAILY_FACTOR_CATALOG,
    *A3_DAILY_FACTOR_CATALOG,
)
A3_ALL_DAILY_FACTOR_NAMES: tuple[str, ...] = (
    *ALL_DAILY_FACTOR_NAMES,
    *A3_DAILY_FACTOR_NAMES,
)
A3_ALL_DAILY_DIRECTIONS: dict[str, int] = {
    **ALL_DAILY_DIRECTIONS,
    **A3_DAILY_DIRECTIONS,
}
A3_ALL_DAILY_FAMILIES: dict[str, str] = {
    **ALL_DAILY_FAMILIES,
    **A3_DAILY_FAMILIES,
}
