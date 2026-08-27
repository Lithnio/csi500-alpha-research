import numpy as np
import pandas as pd

from csi500_alpha.config import OptimizerSettings, ResearchSettings, RiskSettings
from csi500_alpha.execution.backtest import SmokeEventBacktester
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer


def _settings() -> ResearchSettings:
    return ResearchSettings(
        factor_window=1,
        rebalance_every=3,
        top_n=1,
        initial_cash=1.0,
        linear_cost_bps=5.0,
        stamp_duty_change_date="20230828",
        stamp_duty_before=0.001,
        stamp_duty_after=0.0005,
        price_limit_tolerance=1e-6,
    )


def _synthetic_inputs(block_first_buy: bool = False) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=8).strftime("%Y%m%d").tolist()
    calendar = pd.DataFrame(
        {"trade_date": dates, "is_open": 1, "prev_trade_date": [""] + dates[:-1]}
    )
    weights = pd.DataFrame(
        {
            "snapshot_date": ["20241231", "20241231"],
            "instrument": ["A", "B"],
            "weight": [0.5, 0.5],
        }
    )
    rows = []
    for position, date in enumerate(dates):
        for instrument, base in (("A", 10.0), ("B", 20.0)):
            price = base + position
            up_limit = price * 1.1
            if block_first_buy and date == dates[1] and instrument == "A":
                up_limit = price
            rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "open": price,
                    "close": price,
                    "adjusted_open": price,
                    "adjusted_close": price,
                    "up_limit": up_limit,
                    "down_limit": price * 0.9,
                }
            )
    panel = pd.DataFrame(rows)
    index_bars = pd.DataFrame(
        {"trade_date": dates, "index_code": "000905.SH", "close": np.arange(100.0, 108.0)}
    )
    score_rows = []
    for position, date in enumerate(dates):
        scores = {"A": 1.0, "B": 0.0} if position < 3 else {"A": 0.0, "B": 1.0}
        score_rows.extend(
            {"trade_date": date, "instrument": instrument, "score": score}
            for instrument, score in scores.items()
        )
    return {
        "calendar": calendar,
        "benchmark_weights": weights,
        "index_bars": index_bars,
        "market_panel": panel,
        "signals": pd.DataFrame(score_rows),
    }


def test_signal_executes_next_open_and_sell_pays_stamp_duty() -> None:
    inputs = _synthetic_inputs()
    result = SmokeEventBacktester(_settings()).run(
        **inputs,
        start_date=inputs["calendar"]["trade_date"].iloc[0],
        end_date=inputs["calendar"]["trade_date"].iloc[-1],
    )
    executed = result.trades[result.trades["status"].isin(["filled", "partial"])]
    first_trade = executed.iloc[0]
    assert first_trade["signal_date"] == inputs["calendar"]["trade_date"].iloc[0]
    assert first_trade["trade_date"] == inputs["calendar"]["trade_date"].iloc[1]
    sells = executed[executed["side"] == "sell"]
    assert not sells.empty
    assert np.isclose(
        sells["stamp_duty"].iloc[0], sells["gross_value"].iloc[0] * 0.0005
    )
    assert result.daily["cash"].min() >= 0.0


def test_open_at_up_limit_blocks_buy() -> None:
    inputs = _synthetic_inputs(block_first_buy=True)
    result = SmokeEventBacktester(_settings()).run(
        **inputs,
        start_date=inputs["calendar"]["trade_date"].iloc[0],
        end_date=inputs["calendar"]["trade_date"].iloc[-1],
    )
    blocked = result.trades[
        (result.trades["status"] == "blocked") & (result.trades["side"] == "buy")
    ]
    assert not blocked.empty
    assert blocked.iloc[0]["reason"] == "open_at_up_limit"


def test_nominal_limit_is_ignored_when_open_proves_it_inapplicable() -> None:
    inputs = _synthetic_inputs(block_first_buy=True)
    first_execution = inputs["calendar"]["trade_date"].iloc[1]
    special = (
        inputs["market_panel"]["trade_date"].eq(first_execution)
        & inputs["market_panel"]["instrument"].eq("A")
    )
    inputs["market_panel"].loc[special, "price_limit_applicable"] = False

    result = SmokeEventBacktester(_settings()).run(
        **inputs,
        start_date=inputs["calendar"]["trade_date"].iloc[0],
        end_date=inputs["calendar"]["trade_date"].iloc[-1],
    )

    first_a = result.trades[
        (result.trades["trade_date"] == first_execution)
        & (result.trades["instrument"] == "A")
    ]
    assert not first_a.empty
    assert not (first_a["reason"] == "open_at_up_limit").any()


def test_explicit_suspension_is_distinct_from_missing_market_data() -> None:
    inputs = _synthetic_inputs()
    first_execution = inputs["calendar"]["trade_date"].iloc[1]
    result = SmokeEventBacktester(_settings()).run(
        **inputs,
        suspensions=pd.DataFrame(
            {
                "trade_date": [first_execution],
                "instrument": ["A"],
                "suspend_type": ["S"],
            }
        ),
        start_date=inputs["calendar"]["trade_date"].iloc[0],
        end_date=inputs["calendar"]["trade_date"].iloc[-1],
    )

    blocked = result.trades[
        (result.trades["trade_date"] == first_execution)
        & (result.trades["instrument"] == "A")
    ]
    assert blocked.iloc[0]["status"] == "blocked"
    assert blocked.iloc[0]["reason"] == "suspended"


def test_later_intraday_suspension_does_not_block_open_execution() -> None:
    inputs = _synthetic_inputs()
    first_execution = inputs["calendar"]["trade_date"].iloc[1]
    result = SmokeEventBacktester(_settings()).run(
        **inputs,
        suspensions=pd.DataFrame(
            {
                "trade_date": [first_execution, first_execution],
                "instrument": ["A", "B"],
                "suspend_timing": ["10:07-10:17", None],
                "suspend_type": ["S", "R"],
            }
        ),
        start_date=inputs["calendar"]["trade_date"].iloc[0],
        end_date=inputs["calendar"]["trade_date"].iloc[-1],
    )

    first_a = result.trades[
        (result.trades["trade_date"] == first_execution)
        & (result.trades["instrument"] == "A")
    ]
    assert not first_a.empty
    assert not (first_a["reason"] == "suspended").any()


def test_execution_capacity_creates_partial_fill_and_impact_cost() -> None:
    optimizer_settings = OptimizerSettings(
        enabled=True,
        risk_aversion=1.0,
        risk_horizon_days=5,
        l2_penalty=0.01,
        active_cap=0.5,
        name_cap=1.0,
        turnover_cap=1.0,
        initial_turnover_cap=1.0,
        exposure_cap=0.1,
        solvers=("CLARABEL",),
        liquidity_enabled=True,
        portfolio_aum_cny=10_000.0,
        adv_lookback=2,
        min_adv_observations=1,
        max_adv_participation=0.10,
        impact_bps_at_max_participation=10.0,
    )
    risk_settings = RiskSettings(
        lookback=2,
        min_history=2,
        annualization=252,
        missing_annual_volatility=0.8,
        variance_floor=1e-8,
        return_clip=0.2,
    )
    optimizer = ActivePortfolioOptimizer(
        optimizer_settings,
        _settings(),
        risk_settings,
    )
    backtester = SmokeEventBacktester(
        _settings(),
        risk_model=object(),  # type: ignore[arg-type]
        optimizer=optimizer,
    )
    day = pd.DataFrame(
        {
            "instrument": ["A"],
            "open": [10.0],
            "adjusted_open": [10.0],
            "up_limit": [11.0],
            "down_limit": [9.0],
            "amount_cny": [1_000.0],
        }
    ).set_index("instrument", drop=False)
    holdings: dict[str, float] = {}

    cash, _, cost, records = backtester._execute_target(
        signal_date="20250102",
        trade_date="20250103",
        target=pd.Series({"A": 1.0}),
        holdings=holdings,
        cash=1.0,
        pre_nav=1.0,
        day=day,
        last_close={},
        suspended_instruments=set(),
    )

    assert records[0]["status"] == "partial"
    assert records[0]["reason"] == "volume_participation_cap"
    assert np.isclose(records[0]["gross_value"], 0.01)
    assert records[0]["impact_cost"] > 0
    assert cost > records[0]["linear_cost"]
    assert cash >= 0.0


def test_carried_name_without_current_exposure_uses_unknown_bucket() -> None:
    backtester = SmokeEventBacktester(_settings())
    context = backtester._portfolio_context(
        decision_date="20250102",
        names=["CURRENT", "EXITED"],
        pre_cash_weight=0.0,
        exposures=pd.DataFrame(
            {"industry_A": [1.0]},
            index=pd.Index(["CURRENT"], name="instrument"),
        ),
        restrictions=None,
        liquidity_caps=None,
    )

    assert context.exposures is not None
    assert context.exposures.loc["CURRENT", "__unknown_exposure__"] == 0.0
    assert context.exposures.loc["EXITED", "__unknown_exposure__"] == 1.0
    assert context.exposures.loc["EXITED", "industry_A"] == 0.0
