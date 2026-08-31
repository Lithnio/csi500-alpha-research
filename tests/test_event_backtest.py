import numpy as np
import pandas as pd
import pytest

from csi500_alpha.config import OptimizerSettings, ResearchSettings, RiskSettings
from csi500_alpha.execution.backtest import (
    SmokeEventBacktester,
    calculate_backtest_metrics,
    enrich_active_performance,
)
from csi500_alpha.execution.liquidity import LiquiditySnapshot
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.risk.model import LedoitWolfRiskModel, RiskEstimate


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
                    "amount_cny": 1_000_000.0,
                }
            )
    panel = pd.DataFrame(rows)
    index_bars = pd.DataFrame(
        {
            "trade_date": dates,
            "index_code": "000905.SH",
            "benchmark_close": np.arange(100.0, 108.0),
        }
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
    expected_active_nav = (
        result.daily["nav"] / result.daily["nav"].iloc[0]
    ) / (
        result.daily["benchmark_nav"] / result.daily["benchmark_nav"].iloc[0]
    )
    np.testing.assert_allclose(result.daily["active_nav"], expected_active_nav)
    np.testing.assert_allclose(
        result.daily["active_return"],
        expected_active_nav.pct_change().fillna(0.0),
    )
    assert result.metrics["max_drawdown"] == result.metrics["active_max_drawdown"]
    assert "portfolio_max_drawdown" in result.metrics
    assert "capm_alpha_annualized" in result.metrics
    assert "capm_beta" in result.metrics


def test_active_metrics_recover_known_capm_and_relative_wealth() -> None:
    benchmark_returns = np.resize(np.array([-0.01, -0.004, 0.003, 0.012]), 100)
    alpha_daily = 0.0002
    beta = 0.8
    portfolio_returns = alpha_daily + beta * benchmark_returns
    benchmark_nav = np.r_[1.0, np.cumprod(1.0 + benchmark_returns)]
    portfolio_nav = np.r_[1.0, np.cumprod(1.0 + portfolio_returns)]
    daily = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2025-01-02", periods=101).strftime(
                "%Y%m%d"
            ),
            "nav": portfolio_nav,
            "benchmark_nav": benchmark_nav,
            "turnover": 0.0,
        }
    )

    enriched = enrich_active_performance(daily)
    metrics = calculate_backtest_metrics(enriched, pd.DataFrame())

    expected_active_growth = portfolio_nav[-1] / benchmark_nav[-1]
    np.testing.assert_allclose(
        metrics["relative_active_total_return"],
        expected_active_growth - 1.0,
    )
    assert np.isclose(metrics["capm_alpha_annualized"], alpha_daily * 252.0)
    assert np.isclose(metrics["capm_beta"], beta)
    assert np.isclose(
        metrics["rolling_beta_abs_deviation_p95"],
        1.0 - beta,
        atol=3e-6,
    )


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
    liquidity = LiquiditySnapshot(
        reference_date="20250102",
        adv_cny=pd.Series({"A": 1_000.0}),
        observation_count=pd.Series({"A": 2}),
    )

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
        liquidity=liquidity,
    )

    assert records[0]["status"] == "partial"
    assert records[0]["reason"] == "ex_ante_adv_participation_cap"
    assert np.isclose(records[0]["gross_value"], 0.01)
    assert records[0]["impact_cost"] > 0
    assert records[0]["liquidity_reference_date"] == "20250102"
    assert records[0]["adv_cny"] == 1_000.0
    assert records[0]["adv_observations"] == 2
    assert records[0]["executed_adv_participation"] == 0.10
    assert cost > records[0]["linear_cost"]
    assert cash >= 0.0


def test_execution_day_volume_cannot_change_open_fill_or_impact() -> None:
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
    backtester = SmokeEventBacktester(
        _settings(),
        risk_model=object(),  # type: ignore[arg-type]
        optimizer=ActivePortfolioOptimizer(
            optimizer_settings,
            _settings(),
            RiskSettings(
                lookback=2,
                min_history=2,
                annualization=252,
                missing_annual_volatility=0.8,
                variance_floor=1e-8,
                return_clip=0.2,
            ),
        ),
    )
    liquidity = LiquiditySnapshot(
        reference_date="20250102",
        adv_cny=pd.Series({"A": 1_000.0}),
        observation_count=pd.Series({"A": 2}),
    )

    def execute(
        execution_day_amount: float,
        snapshot: LiquiditySnapshot = liquidity,
    ) -> tuple[float, float, dict[str, object]]:
        day = pd.DataFrame(
            {
                "instrument": ["A"],
                "open": [10.0],
                "adjusted_open": [10.0],
                "up_limit": [11.0],
                "down_limit": [9.0],
                "amount_cny": [execution_day_amount],
            }
        ).set_index("instrument", drop=False)
        cash, _, cost, records = backtester._execute_target(
            signal_date="20250102",
            trade_date="20250103",
            target=pd.Series({"A": 1.0}),
            holdings={},
            cash=1.0,
            pre_nav=1.0,
            day=day,
            last_close={},
            suspended_instruments=set(),
            liquidity=snapshot,
        )
        return cash, cost, records[0]

    low_volume = execute(100.0)
    high_volume = execute(1_000_000.0)

    assert low_volume[0] == high_volume[0]
    assert low_volume[1] == high_volume[1]
    assert low_volume[2]["gross_value"] == high_volume[2]["gross_value"]
    assert low_volume[2]["impact_cost"] == high_volume[2]["impact_cost"]
    assert low_volume[2]["realized_day_participation"] != high_volume[2][
        "realized_day_participation"
    ]
    with pytest.raises(ValueError, match="must predate the execution session"):
        execute(
            100.0,
            LiquiditySnapshot(
                reference_date="20250103",
                adv_cny=pd.Series({"A": 1_000.0}),
                observation_count=pd.Series({"A": 2}),
            ),
        )


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
        liquidity=None,
    )

    assert context.exposures is not None
    assert context.exposures.loc["CURRENT", "__unknown_exposure__"] == 0.0
    assert context.exposures.loc["EXITED", "__unknown_exposure__"] == 1.0
    assert context.exposures.loc["EXITED", "industry_A"] == 0.0


def test_optimized_backtest_emits_realized_constraint_audit() -> None:
    inputs = _synthetic_inputs()
    signal_date = str(inputs["calendar"]["trade_date"].iloc[2])
    execution_date = str(inputs["calendar"]["trade_date"].iloc[3])
    inputs["signals"]["expected_return"] = (
        inputs["signals"]["score"].astype(float) * 0.02 - 0.01
    )
    optimizer_settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.1,
        risk_horizon_days=5,
        l2_penalty=0.001,
        active_cap=0.40,
        name_cap=0.90,
        turnover_cap=0.50,
        initial_turnover_cap=1.0,
        exposure_cap=0.20,
        solvers=("CLARABEL",),
        beta_constraint_enabled=True,
        beta_active_cap=0.05,
        liquidity_enabled=True,
        portfolio_aum_cny=100_000.0,
        adv_lookback=2,
        min_adv_observations=1,
        max_adv_participation=0.10,
        impact_bps_at_max_participation=10.0,
    )
    risk_settings = RiskSettings(
        lookback=2,
        min_history=1,
        annualization=252,
        missing_annual_volatility=0.8,
        variance_floor=1e-8,
        return_clip=0.2,
    )
    risk_model = LedoitWolfRiskModel(
        risk_settings,
        inputs["market_panel"],
        inputs["calendar"]["trade_date"].astype(str).tolist(),
    )
    optimizer = ActivePortfolioOptimizer(
        optimizer_settings,
        _settings(),
        risk_settings,
    )
    exposures = pd.DataFrame(
        {
            "trade_date": [signal_date, signal_date],
            "instrument": ["A", "B"],
            "industry_A": [1.0, 0.0],
            "industry_B": [0.0, 1.0],
        }
    )
    styles = pd.DataFrame(
        {
            "trade_date": [signal_date, signal_date],
            "instrument": ["A", "B"],
            "market_beta_60": [1.2, 0.8],
            "small_size_z": [1.0, -1.0],
        }
    )

    result = SmokeEventBacktester(
        _settings(),
        risk_model=risk_model,
        optimizer=optimizer,
    ).run(
        **inputs,
        portfolio_exposures=exposures,
        portfolio_styles=styles,
        start_date=str(inputs["calendar"]["trade_date"].iloc[0]),
        end_date=str(inputs["calendar"]["trade_date"].iloc[-1]),
        rebalance_dates=[signal_date],
    )

    assert len(result.constraint_audits) == 1
    audit = result.constraint_audits.iloc[0]
    assert audit["execution_date"] == execution_date
    assert bool(audit["beta_audit_complete"])
    assert abs(float(audit["actual_active_beta"])) <= 0.05 + 1e-6
    assert not bool(audit["has_policy_violation"])
    assert set(result.positions["instrument"]) == {"A", "B"}
    assert {
        "actual_weight",
        "benchmark_weight",
        "active_weight",
        "target_deviation",
        "market_beta_60",
    }.issubset(result.positions.columns)
    assert result.metrics["post_trade_audit_count"] == 1
    assert result.metrics["beta_audit_complete_fraction"] == 1.0
    assert result.trades["liquidity_reference_date"].dropna().unique().tolist() == [
        signal_date
    ]
    assert result.trades["executed_adv_participation"].max() <= 0.10 + 1e-12
    assert audit["maximum_ex_ante_adv_participation"] <= 0.10 + 1e-12


def test_backtester_uses_risk_model_beta_for_optimization_and_audit() -> None:
    inputs = _synthetic_inputs()
    signal_date = str(inputs["calendar"]["trade_date"].iloc[2])
    inputs["signals"]["expected_return"] = np.where(
        inputs["signals"]["instrument"].eq("A"),
        0.05,
        0.0,
    )
    risk_settings = RiskSettings(
        lookback=2,
        min_history=1,
        annualization=252,
        missing_annual_volatility=0.8,
        variance_floor=1e-8,
        return_clip=0.2,
    )
    optimizer = ActivePortfolioOptimizer(
        OptimizerSettings(
            enabled=True,
            risk_aversion=0.01,
            risk_horizon_days=5,
            l2_penalty=0.0,
            active_cap=0.40,
            name_cap=0.90,
            turnover_cap=1.0,
            initial_turnover_cap=1.0,
            exposure_cap=0.20,
            solvers=("CLARABEL",),
            beta_constraint_enabled=True,
            beta_active_cap=0.01,
        ),
        _settings(),
        risk_settings,
    )

    class SyntheticRiskModel:
        def estimate(self, as_of_date: str, instruments: list[str]) -> RiskEstimate:
            names = pd.Index(sorted(instruments), name="instrument")
            return RiskEstimate(
                as_of_date=as_of_date,
                covariance=pd.DataFrame(
                    np.eye(len(names)) * 0.0001,
                    index=names,
                    columns=names,
                ),
                eligible=pd.Series(True, index=names),
                method="synthetic_factor_risk",
                observations=20,
                market_beta=pd.Series({"A": 1.5, "B": 0.5}).reindex(names),
                beta_method="synthetic_shrunk_beta",
                diagnostics={"factor_count": 3, "factor_model_fallback": False},
            )

    styles = pd.DataFrame(
        {
            "trade_date": [signal_date, signal_date],
            "instrument": ["A", "B"],
            "market_beta_60": [1.0, 1.0],
            "small_size_z": [1.0, -1.0],
        }
    )
    result = SmokeEventBacktester(
        _settings(),
        risk_model=SyntheticRiskModel(),
        optimizer=optimizer,
    ).run(
        **inputs,
        portfolio_styles=styles,
        start_date=str(inputs["calendar"]["trade_date"].iloc[0]),
        end_date=str(inputs["calendar"]["trade_date"].iloc[-1]),
        rebalance_dates=[signal_date],
    )

    target = result.targets.set_index("instrument")["target_weight"]
    assert target["A"] <= 0.51 + 1e-6
    positions = result.positions.set_index("instrument")
    assert positions.loc["A", "market_beta_60"] == 1.5
    assert positions.loc["A", "market_beta_raw_60"] == 1.0
    audit = result.constraint_audits.iloc[0]
    assert audit["risk_method"] == "synthetic_factor_risk"
    assert audit["beta_method"] == "synthetic_shrunk_beta"
    optimization = result.optimization.iloc[0]
    assert optimization["risk_factor_count"] == 3
    assert not bool(optimization["risk_factor_model_fallback"])
