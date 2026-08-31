from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from csi500_alpha.config import OptimizerSettings, ResearchSettings, RiskSettings
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.risk.model import (
    FactorEWMARiskModel,
    LedoitWolfRiskModel,
    RiskEstimate,
    build_risk_model,
)


def _research_settings() -> ResearchSettings:
    return ResearchSettings(
        factor_window=5,
        rebalance_every=5,
        top_n=30,
        initial_cash=1.0,
        linear_cost_bps=5.0,
        stamp_duty_change_date="20230828",
        stamp_duty_before=0.001,
        stamp_duty_after=0.0005,
        price_limit_tolerance=1e-6,
    )


def _risk_settings() -> RiskSettings:
    return RiskSettings(
        lookback=8,
        min_history=5,
        annualization=252,
        missing_annual_volatility=0.8,
        variance_floor=1e-8,
        return_clip=0.2,
    )


def test_risk_estimate_does_not_use_prices_after_as_of_date() -> None:
    dates = pd.bdate_range("2025-01-02", periods=15).strftime("%Y%m%d").tolist()
    rows = [
        {
            "trade_date": date,
            "instrument": instrument,
            "adjusted_close": base * (1.0 + 0.01 * position),
        }
        for position, date in enumerate(dates)
        for instrument, base in (("A", 10.0), ("B", 20.0), ("C", 30.0))
    ]
    panel = pd.DataFrame(rows)
    changed = panel.copy()
    changed.loc[changed["trade_date"] > dates[9], "adjusted_close"] *= 10.0
    original_model = LedoitWolfRiskModel(_risk_settings(), panel, dates)
    changed_model = LedoitWolfRiskModel(_risk_settings(), changed, dates)

    original = original_model.estimate(dates[9], ["A", "B", "C"])
    revised = changed_model.estimate(dates[9], ["A", "B", "C"])

    np.testing.assert_allclose(original.covariance, revised.covariance)
    assert original.as_of_date == dates[9]


def _factor_risk_inputs() -> tuple[
    RiskSettings,
    list[str],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2024-01-02", periods=90).strftime("%Y%m%d").tolist()
    names = [f"S{position:03d}" for position in range(24)]
    market_returns = rng.normal(0.0002, 0.012, len(dates))
    true_beta = np.linspace(0.55, 1.45, len(names))
    residual_returns = rng.normal(0.0, 0.009, (len(dates), len(names)))
    stock_returns = market_returns[:, None] * true_beta + residual_returns
    stock_prices = 100.0 * np.cumprod(1.0 + stock_returns, axis=0)
    panel = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, len(names)),
            "instrument": np.tile(names, len(dates)),
            "adjusted_close": stock_prices.reshape(-1),
        }
    )
    index_bars = pd.DataFrame(
        {
            "trade_date": dates,
            "benchmark_close": 100.0 * np.cumprod(1.0 + market_returns),
        }
    )
    exposure_date = dates[60]
    industry = pd.DataFrame(
        {
            "trade_date": exposure_date,
            "instrument": names,
            **{
                f"industry_{group}": [
                    float(position % 3 == group) for position in range(len(names))
                ]
                for group in range(3)
            },
        }
    )
    styles = pd.DataFrame(
        {
            "trade_date": exposure_date,
            "instrument": names,
            "market_beta_60": true_beta,
            "small_size_z": np.linspace(-2.0, 2.0, len(names)),
            "momentum_120_20_z": rng.normal(size=len(names)),
            "low_idio_volatility_z": rng.normal(size=len(names)),
            "turnover_z": rng.normal(size=len(names)),
            "value_z": rng.normal(size=len(names)),
        }
    )
    settings = RiskSettings(
        lookback=40,
        min_history=30,
        annualization=252,
        missing_annual_volatility=0.8,
        variance_floor=1e-8,
        return_clip=0.2,
        model="factor_ewma",
        beta_model="ewma_shrunk",
        min_factor_cross_section=15,
        beta_lookback=40,
        beta_min_history=25,
        beta_half_life=20.0,
    )
    return settings, dates, names, panel, index_bars, industry, styles


def test_factor_risk_model_is_psd_and_produces_shrunk_ewma_beta() -> None:
    settings, dates, names, panel, index_bars, industry, styles = _factor_risk_inputs()
    model = FactorEWMARiskModel(
        settings,
        panel,
        dates,
        index_bars=index_bars,
        industry_exposures=industry,
        style_exposures=styles,
    )

    estimate = model.estimate(dates[60], names)

    assert estimate.method == "factor_ewma"
    assert estimate.beta_method == "ewma_shrunk_to_one"
    assert estimate.market_beta is not None
    assert estimate.market_beta.notna().all()
    assert estimate.market_beta.iloc[0] < estimate.market_beta.iloc[-1]
    assert not bool(estimate.diagnostics["factor_model_fallback"])
    assert int(estimate.diagnostics["factor_count"]) >= 9
    assert 0.0 <= float(estimate.diagnostics["beta_clip_fraction"]) <= 1.0
    assert np.linalg.eigvalsh(estimate.covariance.to_numpy()).min() >= -1e-12


def test_factor_risk_model_does_not_use_future_prices_index_or_exposures() -> None:
    settings, dates, names, panel, index_bars, industry, styles = _factor_risk_inputs()
    as_of_date = dates[60]
    future_industry = industry.assign(trade_date=dates[-1]).copy()
    future_styles = styles.assign(trade_date=dates[-1]).copy()
    original_industry = pd.concat([industry, future_industry], ignore_index=True)
    original_styles = pd.concat([styles, future_styles], ignore_index=True)

    changed_panel = panel.copy()
    changed_panel.loc[changed_panel["trade_date"] > as_of_date, "adjusted_close"] *= 4.0
    changed_index = index_bars.copy()
    changed_index.loc[changed_index["trade_date"] > as_of_date, "benchmark_close"] *= 3.0
    changed_industry = original_industry.copy()
    changed_industry.loc[
        changed_industry["trade_date"] > as_of_date,
        [column for column in changed_industry if column.startswith("industry_")],
    ] = 0.0
    changed_styles = original_styles.copy()
    changed_styles.loc[
        changed_styles["trade_date"] > as_of_date,
        [
            column
            for column in changed_styles
            if column not in {"trade_date", "instrument"}
        ],
    ] = 99.0

    original = FactorEWMARiskModel(
        settings,
        panel,
        dates,
        index_bars=index_bars,
        industry_exposures=original_industry,
        style_exposures=original_styles,
    ).estimate(as_of_date, names)
    changed = FactorEWMARiskModel(
        settings,
        changed_panel,
        dates,
        index_bars=changed_index,
        industry_exposures=changed_industry,
        style_exposures=changed_styles,
    ).estimate(as_of_date, names)

    np.testing.assert_allclose(original.covariance, changed.covariance)
    assert original.market_beta is not None and changed.market_beta is not None
    np.testing.assert_allclose(original.market_beta, changed.market_beta)


def test_factor_risk_factory_requires_point_in_time_context() -> None:
    settings, dates, _names, panel, index_bars, industry, styles = _factor_risk_inputs()
    model = build_risk_model(
        settings,
        panel,
        dates,
        index_bars=index_bars,
        industry_exposures=industry,
        style_exposures=styles,
    )
    assert isinstance(model, FactorEWMARiskModel)

    beta_only = build_risk_model(
        replace(settings, model="ledoit_wolf"),
        panel,
        dates,
        index_bars=index_bars,
    ).estimate(dates[60], _names)
    assert beta_only.method == "ledoit_wolf"
    assert beta_only.beta_method == "ewma_shrunk_to_one"
    assert beta_only.market_beta is not None

    factor_only = build_risk_model(
        replace(settings, beta_model="feature_60"),
        panel,
        dates,
        index_bars=index_bars,
        industry_exposures=industry,
        style_exposures=styles,
    ).estimate(dates[60], _names)
    assert factor_only.method == "factor_ewma"
    assert factor_only.beta_method == "feature_beta_60"
    assert factor_only.market_beta is None

    with pytest.raises(ValueError, match="requires point-in-time context"):
        build_risk_model(settings, panel, dates, index_bars=index_bars)


def test_ewma_beta_rejects_forward_filled_sparse_price_history() -> None:
    settings, dates, names, panel, index_bars, _industry, _styles = (
        _factor_risk_inputs()
    )
    sparse_name = names[0]
    sparse_dates = set(dates[:5])
    sparse_panel = panel.loc[
        panel["instrument"].ne(sparse_name)
        | panel["trade_date"].isin(sparse_dates)
    ].copy()
    estimate = LedoitWolfRiskModel(
        replace(settings, model="ledoit_wolf"),
        sparse_panel,
        dates,
        index_bars=index_bars,
    ).estimate(dates[60], names)

    assert estimate.market_beta is not None
    assert estimate.market_beta.loc[sparse_name] == 1.0
    assert estimate.diagnostics["beta_observed_fraction"] < 1.0


def test_optimizer_respects_budget_active_cap_and_missing_signal_buy_block() -> None:
    names = pd.Index(["A", "B", "C"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(3) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.1,
        risk_horizon_days=5,
        l2_penalty=0.001,
        active_cap=0.10,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL",),
    )
    optimizer = ActivePortfolioOptimizer(settings, _research_settings(), _risk_settings())
    benchmark = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": 0.05, "B": 0.0, "C": np.nan}),
        benchmark=benchmark,
        pre_weights=benchmark,
        pre_cash_weight=0.0,
        risk_estimate=risk,
    )

    assert result.target is not None
    target = result.target
    assert np.isclose(target.sum(), 1.0, atol=1e-7)
    assert (target >= -1e-8).all()
    assert (target - benchmark).abs().max() <= settings.active_cap + 1e-6
    assert target["A"] > benchmark["A"]
    assert target["C"] <= benchmark["C"] + 1e-7
    assert result.diagnostics["maximum_violation"] <= 1e-6


def test_optimizer_maps_a_globally_inactive_alpha_to_benchmark() -> None:
    names = pd.Index(["A", "B", "C"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(3) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=1.0,
        risk_horizon_days=5,
        l2_penalty=0.01,
        active_cap=0.10,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL",),
    )
    optimizer = ActivePortfolioOptimizer(
        settings,
        _research_settings(),
        _risk_settings(),
    )
    benchmark = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})

    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series(np.nan, index=names),
        benchmark=benchmark,
        pre_weights=pd.Series(0.0, index=names),
        pre_cash_weight=1.0,
        risk_estimate=risk,
    )

    assert result.target is not None
    pd.testing.assert_series_equal(
        result.target,
        benchmark.rename("target_weight"),
        check_exact=False,
        atol=1e-6,
        rtol=0.0,
        check_names=False,
    )
    assert result.diagnostics["alpha_state"] == "inactive_zero_alpha"
    assert result.diagnostics["active_eligible"] == len(names)


def test_optimizer_enforces_exposure_freeze_and_liquidity_constraints() -> None:
    names = pd.Index(["A", "B", "C", "D"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(4) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.1,
        risk_horizon_days=5,
        l2_penalty=0.001,
        active_cap=0.20,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.01,
        solvers=("CLARABEL",),
    )
    optimizer = ActivePortfolioOptimizer(settings, _research_settings(), _risk_settings())
    benchmark = pd.Series(0.25, index=names)
    exposures = pd.DataFrame(
        {
            "sector_1": [1.0, 1.0, 0.0, np.nan],
            "sector_2": [0.0, 0.0, 1.0, np.nan],
        },
        index=names,
    )
    trade_caps = pd.Series(0.03, index=names)
    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": 0.05, "B": -0.02, "C": 0.01, "D": 0.03}),
        benchmark=benchmark,
        pre_weights=benchmark,
        pre_cash_weight=0.0,
        risk_estimate=risk,
        exposures=exposures,
        cannot_buy={"A"},
        cannot_sell={"B"},
        max_trade_weights=trade_caps,
    )

    assert result.target is not None
    active = result.target - benchmark
    assert result.target["A"] <= benchmark["A"] + 1e-7
    assert result.target["B"] >= benchmark["B"] - 1e-7
    assert np.isclose(active["D"], 0.0, atol=1e-7)
    assert active.abs().max() <= 0.03 + 1e-7
    assert np.abs(exposures.fillna(0.0).T @ active).max() <= 0.01 + 1e-7
    assert result.diagnostics["maximum_violation"] <= settings.feasibility_tolerance


def test_optimizer_enforces_active_beta_cap() -> None:
    names = pd.Index(["A", "B"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(2) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.0,
        risk_horizon_days=5,
        l2_penalty=0.0,
        active_cap=0.20,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL",),
        beta_constraint_enabled=True,
        beta_active_cap=0.02,
    )
    optimizer = ActivePortfolioOptimizer(
        settings,
        _research_settings(),
        _risk_settings(),
    )
    benchmark = pd.Series({"A": 0.5, "B": 0.5})

    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": 0.10, "B": -0.10}),
        benchmark=benchmark,
        pre_weights=benchmark,
        pre_cash_weight=0.0,
        risk_estimate=risk,
        market_beta=pd.Series({"A": 1.5, "B": 0.5}),
    )

    assert result.target is not None
    active_beta = float(
        pd.Series({"A": 1.5, "B": 0.5}) @ (result.target - benchmark)
    )
    assert abs(active_beta) <= settings.beta_active_cap + 1e-7
    assert result.diagnostics["target_active_beta"] == pytest.approx(active_beta)
    assert (
        result.diagnostics["configured_beta_cap_breach_after"]
        <= settings.feasibility_tolerance
    )


def test_ineligible_name_can_reduce_but_not_increase_under_tight_trade_caps() -> None:
    names = pd.Index(["A", "B", "C"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(3) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.1,
        risk_horizon_days=5,
        l2_penalty=0.001,
        active_cap=0.02,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL", "OSQP"),
    )
    optimizer = ActivePortfolioOptimizer(settings, _research_settings(), _risk_settings())
    benchmark = pd.Series({"A": 0.50, "B": 0.00, "C": 0.50})
    pretrade = pd.Series({"A": 0.45, "B": 0.25, "C": 0.30})

    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": 0.05, "B": np.nan, "C": -0.01}),
        benchmark=benchmark,
        pre_weights=pretrade,
        pre_cash_weight=0.0,
        risk_estimate=risk,
        max_trade_weights=pd.Series(0.01, index=names),
    )

    assert result.target is not None
    assert result.target["B"] <= pretrade["B"] + 1e-7
    assert result.target["B"] < pretrade["B"] - 1e-4
    assert result.target["A"] <= pretrade["A"] + 0.010001
    assert result.diagnostics["ineligible_position_policy"] == (
        "cannot_exceed_pretrade_or_benchmark_may_reduce"
    )
    assert result.diagnostics["ineligible_held"] == 1
    assert result.diagnostics["preexisting_active_cap_breaches"] == 3
    assert result.diagnostics["maximum_violation"] <= settings.feasibility_tolerance


def test_optimizer_repairs_preexisting_gaps_before_pursuing_alpha() -> None:
    names = pd.Index(["A", "B", "C"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(3) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.0,
        risk_horizon_days=5,
        l2_penalty=0.0,
        active_cap=0.02,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL",),
    )
    optimizer = ActivePortfolioOptimizer(
        settings,
        _research_settings(),
        _risk_settings(),
    )
    benchmark = pd.Series({"A": 0.50, "B": 0.25, "C": 0.25})
    pretrade = pd.Series({"A": 0.30, "B": 0.45, "C": 0.25})

    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": -0.10, "B": 0.10, "C": 0.0}),
        benchmark=benchmark,
        pre_weights=pretrade,
        pre_cash_weight=0.0,
        risk_estimate=risk,
        max_trade_weights=pd.Series(0.05, index=names),
    )

    assert result.target is not None
    assert result.target["A"] > pretrade["A"] + 0.049
    assert result.target["B"] < pretrade["B"] - 0.049
    assert result.diagnostics["constraint_policy"] == (
        "lexicographic_minimum_configured_gap"
    )
    assert result.diagnostics["configured_gap_score_improvement"] > 0.0
    assert {
        attempt["stage"] for attempt in result.diagnostics["attempts"]
    } >= {"strict_alpha", "repair", "alpha_after_repair"}
    assert (
        result.diagnostics["configured_gap_score_after"]
        <= result.diagnostics["repair_gap_budget"]
    )
    assert result.diagnostics["configured_active_cap_breaches_after"] == 2


def test_optimizer_enforces_annualized_tracking_error_budget() -> None:
    names = pd.Index(["A", "B"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(2) * 0.0004, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.0,
        risk_horizon_days=5,
        l2_penalty=0.0,
        active_cap=0.30,
        name_cap=0.90,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.20,
        solvers=("CLARABEL",),
        tracking_error_cap=0.05,
    )
    optimizer = ActivePortfolioOptimizer(
        settings,
        _research_settings(),
        _risk_settings(),
    )
    benchmark = pd.Series({"A": 0.50, "B": 0.50})

    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": 0.20, "B": -0.20}),
        benchmark=benchmark,
        pre_weights=benchmark,
        pre_cash_weight=0.0,
        risk_estimate=risk,
    )

    assert result.target is not None
    assert result.diagnostics["ex_ante_tracking_error"] <= 0.050001
    assert result.diagnostics["tracking_error_utilization"] <= 1.00002
    assert [
        attempt["stage"] for attempt in result.diagnostics["attempts"]
    ] == ["strict_alpha"]
    assert result.diagnostics["repair_status"] == (
        "not_required_strict_feasible"
    )
    assert (
        result.diagnostics["configured_tracking_error_cap_breach_after"]
        <= settings.feasibility_tolerance
    )


def test_optimizer_accepts_repair_score_noise_within_raw_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = pd.Index([f"S{position:02d}" for position in range(40)], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(40) * 1e-6, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=252,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.0,
        risk_horizon_days=5,
        l2_penalty=0.0,
        active_cap=0.01,
        name_cap=0.03,
        turnover_cap=0.35,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL",),
        feasibility_tolerance=1e-6,
    )
    benchmark = pd.Series(1.0 / len(names), index=names)
    expected_returns = pd.Series(0.0, index=names)
    expected_returns.iloc[0] = 0.10
    original_solve = cp.Problem.solve

    def add_small_cap_noise(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        value = original_solve(problem, *args, **kwargs)
        target = next(
            variable
            for variable in problem.variables()
            if variable.name() == "target_weight"
        )
        noisy = np.asarray(target.value, dtype=float).copy()
        adjustment = settings.name_cap + 2e-7 - noisy[0]
        noisy[0] += adjustment
        noisy[1:] -= adjustment / (len(noisy) - 1)
        target.value = noisy
        return float(value)

    monkeypatch.setattr(cp.Problem, "solve", add_small_cap_noise)
    result = ActivePortfolioOptimizer(
        settings,
        _research_settings(),
        _risk_settings(),
    ).solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=expected_returns,
        benchmark=benchmark,
        pre_weights=benchmark,
        pre_cash_weight=0.0,
        risk_estimate=risk,
    )

    assert result.target is not None
    assert result.diagnostics["repair_gap_score_excess"] > settings.feasibility_tolerance
    assert (
        result.diagnostics["repair_gap_equivalent_violation"]
        <= settings.feasibility_tolerance
    )
    assert result.diagnostics["maximum_violation"] <= settings.feasibility_tolerance
    assert result.diagnostics["violations"]["repair_gap_lock"] == pytest.approx(
        result.diagnostics["repair_gap_equivalent_violation"]
    )


def test_optimizer_rejects_constraint_violating_solver_output(monkeypatch: object) -> None:
    names = pd.Index(["A", "B", "C"], name="instrument")
    risk = RiskEstimate(
        as_of_date="20250110",
        covariance=pd.DataFrame(np.eye(3) * 0.0001, index=names, columns=names),
        eligible=pd.Series(True, index=names),
        method="synthetic",
        observations=20,
    )
    settings = OptimizerSettings(
        enabled=True,
        risk_aversion=0.1,
        risk_horizon_days=5,
        l2_penalty=0.001,
        active_cap=0.10,
        name_cap=0.80,
        turnover_cap=0.40,
        initial_turnover_cap=1.0,
        exposure_cap=0.02,
        solvers=("CLARABEL",),
    )
    original_solve = cp.Problem.solve

    def corrupt_solution(problem: cp.Problem, *args: object, **kwargs: object) -> float:
        value = original_solve(problem, *args, **kwargs)
        target = next(
            variable
            for variable in problem.variables()
            if variable.name() == "target_weight"
        )
        target.value = np.array([1.0, 0.0, 0.0])
        return float(value)

    monkeypatch.setattr(cp.Problem, "solve", corrupt_solution)  # type: ignore[attr-defined]
    optimizer = ActivePortfolioOptimizer(settings, _research_settings(), _risk_settings())
    benchmark = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    result = optimizer.solve(
        decision_date="20250110",
        execution_date="20250113",
        expected_returns=pd.Series({"A": 0.05, "B": 0.0, "C": -0.01}),
        benchmark=benchmark,
        pre_weights=benchmark,
        pre_cash_weight=0.0,
        risk_estimate=risk,
    )

    assert result.target is None
    assert result.diagnostics["status"] == "postsolve_infeasible"
    assert result.diagnostics["action"] == "hold_pre_trade_portfolio"
    assert result.diagnostics["repair_gap_equivalent_violation"] > (
        settings.feasibility_tolerance
    )
