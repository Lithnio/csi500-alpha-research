import cvxpy as cp
import numpy as np
import pandas as pd

from csi500_alpha.config import OptimizerSettings, ResearchSettings, RiskSettings
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.risk.model import LedoitWolfRiskModel, RiskEstimate


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
    assert result.diagnostics["ineligible_position_policy"] == "cannot_increase_may_reduce"
    assert result.diagnostics["ineligible_held"] == 1
    assert result.diagnostics["preexisting_active_cap_breaches"] == 3
    assert result.diagnostics["maximum_violation"] <= settings.feasibility_tolerance


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
