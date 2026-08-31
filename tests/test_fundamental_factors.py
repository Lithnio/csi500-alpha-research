from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.features.catalog import A2_DAILY_FACTOR_NAMES
from csi500_alpha.features.fundamental import (
    FUNDAMENTAL_DIRECTIONS,
    FUNDAMENTAL_FACTOR_CATALOG,
    FUNDAMENTAL_FACTOR_NAMES,
    FUNDAMENTAL_FAMILIES,
    build_fundamental_factor_panel,
)
from csi500_alpha.workflow.components import default_component_registry


def _available_date(report_period: str) -> str:
    year = int(report_period[:4])
    ending = report_period[4:]
    if ending == "0331":
        return f"{year}0430"
    if ending == "0630":
        return f"{year}0831"
    if ending == "0930":
        return f"{year}1031"
    return f"{year + 1}0430"


def _financial_tables(*, company_type: str = "1") -> dict[str, pd.DataFrame]:
    income_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    cashflow_rows: list[dict[str, Any]] = []
    quarter_revenue = {2022: 20.0, 2023: 25.0, 2024: 30.0}
    quarter_net_income = {2022: 3.0, 2023: 4.0, 2024: 5.0}
    periods = pd.period_range("2022Q1", "2024Q4", freq="Q-DEC")
    for position, period in enumerate(periods):
        report_period = period.end_time.strftime("%Y%m%d")
        year = int(period.year)
        quarter = int(period.quarter)
        revenue_ytd = quarter_revenue[year] * quarter
        net_income_ytd = quarter_net_income[year] * quarter
        available_date = _available_date(report_period)
        common = {
            "instrument": "000001.SZ",
            "report_period": report_period,
            "report_type": "1",
            "available_date": available_date,
            "version_sequence": 1,
            "company_type": company_type,
        }
        income_rows.append(
            {
                **common,
                "total_revenue": revenue_ytd,
                "revenue": revenue_ytd,
                "oper_cost": revenue_ytd * 0.60,
                "operate_profit": revenue_ytd * 0.20,
                "n_income_attr_p": net_income_ytd,
            }
        )
        if report_period == "20221231":
            assets, equity = 90.0, 55.0
        elif report_period == "20231231":
            assets, equity = 100.0, 60.0
        elif report_period == "20241231":
            assets, equity = 110.0, 65.0
        else:
            assets = 75.0 + position * 3.0
            equity = 45.0 + position * 1.5
        balance_rows.append(
            {
                **common,
                "total_assets": assets,
                "total_hldr_eqy_exc_min_int": equity,
            }
        )
        cashflow_rows.append(
            {
                **common,
                "n_cashflow_act": net_income_ytd * 1.20,
                "c_pay_acq_const_fiolta": quarter * (0.8 + 0.2 * (year - 2022)),
            }
        )
    return {
        "income": pd.DataFrame(income_rows),
        "balancesheet": pd.DataFrame(balance_rows),
        "cashflow": pd.DataFrame(cashflow_rows),
    }


def _base(decision_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": [decision_date],
            "instrument": ["000001.SZ"],
            "total_mv_cny": [200.0],
        }
    )


def test_fundamental_catalog_has_five_auditable_families() -> None:
    assert len(FUNDAMENTAL_FACTOR_CATALOG) == 17
    assert len(set(FUNDAMENTAL_FACTOR_NAMES)) == len(FUNDAMENTAL_FACTOR_NAMES)
    assert set(FUNDAMENTAL_DIRECTIONS.values()) == {-1, 1}
    assert set(FUNDAMENTAL_FAMILIES.values()) == {
        "quality",
        "value",
        "growth",
        "investment",
        "accrual",
    }
    for definition in FUNDAMENTAL_FACTOR_CATALOG:
        assert definition.input_fields
        assert definition.formula
        assert definition.missing_rule
        assert definition.availability

    provider = default_component_registry().create_feature_provider(
        "builtin_daily_fundamental",
        {"max_financial_age_days": 180},
    )
    assert set(FUNDAMENTAL_FACTOR_NAMES).issubset(provider.factor_names)
    assert all(provider.directions[name] in {-1, 1} for name in provider.factor_names)


def test_a2_fundamental_provider_exposes_expanded_daily_catalog() -> None:
    provider = default_component_registry().create_feature_provider(
        "builtin_a2_daily_fundamental",
        {"max_financial_age_days": 180},
    )

    assert set(A2_DAILY_FACTOR_NAMES).issubset(provider.factor_names)
    assert set(FUNDAMENTAL_FACTOR_NAMES).issubset(provider.factor_names)


def test_cumulative_reports_are_converted_to_ttm_and_yoy_factors() -> None:
    result = build_fundamental_factor_panel(
        _base("20250502"),
        _financial_tables(),
    )
    row = result.panel.iloc[0]

    assert row["financial_report_period"] == "20241231"
    assert row["financial_available_date"] == "20250430"
    assert not bool(row["financial_stale"])
    assert np.isclose(row["gross_profitability_ttm"], 48.0 / 105.0)
    assert np.isclose(row["roe_ttm"], 20.0 / 62.5)
    assert np.isclose(row["roa_ttm"], 20.0 / 105.0)
    assert np.isclose(row["operating_margin_ttm"], 0.20)
    assert np.isclose(row["cash_return_on_assets_ttm"], 24.0 / 105.0)
    assert np.isclose(row["book_to_market"], 65.0 / 200.0)
    assert np.isclose(row["earnings_yield_ttm"], 20.0 / 200.0)
    assert np.isclose(row["sales_to_price_ttm"], 120.0 / 200.0)
    assert np.isclose(row["cfo_yield_ttm"], 24.0 / 200.0)
    assert np.isclose(row["revenue_growth_ttm_yoy"], 0.20)
    assert np.isclose(row["earnings_growth_ttm_yoy"], 0.25)
    assert np.isclose(row["cfo_growth_ttm_yoy"], 0.25)
    assert np.isclose(row["roe_change_yoy"], 20.0 / 62.5 - 16.0 / 57.5)
    assert np.isclose(row["asset_growth_yoy"], 0.10)
    assert np.isclose(row["capex_intensity_ttm"], 4.8 / 105.0)
    assert np.isclose(row["total_accruals_ttm"], -4.0 / 105.0)
    assert np.isclose(row["cash_conversion_ttm"], 1.20)
    assert result.quality["lookahead_violations"].eq(0).all()
    assert result.quality["coverage"].eq(1.0).all()


def test_future_revision_cannot_change_an_earlier_decision_date() -> None:
    baseline_tables = _financial_tables()
    revised_tables = {name: table.copy() for name, table in baseline_tables.items()}
    revised = revised_tables["income"].loc[
        revised_tables["income"]["report_period"].eq("20241231")
    ].iloc[0].copy()
    revised["available_date"] = "20250505"
    revised["version_sequence"] = 2
    revised["total_revenue"] = 1200.0
    revised["revenue"] = 1200.0
    revised["oper_cost"] = 720.0
    revised["operate_profit"] = 240.0
    revised["n_income_attr_p"] = 200.0
    revised_tables["income"] = pd.concat(
        [revised_tables["income"], revised.to_frame().T],
        ignore_index=True,
    )

    baseline = build_fundamental_factor_panel(
        _base("20250502"), baseline_tables
    ).panel
    unchanged = build_fundamental_factor_panel(
        _base("20250502"), revised_tables
    ).panel
    changed = build_fundamental_factor_panel(
        _base("20250506"), revised_tables
    ).panel

    pd.testing.assert_frame_equal(
        baseline[list(FUNDAMENTAL_FACTOR_NAMES)],
        unchanged[list(FUNDAMENTAL_FACTOR_NAMES)],
    )
    assert changed.iloc[0]["earnings_yield_ttm"] > baseline.iloc[0]["earnings_yield_ttm"]


def test_incremental_panel_matches_separate_point_in_time_snapshots() -> None:
    tables = _financial_tables()
    revised = tables["income"].loc[
        tables["income"]["report_period"].eq("20241231")
    ].iloc[0].copy()
    revised["available_date"] = "20250505"
    revised["version_sequence"] = 2
    revised["n_income_attr_p"] = 200.0
    tables["income"] = pd.concat(
        [tables["income"], revised.to_frame().T],
        ignore_index=True,
    )
    base = pd.concat([_base("20250502"), _base("20250506")], ignore_index=True)

    incremental = build_fundamental_factor_panel(base, tables).panel
    separate = pd.concat(
        [
            build_fundamental_factor_panel(_base(date), tables).panel
            for date in ("20250502", "20250506")
        ],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(incremental, separate)


def test_stale_reports_are_disabled_instead_of_forward_filled() -> None:
    result = build_fundamental_factor_panel(
        _base("20250715"),
        _financial_tables(),
        max_age_days=180,
    )

    assert bool(result.panel.iloc[0]["financial_stale"])
    assert result.panel[list(FUNDAMENTAL_FACTOR_NAMES)].isna().all().all()
    assert result.quality["stale_instruments"].eq(1).all()


def test_nonindustrial_companies_only_keep_comparable_factors() -> None:
    result = build_fundamental_factor_panel(
        _base("20250502"),
        _financial_tables(company_type="2"),
    ).panel.iloc[0]

    assert np.isnan(result["gross_profitability_ttm"])
    assert np.isnan(result["cash_return_on_assets_ttm"])
    assert np.isnan(result["asset_growth_yoy"])
    assert np.isnan(result["total_accruals_ttm"])
    assert np.isfinite(result["roe_ttm"])
    assert np.isfinite(result["book_to_market"])
    assert np.isfinite(result["earnings_yield_ttm"])
