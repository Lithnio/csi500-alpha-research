from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from csi500_alpha.errors import ConfigurationError


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing configuration key: {section}.{key}")
    return mapping[key]


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{section} must be a mapping")
    return value


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be true or false")
    return value


def _sequence(value: Any, key: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{key} must be a sequence")
    return tuple(value)


def _reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        prefix = f"{section}." if section else ""
        names = ", ".join(f"{prefix}{name}" for name in unknown)
        raise ConfigurationError(f"Unknown configuration keys: {names}")


def _date(value: Any, key: str) -> str:
    text = str(value)
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise ConfigurationError(f"{key} must use YYYYMMDD: {text}") from exc
    return text


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    run_root: Path
    dataset: str

    @property
    def silver_root(self) -> Path:
        return self.data_root / "silver" / self.dataset

    @property
    def quality_root(self) -> Path:
        return self.data_root / "quality" / self.dataset


@dataclass(frozen=True)
class SourceSettings:
    token_env: str
    exchange: str
    index_code: str
    total_return_index_code: str
    max_attempts: int
    request_timeout_seconds: float
    backoff_base_seconds: float
    backoff_max_seconds: float
    min_request_interval_seconds: float
    calls_per_minute_limit: int

    @property
    def effective_min_request_interval_seconds(self) -> float:
        """Return a conservative fixed interval for the configured account limit."""

        rate_interval = 60.0 / self.calls_per_minute_limit
        return max(self.min_request_interval_seconds, rate_interval + 0.01)


@dataclass(frozen=True)
class DateSettings:
    raw_start: str
    backtest_start: str
    end: str


@dataclass(frozen=True)
class DownloadSettings:
    include_daily_basic: bool
    include_suspensions: bool
    include_instrument_master: bool
    include_industry: bool
    industry_taxonomies: tuple[str, ...]
    supplement_industry_by_instrument: bool
    reference_cache_tag: str | None = None
    eligibility_refresh_start: str | None = None


@dataclass(frozen=True)
class ResearchSettings:
    factor_window: int
    rebalance_every: int
    top_n: int
    initial_cash: float
    linear_cost_bps: float
    stamp_duty_change_date: str
    stamp_duty_before: float
    stamp_duty_after: float
    price_limit_tolerance: float


@dataclass(frozen=True)
class RiskSettings:
    lookback: int
    min_history: int
    annualization: int
    missing_annual_volatility: float
    variance_floor: float
    return_clip: float
    model: str = "ledoit_wolf"
    beta_model: str = "feature_60"
    factor_half_life: float = 63.0
    specific_half_life: float = 63.0
    factor_covariance_shrinkage: float = 0.25
    specific_variance_shrinkage: float = 0.50
    factor_ridge: float = 1e-3
    min_factor_cross_section: int = 100
    style_exposure_clip: float = 3.0
    beta_lookback: int = 252
    beta_min_history: int = 60
    beta_half_life: float = 63.0
    beta_shrinkage: float = 0.25
    beta_clip_min: float = 0.25
    beta_clip_max: float = 1.75


@dataclass(frozen=True)
class OptimizerSettings:
    enabled: bool
    risk_aversion: float
    risk_horizon_days: int
    l2_penalty: float
    active_cap: float
    name_cap: float
    turnover_cap: float
    initial_turnover_cap: float
    exposure_cap: float
    solvers: tuple[str, ...]
    beta_constraint_enabled: bool = False
    beta_active_cap: float = 0.05
    tracking_error_cap: float = 0.05
    feasibility_tolerance: float = 1e-6
    constraint_materiality_tolerance: float = 1e-4
    liquidity_enabled: bool = False
    portfolio_aum_cny: float = 100_000_000.0
    adv_lookback: int = 20
    min_adv_observations: int = 10
    max_adv_participation: float = 0.05
    impact_bps_at_max_participation: float = 10.0


@dataclass(frozen=True)
class FeatureSettings:
    label_horizon: int
    min_factor_coverage: float
    mad_clip: float
    industry_coverage_threshold: float
    industry_transition_date: str


@dataclass(frozen=True)
class ComponentSettings:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class WorkflowSettings:
    feature_start: str
    portfolio_start: str
    refit_every: int
    factor_names: tuple[str, ...]
    feature_provider: ComponentSettings
    selector: ComponentSettings
    model: ComponentSettings
    calibrator: ComponentSettings


@dataclass(frozen=True)
class ExperimentSettings:
    stage: str
    protocol_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    embargo_days: int
    allow_frozen_test: bool = False

    @property
    def enabled(self) -> bool:
        return self.stage != "walk_forward"


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    paths: ProjectPaths
    source: SourceSettings
    dates: DateSettings
    download: DownloadSettings
    research: ResearchSettings
    risk: RiskSettings
    optimizer: OptimizerSettings
    features: FeatureSettings
    workflow: WorkflowSettings
    experiment: ExperimentSettings

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> AppConfig:
        path = Path(config_path).resolve()
        if not path.exists():
            raise ConfigurationError(f"Configuration file does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError("Configuration root must be a mapping")

        _reject_unknown(
            raw,
            {
                "project",
                "source",
                "dates",
                "download",
                "research",
                "risk",
                "optimizer",
                "features",
                "workflow",
                "experiment",
            },
            "",
        )
        project = _mapping(raw.get("project", {}), "project")
        source = _mapping(raw.get("source", {}), "source")
        dates = _mapping(raw.get("dates", {}), "dates")
        download = _mapping(raw.get("download", {}), "download")
        research = _mapping(raw.get("research", {}), "research")
        risk = _mapping(raw.get("risk", {}), "risk")
        optimizer = _mapping(raw.get("optimizer", {}), "optimizer")
        features = _mapping(raw.get("features", {}), "features")
        workflow = _mapping(raw.get("workflow", {}), "workflow")
        experiment = _mapping(raw.get("experiment", {}), "experiment")
        section_keys = {
            "project": {"data_root", "run_root", "dataset"},
            "source": {
                "token_env",
                "exchange",
                "index_code",
                "total_return_index_code",
                "max_attempts",
                "request_timeout_seconds",
                "backoff_base_seconds",
                "backoff_max_seconds",
                "min_request_interval_seconds",
                "calls_per_minute_limit",
            },
            "dates": {"raw_start", "backtest_start", "end"},
            "download": {
                "include_daily_basic",
                "include_suspensions",
                "include_instrument_master",
                "include_industry",
                "industry_taxonomies",
                "supplement_industry_by_instrument",
                "reference_cache_tag",
                "eligibility_refresh_start",
            },
            "research": {
                "factor_window",
                "rebalance_every",
                "top_n",
                "initial_cash",
                "linear_cost_bps",
                "stamp_duty_change_date",
                "stamp_duty_before",
                "stamp_duty_after",
                "price_limit_tolerance",
            },
            "risk": {
                "lookback",
                "min_history",
                "annualization",
                "missing_annual_volatility",
                "variance_floor",
                "return_clip",
                "model",
                "beta_model",
                "factor_half_life",
                "specific_half_life",
                "factor_covariance_shrinkage",
                "specific_variance_shrinkage",
                "factor_ridge",
                "min_factor_cross_section",
                "style_exposure_clip",
                "beta_lookback",
                "beta_min_history",
                "beta_half_life",
                "beta_shrinkage",
                "beta_clip_min",
                "beta_clip_max",
            },
            "optimizer": {
                "enabled",
                "risk_aversion",
                "risk_horizon_days",
                "l2_penalty",
                "active_cap",
                "name_cap",
                "turnover_cap",
                "initial_turnover_cap",
                "exposure_cap",
                "solvers",
                "beta_constraint_enabled",
                "beta_active_cap",
                "tracking_error_cap",
                "feasibility_tolerance",
                "constraint_materiality_tolerance",
                "liquidity_enabled",
                "portfolio_aum_cny",
                "adv_lookback",
                "min_adv_observations",
                "max_adv_participation",
                "impact_bps_at_max_participation",
            },
            "features": {
                "label_horizon",
                "min_factor_coverage",
                "mad_clip",
                "industry_coverage_threshold",
                "industry_transition_date",
            },
            "workflow": {
                "feature_start",
                "portfolio_start",
                "refit_every",
                "factors",
                "feature_provider",
                "selector",
                "model",
                "calibrator",
            },
            "experiment": {
                "stage",
                "protocol_id",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
                "embargo_days",
                "allow_frozen_test",
            },
        }
        for section, mapping in {
            "project": project,
            "source": source,
            "dates": dates,
            "download": download,
            "research": research,
            "risk": risk,
            "optimizer": optimizer,
            "features": features,
            "workflow": workflow,
            "experiment": experiment,
        }.items():
            _reject_unknown(mapping, section_keys[section], section)
        raw_workflow_factors = _sequence(
            workflow.get("factors", ()),
            "workflow.factors",
        )
        root = path.parent.parent.resolve()
        default_workflow_start = _date(
            _require(dates, "backtest_start", "dates"),
            "dates.backtest_start",
        )

        def component(
            key: str,
            default_name: str,
            default_params: dict[str, Any] | None = None,
        ) -> ComponentSettings:
            raw_component = workflow.get(key)
            if raw_component is None:
                return ComponentSettings(
                    name=default_name,
                    params=dict(default_params or {}),
                )
            if not isinstance(raw_component, dict):
                raise ConfigurationError(f"workflow.{key} must be a mapping")
            _reject_unknown(raw_component, {"name", "params"}, f"workflow.{key}")
            raw_params = raw_component.get("params", {})
            if not isinstance(raw_params, dict):
                raise ConfigurationError(f"workflow.{key}.params must be a mapping")
            return ComponentSettings(
                name=str(raw_component.get("name", default_name)),
                params=dict(raw_params),
            )

        workflow_settings = WorkflowSettings(
            feature_start=_date(
                workflow.get("feature_start", default_workflow_start),
                "workflow.feature_start",
            ),
            portfolio_start=_date(
                workflow.get("portfolio_start", default_workflow_start),
                "workflow.portfolio_start",
            ),
            refit_every=int(workflow.get("refit_every", 4)),
            factor_names=tuple(str(name) for name in raw_workflow_factors),
            feature_provider=component("feature_provider", "builtin_daily"),
            selector=component("selector", "all"),
            model=component("model", "direction_equal_weight"),
            calibrator=component(
                "calibrator",
                "robust_cross_section",
                {
                    "target_scale": 0.01,
                    "score_clip": 3.0,
                },
            ),
        )
        experiment_stage = str(experiment.get("stage", "walk_forward"))
        if experiment_stage == "walk_forward":
            experiment_settings = ExperimentSettings(
                stage=experiment_stage,
                protocol_id=str(experiment.get("protocol_id", "local-walk-forward")),
                train_start=workflow_settings.feature_start,
                train_end=workflow_settings.portfolio_start,
                validation_start=workflow_settings.portfolio_start,
                validation_end=_date(_require(dates, "end", "dates"), "dates.end"),
                test_start=_date(_require(dates, "end", "dates"), "dates.end"),
                test_end=_date(_require(dates, "end", "dates"), "dates.end"),
                embargo_days=int(experiment.get("embargo_days", 0)),
            )
        else:
            experiment_settings = ExperimentSettings(
                stage=experiment_stage,
                protocol_id=str(_require(experiment, "protocol_id", "experiment")),
                train_start=_date(
                    _require(experiment, "train_start", "experiment"),
                    "experiment.train_start",
                ),
                train_end=_date(
                    _require(experiment, "train_end", "experiment"),
                    "experiment.train_end",
                ),
                validation_start=_date(
                    _require(experiment, "validation_start", "experiment"),
                    "experiment.validation_start",
                ),
                validation_end=_date(
                    _require(experiment, "validation_end", "experiment"),
                    "experiment.validation_end",
                ),
                test_start=_date(
                    _require(experiment, "test_start", "experiment"),
                    "experiment.test_start",
                ),
                test_end=_date(
                    _require(experiment, "test_end", "experiment"),
                    "experiment.test_end",
                ),
                embargo_days=int(experiment.get("embargo_days", 0)),
                allow_frozen_test=_boolean(
                    experiment.get("allow_frozen_test", False),
                    "experiment.allow_frozen_test",
                ),
            )

        result = cls(
            config_path=path,
            paths=ProjectPaths(
                root=root,
                data_root=(root / str(_require(project, "data_root", "project"))).resolve(),
                run_root=(root / str(_require(project, "run_root", "project"))).resolve(),
                dataset=str(project.get("dataset", path.stem)),
            ),
            source=SourceSettings(
                token_env=str(_require(source, "token_env", "source")),
                exchange=str(_require(source, "exchange", "source")),
                index_code=str(_require(source, "index_code", "source")),
                total_return_index_code=str(
                    source.get("total_return_index_code", "H00905.CSI")
                ),
                max_attempts=int(source.get("max_attempts", 3)),
                request_timeout_seconds=float(source.get("request_timeout_seconds", 30.0)),
                backoff_base_seconds=float(source.get("backoff_base_seconds", 1.0)),
                backoff_max_seconds=float(source.get("backoff_max_seconds", 30.0)),
                min_request_interval_seconds=float(source.get("min_request_interval_seconds", 0.0)),
                calls_per_minute_limit=int(source.get("calls_per_minute_limit", 200)),
            ),
            dates=DateSettings(
                raw_start=_date(_require(dates, "raw_start", "dates"), "dates.raw_start"),
                backtest_start=_date(
                    _require(dates, "backtest_start", "dates"), "dates.backtest_start"
                ),
                end=_date(_require(dates, "end", "dates"), "dates.end"),
            ),
            download=DownloadSettings(
                include_daily_basic=_boolean(
                    download.get("include_daily_basic", False),
                    "download.include_daily_basic",
                ),
                include_suspensions=_boolean(
                    download.get("include_suspensions", False),
                    "download.include_suspensions",
                ),
                include_instrument_master=_boolean(
                    download.get("include_instrument_master", False),
                    "download.include_instrument_master",
                ),
                include_industry=_boolean(
                    download.get("include_industry", False),
                    "download.include_industry",
                ),
                industry_taxonomies=tuple(
                    str(value)
                    for value in _sequence(
                        download.get("industry_taxonomies", ("SW2021",)),
                        "download.industry_taxonomies",
                    )
                ),
                supplement_industry_by_instrument=_boolean(
                    download.get("supplement_industry_by_instrument", True),
                    "download.supplement_industry_by_instrument",
                ),
                reference_cache_tag=(
                    str(download["reference_cache_tag"])
                    if download.get("reference_cache_tag") is not None
                    else None
                ),
                eligibility_refresh_start=(
                    _date(
                        download["eligibility_refresh_start"],
                        "download.eligibility_refresh_start",
                    )
                    if download.get("eligibility_refresh_start") is not None
                    else None
                ),
            ),
            research=ResearchSettings(
                factor_window=int(research.get("factor_window", 5)),
                rebalance_every=int(research.get("rebalance_every", 5)),
                top_n=int(research.get("top_n", 30)),
                initial_cash=float(research.get("initial_cash", 1.0)),
                linear_cost_bps=float(research.get("linear_cost_bps", 5.0)),
                stamp_duty_change_date=_date(
                    research.get("stamp_duty_change_date", "20230828"),
                    "research.stamp_duty_change_date",
                ),
                stamp_duty_before=float(research.get("stamp_duty_before", 0.001)),
                stamp_duty_after=float(research.get("stamp_duty_after", 0.0005)),
                price_limit_tolerance=float(research.get("price_limit_tolerance", 1e-6)),
            ),
            risk=RiskSettings(
                lookback=int(risk.get("lookback", 252)),
                min_history=int(risk.get("min_history", 120)),
                annualization=int(risk.get("annualization", 252)),
                missing_annual_volatility=float(risk.get("missing_annual_volatility", 0.80)),
                variance_floor=float(risk.get("variance_floor", 1e-8)),
                return_clip=float(risk.get("return_clip", 0.20)),
                model=str(risk.get("model", "ledoit_wolf")),
                beta_model=str(risk.get("beta_model", "feature_60")),
                factor_half_life=float(risk.get("factor_half_life", 63.0)),
                specific_half_life=float(risk.get("specific_half_life", 63.0)),
                factor_covariance_shrinkage=float(
                    risk.get("factor_covariance_shrinkage", 0.25)
                ),
                specific_variance_shrinkage=float(
                    risk.get("specific_variance_shrinkage", 0.50)
                ),
                factor_ridge=float(risk.get("factor_ridge", 1e-3)),
                min_factor_cross_section=int(
                    risk.get("min_factor_cross_section", 100)
                ),
                style_exposure_clip=float(risk.get("style_exposure_clip", 3.0)),
                beta_lookback=int(risk.get("beta_lookback", 252)),
                beta_min_history=int(risk.get("beta_min_history", 60)),
                beta_half_life=float(risk.get("beta_half_life", 63.0)),
                beta_shrinkage=float(risk.get("beta_shrinkage", 0.25)),
                beta_clip_min=float(risk.get("beta_clip_min", 0.25)),
                beta_clip_max=float(risk.get("beta_clip_max", 1.75)),
            ),
            optimizer=OptimizerSettings(
                enabled=_boolean(optimizer.get("enabled", False), "optimizer.enabled"),
                risk_aversion=float(optimizer.get("risk_aversion", 3.0)),
                risk_horizon_days=int(optimizer.get("risk_horizon_days", 5)),
                l2_penalty=float(optimizer.get("l2_penalty", 0.01)),
                active_cap=float(optimizer.get("active_cap", 0.01)),
                name_cap=float(optimizer.get("name_cap", 0.03)),
                turnover_cap=float(optimizer.get("turnover_cap", 0.35)),
                initial_turnover_cap=float(optimizer.get("initial_turnover_cap", 1.0)),
                exposure_cap=float(optimizer.get("exposure_cap", 0.02)),
                solvers=tuple(
                    str(value)
                    for value in _sequence(
                        optimizer.get("solvers", ("CLARABEL", "OSQP")),
                        "optimizer.solvers",
                    )
                ),
                beta_constraint_enabled=_boolean(
                    optimizer.get("beta_constraint_enabled", False),
                    "optimizer.beta_constraint_enabled",
                ),
                beta_active_cap=float(optimizer.get("beta_active_cap", 0.05)),
                tracking_error_cap=float(optimizer.get("tracking_error_cap", 0.05)),
                feasibility_tolerance=float(optimizer.get("feasibility_tolerance", 1e-6)),
                constraint_materiality_tolerance=float(
                    optimizer.get("constraint_materiality_tolerance", 1e-4)
                ),
                liquidity_enabled=_boolean(
                    optimizer.get("liquidity_enabled", False),
                    "optimizer.liquidity_enabled",
                ),
                portfolio_aum_cny=float(optimizer.get("portfolio_aum_cny", 100_000_000.0)),
                adv_lookback=int(optimizer.get("adv_lookback", 20)),
                min_adv_observations=int(optimizer.get("min_adv_observations", 10)),
                max_adv_participation=float(optimizer.get("max_adv_participation", 0.05)),
                impact_bps_at_max_participation=float(
                    optimizer.get("impact_bps_at_max_participation", 10.0)
                ),
            ),
            features=FeatureSettings(
                label_horizon=int(features.get("label_horizon", 5)),
                min_factor_coverage=float(features.get("min_factor_coverage", 0.80)),
                mad_clip=float(features.get("mad_clip", 5.0)),
                industry_coverage_threshold=float(
                    features.get("industry_coverage_threshold", 0.90)
                ),
                industry_transition_date=_date(
                    features.get("industry_transition_date", "20211213"),
                    "features.industry_transition_date",
                ),
            ),
            workflow=workflow_settings,
            experiment=experiment_settings,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not (self.dates.raw_start <= self.dates.backtest_start <= self.dates.end):
            raise ConfigurationError("Expected raw_start <= backtest_start <= end")
        if self.source.max_attempts < 1:
            raise ConfigurationError("source.max_attempts must be positive")
        if self.source.request_timeout_seconds <= 0:
            raise ConfigurationError("source.request_timeout_seconds must be positive")
        if self.source.min_request_interval_seconds < 0:
            raise ConfigurationError("source.min_request_interval_seconds cannot be negative")
        if self.source.calls_per_minute_limit < 1:
            raise ConfigurationError("source.calls_per_minute_limit must be positive")
        if not self.source.total_return_index_code.strip():
            raise ConfigurationError("source.total_return_index_code cannot be empty")
        if self.source.total_return_index_code == self.source.index_code:
            raise ConfigurationError(
                "source.total_return_index_code must differ from source.index_code"
            )
        if self.research.factor_window < 1 or self.research.rebalance_every < 1:
            raise ConfigurationError("Factor and rebalance windows must be positive")
        if self.research.top_n < 1:
            raise ConfigurationError("research.top_n must be positive")
        if self.research.initial_cash <= 0:
            raise ConfigurationError("research.initial_cash must be positive")
        if self.research.linear_cost_bps < 0:
            raise ConfigurationError("research.linear_cost_bps cannot be negative")
        if self.risk.lookback < 2 or not 2 <= self.risk.min_history <= self.risk.lookback:
            raise ConfigurationError("Expected 2 <= risk.min_history <= risk.lookback")
        if self.risk.annualization < 1 or self.risk.missing_annual_volatility <= 0:
            raise ConfigurationError("Risk annualization and missing volatility must be positive")
        if self.risk.variance_floor <= 0 or self.risk.return_clip <= 0:
            raise ConfigurationError("Risk floors and clipping thresholds must be positive")
        allowed_risk_models = {"ledoit_wolf", "factor_ewma"}
        if self.risk.model not in allowed_risk_models:
            raise ConfigurationError(
                f"risk.model must be one of {sorted(allowed_risk_models)}"
            )
        allowed_beta_models = {"feature_60", "ewma_shrunk"}
        if self.risk.beta_model not in allowed_beta_models:
            raise ConfigurationError(
                f"risk.beta_model must be one of {sorted(allowed_beta_models)}"
            )
        positive_risk_parameters = {
            "risk.factor_half_life": self.risk.factor_half_life,
            "risk.specific_half_life": self.risk.specific_half_life,
            "risk.factor_ridge": self.risk.factor_ridge,
            "risk.min_factor_cross_section": self.risk.min_factor_cross_section,
            "risk.style_exposure_clip": self.risk.style_exposure_clip,
            "risk.beta_lookback": self.risk.beta_lookback,
            "risk.beta_min_history": self.risk.beta_min_history,
            "risk.beta_half_life": self.risk.beta_half_life,
        }
        if any(value <= 0 for value in positive_risk_parameters.values()):
            raise ConfigurationError("Factor-risk and beta windows must be positive")
        if not 2 <= self.risk.beta_min_history <= self.risk.beta_lookback:
            raise ConfigurationError(
                "Expected 2 <= risk.beta_min_history <= risk.beta_lookback"
            )
        shrinkages = {
            "risk.factor_covariance_shrinkage": (
                self.risk.factor_covariance_shrinkage
            ),
            "risk.specific_variance_shrinkage": (
                self.risk.specific_variance_shrinkage
            ),
            "risk.beta_shrinkage": self.risk.beta_shrinkage,
        }
        if any(not 0 <= value <= 1 for value in shrinkages.values()):
            raise ConfigurationError("Risk shrinkage parameters must be in [0, 1]")
        if self.risk.beta_clip_min <= 0 or self.risk.beta_clip_min >= self.risk.beta_clip_max:
            raise ConfigurationError(
                "Expected 0 < risk.beta_clip_min < risk.beta_clip_max"
            )
        if self.optimizer.enabled and not self.optimizer.solvers:
            raise ConfigurationError("optimizer.solvers cannot be empty when enabled")
        bounded_positive = {
            "optimizer.risk_horizon_days": self.optimizer.risk_horizon_days,
            "optimizer.active_cap": self.optimizer.active_cap,
            "optimizer.name_cap": self.optimizer.name_cap,
            "optimizer.turnover_cap": self.optimizer.turnover_cap,
            "optimizer.initial_turnover_cap": self.optimizer.initial_turnover_cap,
            "optimizer.beta_active_cap": self.optimizer.beta_active_cap,
            "optimizer.tracking_error_cap": self.optimizer.tracking_error_cap,
        }
        if any(value <= 0 for value in bounded_positive.values()):
            raise ConfigurationError(
                "Optimizer horizons, caps and turnover limits must be positive"
            )
        if self.optimizer.risk_aversion < 0 or self.optimizer.l2_penalty < 0:
            raise ConfigurationError("Optimizer penalties cannot be negative")
        if self.optimizer.feasibility_tolerance <= 0:
            raise ConfigurationError("optimizer.feasibility_tolerance must be positive")
        if self.optimizer.constraint_materiality_tolerance < self.optimizer.feasibility_tolerance:
            raise ConfigurationError(
                "optimizer.constraint_materiality_tolerance must be at least "
                "optimizer.feasibility_tolerance"
            )
        if self.optimizer.portfolio_aum_cny <= 0:
            raise ConfigurationError("optimizer.portfolio_aum_cny must be positive")
        if not (1 <= self.optimizer.min_adv_observations <= self.optimizer.adv_lookback):
            raise ConfigurationError(
                "Expected 1 <= optimizer.min_adv_observations <= optimizer.adv_lookback"
            )
        if not 0 < self.optimizer.max_adv_participation <= 1:
            raise ConfigurationError("optimizer.max_adv_participation must be in (0, 1]")
        if self.optimizer.impact_bps_at_max_participation < 0:
            raise ConfigurationError("optimizer.impact_bps_at_max_participation cannot be negative")
        if not self.paths.dataset or any(char in self.paths.dataset for char in "\\/:"):
            raise ConfigurationError("project.dataset must be a simple non-empty name")
        allowed_taxonomies = {"SW2014", "SW2021"}
        if not set(self.download.industry_taxonomies).issubset(allowed_taxonomies):
            raise ConfigurationError("Industry taxonomies must be SW2014 and/or SW2021")
        if self.download.include_industry and not self.download.industry_taxonomies:
            raise ConfigurationError("At least one industry taxonomy is required")
        if self.download.reference_cache_tag is not None:
            cache_tag = self.download.reference_cache_tag
            if not cache_tag or any(
                not (character.isalnum() or character in "._-")
                for character in cache_tag
            ):
                raise ConfigurationError(
                    "download.reference_cache_tag must use only letters, numbers, '.', '-', or '_'"
                )
        if (
            self.download.eligibility_refresh_start is not None
            and not self.dates.raw_start
            <= self.download.eligibility_refresh_start
            <= self.dates.end
        ):
            raise ConfigurationError(
                "download.eligibility_refresh_start must fall within the configured data range"
            )
        if self.features.label_horizon < 1 or self.features.mad_clip <= 0:
            raise ConfigurationError("Feature horizon and MAD clipping must be positive")
        proportions = {
            "features.min_factor_coverage": self.features.min_factor_coverage,
            "features.industry_coverage_threshold": (self.features.industry_coverage_threshold),
        }
        if any(not 0 < value <= 1 for value in proportions.values()):
            raise ConfigurationError("Feature coverage thresholds must be in (0, 1]")
        if not (
            self.dates.raw_start
            <= self.workflow.feature_start
            <= self.workflow.portfolio_start
            <= self.dates.end
        ):
            raise ConfigurationError(
                "Expected raw_start <= workflow.feature_start <= workflow.portfolio_start <= end"
            )
        if self.workflow.refit_every < 1:
            raise ConfigurationError("workflow.refit_every must be positive")
        components = {
            "workflow.feature_provider": self.workflow.feature_provider,
            "workflow.selector": self.workflow.selector,
            "workflow.model": self.workflow.model,
            "workflow.calibrator": self.workflow.calibrator,
        }
        if any(not component.name.strip() for component in components.values()):
            raise ConfigurationError("Workflow component names cannot be empty")
        if len(set(self.workflow.factor_names)) != len(self.workflow.factor_names):
            raise ConfigurationError("workflow.factors cannot contain duplicates")
        allowed_stages = {"walk_forward", "validation", "frozen_test"}
        if self.experiment.stage not in allowed_stages:
            raise ConfigurationError(f"experiment.stage must be one of {sorted(allowed_stages)}")
        if self.experiment.embargo_days < 0:
            raise ConfigurationError("experiment.embargo_days cannot be negative")
        if (
            self.experiment.stage == "frozen_test"
            and not self.experiment.allow_frozen_test
        ):
            raise ConfigurationError(
                "frozen_test is disabled for this protocol config; use the dedicated "
                "final-holdout config"
            )
        if not self.experiment.protocol_id or any(
            character in self.experiment.protocol_id for character in "\\/:"
        ):
            raise ConfigurationError("experiment.protocol_id must be a simple name")
        if self.experiment.enabled:
            ordered_phases = (
                self.workflow.feature_start
                <= self.experiment.train_start
                <= self.experiment.train_end
                < self.experiment.validation_start
                <= self.experiment.validation_end
                < self.experiment.test_start
                <= self.experiment.test_end
            )
            visible_end_is_valid = (
                self.experiment.validation_end <= self.dates.end
                if self.experiment.stage == "validation"
                else self.experiment.test_end <= self.dates.end
            )
            if not ordered_phases or not visible_end_is_valid:
                raise ConfigurationError(
                    "Expected feature_start <= train <= validation < test, with the "
                    "active stage ending no later than dates.end"
                )
