from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from csi500_alpha.errors import Csi500AlphaError
from csi500_alpha.final_reporting import build_final_holdout_report
from csi500_alpha.logging_utils import configure_logging
from csi500_alpha.pipeline import (
    doctor,
    download_data,
    download_eligibility_data,
    download_smoke,
    plan_data_download,
    plan_eligibility_download,
    plan_portfolio_stress,
    plan_study,
    run_portfolio_stress,
    run_research_workflow,
    run_smoke,
    run_study,
    validate_existing_smoke,
)
from csi500_alpha.reporting import build_public_report

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/smoke.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csi500-alpha")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check environment and credentials")
    doctor_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor_parser.add_argument("--probe", action="store_true")

    download_parser = subparsers.add_parser("download-smoke", help="Download smoke data")
    download_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    download_parser.add_argument("--force", action="store_true")

    plan_data_parser = subparsers.add_parser(
        "plan-data",
        help="Plan a resumable annual market-data snapshot without network access",
    )
    plan_data_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full.yaml"),
    )

    download_data_parser = subparsers.add_parser(
        "download-data",
        help="Download, materialize and validate an annual-partitioned data snapshot",
    )
    download_data_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full.yaml"),
    )
    download_data_parser.add_argument("--force", action="store_true")
    download_data_parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="Refresh mutable security-master and industry reference requests only",
    )

    plan_eligibility_parser = subparsers.add_parser(
        "plan-eligibility",
        help="Plan historical-name and resumption supplements without network access",
    )
    plan_eligibility_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full.yaml"),
    )

    eligibility_parser = subparsers.add_parser(
        "download-eligibility",
        help="Download resumable historical-name and resumption supplements",
    )
    eligibility_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full.yaml"),
    )
    eligibility_parser.add_argument("--force", action="store_true")
    eligibility_parser.add_argument(
        "--refresh-names-from",
        metavar="YYYYMMDD",
        help="Refresh name/ST history only for constituents active from this date",
    )

    validate_parser = subparsers.add_parser("validate-smoke", help="Validate silver smoke data")
    validate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    run_parser = subparsers.add_parser("run-smoke", help="Run the complete smoke workflow")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--force", action="store_true")

    workflow_parser = subparsers.add_parser(
        "run-workflow",
        help="Run the pluggable factor-to-portfolio research workflow",
    )
    workflow_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/framework_smoke.yaml"),
    )
    workflow_parser.add_argument("--force", action="store_true")
    workflow_parser.add_argument(
        "--stage",
        choices=("walk_forward", "validation", "frozen_test"),
        help="Override the configured experiment stage without editing the YAML file",
    )
    workflow_parser.add_argument(
        "--confirm-final-holdout",
        action="store_true",
        help="Explicitly authorize the one-time frozen-test stage",
    )

    plan_study_parser = subparsers.add_parser(
        "plan-study",
        help="Validate and describe a bounded research study without loading data",
    )
    plan_study_parser.add_argument(
        "--study",
        type=Path,
        default=Path("configs/studies/core_baselines.yaml"),
    )

    study_parser = subparsers.add_parser(
        "run-study",
        help="Run or resume all trials in a research study",
    )
    study_parser.add_argument(
        "--study",
        type=Path,
        default=Path("configs/studies/core_baselines.yaml"),
    )
    study_parser.add_argument("--force", action="store_true")

    plan_stress_parser = subparsers.add_parser(
        "plan-stress",
        help="Validate a selected-trial cost and capacity stress matrix",
    )
    plan_stress_parser.add_argument(
        "--stress",
        type=Path,
        default=Path("configs/stress/core_cost_capacity.yaml"),
    )

    stress_parser = subparsers.add_parser(
        "run-stress",
        help="Run or resume portfolio-only stresses for a selected Study trial",
    )
    stress_parser.add_argument(
        "--stress",
        type=Path,
        default=Path("configs/stress/core_cost_capacity.yaml"),
    )

    report_parser = subparsers.add_parser(
        "build-report",
        help="Build public-safe aggregate figures and a reproducibility manifest",
    )
    report_parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Completed Study directory that supplies the selected backtest",
    )
    report_parser.add_argument(
        "--stress-root",
        type=Path,
        required=True,
        help="Completed stress directory tied to the selected Study trial",
    )
    report_parser.add_argument(
        "--ablation-study-root",
        type=Path,
        help="Study containing C0-C4; defaults to --study-root",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets"),
    )
    report_parser.add_argument(
        "--evaluation-role",
        choices=("method_selection", "extended_validation", "final_holdout"),
        default="method_selection",
    )

    final_report_parser = subparsers.add_parser(
        "build-final-report",
        help="Audit one frozen-test run and publish aggregate final-holdout evidence",
    )
    final_report_parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Completed frozen-test run directory",
    )
    final_report_parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    configure_logging(args.verbose)
    try:
        if args.command == "doctor":
            payload = doctor(args.config, probe=args.probe)
        elif args.command == "plan-data":
            payload = plan_data_download(args.config)
        elif args.command == "download-data":
            download_summary, quality = download_data(
                args.config,
                force=args.force,
                refresh_reference=args.refresh_reference,
            )
            payload = {
                "download": download_summary.to_dict(),
                "quality": quality.to_dict(),
            }
        elif args.command == "plan-eligibility":
            payload = plan_eligibility_download(args.config)
        elif args.command == "download-eligibility":
            payload = download_eligibility_data(
                args.config,
                force=args.force,
                refresh_names_from=args.refresh_names_from,
            ).to_dict()
        elif args.command == "download-smoke":
            download_summary = download_smoke(args.config, force=args.force)
            payload = download_summary.to_dict()
        elif args.command == "validate-smoke":
            payload = validate_existing_smoke(args.config).to_dict()
        elif args.command == "run-smoke":
            run_id, result = run_smoke(args.config, force=args.force)
            payload = {"run_id": run_id, "metrics": result.metrics}
        elif args.command == "run-workflow":
            run_id, summary = run_research_workflow(
                args.config,
                force=args.force,
                experiment_stage=args.stage,
                confirm_final_holdout=args.confirm_final_holdout,
            )
            payload = {"run_id": run_id, "summary": summary}
        elif args.command == "plan-study":
            payload = plan_study(args.study)
        elif args.command == "run-study":
            payload = run_study(args.study, force=args.force).to_dict()
        elif args.command == "plan-stress":
            payload = plan_portfolio_stress(args.stress)
        elif args.command == "run-stress":
            payload = run_portfolio_stress(args.stress).to_dict()
        elif args.command == "build-report":
            payload = build_public_report(
                study_root=args.study_root,
                stress_root=args.stress_root,
                ablation_study_root=args.ablation_study_root,
                output_root=args.output,
                evaluation_role=args.evaluation_role,
            ).to_dict()
        elif args.command == "build-final-report":
            payload = build_final_holdout_report(
                run_root=args.run_root,
                output_root=args.output,
            ).to_dict()
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    except Csi500AlphaError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
