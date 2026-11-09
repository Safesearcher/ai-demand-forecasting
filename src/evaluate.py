"""Run Phase 4 evaluation for one pipeline without generating model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.classification import (
    evaluate_stockout_alerts,
    plot_confusion_matrices,
    plot_precision_recall_curves,
    plot_roc_curves,
    save_classification_latex,
)
from evaluation.stockout import (
    apply_stockout_labels,
    choose_stock_multiplier,
    compute_origin_known_stock,
)
from evaluation.config import (
    EvaluationSpec,
    canonicalize_data_columns,
    canonicalize_prediction_columns,
    canonicalize_subgroup_columns,
    get_evaluation_spec,
)
from evaluation.error_analysis import (
    add_volume_tiers,
    calculate_bias_gaps,
    calculate_subgroup_errors,
    find_worst_dates,
    find_worst_series,
    plot_residuals_over_time,
    plot_subgroup_errors,
)
from evaluation.figures import (
    plot_coverage_calibration,
    plot_loss_curves,
    plot_model_comparison,
    plot_prediction_bands,
    plot_residual_histogram,
)
from evaluation.metrics import (
    evaluate_models,
    save_metrics_csv,
    save_metrics_latex,
)
from evaluation.validation import load_and_split


def load_prediction_frames(
    spec: EvaluationSpec,
    results_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load only existing predictions for the requested pipeline."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob(f"predictions_{spec.pipeline}_*.parquet")):
        model = path.stem.replace(f"predictions_{spec.pipeline}_", "")
        frames[model] = canonicalize_prediction_columns(
            pd.read_parquet(path), spec
        )
    if not frames:
        raise FileNotFoundError(
            f"No prediction files found for pipeline {spec.pipeline} in {results_dir}."
        )
    return frames


def _write_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_latex(
        path,
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
        caption=caption,
        label=label,
    )


def _enrich_predictions(
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    spec: EvaluationSpec,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Attach real subgroup columns and training-derived volume tiers."""
    base = canonicalize_prediction_columns(frame, spec)
    source = canonicalize_subgroup_columns(canonicalize_data_columns(raw, spec), spec)
    columns = list(spec.series_id_columns) + ["date"]
    columns.extend(
        column
        for column in spec.subgroup_columns
        if column in source.columns
        and column not in columns
        and column not in {"volume_tier", "item_volume_tier"}
    )
    source = source[columns].drop_duplicates(
        [*spec.series_id_columns, "date"]
    )
    result = base.merge(
        source,
        on=[*spec.series_id_columns, "date"],
        how="left",
        validate="one_to_one",
    )
    result = add_volume_tiers(train, result, spec)
    return canonicalize_subgroup_columns(result, spec)


def _evaluate_error_analysis(
    frames: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    spec: EvaluationSpec,
    train: pd.DataFrame,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    subgroup_frames = []
    for model, frame in frames.items():
        enriched = _enrich_predictions(frame, raw, spec, train)
        enriched["model"] = model
        subgroup_frames.append(
            calculate_subgroup_errors(enriched, spec, spec.subgroup_columns).assign(
                model=model
            )
        )
        worst_series = find_worst_series(enriched, spec, 10)
        worst_series.insert(0, "model", model)
        worst_series.to_csv(output_dir / f"worst_series_{model}.csv", index=False)
        worst_dates = find_worst_dates(enriched, 10)
        worst_dates.insert(0, "model", model)
        worst_dates.to_csv(output_dir / f"worst_dates_{model}.csv", index=False)
        plot_residuals_over_time(
            enriched,
            figure_dir / f"residuals_over_time_{model}.png",
        )
    subgroup_metrics = pd.concat(subgroup_frames, ignore_index=True)
    subgroup_metrics.to_csv(output_dir / f"subgroup_metrics_{spec.pipeline}.csv", index=False)
    _write_latex(
        subgroup_metrics,
        output_dir / f"subgroup_metrics_{spec.pipeline}.tex",
        f"{spec.pipeline} subgroup forecast errors",
        f"tab:subgroup-errors-{spec.pipeline}",
    )
    bias_gaps = pd.concat(
        [
            calculate_bias_gaps(group)
            for _, group in subgroup_metrics.groupby("model")
        ],
        keys=sorted(subgroup_metrics["model"].unique()),
        names=["model", "row"],
    ).reset_index(level="model")
    bias_gaps.to_csv(output_dir / f"bias_gaps_{spec.pipeline}.csv", index=False)
    _write_latex(
        bias_gaps,
        output_dir / f"bias_gaps_{spec.pipeline}.tex",
        f"{spec.pipeline} subgroup bias gaps",
        f"tab:bias-gaps-{spec.pipeline}",
    )
    plot_subgroup_errors(subgroup_metrics, figure_dir)


def evaluate_pipeline(
    pipeline: str,
    root: Path,
    run_cv: bool = False,
) -> dict[str, Any]:
    """Evaluate available artifacts for one pipeline and return generated tables."""
    spec = get_evaluation_spec(pipeline, root)
    output_dir = root / "results" / "phase4"
    figure_dir = root / "report" / "figures" / "phase4"
    results_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(spec.processed_path)
    splits = load_and_split(spec, output_dir)
    frames = load_prediction_frames(spec, results_dir)

    metrics = evaluate_models(frames, spec)
    save_metrics_csv(metrics, output_dir / f"metrics_{spec.pipeline}.csv")
    save_metrics_latex(
        metrics,
        output_dir / f"metrics_{spec.pipeline}.tex",
        f"{spec.pipeline} test forecast metrics",
        f"tab:metrics-{spec.pipeline}",
    )

    curve_paths = sorted(results_dir.glob(f"curves_{spec.pipeline}_*.csv"))
    plot_loss_curves(curve_paths, figure_dir)
    combined = pd.concat(frames.values(), ignore_index=True)
    plot_prediction_bands(next(iter(frames.values())), spec, figure_dir)
    plot_coverage_calibration(metrics, figure_dir / f"coverage_{spec.pipeline}.png")
    plot_residual_histogram(combined, figure_dir / f"residuals_{spec.pipeline}.png")
    plot_model_comparison(metrics, figure_dir / f"model_comparison_{spec.pipeline}.png")

    _evaluate_error_analysis(
        frames,
        raw,
        spec,
        splits.train,
        output_dir,
        figure_dir,
    )

    validation_paths = sorted(
        results_dir.glob(f"validation_predictions_{spec.pipeline}_*.parquet")
    )
    if not validation_paths:
        print(
            "Stockout classification skipped: validation prediction artifacts are "
            "required to choose multiplier k; test data was not used."
        )
    else:
        validation_frames = {
            path.stem.replace(f"validation_predictions_{spec.pipeline}_", ""): pd.read_parquet(path)
            for path in validation_paths
        }
        k = choose_stock_multiplier(
            validation_frames,
            splits.validation,
            spec,
            horizon=7,
            history_sales=raw,
        )
        print(f"Pipeline {spec.pipeline}: chosen stockout multiplier k = {k:.3f}")
        labeled_by_model: dict[str, pd.DataFrame] = {}
        for model_name, test_frame in frames.items():
            test_basis = compute_origin_known_stock(raw, test_frame, spec, horizon=7)
            labeled_by_model[model_name] = apply_stockout_labels(test_frame, test_basis, k)
        
        classification_metrics = evaluate_stockout_alerts(labeled_by_model, spec.pipeline)
        classification_metrics.to_csv(output_dir / f"classification_{spec.pipeline}.csv", index=False)
        save_classification_latex(
            classification_metrics,
            output_dir / f"classification_{spec.pipeline}.tex",
            f"{spec.pipeline} stockout classification metrics",
            f"tab:classification-{spec.pipeline}",
        )
        plot_roc_curves(labeled_by_model, figure_dir / f"roc_{spec.pipeline}.png")
        plot_precision_recall_curves(labeled_by_model, figure_dir / f"pr_curve_{spec.pipeline}.png")
        plot_confusion_matrices(labeled_by_model, figure_dir / f"confusion_{spec.pipeline}.png")
    if run_cv:
        print(
            "CV requested, but fold predictions are not available. "
            "No test-based tuning or fabricated CV metrics were produced."
        )
    return {"metrics": metrics, "split_summary": pd.read_csv(output_dir / "split_summary.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 evaluation.")
    parser.add_argument("--pipeline", choices=("A", "B"), required=True)
    parser.add_argument("--cv", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = evaluate_pipeline(args.pipeline, root, args.cv)
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
