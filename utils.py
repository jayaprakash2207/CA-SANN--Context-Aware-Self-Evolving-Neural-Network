from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def line(values: list[str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    separator = "-+-".join("-" * width for width in widths)
    output = [line(headers), separator]
    output.extend(line(row) for row in rows)
    return "\n".join(output)


def render_research_analysis(
    aggregate_rows: list[dict[str, Any]],
    verdict_counts: dict[str, int],
) -> str:
    static_row = next((row for row in aggregate_rows if row["model"] == "Static"), None)
    sann_row = next((row for row in aggregate_rows if row["model"] == "SANN"), None)
    ca_row = next((row for row in aggregate_rows if row["model"] == "CA-SANN"), None)

    lines: list[str] = []
    lines.append("Research-Style Analysis")
    lines.append("======================")

    if static_row is None or sann_row is None:
        lines.append("Insufficient rows to compute comparative analysis.")
        return "\n".join(lines)

    acc_delta = float(sann_row["test_accuracy_mean"]) - float(static_row["test_accuracy_mean"])
    loss_delta = float(sann_row["test_loss_mean"]) - float(static_row["test_loss_mean"])
    size_ratio = float(sann_row["model_size_mean"]) / max(1.0, float(static_row["model_size_mean"]))

    lines.append("Key observations")
    lines.append(f"- SANN test accuracy delta vs static: {acc_delta:+.4f}")
    lines.append(f"- SANN test loss delta vs static: {loss_delta:+.6f}")
    lines.append(f"- SANN/Static mean model size ratio: {size_ratio:.3f}x")

    if ca_row is not None:
        ca_acc_delta = float(ca_row["test_accuracy_mean"]) - float(static_row["test_accuracy_mean"])
        ca_loss_delta = float(ca_row["test_loss_mean"]) - float(static_row["test_loss_mean"])
        ca_size_ratio = float(ca_row["model_size_mean"]) / max(1.0, float(static_row["model_size_mean"]))
        lines.append(f"- CA-SANN test accuracy delta vs static: {ca_acc_delta:+.4f}")
        lines.append(f"- CA-SANN test loss delta vs static: {ca_loss_delta:+.6f}")
        lines.append(f"- CA-SANN/Static mean model size ratio: {ca_size_ratio:.3f}x")
    lines.append(
        "- Verdict counts across seeds: "
        f"better={verdict_counts.get('better', 0)}, "
        f"similar={verdict_counts.get('similar', 0)}, "
        f"worse={verdict_counts.get('worse', 0)}"
    )

    lines.append("Growth and pruning impact")
    lines.append(
        f"- Mean SANN growth events: {float(sann_row['growth_events_mean']):.2f} "
        f"(+/- {float(sann_row['growth_events_std']):.2f})"
    )
    if ca_row is not None:
        lines.append(
            f"- Mean CA-SANN growth events: {float(ca_row['growth_events_mean']):.2f} "
            f"(+/- {float(ca_row['growth_events_std']):.2f})"
        )
    lines.append(
        "- If accuracy improves with a modest size ratio increase, dynamic growth appears beneficial. "
        "If size grows without accuracy gain, regularization and controller thresholds should be tightened."
    )

    lines.append("Potential weaknesses")
    lines.append("- Multi-seed variance may remain high on noisy labels and limited epoch budgets.")
    lines.append("- Forced-growth debug mode can bias conclusions and should be disabled for final claims.")

    lines.append("Actionable suggestions")
    lines.append("- Increase epochs for all seeds and compare early-stop checkpoints by validation loss.")
    lines.append("- Tune complexity penalty and max model parameter cap jointly.")
    lines.append("- Add confidence intervals via bootstrap over seed-level metrics.")

    return "\n".join(lines)
