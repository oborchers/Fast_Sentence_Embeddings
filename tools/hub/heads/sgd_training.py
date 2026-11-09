"""Fixed-epoch streaming updates shared across the predeclared alpha grid."""

from __future__ import annotations

import joblib
from sgd_data import batches, receipt
from sgd_model import create, update


def metadata(groups):
    return {key: {k: v for k, v in row.items() if k != "models"} for key, row in groups.items()}


def checkpoint(work, key, row, scaler, config):
    path = work / f"C{key}-epoch{row['epochs_completed']}.joblib"
    temporary = path.with_name(path.name + ".pending")
    joblib.dump(
        {
            "models": row["models"],
            "scaler": scaler,
            "labels": config["labels"],
            "C_reference": row["C_reference"],
            "alpha": row["alpha"],
            "epochs_completed": row["epochs_completed"],
            "rows_seen": row["rows_seen"],
        },
        temporary,
    )
    temporary.replace(path)
    row["checkpoint"] = path.name


def update_groups(groups, x, y, budget):
    for row in groups.values():
        if row["status"] == "invalid":
            continue
        try:
            update(row["models"], x, y, budget)
        except FloatingPointError:
            row["status"], row["reason"] = "invalid", "nonfinite coefficients"
            continue
        row["rows_seen"] += len(x)


def finish_epoch(groups, epoch, rows, work, scaler, config):
    for key, row in groups.items():
        if row["status"] == "invalid":
            continue
        expected = (epoch + 1) * rows
        if not all(model.t_ == expected + 1 for model in row["models"]):
            raise AssertionError("a label did not receive the entire epoch")
        row["epochs_completed"] = epoch + 1
        checkpoint(work, key, row, scaler, config)


def train(source, work, scaler, weights, config, budget, *, rows):
    groups = {
        str(c): {
            "models": create(config, weights, c),
            "C_reference": c,
            "alpha": 1 / (c * config["train_rows"]),
            "status": "not_finished",
            "rows_seen": 0,
            "epochs_completed": 0,
        }
        for c in config["C"]
    }
    for epoch in range(config["epochs"]):
        budget.check()
        current_rows = 0
        for x, y in batches(source, config, epoch, rows):
            budget.check()
            update_groups(groups, x, y, budget)
            current_rows += len(x)
            receipt(
                work / "progress.json",
                {
                    "phase": "training",
                    "epoch": epoch + 1,
                    "rows_this_epoch": current_rows,
                    "candidates": metadata(groups),
                },
            )
            del x, y
        if current_rows != rows:
            raise AssertionError("streaming epoch omitted or duplicated rows")
        finish_epoch(groups, epoch, rows, work, scaler, config)
        budget.check()
    for row in groups.values():
        if row["status"] != "invalid":
            row["status"] = "trained_not_evaluated"
    receipt(work / "progress.json", {"phase": "training_finished", "candidates": metadata(groups)})
    return groups
