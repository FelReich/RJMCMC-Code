"""PDF table of final Brooks-Giudici variance estimates and their ratios."""

import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
SYMBOL = {
    "V_hat": r"$\hat{V}$",
    "W_c": r"$W_c$",
    "W_m": r"$W_m$",
    "W_m_W_c": r"$W_mW_c$",
    "B_m": r"$B_m$",
    "B_m_W_c": r"$B_mW_c$",
}
RATIO_SYMBOL = {
    ("V_hat", "W_c"): r"$\hat{V}/W_c$",
    ("W_m", "W_m_W_c"): r"$W_m/W_mW_c$",
    ("B_m", "B_m_W_c"): r"$B_m/B_mW_c$",
}


def load_rows():
    path = HERE / "convergence_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Run {HERE / 'convergence_diagnostics.py'} first to generate {path}"
        )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["model_subset"] == "fixed_progression"]
    if not rows:
        raise ValueError(f"No fixed-model diagnostics in {path}")
    for row in rows:
        numerator = float(row["numerator"])
        denominator = float(row["denominator"])
        ratio = float(row["variance_ratio"])
        if (denominator <= 0 or not all(map(math.isfinite,
                                            (numerator, denominator, ratio)))
                or not math.isclose(numerator / denominator, ratio,
                                    rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("Invalid variance ratio in diagnostics CSV")
    return rows


def model_label(row):
    models = json.loads(row["selected_models"])
    if models == list(range(models[0], models[-1] + 1)):
        return f"{models[0]}-{models[-1]}"
    return ", ".join(map(str, models))


def main():
    rows = load_rows()
    style()
    table_rows = []
    for row in rows:
        numerator = SYMBOL[row["numerator_name"]]
        denominator = SYMBOL[row["denominator_name"]]
        ratio = RATIO_SYMBOL[(row["numerator_name"], row["denominator_name"])]
        table_rows.append((
            row["statistic"], model_label(row),
            row["comparison"].replace("_", " ").title(),
            f"{numerator} = {float(row['numerator']):.4f}",
            f"{denominator} = {float(row['denominator']):.4f}",
            f"{ratio} = {float(row['variance_ratio']):.4f}",
        ))

    height = 0.31 * (len(table_rows) + 1) + 0.10
    figure, axis = plt.subplots(figsize=(12.7, height))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=("Statistic", "Models", "Comparison",
                   "Numerator", "Denominator", "Ratio"),
        colWidths=(0.07, 0.19, 0.15, 0.19, 0.20, 0.20),
        cellLoc="center", colLoc="center", bbox=(0.01, 0.02, 0.98, 0.96),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (index, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dde5")
        cell.set_linewidth(0.6)
        if index == 0:
            cell.set_facecolor("#e7edf3")
            cell.set_text_props(weight="bold")
        elif index % 2 == 0:
            cell.set_facecolor("#f8fafc")
    output = HERE / "convergence_diagnostics_table.pdf"
    figure.savefig(output, pad_inches=0.02)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
