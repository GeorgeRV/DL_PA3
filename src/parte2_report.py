from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrige os gráficos e o ganho por classe da Parte 2."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/parte2"),
    )
    return parser.parse_args()


def save_budget_curve(output_dir: Path) -> tuple[pd.DataFrame, float]:
    execution = pd.read_csv(output_dir / "test_public_final_por_execucao.csv")
    budget5 = execution[execution["budget"] == 5].copy()

    summary = execution.groupby(["method", "budget"], as_index=False).agg(
        test_f1_mean=("test_f1_macro", "mean"),
        test_f1_std=("test_f1_macro", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
    )
    summary["test_f1_std"] = summary["test_f1_std"].fillna(0)
    summary["test_accuracy_std"] = summary["test_accuracy_std"].fillna(0)
    summary.to_csv(output_dir / "curva_orcamento_test_public_corrigida.csv", index=False)

    ceiling_row = summary[
        (summary["method"] == "supervised") & (summary["budget"] == 20)
    ]
    if ceiling_row.empty:
        raise RuntimeError("Não existe resultado supervisionado no orçamento 20 para definir o teto.")
    ceiling = float(ceiling_row.iloc[0]["test_f1_mean"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    methods = [
        ("supervised", "Piso supervisionado"),
        ("fixmatch_aligned", "FixMatch aligned"),
    ]
    colors = {"supervised": "#1f77b4", "fixmatch_aligned": "#ff7f0e"}
    for method, label in methods:
        data = summary[summary["method"] == method].sort_values("budget")
        ax.errorbar(
            data["budget"],
            data["test_f1_mean"],
            yerr=data["test_f1_std"],
            marker="o",
            linewidth=2,
            markersize=7,
            capsize=5,
            color=colors[method],
            label=label,
        )

    ax.axhline(
        ceiling,
        color="#7a7a7a",
        linestyle="--",
        linewidth=2,
        label=f"Teto supervisionado (20 rótulos/classe): {ceiling:.3f}",
    )

    ax.set_xscale("log", base=10)
    ax.set_xticks([1, 5, 20], labels=["1", "5", "20"])
    ax.set_xlim(0.85, 24)
    ax.set_xlabel("Orçamento (rótulos por classe)")
    ax.set_ylabel("F1 Macro no test_public")
    ax.set_title("Curva de orçamento — piso e FixMatch aligned")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "curva_orcamento_test_public_corrigida.png", dpi=200)
    plt.close(fig)

    return budget5, ceiling


def save_class_gain(output_dir: Path) -> pd.DataFrame:
    per_class = pd.read_csv(output_dir / "test_public_f1_por_classe.csv")
    budget5 = per_class[per_class["budget"] == 5].copy()

    pivot = budget5.pivot(index=["seed", "class"], columns="method", values="test_f1")
    pivot = pivot.reset_index()
    required = {"supervised", "fixmatch_aligned"}
    if not required.issubset(pivot.columns):
        raise RuntimeError("Faltam resultados do piso ou do FixMatch no orçamento 5.")
    pivot["ganho"] = pivot["fixmatch_aligned"] - pivot["supervised"]

    summary = pivot.groupby("class", as_index=False).agg(
        fixmatch_f1_mean=("fixmatch_aligned", "mean"),
        fixmatch_f1_std=("fixmatch_aligned", "std"),
        supervised_f1_mean=("supervised", "mean"),
        supervised_f1_std=("supervised", "std"),
        ganho_medio=("ganho", "mean"),
        ganho_std=("ganho", "std"),
    )
    summary["ganho_std"] = summary["ganho_std"].fillna(0)
    summary["fixmatch_f1_std"] = summary["fixmatch_f1_std"].fillna(0)
    summary["supervised_f1_std"] = summary["supervised_f1_std"].fillna(0)
    summary["ganho_medio"] = summary["ganho_medio"].round(6)
    summary = summary.sort_values("ganho_medio", ascending=False)
    summary.to_csv(
        output_dir / "ganho_por_classe_orcamento_5_test_public.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2ca02c" if gain > 0 else "#d62728" for gain in summary["ganho_medio"]]
    ax.barh(
        summary["class"],
        summary["ganho_medio"],
        xerr=summary["ganho_std"],
        color=colors,
        capsize=2,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Ganho médio de F1 por classe — FixMatch - piso")
    ax.set_ylabel("Classe")
    ax.set_title("Ganho por classe no orçamento de 5 rótulos/classe")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ganho_por_classe_orcamento_5_test_public.png", dpi=200)
    plt.close(fig)
    return summary


def write_report(
    output_dir: Path,
    budget_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    ceiling: float,
) -> None:
    budget_summary = budget_summary.sort_values(["method", "budget"])
    top_gain = class_summary.head(5)
    no_gain = class_summary[class_summary["ganho_medio"] <= 0].sort_values("ganho_medio")

    lines = [
        "# Parte 2 — Curva de orçamento corrigida",
        "",
        "## Método de avaliação",
        "",
        "- Curva: F1 macro calculada no `test_public`, conjunto não usado no treinamento.",
        "- Piso: ResNet-18 supervisionado.",
        "- Método: FixMatch aligned.",
        "- Orçamento 5: média ± desvio padrão das 3 seeds 42, 100 e 200.",
        "- Orçamentos 1 e 20: seed 42.",
        "- Teto: linha horizontal no melhor F1 supervisionado disponível no orçamento máximo "
        "de 20 rótulos por classe.",
        "",
        "## Resultados no test_public",
        "",
    ]

    for method in ("supervised", "fixmatch_aligned"):
        label = "Piso supervisionado" if method == "supervised" else "FixMatch aligned"
        lines.append(f"### {label}")
        for row in budget_summary[budget_summary["method"] == method].itertuples():
            std = row.test_f1_std
            lines.append(
                f"- **{row.budget} rótulos/classe:** "
                f"{row.test_f1_mean:.4f}"
                + (f" ± {std:.4f}" if row.budget == 5 else "")
            )
        lines.append("")

    lines.extend([
        f"- Teto supervisionado usado na curva: **{ceiling:.4f}**.",
        "",
        "![Curva corrigida](curva_orcamento_test_public_corrigida.png)",
        "",
        "## Ganho por classe no orçamento 5",
        "",
        "O ganho foi calculado por seed e depois agregado: "
        "`ganho médio = média(F1 FixMatch seed - F1 piso seed)`.",
        "",
        "### Cinco maiores ganhos",
        "",
    ])
    for _, values in top_gain.iterrows():
        lines.append(
            f"- **{values['class']}:** ganho médio {values['ganho_medio']:+.4f} "
            f"(± {values['ganho_std']:.4f}); FixMatch {values['fixmatch_f1_mean']:.4f}; "
            f"Piso {values['supervised_f1_mean']:.4f}"
        )

    lines.extend(["", "### Classes sem ganho", ""])
    if no_gain.empty:
        lines.append("- Todas as classes apresentaram ganho médio positivo.")
    else:
        for _, values in no_gain.iterrows():
            lines.append(
                f"- **{values['class']}:** ganho médio {values['ganho_medio']:+.4f} "
                f"(± {values['ganho_std']:.4f}); FixMatch {values['fixmatch_f1_mean']:.4f}; "
                f"Piso {values['supervised_f1_mean']:.4f}"
            )

    lines.extend([
        "",
        "![Ganho por classe](ganho_por_classe_orcamento_5_test_public.png)",
        "",
        "## Arquivos corrigidos",
        "",
        "- `curva_orcamento_test_public_corrigida.csv`",
        "- `curva_orcamento_test_public_corrigida.png`",
        "- `ganho_por_classe_orcamento_5_test_public.csv`",
        "- `ganho_por_classe_orcamento_5_test_public.png`",
        "",
    ])
    (output_dir / "diagnostico_parte2.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    budget5, ceiling = save_budget_curve(args.output_dir)
    class_summary = save_class_gain(args.output_dir)
    budget_summary = pd.read_csv(
        args.output_dir / "curva_orcamento_test_public_corrigida.csv"
    )
    write_report(args.output_dir, budget_summary, class_summary, ceiling)
    print("Relatório corrigido:", args.output_dir)


if __name__ == "__main__":
    main()
