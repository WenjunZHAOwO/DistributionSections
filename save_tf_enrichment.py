from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def save_tf_enrichment(
    gene_scores,
    trrust_path,
    output_file,
    score_column="regression_score_raw",
    gene_column="clean_name",
):
    """
    Reproduce the statistical calculation in STORIES tf_enrich.
    Save and return a table rather than only plotting it.

    Each gene-ID row is retained separately, including rows sharing
    the same cleaned gene name, as in STORIES.
    """
    if score_column not in gene_scores:
        raise KeyError(f"Missing score column: {score_column}")
    if gene_column not in gene_scores:
        raise KeyError(f"Missing gene-name column: {gene_column}")
    if not gene_scores.index.is_unique:
        raise ValueError("Duplicate gene IDs")

    table = gene_scores.copy()
    values = pd.to_numeric(
        table[score_column], errors="coerce"
    ).to_numpy(dtype=float)

    finite = np.isfinite(values)
    if not finite.all():
        print(
            f"Excluding {(~finite).sum()} nonfinite gene scores "
            f"from {Path(output_file).name}"
        )

    table = table.loc[finite].copy()
    values = values[finite]
    names = table[gene_column].fillna("").astype(str).to_numpy()

    if len(values) < 2:
        raise ValueError("Too few finite gene scores")

    trrust = pd.read_csv(
        trrust_path,
        sep="\t",
        header=None,
        names=["TF", "Target", "Mode", "References"],
    )

    # Match the original STORIES target filtering.
    trrust = trrust[
        trrust["Target"].isin(names)
    ].copy()

    if trrust.empty:
        raise ValueError(
            "No TRRUST targets match clean_name. "
            "Check the gene names and TRRUST file."
        )

    records = []

    for tf, group in trrust.groupby("TF", sort=False):
        target_mask = np.isin(
            names, group["Target"].unique()
        )
        target_scores = values[target_mask]
        background_scores = values[~target_mask]

        if len(target_scores) == 0 or len(background_scores) == 0:
            continue

        stat, p_value = ranksums(
            target_scores,
            background_scores,
            alternative="greater",
        )

        records.append({
            "TF": tf,
            "stat": float(stat),
            "p_value": float(p_value),
            "n_targets": len(target_scores),
            "n_nontargets": len(background_scores),
            "mean_target_score": float(target_scores.mean()),
            "mean_nontarget_score": float(background_scores.mean()),
            "score_column": score_column,
        })

    if not records:
        raise ValueError("No TFs could be tested")

    result = pd.DataFrame(records).set_index("TF")
    result["neglog10_p"] = -np.log10(
        result["p_value"].clip(lower=np.finfo(float).tiny)
    )
    result = result.sort_values(
        "p_value", kind="stable"
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )
    result.to_csv(temporary_file)
    temporary_file.replace(output_file)

    return result
