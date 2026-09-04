# ============================================================
# stat_clinical_significance.py
# ============================================================
# Rewritten to replace the lost notebook, matching the actual
# columns in your CSVs:
#   ensemble_predictions.csv: subject_id, true_label, predicted_label,
#                              true_class, predicted_class,
#                              prob_Wake, prob_N1, prob_N2, prob_N3, prob_REM
#   per_seed_predictions.csv: seed, subject_id, true_label, predicted_label
#
# Save this file as "stat_clinical_significance.py" inside:
#   D:\22\AA\AA journal\code\student\stat_clinical_significance\
# (next to, or replacing, the old empty .ipynb). Then your original code:
#   sys.path.insert(0, SIG_MODULE_PATH)
#   import stat_clinical_significance as sig
# will work as-is -- no notebook loader needed.
# ============================================================
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from scipy import stats


# ------------------------------------------------------------
# 1) Load predictions + build label<->class name mapping
# ------------------------------------------------------------
def load_predictions_with_subjects(csv_path, subject_filter=None):
    """
    Loads ensemble_predictions.csv or per_seed_predictions.csv.
    If subject_filter is given (a list of subject_id), keeps only those
    subjects; pass None to keep everyone.
    """
    df = pd.read_csv(csv_path)
    if subject_filter is not None:
        df = df[df["subject_id"].isin(subject_filter)].reset_index(drop=True)
    return df


def _build_label_map(df):
    """
    Builds a numeric label <-> class name (Wake/N1/N2/N3/REM) mapping
    directly from the data, so we never have to hardcode 0/4 etc.
    Returns a dict like {"Wake": 0, "N1": 1, ...}.
    """
    name_to_num = {}
    if "true_label" in df.columns and "true_class" in df.columns:
        for num, name in df[["true_label", "true_class"]].drop_duplicates().values:
            name_to_num[name] = num
    if "predicted_label" in df.columns and "predicted_class" in df.columns:
        for num, name in df[["predicted_label", "predicted_class"]].drop_duplicates().values:
            name_to_num.setdefault(name, num)
    return name_to_num


# ------------------------------------------------------------
# 2) Per-subject performance metrics (Macro-F1, Kappa)
# ------------------------------------------------------------
def per_subject_metrics(df):
    """
    Computes macro-F1 and Cohen's kappa separately for each subject_id.
    Returns a DataFrame with columns: subject_id, macro_f1, kappa.
    """
    rows = []
    for sid, g in df.groupby("subject_id"):
        y_true = g["true_label"].values
        y_pred = g["predicted_label"].values
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        rows.append({"subject_id": sid, "macro_f1": f1, "kappa": kappa})
    return pd.DataFrame(rows)


def per_seed_subject_metrics(per_seed_csv_path):
    """
    From per_seed_predictions.csv, computes macro_f1 and kappa separately
    for every (seed, subject_id) combination.
    Columns: seed, subject_id, macro_f1, kappa
    """
    df = pd.read_csv(per_seed_csv_path)
    rows = []
    for (seed, sid), g in df.groupby(["seed", "subject_id"]):
        y_true = g["true_label"].values
        y_pred = g["predicted_label"].values
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        rows.append({"seed": seed, "subject_id": sid, "macro_f1": f1, "kappa": kappa})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 3) Paired statistical significance test
# ------------------------------------------------------------
def paired_test(values_b, values_a, label=""):
    """
    values_b, values_a: metric values for two conditions on the SAME
    subjects (e.g. macro_f1 for stage B and stage A, in matching subject
    order).

    Automatically checks normality (Shapiro-Wilk) and picks:
      - normal -> paired t-test (effect size: Cohen's d for paired samples)
      - not normal -> Wilcoxon signed-rank test (effect size: matched-pairs
        rank-biserial correlation)

    Returns a dict: test, p_value, effect_size, effect_size_name, n
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    diff = values_b - values_a
    n = len(diff)

    if n < 3:
        return {"test": "insufficient_data", "p_value": np.nan,
                "effect_size": np.nan, "effect_size_name": "n/a", "n": n}

    # normality of the paired differences
    try:
        _, p_norm = stats.shapiro(diff)
    except Exception:
        p_norm = 0.0  # if Shapiro fails, fall back to the non-parametric test

    if p_norm > 0.05:
        stat, p_value = stats.ttest_rel(values_b, values_a)
        sd_diff = diff.std(ddof=1)
        cohens_d = diff.mean() / sd_diff if sd_diff > 0 else 0.0
        return {"test": "paired t-test", "p_value": p_value,
                "effect_size": cohens_d, "effect_size_name": "cohen_d", "n": n}
    else:
        nonzero = diff[diff != 0]
        if len(nonzero) == 0:
            return {"test": "wilcoxon", "p_value": 1.0,
                    "effect_size": 0.0, "effect_size_name": "rank_biserial", "n": n}
        stat, p_value = stats.wilcoxon(values_b, values_a)
        n_pos = (nonzero > 0).sum()
        n_neg = (nonzero < 0).sum()
        rank_biserial = (n_pos - n_neg) / len(nonzero)
        return {"test": "wilcoxon", "p_value": p_value,
                "effect_size": rank_biserial, "effect_size_name": "rank_biserial", "n": n}


# ------------------------------------------------------------
# 4) Per-seed robustness check
# ------------------------------------------------------------
def per_seed_robustness_check(ps_a, ps_b, metric="macro_f1"):
    """
    ps_a, ps_b: output of per_seed_subject_metrics() for the two stages
    being compared.

    Does two things:
      1) Runs a separate subject-level paired test for each seed
      2) Runs one "seed-averaged" paired test using each subject's
         metric averaged across all seeds

    Returns a dict:
      seed_averaged_test, per_seed_tests (DataFrame),
      n_seeds_significant_same_direction, n_seeds_total
    """
    common_seeds = sorted(set(ps_a["seed"]).intersection(set(ps_b["seed"])))

    # ---- determine the overall direction from the seed-averaged diff ----
    a_avg = ps_a.groupby("subject_id")[metric].mean()
    b_avg = ps_b.groupby("subject_id")[metric].mean()
    common_subjects = a_avg.index.intersection(b_avg.index)
    a_avg = a_avg.loc[common_subjects]
    b_avg = b_avg.loc[common_subjects]
    sa_test = paired_test(b_avg.values, a_avg.values, f"seed-averaged {metric}")
    overall_sign = np.sign((b_avg - a_avg).mean())

    # ---- each seed separately ----
    per_seed_rows = []
    n_sig_same_direction = 0
    for seed in common_seeds:
        a_s = ps_a[ps_a["seed"] == seed].set_index("subject_id")[metric]
        b_s = ps_b[ps_b["seed"] == seed].set_index("subject_id")[metric]
        common = a_s.index.intersection(b_s.index)
        a_s, b_s = a_s.loc[common], b_s.loc[common]
        t = paired_test(b_s.values, a_s.values, f"seed {seed} {metric}")
        mean_diff = (b_s - a_s).mean()
        is_sig_same_dir = bool(t["p_value"] < 0.05 and np.sign(mean_diff) == overall_sign)
        if is_sig_same_dir:
            n_sig_same_direction += 1
        per_seed_rows.append({
            "seed": seed, "test": t["test"], "p_value": t["p_value"],
            "effect_size": t["effect_size"], "mean_diff": mean_diff,
            "significant_same_direction": is_sig_same_dir,
        })

    return {
        "seed_averaged_test": sa_test,
        "per_seed_tests": pd.DataFrame(per_seed_rows),
        "n_seeds_significant_same_direction": n_sig_same_direction,
        "n_seeds_total": len(common_seeds),
    }


# ------------------------------------------------------------
# 5) Clinical sleep parameters (per subject: TST, SE, WASO, REM latency)
# ------------------------------------------------------------
def per_subject_sleep_parameters(df, epoch_len_sec=30):
    """
    For each subject, computes standard clinical sleep parameters from
    both the true and predicted labels:
      TST  (Total Sleep Time, minutes)
      SE   (Sleep Efficiency, %)
      WASO (Wake After Sleep Onset, minutes)
      REM latency (time from sleep onset to the first REM epoch, minutes)

    Assumes row order within each subject reflects temporal (epoch) order --
    this holds as long as your ensemble_predictions.csv is already sorted
    by epoch, as in your sample output.
    """
    label_map = _build_label_map(df)
    wake_label = label_map.get("Wake")
    rem_label = label_map.get("REM")
    if wake_label is None or rem_label is None:
        raise ValueError(
            "Could not determine the label number for 'Wake' or 'REM' from "
            "the data -- check the true_class/predicted_class columns."
        )

    def calc_params(labels):
        labels = np.asarray(labels)
        n_epochs = len(labels)
        tib_min = n_epochs * epoch_len_sec / 60.0
        sleep_mask = labels != wake_label
        tst_min = sleep_mask.sum() * epoch_len_sec / 60.0
        se = 100.0 * tst_min / tib_min if tib_min > 0 else np.nan

        sleep_idx = np.where(sleep_mask)[0]
        if len(sleep_idx) > 0:
            onset = sleep_idx[0]
            waso_min = (labels[onset:] == wake_label).sum() * epoch_len_sec / 60.0
            rem_idx = np.where(labels == rem_label)[0]
            rem_idx = rem_idx[rem_idx > onset]
            rem_latency_min = (rem_idx[0] - onset) * epoch_len_sec / 60.0 if len(rem_idx) > 0 else np.nan
        else:
            waso_min = np.nan
            rem_latency_min = np.nan

        return tst_min, se, waso_min, rem_latency_min

    rows = []
    for sid, g in df.groupby("subject_id"):
        g = g.reset_index(drop=True)
        tst_t, se_t, waso_t, remlat_t = calc_params(g["true_label"].values)
        tst_p, se_p, waso_p, remlat_p = calc_params(g["predicted_label"].values)
        rows.append({
            "subject_id": sid,
            "TST_true": tst_t, "TST_pred": tst_p,
            "SE_true": se_t, "SE_pred": se_p,
            "WASO_true": waso_t, "WASO_pred": waso_p,
            "REM_latency_true": remlat_t, "REM_latency_pred": remlat_p,
        })
    return pd.DataFrame(rows)


def clinical_validity_table(sleep_params_df):
    """
    Takes the output of per_subject_sleep_parameters() and, for each
    parameter, computes bias (mean pred - true), MAE, RMSE, and
    correlation -- producing the summary table used as the "clinical
    validity" table in the thesis.
    """
    parameters = ["TST", "SE", "WASO", "REM_latency"]
    rows = []
    for p in parameters:
        true_col, pred_col = f"{p}_true", f"{p}_pred"
        if true_col not in sleep_params_df.columns:
            continue
        valid = sleep_params_df[[true_col, pred_col]].dropna()
        if len(valid) == 0:
            rows.append({"parameter": p, "n": 0, "bias": np.nan, "MAE": np.nan,
                         "RMSE": np.nan, "correlation": np.nan})
            continue
        diff = valid[pred_col] - valid[true_col]
        bias = diff.mean()
        mae = diff.abs().mean()
        rmse = np.sqrt((diff ** 2).mean())
        corr = valid[true_col].corr(valid[pred_col]) if len(valid) > 1 else np.nan
        rows.append({"parameter": p, "n": len(valid), "bias": bias,
                     "MAE": mae, "RMSE": rmse, "correlation": corr})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 6) Kappa benchmark (Landis & Koch, 1977)
# ------------------------------------------------------------
def kappa_benchmark_report(kappa_value):
    """
    Prints the overall Cohen's kappa and reports the agreement category
    per the Landis & Koch (1977) benchmark.
    """
    if kappa_value < 0:
        category = "Poor"
    elif kappa_value < 0.20:
        category = "Slight"
    elif kappa_value < 0.40:
        category = "Fair"
    elif kappa_value < 0.60:
        category = "Moderate"
    elif kappa_value < 0.80:
        category = "Substantial"
    else:
        category = "Almost Perfect"
    print(f"\nOverall Cohen's kappa: {kappa_value:.4f} -> {category} agreement "
          f"(Landis & Koch, 1977 benchmark)")
    return category