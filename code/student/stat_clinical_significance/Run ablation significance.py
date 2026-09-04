# ============================================================
# S1 -> S6a FULL ABLATION CHAIN:
# Statistical and Clinical Significance Analysis
# ============================================================
#
# This script compares the complete KD ablation chain:
#
#   S1 -> S2
#   S2 -> S4
#   S4 -> S5
#   S5 -> S6a
#
# For each comparison, the script performs:
#
# 1. Ensemble-level paired significance tests
# 2. Per-seed robustness analysis
# 3. Clinical validity analysis for the later stage
# 4. Cohen's Kappa benchmark reporting
#
# Required files for every stage:
#   - ensemble_predictions.csv
#   - per_seed_predictions.csv
#
# This script imports the analysis functions from:
#   stat_clinical_significance.py
#
# Both Python files should be in the same folder.
# ============================================================


import os
import sys
import importlib.util
import pandas as pd


# ============================================================
# 1. IMPORT stat_clinical_significance.py
# ============================================================

# Full path of the Python module containing the analysis functions.
MODULE_PATH = (
    r"D:\22\AA\AA journal\code\student\stat_clinical_significance"
    r"\stat_clinical_significance.py"
)


def load_python_module(module_path, module_name="stat_clinical_significance"):
    """
    Load a Python file as a module using its complete file path.

    This is used instead of a normal import because the module is
    stored in a custom folder and may not be available in Python's
    default import path.
    """

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"\nPython module not found:\n{module_path}\n\n"
            "Please check MODULE_PATH and make sure the filename "
            "is correct."
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create an import specification for:\n{module_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Register the module so that it can be accessed normally.
    sys.modules[module_name] = module

    # Execute the Python file.
    spec.loader.exec_module(module)

    return module


# Load the statistical and clinical significance functions.
sig = load_python_module(MODULE_PATH)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Root folder containing all teacher-student experiment outputs.
TEACHER_STUDENT_DIR = (
    r"D:\22\AA\AA journal\evaluation\teacher-student"
)


# ============================================================
# 3. STAGE CONFIGURATION
# ============================================================
#
# Update the folder names below if your actual output folders
# have different names.
# ============================================================

STAGES = {
    "S1": {
        "name": "S1 (wearable baseline)",

        "ensemble_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s1_wearable_baseline",
            "ensemble_predictions.csv"
        ),

        "per_seed_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s1_wearable_baseline",
            "per_seed_predictions.csv"
        ),
    },

    "S2": {
        "name": "S2 (KD student)",

        "ensemble_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s2_kd_student",
            "ensemble_predictions.csv"
        ),

        "per_seed_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s2_kd_student",
            "per_seed_predictions.csv"
        ),
    },

    "S4": {
        "name": "S4 (KD + artifact-weighted CE)",

        "ensemble_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s4_artifact_weighted_kd",
            "ensemble_predictions.csv"
        ),

        "per_seed_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s4_artifact_weighted_kd",
            "per_seed_predictions.csv"
        ),
    },

    "S5": {
        "name": "S5 (context-window student)",

        "ensemble_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s5_studentC7",
            "ensemble_predictions.csv"
        ),

        "per_seed_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s5_studentC7",
            "per_seed_predictions.csv"
        ),
    },

    "S6a": {
        "name": "S6a (modality-aware KD)",

        "ensemble_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s6a_studentC7",
            "ensemble_predictions.csv"
        ),

        "per_seed_csv": os.path.join(
            TEACHER_STUDENT_DIR,
            "ensemble_s6a_studentC7",
            "per_seed_predictions.csv"
        ),
    },
}


# ============================================================
# 4. COMPARISON CHAIN
# ============================================================

COMPARISON_CHAIN = [
    ("S1", "S2"),
    ("S2", "S4"),
    ("S4", "S5"),
    ("S5", "S6a"),
]


# ============================================================
# 5. OUTPUT DIRECTORY
# ============================================================

OUT_DIR = os.path.join(
    TEACHER_STUDENT_DIR,
    "significance_analysis_S1_to_S6a"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 6. RUN ONE STAGE COMPARISON
# ============================================================

def run_one_comparison(stage_a_key, stage_b_key):
    """
    Compare two consecutive stages in the ablation chain.

    The later stage is tested against the earlier stage using:

    - Per-subject Macro-F1
    - Per-subject Cohen's Kappa
    - Paired significance tests
    - Per-seed robustness analysis
    - Clinical validity analysis
    """

    stage_a = STAGES[stage_a_key]
    stage_b = STAGES[stage_b_key]

    print("\n" + "#" * 78)
    print(f"#  {stage_a['name']}  -->  {stage_b['name']}")
    print("#" * 78)

    # --------------------------------------------------------
    # Check whether the required ensemble files exist.
    # --------------------------------------------------------

    required_files = [
        stage_a["ensemble_csv"],
        stage_b["ensemble_csv"]
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not os.path.exists(file_path)
    ]

    if missing_files:
        print("\nSKIPPED: The following files were not found:")

        for file_path in missing_files:
            print(f"  - {file_path}")

        print(
            "\nPlease run the corresponding ensemble scripts first."
        )

        return None

    # --------------------------------------------------------
    # Load ensemble predictions.
    # --------------------------------------------------------

    df_a = sig.load_predictions_with_subjects(
        stage_a["ensemble_csv"],
        None
    )

    df_b = sig.load_predictions_with_subjects(
        stage_b["ensemble_csv"],
        None
    )

    # --------------------------------------------------------
    # Calculate subject-level performance metrics.
    # --------------------------------------------------------

    subj_a = sig.per_subject_metrics(df_a)
    subj_b = sig.per_subject_metrics(df_b)

    # Merge the two stages using subject_id.
    merged = subj_a.merge(
        subj_b,
        on="subject_id",
        suffixes=("_A", "_B")
    )

    # --------------------------------------------------------
    # Paired test for Macro-F1.
    # --------------------------------------------------------

    f1_test = sig.paired_test(
        merged["macro_f1_B"].values,
        merged["macro_f1_A"].values,
        f"macro_f1 ({stage_b_key} - {stage_a_key})"
    )

    # --------------------------------------------------------
    # Paired test for Cohen's Kappa.
    # --------------------------------------------------------

    kappa_test = sig.paired_test(
        merged["kappa_B"].values,
        merged["kappa_A"].values,
        f"kappa ({stage_b_key} - {stage_a_key})"
    )

    # --------------------------------------------------------
    # Print ensemble-level Macro-F1 results.
    # --------------------------------------------------------

    mean_f1_a = merged["macro_f1_A"].mean()
    mean_f1_b = merged["macro_f1_B"].mean()
    f1_difference = mean_f1_b - mean_f1_a

    print(
        f"\nEnsemble-level per-subject Macro-F1:"
        f" {stage_a_key}={mean_f1_a:.4f}"
        f"  {stage_b_key}={mean_f1_b:.4f}"
        f"  difference={f1_difference:+.4f}"
    )

    print(
        f"  Test: {f1_test['test']}"
        f"  p={f1_test['p_value']:.4g}"
        f"  {f1_test.get('effect_size_name', '')}"
        f"={f1_test['effect_size']:.3f}"
        f"  n={f1_test['n']}"
    )

    # --------------------------------------------------------
    # Print ensemble-level Kappa results.
    # --------------------------------------------------------

    mean_kappa_a = merged["kappa_A"].mean()
    mean_kappa_b = merged["kappa_B"].mean()
    kappa_difference = mean_kappa_b - mean_kappa_a

    print(
        f"\nEnsemble-level per-subject Kappa:"
        f" {stage_a_key}={mean_kappa_a:.4f}"
        f"  {stage_b_key}={mean_kappa_b:.4f}"
        f"  difference={kappa_difference:+.4f}"
    )

    print(
        f"  Test: {kappa_test['test']}"
        f"  p={kappa_test['p_value']:.4g}"
        f"  {kappa_test.get('effect_size_name', '')}"
        f"={kappa_test['effect_size']:.3f}"
        f"  n={kappa_test['n']}"
    )

    # ========================================================
    # 7. PER-SEED ROBUSTNESS ANALYSIS
    # ========================================================

    robustness_row = None

    per_seed_a_exists = os.path.exists(
        stage_a["per_seed_csv"]
    )

    per_seed_b_exists = os.path.exists(
        stage_b["per_seed_csv"]
    )

    if per_seed_a_exists and per_seed_b_exists:

        ps_a = sig.per_seed_subject_metrics(
            stage_a["per_seed_csv"]
        )

        ps_b = sig.per_seed_subject_metrics(
            stage_b["per_seed_csv"]
        )

        robustness = sig.per_seed_robustness_check(
            ps_a,
            ps_b,
            metric="macro_f1"
        )

        seed_averaged_test = robustness["seed_averaged_test"]

        print(
            "\nPer-seed robustness analysis "
            "(seed-averaged per-subject Macro-F1):"
        )

        print(
            f"  Test: {seed_averaged_test['test']}"
            f"  p={seed_averaged_test['p_value']:.4g}"
            f"  {seed_averaged_test.get('effect_size_name', '')}"
            f"={seed_averaged_test['effect_size']:.3f}"
        )

        print(
            f"  {robustness['n_seeds_significant_same_direction']}"
            f"/{robustness['n_seeds_total']}"
            " individual seeds reached p < 0.05 "
            "in the same direction."
        )

        # Save individual per-seed test results.
        per_seed_output = os.path.join(
            OUT_DIR,
            f"{stage_a_key}_vs_{stage_b_key}_per_seed_tests.csv"
        )

        robustness["per_seed_tests"].to_csv(
            per_seed_output,
            index=False
        )

        robustness_row = {
            "seed_avg_p": seed_averaged_test["p_value"],
            "seed_avg_effect_size": seed_averaged_test["effect_size"],
            "n_seeds_significant": (
                robustness["n_seeds_significant_same_direction"]
            ),
            "n_seeds_total": robustness["n_seeds_total"],
        }

    else:

        print(
            "\nPer-seed robustness analysis skipped."
        )

        print(
            "The per_seed_predictions.csv file was not found "
            f"for {stage_a_key} and/or {stage_b_key}."
        )

    # ========================================================
    # 8. CLINICAL VALIDITY ANALYSIS
    # ========================================================
    #
    # Clinical validity is calculated for the later stage
    # against the PSG ground truth.
    # ========================================================

    sleep_parameters_b = sig.per_subject_sleep_parameters(
        df_b
    )

    clinical_table_b = sig.clinical_validity_table(
        sleep_parameters_b
    )

    clinical_output = os.path.join(
        OUT_DIR,
        f"{stage_b_key}_clinical_validity.csv"
    )

    clinical_table_b.to_csv(
        clinical_output,
        index=False
    )

    print(
        f"\nClinical validity -- {stage_b['name']} "
        "against PSG ground truth:"
    )

    print(
        clinical_table_b.to_string(index=False)
    )

    # --------------------------------------------------------
    # Cohen's Kappa benchmark report.
    # --------------------------------------------------------

    overall_kappa_b = subj_b["kappa"].mean()

    sig.kappa_benchmark_report(
        overall_kappa_b
    )

    # ========================================================
    # 9. CREATE SUMMARY ROW
    # ========================================================

    summary_row = {
        "comparison": f"{stage_a_key} -> {stage_b_key}",

        "macro_f1_A": mean_f1_a,
        "macro_f1_B": mean_f1_b,
        "macro_f1_diff": f1_difference,

        "macro_f1_test": f1_test["test"],
        "macro_f1_p": f1_test["p_value"],
        "macro_f1_effect_size": f1_test["effect_size"],

        "kappa_A": mean_kappa_a,
        "kappa_B": mean_kappa_b,
        "kappa_diff": kappa_difference,

        "kappa_test": kappa_test["test"],
        "kappa_p": kappa_test["p_value"],
        "kappa_effect_size": kappa_test["effect_size"],
    }

    if robustness_row is not None:
        summary_row.update(robustness_row)

    return summary_row


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():

    print(
        "\nRunning significance analysis across the "
        "S1 -> S6a ablation chain."
    )

    print(
        f"Comparisons: {COMPARISON_CHAIN}"
    )

    summary_rows = []

    # Run every consecutive comparison.
    for stage_a_key, stage_b_key in COMPARISON_CHAIN:

        result = run_one_comparison(
            stage_a_key,
            stage_b_key
        )

        if result is not None:
            summary_rows.append(result)

    # ========================================================
    # 11. SAVE FINAL SUMMARY
    # ========================================================

    if summary_rows:

        summary_df = pd.DataFrame(
            summary_rows
        )

        summary_csv = os.path.join(
            OUT_DIR,
            "S1_to_S6a_ablation_significance_summary.csv"
        )

        summary_df.to_csv(
            summary_csv,
            index=False
        )

        print("\n" + "=" * 78)
        print(
            "FULL ABLATION CHAIN SUMMARY"
        )
        print("=" * 78)

        print(
            summary_df.to_string(index=False)
        )

        print(
            f"\nSaved summary file:\n{summary_csv}"
        )

    else:

        print(
            "\nNo comparison could be completed."
        )

        print(
            "Please check whether all ensemble scripts have been "
            "run and whether the output paths in STAGES are correct."
        )

    print(
        f"\nAll outputs are saved in:\n{OUT_DIR}"
    )

    print("\nDone.")


# ============================================================
# 12. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()