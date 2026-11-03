"""Apply the non-destructive Midnight/Cobalt/Lime UI redesign to VibeGuard Cloud.

Run from the existing Cloud-Computing-Final repository root:
    python apply_redesign.py

Creates app.py.before-ui-redesign.bak (do NOT commit that backup).
Does not alter trained checkpoints, evaluation metrics or AWS provisioning.
"""
from __future__ import annotations

import ast
from pathlib import Path


def replace_between(text: str, left: str, right: str, replacement: str) -> str:
    if left not in text or right not in text[text.index(left) + len(left):]:
        raise RuntimeError(f"Could not find original app.py boundary: {left!r} -> {right!r}")
    i = text.index(left)
    j = text.index(right, i+len(left))
    return text[:i] + replacement + text[j:]


def replace_one(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {description} match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    p=Path('app.py')
    if not p.is_file():
        raise SystemExit('Open a terminal in the original Cloud-Computing-Final repository root.')
    if not Path('ui_design.py').is_file():
        raise SystemExit('Copy ui_design.py into repository root beside app.py first.')
    original=p.read_text(encoding='utf-8')
    if 'from ui_design import (' in original:
        raise SystemExit('The dashboard already imports ui_design; refusing to apply the patch twice.')

    source=replace_one(original,'import streamlit as st\n','''import streamlit as st

from ui_design import (
    mount_theme, hero, scope_note, section, sidebar_status,
    metrics_row, result_hero, provenance_card, eval_limitation_note,
    score_plot, signal_plot, confusion_plot, training_plot, architecture,
)
''','Streamlit import')

    # Keep the exported plot function and exact numerical calculations, but change its styling.
    source=replace_between(source,'def render_waveform_and_spectrum(', '\n\ndef main() -> None:', '''def render_waveform_and_spectrum(
    raw_signal: np.ndarray,
    first_window_norm: np.ndarray,
    sampling_hz: int = NOMINAL_SAMPLING_HZ,
) -> plt.Figure:
    """Dark-themed three-panel vibration analysis; data are unchanged."""
    return signal_plot(raw_signal, first_window_norm, sampling_hz)
''')

    # Replace only the page heading + helper note. Metrics and underlying logic stay intact.
    source=replace_between(source,'    st.title("⚙️ VibeGuard Cloud:', '    metrics_data = load_json_if_exists(METRICS_PATH)', '''    mount_theme()
    hero()
    scope_note()

''')

    # Replace sidebar presentation, not the S3 probe or metrics functions.
    source=replace_between(source,'    with st.sidebar:\n', '    tab_infer, tab_eval, tab_arch = st.tabs(', '''    with st.sidebar:
        checkpoint_sha = compute_sha256(MODEL_PATH)[:16] if MODEL_PATH.exists() else None
        sidebar_cnn = metrics_data.get("cnn_1d", {}).get("test_3hp", {}) if metrics_data else {}
        sidebar_rf = metrics_data.get("random_forest_baseline", {}).get("test_3hp", {}) if metrics_data else {}
        sidebar_status(
            model_sha=checkpoint_sha,
            run_id=metrics_data.get("run_id", "N/A") if metrics_data else "N/A",
            epoch=metrics_data.get("selected_epoch", "N/A") if metrics_data else "N/A",
            cnn_f1=float(sidebar_cnn["macro_f1"]) if "macro_f1" in sidebar_cnn else None,
            rf_f1=float(sidebar_rf["macro_f1"]) if "macro_f1" in sidebar_rf else None,
            s3_connected=bool(s3_status["connected"]),
            s3_message=str(s3_status["message"]),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD,
        )

''')

    for old,new in [
        ('"1. Signal Upload & Fault Inference"','"01  /  Diagnose"'),
        ('"2. Held-Out 3 HP Evaluation & Baseline Comparison"','"02  /  Evaluate"'),
        ('"3. Architecture, Provenance & Limitations"','"03  /  Architecture"'),
        ('            st.subheader("Upload Vibration Recording")','            st.markdown("##### Upload a vibration recording")'),
        ('            st.subheader("Or Select a Held-Out 3 HP Sample")','            st.markdown("##### Select a verified 3 HP sample")'),
        ('                options=list(HELD_OUT_DEMO_FILES.keys()),','                options=list(HELD_OUT_DEMO_FILES.keys()),\n                index=2,'),
    ]:
        source=replace_one(source,old,new,old)

    source=replace_one(source,'    with tab_infer:\n        col_in1, col_in2 = st.columns([1.2, 1.0])', '''    with tab_infer:
        section("01 / LIVE INFERENCE", "What is your motor telling you?", "Choose a verified CWRU recording or upload a compatible single-channel vibration signal.")
        col_in1, col_in2 = st.columns([1.2, 1.0])''','infer section')

    source=replace_one(source,'''                    st.divider()
                    st.subheader(f"Diagnostic Result for `{source_name}`")''', '''                    section("DIAGNOSTIC OUTPUT", "Signal analysis complete", "Predictions shown below were computed from the loaded, unchanged TinyFaultCNN checkpoint.")''', 'diagnostic heading')

    source=replace_between(source,'                    m1, m2, m3, m4 = st.columns(4)', '                    score_col, meta_col = st.columns([1.2, 1.0])', '''                    result_hero(
                        predicted_display=pred_result["predicted_display"],
                        source_file=source_name,
                        manual_review=bool(pred_result["manual_review_required"]),
                        note="3 HP held-out CWRU recording" if demo_meta is not None else "User-supplied recording",
                    )
                    metrics_row([
                        ("Top model score", f"{pred_result['top_uncalibrated_score']:.4f}", "Uncalibrated softmax"),
                        ("Segment agreement", f"{pred_result['segment_agreement']:.1%}", "Window-level consistency"),
                        ("Analyzed windows", str(pred_result["num_windows"]), f"{WINDOW_SIZE:,} samples per window"),
                        ("Operating load", f"{demo_meta['load_hp']} HP" if demo_meta else "Not specified", "Held-out test" if demo_meta else "Uploaded recording"),
                    ], featured=0)

''')

    source=replace_between(source,'                    score_col, meta_col = st.columns([1.2, 1.0])', '                    st.markdown("#### Vibration Waveform & Spectral Inspection (12 kHz Nominal)")', '''                    score_col, meta_col = st.columns([1.2, 1.0])
                    with score_col:
                        section("MODEL OUTPUT", "Four-class score comparison", "Scores are not calibrated probabilities.")
                        score_fig = score_plot(pred_result["mean_scores"], CLASS_DISPLAY_NAMES, CLASS_NAMES)
                        st.pyplot(score_fig, clear_figure=True)
                        plt.close(score_fig)
                        scores_df = pd.DataFrame(
                            [
                                {
                                    "Class Key": cls_name,
                                    "Bearing Condition": CLASS_DISPLAY_NAMES[cls_name],
                                    "Mean Uncalibrated Softmax Score": pred_result["mean_scores"][cls_name],
                                    "Winning Windows": pred_result["per_class_window_counts"][cls_name],
                                }
                                for cls_name in CLASS_NAMES
                            ]
                        )
                        st.dataframe(scores_df, width="stretch", hide_index=True)

                    with meta_col:
                        section("SIGNAL PROVENANCE", "Recording & preprocessing metadata", "Verified raw data, channel and window parameters.")
                        provenance_card(
                            source_file=source_name,
                            channel_key=str(sig_meta["channel_key"]),
                            sampling_hz=int(sig_meta["sampling_hz_nominal"]),
                            window_size=WINDOW_SIZE,
                            num_windows=int(seg_meta["num_windows"]),
                            manual_review=bool(pred_result["manual_review_required"]),
                        )
                        with st.expander("Inspect full provenance & preprocessing JSON"):
                            st.json(
                                {
                                    "source_file": source_name,
                                    "channel_key": sig_meta["channel_key"],
                                    "raw_samples": sig_meta["num_samples"],
                                    "recorded_rpm": sig_meta["recorded_rpm"],
                                    "nominal_sampling_hz": sig_meta["sampling_hz_nominal"],
                                    "windows_extracted": seg_meta["num_windows"],
                                    "sample_index_span": f"[0, {seg_meta['end_sample']})",
                                    "normalization": "Per-window z-score, clamped [-10, 10]",
                                    "calibration_notice": pred_result["calibration_note"],
                                }
                            )

''')
    source=replace_one(source,'                    st.markdown("#### Vibration Waveform & Spectral Inspection (12 kHz Nominal)")', '                    section("SIGNAL FORENSICS", "Vibration & frequency analysis", "Three complementary views of the same recording. Nominal sample rate: 12 kHz.")','signal heading')

    source=replace_one(source,'        st.subheader("Locked Cross-Load Evaluation (Trained on 0 & 1 HP → Validated on 2 HP → Tested on 3 HP)")', '''        section("02 / MODEL EVALUATION", "Performance under load shift", "Training: 0 & 1 HP · Validation: 2 HP · Final held-out test: 3 HP. Scores come from frozen evaluation artifacts.")''','evaluation heading')
    source=replace_between(source,'            c1, c2, c3, c4 = st.columns(4)', '            st.markdown("#### Recording-Level Predictions on Held-Out 3 HP Test Split (4 Recordings)")', '''            metrics_row([
                ("CNN · Validation macro-F1", f"{cnn_val['macro_f1']:.4f}", "2 HP unseen validation"),
                ("CNN · Test macro-F1", f"{cnn_test['macro_f1']:.4f}", "3 HP locked test"),
                ("RF · Validation macro-F1", f"{rf_val['macro_f1']:.4f}", "2 HP unseen validation"),
                ("RF · Test macro-F1", f"{rf_test['macro_f1']:.4f}", "3 HP locked test"),
            ], featured=1)
            eval_limitation_note()

''')
    source=replace_one(source,'            st.markdown("#### Recording-Level Predictions on Held-Out 3 HP Test Split (4 Recordings)")', '            section("RECORDING AUDIT", "Four held-out test recordings", "The same four recordings were never included in training or validation selection.")','table heading')

    source=replace_between(source,'            fig_col1, fig_col2 = st.columns(2)', '\n    with tab_arch:', '''            section("DIAGNOSTIC EVIDENCE", "Confusion matrices", "Actual versus predicted windows, on the locked 3 HP test set.")
            confusion_fig = confusion_plot(
                cnn_test["confusion_matrix"], rf_test["confusion_matrix"], cnn_test["labels"]
            )
            st.pyplot(confusion_fig, clear_figure=True)
            plt.close(confusion_fig)

            history_data = load_json_if_exists(Path("outputs/training_history.json"))
            if history_data:
                section("TRAINING HISTORY", "Learning dynamics", "Validation data were used for model selection; final test data were held out.")
                history_fig = training_plot(history_data)
                st.pyplot(history_fig, clear_figure=True)
                plt.close(history_fig)

''')

    source=replace_one(source,'        st.subheader("System Architecture, Dataset Provenance & Methodological Limitations")', '''        section("03 / SYSTEM ARCHITECTURE", "From raw signal to diagnosis", "One reproducible pipeline, two purpose-built deployment environments.")
        architecture()''','architecture heading')

    source=replace_one(source,'''        if arch_img.exists():
            st.image(str(arch_img), caption="VibeGuard Cloud End-to-End Architecture")''','''        if arch_img.exists():
            with st.expander("View original technical architecture figure"):
                st.image(str(arch_img), caption="Original AWS end-to-end architecture")''','architecture image')

    # Fail closed if the resulting Python is syntactically incorrect.
    ast.parse(source,filename='app.py')
    backup=Path('app.py.before-ui-redesign.bak')
    backup.write_text(original,encoding='utf-8')
    ignore_path = Path('.gitignore')
    existing_ignore = ignore_path.read_text(encoding='utf-8') if ignore_path.exists() else ''
    if 'app.py.before-ui-redesign.bak' not in existing_ignore:
        ignore_path.write_text(existing_ignore.rstrip() + '\napp.py.before-ui-redesign.bak\n', encoding='utf-8')
    p.write_text(source,encoding='utf-8')

    cfg=Path('.streamlit/config.toml')
    cfg.parent.mkdir(parents=True,exist_ok=True)
    old=cfg.read_text(encoding='utf-8') if cfg.exists() else ''
    theme='''
[theme]
base = "dark"
primaryColor = "#D4FF24"
backgroundColor = "#030407"
secondaryBackgroundColor = "#0B1220"
textColor = "#F6F8FF"
'''
    if '[theme]' not in old:
        cfg.write_text(old.rstrip()+'\n'+theme,encoding='utf-8')
    else:
        print('Existing [theme] found: review manually instead of overwriting.')
    print('Redesign patch applied. Review git diff before committing; do not commit app.py.before-ui-redesign.bak.')


if __name__ == '__main__':
    main()
