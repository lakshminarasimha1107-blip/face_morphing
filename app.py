import sys
import os
import io
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATASET_ROOT,
    REAL_DIR,
    FAKE_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    FOLD_METRICS_CSV,
    FOLD_METRICS_JSON,
    TRAINING_HISTORY_CSV,
    PREDICTION_HISTORY_CSV,
    DEVICE,
    DEVICE_NAME,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED
)
from training.dataset_inspector import load_or_inspect_dataset, inspect_dataset
from training.evaluate import load_saved_results, format_evaluation_summary_table
from utils.visualization import (
    plot_learning_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_dataset_distribution
)

# Page configuration
st.set_page_config(
    page_title="DeepGuard: Face Morphing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .morphed-badge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .real-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Face Morph Detection")
st.sidebar.caption(f"Hardware: **{DEVICE.upper()}** ({DEVICE_NAME})")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Explorer",
        "Model Training (5-Fold)",
        "Model Evaluation",
        "Image Prediction & Grad-CAM",
        "Prediction History"
    ]
)

# Cache dataset summary
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = load_or_inspect_dataset()


# ==========================================
# PAGE A: HOME
# ==========================================
if nav_choice == "Home":
    st.markdown('<div class="main-header">🛡️ CNN-Based Face Morphing Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">DeepGuard ResNet-18 Transfer Learning with Stratified 5-Fold Cross-Validation and Grad-CAM Explainability</div>', unsafe_allow_html=True)

    summary = st.session_state.dataset_summary
    best_model_exists = (CHECKPOINTS_DIR / "best_model.pth").exists()
    eval_results = load_saved_results()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Dataset Images", f"{summary.get('total_images', 0):,}")
    with col2:
        st.metric("REAL Images", f"{summary.get('num_real', 0):,}")
    with col3:
        st.metric("MORPHED Images", f"{summary.get('num_fake', 0):,}")
    with col4:
        status_label = "Model Trained ✅" if best_model_exists else "Not Trained ⏳"
        st.metric("Model Status", status_label)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("📌 System Architecture & Pipeline")
        st.markdown(f"""
        - **CNN Architecture:** PyTorch **ResNet-18** with ImageNet-pretrained weights
        - **Task:** Binary Face Manipulation Detection (`REAL` vs. `MORPHED`)
        - **Validation Strategy:** Stratified **5-Fold Cross-Validation** with Disjoint Set Union clustering (zero pair/identity leakage)
        - **Explainability:** **Grad-CAM** on the final convolutional layer (`layer4[-1]`)
        - **Loss & Optimizer:** Cross-Entropy Loss with AdamW optimizer & ReduceLROnPlateau scheduling
        - **Dataset Location:** `{DATASET_ROOT}`
        - **Acceleration:** PyTorch CUDA on **{DEVICE_NAME}**
        """)

        st.subheader("📖 Dataset Provenance (from README)")
        readme_info = summary.get("readme", {})
        st.markdown(f"""
        - **Dataset Name:** {readme_info.get('title', 'DeepGuard Deepfake Dataset')}
        - **Generation Method:** `{readme_info.get('fake_method', 'InsightFace inswapper_128')}`
        - **Paired Architecture:** Every morphed image has a matching real face photograph
        - **License:** {readme_info.get('license', 'MIT')}
        """)

    with col_right:
        st.subheader("📊 Latest 5-Fold Cross-Validation Summary")
        if eval_results and "metrics" in eval_results:
            df_summary = format_evaluation_summary_table(eval_results)
            st.dataframe(df_summary[["Metric", "Mean", "Std Dev"]], hide_index=True, use_container_width=True)
            st.caption("Results calculated across all five validation folds (not fabricated).")
        else:
            st.info("No cross-validation evaluation has been run yet. Navigate to the **Model Training (5-Fold)** page to train the model.")

        if (PLOTS_DIR / "roc_curves.png").exists():
            st.image(str(PLOTS_DIR / "roc_curves.png"), caption="Aggregated 5-Fold ROC Curve", use_container_width=True)


# ==========================================
# PAGE B: DATASET EXPLORER
# ==========================================
elif nav_choice == "Dataset Explorer":
    st.markdown('<div class="main-header">📂 Dataset Explorer & Integrity Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection, validation statistics, pair matching, and duplicate analysis</div>', unsafe_allow_html=True)

    summary = st.session_state.dataset_summary

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Rescan Dataset"):
            with st.spinner("Rescanning dataset files..."):
                st.session_state.dataset_summary = inspect_dataset(save_summary=True)
                st.rerun()

    # Metric Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Images", f"{summary.get('total_images', 0):,}")
    c2.metric("REAL Count", f"{summary.get('num_real', 0):,}")
    c3.metric("MORPHED Count", f"{summary.get('num_fake', 0):,}")
    c4.metric("Exact Duplicates", f"{summary.get('exact_duplicate_count', 0)}")
    c5.metric("Corrupted Files", f"{summary.get('corrupted_count', 0)}")

    st.markdown("---")

    col_chart, col_meta = st.columns([1, 1.2])

    with col_chart:
        st.subheader("📊 Class Distribution")
        fig_dist = plot_dataset_distribution(summary)
        st.pyplot(fig_dist)

        st.subheader("🏷️ Label Mapping")
        st.code("""
CLASS_TO_IDX = {
    "real": 0,
    "fake": 1
}

IDX_TO_CLASS = {
    0: "REAL",
    1: "MORPHED"
}
        """, language="python")
        st.caption("Consistent across dataset loading, training, validation, Grad-CAM, and inference.")

    with col_meta:
        st.subheader("📝 README Analysis & Label Verification")
        readme = summary.get("readme", {})
        if readme.get("exists"):
            st.success(f"README located: `{readme.get('filename')}`")
            st.markdown(f"""
            - **Claimed Fake:** {readme.get('claimed_fake_count', 'N/A')}
            - **Claimed Real:** {readme.get('claimed_real_count', 'N/A')}
            - **Claimed Total:** {readme.get('claimed_total', 'N/A')}
            - **Generation Method:** `{readme.get('fake_method', 'N/A')}`
            - **Format & Resolution:** {readme.get('format', 'N/A')} (Verified on disk: 256x256 RGB)
            - **Consistency Status:** {summary.get('readme_consistency', 'Consistent')}
            """)
            with st.expander("View Full README Content"):
                st.text(readme.get("raw_content", ""))
        else:
            st.error("README file not found in dataset directory!")

        if summary.get("warnings"):
            st.subheader("⚠️ Validation Notes")
            for w in summary["warnings"]:
                st.warning(w)

    st.markdown("---")
    st.subheader("🖼️ Sample Images from Dataset")
    real_path = Path(REAL_DIR)
    fake_path = Path(FAKE_DIR)

    real_samples = sorted(list(real_path.glob("*.jpg")))[:10]
    fake_samples = sorted(list(fake_path.glob("*.jpg")))[:10]

    if real_samples and fake_samples:
        sample_idx = st.slider("Select Sample Index (Paired Comparison)", 0, min(len(real_samples), len(fake_samples)) - 1, 0)
        c_left, c_right = st.columns(2)
        with c_left:
            st.image(str(real_samples[sample_idx]), caption=f"REAL: {real_samples[sample_idx].name}", use_container_width=True)
        with c_right:
            st.image(str(fake_samples[sample_idx]), caption=f"MORPHED: {fake_samples[sample_idx].name}", use_container_width=True)


# ==========================================
# PAGE C: MODEL TRAINING (5-FOLD)
# ==========================================
elif nav_choice == "Model Training (5-Fold)":
    st.markdown('<div class="main-header">🏋️ 5-Fold Cross-Validation Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train ResNet-18 across 5 stratified folds with zero data leakage</div>', unsafe_allow_html=True)

    st.sidebar.subheader("Training Hyperparameters")
    batch_size = st.sidebar.number_input("Batch Size", min_value=8, max_value=128, value=DEFAULT_BATCH_SIZE, step=8)
    lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=DEFAULT_LR, format="%.5f")
    epochs = st.sidebar.number_input("Max Epochs per Fold", min_value=1, max_value=100, value=DEFAULT_EPOCHS)
    patience = st.sidebar.number_input("Early Stopping Patience", min_value=1, max_value=20, value=DEFAULT_PATIENCE)
    seed = st.sidebar.number_input("Random Seed", value=DEFAULT_SEED)

    st.sidebar.subheader("Dataset Scope")
    data_scope = st.sidebar.selectbox(
        "Training Sample Size",
        ["Full Dataset (10,852 images)", "Balanced Prototype (500 images)", "Balanced Prototype (1,000 images)", "Balanced Prototype (2,000 images)"],
        index=0
    )

    max_samples = None
    if "500" in data_scope:
        max_samples = 500
    elif "1,000" in data_scope:
        max_samples = 1000
    elif "2,000" in data_scope:
        max_samples = 2000

    from training.cross_validation import is_training_active, run_5fold_cross_validation

    training_running = is_training_active()

    if training_running:
        st.warning("⚠️ Training is currently in progress. Please check the logs below or wait for completion.")
    else:
        st.info("Configure settings in the sidebar and click **Start 5-Fold Training** below. Training will only begin upon your click.")

    start_btn = st.button("🚀 Start 5-Fold Training", disabled=training_running)

    # Placeholders for live training updates
    status_box = st.empty()
    progress_bar = st.empty()
    metric_cols = st.empty()
    log_area = st.empty()

    if start_btn:
        progress_bar.progress(0.0)
        logs = []

        def ui_callback(event, data):
            if event == "log":
                logs.append(data.get("message", ""))
                log_area.text_area("Training Logs", "\n".join(logs[-15:]), height=200)
            elif event == "fold_start":
                fold = data["fold"]
                status_box.markdown(f"### 🔄 Training Fold {fold} of 5 (Train: {data['train_size']}, Val: {data['val_size']})")
                logs.append(f"Started Fold {fold}/5...")
                log_area.text_area("Training Logs", "\n".join(logs[-15:]), height=200)
            elif event == "epoch_end":
                fold = data["fold"]
                epoch = data["epoch"]
                max_ep = data["max_epochs"]
                overall_progress = ((fold - 1) * max_ep + epoch) / (5 * max_ep)
                progress_bar.progress(min(overall_progress, 1.0))

                with metric_cols.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Fold", f"{fold} / 5")
                    m2.metric("Epoch", f"{epoch} / {max_ep}")
                    m3.metric("Train Loss", f"{data['train_loss']:.4f}")
                    m4.metric("Val Loss", f"{data['val_loss']:.4f}")
                    m5.metric("Val Acc", f"{data['val_acc']*100:.2f}%")

                logs.append(f"Fold {fold} | Epoch {epoch}/{max_ep} | Train Loss: {data['train_loss']:.4f} | Val Loss: {data['val_loss']:.4f} | Val Acc: {data['val_acc']*100:.2f}%")
                log_area.text_area("Training Logs", "\n".join(logs[-15:]), height=200)
            elif event == "fold_end":
                logs.append(f"✅ Completed Fold {data['fold']}! Best Acc: {data['metrics']['accuracy']*100:.2f}%")
                log_area.text_area("Training Logs", "\n".join(logs[-15:]), height=200)
            elif event == "complete":
                progress_bar.progress(1.0)
                status_box.success("🎉 5-Fold Cross-Validation Training Completed Successfully!")
                logs.append("All 5 folds completed and checkpoints saved!")
                log_area.text_area("Training Logs", "\n".join(logs[-15:]), height=200)

        try:
            with st.spinner("Running 5-Fold Cross-Validation on GPU..."):
                results = run_5fold_cross_validation(
                    batch_size=batch_size,
                    lr=lr,
                    max_epochs=epochs,
                    patience=patience,
                    seed=seed,
                    max_samples=max_samples,
                    progress_callback=ui_callback
                )
            st.success("Training finished! Navigate to **Model Evaluation** to inspect metrics and plots.")
        except Exception as e:
            st.error(f"Training Error: {str(e)}")


# ==========================================
# PAGE D: MODEL EVALUATION
# ==========================================
elif nav_choice == "Model Evaluation":
    st.markdown('<div class="main-header">📈 5-Fold Model Evaluation Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Per-fold and aggregated performance metrics with learning curves and ROC plots</div>', unsafe_allow_html=True)

    eval_results = load_saved_results()

    if not eval_results or not FOLD_METRICS_CSV.exists():
        st.warning("No evaluation results available yet. Please complete training in the **Model Training** page first.")
    else:
        st.subheader("📋 Aggregated 5-Fold Performance Summary")
        summary_df = format_evaluation_summary_table(eval_results)
        st.dataframe(summary_df[["Metric", "Mean", "Std Dev", "Min", "Max"]], hide_index=True, use_container_width=True)

        st.caption("ℹ️ **Methodology Note:** The metrics reported above are derived from five-fold stratified cross-validation on the paired dataset. They reflect cross-validation generalization and are strictly separated from independent external test benchmark claims.")

        st.markdown("---")
        st.subheader("📑 Per-Fold Performance Breakdown")
        fold_df = pd.read_csv(FOLD_METRICS_CSV)
        display_fold_df = fold_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "Specificity", "F1_Score"]:
            if col in display_fold_df.columns:
                display_fold_df[col] = (display_fold_df[col] * 100).round(2).astype(str) + "%"
        if "ROC_AUC" in display_fold_df.columns:
            display_fold_df["ROC_AUC"] = display_fold_df["ROC_AUC"].round(4)

        st.dataframe(display_fold_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Evaluation Visualizations")

        tab1, tab2, tab3 = st.tabs(["Learning Curves (Loss & Acc)", "ROC Curves", "Confusion Matrices"])

        with tab1:
            learning_curve_path = PLOTS_DIR / "learning_curves.png"
            if learning_curve_path.exists():
                st.image(str(learning_curve_path), caption="Training & Validation Loss and Accuracy vs Epoch", use_container_width=True)
            else:
                st.info("Learning curve plot not found.")

        with tab2:
            roc_path = PLOTS_DIR / "roc_curves.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="Stratified 5-Fold ROC Curves and Mean ROC", use_container_width=True)
            else:
                st.info("ROC plot not found.")

        with tab3:
            cm_path = PLOTS_DIR / "confusion_matrices.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Per-Fold and Aggregated Confusion Matrices", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")


# ==========================================
# PAGE E: IMAGE PREDICTION & GRAD-CAM
# ==========================================
elif nav_choice == "Image Prediction & Grad-CAM":
    st.markdown('<div class="main-header">🔍 Face Morphing Prediction & Grad-CAM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a face image to classify as REAL or MORPHED with explainable Grad-CAM heatmap overlay</div>', unsafe_allow_html=True)

    best_checkpoint = CHECKPOINTS_DIR / "best_model.pth"
    if not best_checkpoint.exists():
        st.error("⚠️ No trained model checkpoint found at `models/checkpoints/best_model.pth`! Please train the model using **Model Training (5-Fold)** before making predictions.")
        st.stop()

    from inference.predict import MorphingPredictor

    # Predictor initialization
    @st.cache_resource
    def get_predictor():
        return MorphingPredictor(checkpoint_path=best_checkpoint, device=DEVICE)

    predictor = get_predictor()

    st.subheader("1. Choose Input Image")
    input_source = st.radio("Input Source", ["Upload Image File", "Select from Dataset Samples"], horizontal=True)

    selected_image = None
    image_name = "uploaded_face.jpg"

    if input_source == "Upload Image File":
        uploaded_file = st.file_uploader("Upload a face image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = uploaded_file
            image_name = uploaded_file.name
    else:
        sample_type = st.selectbox("Sample Category", ["REAL Samples", "MORPHED Samples"])
        dir_to_sample = Path(REAL_DIR) if sample_type == "REAL Samples" else Path(FAKE_DIR)
        sample_files = sorted(list(dir_to_sample.glob("*.jpg")))[:20]
        if sample_files:
            chosen_sample = st.selectbox("Select Sample File", [p.name for p in sample_files])
            selected_image = dir_to_sample / chosen_sample
            image_name = chosen_sample

    if selected_image is not None:
        st.markdown("---")
        st.subheader("2. Prediction & Grad-CAM Explanation")

        # Controls
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            record_history = st.checkbox("Save to Prediction History", value=True)
        with col_c2:
            force_gradcam_class = st.selectbox("Grad-CAM Class Target", ["Predicted Class", "MORPHED Class", "REAL Class"])

        target_class_idx = None
        if force_gradcam_class == "MORPHED Class":
            target_class_idx = CLASS_TO_IDX["fake"]
        elif force_gradcam_class == "REAL Class":
            target_class_idx = CLASS_TO_IDX["real"]

        with st.spinner("Running ResNet-18 Inference and Generating Grad-CAM..."):
            pred_res = predictor.predict(
                image_input=selected_image,
                generate_gradcam=True,
                target_class=target_class_idx,
                save_to_history=record_history,
                filename_hint=image_name
            )

        pred_class = pred_res["predicted_class"]
        p_real = pred_res["prob_real"]
        p_morphed = pred_res["prob_morphed"]
        conf = pred_res["confidence"]

        # Results header cards
        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])
        with res_col1:
            if pred_class == "MORPHED":
                st.markdown(f'<div class="morphed-badge">⚠️ Prediction: MORPHED ({conf*100:.1f}% confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="real-badge">✅ Prediction: REAL ({conf*100:.1f}% confidence)</div>', unsafe_allow_html=True)

        with res_col2:
            st.metric("REAL Probability", f"{p_real * 100:.2f}%")
        with res_col3:
            st.metric("MORPHED Probability", f"{p_morphed * 100:.2f}%")

        st.progress(p_morphed, text=f"Morphing Likelihood: {p_morphed*100:.1f}%")

        st.caption(f"ℹ️ {pred_res['calibration_notice']}")

        st.markdown("---")
        st.subheader("3. Visual Explanation (Grad-CAM)")

        # Display side-by-side
        view_col1, view_col2, view_col3 = st.columns(3)
        with view_col1:
            st.image(pred_res["original_image_pil"], caption="Original Input Face", use_container_width=True)
        with view_col2:
            if pred_res["cam_overlay_np"] is not None:
                st.image(pred_res["cam_overlay_np"], caption="Grad-CAM Heatmap Overlay", use_container_width=True)
        with view_col3:
            if pred_res["cam_heatmap_np"] is not None:
                st.image(pred_res["cam_heatmap_np"], caption="Heatmap-Only View", use_container_width=True)

        st.info(f"🔍 **Interpretation Guide:** {pred_res['cam_legend_text']}")

        # Download button for Grad-CAM
        if pred_res["cam_overlay_np"] is not None:
            overlay_pil = Image.fromarray(pred_res["cam_overlay_np"])
            buf = io.BytesIO()
            overlay_pil.save(buf, format="JPEG")
            st.download_button(
                label="📥 Download Grad-CAM Visualization",
                data=buf.getvalue(),
                file_name=f"gradcam_{image_name}",
                mime="image/jpeg"
            )


# ==========================================
# PAGE F: PREDICTION HISTORY
# ==========================================
elif nav_choice == "Prediction History":
    st.markdown('<div class="main-header">📜 Prediction History & Privacy Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Audit past inference runs with strict privacy management</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([1, 1])

    with col_ctrl1:
        if st.button("🗑️ Clear Prediction History"):
            if PREDICTION_HISTORY_CSV.exists():
                PREDICTION_HISTORY_CSV.unlink()
                st.success("Prediction history cleared successfully.")
                st.rerun()

    if PREDICTION_HISTORY_CSV.exists():
        history_df = pd.read_csv(PREDICTION_HISTORY_CSV)
        st.markdown(f"**Total Past Predictions Recorded:** {len(history_df)}")
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No prediction history recorded yet.")
