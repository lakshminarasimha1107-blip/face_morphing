import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.cross_validation import run_5fold_cross_validation
from inference.predict import MorphingPredictor
from config import CHECKPOINTS_DIR, PLOTS_DIR, FOLD_METRICS_CSV, DATASET_ROOT

def test_pipeline():
    print("==================================================")
    print("STEP 1: Running 5-Fold Stratified Cross-Validation")
    print("==================================================")

    def print_progress(event, data):
        if event == "fold_start":
            print(f"--> Starting Fold {data['fold']} of {data['total_folds']} (Train: {data['train_size']}, Val: {data['val_size']})")
        elif event == "epoch_end":
            print(f"    Epoch {data['epoch']}/{data['max_epochs']} | Train Loss: {data['train_loss']:.4f} | Val Loss: {data['val_loss']:.4f} | Val Acc: {data['val_acc']*100:.2f}% | AUC: {data['val_auc']:.4f}")
        elif event == "fold_end":
            print(f"--> Fold {data['fold']} Completed! Accuracy: {data['metrics']['accuracy']*100:.2f}%\n")

    results = run_5fold_cross_validation(
        batch_size=32,
        lr=1e-4,
        max_epochs=3,
        patience=2,
        seed=42,
        max_samples=400,  # 400 balanced images for rapid, verified end-to-end testing
        use_grouping=True,
        progress_callback=print_progress
    )

    print("\n==================================================")
    print("STEP 2: 5-Fold Aggregated Evaluation Metrics")
    print("==================================================")
    for metric, stats in results["metrics"].items():
        if metric != "roc_auc":
            print(f"  {metric:15s}: {stats['mean']*100:.2f}% \u00b1 {stats['std']*100:.2f}% (Min: {stats['min']*100:.2f}%, Max: {stats['max']*100:.2f}%)")
        else:
            print(f"  {metric:15s}: {stats['mean']:.4f} \u00b1 {stats['std']:.4f} (Min: {stats['min']:.4f}, Max: {stats['max']:.4f})")

    print("\n==================================================")
    print("STEP 3: Checkpoint & Visualization File Verification")
    print("==================================================")
    checkpoints = sorted([p.name for p in CHECKPOINTS_DIR.glob("*.pth")])
    print(f"Checkpoints created ({len(checkpoints)}): {checkpoints}")
    assert "best_model.pth" in checkpoints, "best_model.pth missing!"
    for i in range(1, 6):
        assert f"best_model_fold_{i}.pth" in checkpoints, f"Fold {i} checkpoint missing!"

    plots = sorted([p.name for p in PLOTS_DIR.glob("*.png")])
    print(f"Plots generated ({len(plots)}): {plots}")
    assert "learning_curves.png" in plots, "learning_curves.png missing!"
    assert "roc_curves.png" in plots, "roc_curves.png missing!"
    assert "confusion_matrices.png" in plots, "confusion_matrices.png missing!"

    print("\n==================================================")
    print("STEP 4: Real Model Inference and Grad-CAM Testing")
    print("==================================================")
    predictor = MorphingPredictor()

    real_sample = Path(DATASET_ROOT) / "real" / "real_00000.jpg"
    fake_sample = Path(DATASET_ROOT) / "fake" / "fake_00000.jpg"

    # Test REAL sample
    res_real = predictor.predict(
        real_sample,
        generate_gradcam=True,
        save_to_history=True,
        filename_hint="real_00000.jpg"
    )
    print(f"Image: real_00000.jpg -> Predicted: {res_real['predicted_class']}")
    print(f"  Confidence: {res_real['confidence']*100:.2f}% | Real Prob: {res_real['prob_real']*100:.2f}% | Morphed Prob: {res_real['prob_morphed']*100:.2f}%")
    print(f"  Grad-CAM overlay shape: {res_real['cam_overlay_np'].shape}")
    print(f"  Grad-CAM heatmap shape: {res_real['cam_heatmap_np'].shape}")
    assert res_real["cam_overlay_np"].shape == (224, 224, 3)

    # Test MORPHED sample
    res_fake = predictor.predict(
        fake_sample,
        generate_gradcam=True,
        save_to_history=True,
        filename_hint="fake_00000.jpg"
    )
    print(f"Image: fake_00000.jpg -> Predicted: {res_fake['predicted_class']}")
    print(f"  Confidence: {res_fake['confidence']*100:.2f}% | Real Prob: {res_fake['prob_real']*100:.2f}% | Morphed Prob: {res_fake['prob_morphed']*100:.2f}%")
    print(f"  Grad-CAM overlay shape: {res_fake['cam_overlay_np'].shape}")
    print(f"  Grad-CAM heatmap shape: {res_fake['cam_heatmap_np'].shape}")
    assert res_fake["cam_overlay_np"].shape == (224, 224, 3)

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
