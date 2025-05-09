# classification_nested_cv.py (혹은 main 스크립트 맨 아래)
from utils.visualization import (
    plot_confusion_matrix,        # 1. Confusion Matrix
    plot_roc_curve,               # 2. ROC Curve
    plot_precision_recall_curve,  # 3. Precision–Recall Curve
    plot_calibration_curve        # 4. Calibration Curve
)

def main():
    # ... (기존 nested CV 학습/평가 로직)

    # --- 모든 모델 학습 & 평가가 끝난 뒤 (또는 fold 루프 안에서) ---
    for model_name in trained_models:
        model_dir = os.path.join(args.output_dir, model_name)

        # 1) 모든 fold를 합친 예측 결과 로드
        preds = pd.read_csv(os.path.join(model_dir, "all_folds_predictions.csv"), index_col=0)
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        y_prob = preds["y_prob"]

        # 2) Confusion Matrix
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=[0,1],  # 클래스 레이블
            normalize="true",  # 비율로 보고 싶으면 "true"
            title=f"{model_name} Confusion Matrix",
            output_path=os.path.join(model_dir, "confusion_matrix.png")
        )

        # 3) ROC Curve
        plot_roc_curve(
            y_true=y_true,
            y_score=y_prob,
            title=f"{model_name} ROC Curve",
            output_path=os.path.join(model_dir, "roc_curve.png")
        )

        # 4) Precision–Recall Curve
        plot_precision_recall_curve(
            y_true=y_true,
            y_score=y_prob,
            title=f"{model_name} Precision–Recall Curve",
            output_path=os.path.join(model_dir, "pr_curve.png")
        )

        # 5) Calibration Curve
        plot_calibration_curve(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=10,
            title=f"{model_name} Calibration Curve",
            output_path=os.path.join(model_dir, "calibration_curve.png")
        )