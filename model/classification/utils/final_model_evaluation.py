import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve

class ModelEvaluator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def evaluate_model(self, model_name, y_true, y_pred, y_pred_proba, metrics):
        """모델 평가 및 시각화"""
        # 1. Confusion Matrix
        self._plot_confusion_matrix(model_name, y_true, y_pred)
        
        # 2. Calibration Curve & Brier Score
        self._plot_calibration_curve(model_name, y_true, y_pred_proba)
        
        # 3. Threshold-Metric 곡선
        self._plot_threshold_metric_curves(model_name, y_true, y_pred_proba)
        
        # 4. 예측확률 분포
        self._plot_probability_distribution(model_name, y_true, y_pred_proba, metrics)
        
        # 5. 성능 지표 저장
        self._save_metrics(model_name, metrics)
    
    def _plot_confusion_matrix(self, model_name, y_true, y_pred):
        """Confusion Matrix 시각화"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
    
    def _plot_calibration_curve(self, model_name, y_true, y_pred_proba):
        """Calibration Curve 시각화"""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label=f'{model_name}')
        plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('True probability')
        plt.title(f'{model_name} Calibration Curve (Brier Score: {brier_score:.3f})')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_calibration_curve.png"))
        plt.close()
    
    def _plot_threshold_metric_curves(self, model_name, y_true, y_pred_proba):
        """Threshold-Metric 곡선 시각화"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        metric_values = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            metric_values['accuracy'].append(accuracy_score(y_true, y_pred_threshold))
            metric_values['precision'].append(precision_score(y_true, y_pred_threshold))
            metric_values['recall'].append(recall_score(y_true, y_pred_threshold))
            metric_values['f1'].append(f1_score(y_true, y_pred_threshold))
        
        plt.figure(figsize=(10, 6))
        for metric, values in metric_values.items():
            plt.plot(thresholds, values, label=metric)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{model_name} Threshold-Metric Curves')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_threshold_metric_curves.png"))
        plt.close()
    
    def _plot_probability_distribution(self, model_name, y_true, y_pred_proba, metrics):
        """예측확률 분포 시각화 (클래스별 분리)"""
        # 클래스별 확률 분리
        neg_probs = y_pred_proba[y_true == 0]  # 음성 샘플 확률
        pos_probs = y_pred_proba[y_true == 1]  # 양성 샘플 확률
        
        plt.figure(figsize=(10, 6))
        
        # 클래스별 히스토그램
        plt.hist(neg_probs, bins=50, alpha=0.5, label='Negative samples', color='red')
        plt.hist(pos_probs, bins=50, alpha=0.5, label='Positive samples', color='blue')
        
        # 임계값 표시
        plt.axvline(x=0.5, color='black', linestyle='--', label='Default threshold (0.5)')
        plt.axvline(x=metrics["optimal_threshold"], color='green', linestyle='--', 
                    label=f'Optimal threshold ({metrics["optimal_threshold"]:.2f})')
        
        # 그래프 설정
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title(f'{model_name} Prediction Probability Distribution by True Class')
        plt.legend(loc='upper right')
        
        # 클래스별 샘플 수 표시
        plt.text(0.02, 0.95, f'Negative samples: {len(neg_probs)}', 
                 transform=plt.gca().transAxes, color='red')
        plt.text(0.02, 0.90, f'Positive samples: {len(pos_probs)}', 
                 transform=plt.gca().transAxes, color='blue')
        
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_probability_distribution.png"))
        plt.close()
    
    def _save_metrics(self, model_name, metrics):
        """성능 지표 저장"""
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, f"{model_name}_detailed_metrics.csv"), 
                         index=False)
