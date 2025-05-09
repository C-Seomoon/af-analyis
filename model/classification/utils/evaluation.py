from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score
import numpy as np
import pandas as pd
import json
import os

def calculate_classification_metrics(y_true, y_pred, y_prob):
    """
    이진 분류 모델의 주요 성능 지표를 계산합니다.

    Args:
        y_true (array-like): 실제 레이블.
        y_pred (array-like): 모델이 예측한 레이블 (0 또는 1).
        y_prob (array-like): 클래스 1에 대한 예측 확률.

    Returns:
        dict: 계산된 성능 지표들을 담은 딕셔너리.
              {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
               'pr_auc', 'balanced_accuracy', 'mcc'} 포함.
               오류 발생 시 해당 메트릭 값은 None으로 설정됩니다.
    """
    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Could not calculate accuracy: {e}")
        metrics['accuracy'] = None
        
    try:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"Warning: Could not calculate precision: {e}")
        metrics['precision'] = None
        
    try:
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"Warning: Could not calculate recall: {e}")
        metrics['recall'] = None
        
    try:
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"Warning: Could not calculate f1-score: {e}")
        metrics['f1'] = None
        
    try:
        # ROC AUC는 두 클래스가 모두 존재해야 계산 가능
        if len(np.unique(y_true)) > 1:
             metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            print("Warning: ROC AUC requires at least two classes in y_true. Setting to None.")
            metrics['roc_auc'] = None
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC: {e}")
        metrics['roc_auc'] = None

    try:
        # PR AUC 계산 (Average Precision) - 클래스 개수 확인 추가
        if len(np.unique(y_true)) > 1:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        else:
            print("Warning: PR AUC (Average Precision) requires at least two classes in y_true. Setting to None.")
            metrics['pr_auc'] = None
    except Exception as e:
        print(f"Warning: Could not calculate PR AUC (Average Precision): {e}")
        metrics['pr_auc'] = None

    try:
        # Balanced Accuracy 계산 (불균형 데이터셋에 유용)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Could not calculate Balanced Accuracy: {e}")
        metrics['balanced_accuracy'] = None
        
    try:
        # Matthews Correlation Coefficient (MCC) 계산 (불균형 데이터셋에 유용)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Could not calculate Matthews Correlation Coefficient (MCC): {e}")
        metrics['mcc'] = None

    return metrics

def save_results(results, output_dir, filename="results.json"):
    """
    딕셔너리 형태의 결과를 JSON 파일로 저장합니다.

    Args:
        results (dict): 저장할 결과 데이터.
        output_dir (str): 결과를 저장할 디렉토리 경로.
        filename (str, optional): 저장될 JSON 파일 이름. Defaults to "results.json".
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to standard Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # Handle arrays if needed
            return obj.tolist() # Convert arrays to lists
        elif isinstance(obj, pd.Timestamp): # Handle pandas Timestamps if present
             return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist() # Series는 리스트로 변환
        elif isinstance(obj, pd.DataFrame):
            # DataFrame은 {column: [values...]} 형태의 딕셔너리로 변환 (예시)
            # 또는 다른 적절한 형태로 변환 (예: to_dict('records'))
            return obj.to_dict(orient='list')
        return obj

    try:
        with open(filepath, 'w') as f:
            # Use json.dump with default=convert_types for broader compatibility
            json.dump(results, f, indent=4, default=convert_types) 
        print(f"Results saved to {filepath}")
    except TypeError as e:
        print(f"Error saving results to JSON: {e}. Serialization failed for some data types.")
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")


def save_predictions(y_true, y_pred, y_prob, output_dir, filename="predictions.csv", index=None):
    """
    실제값, 예측값, 예측 확률을 CSV 파일로 저장합니다.

    Args:
        y_true (array-like): 실제 레이블.
        y_pred (array-like): 모델이 예측한 레이블.
        y_prob (array-like): 클래스 1에 대한 예측 확률.
        output_dir (str): 파일을 저장할 디렉토리 경로.
        filename (str, optional): 저장될 CSV 파일 이름. Defaults to "predictions.csv".
        index (array-like, optional): CSV 파일의 인덱스로 사용할 값 (예: 원본 데이터 인덱스). Defaults to None.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Ensure consistent lengths
    if not (len(y_true) == len(y_pred) == len(y_prob)):
        print("Warning: Lengths of y_true, y_pred, and y_prob do not match. Cannot save predictions.")
        return
        
    # 인덱스 유효성 검사 강화
    use_index = False
    if index is not None:
        # Check if index is array-like and has the correct length
        if hasattr(index, '__len__') and len(index) == len(y_true):
            use_index = True
        else:
            print(f"Warning: Provided index is not valid (length mismatch or not array-like). Predictions will be saved without index.")
            index = None # Invalidate index if it's not usable

    pred_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }, index=index) # Use validated index

    try: # 파일 저장 시 예외 처리 추가
        pred_df.to_csv(filepath, index=use_index) # Save index only if valid and provided
        print(f"Predictions saved to {filepath}")
    except Exception as e:
         print(f"Error saving predictions to CSV: {e}")
