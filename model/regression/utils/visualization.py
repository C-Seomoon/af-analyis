import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def plot_actual_vs_predicted(y_true, y_pred, output_dir, fold_num):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Fold {fold_num} — Actual vs. Predicted")
    plt.tight_layout()
    path = os.path.join(output_dir, f"actual_vs_predicted_fold_{fold_num}.png")
    plt.savefig(path)
    plt.close()

def plot_residuals_vs_predicted(y_true, y_pred, output_dir, fold_num):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Fold {fold_num} — Residuals vs. Predicted")
    plt.tight_layout()
    path = os.path.join(output_dir, f"residuals_vs_predicted_fold_{fold_num}.png")
    plt.savefig(path)
    plt.close()

def plot_residual_distribution(y_true, y_pred, output_dir, fold_num):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True, stat="density", alpha=0.6)
    plt.xlabel("Residuals")
    plt.title(f"Fold {fold_num} — Residual Distribution")
    plt.tight_layout()
    path = os.path.join(output_dir, f"residuals_distribution_fold_{fold_num}.png")
    plt.savefig(path)
    plt.close()

def plot_qq(residuals, output_dir, fold_num):
    plt.figure(figsize=(6,6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Fold {fold_num} — Q–Q Plot of Residuals")
    plt.tight_layout()
    path = os.path.join(output_dir, f"residuals_qq_fold_{fold_num}.png")
    plt.savefig(path)
    plt.close()