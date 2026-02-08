"""
Model Evaluation Utilities
Comprehensive metrics and visualization for model assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)

from sklearn.calibration import calibration_curve as sklearn_calibration_curve

from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px


class ModelEvaluator:
    """Comprehensive model evaluation toolkit"""
    
    def __init__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 y_pred_binary: Optional[np.ndarray] = None, threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            y_true: True labels (0/1)
            y_pred_proba: Predicted probabilities
            y_pred_binary: Predicted binary labels (optional, will be computed)
            threshold: Classification threshold
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        
        if y_pred_binary is None:
            self.y_pred_binary = (y_pred_proba >= threshold).astype(int)
        else:
            self.y_pred_binary = y_pred_binary
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        
        # ROC curve metrics
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_true, self.y_pred_proba)
        
        # Classification metrics
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred_binary).ravel()
        
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'brier_score': brier_score_loss(self.y_true, self.y_pred_proba),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        # Positive/Negative Predictive Value
        metrics['ppv'] = metrics['precision']
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def print_report(self):
        """Print comprehensive evaluation report"""
        metrics = self.compute_metrics()
        
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        
        print("\n📊 DISCRIMINATION METRICS")
        print(f"  ROC-AUC:           {metrics['roc_auc']:.3f}")
        print(f"  PR-AUC:            {metrics['pr_auc']:.3f}")
        
        print("\n🎯 CLASSIFICATION METRICS (threshold={:.2f})".format(self.threshold))
        print(f"  Accuracy:          {metrics['accuracy']:.3f}")
        print(f"  Precision (PPV):   {metrics['precision']:.3f}")
        print(f"  Recall (Sens):     {metrics['recall']:.3f}")
        print(f"  Specificity:       {metrics['specificity']:.3f}")
        print(f"  F1 Score:          {metrics['f1_score']:.3f}")
        print(f"  NPV:               {metrics['npv']:.3f}")
        
        print("\n📈 CALIBRATION")
        print(f"  Brier Score:       {metrics['brier_score']:.3f}")
        
        print("\n🔢 CONFUSION MATRIX")
        print(f"  True Positives:    {metrics['true_positives']}")
        print(f"  True Negatives:    {metrics['true_negatives']}")
        print(f"  False Positives:   {metrics['false_positives']}")
        print(f"  False Negatives:   {metrics['false_negatives']}")
        
        print("\n💡 INTERPRETATION")
        if metrics['roc_auc'] >= 0.9:
            print("  ✓ Excellent discrimination")
        elif metrics['roc_auc'] >= 0.8:
            print("  ✓ Good discrimination")
        elif metrics['roc_auc'] >= 0.7:
            print("  ⚠ Fair discrimination")
        else:
            print("  ✗ Poor discrimination - model needs improvement")
        
        if metrics['precision'] >= 0.8 and metrics['recall'] >= 0.7:
            print("  ✓ Good precision-recall balance")
        elif metrics['precision'] < 0.7:
            print("  ⚠ Low precision - many false positives (alert fatigue risk)")
        elif metrics['recall'] < 0.6:
            print("  ⚠ Low recall - missing at-risk patients")
        
        print("=" * 60)
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal reference
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_true, self.y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR (AUC = {pr_auc:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline
        baseline = self.y_true.mean()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Baseline ({baseline:.2f})',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_calibration_curve(self, n_bins: int = 10, save_path: Optional[str] = None):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = sklearn_calibration_curve(
            self.y_true, self.y_pred_proba, n_bins=n_bins
        )
        
        fig = go.Figure()
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Model',
            line=dict(color='blue', width=2)
        ))
        
        # Perfect calibration
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Calibration Curve',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=600,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred_binary)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Adherent', 'Non-Adherent'],
            y=['Adherent', 'Non-Adherent'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            width=500,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_threshold_analysis(self, save_path: Optional[str] = None):
        """Plot metrics vs threshold"""
        thresholds = np.linspace(0, 1, 101)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred = (self.y_pred_proba >= thresh).astype(int)
            
            if y_pred.sum() == 0:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                continue
            
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, name='Precision', mode='lines'))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, name='Recall', mode='lines'))
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, name='F1 Score', mode='lines'))
        
        # Mark current threshold
        fig.add_vline(x=self.threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Current ({self.threshold})")
        
        fig.update_layout(
            title='Metrics vs Classification Threshold',
            xaxis_title='Threshold',
            yaxis_title='Score',
            width=800,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_report_html(self, save_path: str = "evaluation_report.html"):
        """Generate comprehensive HTML report"""
        metrics = self.compute_metrics()
        
        # Create all plots
        roc_fig = self.plot_roc_curve()
        pr_fig = self.plot_precision_recall_curve()
        cal_fig = self.plot_calibration_curve()
        cm_fig = self.plot_confusion_matrix()
        thresh_fig = self.plot_threshold_analysis()
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #FF4B4B; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>🏥 Model Evaluation Report</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <div class="section">
                <h2>📊 Key Metrics</h2>
                <div class="metric"><b>ROC-AUC:</b> {metrics['roc_auc']:.3f}</div>
                <div class="metric"><b>PR-AUC:</b> {metrics['pr_auc']:.3f}</div>
                <div class="metric"><b>Precision:</b> {metrics['precision']:.3f}</div>
                <div class="metric"><b>Recall:</b> {metrics['recall']:.3f}</div>
                <div class="metric"><b>F1 Score:</b> {metrics['f1_score']:.3f}</div>
                <div class="metric"><b>Brier Score:</b> {metrics['brier_score']:.3f}</div>
            </div>
            
            <div class="section">
                <h2>📈 ROC Curve</h2>
                {roc_fig.to_html(full_html=False)}
            </div>
            
            <div class="section">
                <h2>🎯 Precision-Recall Curve</h2>
                {pr_fig.to_html(full_html=False)}
            </div>
            
            <div class="section">
                <h2>📉 Calibration Curve</h2>
                {cal_fig.to_html(full_html=False)}
            </div>
            
            <div class="section">
                <h2>🔢 Confusion Matrix</h2>
                {cm_fig.to_html(full_html=False)}
            </div>
            
            <div class="section">
                <h2>⚖️ Threshold Analysis</h2>
                {thresh_fig.to_html(full_html=False)}
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"✓ Report saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y_true = make_classification(n_samples=500, n_features=10, random_state=42)
    y_pred_proba = np.random.rand(500)
    
    # Evaluate
    evaluator = ModelEvaluator(y_true, y_pred_proba, threshold=0.5)
    evaluator.print_report()
    evaluator.generate_report_html("test_evaluation.html")
