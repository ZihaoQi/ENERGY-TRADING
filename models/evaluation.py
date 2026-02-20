import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ModelEvaluator:
    """
    Model evaluation utility
    Designed for ML trading pipelines
    """

    def __init__(self, model):
        self.model = model

    def _plot_feature_importance(self, ax, feature_names):
        # if not hasattr(self.model, "feature_importances_"):
        #     ax.text(0.5, 0.5, "Feature importance not available",
        #             ha="center", va="center")
        #     return

        importance = self.model.model.feature_importances_

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        ax.barh(df["feature"], df["importance"])
        ax.set_title("Feature Importance", fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")

    def _plot_confusion_matrix(self, ax, y_true, y_pred, target_names):
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax
        )

        ax.set_title("Confusion Matrix (Test)", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    def _plot_class_distribution(self, ax, y_train, y_test, target_names):
        train_dist = pd.Series(y_train).value_counts().sort_index()
        test_dist = pd.Series(y_test).value_counts().sort_index()

        x = np.arange(len(train_dist))
        width = 0.35

        ax.bar(x - width/2, train_dist.values, width, label="Train")
        ax.bar(x + width/2, test_dist.values, width, label="Test")

        ax.set_xticks(x)
        ax.set_xticklabels(target_names)
        ax.set_title("Class Distribution", fontweight="bold")
        ax.set_ylabel("Count")
        ax.legend()

    def _plot_prediction_confidence(self, ax, X_test, trading_threshold):
        if not hasattr(self.model, "predict_proba"):
            ax.text(0.5, 0.5, "predict_proba not available",
                    ha="center", va="center")
            return

        y_proba = self.model.predict_proba(X_test)
        max_proba = np.max(y_proba, axis=1)

        ax.hist(max_proba, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(
            x=trading_threshold,
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({trading_threshold})"
        )

        ax.set_title("Prediction Confidence", fontweight="bold")
        ax.set_xlabel("Max Probability")
        ax.set_ylabel("Frequency")
        ax.legend()

    def evaluate(
        self,
        X_test,
        y_test,
        y_train,
        y_pred_test,
        feature_names,
        target_names,
        trading_threshold=0.6,
        save_path=None
    ):
        """
        Create 2x2 evaluation dashboard
        """

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1 Feature Importance
        self._plot_feature_importance(axes[0, 0], feature_names)

        # 2 Confusion Matrix
        self._plot_confusion_matrix(
            axes[0, 1], y_test, y_pred_test, target_names
        )

        # 3 Class Distribution
        self._plot_class_distribution(
            axes[1, 0], y_train, y_test, target_names
        )

        # 4 Confidence Distribution
        self._plot_prediction_confidence(
            axes[1, 1], X_test, trading_threshold
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Evaluation saved to {save_path}")

        return fig

    def print_classification_report(self, y_test, y_pred_test, target_names):
        print("\nClassification Report:\n")
        print(classification_report(
            y_test,
            y_pred_test,
            target_names=target_names
        ))