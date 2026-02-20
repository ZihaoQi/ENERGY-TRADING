from xgboost import XGBClassifier
from base_classes import BaseModel
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score


class XGBoostModel(BaseModel):

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = XGBClassifier(
            n_estimators=self.config.get("n_estimators", 200),
            max_depth=self.config.get("max_depth", 5),
            learning_rate=self.config.get("learning_rate", 0.1),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            random_state=self.config.get("random_state", 42),
            eval_metric="mlogloss"
        )

    def optimize(self, X, y, n_trials=100):
        """Optimize model hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = XGBClassifier(**params, random_state=42, eval_metric="mlogloss")
            score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
            return score

        study = optuna.create_study(direction='maximize', pruner=MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.model.set_params(**best_params)
        return study.best_value

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        return self.model.model.feature_importances_
