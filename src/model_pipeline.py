import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
from src.feature_engineering import FeatureEngineering


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    


class NBAChampionshipPredictor:
    """Main ML pipeline for NBA championship prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineering()
        self.models = {}
        self.ensemble_weights = {}
        self.is_fitted = False
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_base_models(self) -> Dict:
        """Define base models for ensemble"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=self.config.random_state,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_state
            ),
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=self.config.random_state,
                    class_weight='balanced',
                    max_iter=1000
                ))
            ])
        }
    
    def _time_series_split(self, X: pd.DataFrame, y: pd.Series) -> TimeSeriesSplit:
        """Custom time series split for NBA seasons"""
        # Use TimeSeriesSplit to respect temporal order
        return TimeSeriesSplit(n_splits=self.config.cv_folds)
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train ensemble of models with MLflow tracking"""
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_models", len(self._get_base_models()))
            mlflow.log_param("cv_folds", self.config.cv_folds)
            
            # Feature engineering
            X_engineered = self.feature_engineer.engineer_features(X.copy())
            
            # Get feature columns (exclude target and metadata)
            feature_cols = [col for col in X_engineered.columns 
                          if col not in ['team', 'season', 'champion']]
            X_features = X_engineered[feature_cols]
            
            # Time series cross-validation
            tscv = self._time_series_split(X_features, y)
            
            # Train each model
            self.models = self._get_base_models()
            model_scores = {}
            
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_features, y, 
                    cv=tscv, scoring='roc_auc'
                )
                
                model_scores[name] = cv_scores.mean()
                
                # Fit on full data
                model.fit(X_features, y)
                
                # Log model metrics
                mlflow.log_metric(f"{name}_cv_auc", cv_scores.mean())
                mlflow.log_metric(f"{name}_cv_std", cv_scores.std())
                
                self.logger.info(f"{name} AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Calculate ensemble weights based on performance
            total_score = sum(model_scores.values())
            self.ensemble_weights = {
                name: score / total_score 
                for name, score in model_scores.items()
            }
            
            # Log ensemble weights
            for name, weight in self.ensemble_weights.items():
                mlflow.log_metric(f"{name}_ensemble_weight", weight)
            
            # Save feature names
            self.feature_names = feature_cols
            mlflow.log_param("n_features", len(feature_cols))
            
            self.is_fitted = True
            
            # Log the ensemble model
            mlflow.sklearn.log_model(self, "nba_championship_ensemble")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction with probability outputs"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_features(X.copy())
        X_features = X_engineered[self.feature_names]
        
        # Get predictions from each model
        ensemble_probs = np.zeros((len(X_features), 2))
        
        for name, model in self.models.items():
            model_probs = model.predict_proba(X_features)
            weight = self.ensemble_weights[name]
            ensemble_probs += weight * model_probs
        
        return ensemble_probs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get weighted feature importance across models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({'feature': self.feature_names})
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[f'{name}_importance'] = model.feature_importances_
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'coef_'):
                # For logistic regression
                importance_df[f'{name}_importance'] = np.abs(model.named_steps['classifier'].coef_[0])
        
        # Calculate weighted average importance
        importance_cols = [col for col in importance_df.columns if 'importance' in col]
        weights = [self.ensemble_weights[col.split('_')[0]] for col in importance_cols]
        
        importance_df['ensemble_importance'] = np.average(
            importance_df[importance_cols].values, 
            axis=1, 
            weights=weights
        )
        
        return importance_df.sort_values('ensemble_importance', ascending=False)