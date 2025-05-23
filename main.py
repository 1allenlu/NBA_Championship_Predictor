# NBA Championship Prediction ML Pipeline
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from typing import Dict, List, Tuple
import joblib
import logging
from src.model_pipeline import ModelConfig, NBAChampionshipPredictor
from src.data_loader import load_nba_data


#source myenv/bin/activate



def evaluate_model():
    df = load_nba_data()
    
    # Target and features
    y = df['champion']
    X = df.drop(columns=['champion'])

    config = ModelConfig()
    predictor = NBAChampionshipPredictor(config)

    print("NBA Championship Prediction ML Pipeline")
    print("=======================================")
    
    predictor.train_ensemble(X, y)
    print("Training complete.")

if __name__ == "__main__":
    evaluate_model()