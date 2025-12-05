import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform, loguniform
from config import settings

def load_data(filepath):
    """Loads the model-ready dataset."""
    df = pd.read_csv(filepath)
    return df

def get_feature_target_split(df, target_col='deficit'):
    """Separates features (X) and target (y)."""
    # Drop non-feature columns
    drop_cols = ['N_HC', 'ultima ventana', target_col]
    # Also drop any other metadata if present
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(int)
    
    return X, y

def train_models(X, y):
    """
    Trains baseline and optimized models using RandomizedSearchCV.
    Returns a dictionary of all trained models and the test sets.
    """
    # 1. Stratified Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Define Base Models
    base_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    }

    # 3. Define Hyperparameter Grids
    param_grids = {
        'LogisticRegression': {
            'classifier__C': loguniform(1e-4, 100),
            'classifier__penalty': ['l2'] # l1 requires specific solvers
        },
        'RandomForest': {
            'classifier__n_estimators': randint(50, 300),
            'classifier__max_depth': randint(3, 20),
            'classifier__min_samples_split': randint(2, 20),
            'classifier__min_samples_leaf': randint(1, 10)
        },
        'XGBoost': {
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__n_estimators': randint(50, 300),
            'classifier__max_depth': randint(3, 10),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4)
        },
        'LightGBM': {
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__n_estimators': randint(50, 300),
            'classifier__max_depth': randint(3, 10),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
            'classifier__num_leaves': randint(20, 100)
        }
    }

    trained_models = {}

    # 4. Train Loop
    for name, model in base_models.items():
        print(f"\n--- Processing {name} ---")
        
        # Base Pipeline
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # A. Train Baseline
        print(f"Training Baseline {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[f"{name}_Baseline"] = pipeline
        print(f"✓ {name}_Baseline trained.")
        
        # B. Train Optimized
        print(f"Optimizing {name}...")
        if name in param_grids:
            search = RandomizedSearchCV(
                pipeline, 
                param_distributions=param_grids[name], 
                n_iter=20, # 20 iterations for better coverage
                cv=3,      # 3-fold CV to save time
                scoring='roc_auc', 
                n_jobs=-1, 
                random_state=42,
                verbose=1
            )
            search.fit(X_train, y_train)
            trained_models[f"{name}_Optimized"] = search.best_estimator_
            print(f"✓ {name}_Optimized trained. Best params: {search.best_params_}")

    return trained_models, X_test, y_test
