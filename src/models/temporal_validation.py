"""
Temporal validation module for TANI child development prediction.

This module implements temporal (out-of-time) validation to assess model
performance on future, unseen time periods - critical for longitudinal data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import settings


def create_temporal_split(df_model, df_analytical, temporal_percentile=0.8):
    """
    Creates temporal train/test split based on patient's last observation date.

    Parameters:
    -----------
    df_model : pd.DataFrame
        Model-ready dataset (one row per patient observation window)
    df_analytical : pd.DataFrame
        Analytical dataset with temporal information (Fecha column)
    temporal_percentile : float
        Percentile cutoff for temporal split (default 0.8 = 80th percentile)

    Returns:
    --------
    dict with keys:
        - X_train_temporal: Features for temporal training set
        - X_test_temporal: Features for temporal test set
        - y_train_temporal: Target for temporal training set
        - y_test_temporal: Target for temporal test set
        - cutoff_date: Date used as temporal cutoff
        - train_patients: List of N_HC in training set
        - test_patients: List of N_HC in test set
    """

    # 1. Get max date per patient from analytical dataset
    df_analytical['Fecha'] = pd.to_datetime(df_analytical['Fecha'])

    patient_max_dates = df_analytical.groupby('N_HC')['Fecha'].max().reset_index()
    patient_max_dates.columns = ['N_HC', 'max_fecha']

    # 2. Determine temporal cutoff
    cutoff_date = patient_max_dates['max_fecha'].quantile(temporal_percentile)

    print("=" * 80)
    print("TEMPORAL SPLIT CONFIGURATION")
    print("=" * 80)
    print(f"Temporal percentile: {temporal_percentile*100:.0f}%")
    print(f"Cutoff date: {cutoff_date.date()}")

    # 3. Assign patients to temporal train/test based on their max date
    patient_max_dates['temporal_set'] = patient_max_dates['max_fecha'].apply(
        lambda x: 'train' if x <= cutoff_date else 'test'
    )

    train_patients = patient_max_dates[patient_max_dates['temporal_set'] == 'train']['N_HC'].values
    test_patients = patient_max_dates[patient_max_dates['temporal_set'] == 'test']['N_HC'].values

    print(f"\nPatients in temporal training: {len(train_patients):,} ({len(train_patients)/len(patient_max_dates)*100:.1f}%)")
    print(f"Patients in temporal test: {len(test_patients):,} ({len(test_patients)/len(patient_max_dates)*100:.1f}%)")

    # 4. Filter model-ready dataset by patient assignment
    df_train_temporal = df_model[df_model['N_HC'].isin(train_patients)].copy()
    df_test_temporal = df_model[df_model['N_HC'].isin(test_patients)].copy()

    print(f"\nObservations in temporal training: {len(df_train_temporal):,}")
    print(f"Observations in temporal test: {len(df_test_temporal):,}")

    # 5. Separate features and target
    drop_cols = ['N_HC', 'ultima ventana', 'deficit']

    X_train_temporal = df_train_temporal.drop(columns=[c for c in drop_cols if c in df_train_temporal.columns])
    y_train_temporal = df_train_temporal['deficit'].astype(int)

    X_test_temporal = df_test_temporal.drop(columns=[c for c in drop_cols if c in df_test_temporal.columns])
    y_test_temporal = df_test_temporal['deficit'].astype(int)

    # 6. Check class balance
    print("\nClass distribution in temporal training:")
    print(f"  No deficit: {(y_train_temporal == 0).sum():,} ({(y_train_temporal == 0).mean()*100:.2f}%)")
    print(f"  Deficit:    {(y_train_temporal == 1).sum():,} ({(y_train_temporal == 1).mean()*100:.2f}%)")

    print("\nClass distribution in temporal test:")
    print(f"  No deficit: {(y_test_temporal == 0).sum():,} ({(y_test_temporal == 0).mean()*100:.2f}%)")
    print(f"  Deficit:    {(y_test_temporal == 1).sum():,} ({(y_test_temporal == 1).mean()*100:.2f}%)")

    return {
        'X_train_temporal': X_train_temporal,
        'X_test_temporal': X_test_temporal,
        'y_train_temporal': y_train_temporal,
        'y_test_temporal': y_test_temporal,
        'cutoff_date': cutoff_date,
        'train_patients': train_patients,
        'test_patients': test_patients
    }


def evaluate_temporal_performance(models, temporal_split_data):
    """
    Evaluates trained models on temporal test set.

    Parameters:
    -----------
    models : dict
        Dictionary of trained model pipelines (from train_model.py)
    temporal_split_data : dict
        Output from create_temporal_split()

    Returns:
    --------
    pd.DataFrame with temporal evaluation results
    """
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    X_test_temporal = temporal_split_data['X_test_temporal']
    y_test_temporal = temporal_split_data['y_test_temporal']

    results = []

    for model_name, model_pipeline in models.items():
        # Predict
        y_pred_proba = model_pipeline.predict_proba(X_test_temporal)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        auc = roc_auc_score(y_test_temporal, y_pred_proba)
        precision = precision_score(y_test_temporal, y_pred, zero_division=0)
        recall = recall_score(y_test_temporal, y_pred, zero_division=0)
        f1 = f1_score(y_test_temporal, y_pred, zero_division=0)

        results.append({
            'Model': model_name,
            'AUC_Temporal': auc,
            'Precision_Temporal': precision,
            'Recall_Temporal': recall,
            'F1_Temporal': f1
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC_Temporal', ascending=False)

    return results_df


def compare_random_vs_temporal(random_results, temporal_results):
    """
    Compares performance between random stratified split and temporal split.

    Parameters:
    -----------
    random_results : pd.DataFrame
        Results from standard random test set (from model_comparison.csv)
    temporal_results : pd.DataFrame
        Results from temporal test set (from evaluate_temporal_performance)

    Returns:
    --------
    pd.DataFrame with comparison
    """

    # Merge results
    comparison = random_results.merge(
        temporal_results[['Model', 'AUC_Temporal', 'Precision_Temporal', 'Recall_Temporal', 'F1_Temporal']],
        on='Model',
        how='inner'
    )

    # Calculate degradation
    comparison['AUC_Degradation'] = comparison['AUC'] - comparison['AUC_Temporal']
    comparison['Precision_Degradation'] = comparison['Precision'] - comparison['Precision_Temporal']
    comparison['Recall_Degradation'] = comparison['Recall'] - comparison['Recall_Temporal']
    comparison['F1_Degradation'] = comparison['F1'] - comparison['F1_Temporal']

    # Add degradation percentage
    comparison['AUC_Degradation_Pct'] = (comparison['AUC_Degradation'] / comparison['AUC'] * 100).round(2)

    return comparison
