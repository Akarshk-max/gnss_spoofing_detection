import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "/kaggle/input/datasets/akarshkumarshukla/final-data/train_engineered.csv"
MODEL_DIR = "/kaggle/working/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading engineered data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

DROP_COLS = ['time', 'RX_time', 'TOW', 'spoofed']
features = [c for c in df.columns if c not in DROP_COLS]

X = df[features]
y = df['spoofed']

le_maps = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_maps[col] = {str(k): int(v) for k, v in zip(le.classes_, range(len(le.classes_)))}

with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
    json.dump(features, f)

if le_maps:
    with open(os.path.join(MODEL_DIR, 'le_maps.json'), 'w') as f:
        json.dump(le_maps, f)

print(f"Saved {len(features)} feature names and {len(le_maps)} LE mappings.")
print(f"Dataset shape: {X.shape}")
print(f"Class balance: Authentic={sum(y==0)}, Spoofed={sum(y==1)}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

def objective(trial):
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 8.0),
        'early_stopping_rounds': 50,
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_prob = model.predict_proba(X_val)[:, 1]

    best_f1 = 0
    for thresh in np.arange(0.01, 0.61, 0.01):
        preds = (preds_prob >= thresh).astype(int)
        score = f1_score(y_val, preds, average='weighted')
        if score > best_f1:
            best_f1 = score

    return best_f1


if __name__ == "__main__":

    print("\nStarting Optuna Hyperparameter Optimization (Maximize Weighted F1)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)

    print("\n--- Optuna Best Trial ---")
    print(f"Best Weighted F1: {study.best_value:.4f}")
    print("Best Params:", study.best_params)

    print("\nTraining Final Best Model...")
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    best_params['objective'] = 'binary:logistic'
    best_params['n_estimators'] = 1500
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['early_stopping_rounds'] = 100

    final_model = xgb.XGBClassifier(**best_params)

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    val_probs = final_model.predict_proba(X_val)[:, 1]

    print(f"Val prob distribution — min: {val_probs.min():.4f}  max: {val_probs.max():.4f}  mean: {val_probs.mean():.4f}")

    best_thresh = 0.5
    best_f1 = 0

    print("\nThreshold sweep (0.01 → 0.60):")

    for thresh in np.arange(0.01, 0.61, 0.01):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("\n--- Validation Results ---")
    print(f"Optimal Decision Threshold: {best_thresh:.3f}")
    print(f"Final Weighted F1 Score:    {best_f1:.4f}")

    final_preds = (val_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_val, final_preds)

    print("\nConfusion Matrix:")
    print(cm)

    save_path = os.path.join(MODEL_DIR, "xgb_best_model.json")
    final_model.save_model(save_path)

    with open(os.path.join(MODEL_DIR, "best_threshold.txt"), "w") as f:
        f.write(str(best_thresh))

    print(f"\nModel saved to: {save_path}")
    print("Run hackathon_infer.py to generate submission.")


df_test = pd.read_csv("/kaggle/input/datasets/akarshkumarshukla/final-data/test_engineered.csv")

for col, mapping in le_maps.items():
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str).map(mapping).fillna(0).astype(int)

for c in features:
    if c not in df_test.columns:
        df_test[c] = 0

X_test_em = df_test[features].astype(np.float32)

test_probs = final_model.predict_proba(X_test_em)[:, 1]

best_thresh = 0.18

print(f"Probs: min={test_probs.min():.4f} max={test_probs.max():.4f}")

preds = (test_probs >= best_thresh).astype(int)

pd.DataFrame({
    'Id': range(len(preds)),
    'spoofed': preds
}).to_csv('/kaggle/working/submission.csv', index=False)

print(f"Done! Spoofed={sum(preds):,}")
