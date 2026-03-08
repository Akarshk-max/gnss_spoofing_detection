import pandas as pd
import numpy as np
import os

LAMBDA_L1 = 0.190293672798

def engineer_features(df):
    num_cols = ['Carrier_Doppler_hz', 'Pseudorange_m', 'Carrier_phase', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'CN0', 'RX_time', 'TOW']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    n_before = len(df)
    valid_mask = (df['CN0'] > 0) & (df['Pseudorange_m'] != 0)
    df = df[valid_mask].copy()
    n_after = len(df)
    print(f"  Validity filter: dropped {n_before - n_after:,} dead-epoch rows ({(n_before-n_after)/n_before*100:.1f}%) → {n_after:,} valid rows remaining")

    print("Sorting by PRN and Time to ensure no temporal data leakage...")
    df = df.sort_values(['PRN', 'RX_time']).copy()
    
    print("Computing Correlator SQM features...")
    df['correlator_diff'] = df['EC'] - df['LC']
    df['correlator_ratio'] = df['EC'] / (df['LC'] + 1e-6)
    df['prompt_balance'] = df['PC'] / (df['EC'] + df['LC'] + 1e-6)
    df['PC_magnitude'] = (df['PIP']**2 + df['PQP']**2)**0.5

    print("Computing PIR (Pseudorange-Doppler Residual)...")
    df['pseudo_delta'] = df.groupby('PRN')['Pseudorange_m'].diff()
    df['time_delta'] = df.groupby('PRN')['RX_time'].diff()
    
    valid_dt = df['time_delta'] != 0
    df['pseudo_rate'] = 0.0
    df.loc[valid_dt, 'pseudo_rate'] = df.loc[valid_dt, 'pseudo_delta'] / df.loc[valid_dt, 'time_delta']
    
    df['doppler_velocity'] = LAMBDA_L1 * df['Carrier_Doppler_hz']
    df['residual_PD'] = df['pseudo_rate'] + df['doppler_velocity']

    print("Computing Temporal Dynamics...")
    df['doppler_delta'] = df.groupby('PRN')['Carrier_Doppler_hz'].diff()
    df['phase_jump'] = df.groupby('PRN')['Carrier_phase'].diff().abs()
    
    df['cn0_zscore'] = df.groupby('PRN')['CN0'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    print("Computing Clock Consistency...")
    df['clock_error'] = df['RX_time'] - df['TOW']

    print("Computing Per-PRN Statistics...")
    for col in ['CN0', 'Carrier_Doppler_hz', 'Pseudorange_m']:
        df[f'{col}_prn_mean'] = df.groupby('PRN')[col].transform('mean')
        df[f'{col}_prn_std']  = df.groupby('PRN')[col].transform('std')

    print("Handling NaNs...")
    df.fillna(0, inplace=True)
    
    return df

def main():
    base_dir = r"c:\Users\Amogh Shukla\Downloads\gnss_spoofing_plan+analysis\IITD_GNSS_Hackathon_Dataset"
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    
    out_dir = r"c:\Users\Amogh Shukla\Downloads\gnss_spoofing_plan+analysis\engineered_data"
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(train_path):
        print("\nProcessing Training Data...")
        df_train = pd.read_csv(train_path, low_memory=False)
        df_train_eng = engineer_features(df_train)
        out_train = os.path.join(out_dir, "train_engineered.csv")
        df_train_eng.to_csv(out_train, index=False)
        print(f"Saved: {out_train}")
    else:
        print(f"Train file not found: {train_path}")

    if os.path.exists(test_path):
        print("\nProcessing Test Data...")
        df_test = pd.read_csv(test_path, low_memory=False)
        df_test_eng = engineer_features(df_test)
        out_test = os.path.join(out_dir, "test_engineered.csv")
        df_test_eng.to_csv(out_test, index=False)
        print(f"Saved: {out_test}")
    else:
        print(f"Test file not found: {test_path}")
        
    print("\nFeature Engineering Pipeline Complete!")

if __name__ == "__main__":
    main()
