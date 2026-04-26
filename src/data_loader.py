import pandas as pd
import numpy as np
import os


def load_fremtpl2_data():
    """
    Carrega freMTPL2freq e freMTPL2sev.
    Ordem de tentativas:
      1. Cache local (data/raw/)
      2. Hugging Face mirror (mais estável em 2025/2026)
      3. OpenML direto via CSV endpoint (sem sklearn)
      4. OpenML via sklearn com retries
      5. Dados sintéticos realistas como fallback final
    """

    freq_path = '../data/raw/freMTPL2freq.csv'
    sev_path  = '../data/raw/freMTPL2sev.csv'

    # 1. Cache local
    if os.path.exists(freq_path) and os.path.exists(sev_path):
        print("Loading from local cache...")
        df_freq = pd.read_csv(freq_path)
        df_sev  = pd.read_csv(sev_path)
        print(f"Frequency data: {df_freq.shape}")
        print(f"Severity data:  {df_sev.shape}")
        return df_freq, df_sev

    os.makedirs('../data/raw', exist_ok=True)

    # 2. Hugging Face mirror
    try:
        print("Trying Hugging Face mirror...")
        base = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main"
        df_freq = pd.read_csv(f"{base}/freMTPL2freq.csv")
        df_sev  = pd.read_csv(f"{base}/freMTPL2sev.csv")
        df_freq["IDpol"] = df_freq["IDpol"].astype("int64")
        df_freq.to_csv(freq_path, index=False)
        df_sev.to_csv(sev_path,  index=False)
        print(f"Frequency data: {df_freq.shape}")
        print(f"Severity data:  {df_sev.shape}")
        return df_freq, df_sev
    except Exception as e:
        print(f"Hugging Face failed: {e}")

    # 3. OpenML CSV endpoint direto (sem sklearn)
    try:
        print("Trying OpenML CSV endpoint...")
        df_freq = pd.read_csv(
            "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff",
            quotechar="'"
        )
        df_freq.rename(lambda c: c.replace('"', ''), axis="columns", inplace=True)

        df_sev = pd.read_csv(
            "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff",
            index_col=0
        )
        df_freq.to_csv(freq_path, index=False)
        df_sev.to_csv(sev_path,  index=False)
        print(f"Frequency data: {df_freq.shape}")
        print(f"Severity data:  {df_sev.shape}")
        return df_freq, df_sev
    except Exception as e:
        print(f"OpenML CSV endpoint failed: {e}")

    # 4. OpenML via sklearn com retries
    try:
        print("Trying OpenML via sklearn (with retries)...")
        from sklearn.datasets import fetch_openml
        freq_data = fetch_openml(data_id=41214, as_frame=True, n_retries=10, delay=2.0)
        sev_data  = fetch_openml(data_id=41215, as_frame=True, n_retries=10, delay=2.0)
        df_freq = freq_data.frame
        df_sev  = sev_data.frame
        df_freq.to_csv(freq_path, index=False)
        df_sev.to_csv(sev_path,  index=False)
        print(f"Frequency data: {df_freq.shape}")
        print(f"Severity data:  {df_sev.shape}")
        return df_freq, df_sev
    except Exception as e:
        print(f"OpenML sklearn failed: {e}")

    # 5. Fallback sintético
    print("\nAll remote sources failed. Generating realistic synthetic data...")
    return _generate_synthetic_data()


def _generate_synthetic_data(n_policies=100000, random_state=42):
    """
    Gera dados sintéticos com propriedades estatísticas similares ao freMTPL2.
    - freMTPL2freq: 678k apólices, claim frequency ~3.9%
    - freMTPL2sev:  26k sinistros, severidade log-normal com cauda pesada
    """
    np.random.seed(random_state)
    n = n_policies

    df_freq = pd.DataFrame({
        'IDpol':       np.arange(1, n + 1),
        'ClaimNb':     np.random.poisson(0.039, n),
        'Exposure':    np.random.uniform(0.002, 1.0, n).round(4),
        'Area':        np.random.choice(list('ABCDEF'), n,
                                        p=[0.05, 0.10, 0.20, 0.25, 0.25, 0.15]),
        'VehPower':    np.random.randint(4, 15, n),
        'VehAge':      np.random.randint(0, 30, n),
        'DrivAge':     np.random.randint(18, 90, n),
        'BonusMalus':  np.random.randint(50, 230, n),
        'VehBrand':    np.random.choice([f'B{i}' for i in range(1, 12)], n),
        'VehGas':      np.random.choice(['Regular', 'Diesel'], n, p=[0.45, 0.55]),
        'Density':     np.clip(np.random.lognormal(5, 2, n).astype(int), 1, 30000),
        'Region':      np.random.choice([f'R{i:02d}' for i in range(1, 23)], n),
    })
    df_freq['ClaimNb'] = df_freq['ClaimNb'].clip(upper=4)
    df_freq['Exposure'] = df_freq['Exposure'].clip(upper=1.0)

    # Severidade: apenas apólices com sinistro
    mask       = df_freq['ClaimNb'] > 0
    claim_pids = np.repeat(
        df_freq.loc[mask, 'IDpol'].values,
        df_freq.loc[mask, 'ClaimNb'].values
    )
    severity = np.clip(
        np.random.lognormal(mean=7.0, sigma=1.4, size=len(claim_pids)),
        1, 1_000_000
    ).round(2)

    df_sev = pd.DataFrame({
        'IDpol':       claim_pids,
        'ClaimAmount': severity,
    })

    print(f"Synthetic frequency data: {df_freq.shape}")
    print(f"Synthetic severity data:  {df_sev.shape}")
    print(f"Mean claim amount:        {df_sev['ClaimAmount'].mean():,.2f}")
    print(f"P99 claim amount:         {np.percentile(df_sev['ClaimAmount'], 99):,.2f}")
    return df_freq, df_sev


def create_loss_portfolio(df_sev, n_simulations=50000, random_state=42):
    np.random.seed(random_state)
    col    = 'ClaimAmount' if 'ClaimAmount' in df_sev.columns else df_sev.columns[-1]
    sample = df_sev[col].dropna().values
    sample = sample[sample > 0]
    losses = np.random.choice(sample, size=n_simulations, replace=True)
    print(f"Generated {n_simulations:,} simulated losses")
    print(f"Mean:  {losses.mean():>12,.2f}")
    print(f"P99:   {np.percentile(losses, 99):>12,.2f}")
    print(f"Max:   {losses.max():>12,.2f}")
    return losses


def add_catastrophe_events(losses, n_cat_events=10, cat_multiplier=100, random_state=42):
    np.random.seed(random_state)
    losses_cat = losses.copy()
    idx = np.random.choice(len(losses), size=n_cat_events, replace=False)
    losses_cat[idx] *= cat_multiplier
    print(f"Added {n_cat_events} catastrophe events (x{cat_multiplier})")
    print(f"New max loss: {losses_cat.max():>12,.2f}")
    return losses_cat


def save_portfolio(losses, filepath='../data/processed/claims_cleaned.csv'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame({'loss': losses}).to_csv(filepath, index=False)
    print(f"Portfolio saved: {filepath}  ({len(losses):,} rows)")


def load_portfolio(filepath='../data/processed/claims_cleaned.csv'):
    return pd.read_csv(filepath)['loss'].values