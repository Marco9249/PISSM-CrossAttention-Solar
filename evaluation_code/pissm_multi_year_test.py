"""
================================================================================
Scientific Evaluation Module: PI-Cross-Attention Model (2020-2024)
================================================================================
Author : Prepared by: Eng. Mohammed Izzaldeen Babeker Abdullah
Purpose: Evaluating the physics-informed model on unseen 5-year data boundary.
         Generates professional non-overlapping English plots for publication.
================================================================================
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join("..", filename)):
        return os.path.join("..", filename)
    else:
        raise FileNotFoundError(f"Cannot find {filename} in current or parent dir.")

TRAIN_FILE = get_data_path("Hourly_2010_2015.csv")
TEST_FILE  = get_data_path("Hourly_2020_2025.csv")
WEIGHTS_FILE = get_data_path("pissm_saved_weights.pth")

PLOT_DIR = "PISSM_Publishable_Plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CORE ARCHITECTURE DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class HankelEmbeddingLayer(nn.Module):
    def __init__(self, sub_window: int = 5):
        super().__init__()
        self.sub_window = sub_window
    def forward(self, x):
        batch, seq_len, features = x.shape
        x_perm = x.permute(0, 2, 1)
        x_unf = x_perm.unfold(2, self.sub_window, 1)
        return x_unf.permute(0, 2, 1, 3).reshape(batch, x_unf.size(2), features * self.sub_window)

class CNN1DLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.activation(self.conv(x)))
        return x.permute(0, 2, 1)

class TemporalSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        attn_out, _ = self.self_attn(query=x, key=x, value=x)
        return attn_out

class PhysicalCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.phys_proj = nn.Linear(2, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    def forward(self, temporal_features, sza_seq, kt_seq):
        phys_vars = torch.cat([sza_seq, kt_seq], dim=-1)
        phys_proj = self.phys_proj(phys_vars)
        attn_out, _ = self.cross_attn(query=temporal_features, key=phys_proj, value=phys_proj)
        return attn_out

class PhysicsInformedCrossAttention(nn.Module):
    def __init__(self, num_features: int, sub_window: int = 5, conv_filters: int = 64, fc_hidden: int = 128, sza_idx: int = 4, kt_idx: int = 3):
        super().__init__()
        self.sza_idx, self.kt_idx = sza_idx, kt_idx
        hankel_dim = num_features * sub_window
        self.hankel = HankelEmbeddingLayer(sub_window=sub_window)
        self.layer1_cnn = CNN1DLayer(in_channels=hankel_dim, out_channels=conv_filters, dropout=0.2)
        self.layer2_self_attn = TemporalSelfAttentionLayer(embed_dim=conv_filters, num_heads=4)
        self.layer3_cross_attn = PhysicalCrossAttentionLayer(embed_dim=conv_filters, num_heads=4)
        self.layer4_fc = nn.Sequential(
            nn.Linear(conv_filters, fc_hidden), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(fc_hidden, 1)
        )
        self.layer5_relu = nn.ReLU()
        
    def forward(self, x):
        sza_full = x[:, :, self.sza_idx].unsqueeze(-1)
        kt_full  = x[:, :, self.kt_idx].unsqueeze(-1)
        h = self.hankel(x)
        num_windows = h.size(1)
        sza_seq = sza_full[:, -num_windows:, :]
        kt_seq  = kt_full[:, -num_windows:, :]
        
        h_cnn = self.layer1_cnn(h)
        h_self = self.layer2_self_attn(h_cnn)
        h_cross = self.layer3_cross_attn(h_self, sza_seq, kt_seq)
        pred = self.layer4_fc(h_cross[:, -1, :])
        return self.layer5_relu(pred)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA PROCESSING & NORMALIZATION RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────
TARGET_COL = "ALLSKY_SFC_SW_DWN"
PHYSICAL_COLS = ["CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "ALLSKY_KT", "SZA", "T2M", "RH2M", "WS10M", "PS"]
TIME_COLS = ["MO", "DY", "HR"]
KEEP_COLS = [TARGET_COL] + PHYSICAL_COLS + TIME_COLS
RADIATION_COLS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF"]
LOG_COLS = ["ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M"]

def process_data(filepath):
    with open(filepath, "r") as f:
        for idx, line in enumerate(f):
            if "-END HEADER-" in line:
                skip_rows = idx + 1
                break
    df = pd.read_csv(filepath, skiprows=skip_rows)
    df.replace([-999.0, -999], np.nan, inplace=True)
    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    has_year = "YEAR" in df.columns
    cols_to_keep = KEEP_COLS + (["YEAR"] if has_year else [])
    df = df[cols_to_keep].copy()
    
    night_mask = df["CLRSKY_SFC_SW_DWN"] <= 0.0
    for col in RADIATION_COLS:
        df.loc[night_mask, col] = 0.0
        
    t2m_shift = 0.0
    if df["T2M"].min() < 0:
        t2m_shift = abs(df["T2M"].min()) + 1.0
        df["T2M"] = df["T2M"] + t2m_shift
        
    for col in LOG_COLS:
        df[col] = np.log1p(df[col].clip(lower=0))
        
    df["MO_sin"] = np.sin(2 * np.pi * df["MO"] / 12.0)
    df["MO_cos"] = np.cos(2 * np.pi * df["MO"] / 12.0)
    df["DY_sin"] = np.sin(2 * np.pi * df["DY"] / 31.0)
    df["DY_cos"] = np.cos(2 * np.pi * df["DY"] / 31.0)
    df["HR_sin"] = np.sin(2 * np.pi * df["HR"] / 24.0)
    df["HR_cos"] = np.cos(2 * np.pi * df["HR"] / 24.0)
    df.drop(columns=TIME_COLS, inplace=True)
    return df

print(f"\n[INFO] Extracting normalization bounds from base training dataset...")
df_base = process_data(TRAIN_FILE)
n_train = int(len(df_base) * 0.80)
df_train = df_base.iloc[:n_train].copy()

CYCLICAL_COLS = ["MO_sin", "MO_cos", "DY_sin", "DY_cos", "HR_sin", "HR_cos"]
FEATURE_COLS = PHYSICAL_COLS + CYCLICAL_COLS

scaler_features = StandardScaler()
scaler_features.fit(df_train[FEATURE_COLS].values)

scaler_target = MinMaxScaler(feature_range=(0, 1))
scaler_target.fit(df_train[[TARGET_COL]].values)

SZA_IDX = FEATURE_COLS.index("SZA")
KT_IDX  = FEATURE_COLS.index("ALLSKY_KT")
NUM_FEATURES = len(FEATURE_COLS)

# ──────────────────────────────────────────────────────────────────────────────
# 3. LOAD TEST DATA (2020-2024) AND INFER
# ──────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Processing unseen multi-year horizon (2020-2024)...")
df_test_full = process_data(TEST_FILE)
df_test = df_test_full[df_test_full["YEAR"] < 2025].copy()

model = PhysicsInformedCrossAttention(
    num_features=NUM_FEATURES, sub_window=5, conv_filters=64,
    fc_hidden=128, sza_idx=SZA_IDX, kt_idx=KT_IDX
).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
model.eval()

def create_seq(features, targets, window=24, horizon=1):
    X, y = [], []
    for i in range(len(features) - window - horizon + 1):
        X.append(features[i : i + window])
        y.append(targets[i + window + horizon - 1])
    return np.array(X), np.array(y)

years_available = sorted(df_test["YEAR"].unique())
yearly_metrics = {}

print(f"\n{'='*60}\n  Commencing Year-by-Year Robustness Testing\n{'='*60}")

for year in years_available:
    df_y = df_test[df_test["YEAR"] == year].copy()
    if len(df_y) < 25: continue
    
    feats = scaler_features.transform(df_y[FEATURE_COLS].values)
    targs = scaler_target.transform(df_y[[TARGET_COL]].values).flatten()
    
    X_y, Y_y = create_seq(feats, targs)
    X_t = torch.tensor(X_y, dtype=torch.float32).to(DEVICE)
    
    # Process in chunks to save memory
    preds_y = []
    with torch.no_grad():
        for i in range(0, len(X_t), 512):
            p = model(X_t[i:i+512]).cpu().numpy()
            preds_y.append(p)
    preds_y = np.concatenate(preds_y).flatten()
    
    p_orig = scaler_target.inverse_transform(preds_y.reshape(-1, 1)).flatten()
    t_orig = scaler_target.inverse_transform(Y_y.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((p_orig - t_orig)**2))
    mae = np.mean(np.abs(p_orig - t_orig))
    
    yearly_metrics[year] = {"RMSE": rmse, "MAE": mae, "Preds": p_orig, "Actuals": t_orig}
    print(f"  [Year {int(year)}]  -> RMSE: {rmse:6.2f} Wh/m²  |  MAE: {mae:6.2f} Wh/m²")

# ──────────────────────────────────────────────────────────────────────────────
# 4. PROFESSIONAL ENGLISH PUBLICATIONS PLOTS (NON-OVERLAPPING)
# ──────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif', 'font.weight': 'bold',
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'figure.facecolor': 'white'
})

years = list(yearly_metrics.keys())
rmses = [yearly_metrics[y]["RMSE"] for y in years]
maes = [yearly_metrics[y]["MAE"] for y in years]

# PLOT 1: Performance Bar Chart 
fig_bar, ax = plt.subplots(figsize=(10, 6), dpi=300)
x = np.arange(len(years))
width = 0.35

rects1 = ax.bar(x - width/2, rmses, width, label='RMSE (Wh/m²)', color='#1a73e8', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, maes, width, label='MAE (Wh/m²)', color='#ff7f0e', edgecolor='black', alpha=0.9)

ax.set_ylabel('Error Magnitude', fontsize=14, labelpad=10)
ax.set_title('Cross-Attention Model Performance Across Years (2020-2024)', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([str(int(y)) for y in years], fontsize=13)
ax.legend(fontsize=12, loc='upper left', framealpha=1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
                
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/1_MultiYear_BarChart.png", dpi=300, bbox_inches='tight')
plt.close(fig_bar)

# PLOT 2: Regression Scatter Array
num_plots = len(years)
cols = min(3, num_plots)
rows = math.ceil(num_plots / cols)

fig_scat, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), dpi=300)
fig_scat.suptitle("Regression Fidelity: Predicted vs Actual Solar Irradiance", fontsize=20, y=1.02)
axes = np.array(axes).flatten()

for i, year in enumerate(years):
    ax = axes[i]
    t_val = yearly_metrics[year]["Actuals"]
    p_val = yearly_metrics[year]["Preds"]
    
    ax.scatter(t_val, p_val, alpha=0.15, s=8, c='#34a853', edgecolors='none')
    max_val = max(np.max(t_val), np.max(p_val))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label="Ideal Fit" if i==0 else "")
    
    ax.set_title(f"Year {int(year)} (RMSE: {rmses[i]:.1f})", fontsize=15, pad=12)
    ax.set_xlabel("Ground Truth (Wh/m²)", fontsize=12)
    ax.set_ylabel("Model Prediction (Wh/m²)", fontsize=12)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle=':', alpha=0.6)
    if i == 0: ax.legend(loc='upper left')

for j in range(len(years), len(axes)):
    fig_scat.delaxes(axes[j])

fig_scat.tight_layout(pad=2.0)
plt.savefig(f"{PLOT_DIR}/2_Regression_Scatter.png", dpi=300, bbox_inches='tight')
plt.close(fig_scat)

# PLOT 3: Deep Dive 100-Hour Trace Selection per Year (Time Series Comparison)
fig_trace, axes = plt.subplots(len(years), 1, figsize=(14, 2.5 * len(years)), dpi=300)
fig_trace.suptitle("100-Hour Micro-Traces: Cross-Attention Target Tracking", fontsize=18, y=1.02)

slice_size = 100
for i, year in enumerate(years):
    ax = axes[i] if len(years) > 1 else axes
    
    t_val = yearly_metrics[year]["Actuals"][2000:2000+slice_size]
    p_val = yearly_metrics[year]["Preds"][2000:2000+slice_size]
    idx = np.arange(len(t_val))
    
    ax.plot(idx, t_val, color='#1a73e8', linewidth=2.0, alpha=0.9, label="Actual Truth")
    ax.plot(idx, p_val, color='#ea4335', linewidth=2.0, alpha=0.9, linestyle="--", label="Model Output")
    ax.fill_between(idx, t_val, p_val, color='#fbbc04', alpha=0.3, label="Error Span")
    
    ax.set_title(f"Year {int(year)} - Spring Trace Specimen", fontsize=14, loc='left', pad=10)
    ax.set_ylabel("Wh/m²", fontsize=12)
    ax.set_xlim(0, slice_size - 1)
    ax.grid(True, alpha=0.4, linestyle='--')
    if i == 0: ax.legend(loc='upper right', ncol=3)
    
axes[-1].set_xlabel("Hourly Index (Continuous Slice)", fontsize=14)

fig_trace.tight_layout(pad=2.0)
plt.savefig(f"{PLOT_DIR}/3_TimeSeries_MicroTraces.png", dpi=300, bbox_inches='tight')
plt.close(fig_trace)

print(f"\n{'='*75}")
print(f" 🎉 Multi-Year Evaluation Complete!")
print(f" 📂 3 Sets of High-Resolution Publication Plots saved to: {PLOT_DIR}/")
print(f"{'='*75}\n")
