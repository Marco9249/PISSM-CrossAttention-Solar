"""
================================================================================
Physics-Informed State Space Model (PISSM) for Solar Irradiance Forecasting
================================================================================

Author : Prepared by: Eng. Mohammed Izzaldeen Babeker Abdullah
Data   : NASA POWER Hourly (2010-2015), Khartoum, Sudan
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import math

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Set reproducibility seeds and device
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ==============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# ==============================================================================

# ── 1.1  Read NASA POWER CSV, skipping all header lines ─────────────────────
DATA_PATH = "Hourly_2010_2015.csv"

# Detect the header end line
with open(DATA_PATH, "r") as f:
    for idx, line in enumerate(f):
        if "-END HEADER-" in line:
            skip_rows = idx + 1   # skip all lines up to and including this one
            break

print(f"[INFO] Skipping {skip_rows} header lines before column names.")
df = pd.read_csv(DATA_PATH, skiprows=skip_rows)
print(f"[INFO] Raw data shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")

# ── 1.2  Replace missing -999 values with time-interpolated neighbors ───────
df.replace(-999.0, np.nan, inplace=True)
df.replace(-999, np.nan, inplace=True)
df.interpolate(method="time" if "time" in dir(df) else "linear", inplace=True)
df.interpolate(method="linear", inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)
print(f"[INFO] Missing values after interpolation: {df.isnull().sum().sum()}")

# ── 1.3  Filter to required columns only (explicit string names) ────────────
# Target variable
TARGET_COL = "ALLSKY_SFC_SW_DWN"

# Physical input columns
PHYSICAL_COLS = [
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_KT",
    "SZA",
    "T2M",
    "RH2M",
    "WS10M",
    "PS",
]

# Time columns for cyclical encoding
TIME_COLS = ["MO", "DY", "HR"]

# Keep only the required columns
KEEP_COLS = [TARGET_COL] + PHYSICAL_COLS + TIME_COLS
df = df[KEEP_COLS].copy()
print(f"[INFO] Filtered DataFrame shape: {df.shape}")

# ── 1.4  Physical Night Mask ────────────────────────────────────────────────
# Set all radiation variables and target to zero where clear-sky <= 0
RADIATION_COLS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
]
night_mask = df["CLRSKY_SFC_SW_DWN"] <= 0.0
for col in RADIATION_COLS:
    df.loc[night_mask, col] = 0.0
print(f"[INFO] Night-masked {night_mask.sum()} hourly records.")

# ── 1.5  Logarithmic Transformation (ln(x+1)) for skewed variables ─────────
LOG_COLS = ["ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M"]
# T2M can be negative; shift to ensure positivity before log
t2m_shift = 0.0
if df["T2M"].min() < 0:
    t2m_shift = abs(df["T2M"].min()) + 1.0
    df["T2M"] = df["T2M"] + t2m_shift

for col in LOG_COLS:
    df[col] = np.log1p(df[col].clip(lower=0))
print(f"[INFO] Applied ln(1+x) to: {LOG_COLS}  (T2M shift={t2m_shift:.2f})")

# ── 1.6  Cyclical Time Encoding (sin/cos) ──────────────────────────────────
df["MO_sin"] = np.sin(2 * np.pi * df["MO"] / 12.0)
df["MO_cos"] = np.cos(2 * np.pi * df["MO"] / 12.0)
df["DY_sin"] = np.sin(2 * np.pi * df["DY"] / 31.0)
df["DY_cos"] = np.cos(2 * np.pi * df["DY"] / 31.0)
df["HR_sin"] = np.sin(2 * np.pi * df["HR"] / 24.0)
df["HR_cos"] = np.cos(2 * np.pi * df["HR"] / 24.0)

CYCLICAL_COLS = ["MO_sin", "MO_cos", "DY_sin", "DY_cos", "HR_sin", "HR_cos"]
print(f"[INFO] Added cyclical time features: {CYCLICAL_COLS}")

# Drop the raw time columns
df.drop(columns=TIME_COLS, inplace=True)

# ── 1.7  Define final feature columns ──────────────────────────────────────
# Physical features (9) + Cyclical features (6) = 15 input features
FEATURE_COLS = PHYSICAL_COLS + CYCLICAL_COLS
NUM_FEATURES = len(FEATURE_COLS)
print(f"[INFO] Total input features: {NUM_FEATURES}  ->  {FEATURE_COLS}")

# ==============================================================================
# SECTION 2 — CHRONOLOGICAL TRAIN/TEST SPLIT (80/20)
# ==============================================================================

TRAIN_RATIO = 0.80
n_total = len(df)
n_train = int(n_total * TRAIN_RATIO)

df_train = df.iloc[:n_train].copy()
df_test  = df.iloc[n_train:].copy()
print(f"[INFO] Train samples: {len(df_train)},  Test samples: {len(df_test)}")

# ==============================================================================
# SECTION 3 — HYBRID NORMALIZATION (fit on train only)
# ==============================================================================

# StandardScaler for physical + cyclical features
scaler_features = StandardScaler()
scaler_features.fit(df_train[FEATURE_COLS].values)

# MinMaxScaler for target variable
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaler_target.fit(df_train[[TARGET_COL]].values)

# Transform both splits
train_features = scaler_features.transform(df_train[FEATURE_COLS].values)
test_features  = scaler_features.transform(df_test[FEATURE_COLS].values)

train_target = scaler_target.transform(df_train[[TARGET_COL]].values).flatten()
test_target  = scaler_target.transform(df_test[[TARGET_COL]].values).flatten()

# Also keep raw SZA and ALLSKY_KT indices for physics gating (normalized)
SZA_IDX = FEATURE_COLS.index("SZA")
KT_IDX  = FEATURE_COLS.index("ALLSKY_KT")
print(f"[INFO] SZA feature index: {SZA_IDX},  KT feature index: {KT_IDX}")

# ==============================================================================
# SECTION 4 — SLIDING WINDOW DATASET CREATION
# ==============================================================================

INPUT_WINDOW  = 24    # 24 time steps as main input window
HORIZON       = 1     # predict the 25th step
SUB_WINDOW    = 5     # sub-window for Hankel matrix
# Number of Hankel sub-windows per sample: 24 - 5 + 1 = 20
NUM_HANKEL_WINDOWS = INPUT_WINDOW - SUB_WINDOW + 1  # = 20

print(f"[INFO] Input window: {INPUT_WINDOW}, Sub-window: {SUB_WINDOW}")
print(f"[INFO] Hankel windows per sample: {NUM_HANKEL_WINDOWS}")


def create_sequences(features: np.ndarray, target: np.ndarray,
                     input_window: int, horizon: int):
    """
    Create (X, y) pairs using a sliding window approach.
    X shape: (num_samples, input_window, num_features)
    y shape: (num_samples,)
    """
    X_list, y_list = [], []
    total_len = len(features)
    for i in range(total_len - input_window - horizon + 1):
        X_list.append(features[i : i + input_window])
        y_list.append(target[i + input_window + horizon - 1])
    return np.array(X_list), np.array(y_list)


X_train, y_train = create_sequences(train_features, train_target,
                                     INPUT_WINDOW, HORIZON)
X_test,  y_test  = create_sequences(test_features, test_target,
                                     INPUT_WINDOW, HORIZON)

print(f"[INFO] X_train: {X_train.shape},  y_train: {y_train.shape}")
print(f"[INFO] X_test : {X_test.shape},   y_test : {y_test.shape}")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

# DataLoaders
BATCH_SIZE = 256
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=False)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           drop_last=False)

# ==============================================================================
# SECTION 5 — MODEL ARCHITECTURE
# ==============================================================================

class HankelEmbeddingLayer(nn.Module):
    """
    Dynamic Hankel Matrix Construction Layer.
    
    Takes an input tensor of shape (batch, seq_len, features) and uses
    PyTorch's unfold operation to extract overlapping sub-windows of size
    `sub_window`, producing a 3D tensor:
        (batch, num_windows, features * sub_window)
    
    For seq_len=24, sub_window=5:
        num_windows = 24 - 5 + 1 = 20
        output_dim  = num_features * 5
    """
    def __init__(self, sub_window: int):
        super().__init__()
        self.sub_window = sub_window
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq_len, features = x.shape
        
        # Permute to (batch, features, seq_len) for unfold on last dim
        x_perm = x.permute(0, 2, 1)
        
        # unfold(dimension, size, step) -> (batch, features, num_windows, sub_window)
        x_unf = x_perm.unfold(2, self.sub_window, 1)
        
        num_windows = x_unf.size(2)
        
        # Arrange to (batch, num_windows, features * sub_window)
        x_unf = x_unf.permute(0, 2, 1, 3)
        x_flat = x_unf.reshape(batch, num_windows, features * self.sub_window)
        
        return x_flat


class CNN1DLayer(nn.Module):
    """
    First Layer: 1D Convolutional Neural Network Feature Extractor
    
    Receives the 3D matrix resulting from the Hankel transformation with dimensions
    (batch, 20, 75). Since Conv1D expects (batch, in_channels, seq_len), the tensor
    is permuted to (batch, 75, 20).
    It then passes through a 1D CNN with 64 filters to form (batch, 64, 20),
    followed by GELU activation and 0.2 Dropout. Dimensions are then permuted back
    to (batch, 20, 64).
    """
    def __init__(self, in_channels: int = 75, out_channels: int = 64, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x input dimensions: (batch, 20, 75)
        # Permute to mathematically align with 1D Convolutional requirements
        x = x.permute(0, 2, 1)     # -> (batch, 75, 20)
        x = self.conv(x)           # -> (batch, 64, 20)
        x = self.activation(x)     
        x = self.dropout(x)        
        # Permute back to standard format for Attention processing
        x = x.permute(0, 2, 1)     # -> (batch, 20, 64)
        return x


class TemporalSelfAttentionLayer(nn.Module):
    """
    Second Layer: Temporal Self-Attention Layer
    
    Processes the extracted temporal features outputted by the convolutional network.
    Constructed using MultiheadAttention with embed_dim=64, num_heads=4 and batch_first=True.
    The Queries (Q), Keys (K), and Values (V) are all exactly the outputs of the CNN layer.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x dimensions: (batch, 20, 64)
        attn_out, _ = self.self_attn(query=x, key=x, value=x)
        return attn_out


class PhysicalCrossAttentionLayer(nn.Module):
    """
    Third Layer: Physical Cross-Attention Layer
    
    Integrates guiding physical variables into the attention mechanism to ensure physics constraints.
    - Queries: Temporal features resulting from self-attention, dimension (batch, 20, 64).
    - Keys & Values: Formed by concatenating SZA and ALLSKY_KT to matrix (batch, 20, 2),
      then passing through a Linear Layer projection to expand dimensions to (batch, 20, 64).
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.phys_proj = nn.Linear(2, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, temporal_features, sza_seq, kt_seq):
        # temporal_features: (batch, 20, 64)
        # sza_seq: (batch, 20, 1), kt_seq: (batch, 20, 1)
        
        # Concatenate guiding physical variables to matrix path -> (batch, 20, 2)
        phys_vars = torch.cat([sza_seq, kt_seq], dim=-1)
        
        # Expand linear dimension representing Keys and Values
        phys_proj = self.phys_proj(phys_vars)            # (batch, 20, 64)
        
        # Cross-Attention interaction mapping
        attn_out, _ = self.cross_attn(query=temporal_features, key=phys_proj, value=phys_proj)
        
        return attn_out


class PhysicsInformedCrossAttention(nn.Module):
    """
    Physics-Informed Cross-Attention Architecture.
    
    Pipeline layout built adhering to exact topological programmatic requests:
      - HankelEmbeddingLayer: Matrix structure extraction from overlapping input sequences.
      - First Layer  : 1D CNN + GELU + Dropout(0.2)
      - Second Layer : Temporal Self-Attention based on MHA
      - Third Layer  : Physical Cross-Attention utilizing projected SZA & ALLSKY_KT variables
      - Fourth Layer : Extract final timesteps mapping through FC path (64 -> 128 -> GELU -> Dropout -> 1)
      - Fifth Layer  : ReLU Threshold constraint against physically disallowed negativity 
    """
    def __init__(self, num_features: int, sub_window: int = 5,
                 conv_filters: int = 64, fc_hidden: int = 128,
                 sza_idx: int = 4, kt_idx: int = 3):
        super().__init__()
        
        self.num_features = num_features
        self.sub_window   = sub_window
        self.sza_idx      = sza_idx
        self.kt_idx       = kt_idx
        hankel_dim        = num_features * sub_window  # e.g., 15 * 5 = 75
        
        # ── Pre-Layer: Hankel Embedding ──────────────────────────────────
        self.hankel = HankelEmbeddingLayer(sub_window=sub_window)
        
        # ── First Layer: 1D Convolutional Network ────────────────────────
        self.layer1_cnn = CNN1DLayer(
            in_channels=hankel_dim, 
            out_channels=conv_filters, 
            dropout=0.2
        )
        
        # ── Second Layer: Temporal Self Attention ────────────────────────
        self.layer2_self_attn = TemporalSelfAttentionLayer(
            embed_dim=conv_filters, 
            num_heads=4
        )
        
        # ── Third Layer: Physical Cross Attention ────────────────────────
        self.layer3_cross_attn = PhysicalCrossAttentionLayer(
            embed_dim=conv_filters, 
            num_heads=4
        )
        
        # ── Fourth Layer: Fully Connected Integrator ─────────────────────
        self.layer4_fc = nn.Sequential(
            nn.Linear(conv_filters, fc_hidden),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(fc_hidden, 1)
        )
        
        # ── Fifth Layer: Safety Post-Processing Filter ───────────────────
        self.layer5_relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor batch of shape (batch, 24, num_features)
        Returns:
            pred (Tensor): Predicted scalar solar irradiance (batch, 1)
        """
        # Step 0: Extract full required temporal physical dependencies for the guiding layers
        sza_full = x[:, :, self.sza_idx].unsqueeze(-1)  # (batch, 24, 1)
        kt_full  = x[:, :, self.kt_idx].unsqueeze(-1)   # (batch, 24, 1)
        
        # Pre-Layer: Hankel Embedding Generation -> (batch, 20, 75)
        h = self.hankel(x)
        num_windows = h.size(1)  # Equals 20 when seq_len is 24 and sub_window is 5
        
        # Truncating physical variables context down to match the Hankel reduced windows size
        sza_seq = sza_full[:, -num_windows:, :]         # (batch, 20, 1)
        kt_seq  = kt_full[:, -num_windows:, :]          # (batch, 20, 1)
        
        # First Layer implementation
        h_cnn = self.layer1_cnn(h)                      # Output mapping dimensions: (batch, 20, 64)
        
        # Second Layer implementation
        h_self = self.layer2_self_attn(h_cnn)           # Output mapping dimensions: (batch, 20, 64)
        
        # Third Layer implementation 
        h_cross = self.layer3_cross_attn(h_self, sza_seq, kt_seq) # Output mapping dimensions: (batch, 20, 64)
        
        # Fourth Layer implementation targeting the last timestep solely 
        h_last = h_cross[:, -1, :]                      # Extract last index making dimensions (batch, 64)
        pred = self.layer4_fc(h_last)                   # Dense pathway mapping to generating value (batch, 1)
        
        # Fifth Layer physical threshold compliance logic
        pred = self.layer5_relu(pred)
        
        return pred

# ==============================================================================
# SECTION 6 — MODEL INSTANTIATION
# ==============================================================================

model = PhysicsInformedCrossAttention(
    num_features=NUM_FEATURES,
    sub_window=SUB_WINDOW,
    conv_filters=64,
    fc_hidden=128,
    sza_idx=SZA_IDX,
    kt_idx=KT_IDX
).to(DEVICE)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*60}")
print(f"  PISSM Model Architecture Summary")
print(f"{'='*60}")
print(model)
print(f"\n  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")
print(f"{'='*60}\n")

# ==============================================================================
# SECTION 7 — TRAINING LOOP
# ==============================================================================

NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=15
)
criterion = nn.MSELoss()


def compute_rmse(preds, targets):
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def compute_mae(preds, targets):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(preds - targets)).item()


print(f"{'='*70}")
print(f"  Training PISSM — {NUM_EPOCHS} Epochs, LR={LEARNING_RATE}")
print(f"{'='*70}")

train_losses = []
val_losses   = []

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Training Phase ───────────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = epoch_loss / num_batches
    train_rmse = math.sqrt(avg_train_loss)
    train_losses.append(avg_train_loss)
    
    # ── Validation Phase ─────────────────────────────────────────────────
    model.eval()
    val_preds_list = []
    val_targets_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            preds = model(X_batch).squeeze(1)
            val_preds_list.append(preds.cpu())
            val_targets_list.append(y_batch.cpu())
    
    val_preds_all   = torch.cat(val_preds_list)
    val_targets_all = torch.cat(val_targets_list)
    val_rmse = compute_rmse(val_preds_all, val_targets_all)
    val_mae  = compute_mae(val_preds_all, val_targets_all)
    val_losses.append(val_rmse ** 2)
    
    scheduler.step(val_rmse)
    
    print(f"  Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
          f"Train RMSE: {train_rmse:.6f}  |  "
          f"Val RMSE: {val_rmse:.6f}  |  "
          f"Val MAE: {val_mae:.6f}")

print(f"\n{'='*70}")
print(f"  Training Complete!")
print(f"{'='*70}\n")
# 💾 حفظ الأوزان هنا في النهاية )
torch.save(model.state_dict(), "pissm_saved_weights.pth")
print(f"\n[INFO] Model weights saved successfully to: pissm_saved_weights.pth")
# ==============================================================================
# SECTION 8 — INVERSE TRANSFORM & FINAL EVALUATION
# ==============================================================================

model.eval()
all_preds   = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).squeeze(1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

all_preds   = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# ── Inverse transform to original physical scale ────────────────────────
preds_original   = scaler_target.inverse_transform(
    all_preds.reshape(-1, 1)
).flatten()
targets_original = scaler_target.inverse_transform(
    all_targets.reshape(-1, 1)
).flatten()

# ── Compute final metrics in original Wh/m² units ───────────────────────
final_rmse = np.sqrt(np.mean((preds_original - targets_original) ** 2))
final_mae  = np.mean(np.abs(preds_original - targets_original))

print(f"{'='*60}")
print(f"  Final Test Metrics (Original Scale — Wh/m²)")
print(f"{'='*60}")
print(f"  RMSE : {final_rmse:.4f} Wh/m²")
print(f"  MAE  : {final_mae:.4f} Wh/m²")
print(f"{'='*60}\n")

# ==============================================================================
# SECTION 9 — PROFESSIONAL VISUALIZATION
# ==============================================================================

def plot_predictions(targets, predictions, rmse, mae, num_points=500):
    """
    Create a professional comparison plot of predictions vs actual values.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        "Physics-Informed State Space Model (PISSM)\n"
        "Solar Irradiance Forecasting — Test Set Performance",
        fontsize=16, fontweight="bold", y=0.98
    )
    
    # Limit to `num_points` for readability
    n = min(num_points, len(targets))
    idx = np.arange(n)
    t = targets[:n]
    p = predictions[:n]
    
    # ── Top panel: Overlay plot ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(idx, t, color="#1a73e8", linewidth=1.2, alpha=0.85,
             label="Actual (Ground Truth)")
    ax1.plot(idx, p, color="#ea4335", linewidth=1.0, alpha=0.80,
             linestyle="--", label="Predicted (PISSM)")
    ax1.fill_between(idx, t, p, alpha=0.15, color="#fbbc04")
    ax1.set_ylabel("Solar Irradiance (Wh/m²)", fontsize=13)
    ax1.set_title(
        f"RMSE = {rmse:.2f} Wh/m²   |   MAE = {mae:.2f} Wh/m²",
        fontsize=12, color="#555555"
    )
    ax1.legend(fontsize=12, loc="upper right",
               framealpha=0.9, edgecolor="#cccccc")
    ax1.set_xlim(0, n)
    ax1.grid(True, alpha=0.3)
    
    # ── Bottom panel: Error plot ─────────────────────────────────────────
    ax2 = axes[1]
    errors = t - p
    colors = np.where(errors >= 0, "#34a853", "#ea4335")
    ax2.bar(idx, errors, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color="#333333", linewidth=0.8)
    ax2.set_xlabel("Test Sample Index", fontsize=13)
    ax2.set_ylabel("Error (Wh/m²)", fontsize=13)
    ax2.set_title("Prediction Error (Actual − Predicted)", fontsize=11,
                  color="#555555")
    ax2.set_xlim(0, n)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("pissm_forecast_results.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("[INFO] Plot saved to:  pissm_forecast_results.png")


def plot_training_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, [math.sqrt(l) for l in train_losses],
            color="#1a73e8", linewidth=2, marker="o", markersize=3,
            label="Train RMSE")
    ax.plot(epochs, [math.sqrt(l) for l in val_losses],
            color="#ea4335", linewidth=2, marker="s", markersize=3,
            label="Validation RMSE")
    
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("RMSE (Normalized)", fontsize=13)
    ax.set_title("PISSM Training Convergence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pissm_training_curves.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("[INFO] Plot saved to:  pissm_training_curves.png")


# Generate plots
plot_training_curves(train_losses, val_losses)
plot_predictions(targets_original, preds_original, final_rmse, final_mae,
                 num_points=500)

print("\n" + "=" * 60)
print("  PISSM Pipeline Complete — All outputs generated.")
print("=" * 60)
