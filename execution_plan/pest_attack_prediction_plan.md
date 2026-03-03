# Execution Plan: Pest Attack Prediction using Spatio-Temporal Epidemic Modelling

## Dataset Summary

| Property         | Details |
|------------------|---------|
| File             | `data/Normalised_KCC_Data.csv` |
| Rows             | 686,118 |
| Date Range       | 2013-01-01 to 2020-07-27 |
| Spatial Coverage | 31 Indian States, 550 Districts |
| Target Variable  | `Count` (number of pest attack reports) |
| Pest Types       | 65 unique (top: insect, caterpiller, aphid, fruit&shootborer, thrip, whitefly, stemborer, termite) |
| Crops            | 283 unique |
| Features         | Year, Month, State, District, Latitude, Longitude, Pest, Crop, Rainfall (MM), Harvest Area |
| Missing Data     | Rainfall: ~70% missing; Harvest Area: ~58% missing |

---

## Epidemic Analogy for Pest Spread

Pest attacks propagate through agricultural regions in a manner analogous to disease epidemics:

| SIR Epidemic Concept | Pest Attack Equivalent |
|----------------------|------------------------|
| Susceptible (S) | Crop area not yet attacked |
| Infected (I) | Crop area currently under pest attack (Count) |
| Recovered (R) | Crop area where pest attack has ceased |
| Transmission rate (β) | Pest spread rate (influenced by climate, adjacency) |
| Recovery rate (γ) | Pest control / natural die-off rate |
| Spatial diffusion | Pest migration across neighboring districts/states |

The existing `SSIR_STGCN` / `SSIR_STGAT` model (Spatio-temporal GCN + SIR ODE) is directly applicable.

---

## Phase 1: Data Preprocessing (`kcc_codebase/data_preprocessing.py`)

### 1.1 Load and Clean Raw Data
- Load `Normalised_KCC_Data.csv`
- Parse `CreatedOn` as datetime; derive `Year`, `Month` columns
- Standardize string columns (lowercase strip): `State Name`, `Dist Name`, `Pest`, `Crop`

### 1.2 Handle Missing Values
- `Rainfall (MM)`: Impute using **monthly-district-level median** from available records; where still missing, use state-level median; remaining use 0
- `Harvest Area`: Impute using district-level mean over time; where still missing use state-level mean
- Drop rows where `Count` is null (there are none, but as a safety check)

### 1.3 Aggregate to Monthly Time Series
- Granularity: **State × Month** (31 states × ~84 months = ~2604 time slots)
- Aggregation:
  - `Count` → sum per (State, Month) → this is the "I" signal
  - `Rainfall (MM)` → mean per (State, Month)
  - `Harvest Area` → mean per (State, Month)
- Also compute per-pest aggregations for pest-specific models (optional in Phase 4)

### 1.4 Construct SIR State Variables
From the aggregated monthly `Count` per state:
- **I(t)** = Count (pest-attacked crop area units) at time t
- **S(t)** = Total Harvest Area - cumulative I up to t (approximate)
- **R(t)** = Harvest Area - S(t) - I(t)
- Normalize: divide S, I, R by total Harvest Area per state → values in [0, 1]

### 1.5 Output
- Save: `kcc_codebase/processed_data/kcc_monthly_sir.csv`
  - Columns: `date`, `state`, `S`, `I`, `R`, `rainfall`, `harvest_area`
- Shape after pivot: `(T=84, N=31, F=3)` → matches existing model input `(B, T, N, F)`

---

## Phase 2: Spatial Graph Construction (`kcc_codebase/graph_construction.py`)

### 2.1 State-Level Adjacency Matrix (Static)
- Use geographic adjacency (which Indian states border each other)
- Build a 31×31 binary adjacency matrix based on physical borders
- Row-normalize the matrix (or use degree-normalized Laplacian)
- Save as: `kcc_codebase/processed_data/neighbor_adjacency_matrix.csv`

### 2.2 Distance-Based Soft Adjacency (Alternative)
- Use state centroid lat/long (average of district lat/long within state)
- Compute pairwise haversine distance matrix
- Apply Gaussian kernel: `A[i,j] = exp(-d²/σ²)` with threshold cutoff
- Save as: `kcc_codebase/processed_data/neighbor_adjacency_matrix_soft.csv`

### 2.3 Correlation-Based Adjacency (For Dynamic Graph baseline)
- Compute Pearson correlation of I(t) time series between all state pairs
- Threshold at r > 0.5 → binary adjacency
- Used to validate whether Dynamic graph type captures real correlations

---

## Phase 3: Dataset Preparation & DataLoader (`kcc_codebase/dataset_builder.py`)

### 3.1 Sliding Window Construction
- Follow existing `window_rolling` logic from `Main.py`
- Input window (`obs_len`): 12 months (1 year look-back)
- Prediction window (`pre_len`): 3 months (quarterly forecast)
- Stride: 1 month
- Result: sliding windows of shape `(B, T_obs+T_pre, N, F=3)`

### 3.2 Train / Validation / Test Split
- Temporal split (no shuffle to avoid data leakage):
  - Train: 2013-01 → 2017-12 (60%)
  - Validation: 2018-01 → 2018-12 (20%)
  - Test: 2019-01 → 2020-07 (20%)
- Split ratio configurable (default `[6, 2, 2]` matching existing args)

### 3.3 Normalization
- Min-max normalize I per state (track `imin`, `imax` per state for de-normalization)
- Save `auxdata` dict: `{'prov_pop': harvest_area_per_state, 'imax': ..., 'imin': ...}`
- Match existing de-normalization logic in `Train.py` (lines 519-535)

### 3.4 Output
- PyTorch `DataLoader` objects: `{'training': ..., 'validation': ..., 'test': ...}`
- Tensors of shape `(B, T_total, N=31, F=3)` with labels `yd` (daily/monthly I values)

---

## Phase 4: Model Configuration (`kcc_codebase/model_config.py`)

### 4.1 Baseline Models
| Model | Description |
|-------|-------------|
| `SSIR_ODEFIT` | Pure SIR ODE fitting — learn β, γ per state; no deep learning |
| `LSTM` (standalone) | Per-state LSTM without spatial graph |

### 4.2 Primary Model
| Model | Config |
|-------|--------|
| `SSIR_STGCN` (Static graph) | 31 states, obs=12, pre=3, static adjacency from Phase 2 |
| `SSIR_STGCN` (Dynamic graph) | Learn adjacency from data |
| `SSIR_STGCN` (Adaptive graph) | Learnable adjacency parameter |

### 4.3 Hyperparameters (starting config)

```python
params = {
    "data_type": "kcc",          # new data_type identifier
    "graph_type": "Dynamic",     # or Static / Adaptive
    "obs_len": 12,               # 12 months look-back
    "pre_len": 3,                # 3 months forecast
    "batch_size": 16,
    "kernel_size": 3,
    "num_layers": 3,
    "t_out_dim": 16,
    "s_out_dim": 16,
    "dropout": 0.1,
    "loss_type": "cMAE",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epoch": 200,
    "early_stop": 20,
    "scheduler": "ReduceLROnPlateau",
    "normalize": "minmax",
    "beta_incorporated": True,
    "ssir": "ssir",              # or 'sir'
    "w4phy": 0.01,               # physics loss weight
    "daily": False,              # monthly aggregation
    "dev": "cpu"                 # pytorch-cpu env
}
```

### 4.4 Constant.py Updates Required
- Add `"kcc"` as a new `data_type` in `Constant.py`
- Add path entry: `KCC_DATA = os.path.join(DATA_REPO, "kcc_monthly_sir.csv")`
- Update `nsize` in `Train.py` `get_model()`: add `elif data_type == 'kcc': nsize = 31`

---

## Phase 5: Training & Evaluation (`kcc_codebase/train_eval.py`)

### 5.1 Training Loop
- Use existing `Trainer.train()` from `code/Train.py`
- Run with KCC-specific DataLoader and params from Phase 4
- Log training/validation loss curve
- Save best checkpoint: `SSIR_STGCN_optmodel.pkl`

### 5.2 Evaluation Metrics (already implemented in `Trainer.evaluate()`)
- MAE, cMAE, MSE, cMSE, RMSE, MAPE, RAE, PCC, CCC
- Evaluated on: Test set, rolling horizons (1-month, 2-month, 3-month ahead)

### 5.3 Baselines for Comparison
| Baseline | Purpose |
|----------|---------|
| Historical mean | Naive benchmark |
| `SSIR_ODEFIT` | Physics-only benchmark |
| LSTM (no graph) | Deep learning without spatial structure |
| `SSIR_STGCN` Static | Full model, static graph |
| `SSIR_STGCN` Dynamic | Full model, dynamic graph |

---

## Phase 6: Visualization & Analysis (`kcc_codebase/visualization.py`)

### 6.1 Forecast Curves
- Per-state plot: ground truth vs prediction vs physics-only curve
- Plot for Test period (2019-2020)
- Reuse `Toolkits.plot_forecurve()`

### 6.2 Spatial Heatmaps
- Map of India: color-coded by predicted pest attack intensity per state per month
- Compare predicted vs actual heatmaps side by side

### 6.3 Epidemic Parameter Analysis
- Plot learned β (transmission rate) per state over time
- States with high β = high pest spread risk
- Correlate β with Rainfall patterns

### 6.4 Pest-Specific Analysis (Optional)
- Repeat Phase 1–5 for top 5 pests separately (insect, caterpiller, aphid, stemborer, whitefly)
- Compare spatial spread patterns per pest type

---

## Phase 7: Pest Risk Alert System (Optional Extension)

### 7.1 Risk Score Definition
- Risk = predicted I(t+1) / S(t) per state
- Risk levels: Low (<0.1), Medium (0.1–0.3), High (>0.3)

### 7.2 Early Warning Output
- Output a monthly risk table per state
- Flag states exceeding threshold for next quarter

---

## Directory Structure for `kcc_codebase/`

```
kcc_codebase/
├── data_preprocessing.py    # Data cleaning, aggregation, SIR construction
├── graph_construction.py    # Adjacency matrix (static, soft, correlation)
├── dataset_builder.py       # Sliding window, DataLoader, normalization
├── model_config.py          # Hyperparameter config, Constant.py patches
├── train_eval.py            # Training runner using existing Trainer class
├── visualization.py         # Forecast plots, heatmaps, epi-params
├── processed_data/
│   ├── kcc_monthly_sir.csv         # Aggregated monthly SIR per state
│   └── neighbor_adjacency_matrix.csv  # 31×31 adjacency matrix
└── results/                        # Model outputs, error CSVs, plots
```

---

## Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                                    ↕
                             (iterate hyperparams)
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 70% missing Rainfall | Use median imputation; include missingness indicator feature |
| Sparse monthly counts (most = 1) | Aggregate at state level; consider log1p transform |
| No 2019 data (year gap) | Use 2013–2018 for train, 2020 (partial) for test only |
| 550 districts too sparse | Use state-level (31) aggregation for first experiments |
| SIR assumes closed population | Use Harvest Area as proxy for N (total population) |
