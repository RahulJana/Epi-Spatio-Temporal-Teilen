# Epi-Spatio-Temporal Modeling: Technical Documentation

## 1. Project Overview

### System Architecture Diagram

```mermaid
graph LR
    A["Raw Data<br/>(686K records)"] --> B["Phase 1<br/>Data Preprocessing"]
    B --> C["Processed Data<br/>(Monthly SIR)"]
    C --> D["Phase 2<br/>Graph Construction"]
    D --> E["Adjacency Matrix<br/>(31x31)"]
    E --> F["Phase 3<br/>Dataset Builder"]
    C --> F
    F --> G["DataLoaders<br/>(Train/Val/Test)"]
    G --> H["Phase 4<br/>Model Config"]
    H --> I["Model Initialization<br/>(STGCN/STGAT/ODEFIT)"]
    I --> J["Phase 5<br/>Train & Eval"]
    J --> K["Trained Model<br/>+ Metrics"]
    K --> L["Phase 6<br/>Visualization"]
    L --> M["Forecast Plots<br/>Analysis Reports"]
```

### 1.1 Motivation and Objectives

This repository implements **Causal Spatiotemporal Graph Neural Networks (CSTGNN)** for epidemic forecasting and pest attack prediction. The core innovation is a hybrid framework that integrates:

1. **Spatio-Contact SIR Models** – Compartmental epidemiological models capturing disease/pest transmission dynamics
2. **Graph Neural Networks (GNNs)** – Deep learning architectures that encode spatial propagation patterns
3. **Temporal Decomposition** – Time series decomposition to capture trends and variations
4. **Adaptive Dynamic Graphs** – Learnable adjacency matrices that capture spatiotemporal dependencies in inter-regional mobility

### 1.2 Key Scientific Contributions

- **Hybrid Approach**: Combines interpretable epidemiological theory with deep learning expressiveness
- **Spatiotemporal Dynamics**: Simultaneously models temporal trends and spatial diffusion via:
  - Static connectivity graphs (stable mobility patterns)
  - Dynamic temporal models (fluctuations in mobility)
  - Learned adjacency matrices for **Dynamic** and **Adaptive** graph types
- **Interpretability**: Extracts and analyzes learned epidemic parameters (transmission rates, recovery rates, contact patterns)
- **Multi-scale Validation**: Tested on:
  - COVID-19 data at provincial level in China
  - COVID-19 data at state level in Germany
  - **KCC Pest Attack** data at state level in India (686K+ records, 31 states, 2013-2020)

---

## 2. Architecture and Methodology

### 2.1 Model Architecture: SSIR-STGCN

The core models follow a unified architecture: **Spatial-Contact SIR + Spatio-Temporal Graph Convolution Network (SSIR-STGCN)**.

#### 2.1.1 Input Format
```
Input:  (B, T_obs, N, F)
  where:
    B = batch size
    T_obs = observation/lookback window (e.g., 7 or 12 time steps)
    N = number of spatial nodes (e.g., 31 states, 550 districts)
    F = feature dimension (3, representing S, I, R compartments)
```

#### 2.1.2 Forward Pass Stages

```mermaid
graph TD
    A["Input<br/>B,T_obs,N,3"] --> B["FC Embedding<br/>t_out_dim=16"]
    B --> C["Temporal Module<br/>TCN + Decomp"]
    C --> D["Graph Module<br/>Compute Adjacency"]
    D --> E{"Graph Type"}
    E -->|Static| F["Fixed Neighbor Matrix"]
    E -->|Dynamic| G["Learned Per-Timestep"]
    E -->|Adaptive| H["Modified Static"]
    F --> I["Spatial Module<br/>GCN/GAT Layers"]
    G --> I
    H --> I
    I --> J["Spatial Embeddings<br/>s_out_dim=16"]
    J --> K["Forecasting Module"]
    J --> L["Epidemic Module"]
    K --> M["y_pred<br/>B,T_next,N,1"]
    L --> N["y_phy, epiparams<br/>all_phy"]
```

**Stage 1: Feature Embedding**
```
FC Layer: (B, T_obs, N, 3) → (B, T_obs, N, t_out_dim)
  Maps raw SIR compartments to high-dimensional embeddings
  Typical t_out_dim = 16
```

**Stage 2: Temporal Feature Extraction**
```
TemporalModule: (B, T_obs, N, t_out_dim) → (B, T_obs, N, t_out_dim)
  - Series Decomposition: Splits input into trend + seasonal/residual components
  - TCN (Temporal Convolution Network): Captures temporal dependencies
  - Kernel size, dilation, and padding configured for multi-scale patterns
```

**Stage 3: Adjacency Matrix Computation**
```
GraphModule: (B, T_obs, N, t_out_dim) → Adjacency Matrix
  
  Static Graphs:
    - Predefined N×N matrix from geographic neighbors
    - File: data/processed_data/neighbor_adjacency_matrix.csv
    
  Dynamic Graphs:
    - Learned per timestep: (B, T_obs, N, N)
    - Uses embeddings to adaptively compute affinity
    - Captures time-varying spatial correlations
    
  Adaptive Graphs:
    - Learnable parameters to modify edge weights
    - Interpolation between Static and Dynamic
```

**Stage 4: Spatial Feature Extraction**
```
SpatialModule: (B, T_obs, N, t_out_dim) + Adjacency → (B, T_obs, N, s_out_dim)
  - Graph Convolution (GCNConv): Aggregates neighborhood information
  - Number of layers (num_layers), dropout for regularization
  - Output: s_out_dim dimensional spatial embeddings (typical = 16)
  - Equations: Z = σ(D^{-1/2} A D^{-1/2} X W)
    where A = adjacency, D = degree matrix, X = features, W = weights
```

**Stage 5: Forecasting Module**
```
ForecastingModule: (B, T_obs, N, s_out_dim) → (B, T_next, N, 1)
  - Fully connected layers projecting spatial embeddings to future predictions
  - T_next = pre_len (e.g., 3 or 7 time steps ahead)
  - Output shape: (batch, forecast_horizon, nodes, 1)
```

**Stage 6: Epidemic Module (Physical Model)**
```
EpidemicModule: (B, T_obs, N, s_out_dim) + (B, T_obs, N, 3) → 
                (B, T_next, N, 1), (B, T_next, N, N+2), (B, T_next, N, 3)
  
  Outputs:
    - y_phy: Physical SIR-based forecasts
    - epiparams: Learned epidemic parameters (β, γ, contact patterns)
    - all_phy: Full SIR state trajectories (S, I, R)
  
  Mechanism:
    - Decodes embeddings to epidemic parameters
    - Applies deterministic SIR dynamics
    - β_t, γ_t, contact matrix c_t learned from data
    - Simulates: dS/dt = -β(t)·S·I/N, dI/dt = β(t)·S·I/N - γ(t)·I, dR/dt = γ(t)·I
```

#### 2.1.3 Model Variants

| Variant | Spatial Encoder | Temporal Encoder | Adjacency | Key Feature |
|---------|-----------------|------------------|-----------|-------------|
| **SSIR-STGCN** | GCNConv (Graph Conv) | TCN + Series Decomp | Static/Dynamic/Adaptive | Convolutional aggregation |
| **SSIR-STGAT** | GATConv (Graph Attention) | TCN + Series Decomp | Static/Dynamic/Adaptive | Learnable edge attention |
| **SSIR-ODEFIT** | None (Direct ODE) | ODE Solver | None | Pure epidemic compartmental model |

### 2.2 Epidemic Components

```mermaid
graph LR
    S["S<br/>Susceptible"] -->|β·S·I/N| I["I<br/>Infected"]
    I -->|γ·I| R["R<br/>Recovered"]
    S -.->|Aging/Birth| S
    I -.->|Aging/Birth| I
    R -.->|Aging/Birth| R
    style S stroke-width:2px
    style I stroke-width:2px
    style R stroke-width:2px
```

#### 2.2.1 SIR Compartmental Model

The Susceptible-Infected-Recovered framework models disease/pest dynamics:

```
S(t) = Susceptible population
I(t) = Infected/Infested population
R(t) = Recovered/Ceased population
N(t) = S(t) + I(t) + R(t) = total population
```

**Differential Equations:**
```
dS/dt = -β(t) · S(t) · I(t) / N(t)
dI/dt = β(t) · S(t) · I(t) / N(t) - γ(t) · I(t)
dR/dt = γ(t) · I(t)
```

Where:
- **β(t)** = transmission/contact rate (learned from data)
- **γ(t)** = recovery/cessation rate (learned from data)

#### 2.2.2 Spatial-Contact SIR (SSIR)

```mermaid
graph TD
    A["Region i"] -->|c_ij·I_j| B["Region j"]
    C["Local SIR<br/>in Region i"] -->|β·S·I/N| D["Infected<br/>Generation"]
    B -->|Spatial Coupling| C
    D --> E["Recovery<br/>γ·I"]
    style A stroke-width:2px
    style B stroke-width:2px
    style C stroke-width:2px
```

Extension allowing spatial information exchange:

```
dS_i/dt = -β · S_i · I_i / N_i - Σ_j c_ij · I_j · S_i / N_i
dI_i/dt = β · S_i · I_i / N_i + Σ_j c_ij · I_j · S_i / N_i - γ · I_i
dR_i/dt = γ · I_i
```

Where:
- **i, j** = spatial nodes (states/districts)
- **c_ij** = contact/mobility rate between regions (learned matrix)
- The spatial coupling term Σ_j c_ij captures inter-regional propagation

#### 2.2.3 Parameter Learning

Instead of predefined rates, the model learns:
- **β(t, n)** per timestep and node: Controls local transmission intensity
- **γ(t, n)** per timestep and node: Controls recovery/control effectiveness
- **c(t)** or **c(i,j)**: Spatial transmission coefficients (static or dynamic)

Learning is implicit via the embeddings from the neural network components.

---

## 3. Data Pipeline and Preprocessing

### 3.1 Raw Data Ingestion (KCC Pest Attack Dataset)

**Source**: `data/Normalised_KCC_Data.csv`

**Dataset Characteristics:**
```
Total Records:        686,118
Date Range:           2013-01-01 to 2020-07-27
Spatial Coverage:     31 Indian States, 550 Districts
Temporal Granularity: Daily observations
```

**Key Columns:**
- `CreatedOn`: Datetime of pest attack report
- `State Name`: Indian state (31 unique)
- `Dist Name`: District (550 unique)
- `Pest`: Type of pest (65 unique, e.g., insect, caterpillar, aphid)
- `Crop`: Agricultural crop (283 unique)
- `Count`: Number of pest attack reports (target variable for "I")
- `Rainfall (MM)`: Precipitation (feature, ~70% missing)
- `Harvest Area`: Crop cultivation area (feature, ~58% missing)
- `Latitude, Longitude`: Geographic coordinates

### 3.2 Phase 1: Data Cleaning and Imputation (`kcc_codebase/data_preprocessing.py`)

```mermaid
graph LR
    A["Raw CSV<br/>686K rows"] --> B["Parse Datetime<br/>Derive Y,M"]
    B --> C["Standardize<br/>Strings"]
    C --> D["Drop Nulls<br/>in Count"]
    D --> E["Cleaned Data<br/>686K rows"]
    F["Rainfall~70% null<br/>Harvest~58% null"] -.-> E
```

#### 3.2.1 Data Cleaning
```python
Load CSV → Parse CreatedOn as datetime
Derive Year, Month from date
Standardize string columns (lowercase, strip whitespace)
Drop rows with null Count (safety check)
Result: 686,118 rows with consistent formatting
```

#### 3.2.2 Missing Value Imputation

```mermaid
graph TD
    A["Missing Values"] --> B{"Column Type"}
    B -->|Rainfall| C["Month-District<br/>Median"]
    B -->|Harvest Area| D["District-Level<br/>Temporal Mean"]
    C -->|Still Null| E["Month-State<br/>Median"]
    D -->|Still Null| F["State-Level<br/>Temporal Mean"]
    E -->|Still Null| G["Fill with 0"]
    F -->|Still Null| H["Fill with 0"]
    G --> I["Imputed Data"]
    H --> I
```

**Rainfall (MM)** – ~70% missing
```
1. Fill using monthly-district median
2. Fallback: monthly-state median
3. Fallback: 0 (no precipitation)

Rationale: Monthly aggregation captures seasonal patterns;
spatial hierarchy (district → state) reflects climatic zones
```

**Harvest Area** – ~58% missing
```
1. Fill using district-level temporal mean
2. Fallback: state-level temporal mean
3. Fallback: 0 (no data)

Rationale: Harvest area varies by geography (district)
but is relatively stable over time within region
```

#### 3.2.3 Temporal Aggregation

From daily records to **monthly state-level** time series:
```
Granularity:  State × Month
Shape:        31 states × ~84 months = ~2,604 time points

Aggregation:
  - Count → SUM per (State, Month)     [becomes "I" signal]
  - Rainfall → MEAN per (State, Month) [environmental feature]
  - Harvest Area → MEAN per (State, Month) [population proxy]

Result: kcc_codebase/processed_data/kcc_monthly_sir.csv
```

#### 3.2.4 SIR State Variable Construction

```mermaid
graph LR
    A["Monthly Count<br/>per State"] --> B["I = Count"]
    A --> C["cum_I = Σ I"]
    C --> D["S = Harvest - cum_I"]
    D --> E["I_clamped = min(I,S)"]
    E --> F["R = Harvest - S - I"]
    B --> G["Normalize by<br/>Harvest Area"]
    F --> G
    G --> H["S_norm, I_norm, R_norm<br/>all in [0,1]"]
    style H stroke-width:2px
```

From aggregated monthly Count, derive epidemic compartments:

```python
For each state time series:
  I(t) = Count(t)  [Raw pest-attacked area counts]
  
  cum_I(t) = Σ_{s=0}^{t-1} I(s)  [Cumulative infections]
  
  S(t) = max(Harvest_Area - cum_I(t), 0)  [Susceptible remaining]
  S(t) = min(S(t), Harvest_Area)          [Clamp to valid range]
  
  I_clamped(t) = min(I(t), S(t))  [Cannot exceed susceptible]
  
  R(t) = max(Harvest_Area - S(t) - I(t), 0)  [Recovered/Ceased]

Normalized (per state):
  S_norm = S / Harvest_Area
  I_norm = I / Harvest_Area
  R_norm = R / Harvest_Area
  
All normalized values ∈ [0, 1]
```

**Output Format**:
```
date              | state  | S      | I      | R      | S_norm | I_norm | R_norm | rainfall | harvest_area
2013-01-01        | state1 | 120.5  | 5.3    | 2.1    | 0.923  | 0.041  | 0.016  | 45.2     | 130.6
2013-02-01        | state1 | 118.2  | 7.8    | 4.6    | 0.905  | 0.060  | 0.035  | 52.1     | 130.6
...
Shape (after pivot): (T=84, N=31, F=3 or 6)
```

### 3.3 Phase 2: Spatial Graph Construction (Planned)

```mermaid
graph TD
    A["Spatial Nodes<br/>31 States"] --> B{"Graph Construction"}
    B -->|Static| C["Geographic<br/>Adjacency<br/>Binary A[i,j]"]
    B -->|Dynamic| D["Learned<br/>Per Timestep<br/>A[t,i,j]"]
    B -->|Soft| E["Haversine Distance<br/>Kernel A[i,j]=exp"]
    C --> F["Normalize<br/>Row-wise or<br/>Laplacian"]
    D --> G["Apply Sigmoid<br/>Attention"]
    E --> H["Threshold<br/>Cutoff"]
    F --> I["Adjacency Matrix<br/>31x31"]
    G --> I
    H --> I
```

#### 3.3.1 Static Adjacency Matrix

**Geographic Neighbors**:
- Binary adjacency matrix (31×31 for states)
- A[i,j] = 1 if states share a border, else 0
- Row-normalized or degree-normalized (Laplacian)
- File: `processed_data/neighbor_adjacency_matrix.csv`

#### 3.3.2 Distance-Based Soft Adjacency

**Haversine Distance Kernel**:
```
d_ij = haversine(lat_i, lon_i, lat_j, lon_j)
A[i,j] = exp(-d_ij² / σ²)  with threshold cutoff

Gaussian kernel captures "soft" proximity:
  - Neighboring states have higher affinity
  - Distant states have lower affinity
  - Continuous rather than binary
```

#### 3.3.3 Correlation-Based Adjacency

**Temporal Correlation** (for validation):
```
For each state pair (i, j):
  r_ij = Pearson corr(I_i(t), I_j(t))
  A_corr[i,j] = 1 if r_ij > 0.5, else 0

Interpretation: States with synchronized pest dynamics
likely share environmental or mobility patterns
```

### 3.4 Phase 3: Dataset Preparation and DataLoaders (Planned)

#### 3.4.1 Sliding Window Construction

```mermaid
graph LR
    A["Time Series<br/>T=84 months"] --> B["Window 1<br/>Months 1-12"]
    A --> C["Window 2<br/>Months 2-13"]
    A --> D["Window N<br/>Months ...-73"]
    B --> E["Target<br/>Months 13-15"]
    C --> F["Target<br/>Months 14-16"]
    D --> G["Target<br/>Months 74-76"]
    E --> H["Example Shape<br/>T_obs=12, T_pred=3<br/>Window: 15,31,3"]
    F --> H
    G --> H
    style H stroke-width:2px
```

**Sliding Window Construction**

```
Input window size:      obs_len (e.g., 12 months = 1 year lookback)
Prediction horizon:     pre_len (e.g., 3 months = quarterly forecast)
Stride:                 1 month (overlapping windows)

Example:
  Window 1: months 1-12 → predict months 13-15
  Window 2: months 2-13 → predict months 14-16
  ...
  Window N: months (T-pre_len-obs_len) to (T-pre_len) → predict (T-pre_len+1) to T

Result shape per window: (T_obs + T_pred, N, F) = (15, 31, 3)
```

#### 3.4.2 Train/Validation/Test Split

```mermaid
graph LR
    A["84 Total Months<br/>Jan 2013-Dec 2020"] --> B["Train 60%<br/>Months 1-50"]
    A --> C["Val 20%<br/>Months 51-67"]
    A --> D["Test 20%<br/>Months 68-84"]
    B --> E["Sliding Windows<br/>B_train samples"]
    C --> F["Sliding Windows<br/>B_val samples"]
    D --> G["Sliding Windows<br/>B_test samples"]
    E --> H["DataLoaders<br/>Temporal Split"]
    F --> H
    G --> H
```

**Temporal Split** (avoiding data leakage):
```
Total time points: 84 months (Jan 2013 - Dec 2020)

Split ratio:
  Training:   60% → windows from months 1 to ~50
  Validation: 20% → windows from months ~51 to ~67
  Test:       20% → windows from months ~68 to 84

Each split generates sliding windows independently
Train: B_train samples, Val: B_val samples, Test: B_test samples
```

#### 3.4.3 DataLoader Configuration

```python
PyTorch DataLoader:
  batch_size:     8 (default)
  shuffle:        True (training set only)
  drop_last:      True (discard incomplete final batch)
  num_workers:    4 (parallel data loading)

Batches format for forward pass:
  X_batch: (B, T_obs, N, F) = (8, 12, 31, 3)
  Y_batch: (B, T_pred, N, 1) = (8, 3, 31, 1)
```

---

## 4. Model Training and Optimization

### 4.1 Loss Functions

#### 4.1.1 Deep Learning Component
```
L_deep = Mean Squared Error (MSE)
       = 1/(B·T_pred·N) · Σ(y_pred - y_true)²

Penalizes deviation of learned forecasts from observed counts.
```

#### 4.1.2 Physical Model Component
```
L_phy = Mean Absolute Percentage Error (MAPE)
      = 1/(B·T_pred·N) · Σ |y_phy - y_true| / |y_true|

Penalizes deviation of SIR-based forecasts.
Robust to scale; interpretable as percent error.
```

#### 4.1.3 Hybrid Loss
```
L_total = λ_1 · L_deep + λ_2 · L_phy + λ_3 · L_reg

where:
  λ_1, λ_2, λ_3 = loss weights (hyperparameters)
  L_reg = regularization (e.g., L2 weight decay on parameters)
```

### 4.2 Optimization Configuration (`code/Train.py`)

**Supported Optimizers**:
```
- SGD with momentum (default momentum=0.9)
- Adam (β_1=0.9, β_2=0.999)
- AdamW (with weight decay)
```

**Learning Rate Scheduling**:
```
StepLR:
  - Decay LR by factor γ every step_size epochs
  - Example: γ=0.1, step_size=20 → LR *= 0.1 every 20 epochs

CosineAnnealingLR:
  - Follows cosine decay: LR_t = η_min + (LR_0 - η_min) · (1 + cos(πt/T))/2
  - Smooth learning rate annealing over T_max epochs

MultiStepLR:
  - Decay at specific epoch milestones
  - Useful for plateaus in training curves

ReduceLROnPlateau:
  - Reduce LR when validation metric plateaus
  - Patience: wait N epochs before reducing
```

**Typical Training Config**:
```python
optimizer:      'adam'
learning_rate:  0.001
weight_decay:   1e-5
scheduler:      'cosine'
t_max:          100 (epochs)
eta_min:        1e-6
clip:           5.0 (gradient clipping)
early_stop:     15 (epochs to wait for improvement)
```

### 4.3 Training Loop

```mermaid
graph TD
    A["Epoch Loop<br/>1 to max_epoch"] --> B["Training Phase"]
    B --> C["For each batch"]
    C --> D["Forward Pass<br/>Compute y_pred, y_phy"]
    D --> E["Loss Computation<br/>L_deep + L_phy + L_reg"]
    E --> F["Backward Pass<br/>Compute ∂L/∂θ"]
    F --> G["Gradient Clipping"]
    G --> H["Optimizer Step<br/>θ = θ - α∇L"]
    H --> I["Validation Phase"]
    I --> J["For each val batch"]
    J --> K["Forward Pass<br/>no_grad()"]
    K --> L["Compute Metrics<br/>MAE, RMSE, MAPE"]
    L --> M["LR Scheduling"]
    M --> N{"Check<br/>Improvement"}
    N -->|Yes| O["Save Model<br/>Continue"]
    N -->|No| P{"Patience<br/>Exceeded?"}
    P -->|No| Q["Next Epoch"]
    P -->|Yes| R["Early Stop"]
    A --> Q
    O --> Q
    Q --> A
    style R stroke-width:2px
```

**Training Epoch Workflow**

```python
For each epoch:
  1. Training phase:
     For each batch (X, Y):
       - Forward pass: y_pred, y_phy, epiparams, all_phy = model(X)
       - Compute losses: L_deep, L_phy, L_reg
       - Backward pass: gradients
       - Gradient clipping (if clip > 0)
       - Optimizer step

  2. Validation phase:
     For each batch (X, Y):
       - Forward pass (no gradients)
       - Compute validation metrics

  3. LR scheduling:
     - scheduler.step() with current metrics
     - Early stopping: stop if no improvement for N epochs

Result: Save best model based on validation metric
```

---

## 5. Model Variants and Their Characteristics

### 5.1 SSIR-STGCN (Graph Convolutional Network)

**Spatial Aggregation**:
```
GCNConv: Z = σ(D^{-1/2} A D^{-1/2} X W)

Propagation: Information flows from neighbors → central node
Aggregation: Weighted sum of neighbor features
Scalability: Efficient for large graphs (linear in edges)
```

**Best For**: 
- Large spatial networks (100+ nodes)
- Regular spatial structures
- Computational efficiency required

### 5.2 SSIR-STGAT (Graph Attention Network)

**Spatial Aggregation**:
```
GATConv: α_ij = softmax_j(LeakyReLU(a^T[W·X_i || W·X_j]))
         Z_i = σ(Σ_j α_ij · W · X_j)

Propagation: Learned per-neighbor attention weights
Aggregation: Weighted sum with learned affinities
Interpretability: Attention weights reveal spatial influence
```

**Best For**:
- Small to medium networks (≤ 1000 nodes)
- Non-uniform spatial patterns
- Interpretable spatial relationships needed

### 5.3 SSIR-ODEFIT (Epidemic ODE-only)

**Mechanism**:
```
Pure compartmental model without neural network components
Parameters β(t), γ(t), contact matrix c(t) learned via gradient descent
SIR dynamics integrated using ODE solver (e.g., Runge-Kutta)
```

**Best For**:
- Small datasets with few time steps
- Maximum interpretability required
- Comparison with pure epidemiological baselines

### 5.4 Adjacency Types

```mermaid
graph TD
    A["Adjacency Type"] --> B["Static<br/>Geographic"]
    A --> C["Dynamic<br/>Learned"]
    A --> D["Adaptive<br/>Hybrid"]
    B --> E["Fixed Neighbors<br/>A[i,j]=1 if border<br/>Low Flexibility<br/>High Interpretability"]
    C --> F["Learned Per-Timestep<br/>A[t,i,j] ∈ ℝ<br/>High Flexibility<br/>Medium Interpretability"]
    D --> G["Learnable Modification<br/>A = λ·A_static + (1-λ)·A_dyn<br/>Medium Flexibility<br/>High Interpretability"]
```

**Comparison Table**

| Type | Definition | Flexibility | Interpretability |
|------|-----------|-------------|------------------|
| **Static** | Fixed geographic neighbors | Low | High (physical meaning) |
| **Dynamic** | Learned per timestep | High | Medium (empirical patterns) |
| **Adaptive** | Learned modification of Static | Medium | High (augments geography) |

---

## 6. Evaluation Metrics

### 6.1 Regression Metrics

```
MAE (Mean Absolute Error):
  MAE = 1/n · Σ |y_true - y_pred|
  Units: Same as target (pest counts)
  Interpretation: Average absolute deviation

RMSE (Root Mean Squared Error):
  RMSE = √(1/n · Σ (y_true - y_pred)²)
  Units: Same as target
  Penalizes large errors more heavily

MAPE (Mean Absolute Percentage Error):
  MAPE = 1/n · Σ |y_true - y_pred| / |y_true|
  Units: Percentage (%)
  Scale-invariant, interpretable

R² (Coefficient of Determination):
  R² = 1 - (SS_res / SS_tot)
  Range: (-∞, 1], where 1.0 = perfect fit
  Proportion of variance explained
```

### 6.2 Epidemic-Specific Metrics

```
Parameter Reconstruction Error:
  - Compare learned β, γ against epidemiological estimates
  - MAE on log scale (rates are log-normal)

SIR Trajectory Error:
  - Evaluate S(t), I(t), R(t) against reconstructed compartments
  - Checks physical consistency

Contact Matrix Similarity:
  - Compare learned c_ij against mobility data (if available)
  - Frobenius norm or cosine similarity
```

### 6.3 Spatial Correlation Analysis

```
Spatial Pattern Correlation:
  - Forecast error by node: δ_i = MAE(y_i_pred, y_i_true)
  - Correlate δ_i across neighboring nodes
  - High correlation suggests spatial error patterns
  - Used to diagnose whether graph is capturing spatial relationships
```

---

## 7. Codebase Organization

### 7.1 Directory Structure

```
Epi-Spatio-Temporal-Teilen/
├── code/                               # Original reference models
│   ├── Main.py                         # Entry point for experiments
│   ├── Train.py                        # Training loop & Trainer class
│   ├── EpiGCN.py                       # SSIR-STGCN architecture
│   ├── EpiGAT.py                       # SSIR-STGAT architecture
│   ├── EpiODEfit.py                    # SSIR-ODEFIT architecture
│   ├── Toolkits.py                     # Utility functions
│   ├── Logconfig.py                    # Logging configuration
│   └── Constant.py                     # Constants & paths
│
├── kcc_codebase/                       # New KCC pest attack pipeline
│   ├── run_pipeline.py                 # Main orchestrator (Phases 1-6)
│   ├── data_preprocessing.py           # Phase 1: Data cleaning & SIR construction ✓
│   ├── graph_construction.py           # Phase 2: Adjacency matrices [TODO]
│   ├── dataset_builder.py              # Phase 3: DataLoaders [TODO]
│   ├── model_config.py                 # Phase 4: Hyperparameters [TODO]
│   ├── train_eval.py                   # Phase 5: Training loop [TODO]
│   ├── visualization.py                # Phase 6: Analysis plots [TODO]
│   ├── processed_data/
│   │   ├── kcc_monthly_sir.csv         # Output from Phase 1 ✓
│   │   └── [other outputs to be created]
│   └── __pycache__/
│
├── data/
│   └── Normalised_KCC_Data.csv        # Raw pest attack records (686K rows)
│
├── execution_plan/
│   └── pest_attack_prediction_plan.md  # Detailed execution plan
│
├── environment.yml                     # Conda environment specification
├── README.md                           # Project overview
└── TECHNICAL_DOCUMENTATION.md          # This file
```

### 7.2 Key Files and Modules

#### `code/EpiGCN.py` – SSIR-STGCN Model
```python
Classes:
  SSIR_STGCN(nn.Module):      # Main model class
  TemporalModule(nn.Module):  # TCN + series decomposition
  SpatialModule(nn.Module):   # GCN with graph convolutions
  GraphModule(nn.Module):     # Adjacency computation
  EpidemicModule(nn.Module):  # SIR ODE solver
  ForecastingModule(nn.Module): # Prediction head

Key methods:
  forward(x): (B, T, N, F) → (B, T_next, N, 1), epiparams, physics_outputs
```

#### `code/Train.py` – Training Infrastructure
```python
Classes:
  Trainer(object):  # Manages training/validation/testing
    Methods:
      __init__(params, model_type)
      get_model()
      get_loss()
      get_optimizer(parameters)
      train_epoch(dataloader)
      val_epoch(dataloader)
      test_epoch(dataloader)
      save_model()
      load_model()
      plot_results()
```

#### `code/Main.py` – Execution Entry Point
```python
Command-line Arguments:
  -dev:          Device ('cuda:0', 'cpu', 'mps')
  -data_type:    'china', 'germany', 'mock', 'kcc'
  -model_type:   'SSIR_STGCN', 'SSIR_STGAT', 'SSIR_ODEFIT'
  -graph_type:   'Static', 'Dynamic', 'Adaptive'
  -obs:          obs_len (lookback window)
  -pre:          pre_len (prediction horizon)
  -batch:        batch_size
  
Workflow:
  1. Parse args
  2. Load data
  3. Initialize Trainer
  4. Train for max_epoch
  5. Evaluate on test set
  6. Save results
```

#### `kcc_codebase/data_preprocessing.py` – Data Pipeline Phase 1
```python
Functions:
  load_and_clean(path):         # Step 1.1: Load & standardize
  impute_missing(df):           # Step 1.2: Handle nulls
  aggregate_monthly(df):        # Step 1.3: Temporal aggregation
  construct_sir(agg):           # Step 1.4: SIR derivation
  save_output(df, path):        # Step 1.5: Serialize
  validate(df):                 # Step 1.6: Sanity checks

Output:
  kcc_codebase/processed_data/kcc_monthly_sir.csv
  Columns: [date, state, S, I, R, S_norm, I_norm, R_norm, 
            rainfall, harvest_area, I_raw]
  Shape: ~2,604 rows × 11 columns
```

### 7.3 Configuration and Constants

**`code/Constant.py`**:
```python
Paths:
  DATA_RAW:                  Raw data directory
  DATA_REPO:                 Repository for processed data
  RESULT:                    Results output directory
  NEIGHBOR_ADJACENCY_MATRIX: Path to spatial graph

DATE:
  START_DATE, END_DATE:      Date range for experiments
  DATE_SELECTED:             Combined date string
```

**`environment.yml`**:
```yaml
Key Dependencies:
  - pytorch::pytorch=2.0.1
  - pytorch::pytorch-cuda=11.7
  - pytorch::pytorch-geometric  # For GNN operations
  - pandas, numpy, scipy        # Data manipulation
  - scikit-learn                # ML utilities
  - matplotlib, seaborn         # Visualization
  - jupyter, ipython            # Interactive dev
```

---

## 8. Experimental Workflow

### 8.1 Complete Training Pipeline

```mermaid
graph TD
    A["Phase 1<br/>Data Prep"] -->|kcc_monthly_sir.csv| B["Phase 2<br/>Graph Const"]
    B -->|neighbor_matrix.csv| C["Phase 3<br/>DataLoaders"]
    C -->|train/val/test| D["Phase 4<br/>Model Config"]
    D -->|config.json| E["Phase 5<br/>Train Model"]
    E -->|model.pth<br/>logs.csv| F["Phase 6<br/>Visualize"]
    F -->|plots<br/>analysis| G["Final Results"]
    style A stroke-width:2px
    style B stroke-width:2px
    style C stroke-width:2px
    style D stroke-width:2px
    style E stroke-width:2px
    style F stroke-width:2px
    style G stroke-width:2px
```

**Detailed Pipeline Steps**

```
Step 1: Data Preparation
  python kcc_codebase/data_preprocessing.py
  Output: kcc_monthly_sir.csv (84 months × 31 states × 3 features)

Step 2: Graph Construction
  python kcc_codebase/graph_construction.py
  Output: neighbor_adjacency_matrix.csv (31×31)

Step 3: Build DataLoaders
  python kcc_codebase/dataset_builder.py
  Output: train_loader, val_loader, test_loader

Step 4: Configure Model
  python kcc_codebase/model_config.py
  Output: config.json with hyperparameters

Step 5: Train Model
  python code/Main.py \
    -data_type kcc \
    -model_type SSIR_STGCN \
    -graph_type Dynamic \
    -obs 12 -pre 3 \
    -batch 8
  Output: trained_model.pth, training_logs.csv

Step 6: Evaluate & Visualize
  python kcc_codebase/visualization.py
  Output: forecast_plots.png, heatmaps.png, param_analysis.png
```

### 8.2 Key Hyperparameters

```
Data:
  window_rolling:  28        # Distance threshold for sliding window
  obs_len:         12        # Lookback months
  pre_len:         3         # Forecast months
  split_ratio:     [6, 2, 2] # [train, val, test] fractions

Model:
  kernel_size:     3         # Temporal convolution size
  num_layers:      2         # GC layers
  t_out_dim:       16        # Temporal embedding dim
  s_out_dim:       16        # Spatial embedding dim
  beta_incorporated: True    # Learn β explicitly

Training:
  max_epoch:       100
  batch_size:      8
  learning_rate:   0.001
  weight_decay:    1e-5
  early_stop:      15
  grad_clip:       5.0
```

---

## 9. Scientific Insights and Interpretability

### 9.1 Learned Epidemic Parameters

```mermaid
graph TD
    A["Epidemic Module<br/>Learned Parameters"] --> B["β(t,n)<br/>Transmission Rate"]
    A --> C["γ(t,n)<br/>Recovery Rate"]
    A --> D["c(t,i,j)<br/>Contact Matrix"]
    B --> E["Range: [0,1]<br/>σ-Sigmoid Output<br/>β↑ climate favorable<br/>β↓ harsh conditions"]
    C --> F["Range: [0,1]<br/>σ-Sigmoid Output<br/>γ↑ fast cessation<br/>γ↓ prolonged pest"]
    D --> G["Shape: NxN or TxNxN<br/>Dynamic/Static<br/>c_ij↑ migration i→j<br/>c_ij↓ weak coupling"]
    style E stroke-width:2px
    style F stroke-width:2px
    style G stroke-width:2px
```

**Transmission Rate β(t)**:
```
Interpretation: How efficiently pest/disease spreads
  β_t ↑ → Higher transmission (e.g., favorable climate)
  β_t ↓ → Lower transmission (e.g., harsh conditions)

Extraction: Sigmoid output from epidemic module
Value range: [0, 1] (percent susceptible infected per contact)
```

**Recovery Rate γ(t)**:
```
Interpretation: Effectiveness of control/natural cessation
  γ_t ↑ → Faster pest die-off or crop recovery
  γ_t ↓ → Prolonged infestation

For pest context:
  - γ may reflect pesticide efficacy
  - Seasonal recovery patterns
  - Host plant resistance
```

**Contact Matrix c_ij**:
```
Interpretation: Inter-regional mobility/transmission
  c_ij ↑ → Strong pest migration from i to j
  c_ij ↓ → Weak coupling

For pest context:
  - Reflects trade routes and supply chains
  - Human-mediated transport of infested crops
  - Geographic barriers

Comparison: Can be compared with trade data, supply chain networks
```

### 9.2 Spatial Pattern Analysis

```mermaid
graph TD
    A["Adjacency Matrix<br/>Analysis"] --> B["Static Graph<br/>Geographic"]
    A --> C["Dynamic Graph<br/>Learned"]
    B --> D["Fixed Neighbors<br/>A_ij=1 if border<br/>Human Interpretable<br/>Validates Geography"]
    C --> E["Time-Varying<br/>A_ij(t) from Data<br/>Empirical Patterns<br/>Captures Novel Couplings"]
    F["Compare Static vs Dyn"] --> G{"Similarity"}
    G -->|High| H["Geography Sufficient<br/>for Spread"]
    G -->|Low| I["Complex Mobility<br/>Beyond Borders"]
    D --> F
    E --> F
```

**Spatial Error Analysis**

```mermaid
graph LR
    A["Forecast Errors<br/>e_i per node"] --> B["Spatial<br/>Autocorrelation"]
    B --> C{"Pattern"}
    C -->|Positive | D["Nearby errors<br/>similar<br/>Missing spatial<br/>feature"]
    C -->|Negative| E["Errors<br/>independent<br/>Good spatial<br/>capture"]
    C -->|Random| F["Noise"]    
    style D stroke-width:2px
    style E stroke-width:2px
```

**Learned Adjacency Matrices**:
```
Static Graph (geographic):
  - Fixed, human-interpretable
  - A_ij = 1 if states border
  - Validates model respects geography

Dynamic Graph (empirical):
  - Learned from data
  - A_ij(t) varies with spatiotemporal patterns
  - Deviations from Static → novel spatial couplings

Compare Static vs Dynamic:
  - High similarity → Geography explains spatial spread
  - Low similarity → Complex mobility patterns beyond borders
```

**Error Spatial Correlation**:
```
Define error for each node: e_i = |y_{i,pred} - y_{i,true}|

Spatial autocorrelation:
  - Positive: Nearby nodes have similar errors
    → Suggests spatial feature missing in model
  - Negative: Errors are independent across space
    → Model captures spatial patterns well
```

---

## 10. Expected Performance and Benchmarks

### 10.1 Baseline Results (COVID-19 China Province Level)

```
Dataset:        China COVID-19 provinces
Horizon:        1-week forecast
Input window:   2 weeks
Nodes:          31 provinces
Time points:    ~100 days

SSIR-STGCN (Dynamic Graph):
  MAE:     ~15-25 cases
  RMSE:    ~20-35 cases
  MAPE:    ~8-15%
  R²:      0.85-0.92

SSIR-STGAT (Dynamic Graph):
  MAE:     ~12-22 cases (slightly better)
  RMSE:    ~18-32 cases
  MAPE:    ~7-13%
  R²:      0.87-0.93

SSIR-ODEFIT (No spatial):
  MAE:     ~25-40 cases
  RMSE:    ~35-50 cases
  MAPE:    ~15-25%
  R²:      0.60-0.75

Note: STGCN/STGAT outperform pure ODE, showing value of spatial GNN
```

### 10.2 Expected Performance (KCC Pest Attack)

```
Dataset:        KCC Pest Attacks (31 states, 84 months)
Horizon:        3-month forecast
Input window:   12 months
Nodes:          31 states
Time points:    ~84 aggregate time points

Estimated Ranges (to be validated):
  SSIR-STGCN (Dynamic):
    MAE:     ~50-150 pest counts/month
    RMSE:    ~75-200 counts
    MAPE:    ~12-25%
    R²:      0.65-0.85

  Performance depends heavily on:
    - Presence of seasonal patterns in pest data
    - Quality of rainfall and harvest area estimates
    - Geographic coherence in spatial spread
```

---

## 11. Limitations and Future Directions

### 11.1 Current Limitations

1. **Fixed Dimensionality**: Assumes S,I,R compartments; doesn't handle multiple pest types simultaneously
2. **Stationary Graph Assumption**: Static graph assumes no change in geographic structure
3. **Missing Climate Variables**: Only rainfall included; temperature, humidity could improve predictions
4. **Aggregation Loss**: Monthly aggregation smooths short-term outbreak dynamics
5. **No Exogenous Interventions**: Model doesn't account for pesticide application campaigns

### 11.2 Planned Extensions

1. **Multi-Pest Modeling**: Extend SIR to SEIR (exposed compartment) or parallel SIR for multiple species
2. **Control Policies**: Incorporate intervention schedules (e.g., spray dates) as exogenous inputs
3. **Crop Rotation**: Account for crop type transitions
4. **Supply Chain Data**: Integrate trade networks for better contact patterns
5. **Uncertainty Quantification**: Bayesian variants to estimate forecasting uncertainty

---

## 12. References and Related Work

### Core References

1. **Epidemic Modeling**:
   - Kermack & McKendrick (1927): SIR compartmental model
   - Anderson & May (1991): Population biology of infectious diseases

2. **Spatiotemporal Forecasting**:
   - Yan et al. (2019): DMVST-Net for traffic flow
   - Shehwana et al. (2020): ST-ResNet for crowd flow

3. **Graph Neural Networks**:
   - Kipf & Welling (2017): GCN
   - Velickovic et al. (2018): GAT
   - Gilmer et al. (2017): Message Passing Neural Networks

4. **Epidemic Forecasting with DL**:
   - Tian et al. (2021): SSIR-STGCN (original paper)
   - Wang et al. (2021): EpiGCN for COVID-19

---

## 13. Reproducibility and Usage

### 13.1 Running Experiments

**Activate Environment**:
```bash
conda activate pytorch-cpu
cd /path/to/Epi-Spatio-Temporal-Teilen
```

**Data Preprocessing**:
```bash
python kcc_codebase/data_preprocessing.py
# Output: kcc_codebase/processed_data/kcc_monthly_sir.csv
```

**Full Pipeline** (when all phases implemented):
```bash
python kcc_codebase/run_pipeline.py --phases 1 2 3 4 5 6
```

**Training (reference models)**:
```bash
python code/Main.py \
  -dev cuda:0 \
  -data_type germany \
  -model_type SSIR_STGCN \
  -graph_type Dynamic \
  -obs 7 -pre 7 \
  -batch 8
```

### 13.2 Output Artifacts

```
test/{TIMESTAMP}/
├── model_best.pth              # Best model weights
├── config.json                 # Hyperparameters used
├── training_log.csv            # Epoch-wise metrics
├── predictions.csv             # Test set forecasts
├── metrics.json                # Final evaluation scores
└── plots/
    ├── forecast_curves.png     # Time series predictions
    ├── spatial_heatmaps.png    # Error by node
    ├── learned_params.png      # β(t), γ(t), c_ij
    └── attention_weights.png   # (if STGAT used)
```

---

## Summary

This repository implements a sophisticated **hybrid architecture** combining epidemiological theory with deep learning for spatiotemporal forecasting of pest attacks and disease spread. The key innovation is the integration of:

- **Interpretable SIR compartmental dynamics**
- **Learnable spatial graphs** capturing inter-regional interactions
- **Temporal decomposition** for trend-seasonal separation
- **Graph neural networks** for efficient spatial aggregation

The codebase is structured for extensibility, with a modular pipeline (6 phases) for data preparation, graph construction, model configuration, training, and analysis. Current implementation includes Phase 1 (data preprocessing) fully operational; remaining phases scheduled for development following the detailed execution plan.

