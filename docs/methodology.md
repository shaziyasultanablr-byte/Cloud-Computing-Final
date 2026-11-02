# Deep Learning & Baseline Methodology (`VibeGuard Cloud`)

## 1. Research Question
**Does a compact 1D Convolutional Neural Network trained exclusively on `0 HP` and `1 HP` motor load vibration recordings maintain accurate four-class bearing fault classification when evaluated on previously unseen `3 HP` operating loads?**

## 2. Signal Preprocessing & Leakage Prevention
1. **Channel Selection:** Only the drive-end accelerometer signal (`*_DE_time`) is extracted from each CWRU MATLAB `.mat` recording.
2. **Deterministic Non-Overlapping Windowing:** Each 1D recording $x \in \mathbb{R}^T$ is segmented into non-overlapping windows of length $L = 2{,}048$ samples (`stride = 2,048`), capped at `MAX_WINDOWS_PER_FILE = 200` windows (`[0, 409600)` samples maximum) for CPU efficiency and balanced evaluation.
3. **Per-Window Standardization & Outlier Clamping:** Each window $w \in \mathbb{R}^{2048}$ is independently standardized to zero mean and unit variance and clamped to $[-10, 10]$:
   $$\tilde{w}_i = \operatorname{clip}\!\left(\frac{w_i - \mu_w}{\sigma_w + 10^{-8}},\, -10.0,\, 10.0\right)$$
   Because normalization uses only statistics $(\mu_w, \sigma_w)$ internal to window $w$, zero information leaks across windows or splits during training, validation, test evaluation, or live user file uploads.
4. **Recording- and Load-Disjoint Split Protocol:**
   - **Train (`0 HP` & `1 HP`):** 8 files (`673` windows: `319` normal, `118` inner, `118` ball, `118` outer)
   - **Validation (`2 HP`):** 4 files (`377` windows: `200` normal, `59` inner, `59` ball, `59` outer)
   - **Held-Out Final Test (`3 HP`):** 4 files (`378` windows: `200` normal, `60` inner, `59` ball, `59` outer)

## 3. 1D Convolutional Neural Network (`TinyFaultCNN`)
Implemented in `model.py` (`15,572` trainable parameters):
- **Block 1:** `Conv1d(1 -> 16, kernel_size=15, padding=7, bias=False)` $\to$ `BatchNorm1d(16)` $\to$ `ReLU` $\to$ `MaxPool1d(4)` (output length: `512`)
- **Block 2:** `Conv1d(16 -> 32, kernel_size=9, padding=4, bias=False)` $\to$ `BatchNorm1d(32)` $\to$ `ReLU` $\to$ `MaxPool1d(4)` (output length: `128`)
- **Block 3:** `Conv1d(32 -> 64, kernel_size=5, padding=2, bias=False)` $\to$ `BatchNorm1d(64)` $\to$ `ReLU` $\to$ `AdaptiveAvgPool1d(1)` (output length: `1`)
- **Classifier Head:** `Flatten()` $\to$ `Dropout(p=0.2)` $\to$ `Linear(64 -> 4)`

### Training & Model Selection Protocol
- **Seed:** `42` across Python `random`, `numpy`, and `torch` CPU generators.
- **Loss & Optimizer:** Multi-class Cross-Entropy Loss (`nn.CrossEntropyLoss`), Adam optimizer (`lr = 0.001`), batch size `64`, `14` epochs on CPU.
- **Checkpoint Selection:** After each epoch, macro-F1 is evaluated on the `2 HP` validation split. The checkpoint achieving the highest `2 HP` validation macro-F1 (tie-broken by lowest `2 HP` validation cross-entropy loss) is saved to `outputs/model.pt`. The `3 HP` test split is never touched during training or selection.

## 4. Classical Machine Learning Baseline (`RandomForestClassifier`)
To verify whether end-to-end 1D temporal convolution offers advantages over classical feature engineering under motor load shift, a `RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)` is trained on the exact same `0 & 1 HP` windows using `18` standardized hand-crafted features per window (`extract_handcrafted_features` in `data_utils.py`):
- **12 Time-Domain Features:** RMS, Peak, Peak-to-Peak, Variance, Mean Absolute Value, Crest Factor, Shape Factor, Impulse Factor, Clearance Factor, Kurtosis, Skewness, Zero-Crossing Rate.
- **6 Frequency-Domain Features (via 2,048-point real FFT):** Spectral Centroid, Spectral Spread, Dominant Frequency, and 3 independent sub-band power ratios (`0–1.5 kHz`, `1.5–3.0 kHz`, `3.0–4.5 kHz`, `4.5–6.0 kHz`).

## 5. Recording-Level Aggregation & Manual-Review Heuristic
For a multi-window recording with $N$ windows, recording-level class scores are computed as the mean of window-level softmax outputs:
$$\bar{s}_c = \frac{1}{N}\sum_{k=1}^{N} p_{k,c}, \quad \hat{y}_{\text{file}} = \arg\max_{c} \bar{s}_c$$
A pre-specified decision-support heuristic (`evaluate_manual_review_flag` in `model.py`) triggers `manual_review_required = True` whenever either condition holds:
1. **Low Uncalibrated Confidence:** $\max_c \bar{s}_c < 0.70$
2. **Low Segment Agreement:** Fraction of windows predicting $\hat{y}_{\text{file}}$ is $< 0.80$ ($80\%$)
