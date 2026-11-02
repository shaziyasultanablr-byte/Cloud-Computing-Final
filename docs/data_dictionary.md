# Data Dictionary & Dataset Provenance (`VibeGuard Cloud`)

## 1. Upstream Data Source
- **Institution:** Case Western Reserve University (CWRU) Bearing Data Center
- **Official Portal:** https://engineering.case.edu/bearingdatacenter/download-data-file
- **Reference Benchmark Study:** Smith, W. A., & Randall, R. B. (2015). *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study.* Mechanical Systems and Signal Processing, 64–65, 100–131.
- **Sensor Channel Used:** Drive-end accelerometer (`*_DE_time`) mounted at the 12 o'clock position on the motor housing supporting an SKF 6205-2RS JEM deep-groove ball bearing.
- **Nominal Sampling Frequency:** `12,000 Hz` (`12 kHz`).
  - *Provenance Note:* CWRU's *Apparatus & Procedures* documentation states that digital data were collected at 12,000 samples/second (and additionally at 48,000 samples/second for a separate drive-end fault series). In benchmark literature (Smith & Randall, 2015), it is noted that CWRU's Normal Baseline table (`97.mat`–`100.mat`) does not independently specify whether those longer baseline files (~244k–485k samples) were sampled at 12 kHz over ~20–40 seconds or 48 kHz over ~5–10 seconds. In VibeGuard Cloud, all 16 recordings are processed using fixed 2,048-sample windows with per-window z-score normalization, and capped at 200 windows per recording (`[0, 409600)` samples max). Any time/frequency axis displayed in the UI explicitly carries a `12 kHz Nominal` notice.

## 2. Critical Data Hygiene Discovery (`99.mat` Workspace Contamination)
During programmatic inspection of the 16 official CWRU `.mat` files, we discovered that `99.mat` (Normal Baseline, 2 HP Validation) contains **two** drive-end time-series variables inside its MATLAB workspace:
- `X098_DE_time` (`483,903` samples — residual variable from `98.mat`, which belongs to the **1 HP Training** split)
- `X099_DE_time` (`485,063` samples — the genuine **2 HP Validation** recording)

A naive loader that selects the first alphabetical `*_DE_time` key (`sorted(keys)[0]`) silently extracts `X098_DE_time` from `99.mat`, causing **100% data leakage** between the 1 HP training normal file (`98.mat`) and the 2 HP validation normal file (`99.mat`).
- **Remediation in `data_utils.py` & `scripts/validate_dataset.py`:** `extract_de_signal_from_mat()` matches the file ID against `X{file_id:03d}_DE_time` (and defaults to `de_keys[-1]` for anonymous byte streams), and `scripts/validate_dataset.py` enforces an explicit assertion `meta["channel_key"] == f"X{int(row['file_id']):03d}_DE_time"`.

## 3. Complete 16-Recording Manifest & Verified Checksums

| File ID | Path | Class (`label`) | Class Index | Fault Diameter (in) | Load (HP) | Approx RPM | Recorded RPM in MAT | Split | MATLAB Channel Key | Total Samples | Extracted Windows (`2048` step) | Bytes | SHA-256 Digest |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| 97 | `normal/97.mat` | `normal` | 0 | 0.000 | 0 | 1797 | 1797 | `train` | `X097_DE_time` | 243,938 | 119 | 3,903,344 | `16bf48babcf1ab6d947eb39ff3cd5fb22ef7dc9bb7fe3fb9cf96bcd3f2eb8d36` |
| 98 | `normal/98.mat` | `normal` | 0 | 0.000 | 1 | 1772 | 1772* | `train` | `X098_DE_time` | 483,903 | 200 | 7,742,720 | `37e6612c05e6ecc2ff0ff7bc7ef6eb9b73bdff04902abcfce638c7eb5a39252d` |
| 99 | `normal/99.mat` | `normal` | 0 | 0.000 | 2 | 1750 | 1750* | `val` | `X099_DE_time` | 485,063 | 200 | 15,503,928 | `4b97e6b5361f0cd4a5f80f9b6f0f0226f508ea014c00bc2479c6d414cf335d40` |
| 100 | `normal/100.mat` | `normal` | 0 | 0.000 | 3 | 1730 | 1725 | `test` | `X100_DE_time` | 485,643 | 200 | 7,770,624 | `88a5990cb5411be1efcfb9abdeff89f2e1f8eb69755f64388cdad64fb7966f0e` |
| 105 | `inner/105.mat` | `inner` | 1 | 0.007 | 0 | 1797 | 1797 | `train` | `X105_DE_time` | 121,265 | 59 | 2,910,768 | `f80b0ea04fd0ef22ef7bfbeed68df0cb0ee8b2a2b0d8602bc3c5e6132c9493f3` |
| 106 | `inner/106.mat` | `inner` | 1 | 0.007 | 1 | 1772 | 1772 | `train` | `X106_DE_time` | 121,991 | 59 | 2,928,192 | `e5cec7cdd138ca038a57d3b6be9ae3adce55e44be096cf8ccfc8a4ae37b6376a` |
| 107 | `inner/107.mat` | `inner` | 1 | 0.007 | 2 | 1750 | 1748 | `val` | `X107_DE_time` | 122,136 | 59 | 2,931,672 | `111ba8996a11b2ce9aaef019b1fb08349d2ed563f9d762945759fc8612b17e9f` |
| 108 | `inner/108.mat` | `inner` | 1 | 0.007 | 3 | 1730 | 1721 | `test` | `X108_DE_time` | 122,917 | 60 | 2,950,416 | `d415f0e65128a70be1b8d39a2b86d60e8626e65efcd1ae4e74ab76bf2a8c28b4` |
| 118 | `ball/118.mat` | `ball` | 2 | 0.007 | 0 | 1797 | 1796 | `train` | `X118_DE_time` | 122,571 | 59 | 2,942,112 | `b00628f8dd8d5284cf835dc1b651f33a8ec8162fbcc3c0fa7d60de621899e577` |
| 119 | `ball/119.mat` | `ball` | 2 | 0.007 | 1 | 1772 | 1774 | `train` | `X119_DE_time` | 121,410 | 59 | 2,914,248 | `bc72d9df7668a73dc580cd2bd33c3cb3b51de8614536cb6fc6ffca991ebbc5bc` |
| 120 | `ball/120.mat` | `ball` | 2 | 0.007 | 2 | 1750 | 1754 | `val` | `X120_DE_time` | 121,556 | 59 | 2,917,752 | `e0b7a584c49a7f346920c5c7508ffdc5d7ff78f986fa9c1efce7c0077b1224d8` |
| 121 | `ball/121.mat` | `ball` | 2 | 0.007 | 3 | 1730 | 1729 | `test` | `X121_DE_time` | 121,556 | 59 | 2,917,752 | `52f686e984ba238d08e3de5bc55a2797e023ff2138e7297cd26a32888b263fa8` |
| 130 | `outer/130.mat` | `outer` | 3 | 0.007 | 0 | 1797 | 1796 | `train` | `X130_DE_time` | 121,991 | 59 | 2,928,192 | `35a095307d09b3e425f9a66b91cd16ef415ad8dc447cfee939a9a7f863bf5492` |
| 131 | `outer/131.mat` | `outer` | 3 | 0.007 | 1 | 1772 | 1771 | `train` | `X131_DE_time` | 122,426 | 59 | 2,938,632 | `7883f7b83bea9f31e2404e71c7c1fb436f7e8a9c5a9717c83e9f67e1c2b9fb21` |
| 132 | `outer/132.mat` | `outer` | 3 | 0.007 | 2 | 1750 | 1750 | `val` | `X132_DE_time` | 121,410 | 59 | 2,914,248 | `17a69ed5d22749697cc74028492e8aa80a3bd3c5b2f8aa45e78c8bf9b0111f35` |
| 133 | `outer/133.mat` | `outer` | 3 | 0.007 | 3 | 1730 | 1725 | `test` | `X133_DE_time` | 122,571 | 59 | 2,942,112 | `53f076cb0d90c3b6fb9df3b72fc8c44bf24db785b024c686495f97e2f4ee98be` |

*\*Note: `98.mat` and `99.mat` do not embed a separate `*RPM` scalar key inside the `.mat` dictionary; their nominal speeds from CWRU's baseline table are `1772` RPM and `1750` RPM.*

## 4. Schema of Generated Artifacts

### 4.1 `docs/dataset_checksums.csv`
Columns: `file,file_id,label,load_hp,split,channel_key,sampling_hz,recorded_rpm,num_samples,num_windows,window_range,bytes,sha256,original_url,checked_utc`.

### 4.2 `outputs/test_predictions.csv`
Columns: `run_id,window_index,file_id,filename,load_hp,true_label,predicted_label,rf_predicted_label,correct,score_normal,score_inner,score_ball,score_outer,top_uncalibrated_score`.
