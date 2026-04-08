## Repository Contents

### Dataset
* The SOP measurement dataset is available via McMaster University Dataverse (Borealis) at https://doi.org/10.5683/SP3/RPCWBY.

### Figure Source Files

- **Fig. 1–4** were generated using [OriginPro](https://www.originlab.com/). The project source file is located at `./OriginPro file/SOP_meas.opju`.

### Python Scripts

| Script | Description |
|--------|-------------|
| `Fig3c_SOP_XGBoost_SHAP_aging.py` | Trains an XGBoost-based SOP estimator and performs SHAP (SHapley Additive exPlanations) analysis to interpret feature contributions. |
| `Fig3f_tryFittingSOPvsR.py` | Identifies the best-fit function for SOP vs. internal resistance (R) during SOP degradation under a CC-discharge aging profile at **50% SOC**. |
| `Fig3g_fitting_RvsSOP.py` | Identifies the best-fit function for SOP vs. internal resistance (R) during SOP degradation under a CC-discharge aging profile at **5% SOC**. |
| `Fig4a_fitting_cycleVSSOP.py` | Identifies the best-fit function for SOP vs. cycle number during SOP degradation under a CC-discharge aging profile. |
| `FigS15_SOP_Spearman_analysis.py` | Performs Spearman rank correlation analysis on the SOP aging dataset to assess feature–target monotonic relationships. |