# DeepDemand  
## Interpretable Long-Term Traffic Modelling on National Road Networks Using Theory-Informed Deep Learning

![Local OD extraction and OD pair screening](images/DeepDemand1.png)  
![Architecture of DeepDemand](images/DeepDemand2.png)

DeepDemand is a theory-informed deep learning framework for long-term, interpretable traffic volume modelling on national road networks.  
This repository provides the official PyTorch implementation accompanying the paper:

> **Yue Li**, Shujuan Chen, Akihiro Shimoda, Ying Jin  
> *Interpretable long-term traffic modelling on national road networks using theory-informed deep learning*  
> (Journal information to be updated)

DeepDemand integrates behavioural travel-demand theory with neural network modelling of origin–destination (OD) interactions. It enables spatially transferable, policy-relevant traffic forecasting with explicit interpretability at both the OD and network levels.

---

# 1. Reproducibility Requirements

## System Configuration

- **Operating System**: Windows 11  
- **Python Version**: 3.10.15  
- **Hardware**:
  - ≥16 GB RAM  
  - ≥16 GB GPU VRAM (CUDA-enabled GPU recommended)

## Core Dependencies

```
python==3.10.15
numpy==2.2.5
pandas==2.3.1
scipy==1.15.3
scikit-learn==1.7.1
numba==0.62.1
networkx==3.4.2
matplotlib==3.10.0
umap-learn==0.5.9.post2
torch==2.5.1
pytorch-cuda==12.1
torchvision==0.20.1
torchaudio==2.5.1
pyarrow==19.0.0
pyproj==3.6.1
```

A CUDA-compatible GPU is required for efficient training.

---

# 2. Repository Structure

## `data/` — Data Construction Pipeline

This directory contains scripts for constructing the complete modelling dataset:

1. Highway network extraction and simplification  
2. Node-level socioeconomic feature engineering  
3. Edge-centric local OD subgraph generation  
4. Traffic volume aggregation  

> **Important**  
> Raw datasets are not stored in this repository.  
> All primary data sources are open-access and should be downloaded from their official providers.  
> Essential processed datasets required for training are hosted on the University of Cambridge Apollo repository.  
> Additional large files can be provided upon request.

### `highway_network/`

- Download `great-britain-20250101.osm.pbf` from Geofabrik.
- Use `osmium` to extract a driving-network subset (`uk_highway_only.pbf`).
- `extract_driving_network.ipynb`:
  - Builds a simplified directed graph (`gpickle`)
  - Exports nodes and edges as GeoJSON
- `add_travel_time_to_uk_driving_graph_simplified.ipynb`:
  - Injects travel-time attributes as edge features

---

### `node_features/`

Constructs LSOA-level features:

- Administrative boundaries (ONS)
- Population
- Employment
- Land use
- Points of interest (POI)
- Additional variables (area, density, households, IMD)

`LSOA_node_pairing.ipynb` matches LSOA centroids to graph nodes.

Final feature banks:

- `lsoa21_features_raw.json`
- `lsoa21_features_normalized.json`

---

### `subgraphs/`

- `subgraph_generation.ipynb`:
  - Extracts edge-centric OD regions
- `HPC/`:
  - Valid OD pair screening scripts
- Output:
  - Per-edge OD sets stored in `od_use.feather`

---

### `traffic_volume/`

- Sensor metadata (`Sites.csv`)
- Sensor availability analysis
- Raw traffic data download and processing
- 8-year AADT aggregation
- Final ground truth file:
  - `GT_AADT_8years.json`

---

## `model/` — Model and Baselines

- `deepdemand.py` — DeepDemand architecture  
- `dataloader.py` — Data loading utilities  
- `utils.py` — Metrics and splitting  
- `baseline_*.py` — Benchmark models  

---

## `eval/` — Evaluation and Diagnostics

- `logs/` — Random CV and spatial CV training logs  
- `baselines/` — Baseline performance outputs  
- `cv.ipynb` — Random CV metric summaries  
- `spatial_cv.ipynb` — Spatial CV summaries  
- `evaluate_log.py` — Log parser  
- Residual analysis and visualisation scripts  

---

## `interpret/` — Interpretability Analysis

- Distance deterrence function analysis  
- UMAP visualisation of node embeddings  
- SHAP-based OD attribution analysis  
- Domain-level and subdomain-level feature importance  
- `od_pair_rf_shap.py` — SHAP protocol wrapper  

---

## Additional Directories

- `param/` — Trained model weights  
- `projection/` — Long-term projection scripts (e.g., 2040 scenarios)  
- `config.py` — Hyperparameter configuration  
- `config-template.py` — Template for batch experiments  
- `train.py` — Random cross-validation training  
- `train_spatial_cv.py` — Spatial cross-validation training  
- `train_multiple_models.ipynb` — Batch experiment automation  

---

# 3. Training Workflow

## Single Model (Random CV)

1. Modify hyperparameters in `config.py`
2. Run:

```bash
python train.py
```

---

## Spatial Cross-Validation

```bash
python train_spatial_cv.py
```

---

## Batch Experiments

Use:

```
train_multiple_models.ipynb
```

to automate multiple configurations sequentially.

---

# 4. Data Availability

- **Raw data**: Obtain directly from official open-data providers (e.g., OSM, ONS, National Highways).
- **Processed training data**: Hosted on the University of Cambridge Apollo repository.
- **Large supplementary files**: Available upon request.

For methodological transparency, full reproduction from raw data is encouraged.

---

# 5. Citation

If you use DeepDemand in your research, please cite:

```bibtex
@article{DeepDemand2026,
  author  = {Yue Li and Shujuan Chen and Akihiro Shimoda and Ying Jin},
  title   = {Interpretable long-term traffic modelling on national road networks using theory-informed deep learning},
  journal = {To be updated},
  year    = {2026}
}
```

---

# 6. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

DeepDemand bridges theory-driven travel demand modelling and modern deep learning, enabling interpretable, scalable, and policy-relevant national-scale traffic forecasting.