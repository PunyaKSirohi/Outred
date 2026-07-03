# Outred  - Outlier Detection Engine

**Outred** is a production-grade outlier detection tool for tabular CSV data. It auto-selects from 5 machine learning algorithms based on your data's characteristics, handles files of any size via streaming, and provides explainable results.

Available as both a **CLI tool** and a **web application** with browser-side and server-side processing options.

---

## Quick Start

### Install

```bash
# Clone and set up
cd outred
python -m venv outredenv
source outredenv/Scripts/activate   # Windows: .\outredenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### CLI Usage

```bash
# Auto-detect the best algorithm
python main.py -i data.csv

# Choose a specific algorithm
python main.py -i data.csv --algorithm hbos

# Run the ensemble (IForest + HBOS + LOF averaged)
python main.py -i data.csv --algorithm ensemble

# Custom contamination rate (expect 10% outliers)
python main.py -i data.csv -n 0.10

# Get SHAP-based explanations for flagged rows
python main.py -i data.csv --explain

# Custom output path
python main.py -i data.csv -o my_results.parquet
```

### Web UI

```bash
python main.py --serve
# Open http://127.0.0.1:8000
```

---

## CLI Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | string | required | Path to input CSV file |
| `--output` | `-o` | string | `results/output.parquet` | Path to output Parquet file |
| `--algorithm` | `-a` | choice | `auto` | `auto\|ensemble\|iforest\|hbos\|lof\|cblof\|ocsvm` |
| `--contamination` | `-n` | float | `0.05` | Expected outlier proportion (0.001–0.20) |
| `--scaling` | `-s` | choice | `robust` | `robust\|standard\|minmax\|none` |
| `--impute` | | choice | `median` | `median\|mean\|zero\|drop` |
| `--chunk-size` | `-c` | int | `100,000` | Rows per chunk for streaming |
| `--explain` | `-e` | flag | off | Compute SHAP feature contributions |
| `--serve` | | flag | off | Launch the web UI server |
| `--host` | | string | `127.0.0.1` | Web server host |
| `--port` | | int | `8000` | Web server port |

---

## Algorithms

| Name | PyOD Model | Best For | Speed |
|------|-----------|----------|-------|
| **IForest** | Isolation Forest | General-purpose, high-dimensional data | Fast |
| **HBOS** | Histogram-Based Outlier Score | Very large datasets (>1M rows) | Fastest |
| **LOF** | Local Outlier Factor | Data with varying cluster densities | Medium |
| **CBLOF** | Clustering-Based LOF | Data that naturally forms groups | Medium |
| **OCSVM** | One-Class SVM | Small, complex non-linear datasets | Slow |
| **Ensemble** | IForest + HBOS + LOF | Best accuracy (averages 3 models) | Medium |

### Smart Auto-Selection

When `--algorithm auto` (the default), Outred profiles your data and selects:

- **Rows > 1M** → HBOS (fastest)
- **Columns > 50** → IForest (handles high dimensionality)
- **Highly skewed data** → LOF (handles varying densities)
- **File > 500MB** → Incremental SGDOneClassSVM (out-of-core)
- **Default** → Ensemble (most robust)

---

## Output

The output Parquet file contains all original columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `anomaly_score` | float (0–1) | Higher = more anomalous |
| `is_outlier` | bool | True if flagged as numeric outlier |
| `cat_anomaly_score` | float (0–1) | Rarity score for categorical values |
| `is_cat_outlier` | bool | True if any category is rare |

---

## Web API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/profile` | POST | Upload CSV, get data profile |
| `/api/analyze` | POST | Upload CSV, run detection, get results |
| `/` | GET | Web frontend |

---

## Architecture

```
main.py (CLI)  ──┐                    ┌── engines/tabular.py (5 PyOD algos)
                  ├── config.py ──┐    ├── engines/incremental.py (SGD SVM)
server.py (API) ──┘               │    ├── engines/categorical.py (frequency)
                        profiler.py    └── engines/timeseries.py (V2 stub)
                        preprocessing.py
                        router/dispatcher.py (smart routing)
                        reporter/aggregator.py (streaming Parquet)
                        explainer.py (SHAP)
```

---

## Privacy

- **Browser Mode**: Data is processed entirely in the browser using JavaScript. Nothing is uploaded to any server.
- **Server Mode**: Data is uploaded to a temp file, processed in memory, and immediately deleted. Zero data retention.

---

## Running Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

---

## License

MIT
