# VoltStream: Sub-Grid Energy Anomaly Detection Framework

VoltStream is a machine learning framework designed to detect operational anomalies, equipment faults, and sudden demand spikes within automated utility structures. Built completely in Python, the system implements a dual-verification safety protocol combining supervised forecasting, supervised classification, and unsupervised outlier detection.

The framework processes historical telemetry from the Grid-India network to establish behavioral baselines, decoupling the heavy data engineering pipelines from real-time interactive execution to maximize computing efficiency.

## System Architecture and Design Reasoning

The system uses a highly decoupled, modular design divided into clear Functional Layers. This architecture separates data processing from pipeline orchestration and runtime execution.

### Architectural Blueprint

```text
VoltStream/
│
├── data/
│   └── hourlyLoadDataIndia.csv   # Raw telemetry dataset
│
├── models/                       # Frozen artifact repository (Serialized via Joblib)
│   ├── anomaly_classifier.joblib
│   ├── load_forecaster.joblib
│   ├── unsupervised_detector.joblib
│   └── feature_names.joblib
│
├── src/
│   ├── __init__.py               # Package initialization marker
│   ├── data_ingestion.py         # Sub-grid extraction logic
│   ├── feature_engineering.py    # Statistical framing and labeling
│   ├── model_pipeline.py         # Supervised classification engine
│   ├── unsupervised_engine.py    # Unsupervised isolation engine
│   ├── load_forecaster.py        # Time-series regression engine
│   └── app.py                    # Live gateway evaluation matrix
│
├── train.py                      # Offline heavy training orchestrator
└── predict.py                    # Real-time interactive execution shell
└── main_evaluation_pipeline.py   # Development sandbox for end-to-end RAM integration testing
```

### Core Design Decisions and Engineering Justification

#### 1. Sub-Grid Isolation Over Macro-Modeling
Instead of building a model on the combined national electricity demand of India, the ingestion layer explicitly isolates the Northern Region sub-grid (Northen Region Hourly Demand).

* **Reasoning:** Climate, industrial shifts, and seasonal weather patterns are highly diverse across different geographic zones. Combining these distinct data channels flattens local variances. A severe localized heatwave in New Delhi causes massive air-conditioning demand spikes that would look like normal noise if averaged with the cooler monsoon zones of Southern India. Sub-grid targeting prevents true facility leaks from being washed out in an aggregate average.

#### 2. Complete Decoupling of Train and Predict Cycles (Serialization)
The heavy data processing engine (train.py) is completely separated from the live prediction engine (predict.py) using the joblib library.

* **Reasoning:** Training machine learning algorithms requires scanning all 46,728 continuous hourly logs, computing rolling matrices, and fitting hundreds of decision trees. This process demands substantial RAM and execution time. In a live industrial environment, a guardian gateway must process incoming telemetry instantly. By saving the trained mathematical states to hard drive assets (.joblib), the application layer loads and runs model logic in milliseconds without touching raw historical files.

#### 3. Dual-Verification Fault Identification Matrix
The system processes data packets through a supervised classifier (Random Forest) and an unsupervised outlier engine (Isolation Forest) simultaneously.

* **Reasoning:** Supervised models excel at matching known patterns based on historical labels but fail when encountering novel, undocumented grid failure modes. Unsupervised models identify structural anomalies based purely on data density profiles without needing labels, but they can trigger false alarms during legitimate holiday shut-downs. Forcing both models to run simultaneously minimizes false alerts while maintaining a highly sensitive shield against equipment failures.

## Code Modules Breakdown

### 1. Ingestion and Feature Layer

* **src/data_ingestion.py**
Dynamically identifies the datetime index, targets the Northern Region channel despite dataset typos (Northen), drops unused geographical load variables, applies sequential sorting, and sanitizes missing sensor telemetry using forward-fill propagation.

* **src/feature_engineering.py**
Extracts time-series cyclic markers (hour, day of week, weekend boolean, calendar month). Computes rolling statistical baselines per hour across 5 years of history to calculate the mean and standard deviation. It algorithmically injects ground truth anomaly labels where load spikes exceed 1.5 standard deviations above the historical hour average.

### 2. Multi-Engine Learning Layer

* **src/model_pipeline.py**
Implements a Supervised Random Forest Classifier. Uses a stratified train-test split to keep imbalanced anomaly distributions identical across data pools. It trains a multi-tree system to recognize the specific features that indicate an overload.

* **src/unsupervised_engine.py**
Implements an Unsupervised Isolation Forest. Evaluates multi-dimensional data density profiles without target variables (y). It isolates rare points based on a strict 5% contamination limit.

* **src/load_forecaster.py**
Implements a Supervised Random Forest Regressor. It models the historical expected baseline consumption for any hour of the year, achieving a low Mean Absolute Error of 5,378.34 MW.

### 3. Application Execution Layer

* **src/app.py**
Maintains the production gateway decision matrix. It checks incoming live loads against the regression forecast. If both the supervised and unsupervised engines confirm a spike, it triggers a critical red alert. If only one system catches the deviation, or if the load exceeds 1.3 times the base forecast, it flags a single-source warning to route the data for secondary checks.

### 4. Development and Prototyping Sandbox

* **main_evaluation_pipeline.py**
This script acts as the framework's integration testing sandbox. It executes the entire end-to-end pipeline (Data Ingestion, Feature Engineering, Training, and Telemetry Simulation) sequentially completely within the system's RAM. It is explicitly separated from the production environment to allow developers to rapidly test new data cleansing laws, evaluate alternative hyperparameters, or prototype a fourth machine learning engine without corrupting the frozen serialized model assets stored on disk.

## Installation and Environment Calibration

1. Clone the repository to your chosen local storage drive:
```bash
cd /d
mkdir VoltStream
cd VoltStream
```

2. Establish and initialize an isolated Python virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate
```

3. Ensure core data science dependencies are fully updated within the local tracking profile:
```bash
pip install --default-timeout=100 pandas numpy scikit-learn joblib
```

4. Place your raw Kaggle data source inside the data/ directory and name it exactly:
```text
data/hourlyLoadDataIndia.csv
```

## Operational Execution Guide

### Phase 1: Compile and Freeze Infrastructure

Run the training pipeline once to ingest files, engineer characteristics, evaluate metrics, and save your machine learning assets to storage.

```bash
python train.py
```

Expected Terminal Verification Output:
```text
============================================================
        VOLTSTREAM: SECURE ML TRAINING PIPELINE        
============================================================
[LOG] Ingesting raw Grid-India hourly data streams...
[LOG] Identified Time Marker: 'datetime' | Targeted Regional Node: 'Northen Region Hourly Demand'
[SUCCESS] Ingested 46728 regional zone hourly data records.
[LOG] Transforming raw logs into temporal and statistical features...
[LOG] Initializing Anomaly Classification Engine...

=============================================
      VOLTSTREAM: ANOMALY CLASSIFIER MATRIX       
=============================================
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      8541
           1       0.99      0.98      0.98       805

    accuracy                           1.00      9346
   macro avg       0.99      0.99      0.99      9346
weighted avg       1.00      1.00      1.00      9346

=============================================

[LOG] Initializing Unsupervised Isolation Forest Engine...
[SIMPLE SUMMARY] Unsupervised Isolation Forest trained successfully on multi-dimensional patterns.
[LOG] Initializing Time-Series Forecasting Engine...

=============================================
      VOLTSTREAM: LOAD FORECASTER REGRESSION      
=============================================
Mean Absolute Error: 5378.34 MW
=============================================

[LOG] Serializing trained ML infrastructure onto disk...

============================================================
[SUCCESS] ALL 3 ENGINES COMPILED & FROZEN TO STORAGE SUCCESSFULLY!
============================================================
```

### Phase 2: Live Real-Time Testing

Run the interactive live evaluation script to test data packets instantly.

```bash
python predict.py
```

Interactive Testing Guide:

* **Scenario A (Normal Baseline Load Testing):**
Enter standard normal load metrics when prompted.
```text
Enter Observed Regional Load in MW (or type 'exit'): 34000

Configure Telemetry Packet Timestamp Context:
  Enter Hour (0-23): 4
  Enter Month (1-12): 12
  Enter Day of Week (0=Monday, 6=Sunday): 3

[LIVE INGESTION] Node Timestamp: 2026-12-15 04:00:00
Observed Load: 34000.0 MW | Expected Base Forecast: 32758.67 MW
Pipeline Decisions -> Supervised Flag: 0 | Unsupervised Flag: 0
[SYSTEM OK] All core parameters match normal grid optimization baselines.
```

* **Scenario B (Critical Spike Threshold Testing):**
Enter an extreme demand spike value to test multi-engine verification fallback tracking.
```text
Enter Observed Regional Load in MW (or type 'exit'): 85000

Configure Telemetry Packet Timestamp Context:
  Enter Hour (0-23): 4
  Enter Month (1-12): 5
  Enter Day of Week (0=Monday, 6=Sunday): 4

[LIVE INGESTION] Node Timestamp: 2026-05-15 04:00:00
Observed Load: 85000.0 MW | Expected Base Forecast: 47190.93 MW
Pipeline Decisions -> Supervised Flag: 1 | Unsupervised Flag: 0
[WARNING] SUSPICIOUS ANOMALY FLAGGED. Single-source deviation detected. Routing to secondary check.
  └─> Trigger Source: Supervised Random Forest Classifier flagged historical spike pattern
  └─> Trigger Source: Observed load exceeded the 30% baseline variance threshold (Limit: 61348.21 MW)
```

To safely disconnect and exit the running framework instance, type `exit` into the prompt.

---

## Technical Summary (STAR Method & Architectural Impact)

### Situation (S)
In automated utility grids and heavy infrastructure facility management (e.g., smart complexes, hospitals, data centers), sudden energy leaks or mechanical equipment overrides occur completely unnoticed. Standard monitoring tools apply crude, static threshold bounds that suffer from high false alarm rates and complete blindness to seasonal changes.

### Task (T)
The task was to design an intelligent, standalone time-series parsing framework capable of ingesting a five-year unmanipulated data stream (46,728 continuous rows from the Grid-India network). A critical engineering decision was made to isolate the Northern Region sub-grid sub-profile instead of modeling combined macro-national data, preventing distinct, climate-driven cooling peaks from being flattened out.

### Action (A)
Developed a modular, highly decoupled object layer consisting of three separate core algorithms:
1. **Supervised Random Forest Regressor:** Built as a continuous trend forecaster to compute time-based baseline load expectations.
2. **Supervised Random Forest Classifier:** Programmed to recognize historical overload signature boundaries based on rolling statistical labels.
3. **Unsupervised Isolation Forest:** Deployed as a zero-day defensive watchdog to score structural layout shape anomalies without target vectors.
The heavy calculations were fully separated from runtime execution loops by implementing `joblib` object serialization.

### Result & Impact (R)
* **High Predictive Precision:** The continuous regression tracker achieved a low Mean Absolute Error of **5,378.34 MW** across a massive power sector network.
* **Elite Classification Metrics:** The risk classification module secured an elite **98% to 99% F1-score** on unexposed training data configurations.
* **Latency Optimization:** Decoupled architectural structure completely bypassed raw CSV storage reading bottlenecks during real-time evaluations, cutting live inference delay to mere milliseconds.
* **Granular Transparency:** The dual-verification pipeline logs itemized root-cause metadata strings, allowing network operators to immediately locate whether a warning stems from an ensemble pattern anomaly or an explicit threshold variance breaker.
