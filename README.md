# Algerian Forest Fires – Fire Weather Index Prediction

A machine learning project that predicts the **Fire Weather Index (FWI)** for two regions in Algeria using meteorological and FWI component data. The final model is a **Ridge Regression** trained on the Algerian Forest Fires dataset (2012), achieving an R² score of **~0.985**.

---

## Project Structure

```
ml codes/
└── model/
    ├── 2.0-EDA And FE Algerian Forest Fires.ipynb   # Data exploration & feature engineering
    ├── 3.0-Model Training.ipynb                      # Model training & evaluation
    ├── Algerian_forest_fires_dataset_UPDATE.csv      # Raw dataset
    ├── Algerian_forest_fires_cleaned_dataset.csv     # Cleaned dataset (output of EDA notebook)
    ├── ridge.pkl                                     # Trained Ridge Regression model
    └── scaler.pkl                                    # Fitted StandardScaler
```

---

## Dataset

**Source:** Algerian Forest Fires Dataset  
**Coverage:** June – September 2012, two Algerian regions:
- **Bejaia Region** (northeast)
- **Sidi Bel-abbes Region** (northwest)

### Features

| Feature       | Description                               |
|---------------|-------------------------------------------|
| `Temperature` | Daily temperature (°C)                    |
| `RH`          | Relative Humidity (%)                     |
| `Ws`          | Wind speed (km/h)                         |
| `Rain`        | Daily rainfall (mm)                      |
| `FFMC`        | Fine Fuel Moisture Code (FWI component)  |
| `DMC`         | Duff Moisture Code (FWI component)       |
| `DC`          | Drought Code (FWI component)             |
| `ISI`         | Initial Spread Index (FWI component)     |
| `BUI`         | Buildup Index (FWI component)            |
| `Classes`     | Fire / Not Fire (binary, encoded 0/1)    |
| `Region`      | 0 = Bejaia, 1 = Sidi Bel-abbes          |

**Target:** `FWI` – Fire Weather Index (continuous regression target)

---

## Workflow

### 1. Exploratory Data Analysis & Feature Engineering (`2.0-EDA And FE ...ipynb`)

- Loaded and merged data from both Algerian regions
- Handled missing values and erroneous entries
- Encoded the `Classes` column: `"not fire"` → `0`, `"fire"` → `1`
- Added a `Region` column to distinguish the two geographical zones
- Dropped temporal columns (`day`, `month`, `year`) as they were not informative for regression

### 2. Model Training (`3.0-Model Training.ipynb`)

#### Data Preparation
- **Train/Test Split:** 75% train / 25% test (`random_state=42`)
- **Correlation-based Feature Selection:** Removed highly correlated features with threshold `> 0.85`
  - Dropped: `BUI`, `DC` (highly correlated with `DMC`)
  - Remaining features: `Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `ISI`, `Classes`, `Region`
- **Feature Scaling:** `StandardScaler` applied to training set; same scaler used at test time

#### Models Evaluated

| Model              | MAE     | R² Score |
|--------------------|---------|----------|
| Linear Regression  | ~0.547  | ~0.985   |
| Lasso Regression   | ~1.133  | ~0.949   |
| **Ridge Regression** (selected) | Best via CV | Best via CV |

Ridge Regression with cross-validation (`RidgeCV`) was selected as the final model based on its performance stability across folds.

#### Serialization
The trained model and scaler are saved using `pickle`:
```python
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(ridge,  open('ridge.pkl',  'wb'))
```

---

## Usage

### Prerequisites

Install the required dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Notebooks

```bash
# 1. EDA & Feature Engineering
jupyter notebook "model/2.0-EDA And FE Algerian Forest Fires.ipynb"

# 2. Model Training
jupyter notebook "model/3.0-Model Training.ipynb"
```

### Running Inference with the Saved Model

```python
import pickle
import numpy as np

# Load artifacts
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
ridge  = pickle.load(open('model/ridge.pkl',  'rb'))

# Example input: [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
sample = np.array([[29, 57, 18, 0, 65.7, 3.4, 0.5, 1, 0]])

# Preprocess and predict
sample_scaled = scaler.transform(sample)
fwi_prediction = ridge.predict(sample_scaled)
print(f"Predicted FWI: {fwi_prediction[0]:.2f}")
```

> **Important:** Always apply `scaler.transform()` before calling `ridge.predict()`. Do **not** refit the scaler on new data.

---

## Dependencies

| Library      | Purpose                         |
|--------------|---------------------------------|
| `numpy`      | Numerical computation           |
| `pandas`     | Data manipulation               |
| `scikit-learn` | ML models, preprocessing, metrics |
| `matplotlib` | Plotting                        |
| `seaborn`    | Statistical visualization       |
| `pickle`     | Model serialization             |

---

## Model Performance

| Metric               | Value   |
|----------------------|---------|
| Algorithm            | Ridge Regression (with Cross-Validation) |
| MAE (Linear Reg.)    | ~0.547  |
| R² Score (Linear Reg.) | ~0.985 |
| MAE (Lasso)          | ~1.133  |
| R² Score (Lasso)     | ~0.949  |

> Ridge Regression was selected as the final production model due to its superior cross-validated performance and resistance to overfitting.

---

## Notes

- The dataset covers fire events and weather observations recorded during the **summer months (June–September)** only.
- The `Region` feature adds a spatial dimension to the model without requiring separate models per region.
- This project is in a **ready-to-deploy** state — the `ridge.pkl` and `scaler.pkl` artifacts can be plugged directly into a Flask/FastAPI inference API.
