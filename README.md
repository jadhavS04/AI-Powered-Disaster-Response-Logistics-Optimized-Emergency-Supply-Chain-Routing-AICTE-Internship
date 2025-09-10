# AI-Powered-Disaster-Response-Logistics-Optimized-Emergency-Supply-Chain-Routing-AICTE-Internship

# 🚑 AI-Powered Disaster Response Logistics  
**Optimized Emergency Supply Chain Routing – AICTE Internship Project**

---

## Project Overview
This project provides an **AI-powered, hazard-aware disaster response routing system** for Mumbai. Key features:

- Select a vehicle and a shelter.
- Predict shelter risk using a trained machine learning model.
- Calculate optimized evacuation routes considering road hazards.
- Visualize routes, vehicles, shelters, and hazard zones interactively.
- Download routes as CSV or GeoJSON for further use.
- Perform data analysis and predictive modeling using Random Forest and Logistic Regression.

---

## Dataset
All datasets required are included in this repository:

| File | Description |
|------|-------------|
| `mumbai_shelters_large.csv` | Shelter information: latitude, longitude, capacity, etc. |
| `mumbai_vehicles_large.csv` | Vehicle information: location, availability, etc. |
| `mumbai_hazard_zones_large.csv` | Hazard zones: risk level, coordinates, hazard type |
| `mumbai_roads_nodes.csv` | Road network nodes (x,y coordinates) |
| `mumbai_roads_edges.csv` | Road network edges (geometry, length, OSM ID) |

### Download Datasets
You can download all datasets directly from this repository by cloning:

```bash
git clone https://github.com/jadhavS04/AI-Powered-Disaster-Response-Logistics-Optimized-Emergency-Supply-Chain-Routing-AICTE-Internship.git
cd AI-Powered-Disaster-Response-Logistics-Optimized-Emergency-Supply-Chain-Routing-AICTE-Internship

Environment Setup
1️⃣ Create Virtual Environment
python -m venv myenv

2️⃣ Activate Environment
Windows:
.\myenv\Scripts\activate

Linux / macOS:
source myenv/bin/activate

3️⃣ Install Dependencies
pip install streamlit pandas joblib networkx shapely geopandas folium streamlit-folium scikit-learn matplotlib seaborn

Run Application
streamlit run disasterapp.py

The application will open in your default browser.



Machine Learning Models
1️⃣ Random Forest Classifier
Train Random Forest on features: latitude, longitude, capacity.
Predict risk_level of shelters.
Evaluate using accuracy, F1-score, confusion matrix.

2️⃣ Logistic Regression
Train Logistic Regression on same features.
Evaluate performance similarly.
Compare with Random Forest.

3️⃣ Model Evaluation
Metrics:
Accuracy
Precision / Recall
F1-score
Confusion matrix
ROC-AUC

4️⃣ Hyperparameter Tuning
RandomizedSearchCV or GridSearchCV for:
Random Forest: n_estimators, max_depth, min_samples_split
Logistic Regression: C (regularization strength), penalty
Optimize for best cross-validated accuracy.

Dependencies
Python >= 3.9
Streamlit
Pandas
Joblib
NetworkX
Shapely
GeoPandas
Folium
Streamlit-Folium
Scikit-learn
Matplotlib
Seaborn
