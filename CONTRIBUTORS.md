## 👤 Contributor Roles

---

### 🧾 Contributor 1 – Dataset Handler

**Responsibilities:**
- Implement CSV upload + validation
- Identify features & target column
- Add built-in datasets using `sklearn.datasets`
- Display basic data insights (e.g., shape, nulls, types)

**Tools:** `pandas`, `sklearn.datasets`, `streamlit.file_uploader`

---

### 🔍 Contributor 2 – Supervised ML Visuals

**Responsibilities:**
- Implement KNN with neighbor highlighting
- Decision Tree with visual splits
- Logistic Regression with sigmoid curve
- SVM with hyperplane + support vectors

**Tools:** `scikit-learn`, `matplotlib`, `plotly`, `seaborn`

---

### 🧭 Contributor 3 – Unsupervised ML Visuals

**Responsibilities:**
- K-Means Clustering (color-coded clusters)
- DBSCAN (density-based clusters)
- PCA (2D projection)
- Hierarchical Clustering (dendrogram)

**Tools:** `scikit-learn`, `plotly`, `seaborn`

---

### 🎛️ Contributor 4 – Streamlit UI & Integration

**Responsibilities:**
- Build a sidebar to toggle "Supervised vs Unsupervised"
- Add algorithm dropdowns and inputs
- Integrate modules from other contributors
- Manage layout, styling, and responsiveness

**Tools:** `Streamlit`

## ✅ Final Deliverables

| File Name                | Description                                        | Owner           |
|--------------------------|----------------------------------------------------|-----------------|
| `data_handler.py`        | Dataset upload, validation, and loading logic      | Contributor 1   |
| `supervised_module.py`   | Implements supervised ML algorithms with visuals   | Contributor 2   |
| `unsupervised_module.py` | Implements unsupervised ML algorithms with visuals | Contributor 3   |
| `app.py`                 | Streamlit UI: layout, toggles, and routing         | Contributor 4   |



## 🔧 Project Module Overview

| Module                  | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| 📂 Dataset Module       | Upload custom CSV or generate synthetic datasets            |
| 🧠 Algorithm Engine     | Run supervised/unsupervised ML algorithms                   |
| 📊 Visualization Layer  | Visualize how algorithms behave and perform                 |
| 🖼️ UI Layer (Streamlit) | Interactive user interface to switch modes and explore algorithms |
