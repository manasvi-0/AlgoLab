
# DBSCAN Clustering

## 📌 What is DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised clustering algorithm based on the **density** of points.

It can find clusters of **arbitrary shapes** and identify outliers (noise).

---

## 🛠️ Parameters

| Parameter | Description |
|-----------|-------------|
| `eps` | Maximum distance between two samples to be considered neighbors |
| `min_samples` | Minimum number of neighbors to form a dense region |

---

## 🎯 Output

- **Clustered Scatter Plot**
- **Silhouette Score** (if valid)
- **Clustered Data Preview**

---

## 🧑‍💻 How It Works in Streamlit UI

1. Upload a `.csv` file with 2 numeric columns
2. Select `eps` and `min_samples` via sliders
3. Click **"Run DBSCAN"**
4. Get visual clusters and evaluation score (if possible)

---

## ⚠️ Special Note

If DBSCAN finds only one cluster or marks all points as noise, silhouette score cannot be calculated.

---

## ✅ Example CSV Format

```csv
x,y
1,2
2,3
3,4
8,7
9,6
10,8
