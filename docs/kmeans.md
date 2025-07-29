# KMeans Clustering

## 📌 What is KMeans?

KMeans is a centroid-based unsupervised clustering algorithm that groups data into **k clusters**. It tries to minimize the distance between data points and their respective cluster centers.

---

## 🛠️ Parameters

| Parameter | Description |
|-----------|-------------|
| `n_clusters` | Number of clusters to form |
| `random_state` | Ensures reproducibility |
| `n_init='auto'` | Stable initialization |

---

## 🎯 Output

- **Clustered Scatter Plot**
- **Silhouette Score** to evaluate cluster separation
- **Preview of Clustered Data**

---

## 🧑‍💻 How It Works in Streamlit UI

1. Upload a `.csv` file with 2 numeric columns (e.g., x, y)
2. Select the number of clusters using a slider
3. Click **"Run KMeans Clustering"**
4. Get visual clusters and evaluation score

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
