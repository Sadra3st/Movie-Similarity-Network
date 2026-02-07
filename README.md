
# Graph Algorithms in Real Networks — MovieLens Analysis

## 📖 Overview
This project analyzes the **MovieLens 100K** dataset by modeling it as a complex network. The goal is to apply fundamental graph algorithms to solve real-world problems in recommendation systems and data exploration.

The project implements:
1.  **Weighted Graph Construction:** Converting user ratings into a Movie-Movie similarity graph.
2.  **Shortest Path Analysis:** Finding "cultural bridges" between dissimilar movies.
3.  **Bipartite Matching:** Optimizing movie assignments to users using Network Flow.
4.  **Graph Embeddings :** Using Spectral Embedding and K-Means to cluster movies by genre automatically.

---

## 📂 Dataset

The project uses the **MovieLens 100K** dataset provided by GroupLens Research.
*   **Nodes:** 1,682 Movies, 943 Users.
*   **Edges:** 100,000 Ratings (1–5 scale).

> **Note:** The script is designed to **automatically download** the necessary data files (`u.data` and `u.item`) from the GroupLens servers upon execution. No manual download is required.

---

## ⚙️ Installation & Requirements

Ensure you have **Python 3.x** installed. The project relies on the following external libraries for data manipulation and machine learning:

```bash
pip install pandas numpy matplotlib scikit-learn requests
```

### Dependencies
*   `pandas`: Dataframe manipulation.
*   `numpy`: Matrix operations.
*   `requests`: Downloading the dataset.
*   `matplotlib`: Visualization of the embedding graph.
*   `sklearn`: Spectral Embedding and K-Means clustering

---

## 🤝 Credits

[Sadra Seyedtabaei](https://github.com/sadra3st) & [Kia Sheykhi](https://github.com/kia8506)

---

## 📄 License

MIT License
Copyright (c) 2025 Sadra Seyedtabaei
