# Airbnb Open Data Analysis

Data science project analyzing factors that influence Airbnb listing prices in New York City.

**Authors:** Roie Shalev & Eve Menachem

## Overview

This project explores the Airbnb Open Data dataset (~100,000 NYC listings) to understand which features most strongly affect nightly price. The workflow includes data cleaning, exploratory analysis, KMeans clustering, and a Decision Tree classifier for price categories.

## Dataset

- **Source:** [Airbnb Open Data on Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)
- **Size:** 102,599 rows × 12 columns
- **Target variable:** `price` (nightly listing price)
- **Features:** host name, neighbourhood group, country, cancellation policy, room type, construction year, service fee, minimum nights, number of reviews, review rate, availability 365

## What the notebook does

1. **Data cleaning** — removes rows with negative `minimum nights` / `availability 365`, drops nulls, normalizes the `neighbourhood group` field (e.g. fixes `brookln` → `Brooklyn`), strips `$` from price strings.
2. **Exploratory analysis** — distribution of listings by borough, price vs. number of reviews, effect of construction year on price variability, average rating vs. number of properties per host, and a market-dynamics view of location × room type.
3. **Algorithm comparison** — benchmarks a manual bubble sort against `numpy.sort` on arrays from 10K to 1M elements, illustrating the practical impact of O(n²) vs. O(n log n).
4. **Feature engineering** — adds `total price`, `price_relative_to_neighborhood`, `host_property_count`, and `host_avg_rating`.
5. **Clustering** — encodes categorical columns with one-hot, normalizes with `MinMaxScaler`, and runs KMeans for k = 2..20, evaluating with SSE (elbow) and silhouette score. Final model uses k = 12.
6. **Cluster analysis** — interprets each cluster across price, minimum nights, availability, host property count, reviews, and rating. Identifies a luxury group (Cluster 7), a budget group (Cluster 9), an under-performing strategy (Cluster 11), and the most-booked standard segments (Clusters 0 and 3).
7. **Classification** — bins prices into Low / Mid / High and trains a `DecisionTreeClassifier` to predict the category from the other features. Reports accuracy and feature importances.

## Key findings

- The NYC Airbnb market splits into three clear price tiers (luxury, budget, standard), with most listings sitting in the standard tier.
- Operational factors (availability, number of reviews, host's average rating) predict price category more strongly than the listing's borough.
- Cluster 7 hosts charge ~$908/night on average; Cluster 9 hosts ~$340/night. Both are run by large, professional hosts.

## Running the notebook

Requires Python 3 with: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook Roie_Shalev_Eve_Menachem.ipynb
```

The notebook expects `Airbnb_Open_Data.csv` in the same directory. The CSV is not committed to this repo — download it from the Kaggle link above.
