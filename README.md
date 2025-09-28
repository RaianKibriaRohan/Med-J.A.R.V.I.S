# Predicting Health Conditions in Medical Students Using Machine Learning (Med J.A.R.V.I.S)

## Project Overview
This project aims to analyze and predict blood pressure (BP) levels based on various health and lifestyle factors among medical students using machine learning techniques. We apply multiple models to classify blood pressure into three categories: Low, Medium, and High. 

The project uses a combination of supervised and unsupervised machine learning models, including:
- **Supervised Learning:** Neural Network, K-Nearest Neighbors (KNN), Decision Tree
- **Unsupervised Learning:** KMeans Clustering

## Key Features
- **Predict BP levels** based on health metrics (BMI, cholesterol, smoking habits, etc.)
- **Visualize correlations** and data distributions with EDA
- **Compare models** for accuracy and performance using various metrics such as ROC, AUC, F1-score, and confusion matrices

## Dataset
The dataset consists of 199,979 records of medical students, including:
- **Features:**
  - Numerical: Age, Height, Weight, BMI, Temperature, Heart Rate, Cholesterol, Blood Pressure
  - Categorical: Gender, Smoking, Diabetes, Blood Type (A, B, AB, O)
- **Target:** BP_Class (Blood Pressure Class: Low, Medium, High) based on quantile splitting of blood pressure

### Preprocessing
- **Handling Missing Data:** Imputation using column means or mode
- **Encoding:** Categorical features (Gender, Smoking, Diabetes, Blood Type) were encoded
- **Scaling:** StandardScaler applied to numerical features
- **Train-Test Split:** 70% train, 30% test, stratified by BP_Class

## Model Training
- **KNN (K-Nearest Neighbors):** Predicts BP_Class based on nearest neighbors.
- **Decision Tree:** Provides an interpretable model highlighting important features.
- **Neural Network:** Captures complex non-linear relationships.
- **KMeans Clustering:** Unsupervised learning model to identify natural clusters in the data.

## Evaluation
- **Supervised Learning:** Achieved the following accuracies:
  - KNN: 37.9%
  - Decision Tree: 33.4%
  - Neural Network: 36.3%
- **Unsupervised Learning:** KMeans Clustering achieved an accuracy of 34.5% and low cluster separation, indicating weak grouping of BP levels.

### Metrics Used
- **Accuracy**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**

## Challenges & Limitations
- **Weak Feature Correlation:** Variables like BMI, cholesterol, and smoking alone cannot effectively predict blood pressure, which is influenced by multiple unaccounted factors.
- **Feature Overlap:** Significant overlap in BMI, cholesterol, and smoking across different BP levels made classification difficult.
- **Imbalanced Complexity:** The dataset lacked sufficient distinct features, making it challenging for neural networks to learn meaningful patterns.

## Conclusion
The project shows that while machine learning models can partially capture patterns in health data, predicting blood pressure accurately requires additional lifestyle and genetic data. KNN was the most effective model, but the current dataset is insufficient for high accuracy.

### Further Work
- Incorporate additional health features (e.g., diet, sleep patterns, genetic factors).
- Use medical thresholds for blood pressure classification instead of statistical quantiles.
- Explore more advanced models to capture complex relationships.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Keras (for Neural Networks)

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/Med-JARVIS.git
cd Med-JARVIS
```

### Contributors
Safin Ahmed Orko - Contributor (ID: 22299060), 
Raian Kibria Rohan - Contributor (ID: 22299407)
