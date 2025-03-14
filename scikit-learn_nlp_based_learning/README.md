# Scikit-Learn NLP

## 1. Introduction
I have done a comprehensive application of the scikit-learn library (version 1.5.2) for natural language processing (NLP) using the 20 Newsgroups dataset. This project demonstrates various functionalities of scikit-learn, including text preprocessing, feature engineering, supervised and unsupervised learning, model evaluation, and more.

## 2. Initial Setup: Environment Check and Imports
I have done an initial setup to ensure that the Python environment is configured correctly and imported all necessary scikit-learn modules.

### **2.1 Environment Check**
- I have used `sys.executable` to print the Python interpreter path to verify compatibility with scikit-learn.
- I have attempted to import `sklearn` and print its version. If not installed, an error is raised with installation instructions.

### **2.2 Imports**
- I have imported necessary modules from `sklearn`, including `datasets`, `model_selection`, `preprocessing`, `linear_model`, `ensemble`, `svm`, `cluster`, `decomposition`, `manifold`, `feature_selection`, `pipeline`, `impute`, `feature_extraction.text`, `gaussian_process`, `calibration`, `metrics`, `inspection`, `neural_network`, and `base`.
- Additional imports include `numpy (np)`, `matplotlib.pyplot (plt)`, and `joblib` for array operations, plotting, and model persistence.

## 3. Dataset: 20 Newsgroups
I have loaded the 20 Newsgroups dataset and selected the following categories:
- `alt.atheism`
- `soc.religion.christian`
- `comp.graphics`
- `sci.med`

I have done the following:
- Downloaded the dataset using `datasets.fetch_20newsgroups()`.
- Extracted the text data (`X`) and category labels (`y`).
- Printed dataset size and target names for verification.

## 4. Feature Engineering
I have done additional feature engineering to enhance the dataset:
- Computed `doc_lengths = [len(doc.split()) for doc in X]` as a numerical feature.
- Created `y_sentiment = np.random.randint(0, 2, size=len(y))` for multi-output classification.
- Converted numerical features to a format compatible with scikit-learn.

## 5. Handling Missing Values
I have demonstrated missing value imputation using `SimpleImputer`:
- Introduced missing values into numerical features.
- Applied mean imputation to fill in missing values.

## 6. Data Splitting
I have split the dataset into training and test sets using `train_test_split()` with a 70/30 split ratio.

## 7. Text Preprocessing and Vectorization
I have transformed text data into numerical representations using `TfidfVectorizer()`:
- Applied TF-IDF transformation, limiting to 5000 terms and removing stop words.
- Fit on training data and transformed test data.

## 8. Numerical Feature Processing
I have done numerical feature preprocessing using:
- `StandardScaler()` for feature standardization.
- `PolynomialFeatures(degree=2, include_bias=False)` for feature expansion.

## 9. Feature Combination
I have combined text and numerical features using:
- `TruncatedSVD(n_components=50, random_state=42)` for dimensionality reduction.
- Stacked the reduced text features with polynomial features.

## 10. Feature Selection
I have done feature selection using `SelectKBest(chi2, k=1000)` to select the top TF-IDF features.

## 11. Data Visualization
I have visualized data using `TSNE()` to reduce dimensions to 2D and plotted using `matplotlib`.

## 12. Supervised Learning Models
I have trained the following models:
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC` (with probability estimates)
- `MLPClassifier`
- Evaluated using `accuracy_score()`.

## 13. Ensemble Learning
I have implemented ensemble techniques using:
- `VotingClassifier` (soft voting)
- `BaggingClassifier` with Logistic Regression

## 14. Unsupervised Learning: Clustering
I have applied clustering using:
- `KMeans(n_clusters=4, random_state=42)`

## 15. Outlier Detection
I have detected anomalies using:
- `IsolationForest(contamination=0.1, random_state=42)`

## 16. Gaussian Process Regression
I have done a simulated regression task using `GaussianProcessRegressor()`.

## 17. Model Calibration
I have calibrated model probabilities using:
- `CalibratedClassifierCV(svm, cv=5, method='sigmoid')`

## 18. Multi-output Classification
I have demonstrated multi-target classification using:
- `MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1)`

## 19. Custom Estimator
I have created a custom classifier by inheriting from `BaseEstimator` and `ClassifierMixin`.

## 20. Time Series Cross-Validation
I have simulated time-series cross-validation using:
- `TimeSeriesSplit(n_splits=3)`

## 21. Cross-Validation and Hyperparameter Tuning
I have optimized models using:
- `cross_val_score()`
- `GridSearchCV()`

## 22. Model Evaluation Metrics
I have provided a detailed evaluation using:
- `classification_report()`
- `roc_curve()`, `auc()`

## 23. Learning and Validation Curves
I have analyzed model performance trends using:
- `learning_curve()`
- `validation_curve()`

## 24. Feature Importance: Partial Dependence Plots
I have visualized feature influence using `PartialDependenceDisplay.from_estimator()`.

## 25. Model Persistence
I have saved and loaded trained models using `joblib.dump()` and `joblib.load()`.

## 26. Building a Machine Learning Pipeline
I have implemented a streamlined workflow using `Pipeline()` with:
- `TfidfVectorizer`
- `TruncatedSVD`
- `RandomForestClassifier`

## 27. Making Predictions
I have applied the trained pipeline to new data using `pipeline.predict()`.

## 28. Running the Program
### **28.1 Install Dependencies:**
```bash
conda install scikit-learn numpy matplotlib joblib
```

### **28.2 Execute the script:**
```bash
python nlp_demo.py
```

### **28.3 Expected Output:**
- Prints evaluation metrics and model performances.
- Displays TSNE visualization, ROC curves, learning curves, and more.

## 29. Conclusion
I have demonstrated a full-fledged NLP pipeline using scikit-learn, covering preprocessing, feature engineering, model training, evaluation, and visualization. This README provides a structured guide to the implementation. For further details, refer to the [scikit-learn documentation](https://scikit-learn.org/).

