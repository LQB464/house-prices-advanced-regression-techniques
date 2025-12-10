# House Prices Advanced Regression Techniques

This project implements a fully modular, end-to-end Machine Learning pipeline for the Ames Housing dataset.  
It includes data preprocessing, baseline model training, hyperparameter tuning using Optuna and GridSearchCV,  
and a final Stacking Ensemble optimized for best predictive performance.

---

## Key Features of the Pipeline

### **1. Data Loading and Splitting**
- Loads the Ames Housing dataset.
- Performs train–test split (default 80–20).
- Separates features and target column using a consistent preprocessing interface.

### **2. Feature Preprocessing Pipeline**
A unified `scikit-learn` `Pipeline` is constructed to ensure consistent transformations across models:
- Ordinal encoding using canonical mapping.
- Domain feature engineering.
- Numerical imputation and scaling.
- Variance threshold filtering.
- Feature selection using Mutual Information (SelectKBest).

This design provides:
- Clean modularity  
- Reproducibility  
- Plug-and-play compatibility with any model  

### **3. Baseline Model Training**
The pipeline evaluates a range of baseline models:
- Linear models: Ridge, Lasso, ElasticNet  
- Tree-based models: RandomForest  
- Kernel models: Support Vector Regression  
- Boosting models (if installed): XGBoost, CatBoost, LightGBM  

Each model is trained inside a full pipeline (`features → model`) and evaluated using:
- 5-fold Cross-Validation RMSE  
- Test RMSE and R²  

Results are logged and stored automatically.

### **4. Model Selection**
Top-K performers (default: 5) are selected based on cross-validated RMSE.  
These models proceed to the tuning stage.

### **5. Hyperparameter Tuning**
Two tuning approaches are used:

#### **Optuna (for advanced models)**
Applied to:
- Random Forest  
- Elastic Net  
- XGBoost  
- CatBoost  
- LightGBM  

Optuna optimizes:
- tree depth  
- number of estimators  
- regularization terms  
- sampling ratios  
and other model-specific parameters.

#### **GridSearchCV (for simpler models)**
Applied to:
- Ridge  
- Lasso  
- SVR  

A curated grid is searched for each model, focusing on key hyperparameters like `alpha`, `C`, and `epsilon`.

All tuned models are re-evaluated and registered in the results table.

### **6. Stacking Ensemble Optimization**
Using Optuna, the pipeline automatically:
- Selects the best combination of 3 tuned models  
- Tunes ElasticNet meta-learner hyperparameters  
- Decides whether to use passthrough features  

The resulting Stacking Ensemble typically outperforms every standalone model.

### **7. Result Saving and Visualization**
- All model metrics are saved into `model_results.csv`.
- RMSE comparison plots are generated.
- Full logs are stored under the `output/` directory.

### **8. Kaggle Submission Support**
The final model can generate a submission file in the correct Kaggle format:
```

Id,SalePrice
1461,xxxx
1462,xxxx
...

```
This makes the pipeline directly usable for competition participation.

---

## 📁 Project Structure
```

project/
│── src/
│   ├── preprocessing/*
│   ├── modeling/*
│   ├── main.py
│── dataset/
│── document/
│── notebooks/
│── outputs/
│── requirements.txt
└── README.md

````

---

## 🛠️ Set Up Environment
```bash
git clone https://github.com/LQB464/house-prices-advanced-regression-techniques.git
cd house-prices-advanced-regression-techniques
pip install -r requirements.txt
````

---

## Run Full Pipeline

```bash
python src/main.py
```

Optional arguments:

```
--top-k 5            # number of top models selected before tuning
--trials-model 30    # Optuna trials for model tuning
--trials-stack 40    # Optuna trials for stacking
--cv-splits 5        # cross-validation folds
--data-path dataset/train.csv
```

---

## Summary

This repository provides:

* A clean modular ML workflow
* Strong baseline models
* Automated hyperparameter tuning
* A powerful stacking ensemble
* Ready-to-submit Kaggle predictions

The pipeline is designed for reliability, expandability, and high-quality performance on tabular regression tasks.

---


