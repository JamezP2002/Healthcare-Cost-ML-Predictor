# Predicting Healthcare Costs with Machine Learning

A data science project that explores drivers of healthcare costs and builds a model to predict medical charges using patient attributes.

**Live Demo:** [Streamlit App](https://healthcare-cost-ml-predictor-interface.streamlit.app/)

## Tech Stack
- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**, **SHAP**
- **Streamlit** – interactive app 
- Jupyter Notebooks for data exploration and modeling

## Getting Started
1. Clone the repo
2. Create a virtual environment
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
4. Look around the notebooks or play around with the streamlit app!

## Key Findings from EDA

- Smokers have significantly higher medical costs compared to non-smokers.
- Age has a moderate positive correlation with charges.
- High BMI increases charges mostly among smokers.
- Children and region don’t seem to have much impact.
- Medical charges are heavily skewed, with a few very high-cost outliers.

## Project Progress

- [x] Dataset loaded and cleaned
- [x] Exploratory data analysis (EDA)
- [x] Feature engineering
- [x] Model building & evaluation (linear regression, random forest, xgboost)
- [x] Streamlit app (live demo in the cloud)

## Notebooks

- [`01_eda.ipynb`](./notebooks/01_eda.ipynb) – Exploratory data analysis
- [`02_preprocessing.ipynb`](./notebooks/02_preprocessing.ipynb) – Preprocessing work
- [`03_modeling.ipynb`](./notebooks/03_modeling.ipynb) – Modeling work

## Model Performance Summary

We trained and compared three different models to predict medical charges:

| Model              | R² Score | MAE      | RMSE     |
|-------------------|----------|----------|----------|
| Linear Regression | 0.78     | $4,176   | $5,794   |
| Random Forest     | 0.86     | $2,719   | $4,704   |
| XGBoost           | 0.86     | **$2,665** | **$4,682** |

### Key Takeaways:
- **Linear Regression** gave a good starting point but couldn’t capture complex relationships in the data.
- **Random Forest** improved performance significantly and captured non-linear patterns better.
- **XGBoost** slightly outperformed Random Forest and gave the most accurate predictions overall.

**Final Model Choice:** **XGBoost** was selected as the final model due to its strong performance on all metrics.

## Streamlit application
The app lets users:
- Input age, sex, BMI, region, smoker status, and children
- Predict their **estimated annual insurance charges**
- View a SHAP **waterfall chart** explaining the prediction
- Compare their charge to the **average patient cost**

 ## Credits
 - Inspired by the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
