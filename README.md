# CareerPath Analytics

CareerPath Analytics is a data analytics and machine learning project developed as part of the HCL Data Analytics program.

The project analyzes employee and career-related data to support career guidance, workforce planning, and employee attrition analysis.

## Features

1. **Executive Overview** – Key workforce metrics and dataset quality indicators.
2. **Career Progression** – Explore filtered career indicators; the static data does not establish promotion history.
3. **Workforce Planning** – Analyze workforce distribution and trends.
4. **ML Attrition Lab** – Compare attrition models with evaluation metrics, permutation importance, and hypothetical-profile explanations using SHAP and LIME.
5. **Recommendations & Limitations** – Review filtered workforce observations, suggested follow-ups, and dataset/model limitations.
6. **Career Assistant** – Get rule-based analytical answers that respect the active filters (not a live AI chatbot).

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Joblib
- SHAP
- LIME

## Machine Learning

The project compares:
- Logistic Regression
- Random Forest

The model training process uses grouped splitting to prevent identical employee profiles from appearing in both training and testing sets.

## Dataset Information

The project uses an IBM HR Analytics-style employee dataset.

The repository includes `hr_data.csv` and the trained model artifacts. The dataset contains resampled employee profiles; repeated profiles are not independent observations. Model evaluation results are educational and should not be interpreted as proof of real-world predictive performance.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/23pa1a4245/CareerPath-Analytics-XAI.git
cd CareerPath-Analytics-XAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Use Python 3.11 for the broadest compatibility with the SHAP and LIME dependencies.
The `requirements.txt` file includes both packages.

The included model artifacts are ready to use. To retrain them after changing the dataset:

```bash
python train_model.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Explainable AI (SHAP and LIME)

The **ML Attrition Lab** includes global SHAP feature importance and a SHAP
summary/beeswarm plot, plus individual SHAP and LIME explanations for a
selected profile. These explain the saved Random Forest model and its
preprocessing pipeline; they do not retrain the model or change the saved
evaluation metrics. Explanations use the current dashboard filters.

To generate reports from the terminal:

```bash
python shap_analysis.py
python lime_analysis.py
```

To explain a particular `EmployeeNumber`:

```bash
python shap_analysis.py --employee-number 1
python lime_analysis.py --employee-number 1
```

Replace `1` with an ID present in your dataset. Reports are written under
`reports/shap/` and `reports/lime/` respectively. The SHAP report contains
global importance and beeswarm plots, per-profile waterfall plots, and CSV/JSON
contributions. The LIME report contains local contribution plots and
CSV/JSON/HTML explanations.

Positive SHAP values increase the model's attrition probability and negative
values decrease it. LIME contributions are for the predicted class, so positive
values support that class and negative values oppose it. These are explanations
of model behavior, not causal findings or employee-level recommendations.
The data is static and contains resampled profiles; do not use predictions or
explanations for employment decisions.

### GitHub and Streamlit Community Cloud

After reviewing your changes, add and push the XAI files and dependency list
to your project repository:

```bash
git add app.py shap_analysis.py lime_analysis.py requirements.txt README.md reports/shap reports/lime
git commit -m "Add SHAP and LIME attrition explanations"
git push
```

To deploy, sign in at [Streamlit Community Cloud](https://share.streamlit.io/), choose
**Create app > Deploy a public app from GitHub**, and select
`23pa1a4245/CareerPath-Analytics-XAI`, branch `main`, and entry point `app.py`.
Cloud installs SHAP and LIME from `requirements.txt`. Push future changes to `main`
to trigger a redeploy.

## Project Structure

```text
CareerPath-Analytics/
│
├── app.py
├── train_model.py
├── shap_analysis.py
├── lime_analysis.py
├── check_data_quality.py
├── expand_dataset.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT_TEMPLATE.md
├── UPGRADE_STEPS.txt
├── VIVA_QUESTIONS.md
├── reports/
│   ├── shap/
│   └── lime/
└── models/
```


## Live Project Deployment

**Live Application:**  
https://careerpath-analytics-starter-dyewd7crik5hdjukxrgidl.streamlit.app/

## Project Repository

**GitHub Repository:**  
https://github.com/23pa1a4245/careerpathanalysis

