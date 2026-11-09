# CareerPath Analytics: Employee Career Progression & Workforce Planning

## 1. Abstract
CareerPath Analytics uses data analytics and an educational machine-learning demonstration to explore workforce structure, career-related indicators, training, compensation, and attrition. The dashboard presents interactive group-level analysis to support career-development discussions and workforce planning.

## 2. Problem Statement
Analyze career progression data to support informed career guidance and workforce planning.

## 3. Objectives
- Understand workforce distribution by department and job level.
- Explore career indicators such as experience, years in the current role, and years since last promotion.
- Compare training frequency and compensation across workforce groups.
- Explore attrition patterns.
- Train and evaluate classification models for the dataset's attrition label.
- Present group-level insights and responsible recommendations through a dashboard.

## 4. Dataset
Source: IBM HR Analytics Employee Attrition & Performance dataset on Kaggle.
Dataset is fictional/synthetic and intended for educational analysis. Record the exact filename and row/column count from the downloaded file.

## 5. Tools and Technologies
Python, Pandas, scikit-learn, Plotly, Streamlit, Joblib, VS Code, CSV.

## 6. Methodology
1. Load and validate the CSV.
2. Inspect data types, missing values, and unique values.
3. Clean target labels and prepare features.
4. Exclude identifiers and selected demographic attributes from the ML demonstration.
5. Split into training and test sets using stratification.
6. Train Logistic Regression and Random Forest pipelines.
7. Evaluate using accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
8. Build dashboard pages for overview, career progression, workforce planning, ML evaluation, and limitations.
9. Translate aggregate patterns into cautious recommendations.

## 7. Results
Complete this section only after running the project. Record:
- Number of records loaded:
- Number of features used by the ML model:
- Selected model:
- Logistic Regression metrics:
- Random Forest metrics:
- Three notable career/workforce patterns from the dashboard:
- Screenshots included:

Do not invent or pre-fill model scores or findings.

## 8. Recommendations
Based on observed group-level patterns, recommend transparent career-development conversations, equitable training access, workforce distribution reviews, and further investigation of attrition patterns.

## 9. Limitations
The data is synthetic and cross-sectional. It does not provide chronological promotion histories, complete skill inventories, or verified promotion outcomes. Attrition is the ML target, not promotion. Results do not establish causality and should not be used for real employment decisions.

## 10. Conclusion
CareerPath Analytics demonstrates how data analysis, interactive visualization, and model evaluation can support exploratory workforce analysis. Future work could use real, properly governed longitudinal data, validated skills inventories, and careful evaluation before making any real-world claims.
