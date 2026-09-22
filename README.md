# Student Performance Analytics & Early Academic Risk Identification

A Machine Learning-based analytics platform that predicts student academic risk and helps educators identify struggling students before major examinations.

---

##  Project Overview

Student success is a key challenge in educational institutions. Traditional monitoring methods often identify struggling students only after poor academic performance has occurred.

This project leverages Machine Learning and Learning Analytics to:

- Predict Mid-Term and Final Exam risk levels
- Identify academically at-risk students early
- Support timely intervention strategies
- Provide actionable insights through interactive dashboards

---

##  Objectives

- Detect academic risk at an early stage
- Improve student retention and success rates
- Assist educators in decision-making
- Provide explainable performance analytics

---

##  Features

### Student Risk Prediction
- Mid-Term (Mid-2) Risk Prediction
- Final Exam Risk Prediction
- Risk Probability Estimation

### Learning Analytics
- Learning Consistency Index (LCI)
- Learning Resilience Index (LRI)
- Performance Trend Analysis

### Interactive Dashboard
- Student-wise Performance Analysis
- Risk Categorization
- Feature Contribution Insights
- Batch Student Evaluation

### Reporting
- PDF Report Generation
- Downloadable Analysis Results

---

##  Machine Learning Workflow

### Stage 1: Mid-Term Risk Prediction

Inputs:
- CAT1 Score
- Quiz1 Score
- Learning Consistency Index (LCI)

Output:
- Mid-Term Risk Category
- Failure Probability

### Stage 2: Final Exam Risk Prediction

Inputs:
- Mid2 Score
- Quiz2 Score
- Quiz3 Score
- Attendance Percentage
- Learning Resilience Index (LRI)

Output:
- Final Exam Risk Category
- Failure Probability

---

##  Custom Academic Metrics

### Learning Consistency Index (LCI)

Measures consistency during the early learning phase.

LCI =

0.55 × (CAT1 / 50)

+ 0.35 × (QUIZ1 / 20)

+ 0.10 × (1 − |CAT1/50 − QUIZ1/20|)

### Learning Resilience Index (LRI)

Measures a student's ability to sustain performance throughout the course based on:

- Mid-Term Performance
- Quiz Performance
- Attendance
- Predicted Final Exam Risk

---

##  Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Dashboard | Streamlit |
| Machine Learning | XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Storage | Joblib |
| Reporting | ReportLab |

---

## Project Structure

```text
Student_Performance_Analytics/
│
├── app.py
├── models.py
├── utils.py
├── train_model.py
├── data_preprocessing.py
├── background.png
├── requirements.txt
├── README.md
```

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Student-Performance-Analytics.git

cd Student-Performance-Analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Dashboard Outputs

- Student Risk Level
- Failure Probability
- Learning Consistency Index
- Learning Resilience Index
- Performance Insights
- Top Contributing Factors
- Batch Analysis Reports

---

## Business Impact

This system helps educational institutions:

- Identify struggling students early
- Improve academic outcomes
- Reduce student failure rates
- Enable data-driven interventions
- Enhance student retention

---

## Application Screenshots

### Dashboard Home
(Add Screenshot)

### Student Risk Analysis
(Add Screenshot)

### LCI & LRI Analytics
(Add Screenshot)

### Batch Analysis Dashboard
(Add Screenshot)

---

## Future Enhancements

- Real-Time Student Monitoring
- Semester-Wise Forecasting
- Faculty Performance Analytics
- Automated Intervention Suggestions
- Multi-Institution Support

---

## 👩‍💻 Author

**Prathipati Sravani**

M.Sc Data Science

Skills:
Python • Machine Learning • Data Analytics • Streamlit • XGBoost • Data Visualization

---

⭐ If you found this project useful, consider giving it a star