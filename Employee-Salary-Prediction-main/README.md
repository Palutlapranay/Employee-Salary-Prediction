# Employee-Salary-Prediction

An AI-powered web app that predicts employee salaries based on demographic and work-related inputs using a Linear Regression model.

---

## 📦 Features

- Predicts salary based on age, gender, role, education, and experience
- Streamlit-based user interface
- Visual output using salary charts
- Trained ML model using scikit-learn

---

## 🎯 Demo

### Input Parameters
- **Age**: Employee's age (18-65 years)
- **Gender**: Male/Female
- **Job Role**: Software Engineer, Data Scientist, Product Manager, etc.
- **Education Level**: Bachelor's, Master's, PhD
- **Years of Experience**: 0-20 years

### Sample Prediction
```
Input:
- Age: 28
- Gender: Male
- Job Role: Software Engineer
- Education: Master's Degree
- Experience: 5 years

Predicted Salary: $85,000 - $95,000
```

### Features Demo
- **Interactive Form**: Easy-to-use input form
- **Real-time Prediction**: Instant salary predictions
- **Visual Charts**: Salary distribution and comparison charts
- **Responsive Design**: Works on desktop and mobile

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
