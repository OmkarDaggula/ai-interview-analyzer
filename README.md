# 🧠 AI Interview Readiness Analyzer

An interactive Streamlit web app that helps users prepare for common interview questions by analyzing their answers using a hybrid approach—rule-based keyword scoring and machine learning.

🚀 Live App: [Click to try it](https://aiinterviewapppy-cvcmpghgjzjvwqeb46bwfu.streamlit.app/)


## 🚀 Features

- ✅ 5 commonly asked interview questions
- 💡 Suggested keywords to guide your answers
- 🤖 Hybrid scoring system using:
  - Rule-based keyword matching
  - ML-based scoring (trained Random Forest classifier)
- 📊 Visual feedback (confidence vs score) using Plotly
- 💾 Save and review all submitted responses in one session
- 📤 Export all answers to a CSV file for offline use
- 📈 Line chart to track confidence vs performance trends
- 🧹 Clean, user-friendly UI built with Streamlit

## 🎯 How It Works

1. **Select a question** from a dropdown list.
2. **Write your answer** in the input box.
3. **Set your confidence level** using a slider (1–5).
4. **Click Submit** to get immediate scoring and feedback.
5. View **visual insights** from bar and line charts.
6. **Export your data** to a CSV file anytime.

## 📸 Screenshots

> (Insert screenshots here once you take them using the app)

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🌐
- Scikit-learn 🤖
- Pandas 📊
- Plotly 📈


## 🧠 ML Model Details

- Model: Random Forest Classifier
- Features: TF-IDF vectors of combined (question + answer) text
- Target: Human-labeled scores from 0–4
- Additional scoring: Keyword-based rule logic to reinforce quality metrics

## 📦 Installation

```bash
pip install streamlit scikit-learn pandas plotly

## ▶️ Run the App
streamlit run interview_app.py

## ✨ Future Enhancements
Add feedback suggestions for improvements

Support more question types (technical, behavioral)

Allow uploading custom questions






