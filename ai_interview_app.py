import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Define interview questions
questions = [
    "Tell me about yourself.",
    "What are your strengths and weaknesses?",
    "Why should we hire you?",
    "Tell me about a challenge you overcame.",
    "Where do you see yourself in 5 years?"
]

# Keyword mapping for each question (simple version)
keywords_map = {
    "Tell me about yourself.": ["experience", "background", "skills", "education"],
    "What are your strengths and weaknesses?": ["strength", "weakness", "improve", "adapt"],
    "Why should we hire you?": ["fit", "skills", "value", "company"],
    "Tell me about a challenge you overcame.": ["challenge", "problem", "solution", "learned"],
    "Where do you see yourself in 5 years?": ["career", "goal", "future", "growth"]
}

# Expanded dummy training data for ML model
training_data = [
    {"question": "Tell me about yourself.", "answer": "I have experience in data analysis using Python and Excel", "score": 3},
    {"question": "Tell me about yourself.", "answer": "I'm good", "score": 1},
    {"question": "Tell me about yourself.", "answer": "I studied electronics and took a minor in data science", "score": 3},
    {"question": "Tell me about yourself.", "answer": "Love music and watching movies", "score": 0},
    {"question": "Tell me about yourself.", "answer": "I recently graduated with a degree in data science, and I enjoy working with data to solve complex problems.", "score": 4},
    {"question": "Tell me about yourself.", "answer": "I'm a quick learner and eager to contribute.", "score": 2},
    {"question": "Tell me about yourself.", "answer": "I like sports.", "score": 0},
    {"question": "What are your strengths and weaknesses?", "answer": "Problem-solving, attention to detail, and teamwork", "score": 4},
    {"question": "What are your strengths and weaknesses?", "answer": "Python", "score": 1},
    {"question": "What are your strengths and weaknesses?", "answer": "I can lead teams and manage time well", "score": 3},
    {"question": "What are your strengths and weaknesses?", "answer": "I am detail-oriented and work well under pressure.", "score": 4},
    {"question": "What are your strengths and weaknesses?", "answer": "I am friendly.", "score": 1},
    {"question": "What are your strengths and weaknesses?", "answer": "Sometimes I overthink problems, but I am working on it.", "score": 3},
    {"question": "What are your strengths and weaknesses?", "answer": "I am bad at everything.", "score": 0},
    {"question": "Why should we hire you?", "answer": "Because I'm passionate about analytics and want to solve real problems", "score": 4},
    {"question": "Why should we hire you?", "answer": "Job is good", "score": 1},
    {"question": "Why should we hire you?", "answer": "I want to apply my skills in data visualization and machine learning", "score": 3},
    {"question": "Why should we hire you?", "answer": "I am excited to apply my knowledge of machine learning and analytics to help business growth.", "score": 4},
    {"question": "Why should we hire you?", "answer": "Because I want a job.", "score": 0},
    {"question": "Tell me about a challenge you overcame.", "answer": "I solved a data quality problem by creating automated scripts.", "score": 4},
    {"question": "Tell me about a challenge you overcame.", "answer": "Had a problem, fixed it.", "score": 1},
    {"question": "Where do you see yourself in 5 years?", "answer": "I aim to grow into a data science leader in the industry.", "score": 4},
    {"question": "Where do you see yourself in 5 years?", "answer": "Not sure, just working.", "score": 1},
]

# Train the ML model
def train_answer_scorer(data):
    texts = [f"{d['question']} {d['answer']}" for d in data]
    scores = [d["score"] for d in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, scores)
    return vectorizer, clf

vectorizer, score_model = train_answer_scorer(training_data)

# Keyword scoring function
def keyword_score(answer, keywords):
    words = answer.lower().split()
    matched = sum(1 for kw in keywords if kw.lower() in words)
    return min(matched, 4)  # max score 4

# --- UI Polish Start ---

st.title("🧠 AI Interview Readiness Analyzer")

st.markdown("""
Welcome! This app helps you prepare for common interview questions by scoring your answers based on relevant keywords and a trained ML model.  
**How to use:**  
1. Select a question from the dropdown.  
2. Write your answer in the text area.  
3. Set how confident you feel about your answer.  
4. Click Submit to get scored feedback and track your progress!  
""")

if "responses" not in st.session_state:
    st.session_state.responses = []

selected_question = st.selectbox("Choose a question:", questions)

keyword_hints = keywords_map.get(selected_question, [])
st.info(f"💡 Suggested Keywords: {', '.join(keyword_hints)}")

user_answer = st.text_area("Your answer:", height=200)

confidence = st.slider("Confidence level (1 to 5):",
                       min_value=1, max_value=5,
                       help="How confident are you about this answer?")

if st.button("Submit"):
    input_text = f"{selected_question} {user_answer}"
    X_test = vectorizer.transform([input_text])
    ml_score = int(score_model.predict(X_test)[0])
    kw_score = keyword_score(user_answer, keyword_hints)
    final_score = int(round((ml_score + kw_score) / 2))

    st.session_state.responses.append({
        "question": selected_question,
        "answer": user_answer,
        "confidence": confidence,
        "ml_score": ml_score,
        "keyword_score": kw_score,
        "score": final_score
    })

    st.success(f"Answer saved with hybrid score: {final_score}/4 (ML: {ml_score}, Keywords: {kw_score})")

st.markdown("---")

if st.session_state.responses:
    st.write("### 💾 Saved Responses:")
    for idx, r in enumerate(st.session_state.responses):
        st.write(f"**{idx+1}. {r['question']}**")
        st.write(f"Answer: {r['answer']}")
        st.write(f"Confidence: {r['confidence']}/5 | Score: {r['score']}/4 (ML: {r['ml_score']}, Keywords: {r['keyword_score']})")
        if r['score'] < 2:
            st.warning("🔍 Feedback: Consider adding more relevant points to strengthen this answer.")
        st.markdown("---")

if st.session_state.responses:
    st.write("### 📊 Confidence vs Score Bar Chart")
    df = pd.DataFrame(st.session_state.responses)
    df["Entry"] = [f"Ans {i+1}" for i in range(len(df))]
    melted_df = df.melt(id_vars=["Entry"], value_vars=["confidence", "score"],
                        var_name="Metric", value_name="Value")
    fig = px.bar(melted_df, x="Entry", y="Value", color="Metric",
                 barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

if st.button("Export to CSV"):
    if st.session_state.responses:
        df = pd.DataFrame(st.session_state.responses)
        df.to_csv("interview_responses.csv", index=False)
        st.success("✅ Exported to 'interview_responses.csv'")
    else:
        st.warning("No responses to export yet.")

if st.session_state.responses:
    st.write("### 📈 Confidence vs Score Line Chart")
    df = pd.DataFrame(st.session_state.responses)
    df["Entry"] = [f"Ans {i+1}" for i in range(len(df))]
    chart_data = df[["Entry", "confidence", "score"]].set_index("Entry")
    st.line_chart(chart_data)

# --- UI Polish End ---
