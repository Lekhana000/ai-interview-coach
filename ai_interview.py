import streamlit as st
import json
import os
import re
import tempfile
import speech_recognition as sr
from pypdf import PdfReader
from google import genai

# ============================================
# GEMINI API SETUP
# ============================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"⚠️ API Key Error: {e}")
    st.info("Check `.streamlit/secrets.toml` has: GEMINI_API_KEY = \"your_key\"")
    st.stop()

# ============================================
# USER MANAGEMENT
# ============================================

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(username, password, name):
    users = load_users()
    if username in users:
        return False, "❌ Username already exists!"
    users[username] = {"password": password, "name": name}
    save_users(users)
    return True, "✅ Registration successful! Please login."

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "❌ Username not found!"
    if users[username]["password"] != password:
        return False, "❌ Incorrect password!"
    return True, "✅ Login successful!"

# ============================================
# RESUME & AI FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def generate_questions(resume_text):
    prompt = f"""You are an expert interviewer. Based on the following resume, generate EXACTLY 10 interview questions.

Mix:
- 2 HR questions
- 3 Technical questions
- 2 Project questions
- 2 Skills questions
- 1 Education/Experience question

Return ONLY the numbered questions 1 to 10. No extra text.

RESUME:
{resume_text}

10 Questions:"""
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        questions = []
        for line in response.text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            clean = re.sub(r'^(Q?\d+[\.\)\:]\s*)', '', line).strip()
            if clean and len(clean) > 10:
                questions.append(clean)
        return questions[:10]
    except Exception as e:
        return [f"ERROR: {str(e)}"]

def evaluate_answers(questions, answers):
    qa_block = ""
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        qa_block += f"Q{i}: {q}\nAnswer {i}: {a}\n\n"

    prompt = f"""You are an expert interview evaluator. Evaluate each of the following interview answers.

For EACH question, provide:
- Score: X/10
- Strengths: (1-2 short sentences)
- Weaknesses: (1-2 short sentences)
- Suggested: (a better answer, 1-2 sentences)

Use EXACTLY this format:

Q1:
Score: 7/10
Strengths: Good explanation.
Weaknesses: Missing details.
Suggested: Add specific examples.

Q2:
Score: 6/10
Strengths: ...
Weaknesses: ...
Suggested: ...

(Continue for all 10 questions)

Interview Q&A:
{qa_block}

Now evaluate:"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def parse_evaluations(text):
    evaluations = []
    blocks = re.split(r'\nQ\d+:\s*\n?', "\n" + text)
    blocks = [b for b in blocks if b.strip()]

    for block in blocks:
        score = "N/A"
        strengths = "N/A"
        weaknesses = "N/A"
        suggested = "N/A"
        for line in block.split('\n'):
            line = line.strip()
            if line.lower().startswith("score:"):
                score = line.split(":", 1)[1].strip()
            elif line.lower().startswith("strengths:"):
                strengths = line.split(":", 1)[1].strip()
            elif line.lower().startswith("weaknesses:"):
                weaknesses = line.split(":", 1)[1].strip()
            elif line.lower().startswith("suggested:"):
                suggested = line.split(":", 1)[1].strip()
        if score != "N/A":
            evaluations.append({
                "score": score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "suggested": suggested
            })
    return evaluations

def calculate_overall_score(evaluations):
    total = 0
    count = 0
    for ev in evaluations:
        match = re.search(r'(\d+(?:\.\d+)?)', ev["score"])
        if match:
            total += float(match.group(1))
            count += 1
    if count == 0:
        return 0.0
    return round(total / count, 1)

def get_remark(score):
    if score >= 8:
        return "🌟 Excellent! You're well-prepared for interviews."
    elif score >= 6:
        return "👏 Very Good! A little polish will make you great."
    elif score >= 4:
        return "👍 Good! Keep practicing to improve."
    else:
        return "📚 Needs Improvement. Practice more and review fundamentals."

# ============================================
# VOICE → TEXT (Automatic Transcription)
# ============================================

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        os.unlink(tmp_path)
        return text

    except sr.UnknownValueError:
        return "ERROR: Could not understand audio. Please speak clearly and try again."
    except sr.RequestError as e:
        return f"ERROR: Speech service unavailable - {e}"
    except Exception as e:
        return f"ERROR: {str(e)}"

# ============================================
# MAIN APP
# ============================================

def main():
    st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="wide")

    defaults = {
        "logged_in": False,
        "username": "",
        "user_name": "",
        "choice": None,
        "resume_text": "",
        "questions": [],
        "questions_generated": False,
        "current_question": 0,
        "answers": [],
        "interview_started": False,
        "interview_completed": False,
        "evaluation_text": "",
        "evaluations": [],
        "overall_score": 0.0,
        "evaluation_started": False,
        "evaluation_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ---------- LOGIN / REGISTER ----------
    if not st.session_state.logged_in:
        st.title("🎯 AI Interview Coach")
        st.markdown("---")
        st.subheader("Welcome! Please choose an option to continue.")

        if st.session_state.choice is None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Login", use_container_width=True, type="primary"):
                    st.session_state.choice = "login"
                    st.rerun()
            with col2:
                if st.button("📝 Register", use_container_width=True, type="primary"):
                    st.session_state.choice = "register"
                    st.rerun()

        elif st.session_state.choice == "login":
            st.markdown("### 🔐 Login to your account")
            if st.button("← Back to options"):
                st.session_state.choice = None
                st.rerun()
            st.markdown("---")
            u = st.text_input("Username", key="login_username")
            p = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", type="primary", use_container_width=True):
                if not u or not p:
                    st.warning("⚠️ Enter username and password!")
                else:
                    ok, msg = login_user(u, p)
                    if ok:
                        users = load_users()
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        st.session_state.user_name = users[u]["name"]
                        st.session_state.choice = None
                        st.rerun()
                    else:
                        st.error(msg)

        elif st.session_state.choice == "register":
            st.markdown("### 📝 Create a new account")
            if st.button("← Back to options"):
                st.session_state.choice = None
                st.rerun()
            st.markdown("---")
            name = st.text_input("Full Name", key="reg_name")
            u = st.text_input("Choose a Username", key="reg_username")
            p = st.text_input("Create Password", type="password", key="reg_password")
            pc = st.text_input("Confirm Password", type="password", key="reg_confirm")
            if st.button("Register", type="primary", use_container_width=True):
                if not name or not u or not p or not pc:
                    st.warning("⚠️ Fill all fields!")
                elif p != pc:
                    st.error("❌ Passwords do not match!")
                elif len(p) < 4:
                    st.error("❌ Password must be at least 4 characters!")
                else:
                    ok, msg = register_user(u, p, name)
                    if ok:
                        st.success(msg)
                        st.info("💡 Click '← Back to options' and login.")
                    else:
                        st.error(msg)

        st.stop()

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("👤 Profile")
        st.markdown("---")
        st.markdown("**Welcome,**")
        st.markdown(f"### {st.session_state.user_name}")
        st.markdown(f"*@{st.session_state.username}*")
        st.markdown("---")

        if st.session_state.interview_started and not st.session_state.interview_completed:
            idx = st.session_state.current_question
            total = len(st.session_state.questions)
            if total > 0:
                st.markdown("### 📊 Progress")
                st.progress(idx / total)
                st.write(f"**Question {idx + 1} of {total}**")

        st.markdown("---")
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ---------- MAIN ----------
    st.title("🎯 AI Interview Coach")
    st.markdown("---")

    # ===== STEP 1: RESUME UPLOAD =====
    if not st.session_state.resume_text:
        st.header("📄: Upload Your Resume")
        st.write("Please upload your resume in PDF format.")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded is not None:
            with st.spinner("📖 Extracting text..."):
                text = extract_text_from_pdf(uploaded)
            if text.startswith("ERROR:"):
                st.error(f"❌ Failed to read PDF: {text}")
            elif len(text) < 50:
                st.error("❌ Not enough text extracted. Use a PDF with selectable text.")
            else:
                st.session_state.resume_text = text
                st.success("✅ Resume uploaded successfully!")
                st.rerun()
        st.stop()

    # ===== STEP 2 & 3: PREVIEW + GENERATE =====
    if not st.session_state.questions_generated:
        st.header("📋 : Resume Preview")
        with st.expander("👁️ View Extracted Resume Text", expanded=False):
            st.text_area("Resume", st.session_state.resume_text, height=300, disabled=True, label_visibility="collapsed")
        st.markdown("---")

        st.header("🎤 : Generate Interview Questions")
        st.write("Click below to generate  questions.")
        st.info("⏳ This may take 10–20 seconds.")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is generating questions..."):
                    qs = generate_questions(st.session_state.resume_text)
                if len(qs) > 0 and str(qs[0]).startswith("ERROR:"):
                    st.error(f"❌ {qs[0]}")
                    st.info("Please try again in a minute.")
                else:
                    st.session_state.questions = qs
                    st.session_state.questions_generated = True
                    st.success(f"✅ Generated {len(qs)} questions!")
                    st.rerun()
        with col2:
            if st.button("🔄 Upload Different Resume"):
                st.session_state.resume_text = ""
                st.session_state.questions = []
                st.session_state.questions_generated = False
                st.rerun()
        st.stop()

    # ===== STEP 4: READY TO START =====
    if not st.session_state.interview_started:
        st.header("✅ Questions Ready!")
        st.success(f"✅ {len(st.session_state.questions)} questions generated from your resume.")

        with st.expander("👁️ Preview Questions", expanded=False):
            for i, q in enumerate(st.session_state.questions, 1):
                st.markdown(f"**Q{i}.** {q}")

        st.markdown("---")
        st.subheader("🎬 Ready to Start the Interview?")
        st.write("You will be asked **10 questions one at a time**.")
        st.write("Answer by **⌨️ typing** or **🎤 speaking** (auto-transcribed).")
        st.info("⚡ After Q10, AI will automatically evaluate all your answers!")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Start Interview", type="primary", use_container_width=True):
                st.session_state.interview_started = True
                st.session_state.current_question = 0
                st.session_state.answers = [""] * len(st.session_state.questions)
                st.session_state.interview_completed = False
                st.session_state.evaluation_started = False
                st.session_state.evaluation_done = False
                st.rerun()
        with col2:
            if st.button("🔄 Regenerate Questions", use_container_width=True):
                st.session_state.questions = []
                st.session_state.questions_generated = False
                st.rerun()
        st.stop()

    # ===== STEP 5: ANSWERING =====
    if st.session_state.interview_started and not st.session_state.interview_completed:
        total_q = len(st.session_state.questions)
        idx = st.session_state.current_question

        if idx >= total_q:
            st.session_state.interview_completed = True
            st.rerun()

        st.markdown(f"### 📝 Question {idx + 1} of {total_q}")
        st.progress((idx + 1) / total_q)
        st.markdown("---")

        st.markdown("#### 🎤 Question:")
        st.info(st.session_state.questions[idx])
        st.markdown("---")

        st.markdown("#### ✍️ Your Answer:")
        input_method = st.radio(
            "Choose how to answer:",
            ["⌨️ Type Answer", "🎤 Speak Answer (Voice)"],
            horizontal=True,
            key=f"input_method_{idx}"
        )

        # ---- TYPE ANSWER ----
        if input_method == "⌨️ Type Answer":
            current_answer = st.text_area(
                "Type your answer below:",
                value=st.session_state.answers[idx] if idx < len(st.session_state.answers) else "",
                height=150,
                key=f"typed_answer_{idx}",
                placeholder="Type your answer here..."
            )
            if idx < len(st.session_state.answers):
                st.session_state.answers[idx] = current_answer

        # ---- VOICE ANSWER ----
        else:
            st.info("🎤 Press the microphone button below, speak your answer, then press Stop.")
            st.caption("Your speech will be automatically converted to text.")

            audio_value = st.audio_input("🎙️ Press microphone to start recording", key=f"audio_{idx}")

            if audio_value is not None:
                st.audio(audio_value)

                with st.spinner("🔄 Transcribing your voice..."):
                    transcribed_text = transcribe_audio(audio_value)

                if transcribed_text.startswith("ERROR:"):
                    st.error(f"❌ {transcribed_text}")
                    st.warning("You can still type your answer manually below.")
                    current_answer = st.text_area(
                        "📝 Type your answer manually:",
                        value=st.session_state.answers[idx] if idx < len(st.session_state.answers) else "",
                        height=150,
                        key=f"manual_{idx}"
                    )
                    if idx < len(st.session_state.answers):
                        st.session_state.answers[idx] = current_answer
                else:
                    st.success("✅ Voice transcribed successfully!")
                    st.markdown("**📝 Transcribed Text (editable):**")
                    current_answer = st.text_area(
                        "You can edit if needed:",
                        value=transcribed_text,
                        height=150,
                        key=f"voice_answer_{idx}",
                        label_visibility="collapsed"
                    )
                    if idx < len(st.session_state.answers):
                        st.session_state.answers[idx] = current_answer
            else:
                if idx < len(st.session_state.answers) and st.session_state.answers[idx].strip():
                    st.markdown("**📝 Your Saved Answer:**")
                    st.info(st.session_state.answers[idx])
                st.caption("⏳ Waiting for you to press the microphone...")

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if idx > 0:
                if st.button("⬅️ Previous Question", use_container_width=True):
                    st.session_state.current_question -= 1
                    st.rerun()
        with col2:
            st.write("")
        with col3:
            current_text = st.session_state.answers[idx] if idx < len(st.session_state.answers) else ""
            if idx < total_q - 1:
                if st.button("Next Question ➡️", type="primary", use_container_width=True):
                    if not current_text.strip():
                        st.warning("⚠️ Please record or type your answer first!")
                    else:
                        st.session_state.current_question += 1
                        st.rerun()
            else:
                # LAST QUESTION → Finish → AUTO-EVALUATE
                if st.button("✅ Finish & Evaluate", type="primary", use_container_width=True):
                    if not current_text.strip():
                        st.warning("⚠️ Please record or type your answer first!")
                    else:
                        st.session_state.interview_completed = True
                        st.session_state.evaluation_started = True
                        st.rerun()
        st.stop()

    # ===== STEP 6: AUTO-EVALUATION =====
    if st.session_state.interview_completed and not st.session_state.evaluation_done:
        st.success("🎉 Interview Completed! All 10 answers submitted.")
        st.markdown("---")
        st.subheader("🤖 AI is Evaluating Your Answers...")
        st.info("⏳ Please wait — this makes ONE API call and may take 15–30 seconds.")

        # Auto-trigger evaluation
        progress_placeholder = st.empty()

        with progress_placeholder.container():
            with st.spinner("🧠 AI is analyzing your responses..."):
                eval_text = evaluate_answers(st.session_state.questions, st.session_state.answers)

        if eval_text.startswith("ERROR:"):
            st.error(f"❌ {eval_text}")
            st.info("Please wait 60 seconds and try again.")
            if st.button("🔄 Retry Evaluation", type="primary"):
                st.rerun()
        else:
            st.session_state.evaluation_text = eval_text
            parsed = parse_evaluations(eval_text)
            st.session_state.evaluations = parsed
            st.session_state.overall_score = calculate_overall_score(parsed)
            st.session_state.evaluation_done = True
            st.rerun()
        st.stop()

    # ===== STEP 7: FINAL REPORT =====
    if st.session_state.evaluation_done:
        st.success("🎉 Interview Completed!")
        st.markdown("---")

        st.header("📊 Final Result")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Overall Score", f"{st.session_state.overall_score}/10")
        with col2:
            st.metric("Questions Answered", f"{len(st.session_state.answers)}/10")
        with col3:
            st.metric("Remark", "✅ Done")

        st.markdown("---")
        st.subheader("🏆 Overall Remarks")
        st.info(get_remark(st.session_state.overall_score))

        st.markdown("---")
        st.header("📋 Detailed Evaluation")

        evaluations = st.session_state.evaluations
        for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers), 1):
            ev = evaluations[i - 1] if i - 1 < len(evaluations) else None

            with st.expander(f"**Q{i}.** {q[:80]}...", expanded=(i == 1)):
                st.markdown(f"### 🎤 Question {i}")
                st.markdown(f"**{q}**")
                st.markdown("---")

                st.markdown("#### ✍️ Your Answer")
                st.write(a if a.strip() else "*No answer provided*")
                st.markdown("---")

                if ev:
                    st.markdown("#### 🤖 AI Evaluation")
                    st.metric("Score", ev["score"])
                    st.markdown("**✅ Strengths:**")
                    st.success(ev["strengths"])
                    st.markdown("**⚠️ Weaknesses:**")
                    st.warning(ev["weaknesses"])
                    st.markdown("**💡 Suggested Better Answer:**")
                    st.info(ev["suggested"])
                else:
                    st.warning("⚠️ Evaluation not available for this question.")

        st.markdown("---")

        st.subheader("📥 Export Your Report")
        report_text = f"""AI INTERVIEW COACH - FINAL REPORT
=====================================
Candidate: {st.session_state.user_name}
Username: @{st.session_state.username}
Overall Score: {st.session_state.overall_score}/10
Remark: {get_remark(st.session_state.overall_score)}
=====================================
"""
        for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers), 1):
            ev = evaluations[i - 1] if i - 1 < len(evaluations) else None
            report_text += f"\n\nQ{i}: {q}\nYour Answer: {a}\n"
            if ev:
                report_text += f"Score: {ev['score']}\nStrengths: {ev['strengths']}\nWeaknesses: {ev['weaknesses']}\nSuggested: {ev['suggested']}\n"

        st.download_button(
            label="📄 Download Report (TXT)",
            data=report_text,
            file_name=f"interview_report_{st.session_state.username}.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Take Interview Again", use_container_width=True):
                for key in ["interview_started", "interview_completed", "current_question",
                            "answers", "evaluation_text", "evaluations", "overall_score",
                            "evaluation_started", "evaluation_done"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("📄 Upload New Resume", use_container_width=True):
                for key in ["resume_text", "questions", "questions_generated", "interview_started",
                            "interview_completed", "current_question", "answers", "evaluation_text",
                            "evaluations", "overall_score", "evaluation_started", "evaluation_done"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()