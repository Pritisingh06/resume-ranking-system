import os
import re
import sqlite3
from typing import List, Tuple
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder="templete")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max request size
app.secret_key = os.environ.get("SECRET_KEY", "resume-ranker-secret")

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

ALLOWED_EXTENSION = ".pdf"
FALLBACK_SKILLS = [
    "python",
    "sql",
    "machine learning",
    "aws",
    "excel",
    "communication",
]
SKILL_ALIASES = {
    "python": ["python", "py"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "flask": ["flask"],
    "django": ["django"],
    "react": ["react", "reactjs", "react.js"],
    "node": ["node", "nodejs", "node.js"],
    "git": ["git", "github", "gitlab"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "data analysis": ["data analysis", "data analytics", "analytics"],
    "communication": ["communication", "communications", "communicator"],
    "leadership": ["leadership", "team lead", "leading teams"],
    "project management": ["project management", "project manager", "project coordination"],
}
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def init_db() -> None:
    """Create users table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


def extract_text_from_pdf(file_obj) -> str:
    """Extract text from uploaded PDF file object."""
    text_parts: List[str] = []
    try:
        file_obj.stream.seek(0)
        reader = PyPDF2.PdfReader(file_obj.stream)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception:
        return ""
    return " ".join(text_parts).lower().strip()


def normalize_text(text: str) -> str:
    """Normalize casing and spacing for more reliable matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.\s]", " ", text.lower())).strip()


def skill_in_text(skill: str, text: str) -> bool:
    """Case-insensitive skill matching that tolerates PDF spacing noise."""
    normalized_text = normalize_text(text)
    candidate_terms = SKILL_ALIASES.get(skill, [skill])
    for term in candidate_terms:
        normalized_term = normalize_text(term)
        pattern = r"(?<!\w)" + re.escape(normalized_term).replace(r"\ ", r"\s+") + r"(?!\w)"
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            return True
    return False


def extract_skills_from_text(text: str) -> List[str]:
    """Extract known skills mentioned in job description."""
    skills = list(SKILL_ALIASES.keys())
    return [s for s in skills if skill_in_text(s, text)]


def tokenize(text: str) -> set:
    """Simple tokenizer for keyword overlap scoring."""
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}", text.lower()))


def is_valid_email(value: str) -> bool:
    """Basic email validation for auth forms."""
    return bool(EMAIL_REGEX.match(value))


def compute_scores(job_description: str, resume_texts: List[str]) -> Tuple[List[float], List[float]]:
    """Return tf-idf similarity scores and keyword overlap scores."""
    docs = [job_description] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten().tolist()

    jd_tokens = tokenize(job_description)
    overlap_scores: List[float] = []
    for resume_text in resume_texts:
        resume_tokens = tokenize(resume_text)
        overlap = len(jd_tokens & resume_tokens) / max(1, len(jd_tokens))
        overlap_scores.append(overlap)
    return tfidf_scores, overlap_scores


def get_fit_label(score: float) -> str:
    """Human-friendly fit bucket for a candidate score."""
    if score >= 75:
        return "Strong Fit"
    if score >= 50:
        return "Potential Fit"
    return "Needs Improvement"


def build_candidate_summary(
    matched_skills: List[str],
    missing_skills: List[str],
    skill_coverage: float,
    keyword_overlap_score: float,
) -> str:
    """Generate a simple recruiter-style summary for each resume."""
    strengths = []
    if matched_skills:
        strengths.append(f"shows {', '.join(matched_skills[:3])}")
    if skill_coverage >= 0.7:
        strengths.append("covers most required skills")
    elif skill_coverage >= 0.4:
        strengths.append("covers several required skills")

    gaps = []
    if missing_skills:
        gaps.append(f"missing {', '.join(missing_skills[:3])}")
    if keyword_overlap_score < 0.25:
        gaps.append("has limited keyword alignment")

    if strengths and gaps:
        return f"Candidate {strengths[0]}, but is {gaps[0]}."
    if strengths:
        return f"Candidate {strengths[0]} and aligns well with the job description."
    if gaps:
        return f"Candidate is {gaps[0]} and may need resume tailoring."
    return "Candidate has a balanced profile with moderate alignment."


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    results = None
    error = None
    summary = None
    jd_value = ""

    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip().lower()
        jd_value = job_description

        # Support both name="resumes" and name="resumes[]"
        uploaded_files = request.files.getlist("resumes")
        if not uploaded_files:
            uploaded_files = request.files.getlist("resumes[]")

        resume_texts: List[str] = []
        resume_names: List[str] = []
        invalid_files: List[str] = []

        for file in uploaded_files:
            if not file or not file.filename:
                continue
            fname = (file.filename or "").lower()
            if not fname.endswith(ALLOWED_EXTENSION):
                invalid_files.append(file.filename)
                continue
            text = extract_text_from_pdf(file)
            if text:
                resume_texts.append(text)
                resume_names.append(file.filename)
            else:
                invalid_files.append(file.filename)

        if not job_description:
            error = "Please enter a job description."
        elif not resume_texts:
            error = "Please upload at least one valid PDF resume."

        if error:
            return render_template(
                "index.html",
                results=None,
                error=error,
                summary=summary,
                job_description=jd_value,
            )

        skills = extract_skills_from_text(job_description)
        if not skills:
            skills = FALLBACK_SKILLS

        tfidf_scores, keyword_overlap_scores = compute_scores(job_description, resume_texts)

        data = []
        for i in range(len(resume_names)):
            resume_text = resume_texts[i]
            missing = [s for s in skills if not skill_in_text(s, resume_text)]
            matched_skills = [s for s in skills if skill_in_text(s, resume_text)]
            skill_coverage = len(matched_skills) / max(1, len(skills))
            final_score = (
                0.6 * float(tfidf_scores[i]) +
                0.25 * float(skill_coverage) +
                0.15 * float(keyword_overlap_scores[i])
            )
            fit_label = get_fit_label(final_score * 100)
            candidate_summary = build_candidate_summary(
                matched_skills,
                missing,
                skill_coverage,
                keyword_overlap_scores[i],
            )
            data.append({
                "resume": resume_names[i],
                "match": round(final_score * 100, 2),
                "skill_coverage": round(skill_coverage * 100, 2),
                "matched": ", ".join(matched_skills[:8]) if matched_skills else "—",
                "missing": ", ".join(missing[:8]) if missing else "—",
                "matched_count": len(matched_skills),
                "missing_count": len(missing),
                "fit_label": fit_label,
                "candidate_summary": candidate_summary,
            })

        results = sorted(data, key=lambda x: x["match"], reverse=True)
        summary = {
            "total_uploaded": len(uploaded_files),
            "total_processed": len(resume_texts),
            "invalid_count": len(invalid_files),
        }

    return render_template(
        "index.html",
        results=results,
        error=error,
        summary=summary,
        job_description=jd_value,
        username=session.get("username"),
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            error = "Email and password are required."
        elif not is_valid_email(username):
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            conn = sqlite3.connect(DB_PATH)
            try:
                existing_user = conn.execute(
                    "SELECT id FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
                if existing_user:
                    error = "Email already exists."
                    return render_template("register.html", error=error)

                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                conn.commit()
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                error = "Email already exists."
            finally:
                conn.close()

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            user = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        finally:
            conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("index"))

        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not new_password:
            error = "Username and new password are required."
        elif len(new_password) < 6:
            error = "New password must be at least 6 characters."
        elif new_password != confirm_password:
            error = "Passwords do not match."
        else:
            conn = sqlite3.connect(DB_PATH)
            try:
                cur = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
                user = cur.fetchone()
                if not user:
                    error = "No account found with this username."
                else:
                    conn.execute(
                        "UPDATE users SET password_hash = ? WHERE username = ?",
                        (generate_password_hash(new_password), username),
                    )
                    conn.commit()
                    success = "Password updated successfully. You can now login."
            finally:
                conn.close()

    return render_template("forgot_password.html", error=error, success=success)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
