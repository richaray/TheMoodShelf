"""
TheMoodShelf - app.py
=====================
Flask backend for mood-based book recommendations.

Uses:
  - DistilRoBERTa (j-hartmann) for emotion classification
  - Sentence Transformers (all-MiniLM-L6-v2) for semantic book matching
  - MySQL for mood history and saved books
  - Flask sessions to safely track per-user state

Changes in this version:
  - Text cleaning pipeline applied to all book descriptions before embedding
    (fixes latin1/UTF-8 mojibake like ÃÂœ, removes control characters)
  - Improved mood detection: aggregates related emotion scores instead of
    trusting top-1 alone (fixes "calm" misdetection on implied anxiety)
  - Lowered neutral fallback threshold from 0.45 → 0.25
  - Expanded keyword override lists to include anxious + motivated moods
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
load_dotenv()


# ==============================================================
# APP INIT
# ==============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("MOODSHELF_SECRET_KEY", "moodshelf-dev-secret-key-change-in-prod")


# ==============================================================
# DATABASE
# ==============================================================

def get_db_connection():
    """Create and return a fresh DB connection."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.environ.get("MOODSHELF_DB_PASSWORD"),
            database="mood_shelf"
        )
        return conn
    except Error as e:
        print(f"[DB ERROR] Could not connect: {e}")
        return None

def query_db(sql, params=(), fetchone=False):
    """
    Execute a query on a fresh connection+cursor, return results, then close.
    Using a shared cursor causes silent failures when results are left unconsumed —
    this helper avoids that entirely by opening and closing per call.
    """
    conn = get_db_connection()
    if conn is None:
        return None if fetchone else []
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        result = cur.fetchone() if fetchone else cur.fetchall()
        conn.commit()
        return result
    except Error as e:
        print(f"[DB QUERY ERROR] {e}\nSQL: {sql}")
        return None if fetchone else []
    finally:
        cur.close()
        conn.close()

def execute_db(sql, params=()):
    """Execute a write (INSERT/UPDATE/DELETE) and commit."""
    conn = get_db_connection()
    if conn is None:
        return False
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        return True
    except Error as e:
        print(f"[DB EXECUTE ERROR] {e}\nSQL: {sql}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


# ==============================================================
# ML MODELS  (loaded once at startup)
# ==============================================================

def ensure_column(table: str, column: str, definition: str) -> None:
    """
    Add a column to a table only if it doesn't already exist.
    Uses INFORMATION_SCHEMA so it works on MySQL 5.7 and 8.0+.
    (ALTER TABLE … ADD COLUMN IF NOT EXISTS requires MySQL 8.0+.)
    """
    row = query_db(
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (table, column),
        fetchone=True,
    )
    if row and row[0] == 0:
        print(f"[DB] Adding column {column} to {table}...")
        ok = execute_db(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
        if not ok:
            print(f"[DB WARNING] Could not add column {column} to {table}.")
    else:
        print(f"[DB] Column {column} already exists in {table}, skipping.")


ensure_column("mood_entries", "top_emotion", "VARCHAR(50) DEFAULT NULL")
ensure_column("saved_books",  "genre",       "VARCHAR(100) DEFAULT NULL")
ensure_column("saved_books",  "image_url",   "VARCHAR(1000) DEFAULT NULL")


emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("[MODEL] Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================
# TEXT CLEANING
# ==============================================================

def clean_text(text: str) -> str:
    """
    Fix encoding corruption and remove noise from book descriptions.

    The dataset was saved with latin1 encoding but contains characters
    that were originally UTF-8. This causes mojibake like ÃÂœ (which
    should be ") and ÃÂ¦ (which should be ...). We attempt to reverse
    that by re-encoding as latin1 and decoding as UTF-8.

    Steps:
      1. Attempt latin1 -> UTF-8 re-decode to fix mojibake
      2. NFKD unicode normalisation (fancy quotes -> plain, etc.)
      3. Strip control / non-printable characters
      4. Remove remaining garbled non-ASCII sequences (2+ in a row)
      5. Collapse runs of whitespace
    """
    if not isinstance(text, str):
        return ""

    # Step 1 — reverse the latin1-read-as-UTF-8 corruption
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass  # text was already clean or a different encoding — leave it

    # Step 2 — normalise unicode (converts ligatures, fancy punctuation, etc.)
    text = unicodedata.normalize("NFKD", text)

    # Step 3 — remove control characters and non-printable bytes
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Step 4 — remove remaining multi-character non-ASCII garbage sequences
    text = re.sub(r'[^\x00-\x7F]{2,}', ' ', text)

    # Step 5 — collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==============================================================
# BOOK DATASET
# ==============================================================

print("[DATA] Loading book dataset...")
books = pd.read_csv("Book_Dataset_1.csv", encoding="latin1")

# Keep only the columns we need; rename for consistency
books = books[["Title", "Category", "Book_Desc", "Image_Link", "Stars", "Number_of_reviews"]].copy()
books.columns = ["title", "genre", "description", "image_url", "stars", "num_reviews"]

# Drop rows missing any core field
books.dropna(subset=["title", "genre", "description"], inplace=True)

# Clean title and description before embedding
# Genre is only used for string matching so cleaning is optional but consistent
books["title"]       = books["title"].apply(clean_text)
books["description"] = books["description"].apply(clean_text)

# Drop rows where cleaning left a description too short to be meaningful
books = books[books["description"].str.len() > 20]
books.reset_index(drop=True, inplace=True)

books["author"] = "Unknown"  # dataset has no author column

print(f"[DATA] {len(books)} books loaded after cleaning.")

# Precompute description embeddings once on clean text — shape: (num_books, 384)
print("[DATA] Precomputing book embeddings (this may take a moment)...")
book_embeddings = embedder.encode(
    books["description"].tolist(),
    show_progress_bar=True,
    batch_size=64
)
print(f"[DATA] Ready. Embeddings computed for {len(books)} books.")


# ==============================================================
# MOOD CONFIG
# ==============================================================

# Keyword overrides — fire when model confidence is below 0.65.
# Covers moods the model tends to misread because the input is
# tonally neutral but semantically clear (e.g. implied anxiety).
KEYWORD_OVERRIDE = {
    "stressed":  [
        "stress", "stressed", "pressure", "overwhelmed",
        "burnt out", "exhausted", "burnout",
    ],
    "tired":     [
        "tired", "sleepy", "drained", "low energy", "fatigued",
        "can't keep my eyes open",
    ],
    "lonely":    [
        "alone", "lonely", "isolated", "left out",
        "disconnected", "no one around",
    ],
    "anxious":   [
        "anxious", "anxiety", "nervous", "panic", "panicking",
        "worried", "worrying", "dreading", "what if",
        "imagining everything going wrong", "can't stop thinking",
        "keep thinking", "overthinking", "presentation tomorrow",
        "interview tomorrow", "exam tomorrow", "test tomorrow",
    ],
    "motivated": [
        "motivated", "inspired", "determined", "ready to",
        "excited to start", "fired up", "lets go", "let's go",
    ],
}

# Map raw model emotion labels -> our internal mood vocabulary
EMOTION_TO_MOOD = {
    "joy":      "happy",
    "sadness":  "sad",
    "anger":    "stressed",
    "fear":     "anxious",
    "neutral":  "calm",
    "surprise": "motivated",
    "disgust":  "stressed",
}

# Mood -> relevant book genres for candidate filtering
MOOD_GENRE_MAP = {
    "sad":       ["Romance", "Fiction", "Poetry"],
    "stressed":  ["Self Help", "Health", "Spirituality"],
    "happy":     ["Fantasy", "Travel", "Music", "Food", "Horror"],
    "calm":      ["Poetry", "Art", "Philosophy"],
    "anxious":   ["Mystery", "Thriller", "Self Help"],
    "lonely":    ["Fiction", "Contemporary", "Young Adult"],
    "tired":     ["Poetry", "Sequential Art", "Food"],
    "motivated": ["Business", "Science", "Nonfiction", "History"],
}

# Emoji shown next to mood name in the UI
MOOD_EMOJI = {
    "sad":       "😔",
    "stressed":  "😤",
    "happy":     "😊",
    "calm":      "😌",
    "anxious":   "😰",
    "lonely":    "🫂",
    "tired":     "😴",
    "motivated": "🚀",
}


# ==============================================================
# NEGATION HANDLING
# ==============================================================

# Maps each mood to the emotion keywords that represent it.
# Used to detect phrases like "not sad", "don't feel anxious", etc.
MOOD_NEGATION_KEYWORDS = {
    "sad":       ["sad", "sadness", "depressed", "unhappy", "miserable", "gloomy", "grief"],
    "happy":     ["happy", "joyful", "excited", "cheerful", "glad", "elated"],
    "stressed":  ["stressed", "stress", "overwhelmed", "pressured", "burnt out", "burnout"],
    "anxious":   ["anxious", "anxiety", "nervous", "worried", "scared", "afraid", "fearful"],
    "calm":      ["calm", "relaxed", "peaceful", "fine", "okay", "ok", "serene", "composed"],
    "lonely":    ["lonely", "alone", "isolated", "disconnected"],
    "tired":     ["tired", "exhausted", "sleepy", "fatigued", "drained"],
    "motivated": ["motivated", "inspired", "energized", "determined"],
}

# When a mood is explicitly negated, map it to this opposite mood.
NEGATION_FLIP = {
    "sad":       "calm",
    "happy":     "sad",
    "stressed":  "calm",
    "anxious":   "calm",
    "calm":      "anxious",
    "lonely":    "calm",
    "tired":     "motivated",
    "motivated": "calm",
}


def get_negated_moods(text: str) -> set:
    """
    Scan the text for explicitly negated emotion words and return the set
    of moods that are being denied.

    Handles patterns like:
      - "I am not sad"
      - "I don't feel anxious"
      - "not feeling stressed at all"
      - "I'm not really lonely"

    The regex looks for a negation trigger word followed by up to 3
    filler words (adjectives/adverbs) and then the emotion keyword.
    This keeps it simple and avoids false positives on distant negations.
    """
    negated = set()
    text_lower = text.lower()
    # Negation trigger pattern (covers contractions too)
    neg_trigger = r"\b(not|no|never|don'?t|doesn'?t|am\s+not|i'?m\s+not|isn'?t|aren'?t|wasn'?t|weren'?t|haven'?t|hasn'?t|without)\b"

    for mood, keywords in MOOD_NEGATION_KEYWORDS.items():
        for kw in keywords:
            # Allow up to 3 words between the negation and the keyword
            pattern = neg_trigger + r"(?:\s+\w+){0,3}\s+" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                negated.add(mood)
    return negated


# ==============================================================
# MOOD DETECTION
# ==============================================================

def detect_mood(text: str) -> tuple[str, list[dict]]:
    """
    Detect the user's mood from free-text input.

    Returns:
        mood    (str)        -- one of the 8 mood keys
        scores  (list[dict]) -- all 7 emotion scores sorted descending,
                                each dict has keys 'label' and 'score'

    Strategy:
      1. Always run the transformer model to get the full emotion breakdown.
      2. Aggregate related emotions into mood buckets rather than trusting
         the single top label. This fixes cases where moderate fear +
         moderate anger both point to anxiety but neither alone clears
         the old threshold.
      3. If aggregated confidence is still low (< 0.25), default to calm.
      4. Negation check — if the detected mood is explicitly negated in the
         text (e.g. "I am not sad"), flip it to the opposite mood. Transformers
         are blind to negation so this must be done with regex, not the model.
      5. If confidence is below 0.65, allow keyword overrides to take
         priority — catches tonally neutral but semantically clear inputs
         like "my presentation is tomorrow and I keep imagining going wrong".
    """
    # Step 1 — run the model
    raw_results   = emotion_model(text)[0]
    scores_sorted = sorted(raw_results, key=lambda x: x["score"], reverse=True)

    # Build a fast label -> score lookup
    score_map = {r["label"]: r["score"] for r in raw_results}

    # Step 2 — aggregate into mood buckets
    # Related emotions are summed with weights so that two moderate signals
    # beat one weak signal, without requiring any single emotion to dominate.
    mood_scores = {
        "anxious":  score_map.get("fear",    0) + score_map.get("surprise", 0) * 0.4,
        "stressed": score_map.get("anger",   0) + score_map.get("disgust",  0) * 0.5,
        "sad":      score_map.get("sadness", 0),
        "happy":    score_map.get("joy",     0),
        "calm":     score_map.get("neutral", 0),
    }

    best_mood  = max(mood_scores, key=mood_scores.get)
    best_score = mood_scores[best_mood]

    # Step 3 — low-confidence fallback to calm
    if best_score < 0.25:
        best_mood = "calm"

    # Step 4 — negation override (runs regardless of model confidence)
    # The model sees "sad" in "I am not sad" and fires sadness anyway.
    # We catch that here by checking for explicit negation in the raw text.
    negated_moods = get_negated_moods(text)
    if best_mood in negated_moods:
        best_mood = NEGATION_FLIP.get(best_mood, "calm")

    # Step 5 — keyword override when model is not highly confident
    if best_score < 0.65:
        text_lower = text.lower()
        for override_mood, keywords in KEYWORD_OVERRIDE.items():
            # Skip keyword override if that mood was explicitly negated
            if override_mood in negated_moods:
                continue
            if any(kw in text_lower for kw in keywords):
                return override_mood, scores_sorted

    return best_mood, scores_sorted


# ==============================================================
# PERSONALIZATION
# ==============================================================

def get_user_preferences(user_id: int, mood: str) -> tuple[dict, set]:
    """
    Build a genre preference profile and a set of already-saved titles
    for this user + mood combination.

    Returns:
        genre_weights  (dict)  -- {genre: boost_weight} derived from save history.
                                  Genres saved more often for this mood get a
                                  higher weight (max 0.30, i.e. up to 30% score boost).
        saved_titles   (set)   -- titles the user has already saved; these will be
                                  excluded from recommendations so we never repeat a book.
    """
    rows = query_db(
        "SELECT genre FROM saved_books WHERE user_id = %s AND mood = %s AND genre IS NOT NULL",
        (user_id, mood)
    )

    # Count how many times each genre was saved for this mood
    genre_counts: dict[str, int] = {}
    for (genre,) in rows:
        if genre:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Normalise to a 0–0.30 boost weight
    total = sum(genre_counts.values()) or 1
    genre_weights = {g: (cnt / total) * 0.30 for g, cnt in genre_counts.items()}

    # All titles ever saved by this user (across all moods) — never re-recommend these
    title_rows = query_db(
        "SELECT title FROM saved_books WHERE user_id = %s",
        (user_id,)
    )
    saved_titles = {row[0] for row in title_rows if row[0]}

    return genre_weights, saved_titles


# ==============================================================
# BOOK RECOMMENDATION  (semantic matching + personalisation)
# ==============================================================

def recommend_book(mood: str, user_text: str, top_n: int = 5,
                   user_id: int | None = None,
                   extra_exclude: set | None = None) -> list[dict]:
    """
    Recommend books for a given mood using semantic similarity.

    Personalisation (when user_id is provided):
      - Already-saved books are excluded from results so the user never
        sees the same book twice.
      - Genres the user has historically saved for this mood receive a
        score boost (up to +0.30) so preferred genres surface higher.
        The boost is proportional to save frequency, so it grows naturally
        the more the user interacts with the app.
      - Falls back to pure semantic matching for new users with no history.

    extra_exclude:
      - Additional set of titles to skip (used by generate-again to avoid
        returning books that were already shown in earlier result pages).
    """
    # --- Fetch personalisation data (empty defaults for anonymous/new users) ---
    genre_weights: dict[str, float] = {}
    saved_titles:  set[str]         = set()
    if user_id is not None:
        genre_weights, saved_titles = get_user_preferences(user_id, mood)

    # Merge DB-saved titles with any caller-supplied exclusions
    if extra_exclude:
        saved_titles = saved_titles | extra_exclude

    # --- Build candidate pool from mood-appropriate genres ---
    genres        = MOOD_GENRE_MAP.get(mood, ["Fiction"])
    genre_pattern = "|".join(genres)
    genre_mask    = books["genre"].str.contains(genre_pattern, case=False, na=False)
    candidate_indices = books[genre_mask].index.tolist()

    if not candidate_indices:
        candidate_indices = books.index.tolist()

    # --- Semantic similarity ---
    cleaned_user_text    = clean_text(user_text)
    user_embedding       = embedder.encode([cleaned_user_text])
    candidate_embeddings = book_embeddings[candidate_indices]
    similarities         = cosine_similarity(user_embedding, candidate_embeddings)[0]

    # --- Apply personalisation: genre boost + already-read exclusion ---
    adjusted_scores = []
    for local_idx, base_score in enumerate(similarities):
        global_idx  = candidate_indices[local_idx]
        book_title  = books.iloc[global_idx]["title"]
        book_genre  = books.iloc[global_idx]["genre"]

        # Skip books the user has already saved
        if book_title in saved_titles:
            adjusted_scores.append(-1.0)   # sentinel: never selected
            continue

        # Boost score if this genre matches a user preference for this mood.
        # genre_weights values are already normalised to 0–0.30.
        boost = 0.0
        if genre_weights and isinstance(book_genre, str):
            for pref_genre, weight in genre_weights.items():
                if pref_genre.lower() in book_genre.lower():
                    boost = weight
                    break

        adjusted_scores.append(base_score + boost)

    adjusted_scores = np.array(adjusted_scores)

    # Fetch top_n from the adjusted ranking (ignoring sentinel -1 values)
    # Over-fetch a little in case some results are sentinels
    fetch_n           = min(top_n + len(saved_titles), len(adjusted_scores))
    top_local_indices = np.argsort(adjusted_scores)[-fetch_n:][::-1]

    results = []
    for local_idx in top_local_indices:
        if len(results) >= top_n:
            break
        if adjusted_scores[local_idx] < 0:   # skip sentinels
            continue
        global_idx = candidate_indices[local_idx]
        row        = books.iloc[global_idx]
        results.append({
            "title":       row["title"],
            "author":      row["author"],
            "genre":       row["genre"],
            "image_url":   row["image_url"],
            "description": row["description"],
            "stars":       row["stars"],
            "num_reviews": row["num_reviews"],
            "similarity":  round(float(similarities[local_idx]) * 100, 1),
        })
    return results


# ==============================================================
# HELPERS
# ==============================================================

def format_emotion_scores(scores: list[dict]) -> list[dict]:
    """
    Convert raw model output into a display-ready list for templates.
    Maps internal labels to readable names and converts to percentages.
    """
    label_display = {
        "joy":      "Joy",
        "sadness":  "Sadness",
        "anger":    "Anger",
        "fear":     "Fear",
        "neutral":  "Neutral",
        "surprise": "Surprise",
        "disgust":  "Disgust",
    }
    return [
        {
            "label": label_display.get(s["label"], s["label"].capitalize()),
            "score": round(s["score"] * 100, 1),
        }
        for s in scores
    ]


# ==============================================================
# ROUTES
# ==============================================================

@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_text = request.form.get("mood_text", "").strip()
    if not user_text:
        return redirect(url_for("home"))

    mood, raw_scores = detect_mood(user_text)
    emotion_scores   = format_emotion_scores(raw_scores)

    top_emotion = raw_scores[0]["label"] if raw_scores else ""
    user_id     = session.get("user_id")

    execute_db(
        "INSERT INTO mood_entries (user_id, mood, input_text, top_emotion) VALUES (%s, %s, %s, %s)",
        (user_id, mood, user_text, top_emotion)
    )

    session["last_mood"]    = mood
    session["last_text"]    = user_text
    session["saved_titles"] = []   # reset on every fresh analysis

    books_list = recommend_book(mood, user_text, user_id=user_id)

    # Track which titles have been shown so generate-again can skip them
    session["shown_titles"] = [b["title"] for b in books_list]

    return render_template(
        "result.html",
        mood=mood,
        mood_emoji=MOOD_EMOJI.get(mood, "📖"),
        emotion_scores=emotion_scores,
        books=books_list,
        saved_titles=[],
    )

@app.route("/generate-again")
def generate_again():
    mood      = session.get("last_mood")
    user_text = session.get("last_text")

    if not mood:
        result = query_db("SELECT mood FROM mood_entries ORDER BY id DESC LIMIT 1", fetchone=True)
        if not result:
            return redirect(url_for("home"))
        mood      = result[0]
        user_text = mood

    # Collect all titles already shown across previous generate-again calls
    # so each click surfaces a genuinely fresh set of books
    shown_titles  = set(session.get("shown_titles", []))
    saved_titles  = session.get("saved_titles", [])

    books_list = recommend_book(
        mood, user_text,
        user_id=session.get("user_id"),
        extra_exclude=shown_titles,
    )

    # Accumulate shown titles — next generate-again will skip these too
    for b in books_list:
        shown_titles.add(b["title"])
    session["shown_titles"] = list(shown_titles)

    return render_template(
        "result.html",
        mood=mood,
        mood_emoji=MOOD_EMOJI.get(mood, "📖"),
        emotion_scores=[],
        books=books_list,
        saved_titles=saved_titles,
    )


@app.route("/save", methods=["POST"])
def save_book():
    if "user_id" not in session:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {"error": "not logged in"}, 401
        return redirect(url_for("login"))

    user_id   = session.get("user_id")
    title     = request.form.get("title", "")
    author    = request.form.get("author", "")
    mood      = request.form.get("mood", "")
    genre     = request.form.get("genre", "")
    image_url = request.form.get("image_url", "")

    ok = execute_db(
        "INSERT INTO saved_books (user_id, title, author, mood, genre, image_url) VALUES (%s, %s, %s, %s, %s, %s)",
        (user_id, title, author, mood, genre, image_url)
    )

    # Track saved titles in session
    saved_titles = session.get("saved_titles", [])
    if title not in saved_titles:
        saved_titles.append(title)
    session["saved_titles"] = saved_titles

    # AJAX request — return JSON so the page does not reload
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return {"status": "saved" if ok else "error"}, (200 if ok else 500)

    # Non-JS fallback
    return redirect(url_for("saved_books"))


@app.route("/saved-books")
def saved_books():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")

    saved = query_db(
        "SELECT title, author, mood, saved_at FROM saved_books WHERE user_id = %s ORDER BY saved_at DESC",
        (user_id,)
    )

    return render_template("saved_books.html", books=saved)

@app.route("/delete-saved", methods=["POST"])
def delete_saved():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")
    title   = request.form.get("title", "")
    author  = request.form.get("author", "")

    execute_db(
        "DELETE FROM saved_books WHERE user_id = %s AND title = %s AND author = %s LIMIT 1",
        (user_id, title, author)
    )

    # Also remove from session saved_titles if present
    saved_titles = session.get("saved_titles", [])
    if title in saved_titles:
        saved_titles.remove(title)
        session["saved_titles"] = saved_titles

    return redirect(url_for("saved_books"))

@app.route("/journal")
def journal():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")

    journal_rows = query_db(
        "SELECT content, created_at, 'journal' as source FROM journal_entries WHERE user_id = %s",
        (user_id,)
    )

    mood_rows = query_db(
        "SELECT input_text, created_at, mood FROM mood_entries WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,)
    )

    return render_template("journal.html", journal_entries=journal_rows, mood_entries=mood_rows)


@app.route("/journal/add", methods=["POST"])
def add_journal_entry():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")
    content = request.form.get("content", "").strip()

    if content:
        execute_db(
            "INSERT INTO journal_entries (user_id, content) VALUES (%s, %s)",
            (user_id, content)
        )

    return redirect(url_for("journal"))

@app.route("/analytics")
def analytics():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")

    # --- Stat cards ---
    row = query_db("SELECT COUNT(*) FROM mood_entries WHERE user_id = %s", (user_id,), fetchone=True)
    total_moods = row[0] if row else 0

    row = query_db("SELECT COUNT(*) FROM saved_books WHERE user_id = %s", (user_id,), fetchone=True)
    total_saved = row[0] if row else 0

    row = query_db("SELECT COUNT(*) FROM journal_entries WHERE user_id = %s", (user_id,), fetchone=True)
    total_journal = row[0] if row else 0

    row = query_db("SELECT created_at FROM users WHERE id = %s", (user_id,), fetchone=True)
    member_since = row[0].strftime("%d %b %Y") if row and row[0] else "—"

    # --- Mood distribution (donut chart) ---
    mood_dist = query_db(
        "SELECT mood, COUNT(*) as cnt FROM mood_entries "
        "WHERE user_id = %s GROUP BY mood ORDER BY cnt DESC",
        (user_id,)
    )

    # --- Top raw emotions (horizontal bar) ---
    emotion_dist = query_db(
        "SELECT top_emotion, COUNT(*) as cnt FROM mood_entries "
        "WHERE user_id = %s AND top_emotion IS NOT NULL AND top_emotion != '' "
        "GROUP BY top_emotion ORDER BY cnt DESC",
        (user_id,)
    )

    # --- Saved books by mood (bar chart) ---
    saved_by_mood = query_db(
        "SELECT mood, COUNT(*) as cnt FROM saved_books "
        "WHERE user_id = %s GROUP BY mood ORDER BY cnt DESC",
        (user_id,)
    )

    # --- Mood activity last 30 days (line chart) ---
    activity_raw = query_db(
        "SELECT DATE(created_at) as day, COUNT(*) as cnt FROM mood_entries "
        "WHERE user_id = %s AND created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
        "GROUP BY day ORDER BY day ASC",
        (user_id,)
    )
    activity_labels = [str(r[0]) for r in activity_raw]
    activity_data   = [r[1]      for r in activity_raw]

    # --- Most frequent mood (hero label) ---
    top_mood = mood_dist[0][0] if mood_dist else "none"

    return render_template(
        "analytics.html",
        total_moods   = total_moods,
        total_saved   = total_saved,
        total_journal = total_journal,
        member_since  = member_since,
        mood_dist     = mood_dist,
        emotion_dist  = emotion_dist,
        saved_by_mood = saved_by_mood,
        activity_labels = activity_labels,
        activity_data   = activity_data,
        top_mood      = top_mood,
    )



@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        hashed_password = generate_password_hash(password)

        execute_db(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )

        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = query_db(
            "SELECT id, password_hash FROM users WHERE email = %s",
            (email,), fetchone=True
        )

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            return redirect(url_for("home"))

        return "Invalid credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":
    app.run(debug=True)