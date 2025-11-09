# backend/main.py
import os, re, json, math, random
from typing import Dict, List
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from contextlib import asynccontextmanager
from collections import Counter
import joblib
import numpy as np

# =========================
# Models & vectorizers
# =========================
MODEL_PATH = Path(__file__).parent / "best_model.joblib"
VECTORIZER_PATH = Path(__file__).parent / "vectorizer.joblib"

print("Environment variables:")
print(f"USE_SEALION={os.getenv('USE_SEALION')}")
print(f"SEALION_MODEL={os.getenv('SEALION_MODEL')}")
print(f"SEALION_BASE={os.getenv('SEALION_BASE')}")

try:
    sentiment_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Loaded sentiment model and vectorizer.")
except Exception as e:
    print(f"Could not load model/vectorizer: {e}")
    sentiment_model = None
    vectorizer = None

# =========================
# Config & env
# =========================
CSV_PATH = os.getenv("CSV_PATH", "reviews.csv")  # default relative path

# Sanitize SEA-LION base to avoid accidental double /chat/completions
SEALION_BASE_RAW = os.getenv("SEALION_BASE", "https://api.sea-lion.ai/v1")
_SEA_BASE = SEALION_BASE_RAW.rstrip("/")
SEALION_BASE = re.sub(r"/chat/completions$", "", _SEA_BASE)

SEALION_MODEL = os.getenv("SEALION_MODEL", "aisingapore/Gemma-SEA-LION-v4-27B-IT")
SEALION_KEY = os.getenv("SEALION_API_KEY")
if not SEALION_KEY:
    print("Warning: SEALION_API_KEY not set, SEA-LION will be disabled")
SEALION_TIMEOUT = float(os.getenv("SEALION_TIMEOUT", "45"))  # seconds
SEALION_RETRIES = int(os.getenv("SEALION_RETRIES", "2"))     # retry count on timeout

app = FastAPI(title="ReviewLens")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# =========================
# Helpers
# =========================
STOPWORDS = {
    "the","a","an","and","or","but","if","so","of","to","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","its","this","that","these","those",
    "i","you","he","she","we","they","them","my","your","our","their",
    "very","really","just","too","also","than","then","there","here",
}

def english_only_place(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    ascii_part = "".join(ch if ord(ch) < 128 else " " for ch in raw)
    ascii_part = re.sub(r"[^A-Za-z0-9 .,&'()/-]", " ", ascii_part)
    ascii_part = re.sub(r"\s{2,}", " ", ascii_part).strip(" -•|")
    return ascii_part

# Accepts many possible encodings of labels; maps to {positive, negative, neutral}
POS_SET = {"positive", "pos", "positif", "👍", "good", "5", "4", "1", "true"}
NEG_SET = {"negative", "neg", "negatif", "👎", "bad", "poor", "0", "-1", "false"}
NEU_SET = {"neutral", "neu", "mixed", "3", "ok", "meh"}

def _norm_label(x) -> str:
    s = str(x).strip()
    sl = s.lower()
    if sl in POS_SET or s == "Positive":
        return "positive"
    if sl in NEG_SET or s == "Negative":
        return "negative"
    if sl in NEU_SET or s == "Neutral":
        return "neutral"
    try:
        fv = float(sl)
        if fv >= 0.5:
            return "positive"
        if fv <= -0.5 or fv == 0.0:
            return "negative"
    except:
        pass
    return "neutral"

def predict_sentiments(texts: List[str]) -> List[str]:
    if not sentiment_model or not vectorizer:
        raise HTTPException(500, "Sentiment model/vectorizer not loaded")
    X = vectorizer.transform(texts)

    if hasattr(sentiment_model, "predict_proba"):
        proba = sentiment_model.predict_proba(X)
        classes = [str(c) for c in getattr(sentiment_model, "classes_", [])]
        labels = []
        for row in proba:
            idx = int(row.argmax())
            labels.append(_norm_label(classes[idx]))
        return labels

    preds = sentiment_model.predict(X)
    return [_norm_label(p) for p in preds.tolist()]

def aggregate_sentiment(labels: List[str]) -> Dict[str, int]:
    c = Counter(lbl for lbl in labels if lbl is not None)
    return {k: int(v) for k, v in c.items()}

def _rating_histogram(sub: pd.DataFrame) -> dict:
    c = Counter(sub["rating"].astype(int).tolist())
    return {str(k): int(c.get(k, 0)) for k in range(1, 6)}

def _as_percentage_hist(hist: dict) -> dict:
    total = sum(hist.values()) or 1
    return {k: round(v * 100 / total) for k, v in hist.items()}

def _top_strings_from_list(items, maxn=6):
    return [str(x) for x in items if str(x).strip()][:maxn]

def strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

def extract_json_anywhere(text: str):
    if not isinstance(text, str):
        return None
    t = strip_code_fences(text)
    m = re.search(r"\{(?:[^{}]|(?R))*\}", t, flags=re.S) or re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ===== Deny-lists to avoid generic "good food" style noise =====
DENY_UNIGRAMS = {
    "had","have","has","get","got","go","went","come","came","make","made","take","took",
    "put","see","seen","try","tried","would","could","should","visit","visited",
    "am","is","are","was","were","be","been","being","ok","okay","nice","good","great","amazing",
    "not","no","without","never","dont","didnt","doesnt","isnt","arent","cant","wont"
}
ALLOW_UNIGRAMS = {
    "service","staff","food","ambience","atmosphere","music","noise","value","price","portion",
    "queue","wait","booking","reservation","water","menu","allergy","cleanliness","hygiene",
    "parking","seating","table","tables","aircon","toilet","bathroom","ramen","ramyeon","pancake",
    "chicken","pizza","rice","noodles"
}
DENY_BIGRAMS = {"very good","really good","good food","nice food","great food","not bad"}

# =========================
# Model-driven bigram categorization
# =========================
SMALL_WORDS = {"and","or","of","to","in","on","for","with","a","an","the"}

def _nice_title(s: str) -> str:
    parts = s.split()
    return " ".join(p if (i>0 and p in SMALL_WORDS) else p.capitalize() for i,p in enumerate(parts))

def _class_map() -> List[str]:
    raw = [str(c) for c in getattr(sentiment_model, "classes_", [])]
    return [_norm_label(c) for c in raw]  # -> e.g., ["negative","neutral","positive"]

def _bigrams_from_texts(texts: List[str]) -> Counter:
    analyzer = vectorizer.build_analyzer() if vectorizer else (lambda t: re.findall(r"[a-z]+(?:\s[a-z]+)?", t.lower()))
    cnt = Counter()
    for t in texts:
        toks = analyzer(str(t))
        for tok in toks:
            if " " in tok:  # only bigrams if vectorizer is (1,2)
                cnt[tok] += 1
    return cnt

def model_categorized_bigrams(texts: List[str], topn_each: int = 10):
    """
    1) Extract bigrams and count freq.
    2) For each bigram in vocab, look at model coef across classes.
    3) Assign bigram to the class with the largest weight (positive/negative only).
    """
    if not (vectorizer and sentiment_model and hasattr(sentiment_model, "coef_")):
        return [], []

    classes = _class_map()                           # normalized class names
    W = np.asarray(sentiment_model.coef_)           # (n_classes, n_features)
    vocab = vectorizer.vocabulary_
    freq = _bigrams_from_texts(texts)

    pos_rows, neg_rows = [], []
    for bg, f in freq.items():
        if bg not in vocab:
            continue
        if bg in DENY_BIGRAMS:
            continue
        j = vocab[bg]
        col = W[:, j]
        ci = int(np.argmax(col))
        cls = classes[ci] if 0 <= ci < len(classes) else "neutral"
        label = _nice_title(bg)

        if cls == "positive":
            pos_rows.append((label, f, float(col[ci])))
        elif cls == "negative":
            neg_rows.append((label, f, float(col[ci])))
        # ignore neutral to keep lists crisp

    # Sort by: frequency desc, weight desc, length desc
    pos_rows.sort(key=lambda r: (-r[1], -r[2], -len(r[0])))
    neg_rows.sort(key=lambda r: (-r[1], -r[2], -len(r[0])))

    def dedupe(rows):
        out, seen = [], []
        for label, f, w in rows:
            low = label.lower()
            if any(low in s for s in seen):
                continue
            seen.append(low)
            out.append({"label": label, "count": int(f)})
            if len(out) >= topn_each:
                break
        return out

    return dedupe(pos_rows), dedupe(neg_rows)

# ---------- Signed TF-IDF aspect mining (positive vs negative) ----------
def _analyzer():
    try:
        return vectorizer.build_analyzer()
    except Exception:
        def basic_analyzer(t: str):
            return re.findall(r"[a-z]+(?:\s[a-z]+)?", t.lower())
        return basic_analyzer

def _nicify_phrase(s: str) -> str:
    small = {"and","or","of","to","in","on","for","with","a","an","the"}
    parts = s.split()
    return " ".join(p if (i>0 and p in small) else p.capitalize() for i,p in enumerate(parts))

def _collect_term_stats(texts: list[str]) -> dict[str, dict]:
    """
    Returns {term: {"tfidf": <mean tfidf>, "support": <doc freq>}}
    using the fitted vectorizer.
    """
    if not texts:
        return {}
    X = vectorizer.transform([str(t) for t in texts])       # (docs x feats)
    mean_scores = np.asarray(X.mean(axis=0)).ravel()
    support = np.asarray((X > 0).sum(axis=0)).ravel()
    # inverse vocab
    terms = [None]*len(vectorizer.vocabulary_)
    for t, idx in vectorizer.vocabulary_.items():
        terms[idx] = t
    out = {}
    for i, sc in enumerate(mean_scores):
        if sc <= 0: 
            continue
        term = terms[i]
        if not term:
            continue
        out[term] = {"tfidf": float(sc), "support": int(support[i])}
    return out

def _dominance_rank(pos_stats: dict, neg_stats: dict, prefer_bigrams=True,
                    min_support=2, min_ratio=1.5, topn=10):
    """
    Split terms into positive-dominant (praises) and negative-dominant (issues)
    using support ratios with smoothing. Returns two lists of {"label","count"}.
    """
    # union of terms present in either side
    all_terms = set(pos_stats) | set(neg_stats)

    # small stoplists
    deny_bigrams = {"very good","really good","good food","nice food","great food","not bad"}
    deny_unigrams = {
        "had","have","has","get","got","go","went","come","came","make","made","take","took",
        "put","see","seen","try","tried","would","could","should","visit","visited",
        "am","is","are","was","were","be","been","being","ok","okay","nice","good","great","amazing",
        "not","no","without","never","dont","didnt","doesnt","isnt","arent","cant","wont"
    }

    scored_pos, scored_neg = [], []
    for term in all_terms:
        is_bigram = (" " in term)
        if is_bigram:
            if term in deny_bigrams:
                continue
        else:
            if term in deny_unigrams:
                continue

        p_sup = pos_stats.get(term, {}).get("support", 0)
        n_sup = neg_stats.get(term, {}).get("support", 0)
        p_tfidf = pos_stats.get(term, {}).get("tfidf", 0.0)
        n_tfidf = neg_stats.get(term, {}).get("tfidf", 0.0)

        # smooth to avoid div/zero; use normalized dominance ratios
        p_sup_s = p_sup + 1
        n_sup_s = n_sup + 1
        ratio_pos = p_sup_s / n_sup_s
        ratio_neg = n_sup_s / p_sup_s

        # skip very weak terms
        if max(p_sup, n_sup) < min_support:
            continue

        # composite score: emphasize dominance + tfidf + support
        if ratio_pos >= min_ratio and (is_bigram or p_tfidf >= n_tfidf):
            score = (ratio_pos) * (p_tfidf + 1e-9) * (p_sup)
            priority = 0 if (prefer_bigrams and is_bigram) else 1
            scored_pos.append((priority, -score, -p_sup, -len(term), term, p_sup))
        elif ratio_neg >= min_ratio and (is_bigram or n_tfidf >= p_tfidf):
            score = (ratio_neg) * (n_tfidf + 1e-9) * (n_sup)
            priority = 0 if (prefer_bigrams and is_bigram) else 1
            scored_neg.append((priority, -score, -n_sup, -len(term), term, n_sup))

    scored_pos.sort()
    scored_neg.sort()

    praises = [{"label": _nicify_phrase(t), "count": int(c)} 
               for _,_,_,_,t,c in scored_pos[:topn]]
    issues  = [{"label": _nicify_phrase(t), "count": int(c)} 
               for _,_,_,_,t,c in scored_neg[:topn]]

    # enforce exclusivity: if something appears in both because of tie, keep the side with higher count
    pos_labels = {x["label"]: x["count"] for x in praises}
    neg_labels = {x["label"]: x["count"] for x in issues}
    overlap = set(pos_labels) & set(neg_labels)
    for o in overlap:
        if pos_labels[o] >= neg_labels[o]:
            issues = [x for x in issues if x["label"] != o]
        else:
            praises = [x for x in praises if x["label"] != o]

    return praises, issues

# ================
# OFFLINE NARRATIVE (uses model-categorized bigrams)
# ================
def offline_narrative(place: str, metrics: dict, pos_texts: list[str], neg_texts: list[str]) -> dict:
    all_texts = [str(x) for x in (pos_texts + neg_texts)]
    praises, issues = model_categorized_bigrams(all_texts, topn_each=10)

    stance = "positive" if (metrics.get("pos_pct", 0) >= 55) else ("negative" if (metrics.get("neg_pct", 0) > 45) else "mixed")
    summary = (
        f"Overall sentiment for {place} appears {stance}. "
        f"Average rating {metrics.get('avg_rating', 0):.2f}★ across {metrics.get('num_reviews', 0)} reviews. "
        f"{metrics.get('pos_pct', 0):.0f}% positive vs {metrics.get('neg_pct', 0):.0f}% negative per the model."
    )

    return {
        "executive_summary": summary,
        "praises": praises,
        "issues": issues,
        "suggestions": [
            "Stabilise quality on best-selling items; track new reviews weekly.",
            "Triage the two most repeated issues this fortnight and close the loop.",
            "Amplify praised items and staff shout-outs in menu boards and socials.",
        ],
        "sampleQuotes": {
            "positive": [str(x) for x in pos_texts[:5]],
            "negative": [str(x) for x in neg_texts[:5]],
        },
    }

# =========================
# Data load
# =========================
if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV not found at {CSV_PATH}. Mount a file or set CSV_PATH.")

df = pd.read_csv(CSV_PATH)  # columns: place, rating, review
df["clean_place"] = df["place"].apply(english_only_place)
df = df.dropna(subset=["clean_place", "rating", "review"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

# =========================
# HTTP client
# =========================
@asynccontextmanager
async def _client():
    async with httpx.AsyncClient(timeout=SEALION_TIMEOUT) as c:
        yield c

# =========================
# Routes
# =========================
@app.get("/healthz")
def healthz():
    n = len(df)
    return {"ok": True, "rows": n}

@app.get("/api/ratings")
def ratings(place: str):
    cleaned = english_only_place(place)
    sub = df[df["clean_place"].str.casefold() == cleaned.casefold()]
    if sub.empty:
        raise HTTPException(404, f"No reviews for {cleaned}")

    sub = sub.copy()
    if not (sentiment_model and vectorizer):
        raise HTTPException(500, "Sentiment model not loaded")
    X = vectorizer.transform(sub["review"].astype(str).tolist())
    preds = sentiment_model.predict(X)
    norm = [_norm_label(p) for p in preds]
    sub.loc[:, "predicted_sentiment"] = norm
    sentiment_score = (sum(1 for x in norm if x == "positive") / len(norm) * 100) if len(norm) else 0

    return {
        "place": place,
        "cleanedPlace": cleaned,
        "avgRating": round(sub["rating"].mean(), 2),
        "sentimentScore": round(sentiment_score, 2),
        "numReviews": len(sub),
        "predictedSentimentCounts": sub["predicted_sentiment"].value_counts(dropna=True).to_dict(),
    }

@app.get("/api/places")
def places():
    names = sorted(df["clean_place"].dropna().unique().tolist())
    return {"count": len(names), "places": names}

@app.get("/api/debug/normalize")
def debug_normalize(place: str):
    return {"input": place, "cleaned": english_only_place(place)}

# =========================
# SEA-LION client
# =========================
async def sealion_chat_json(messages: List[Dict]) -> Dict:
    if not SEALION_KEY:
        raise HTTPException(500, "Missing SEALION_API_KEY")

    payload = {
        "model": SEALION_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 400,
        "temperature": 0.4,
        "top_p": 0.95,
    }

    last_err = None
    for attempt in range(SEALION_RETRIES + 1):
        try:
            async with _client() as client:
                r = await client.post(
                    f"{SEALION_BASE}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {SEALION_KEY}",
                    },
                    json=payload,
                )
                if r.status_code != 200:
                    raise HTTPException(r.status_code, f"SEA-LION error: {r.text}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]

                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    parsed = extract_json_anywhere(content)
                    if parsed is not None:
                        return parsed
                    clean = strip_code_fences(content)
                    return {
                        "executive_summary": clean.strip(),
                        "praises": [],
                        "issues": [],
                        "suggestions": [],
                        "sampleQuotes": {"positive": [], "negative": []},
                    }
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
        except httpx.HTTPError as e:
            last_err = e
            break
    raise HTTPException(503, f"SEA-LION unavailable: {last_err}")

# =========================
# /api/report
# =========================
@app.get("/api/report")
async def report(place: str):
    cleaned = english_only_place(place)
    sub = df[df["clean_place"].str.casefold() == cleaned.casefold()]
    if sub.empty:
        raise HTTPException(404, f"No reviews for {cleaned}")
    if not (sentiment_model and vectorizer):
        raise HTTPException(500, "Sentiment model not loaded")

    sub = sub.copy()
    texts = sub["review"].astype(str).tolist()
    labels = predict_sentiments(texts)
    sub.loc[:, "predicted_sentiment"] = labels

    n = len(sub)
    avg_rating = float(sub["rating"].mean()) if n else 0.0
    counts = aggregate_sentiment(labels)
    pos = counts.get("positive", 0); neg = counts.get("negative", 0); neu = counts.get("neutral", 0)
    pos_pct = round((pos / n * 100), 2) if n else 0.0
    neg_pct = round((neg / n * 100), 2) if n else 0.0
    neu_pct = round((neu / n * 100), 2) if n else 0.0

    hist_counts = _rating_histogram(sub)
    for k in ["1","2","3","4","5"]:
        hist_counts.setdefault(k, 0)
    hist_pct = _as_percentage_hist(hist_counts)

    # sample quotes (for UI flavor)
    pos_q = _top_strings_from_list(sub[sub["predicted_sentiment"] == "positive"]["review"].tolist(), 6)
    neg_q = _top_strings_from_list(sub[sub["predicted_sentiment"] == "negative"]["review"].tolist(), 6)

    # ===== signed TF-IDF aspects built from FULL corpora =====
    pos_texts = sub.loc[sub["predicted_sentiment"] == "positive", "review"].astype(str).tolist()
    neg_texts = sub.loc[sub["predicted_sentiment"] == "negative", "review"].astype(str).tolist()
    pos_stats = _collect_term_stats(pos_texts)
    neg_stats = _collect_term_stats(neg_texts)
    praises_tfidf, issues_tfidf = _dominance_rank(
        pos_stats, neg_stats,
        prefer_bigrams=True, min_support=2, min_ratio=1.6, topn=12
    )

    metrics = {
        "place": place,
        "cleanedPlace": cleaned,
        "num_reviews": n,
        "avg_rating": round(avg_rating, 2),
        "predicted_sentiment_counts": counts,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct
    }

    # Narrative (LLM if available; otherwise offline)
    try:
        user_payload = {
            "place": place,
            "metrics": metrics,
            "instructions": "Write a concise, business-friendly restaurant insight. Use bullet points for 'Top praises' and 'Top issues' if clear.",
            "sample_positive_reviews": pos_q[:5],
            "sample_negative_reviews": neg_q[:5],
        }
        messages = [
            {"role":"system","content":"You are ReviewLens, an insight writer for F&B operators. Output STRICT JSON with keys: executive_summary (string), praises (array of strings or objects), issues (array of strings or objects), suggestions (array of strings or objects), sampleQuotes{positive[],negative[]}."},
            {"role":"user","content": json.dumps(user_payload)},
        ]
        sea = await sealion_chat_json(messages)
        sea.setdefault("praises", [])
        sea.setdefault("issues", [])
        sea.setdefault("suggestions", [])
        sea.setdefault("sampleQuotes", {"positive": pos_q[:5], "negative": neg_q[:5]})
        # Force our signed TF-IDF lists into the narrative so UI shows correct polarity
        sea["praises"] = praises_tfidf or sea["praises"]
        sea["issues"]  = issues_tfidf  or sea["issues"]
        narrative = {"engine": "model+sealion", **sea}
    except Exception as e:
        print("SEA-LION failed, using offline fallback:", repr(e))
        offline = offline_narrative(place, metrics, pos_texts, neg_texts)
        # overwrite with signed lists
        offline["praises"] = praises_tfidf or offline.get("praises", [])
        offline["issues"]  = issues_tfidf  or offline.get("issues", [])
        narrative = {"engine": "model+offline", **offline}

    # Build strengths/pain points strictly from signed lists
    strengths   = [x["label"] for x in praises_tfidf]
    pain_points = [x["label"] for x in issues_tfidf]

    stance = "positive" if pos_pct >= 55 else ("negative" if neg_pct > 45 else "mixed")
    key_summary = (
        f"Reviews for {cleaned} are {stance} overall. Diners frequently highlight strengths like "
        f"{', '.join([s for s in strengths if s][:2]) if strengths else 'signature dishes and friendly staff'}; "
        f"however, recurring issues include {', '.join([p for p in pain_points if p][:2]) if pain_points else 'wait times and consistency on busy nights'}. "
        f"Average Google rating is {avg_rating:.2f}★ across {n} reviews."
    )

    # Priority bucketing (same as before) ...
    HIGH_KWS = ["wait","queue","slow","cold","undercooked","overcooked","rude","service","missing order","cancelled","late","dirty"]
    MED_KWS  = ["menu","menu clarity","allergen","spicy","payment","google pay","apple pay","crowded","noisy","reservation","booking"]
    def bucketize(txt: str) -> str:
        low = txt.lower()
        if any(k in low for k in HIGH_KWS): return "high"
        if any(k in low for k in MED_KWS):  return "medium"
        return "low"

    requested_improvements = []
    for s in narrative.get("suggestions", []):
        requested_improvements.append(s.get("action") if isinstance(s, dict) else str(s))

    high_priority, medium_priority, low_priority = [], [], []
    def issue_to_action(x: str) -> str:
        base = re.sub(r"\(\d+\)","",str(x)).strip()
        if not base: return ""
        bl = base.lower()
        if "wait" in bl: return "Reduce wait times during peak hours (add floor runner / queue board)."
        if "service" in bl or "staff" in bl: return "Coach FOH on greeting/acknowledgement; add peak-time staffing."
        if "cold" in bl: return "Tighten pass checks: serve hot items immediately; use heat lamps prudently."
        if "reservation" in bl or "booking" in bl: return "Enable simple booking/waitlist to avoid walk-in congestion."
        return f"Address recurring issue: {base}"

    for iss in pain_points:
        action = issue_to_action(iss)
        if action:
            b = bucketize(action)
            (high_priority if b=="high" else medium_priority if b=="medium" else low_priority).append(action)

    if not high_priority:
        high_priority.append("Reduce peak-hour wait times; add a visible queue system and assign a floor greeter.")
    if not medium_priority:
        medium_priority.append("Improve menu clarity & allergy markers; offer quick ‘best-sellers’ panel.")
    if not low_priority:
        low_priority.append("Polish ambience cues (music volume, table resets) between seatings.")

    score_distribution = {
        "1-star": hist_pct.get("1", 0),
        "2-star": hist_pct.get("2", 0),
        "3-star": hist_pct.get("3", 0),
        "4-star": hist_pct.get("4", 0),
        "5-star": hist_pct.get("5", 0),
    }

    owner_voice = (
        f"Tips for {cleaned}: double-down on what guests already love "
        f"({', '.join([s for s in strengths if s][:3]) or 'signature dishes and warm service'}) "
        f"and fix the two most visible friction points first "
        f"({', '.join([p for p in pain_points if p][:3]) or 'wait times and consistency on busy nights'}). "
        f"Right now we’re at {pos_pct:.0f}% positive vs {neg_pct:.0f}% negative — a focused 14-day ops sprint "
        f"should move rating and sentiment noticeably."
    )

    owner_kpis = {
        "avg_rating": round(avg_rating, 2),
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "sample_size": n
    }

    next_14_days = [
        "Peak-hour playbook: assign a door greeter + visible queue board; target <5 min acknowledgement.",
        "Pass discipline: hot dishes leave the pass within 60 seconds; add heat-lamp time limits.",
        "Menu clarity: add top-5 best-sellers panel and allergy markers this week.",
        "Shift brief: 3 key lines for staff to handle delays politely and consistently."
    ]

    return {
        "place": cleaned,
        "summary_of_key_insights": key_summary,
        "quantitative_metrics": {
            "overall_rating_observed": round(avg_rating, 2),
            "score_distribution_percent": score_distribution,
            "sentiment_breakdown_percent": {"positive": pos_pct, "neutral": neu_pct, "negative": neg_pct},
            "data_sample_size": n
        },
        "data_methodology_overview": {
            "source": "Google Maps diner reviews (CSV)",
            "model": "TF-IDF + Logistic Regression (your trained model)",
            "notes": "Predictions aggregated; narrative optionally via SEA-LION; signed TF-IDF used to separate praises (positive-dominant) vs issues (negative-dominant).",
        },
        "key_diner_pain_points": [f'{x["label"]} ({x["count"]})' for x in issues_tfidf][:8],
        "frequently_requested_improvements": requested_improvements[:8],
        "strengths_and_positive_aspects": [f'{x["label"]} ({x["count"]})' for x in praises_tfidf][:8],
        "prioritized_action_recommendations": {
            "high_priority": high_priority[:6],
            "medium_priority": medium_priority[:6],
            "low_priority": low_priority[:6]
        },
        "trends_and_observations": [
            f"Sentiment is {stance} (pos {pos_pct:.0f}% / neu {neu_pct:.0f}% / neg {neg_pct:.0f}%).",
            "Price/value and service consistency commonly drive sentiment swings.",
            "Signature dishes praised should be protected via SOPs during peak periods."
        ],
        "owner_perspective": {"summary": owner_voice, "kpis": owner_kpis, "next_14_days_checklist": next_14_days},
        "conclusion": narrative.get("executive_summary", key_summary),
        "sample_quotes": narrative.get("sampleQuotes", {"positive": pos_q[:3], "negative": neg_q[:3]}),
        "engine": narrative.get("engine", "model+offline")
    }

# =========================
# /api/insights
# =========================
@app.get("/api/insights")
async def insights(place: str):
    cleaned = english_only_place(place)
    sub = df[df["clean_place"].str.casefold() == cleaned.casefold()]
    if sub.empty:
        raise HTTPException(404, f"No reviews for {cleaned}")
    if not (sentiment_model and vectorizer):
        raise HTTPException(500, "Sentiment model not loaded; cannot compute insights")

    sub = sub.copy()
    texts = sub["review"].astype(str).tolist()
    labels = predict_sentiments(texts)
    sub.loc[:, "predicted_sentiment"] = labels

    n = len(sub)
    avg_rating = float(sub["rating"].mean())
    counts = aggregate_sentiment(labels)
    pos = counts.get("positive", 0); neg = counts.get("negative", 0); neu = counts.get("neutral", 0)
    pos_pct = (pos / n * 100) if n else 0.0
    neg_pct = (neg / n * 100) if n else 0.0
    neu_pct = (neu / n * 100) if n else 0.0

    pos_texts = sub.loc[sub["predicted_sentiment"] == "positive", "review"].astype(str).tolist()
    neg_texts = sub.loc[sub["predicted_sentiment"] == "negative", "review"].astype(str).tolist()

    pos_stats = _collect_term_stats(pos_texts)
    neg_stats = _collect_term_stats(neg_texts)
    praises_tfidf, issues_tfidf = _dominance_rank(
        pos_stats, neg_stats, prefer_bigrams=True, min_support=2, min_ratio=1.6, topn=10
    )

    metrics = {
        "place": place,
        "cleanedPlace": cleaned,
        "num_reviews": n,
        "avg_rating": round(avg_rating, 2),
        "predicted_sentiment_counts": counts,
        "pos_pct": round(pos_pct, 2),
        "neg_pct": round(neg_pct, 2),
        "neu_pct": round(neu_pct, 2),
    }

    USE_SEALION = os.getenv("USE_SEALION", "1") not in ("0","false","False")
    if USE_SEALION and SEALION_KEY:
        try:
            user_payload = {
                "place": place,
                "metrics": metrics,
                "instructions": "Write a concise, business-friendly restaurant insight. Use bullet points for 'Top praises' and 'Top issues' if clear.",
                "sample_positive_reviews": pos_texts[:5],
                "sample_negative_reviews": neg_texts[:5],
            }
            messages = [
                {"role":"system","content":"You are ReviewLens, an insight writer for F&B operators. Output STRICT JSON with keys: executive_summary (string), praises (array of strings or objects), issues (array of strings or objects), suggestions (array of strings or objects), sampleQuotes{positive[],negative[]}."},
                {"role":"user","content": json.dumps(user_payload)},
            ]
            sea = await sealion_chat_json(messages)
            sea.setdefault("praises", [])
            sea.setdefault("issues", [])
            sea.setdefault("suggestions", [])
            sea.setdefault("sampleQuotes", {"positive": pos_texts[:5], "negative": neg_texts[:5]})
            # override with signed lists
            sea["praises"] = praises_tfidf or sea["praises"]
            sea["issues"]  = issues_tfidf  or sea["issues"]
            return {"engine":"model+sealion", "metrics": metrics, "narrative": sea}
        except Exception as e:
            print("SEA-LION failed, using offline fallback:", repr(e))

    offline = offline_narrative(place, metrics, pos_texts, neg_texts)
    offline["praises"] = praises_tfidf or offline.get("praises", [])
    offline["issues"]  = issues_tfidf  or offline.get("issues", [])
    stance = "Positive" if metrics["pos_pct"] >= 55 else ("Negative" if metrics["neg_pct"] > 45 else "Mixed")
    offline["overall"] = {"label": stance, "score": int(round(metrics["pos_pct"])), "rationale": offline.get("executive_summary","")}
    return {"engine":"model+offline", "metrics": metrics, "narrative": offline}
