import os
import csv
import time
import html
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
from transformers import pipeline

# ============================================================
# CONFIG
# ============================================================

LOG_FILE = "logs_experiment.csv"

EXPLANATION_CONDITIONS = {
    "Geen uitleg (control)": "no_xai",
    "Alleen highlights": "highlight_only",
    "Highlights + tekstuele uitleg": "full_xai",
}

# ===== PHISHING CUE TAXONOMY =====

PHISHING_CUES: Dict[str, List[str]] = {
    "Urgency": [
        "urgent", "immediately", "asap", "now", "final notice", "last chance", "today"
    ],
    "Credential Harvesting": [
        "password", "verify", "verification", "login", "log in",
        "sign in", "authenticate", "reset", "account details"
    ],
    "Suspicious Links": [
        "http", "https", "bit.ly", "tinyurl", "short url",
        "verify-", "update-", "secure-", "click here"
    ],
    "Threats": [
        "suspended", "terminated", "blocked", "compromised",
        "legal action", "fine", "penalty"
    ],
    "Financial Bait": [
        "winner", "lottery", "prize", "refund", "payment",
        "invoice", "gift card", "bank transfer"
    ],
    "Impersonation": [
        "account team", "support team", "security team",
        "microsoft", "paypal", "dhl", "bpost", "apple", "banking"
    ],
    "Generic Phishing": [
        "click here", "act now", "free", "offer",
        "security alert", "access your account"
    ],
}

CATEGORY_TEXT = {
    "Urgency": "De mail gebruikt urgente taal om je snel te laten reageren zonder na te denken.",
    "Credential Harvesting": "De mail probeert je inloggegevens of wachtwoorden te ontfutselen.",
    "Suspicious Links": "De mail bevat verdachte of verkorte links die naar nepwebsites kunnen leiden.",
    "Threats": "De mail dreigt met blokkering, boetes of andere negatieve gevolgen om druk op je te zetten.",
    "Financial Bait": "De mail lokt je met geld, prijzen of terugbetalingen.",
    "Impersonation": "De mail doet zich voor als een bekend bedrijf of officiële organisatie.",
    "Generic Phishing": "De mail bevat typische advertising-/phishingfrasen die je moeten laten klikken.",
}

# ============================================================
# STIMULI – voeg hier later meer echte e-mails toe
# ============================================================

STIMULI = [
    {
        "id": "mail1",
        "text": """From: Microsoft Support <no-reply@secure-micr0soft.com>
Subject: Final notice - verify your account immediately

Dear user,
Your account will be suspended TODAY. Click here to verify your password and keep your access:
https://bit.ly/verify-365

Regards,
Security Team""",
        "ground_truth": 1,  # 1 = phishing, 0 = legit
    },
    {
        "id": "mail2",
        "text": """Hi,
Here is the agenda for tomorrow's project meeting. Let me know if anything needs to be changed.

Best,
Anne""",
        "ground_truth": 0,
    },
    {
        "id": "mail3",
        "text": """Dear customer,
We noticed unusual activity on your PayPal account. Log in now to confirm your identity or your account will be blocked.

Sincerely,
PayPal Security""",
        "ground_truth": 1,
    },
    {
        "id": "mail4",
        "text": """Hello,
Thanks again for your help last week. The updated report is attached in the shared folder.

Kind regards,
Tom""",
        "ground_truth": 0,
    },
]

# ============================================================
# 1. SURROGATE MODEL (TF-IDF + LR) + SHAP
# ============================================================

SPAM = [
    "Click here to verify your password immediately.",
    "URGENT: Your account will be suspended today.",
    "Final notice: you won a prize! Claim now.",
    "Security alert: login to verify your credentials.",
    "Your DHL package is waiting. Update your delivery info.",
]

HAM = [
    "Here is the report you asked for.",
    "Can we schedule a meeting tomorrow?",
    "Happy birthday! Hope you enjoy your day!",
    "Attached the minutes of last meeting.",
    "Thank you for your help last week."
]

X = SPAM + HAM
y = [1] * len(SPAM) + [0] * len(HAM)


@st.cache_resource(show_spinner=True)
def train_surrogate_model():
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=3000,
        stop_words="english",
    )
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xv, y)
    return vec, clf, Xv


vectorizer, surrogate_clf, Xv_train = train_surrogate_model()
FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())


@st.cache_resource(show_spinner=True)
def build_shap_explainer():
    background = Xv_train[
        np.random.choice(Xv_train.shape[0], min(10, Xv_train.shape[0]), replace=False)
    ]
    return shap.LinearExplainer(surrogate_clf, background)


explainer = build_shap_explainer()


def get_shap_vector(X_new):
    values = explainer.shap_values(X_new)
    if isinstance(values, np.ndarray):  # (1, n_features)
        return values[0]
    if isinstance(values, list):
        if len(values) == 1:
            return values[0][0]
        else:
            return values[1][0]  # class 1 = spam
    return values[0]


# ============================================================
# 2. HUGGINGFACE MODEL VOOR PREDICTIE
# ============================================================

@st.cache_resource(show_spinner=True)
def load_hf_pipeline():
    # Klein sms spam model; vervang eventueel door ander HF-model
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")


hf_clf = load_hf_pipeline()


def hf_predict(text: str):
    out = hf_clf(text, truncation=True)[0]
    label = out["label"]
    score = float(out["score"])
    # Normaliseer labels naar spam/not-spam
    is_spam = label.lower() in ["spam", "label_1", "spam sms"]
    return is_spam, score, label


# ============================================================
# 3. EXPLANATION: CATEGORISEREN + HIGHLIGHTS
# ============================================================

def categorize_phishing_cues(email_text: str, shap_vec: np.ndarray, top_k: int = 12):
    positive_idx = np.where(shap_vec > 0)[0]
    if positive_idx.size == 0:
        return (
            html.escape(email_text),
            {},
            "Voor dit bericht vonden we geen sterke phishing-signalen in de tekst.",
            [],
        )

    sorted_idx = positive_idx[np.argsort(-shap_vec[positive_idx])]
    top_idx = sorted_idx[:top_k]
    top_words = FEATURE_NAMES[top_idx]

    cue_summary: Dict[str, Dict[str, float]] = {}

    for word_raw, idx in zip(top_words, top_idx):
        word = word_raw.lower()
        score = float(shap_vec[idx])
        for category, cues in PHISHING_CUES.items():
            for cue in cues:
                if cue in word:
                    cue_summary.setdefault(category, {})
                    cue_summary[category][word] = max(
                        cue_summary[category].get(word, 0.0), score
                    )

    if not cue_summary:
        explanation = (
            "Het model markeert enkele verdachte woorden, "
            "maar ze vallen niet onder onze vooraf gedefinieerde phishingcategorieën."
        )
        learning_points: List[str] = []
    else:
        explanation_lines = [
            "Deze e-mail lijkt phishing omdat het model sterke signalen vond in de volgende categorieën:",
            "",
        ]
        learning_points: List[str] = []

        for category, words_dict in cue_summary.items():
            words_sorted = sorted(words_dict.items(), key=lambda x: -x[1])
            explanation_lines.append(f"### 📌 {category}")
            explanation_lines.append(CATEGORY_TEXT.get(category, ""))
            explanation_lines.append("")
            for w, s in words_sorted:
                explanation_lines.append(f"- **{w}** (SHAP: {s:.3f})")
            explanation_lines.append("")

            if category == "Urgency":
                learning_points.append(
                    "Let op **urgente taal** zoals “immediately”, “today”, “final notice”."
                )
            elif category == "Credential Harvesting":
                learning_points.append(
                    "Wees argwanend bij mails die om **wachtwoorden of login** vragen."
                )
            elif category == "Suspicious Links":
                learning_points.append(
                    "Controleer altijd **links** (verkorte URL's, rare domeinen) voordat je klikt."
                )
            elif category == "Threats":
                learning_points.append(
                    "Dreigingen zoals “account suspended” worden vaak gebruikt om je te pushen."
                )
            elif category == "Financial Bait":
                learning_points.append(
                    "Belooft de mail **geld of prijzen**? Dat is een klassiek phishinglokmiddel."
                )
            elif category == "Impersonation":
                learning_points.append(
                    "Controleer altijd het **echte afzenderadres** bij mails die een bekend bedrijf imiteren."
                )
            elif category == "Generic Phishing":
                learning_points.append(
                    "Zinnen als “click here” of “act now” zijn generieke phishing-signalen."
                )

        explanation = "\n".join(explanation_lines)

    # Highlight text
    rendered = html.escape(email_text)

    for category, words_dict in cue_summary.items():
        for word in words_dict.keys():
            escaped_word = html.escape(word)
            pattern = re.compile(re.escape(escaped_word), re.IGNORECASE)

            rendered = pattern.sub(
                lambda m: (
                    f"<span style='background-color:#ffe1e1;"
                    f"border-radius:4px;padding:0.1em 0.2em;'>"
                    f"{m.group(0)}</span>"
                ),
                rendered,
            )

    return rendered, cue_summary, explanation, learning_points


# ============================================================
# 4. LOGGING
# ============================================================

def append_log(row: Dict):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 5. STREAMLIT UI – EXPERIMENT
# ============================================================

st.set_page_config(page_title="Phishing XAI Experiment", page_icon="🛡️", layout="wide")
st.title("🛡️ Phishing XAI Experiment – Spamfilter met uitleg")

with st.sidebar:
    st.header("Deelnemer & conditie")
    participant_id = st.text_input("Participant ID")
    cond_label = st.selectbox("Uitlegconditie", list(EXPLANATION_CONDITIONS.keys()))
    condition = EXPLANATION_CONDITIONS[cond_label]
    st.write("Condities:")
    st.markdown("- **Geen uitleg**: alleen AI-label\n"
                "- **Alleen highlights**: verdachte woorden gemarkeerd\n"
                "- **Highlights + tekst**: visueel + categorie-uitleg")
    if st.button("🔄 Reset naar eerste mail"):
        st.session_state["stim_idx"] = 0
        st.session_state["start_time"] = None

# Init state
if "stim_idx" not in st.session_state:
    st.session_state["stim_idx"] = 0
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None

idx = st.session_state["stim_idx"]

if idx >= len(STIMULI):
    st.success("Je bent klaar! Alle e-mails zijn afgewerkt.")
    st.stop()

stim = STIMULI[idx]
email_text = stim["text"]
stim_id = stim["id"]
ground_truth = stim["ground_truth"]

# start timer bij eerste render van deze stimulus
if st.session_state["start_time"] is None:
    st.session_state["start_time"] = time.time()

st.subheader(f"E-mail {idx+1} van {len(STIMULI)} (Stimulus: {stim_id})")

# AI-predictie (HuggingFace)
is_spam_ai, score_ai, raw_label = hf_predict(email_text)
ai_color = "#d9534f" if is_spam_ai else "#5cb85c"
ai_label_str = "SPAM/PHISHING" if is_spam_ai else "NOT SPAM"

st.markdown(
    f"""
    <div style="border-left:6px solid {ai_color}; padding:12px; margin-bottom:16px;">
      <b>AI classificatie (HuggingFace model):</b> 
      <span style="color:{ai_color}; font-weight:bold;">{ai_label_str}</span><br>
      <b>Modelscore:</b> {int(score_ai*100)}%<br>
      <small>Ruw model-label: {raw_label}</small>
    </div>
    """,
    unsafe_allow_html=True,
)

# Surrogate SHAP-explainer
X_new = vectorizer.transform([email_text])
shap_vec = get_shap_vector(X_new)

# Explanations afhankelijk van conditie
if condition == "no_xai":
    st.markdown(f"<div style='white-space:pre-wrap'>{html.escape(email_text)}</div>",
                unsafe_allow_html=True)

elif condition == "highlight_only":
    highlighted_html, _, _, _ = categorize_phishing_cues(email_text, shap_vec, top_k=10)
    st.subheader("🔎 Verdachte woorden (highlights)")
    st.markdown(f"<div style='white-space:pre-wrap'>{highlighted_html}</div>",
                unsafe_allow_html=True)

elif condition == "full_xai":
    highlighted_html, cue_summary, explanation_md, learning_points = categorize_phishing_cues(
        email_text, shap_vec, top_k=10
    )
    st.subheader("🔎 Verdachte woorden (highlights)")
    st.markdown(f"<div style='white-space:pre-wrap'>{highlighted_html}</div>",
                unsafe_allow_html=True)

    st.subheader("🧠 Waarom denkt de AI dat dit phishing is?")
    st.markdown(explanation_md)

    st.subheader("📘 Wat kun je hiervan leren?")
    if learning_points:
        for lp in learning_points:
            st.markdown(f"- {lp}")
    else:
        st.markdown("- In deze mail vonden we geen duidelijke phishingcategorieën.")

st.markdown("---")

# Participant response
st.subheader("Jouw oordeel")

user_label = st.radio(
    "Is deze e-mail volgens JOU phishing?",
    options=["Phishing", "Geen phishing"],
    index=None,
)

confidence = st.slider(
    "Hoe zeker ben je van je antwoord?",
    min_value=1, max_value=10, value=7,
)

if st.button("💾 Antwoord opslaan & volgende"):
    if not participant_id:
        st.warning("Vul eerst Participant ID in.")
        st.stop()
    if user_label is None:
        st.warning("Kies eerst of je denkt dat de mail phishing is.")
        st.stop()

    rt = time.time() - st.session_state["start_time"]
    timestamp = datetime.now().isoformat(timespec="seconds")

    row = {
        "participant_id": participant_id,
        "condition": condition,
        "stimulus_id": stim_id,
        "ground_truth": ground_truth,
        "ai_is_spam": int(is_spam_ai),
        "ai_score": score_ai,
        "user_label": 1 if user_label == "Phishing" else 0,
        "confidence_1_10": confidence,
        "reaction_time_sec": round(rt, 3),
        "timestamp": timestamp,
    }
    append_log(row)

    st.success("Antwoord opgeslagen!")

    # naar volgende mail
    st.session_state["stim_idx"] += 1
    st.session_state["start_time"] = None
    st.rerun()
