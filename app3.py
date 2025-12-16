import os
import csv
import time
import html
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
from transformers import pipeline
import json


@st.cache_data
def load_precomputed_explanations():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "pre_generated_mails.json")

    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return {}


precomputed_data = load_precomputed_explanations()
# ============================================================
# CONFIG
# ============================================================

LOG_FILE = "logs_experiment.csv"

EXPLANATION_CONDITIONS = {
    "Geen uitleg (control)": "no_xai",
    "Alleen highlights": "highlight_only",
    "Highlights + tekstuele uitleg (incl. Gemini)": "full_xai",
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
        "text": """Sender: Help Center info.help-center.co@gmail.com
Subject: Netflix : We're having some trouble with your current billing information
HELLO,   Please note that, your monthly payment has been failed. Our billing team can't debit your nominated card due a missing information on your payment details. Please verify your details again to avoid any delay on your service.   https://netflix.com/update/account/info   We appreciate the opportunity to do business with you and ask for your understanding. Netflix support
""",
        "ground_truth": 1,  # 1 = phishing, 0 = legit
    },
{
        "id": "mail2",
        "text": """Sender: hr-dept@perl.org
Subject: Perl Developer (onsite)
Hello, 
There is a job application available.
Online URL for this job: https://jobs.perl.org/job/7898

To subscribe to this list, send mail to jobs-subscribe@perl.org.
To unsubscribe, send mail to jobs-unsubscribe@perl.org.
""",
        "ground_truth": 0,
    },
    {
        "id": "mail3",
        "text": """Sender: Prize Allocation Dept <winners@lucky-draw-global.com> 
Subject: CLAIM YOUR $500 GIFT CARD NOW
Congratulations!
Your email address was randomly selected in our weekly draw. You've won a $500 gift card.
To claim your prize, simply reply to this email with your:
1.	Full Name
2.	Mailing Address
3.	Phone Number
Do not delay, or your prize will be forfeited.
""",
        "ground_truth": 1,
    },

    {
        "id": "mail4",
        "text": """Sender: support@eprints.dinus.ac.id
Subject: Your Meta wallet will be suspended
Verify your Meta Wallet. Our system has shown that your MetaMask wallet has not yet been verified, this verification can be done easily via the button below. Unverified accounts will be suspended on: Wednesday, 09 November 2022. We are sorry for any inconvenience caused by this, but please note that our intention is to keep our customers safe and happy. Safety is and will remain our highest priority. 
[Verify My MetaMask]
Thank you for choosing us.   
Best regards
""",
        "ground_truth": 1,
    },
    {
        "id": "mail5",
        "text": """Sender: cnnalerts@mail.cnn.com
Subject: CNN Alerts
Maintaining security in Diyala province north of Baghdad will be impossible if U.S. troops are withdrawn from Iraq, according to a U.S. senior ground commander there.
FULL STORY: cnn.com/2022/WORLD/meast/07/05/iraq.commander/


You have agreed to receive this email from cnn.com as a result of your cnn.com preference settings.
To alter your alert criteria or frequency or to unsubscribe from receiving custom email alerts, reply to this email.
Refer a friend or colleague to CNN's FREE personalized alerting service!

© 2022 Cable News Network, LP, LLLP.
A Time Warner Company. All Rights Reserved.
""",
        "ground_truth": 0,
    },
    {
        "id": "mail6",
        "text": """Sender: Security Alert <security@verify-bank-access.net> 
Subject: Alert: Unusual login attempt detected.
We noticed some unusual activity in your account. A sign-in attempt was made from an unrecognized device in Lagos, Nigeria.
Log in to review recent transactions. [Review Activity Now] (http://secure-account-verify-signin.com)

""",
        "ground_truth": 1,
    },
{
        "id": "mail7",
        "text": """Sender: Billing Services <invoicing@service-payment-overdue.com> 
Subject: OVERDUE: Invoice #9982
Your invoice is attached. Please review and pay promptly to avoid penalties.
We have attempted to charge your card on file twice. Please open the attached PDF "Invoice_9982.pdf" to view the outstanding balance and payment instructions.
""",
        "ground_truth": 1,
    },
    {
        "id": "mail8",
        "text": """Sender: Subscription Management <billing@saas-tool.com> 
        Subject: Renewal Success
        Dear Jordan, your subscription has been successfully renewed. Thank you for your continued support.
        You can view your receipt and subscription details on your dashboard.
        """,
        "ground_truth": 0,
    },
    {
        "id": "mail9",
        "text": """Sender: orders@fedex.com 
    Subject: Your Order #34296
    Dear Casey, thank you for your purchase. Your order will be shipped soon.
    You can track your package via FedEx using the link below by providing the next tracking number: 94829285295711083.
    Thank you for shopping with us!
    [https://www.fedex.com/nl-be/tracking.html]
    """,
        "ground_truth": 0,
    },
{
        "id": "mail10",
        "text": """Sender: Bank Alerts <notifications@belfius-message.co> 
Subject: New Message Center Notification
You have a new security message from your bank. Urgently view the message for further action.
Click here to read it.  ( https://beIfius.be/notifications )
""",
        "ground_truth": 1,
    },
]
# ============================================================
# 1. SURROGATE MODEL (TF-IDF + LR) + SHAP VOOR HIGHLIGHTS
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

X_train = SPAM + HAM
y_train = [1] * len(SPAM) + [0] * len(HAM)


@st.cache_resource(show_spinner=True)
def train_surrogate_model():
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=3000,
        stop_words="english",
    )
    Xv = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xv, y_train)
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
# 2. HUGGINGFACE MODEL VOOR PREDICTIE (VOOR UI LABEL)
# ============================================================

@st.cache_resource(show_spinner=True)
def load_hf_pipeline_ui():
    # Klein sms spam model; vervang eventueel door ander HF-model
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")


hf_clf_ui = load_hf_pipeline_ui()


def hf_predict_ui(text: str):
    out = hf_clf_ui(text, truncation=True)[0]
    label = out["label"]
    score = float(out["score"])
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


def build_shap_summary_from_linear(X_new, shap_vec: np.ndarray, feature_names: np.ndarray, top_k: int = 5):
    """
    Bouw een SHAP-samenvatting in hetzelfde formaat als app2,
    zodat we die kunnen doorgeven aan Gemini.
    """
    x_dense = X_new.toarray()
    shap_vals = shap_vec

    idx_sorted_pos = np.argsort(-shap_vals)
    idx_sorted_neg = np.argsort(shap_vals)

    positive = []
    for i in idx_sorted_pos:
        if shap_vals[i] <= 0:
            continue
        if x_dense[0, i] == 0:
            continue
        positive.append({"token": feature_names[i], "shap": float(shap_vals[i])})
        if len(positive) >= top_k:
            break

    negative = []
    for i in idx_sorted_neg:
        if shap_vals[i] >= 0:
            continue
        if x_dense[0, i] == 0:
            continue
        negative.append({"token": feature_names[i], "shap": float(shap_vals[i])})
        if len(negative) >= top_k:
            break
    print(positive,negative)
    return {"positive": positive, "negative": negative}


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
# 5. GEMINI CLIENT + HELPER-FUNCTIES (GEBASEERD OP app2.py)
# ============================================================

GEMINI_MODEL_NAME = "gemini-2.5-flash"


@st.cache_resource
def get_gemini_client():
    """
    Maakt een Gemini client. Verwacht GEMINI_API_KEY in je environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is niet gezet.\n"
            "Doe in je terminal bv.:\n"
            '  export GEMINI_API_KEY="AIza..."\n'
        )
    return genai.Client(api_key=api_key)


def generate_with_gemini(prompt: str, max_tokens: int = 300) -> str:
    """
    Minimalistische Gemini 2.5 Flash wrapper.
    """
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "Gemini kon geen uitleg geven (lege response)."

    except Exception as e:
        return f"Kon geen uitleg genereren via Gemini API: {e}"


@st.cache_resource
def load_teacher_pipeline():
    """
    Teacher-model zoals in app2.py, gebruikt voor Gemini-uitleg.
    """
    teacher = pipeline(
        task="text-classification",
        model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        tokenizer="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        top_k=None,
    )
    return teacher


def get_teacher_raw_and_score(teacher, text: str):
    out = teacher(text)

    if isinstance(out, list):
        if len(out) > 0 and isinstance(out[0], list):
            logits = out[0]
        else:
            logits = out
    else:
        logits = [out]

    score_phish = None
    for item in logits:
        lbl = item["label"].lower()
        if "spam" in lbl or "phish" in lbl:
            score_phish = float(item["score"])
            break

    if score_phish is None:
        score_phish = float(sorted(logits, key=lambda x: -x["score"])[0]["score"])

    return logits, score_phish


def build_explanation_prompt(email_text, teacher_raw, shap_summary, audience):
    shap_json = json.dumps(shap_summary, indent=2, ensure_ascii=False)
    teacher_str = json.dumps(teacher_raw, indent=2, ensure_ascii=False)

    prompt = f"""
Je bent een expert in digitale veiligheid. Je helpt mensen om verdachte e-mails te herkennen,
zodat ze zich beter kunnen beschermen. Je geeft alleen defensieve uitleg
(hoe iemand zich kan beschermen), geen instructies om zelf fraude of misleiding te plegen.

DOELGROEP: {audience}

Je krijgt hieronder:
- de tekst van een e-mail,
- een modelvoorspelling of de e-mail verdacht is,
- en een lijst met woorden die het model belangrijk vindt (SHAP).

TAKEN:
1. Begin met één duidelijke zin: lijkt deze e-mail eerder betrouwbaar of eerder verdacht?
2. Leg daarna in 1 à 2 korte alinea's uit waarom, met verwijzingen naar concrete elementen
   uit de e-mail (toon, dreigende taal, tijdsdruk, beloftes van prijzen, links, vragen om gegevens, ...).
3. Sluit af met 2 à 3 praktische tips speciaal voor deze doelgroep.

REGELS:
- Schrijf in eenvoudig NEDERLANDS.
- Schrijf alleen vloeiende tekst (geen opsommingstekens, geen headings).
- Herhaal de onderstaande instructies niet. Geef alleen de uitleg over de e-mail.

====================================
E-MAIL:
------------------------------------
{email_text}

====================================
MODEL-UITVOER (VERDACHTE-BERICHTEN CLASSIFIER):
------------------------------------
{teacher_str}

====================================
BELANGRIJKSTE WOORDEN VOLGENS SHAP:
------------------------------------
{shap_json}

====================================
UITLEG:
"""
    return prompt.strip()


def generate_explanation_with_gemini(
    email_text: str,
    shap_summary: Dict,
    audience: str,
    max_new_tokens: int = 300,
):
    teacher = load_teacher_pipeline()
    teacher_raw, _ = get_teacher_raw_and_score(teacher, email_text)
    #print(shap_summary)

    prompt = build_explanation_prompt(email_text, teacher_raw, shap_summary, audience)
    gen = generate_with_gemini(prompt, max_tokens=max_new_tokens)

    if "UITLEG:" in gen:
        gen = gen.split("UITLEG:")[-1].strip()

    return gen.strip()


# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Phishing XAI Experiment (met Gemini)", page_icon="🛡️", layout="wide")
st.title("Phishing XAI Experiment – Spamfilter met uitleg")

with st.sidebar:
    st.header("Deelnemer & conditie")
    participant_id = st.text_input("Participant ID")
    cond_label = st.selectbox("Uitlegconditie", list(EXPLANATION_CONDITIONS.keys()))
    condition = EXPLANATION_CONDITIONS[cond_label]
    st.write("Condities:")
    st.markdown(
        "- **Geen uitleg**: alleen AI-label\n"
        "- **Alleen highlights**: verdachte woorden gemarkeerd\n"
        "- **Highlights + tekst**: visueel + categorie-uitleg + Gemini-tekst"
    )
    if st.button("🔄 Reset naar eerste mail"):
        st.session_state["stim_idx"] = 0
        st.session_state["start_time"] = None

# Init state
if "stim_idx" not in st.session_state:
    st.session_state["stim_idx"] = 0
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None
# Cache voor Gemini-output (blijft staan bij reruns)
if "gemini_cache" not in st.session_state:
    st.session_state["gemini_cache"] = {}


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

# AI-predictie (HuggingFace) voor snelle feedback
is_spam_ai, score_ai, raw_label = hf_predict_ui(email_text)
ai_color = "#d9534f" if is_spam_ai else "#5cb85c"
ai_label_str = "SPAM/PHISHING" if is_spam_ai else "NOT SPAM"



# Surrogate SHAP-explainer (LinearExplainer)
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
    # ------------------------------------------------------------
    # 1. VISUAL HIGHLIGHTS & CATEGORIZATION
    # ------------------------------------------------------------
    highlighted_html, cue_summary, explanation_md, learning_points = categorize_phishing_cues(
        email_text, shap_vec, top_k=10
    )

    st.subheader("🔎 Verdachte woorden (highlights)")

    # Render the email text with highlighted words
    st.markdown(
        f"<div style='white-space:pre-wrap; border:1px solid #ddd; padding:15px; border-radius:5px; background-color: #f9f9f9;'>{highlighted_html}</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # 2. GEMINI TEXT EXPLANATION (SINGLE VERSION)
    # ------------------------------------------------------------
    st.subheader("🤖 Uitleg van de veiligheidsexpert")

    # Retrieve the specific explanation for the current email ID
    # We default to a standard message if the ID is missing
    current_explanations = precomputed_data.get(stim_id, {})

    # Get the 'older' text. If missing, show a fallback message.
    explanation_text = current_explanations.get(
        "older",
        "⚠️ Er is nog geen uitleg beschikbaar voor deze e-mail."
    )

    # Display the explanation directly (No tabs)
    st.markdown(explanation_text)

    # ------------------------------------------------------------
    # 3. EXTRA: LEARNING POINTS (OPTIONAL)
    # ------------------------------------------------------------
    if learning_points:
        st.info("**Belangrijkste lessen:**\n\n" + "\n".join([f"- {p}" for p in learning_points]))

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
