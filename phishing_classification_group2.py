import os
import html
import re
import json
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
from transformers import pipeline
from google import genai
from typing import Dict, List

# ============================================================
# CONFIG & CONSTANTS
# ============================================================

GEMINI_MODEL_NAME = "gemini-2.5-flash"

AUDIENCE_DEFAULT = (
    "volwassenen of studenten die regelmatig e-mails krijgen van bank, overheid en werk, "
    "maar zich niet dagelijks bezighouden met IT-beveiliging. Gebruik een rustige, "
    "duidelijke toon en leg stap voor stap uit wat er verdacht is."
)

PHISHING_CUES: Dict[str, List[str]] = {
    "Urgency": ["urgent", "immediately", "asap", "now", "final notice", "last chance", "today"],
    "Credential Harvesting": ["password", "verify", "verification", "login", "log in", "sign in", "authenticate",
                              "reset", "account details"],
    "Suspicious Links": ["http", "https", "bit.ly", "tinyurl", "short url", "verify-", "update-", "secure-",
                         "click here"],
    "Threats": ["suspended", "terminated", "blocked", "compromised", "legal action", "fine", "penalty"],
    "Financial Bait": ["winner", "lottery", "prize", "refund", "payment", "invoice", "gift card", "bank transfer"],
    "Impersonation": ["account team", "support team", "security team", "microsoft", "paypal", "dhl", "bpost", "apple",
                      "banking"],
    "Generic Phishing": ["click here", "act now", "free", "offer", "security alert", "access your account"],
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
# 1. MODEL TRAINING
# ============================================================

SPAM_DATA = [
    "Click here to verify your password immediately.",
    "URGENT: Your account will be suspended today.",
    "Final notice: you won a prize! Claim now.",
    "Security alert: login to verify your credentials.",
    "Your DHL package is waiting. Update your delivery info.",
]
HAM_DATA = [
    "Here is the report you asked for.",
    "Can we schedule a meeting tomorrow?",
    "Happy birthday! Hope you enjoy your day!",
    "Attached the minutes of last meeting.",
    "Thank you for your help last week."
]

X_train = SPAM_DATA + HAM_DATA
y_train = [1] * len(SPAM_DATA) + [0] * len(HAM_DATA)


@st.cache_resource(show_spinner="Loading Analysis Models...")
def load_surrogate_and_shap():
    # 1. Train Surrogate Model (TF-IDF + LR)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=3000, stop_words="english")
    Xv = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xv, y_train)

    feature_names = np.array(vec.get_feature_names_out())

    # 2. Build SHAP Explainer
    background = Xv[np.random.choice(Xv.shape[0], min(10, Xv.shape[0]), replace=False)]
    explainer = shap.LinearExplainer(clf, background)

    return vec, clf, explainer, feature_names


vectorizer, surrogate_clf, explainer, FEATURE_NAMES = load_surrogate_and_shap()


def get_shap_vector(text):
    X_new = vectorizer.transform([text])
    values = explainer.shap_values(X_new)
    if isinstance(values, list):
        return values[1][0] if len(values) > 1 else values[0][0]
    elif isinstance(values, np.ndarray):
        return values[0]
    return values


def build_shap_summary_json(text, shap_vec, top_k=5):
    X_new = vectorizer.transform([text])
    x_dense = X_new.toarray()

    idx_sorted = np.argsort(-shap_vec)  # Sort descending

    important_words = []
    for i in idx_sorted:
        # Filter for positive contributions (words that indicate phishing) present in text
        if shap_vec[i] > 0 and x_dense[0, i] > 0:
            important_words.append({
                "word": FEATURE_NAMES[i],
                "importance_score": float(shap_vec[i])
            })
        if len(important_words) >= top_k:
            break

    return {"risk_indicators": important_words}


# ============================================================
# 2. HUGGINGFACE MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    return pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")


hf_clf = load_hf_pipeline()


def predict_phishing_score(text: str):
    out = hf_clf(text, truncation=True)
    if isinstance(out, list):
        out = out[0]
    if isinstance(out, list):  # handle nested lists
        out = out[0]

    return out


# ============================================================
# 3. GEMINI INTEGRATION
# ============================================================

@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def generate_explanation(email_text, teacher_output, shap_summary):
    client = get_gemini_client()
    if not client:
        return "⚠️ **Error:** Geen API Key gevonden. Stel de GEMINI_API_KEY environment variable in."

    # Format data for prompt
    shap_json = json.dumps(shap_summary, indent=2)
    teacher_json = json.dumps(teacher_output, indent=2)

    prompt = f"""
Je bent een expert in digitale veiligheid. Je analyseert een e-mail om een gebruiker te helpen begrijpen of deze veilig of verdacht is.

DOELGROEP: {AUDIENCE_DEFAULT}

GEGEVENS:
1. E-MAIL TEKST: "{email_text}"
2. AI-MODEL SCORE: {teacher_json}
3. VERDACHTE WOORDEN (SHAP analyse): {shap_json}

TAKEN:
1. Begin met één duidelijke conclusie: Lijkt dit betrouwbaar of verdacht?
2. Leg in 1 à 2 alinea's uit WAAROM. Verwijs specifiek naar de inhoud van de e-mail (bijv. dwingende toon, vreemde links, vragen om geld).
3. Geef 3 korte, concrete tips wat de gebruiker nu moet doen (bijv. "Klik nergens op", "Markeer als spam/phishing").

REGELS:
- Schrijf in eenvoudig, vriendelijk Nederlands.
- Gebruik Markdown voor opmaak (vetgedrukt, bulletpoints).
- Geef GEEN technische uitleg over SHAP of AI-scores, focus op de inhoud.
"""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Kon geen uitleg genereren: {e}"


# ============================================================
# 4. HIGHLIGHTING LOGIC
# ============================================================

def highlight_text(email_text, shap_vec, top_k=10):
    positive_idx = np.where(shap_vec > 0)[0]
    if positive_idx.size == 0:
        return html.escape(email_text), []

    sorted_idx = positive_idx[np.argsort(-shap_vec[positive_idx])]
    top_idx = sorted_idx[:top_k]
    top_words = FEATURE_NAMES[top_idx]

    found_cues = set()
    cue_summary = {}

    for word_raw, idx in zip(top_words, top_idx):
        word = word_raw.lower()
        score = float(shap_vec[idx])

        # Check taxonomy
        for category, cues in PHISHING_CUES.items():
            for cue in cues:
                if cue in word:
                    found_cues.add(category)
                    cue_summary[word] = score

    rendered = html.escape(email_text)

    words_to_highlight = sorted(cue_summary.keys(), key=len, reverse=True)

    for word in words_to_highlight:
        escaped_word = html.escape(word)
        pattern = re.compile(re.escape(escaped_word), re.IGNORECASE)
        rendered = pattern.sub(
            lambda m: (
                f"<span style='background-color:#ffe1e1; border-bottom: 2px solid #ff4b4b; "
                f"border-radius:4px; padding:0 2px; font-weight:bold' title='Verdacht woord'>{m.group(0)}</span>"
            ),
            rendered
        )

    # Generate static learning points based on categories found
    learning_points = []
    for cat in found_cues:
        if cat == "Urgency":
            learning_points.append("⚠️ **Urgentie:** Oplichters creëren haast zodat je niet nadenkt.")
        elif cat == "Credential Harvesting":
            learning_points.append(
                "🔒 **Gegevens:** Een betrouwbaar bedrijf vraagt nooit zomaar om je wachtwoord via mail.")
        elif cat == "Suspicious Links":
            learning_points.append(
                "🔗 **Links:** Klik niet zomaar. Beweeg je muis over de link om het echte adres te zien.")
        elif cat == "Financial Bait":
            learning_points.append("💰 **Geld:** Klinkt het te mooi om waar te zijn? Dan is het dat meestal ook.")

    return rendered, learning_points


# ============================================================
# 5. MAIN APP UI
# ============================================================

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Phishing Detector")
st.markdown("""
Plak hieronder een e-mail die je niet vertrouwt. 
Onze AI analyseert de tekst en legt uit **waarom** het wel of geen phishing is.
""")

# Input Area
user_email = st.text_area("Plak de e-mail hier:", height=250,
                          placeholder="Onderwerp: ...\nBeste klant, ...")

if st.button("🔍 Analyseer E-mail", type="primary"):
    if not user_email.strip():
        st.warning("Plak eerst een e-mail in het vak hierboven.")
    else:
        with st.spinner("AI is de e-mail aan het lezen..."):
            # 1. Technical Analysis
            shap_vec = get_shap_vector(user_email)
            prediction_raw = predict_phishing_score(user_email)
            shap_summary = build_shap_summary_json(user_email, shap_vec)

            # 2. Textual Generation (Gemini)
            explanation = generate_explanation(user_email, prediction_raw, shap_summary)

            # 3. Visual Highlighting
            highlighted_text, learning_points = highlight_text(user_email, shap_vec)

        # --- RESULTS DISPLAY ---

        st.divider()

        # Section A: AI Explanation
        st.subheader("🤖 Oordeel van de Expert")
        st.markdown(explanation)

        # Section B: Visual Evidence
        st.subheader("🔎 Waar keek de AI naar?")
        st.caption("De AI heeft deze woorden gemarkeerd als verdacht:")
        st.markdown(
            f"<div style='white-space:pre-wrap; border:1px solid #ddd; padding:20px; border-radius:10px; background-color: #ffffff; line-height: 1.6;'>{highlighted_text}</div>",
            unsafe_allow_html=True,
        )

        # Section C: Key Takeaways
        if learning_points:
            st.subheader("📚 Belangrijkste lessen")
            for point in learning_points:
                st.markdown(f"- {point}")

# Sidebar info
with st.sidebar:
    st.info("""
    **Hoe werkt dit?**
    1. Een **AI Model** scant de tekst op patronen van phishing.
    2. **SHAP** analyseert welke specifieke woorden de beslissing beïnvloeden.
    3. **Gemini (Google)** vertaalt deze technische data naar een begrijpelijke uitleg.
    """)
    st.warning("**Let op:** Geen enkel systeem is 100% waterdicht. Blijf altijd zelf alert.")