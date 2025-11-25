import os
import json
import numpy as np
import streamlit as st
from tqdm import tqdm
import shap

from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor

from google import genai


# ============================================================
# 0. GEMINI CLIENT (2.5 FLASH) VOOR UITLEG
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
    Minimalistische Gemini 2.5 Flash wrapper – exact zoals het werkt
    in jouw testscript. Geen candidates/parts.
    """
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        # Nieuw API-formaat → response.text is altijd de juiste output
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        # Fallback als het tóch leeg zou zijn
        return "Gemini kon geen uitleg geven (lege response)."

    except Exception as e:
        return f"Kon geen uitleg genereren via Gemini API: {e}"


# ============================================================
# 1. HUGGING FACE TEACHER (SPAM/PHISHING CLASSIFIER)
# ============================================================

@st.cache_resource
def load_teacher_pipeline():
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


# ============================================================
# 2. SURROGAATMODEL + SHAP
# ============================================================

def _build_surrogate_model_internal(teacher, emails):
    emails = np.array(emails)
    scores = [get_teacher_raw_and_score(teacher, t)[1] for t in emails]
    y_teacher = np.array(scores)

    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(emails)

    student = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    student.fit(X, y_teacher)

    explainer = shap.TreeExplainer(student)
    return vectorizer, student, explainer


@st.cache_resource
def build_surrogate_model_cached():
    teacher = load_teacher_pipeline()

    base_emails = [
        "Beste klant, uw bankrekening wordt binnen 24 uur geblokkeerd. "
        "Klik onmiddellijk op de volgende link om uw gegevens te bevestigen.",

        "Hoi, hier is het verslag van onze meeting van gisteren. "
        "Laat me weten als je nog vragen hebt.",

        "GEFELICITEERD! U heeft een smartphone gewonnen. "
        "Bevestig NU uw adres via deze link.",

        "Beste collega, hierbij in bijlage de planning voor volgende week. "
        "Kun je even controleren of alles klopt?",

        "Uw account is tijdelijk geblokkeerd wegens een veiligheidsprobleem. "
        "Log in via onze speciale beveiligde pagina om uw identiteit te verifiëren.",
    ]

    return _build_surrogate_model_internal(teacher, base_emails)


# ============================================================
# 3. SHAP SAMENVATTING
# ============================================================

def get_shap_summary(x_sparse_row, vectorizer, explainer, top_k=5):
    x_dense = x_sparse_row.toarray()
    shap_vals = explainer.shap_values(x_dense)[0]
    feature_names = np.array(vectorizer.get_feature_names_out())

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

    return {"positive": positive, "negative": negative}


# ============================================================
# 4. PROMPT & UITLEGGENERATIE (GEMINI)
# ============================================================

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


def generate_explanation_for_audience(
    email_text,
    x_sparse_row,
    vectorizer,
    explainer,
    teacher_pipeline,
    audience,
    max_new_tokens=300,
):
    teacher_raw, _ = get_teacher_raw_and_score(teacher_pipeline, email_text)
    shap_summary = get_shap_summary(x_sparse_row, vectorizer, explainer, top_k=5)

    prompt = build_explanation_prompt(email_text, teacher_raw, shap_summary, audience)
    gen = generate_with_gemini(prompt, max_tokens=max_new_tokens)

    if "UITLEG:" in gen:
        gen = gen.split("UITLEG:")[-1].strip()

    return gen.strip(), shap_summary


# ============================================================
# 5. STREAMLIT UI (ONVERANDERD DESIGN)
# ============================================================

def main():
    st.set_page_config(page_title="FHMI Phishing XAI Demo (Gemini 2.5 Flash)", layout="wide")

    st.title("📧 FHMI Phishing XAI Demo (Gemini 2.5 Flash)")
    st.write(
        "Deze demo combineert een Hugging Face teacher-model, een surrogaatmodel met SHAP, "
        "en Gemini 2.5 Flash om verdachte e-mails uit te leggen voor verschillende doelgroepen."
    )

    with st.expander("🔧 Hoe werkt dit (kort)?", expanded=False):
        st.markdown(
            """
1. **Teacher-model (Hugging Face)**: bepaalt of een e-mail verdacht/spam is.  
2. **Surrogaatmodel (RandomForest + bag-of-words)**: leert het gedrag van de teacher na, maar is beter uitlegbaar.  
3. **SHAP**: toont welke woorden/uitdrukkingen de score omhoog of omlaag duwen.  
4. **Gemini LLM**: maakt op basis van e-mail + teacher-output + SHAP een menselijke uitleg,  
   afgestemd op **jongeren** en op een **ouder publiek**.
            """
        )

    st.markdown("---")

    email_text = st.text_area(
        "Plak hier een e-mail (bij voorkeur in het Nederlands):",
        height=200,
        placeholder="Beste klant, uw account wordt binnen 24 uur geblokkeerd. Klik op deze link...",
    )

    col_left, col_right = st.columns([1, 2])
    with col_left:
        analyze = st.button("Analyseer e-mail")

    if analyze:
        if not email_text.strip():
            st.warning("Vul eerst een e-mailtekst in.")
            return

        with st.spinner("Modellen laden en analyse uitvoeren..."):
            teacher = load_teacher_pipeline()
            vectorizer, student, shap_explainer = build_surrogate_model_cached()

            teacher_raw, teacher_score = get_teacher_raw_and_score(teacher, email_text)

            x_sparse = vectorizer.transform([email_text])
            x_dense = x_sparse.toarray()
            student_score = float(student.predict(x_dense)[0])

            # Jongeren
            audience_young = (
                "jongeren tussen 16 en 25 jaar, die veel online zijn, social media gebruiken "
                "en geen zin hebben in lange, saaie uitleg. Gebruik een directe, herkenbare toon "
                "en voorbeelden uit hun digitale leven."
            )
            explanation_young, shap_summary = generate_explanation_for_audience(
                email_text=email_text,
                x_sparse_row=x_sparse,
                vectorizer=vectorizer,
                explainer=shap_explainer,
                teacher_pipeline=teacher,
                audience=audience_young,
                max_new_tokens=260,
            )

            # Oudere doelgroep
            audience_older = (
                "volwassenen tussen 40 en 70 jaar, die regelmatig e-mails krijgen van bank, overheid "
                "en werk, maar zich niet dagelijks bezighouden met IT-beveiliging. Gebruik een rustige, "
                "duidelijke toon en leg stap voor stap uit wat er verdacht is."
            )
            explanation_older, _ = generate_explanation_for_audience(
                email_text=email_text,
                x_sparse_row=x_sparse,
                vectorizer=vectorizer,
                explainer=shap_explainer,
                teacher_pipeline=teacher,
                audience=audience_older,
                max_new_tokens=280,
            )

        st.markdown("## 🔍 Model-scores")
        score_cols = st.columns(2)
        with score_cols[0]:
            st.metric("Teacher (HF) phishing-score", f"{teacher_score:.3f}")
        with score_cols[1]:
            st.metric("Surrogaatmodel phishing-score", f"{student_score:.3f}")

        st.markdown("---")
        st.markdown("## 🧩 Belangrijkste SHAP-features (surrogaatmodel)")

        shap_cols = st.columns(2)
        pos = shap_summary.get("positive", [])
        neg = shap_summary.get("negative", [])

        with shap_cols[0]:
            st.markdown("**Positieve bijdragen (maken het meer verdacht):**")
            if pos:
                st.table(pos)
            else:
                st.write("_Geen duidelijke positieve features._")

        with shap_cols[1]:
            st.markdown("**Negatieve bijdragen (maken het minder verdacht):**")
            if neg:
                st.table(neg)
            else:
                st.write("_Geen duidelijke negatieve features._")

        st.markdown("---")
        st.markdown("## 🗣️ Uitleg voor verschillende doelgroepen")

        expl_cols = st.columns(2)
        with expl_cols[0]:
            st.markdown("### Voor jongeren (16–25)")
            st.write(explanation_young)

        with expl_cols[1]:
            st.markdown("### Voor ouder publiek (40–70)")
            st.write(explanation_older)


if __name__ == "__main__":
    main()
