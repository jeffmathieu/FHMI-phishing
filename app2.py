import json
import numpy as np
import streamlit as st
from tqdm import tqdm
import shap

from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 0. UITLEG-LLM (LOKAAL HUGGING FACE INSTRUCT-MODEL)
# ============================================================

@st.cache_resource
def load_explainer_pipeline():
    """
    Laadt een klein, licht instructmodel dat lokaal kan draaien.
    Geen zware downloads, geen sentencepiece, geen GPU nodig.
    """
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    explainer = pipeline(
        task="text-generation",
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=250,
    )
    return explainer



# ============================================================
# 1. HUGGING FACE "TEACHER" MODEL (SPAM/PHISHING CLASSIFIER)
# ============================================================

@st.cache_resource
def load_teacher_pipeline():
    """
    Hugging Face text-classification pipeline als 'teacher'
    voor spam/phishing-detectie.
    """
    teacher = pipeline(
        task="text-classification",
        model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        tokenizer="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        top_k=None
    )
    return teacher


def get_teacher_raw_and_score(teacher, text: str):
    """
    Laat het teacher-model een voorspelling maken en geeft:
      - teacher_raw: ruwe output (list[dict])
      - phishing_score: kans dat het spam/phishing is (0-1)
    """
    teacher_raw = teacher(text)

    if isinstance(teacher_raw, list):
        if len(teacher_raw) > 0 and isinstance(teacher_raw[0], list):
            logits = teacher_raw[0]
        else:
            logits = teacher_raw
    else:
        logits = [teacher_raw]

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

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(emails)

    student = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    student.fit(X, y_teacher)

    explainer = shap.TreeExplainer(student)
    return vectorizer, student, explainer


@st.cache_resource
def build_surrogate_model_cached():
    """
    Mini-dataset om surrogaatmodel te trainen (voor demo).
    """
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
        "Log in via onze speciale beveiligde pagina om uw identiteit te verifiëren."
    ]

    vectorizer, student, explainer = _build_surrogate_model_internal(teacher, base_emails)
    return vectorizer, student, explainer


# ============================================================
# 3. SHAP SAMENVATTING
# ============================================================

def get_shap_summary(x_sparse_row, vectorizer: CountVectorizer, explainer: shap.TreeExplainer, top_k: int = 5):
    """
    Samenvatting van SHAP:
    - top positieve features (verhogen phishing-score)
    - top negatieve features (verlagen phishing-score)
    Alleen features die echt in de e-mail voorkomen.
    """
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
        positive.append({
            "token": feature_names[i],
            "shap": float(shap_vals[i]),
        })
        if len(positive) >= top_k:
            break

    negative = []
    for i in idx_sorted_neg:
        if shap_vals[i] >= 0:
            continue
        if x_dense[0, i] == 0:
            continue
        negative.append({
            "token": feature_names[i],
            "shap": float(shap_vals[i]),
        })
        if len(negative) >= top_k:
            break

    return {
        "positive": positive,
        "negative": negative,
    }


# ============================================================
# 4. PROMPT & UITLEGGENERATIE (MET HF INSTRUCT-MODEL)
# ============================================================

def build_explanation_prompt(email_text: str, teacher_raw, shap_summary: dict, audience: str) -> str:
    """
    Prompt voor de uitleg-LLM, expliciet gefocust op 'verdachte e-mails'
    maar geformuleerd als defensieve, educatieve uitleg.
    """
    shap_json = json.dumps(shap_summary, indent=2, ensure_ascii=False)
    teacher_str = json.dumps(teacher_raw, indent=2, ensure_ascii=False)

    prompt = f"""
Je bent een expert in digitale veiligheid en je helpt gewone gebruikers om verdachte e-mails te herkennen,
zodat ze zich beter kunnen beschermen. Je analyseert ALLEEN de tekst op een educatieve manier.
Je geeft GEEN instructies om zelf misleiding of fraude te plegen.

Doelgroep: {audience}

Je taak:
- Beschrijf of de onderstaande e-mail mogelijk onbetrouwbaar of misleidend is.
- Gebruik daarvoor:
  * De inhoud van de e-mail
  * De model-output van een verdachte-berichten-classifier
  * De SHAP-samenvatting met woorden die meer of minder verdacht zijn

Schrijf twee à drie duidelijke alinea's in het NEDERLANDS met:
- Een korte conclusie: lijkt dit bericht eerder betrouwbaar of eerder verdacht?
- Een uitleg in eenvoudige taal die past bij deze doelgroep.
- Concrete verwijzingen naar delen van de e-mail (bijv. dreigende taal, tijdsdruk, links, vraag om gegevens).
- Praktische tips voor deze doelgroep (bv. "klik niet zomaar op links", "check de afzender", ...).
- Geen technische termen uitleggen als dat niet nodig is; houd het begrijpelijk.

Gebruik GEEN bullet lists, geen code en herhaal de tekst van de e-mail niet letterlijk;
geef alleen een leesbare uitleg.

==============================
TE ANALYSEREN BERICHT:
------------------------------
{email_text}

==============================
MODEL-OUTPUT (VERDACHTE-BERICHTEN CLASSIFIER):
------------------------------
{teacher_str}

==============================
SHAP-SAMENVATTING (VERDACHTE / NIET-VERDACHTE WOORDEN):
------------------------------
{shap_json}

==============================
UITLEG VOOR DEZE DOELGROEP:
"""
    return prompt.strip()


def generate_explanation_for_audience(
    email_text: str,
    x_sparse_row,
    vectorizer: CountVectorizer,
    explainer: shap.TreeExplainer,
    teacher_pipeline,
    explainer_pipeline,
    audience: str,
    max_new_tokens: int = 250
):
    """
    Uitleg genereren voor een bepaalde doelgroep (jongeren / ouder publiek),
    met een lokaal Hugging Face instruct-model.
    """
    teacher_raw, _ = get_teacher_raw_and_score(teacher_pipeline, email_text)
    shap_summary = get_shap_summary(x_sparse_row, vectorizer, explainer, top_k=5)

    prompt = build_explanation_prompt(
        email_text=email_text,
        teacher_raw=teacher_raw,
        shap_summary=shap_summary,
        audience=audience
    )

    result = explainer_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.4,
    )

    gen = result[0]["generated_text"]

    # Prompt eruit knippen als het model hem terug-echoot
    if "UITLEG VOOR DEZE DOELGROEP:" in gen:
        gen = gen.split("UITLEG VOOR DEZE DOELGROEP:")[-1].strip()

    return gen.strip(), shap_summary


# ============================================================
# 5. STREAMLIT UI (jouw “oude” layout)
# ============================================================

def main():
    st.set_page_config(page_title="FHMI Phishing XAI Demo (lokaal HF)", layout="wide")

    st.title("📧 FHMI Phishing XAI Demo (lokaal Hugging Face uitleg-LLM)")
    st.write(
        "Deze demo combineert een Hugging Face model, een surrogaatmodel met SHAP, "
        "en een lokaal instruct-model om verdachte e-mails uit te leggen voor verschillende doelgroepen."
    )

    with st.expander("🔧 Hoe werkt dit (kort)?", expanded=False):
        st.markdown(
            """
1. **Teacher-model (Hugging Face)**: bepaalt of een e-mail verdacht/spam is.  
2. **Surrogaatmodel (RandomForest + bag-of-words)**: leert het gedrag van de teacher na, maar is beter uitlegbaar.  
3. **SHAP**: toont welke woorden/uitdrukkingen de score omhoog of omlaag duwen.  
4. **Lokaal uitleg-LLM**: maakt op basis van e-mail + teacher-output + SHAP een menselijke uitleg,  
   afgestemd op **jongeren** en op een **ouder publiek**.
            """
        )

    st.markdown("---")

    email_text = st.text_area(
        "Plak hier een e-mail (bij voorkeur in het Nederlands):",
        height=200,
        placeholder="Beste klant, uw account wordt binnen 24 uur geblokkeerd. Klik op deze link..."
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
            explainer_llm = load_explainer_pipeline()
            vectorizer, student, shap_explainer = build_surrogate_model_cached()

            # Teacher score
            teacher_raw, teacher_score = get_teacher_raw_and_score(teacher, email_text)

            # Surrogaatmodel score
            x_sparse = vectorizer.transform([email_text])
            x_dense = x_sparse.toarray()
            student_score = float(student.predict(x_dense)[0])

            # Uitleg voor jongeren
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
                explainer_pipeline=explainer_llm,
                audience=audience_young,
                max_new_tokens=220
            )

            # Uitleg voor ouder publiek
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
                explainer_pipeline=explainer_llm,
                audience=audience_older,
                max_new_tokens=260
            )

        # Resultaten tonen
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
