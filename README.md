🛡️ Phishing XAI Experiment

Explainable AI Spam Filter — KU Leuven FHMI project

Dit project is een interactieve Streamlit webapp die deelnemers e-mails toont en test hoe goed ze phishing herkennen onder verschillende uitlegcondities.
De app combineert:

een HuggingFace spammodel (voorspelling)

een surrogaatmodel (TF-IDF + Logistic Regression)

SHAP explainability

een phishing cue taxonomy

experimenteerfunctionaliteit met logging

Het resultaat is een bruikbaar platform om RQ1 (confidence/actionability) en RQ3 (understanding) te meten in user studies.

📦 Installatie

Clone dit project:

git clone https://github.com/<jouw-repo>
cd phishing-xai-experiment


Maak en activeer een virtuele omgeving:

macOS/Linux:

python3 -m venv venv
source venv/bin/activate


Windows (PowerShell):

python -m venv venv
venv\Scripts\Activate.ps1


Installeer dependencies:

pip install streamlit shap scikit-learn transformers torch

🚀 Applicatie runnen

Voer dit uit in de projectmap:

streamlit run app.py


De app opent automatisch in je browser op:
👉 http://localhost:8501

🧪 Hoe het experiment werkt

Deelnemers doorlopen e-mails één voor één.
Voor elke e-mail zien ze:

AI-classificatie (HuggingFace)

SPAM / NOT SPAM

Probability score

Een van de drie uitlegcondities (via sidebar instelbaar):

Geen uitleg

Alleen highlights — SHAP highlights verdachte woorden

Highlights + tekstuitleg — categorieën (Urgency, Credential Harvesting, enz.) + leerteksten

De deelnemer geeft:

eigen oordeel (phishing / niet)

confidence (1–10)

Na “Opslaan & volgende” gaat de app naar de volgende stimulus.

📄 Logging van resultaten

Alle antwoorden worden automatisch opgeslagen in:

logs_experiment.csv


Elke rij bevat:

Kolom	Betekenis
participant_id	ID van de deelnemer
condition	uitlegconditie (no_xai, highlight_only, full_xai)
stimulus_id	ID van de e-mail
ground_truth	1 = phishing, 0 = legitiem
ai_is_spam	AI-voorspelling (1 of 0)
ai_score	AI-probability score
user_label	oordeel van deelnemer (1=phishing)
confidence_1_10	zelfverzekerdheid
reaction_time_sec	reactietijd voor die e-mail
timestamp	tijdstip van antwoord

Deze CSV is direct bruikbaar voor statistische analyse (ANOVA, mixed-effects, etc.).

🧠 Hoe de XAI werkt

De uitlegfunctie gebruikt een surrogaatmodel:

TF-IDF vectorizer (1-grams/2-grams)

Logistic Regression (snelle, interpreteerbare baseline)

SHAP LinearExplainer berekent welke woorden de spam-score verhogen

Deze SHAP-woorden worden gematcht aan een phishing cue taxonomy:

Urgency (“immediately”, “today”, “final notice”)

Credential Harvesting (“verify password”, “login”, “authenticate”)

Suspicious Links (“bit.ly”, “http”, “verify-...”)

Threats (“account suspended”)

Financial Bait, Impersonation, Generic phishing cues

Daarna wordt een natuurlijke taal uitleg gegenereerd zodat gebruikers leren wat phishing-signalen zijn.

Waarom surrogaatmodel?

HuggingFace-transformers zijn moeilijk robuust uit te leggen met SHAP (tokenizers, subtokens, maskers).
Daarom gebruiken we:

HF-model → voorspelling

Surrogaat (TF-IDF + SHAP) → uitlegbaarheid

Dit is een bekende, valide XAI-benadering.

📁 Bestandsstructuur
/
├── app.py                 # Streamlit experiment app
├── logs_experiment.csv    # (wordt automatisch aangemaakt)
├── README.md              # (dit bestand)
└── stimuli/ (optioneel)   # om later meer mails te laden

📘 Stimuli toevoegen

Bovenaan in app.py staat:

STIMULI = [
    { "id": "mail1", "text": "...", "ground_truth": 1 },
    ...
]


Je kunt eenvoudig meer mails toevoegen of genereren via CSV/JSON.

👥 Team gebruik

Iedereen in het team kan:

De code openen

Een eigen branche maken

Nieuwe e-mails toevoegen, SHAP-categories uitbreiden, logging aanpassen

De app lokaal runnen voor testen of user studies

❓ Vragen of uitbreidingen

In de code zit ruimte voor:

condition randomization

UI voor studiablokken

import van externe datasets

automatische generering van stimuli

integratie met Firebase/Sheets i.p.v. CSV

Laat gerust weten als je extra functionaliteit wil.# FHMI-phishing
