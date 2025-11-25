📧 FHMI Phishing XAI Demo — README

Deze app demonstreert hoe je een phishing-detectiemodel kunt uitleggen met behulp van:

Hugging Face teacher-model (BERT-tiny spam classifier)

Surrogaatmodel + SHAP voor uitlegbaarheid

Gemini 2.5 Flash om begrijpelijke uitlegteksten te genereren

Streamlit als UI

De applicatie bevindt zich in app2.py.

📦 1. Installatie

Ga naar de projectmap en maak (optioneel) een virtual environment:

cd FHMI-phishing
python3 -m venv venv
source venv/bin/activate


Installeer alle vereiste Python-packages:

python3 -m pip install --upgrade pip
python3 -m pip install streamlit google-genai transformers torch scikit-learn shap tqdm sentencepiece

🔑 2. Gemini API-sleutel instellen

De app gebruikt Gemini 2.5 Flash om uitlegteksten te genereren.
Je hebt hiervoor een API-key nodig van:

👉 https://ai.google.dev

Sla deze key op als environment variable:

export GEMINI_API_KEY="jouw_api_key_hier"


Tip: wil je dit permanent maken?
Voeg de regel toe aan ~/.zshrc of ~/.bashrc.

De app leest deze sleutel automatisch in via:

os.getenv("GEMINI_API_KEY")

▶️ 3. De applicatie starten

Start de Streamlit-app:

streamlit run app2.py


De UI opent automatisch in je browser op:

http://localhost:8501

🧠 4. Hoe de app werkt

Je plakt een e-mail in het tekstveld.

Het HuggingFace-model geeft een phishing/spam score.

Een surrogaatmodel (RandomForest) wordt gebruikt om SHAP-waarden te berekenen.

SHAP bepaalt welke woorden de score positief of negatief beïnvloeden.

De app stuurt geen ruwe e-mail naar Gemini (veiligheidsrestrictie), maar een modelanalyse.

Gemini genereert twee uitlegggen:

één op maat van jongeren

één op maat van ouder publiek

🤝 5. Support / fouten oplossen
❌ “GEMINI_API_KEY is niet gezet”

Je vergeten de key te exporteren.
Doe:

export GEMINI_API_KEY="xxx"

❌ “Gemini kon geen uitleg geven (lege response)”

Gemini’s veiligheidsfilter blokkeerde de input.
In deze versie wordt dit automatisch opgevangen.

❌ ModuleNotFoundError

Controleer of alles is geïnstalleerd:

python3 -m pip install streamlit google-genai transformers torch scikit-learn shap tqdm sentencepiece