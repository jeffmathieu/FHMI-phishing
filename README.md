These instructions are based on Windows PowerShell. For other operating systems, please refer to online sources.

📦 1. Installation

Navigate to the project directory.

Install all required Python packages:

```
python -m pip install --upgrade pip
python -m pip install streamlit google-genai transformers torch scikit-learn shap tqdm sentencepiece
```

🔑 2. Setup Gemini API Key

The app uses Gemini 2.5 Flash to generate explanation texts. You need an API key for this from:

👉 https://ai.google.dev

Save this key as (key inside double quotes):

```$env:GEMINI_API_KEY = "your_key"```

The app automatically reads this key via:

os.getenv("GEMINI_API_KEY")

▶️ 3. Start the application

Start the Streamlit app:

```streamlit run phishing_classification_group2.py```


The UI opens automatically in your browser at:

http://localhost:8501


