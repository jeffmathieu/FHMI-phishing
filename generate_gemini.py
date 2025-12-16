import json
import os
import time
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
from transformers import pipeline
from google import genai
from typing import List, Dict

# ============================================================
# 1. SETUP & CONFIG
# ============================================================

# Paste your API key here or make sure it's in your env variables
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

GEMINI_MODEL_NAME = "gemini-2.5-flash"
# Reuse one client instance (avoid reconnect overhead)
GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
OUTPUT_FILE = "gemini_cache.json"
OUTPUT_TEXT_FILE = "gemini_explanations.txt"
REQUEST_SPACING_SECONDS = 15  # free-tier friendly pacing (1 request per email)


# Copy your STIMULI from app3.py
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
Online URL for this job: http://jobs.perl.org/job/7898

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
    You can view your receipt and subscription details on your dashboard: [https://dashboard.saas-tool.com/billing]
    """,
        "ground_truth": 0,
    },
    {
        "id": "mail9",
        "text": """Sender: Marketing Team <marketing@brand-store.com> 
Subject: Your Order #12345
Dear Casey, thank you for your purchase. Your order will be shipped soon.
You can track your package via FedEx using the link below: [https://www.fedex.com/track/123456789]
Thank you for shopping with us!
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

# Audiences (Must match what is in app3.py)
AUDIENCE_YOUNG = (
    "jongeren tussen 16 en 25 jaar, die veel online zijn, social media gebruiken,"
    "en geen zin hebben in lange, saaie uitleg. Gebruik een directe, herkenbare toon "
    "en voorbeelden uit hun digitale leven."
)
AUDIENCE_OLDER = (
    "volwassenen tussen 40 en 70 jaar, die regelmatig e-mails krijgen van bank, overheid "
    "en werk, maar zich niet dagelijks bezighouden met IT-beveiliging. Gebruik een rustige, "
    "duidelijke toon en leg stap voor stap uit wat er verdacht is."
)

AUDIENCE_NORMAL = (
    "een gemiddelde gebruiker die een duidelijke, korte uitleg wil. "
    "Gebruik een neutrale toon, benoem de belangrijkste rode vlaggen en geef concrete adviezen."
)

# ============================================================
# 2. HELPER MODELS (Surrogate + Teacher)
# ============================================================
# We need to rebuild the models to generate the exact same SHAP values

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

print("Training surrogate model...")
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=3000, stop_words="english")
Xv = vec.fit_transform(SPAM + HAM)
clf = LogisticRegression(max_iter=2000)
clf.fit(Xv, [1] * len(SPAM) + [0] * len(HAM))
FEATURE_NAMES = np.array(vec.get_feature_names_out())

# Build SHAP explainer
print("Building SHAP explainer...")
background = Xv[np.random.choice(Xv.shape[0], min(10, Xv.shape[0]), replace=False)]
explainer = shap.LinearExplainer(clf, background)

# Load Teacher Pipeline (HuggingFace)
print("Loading Teacher model...")
teacher_pipeline = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection", top_k=None)


# ============================================================
# 3. GENERATION FUNCTIONS
# ============================================================

def get_shap_summary(text):
    X_new = vec.transform([text])
    # Get raw SHAP values
    values = explainer.shap_values(X_new)
    if isinstance(values, list):
        # if list, usually [class0_shap, class1_shap]. We want class1 (spam)
        shap_vals = values[1][0] if len(values) > 1 else values[0][0]
    elif isinstance(values, np.ndarray):
        shap_vals = values[0]
    else:
        shap_vals = values

    # Summarize top 5 positive and negative
    x_dense = X_new.toarray()
    idx_sorted_pos = np.argsort(-shap_vals)
    idx_sorted_neg = np.argsort(shap_vals)

    positive = []
    for i in idx_sorted_pos:
        if shap_vals[i] <= 0: continue
        if x_dense[0, i] == 0: continue
        positive.append({"token": FEATURE_NAMES[i], "shap": float(shap_vals[i])})
        if len(positive) >= 5: break

    negative = []
    for i in idx_sorted_neg:
        if shap_vals[i] >= 0: continue
        if x_dense[0, i] == 0: continue
        negative.append({"token": FEATURE_NAMES[i], "shap": float(shap_vals[i])})
        if len(negative) >= 5: break

    return {"positive": positive, "negative": negative}


def get_teacher_raw(text):
    out = teacher_pipeline(text)
    # Normalize output format
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        return out[0]
    return out if isinstance(out, list) else [out]


def _extract_retry_seconds(err_msg: str) -> int | None:
    """Extract retry seconds from Gemini 429 errors when provided."""
    m = re.search(r"retryDelay'\s*:\s*'(\d+)s'", err_msg)
    if m:
        return int(m.group(1))
    m = re.search(r"Please retry in\s+(\d+(?:\.\d+)?)s", err_msg)
    if m:
        return int(float(m.group(1)))
    return None


def generate_gemini_text(prompt: str, max_retries: int = 8) -> str:
    """Call Gemini with robust retries for 503 overloads and 429 quota responses."""
    base_sleep = 2.0

    for attempt in range(max_retries):
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
            txt = getattr(response, "text", None)
            if txt and txt.strip():
                return txt.strip()
            raise RuntimeError("Gemini returned empty response")

        except Exception as e:
            msg = str(e)

            # 429: quota/rate limits -> respect server hint if present
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                retry_s = _extract_retry_seconds(msg) or 60
                print(f"[Gemini] 429 quota hit. Sleeping {retry_s}s then retrying...")
                time.sleep(retry_s + random.uniform(0, 1.5))
                continue

            # 503: overloaded/unavailable -> exponential backoff + jitter
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                sleep_s = min(60.0, base_sleep * (2 ** attempt)) + random.uniform(0, 1.5)
                print(f"[Gemini] 503 overloaded. Sleeping {sleep_s:.1f}s then retrying...")
                time.sleep(sleep_s)
                continue

            # Other transient errors -> light backoff
            sleep_s = min(10.0, base_sleep + attempt) + random.uniform(0, 1.0)
            print(f"[Gemini] Error: {e}. Sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)

    return "Error generating explanation."


def build_prompt(email_text, teacher_raw, shap_summary, audience):
    shap_json = json.dumps(shap_summary, indent=2)
    teacher_str = json.dumps(teacher_raw, indent=2)
    return f"""
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

E-MAIL:
{email_text}

MODEL-UITVOER:
{teacher_str}

BELANGRIJKSTE WOORDEN VOLGENS SHAP:
{shap_json}

UITLEG:
"""


# ============================================================
# 4. MAIN LOOP
# ============================================================

# Load existing cache (resume-friendly)
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
else:
    cache_data = {}

print(f"Starting generation for {len(STIMULI)} emails...")

for stim in STIMULI:
    e_id = stim["id"]
    text = stim["text"]
    print(f"Processing {e_id}...")

    # Skip if already cached
    if e_id in cache_data and isinstance(cache_data.get(e_id), dict) and cache_data[e_id].get("explanation"):
        print(f" - cached, skipping Gemini call.")
        continue

    # Compute features locally
    shap_data = get_shap_summary(text)
    teacher_data = get_teacher_raw(text)

    # One normal explanation per email (combined: A+B+C+D)
    prompt = build_prompt(text, teacher_data, shap_data, AUDIENCE_NORMAL)
    resp = generate_gemini_text(prompt)

    # Keep only the explanation part if the model echoes the template label
    if "UITLEG:" in resp:
        resp = resp.split("UITLEG:")[-1].strip()

    # Save to cache (incremental, so you don't lose progress on rate limits)
    cache_data[e_id] = {"explanation": resp}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    # Pace requests for free-tier limits (1 request per email)
    time.sleep(REQUEST_SPACING_SECONDS)

# Write a separate, human-readable output file (always rebuilt from cache)
with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
    for stim in STIMULI:
        e_id = stim["id"]
        if e_id not in cache_data:
            continue
        f.write(f"===== {e_id} =====")
        f.write(cache_data[e_id].get("explanation", "").strip() + "")

print(f"Done! Cache saved to {OUTPUT_FILE} and explanations saved to {OUTPUT_TEXT_FILE}")
