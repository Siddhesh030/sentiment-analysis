from pathlib import Path
import re
import pickle
import logging

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request limit
logging.basicConfig(level=logging.INFO)

# Keep model files in the same directory as this app.py file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model (3).pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

model = None
vectorizer = None
MODEL_LOAD_ERROR = None

try:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    app.logger.info("Sentiment model and vectorizer loaded successfully")
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)
    app.logger.exception("Could not load model or vectorizer")


def clean_text(text: str) -> str:
    """Apply the same lightweight cleaning used by the original app."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentiment_details(label: object) -> tuple[str, str, str]:
    """Map common model labels to a presentation label, icon, and description."""
    raw = str(label).strip()
    value = raw.lower()
    if "positive" in value or value in {"pos", "1"}:
        return "Positive", "😊", "The text is predicted to express a positive sentiment."
    if "negative" in value or value in {"neg", "0", "-1"}:
        return "Negative", "😞", "The text is predicted to express a negative sentiment."
    if "neutral" in value or value in {"neu", "2"}:
        return "Neutral", "😐", "The text is predicted to express a neutral sentiment."
    return raw.title() or "Unknown", "🔎", "Sentiment predicted by your trained model."


def predict_sentiment(text: str) -> dict:
    if model is None or vectorizer is None:
        raise RuntimeError(
            "The model files could not be loaded. Make sure model (3).pkl and vectorizer.pkl "
            "are included beside app.py and that scikit-learn matches the training version."
        )

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    label, icon, description = sentiment_details(prediction)

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "sentiment": label,
        "confidence": confidence,
        "icon": icon,
        "description": description,
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0b1020">
  <meta name="description" content="A clean, professional sentiment analysis workspace powered by machine learning.">
  <title>Sentiment Studio — AI Text Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root{color-scheme:dark;--bg:#0b1020;--panel:#111a2d;--panel2:#0e1729;--line:rgba(180,198,235,.13);--text:#f4f7ff;--muted:#9aa8c2;--accent:#8b7cff;--accent2:#58c7f3;--good:#43d6a0;--bad:#ff7185;--warn:#f6c96b;--shadow:0 24px 80px rgba(0,0,0,.24)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;min-height:100vh;background:radial-gradient(ellipse at 10% 0%,rgba(91,105,219,.18),transparent 35%),radial-gradient(ellipse at 95% 25%,rgba(53,176,213,.10),transparent 30%),var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;line-height:1.5}button,textarea{font:inherit}button{cursor:pointer}.shell{width:min(1120px,calc(100% - 40px));margin:0 auto}.topbar{display:flex;align-items:center;justify-content:space-between;padding:25px 0 20px;border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:12px;text-decoration:none;color:var(--text)}.brand-mark{width:42px;height:42px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,var(--accent),#4ebfe7);box-shadow:0 8px 24px #7669e733;font-size:21px;font-weight:800}.brand-name{font-family:Manrope,sans-serif;font-size:16px;font-weight:800;letter-spacing:-.4px}.brand-sub{display:block;color:var(--muted);font-size:11px;font-weight:500;letter-spacing:.5px}.status{display:flex;align-items:center;gap:8px;color:#b7c4dc;font-size:12px;padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#ffffff05}.status-dot{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 0 4px #43d6a015}.hero{padding:56px 0 34px;max-width:760px}.eyebrow{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid #8b7cff44;background:#8b7cff12;border-radius:999px;color:#c6c0ff;font-size:11px;font-weight:700;letter-spacing:1.1px;text-transform:uppercase}.hero h1{font-family:Manrope,sans-serif;font-size:clamp(38px,6vw,64px);line-height:1.08;letter-spacing:-2.6px;margin:20px 0 16px;font-weight:800}.gradient{background:linear-gradient(100deg,#a69cff 5%,#67d3f2 92%);-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{max-width:630px;color:var(--muted);font-size:16px;line-height:1.8;margin:0}.workspace{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(250px,.75fr);gap:18px;align-items:start}.panel{background:linear-gradient(145deg,rgba(19,29,50,.96),rgba(14,22,39,.96));border:1px solid var(--line);border-radius:22px;box-shadow:var(--shadow)}.composer{padding:25px}.section-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}.section-heading h2,.side-heading{font-family:Manrope,sans-serif;font-size:15px;letter-spacing:-.2px;margin:0;font-weight:800}.step{color:#a9b5cc;font-size:11px;font-weight:700;letter-spacing:.8px}.hint{color:var(--muted);font-size:12px;margin:0 0 13px}.input-wrap{position:relative}textarea{display:block;width:100%;min-height:205px;resize:vertical;border:1px solid #a8b8d51f;border-radius:15px;background:#080f1d9c;color:var(--text);padding:18px 18px 38px;outline:none;line-height:1.75;font-size:14px;transition:border .2s,box-shadow .2s}textarea::placeholder{color:#73819b}textarea:focus{border-color:#8b7cffaa;box-shadow:0 0 0 4px #8b7cff18}.char-count{position:absolute;bottom:12px;right:14px;color:#7e8da7;font-size:11px}.examples-label{color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.7px;text-transform:uppercase;margin:18px 0 9px}.examples{display:flex;flex-wrap:wrap;gap:8px}.chip{border:1px solid var(--line);background:#ffffff05;color:#c1cbe0;padding:8px 11px;border-radius:10px;font-size:12px;transition:.2s}.chip:hover{border-color:#8b7cff88;background:#8b7cff12;color:#fff}.actions{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:22px}.privacy{color:#7f8da6;font-size:11px}.primary{border:0;border-radius:12px;background:linear-gradient(110deg,#8174ff,#4abde5);color:white;font-weight:800;font-size:13px;padding:13px 18px;min-width:178px;box-shadow:0 10px 25px #7168df2c;transition:transform .2s,filter .2s}.primary:hover{transform:translateY(-2px);filter:brightness(1.08)}.primary:disabled{opacity:.7;cursor:wait;transform:none}.spinner{display:inline-block;width:13px;height:13px;border:2px solid #ffffff66;border-top-color:white;border-radius:50%;animation:spin .7s linear infinite;vertical-align:-2px;margin-right:7px}@keyframes spin{to{transform:rotate(360deg)}}.side{padding:22px}.side-head{display:flex;align-items:center;gap:10px;margin-bottom:17px}.side-icon{width:34px;height:34px;border:1px solid var(--line);border-radius:11px;background:#ffffff06;display:grid;place-items:center}.side-copy{color:var(--muted);font-size:12px;line-height:1.75;margin:0 0 20px}.step-list{display:grid;gap:16px}.step-item{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:start}.step-number{width:27px;height:27px;border:1px solid var(--line);border-radius:9px;display:grid;place-items:center;color:#bcb5ff;font-size:11px;font-weight:800}.step-item strong{display:block;font-size:12px;margin:1px 0 3px}.step-item span{display:block;color:var(--muted);font-size:11px;line-height:1.6}.divider{height:1px;background:var(--line);margin:20px 0}.result{display:none;margin-top:18px;padding:22px;animation:appear .35s ease}.result.show{display:block}@keyframes appear{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}.result-head{display:flex;align-items:center;gap:14px}.result-emoji{width:54px;height:54px;display:grid;place-items:center;background:#ffffff08;border:1px solid var(--line);border-radius:16px;font-size:27px}.result-kicker{color:var(--muted);font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase}.result-label{font-family:Manrope,sans-serif;font-size:26px;font-weight:800;letter-spacing:-.8px;margin-top:2px}.result-description{color:var(--muted);font-size:12px;margin-top:3px}.confidence{margin-top:22px}.confidence-line{display:flex;align-items:center;justify-content:space-between;color:#c1cbe0;font-size:12px;margin-bottom:9px}.confidence-line strong{font-size:13px;color:var(--text)}.track{height:8px;background:#ffffff0d;border-radius:99px;overflow:hidden}.bar{width:0;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .7s ease}.notice{display:none;margin-top:14px;padding:12px 14px;border-radius:12px;border:1px solid #ff718544;background:#ff71850d;color:#ffb3bd;font-size:12px}.notice.show{display:block}.features{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 48px}.feature{padding:19px;border:1px solid var(--line);border-radius:17px;background:#ffffff03}.feature-symbol{font-size:18px;margin-bottom:12px}.feature h3{font-family:Manrope,sans-serif;font-size:13px;margin:0 0 6px}.feature p{color:var(--muted);font-size:11px;line-height:1.7;margin:0}.footer{border-top:1px solid var(--line);padding:20px 0 28px;display:flex;justify-content:space-between;gap:12px;color:#75839c;font-size:11px}.footer strong{color:#aeb9cf;font-weight:600}.light-mode{color-scheme:light;--bg:#f4f7fc;--panel:#fff;--panel2:#fff;--line:rgba(36,53,84,.13);--text:#17223a;--muted:#63718a;--shadow:0 20px 55px rgba(36,53,84,.08);background:radial-gradient(ellipse at 10% 0%,#dcd9ff99,transparent 35%),radial-gradient(ellipse at 95% 25%,#d5f4ff99,transparent 30%),var(--bg)}.light-mode .panel{background:rgba(255,255,255,.9)}.light-mode textarea{background:#f8faff;color:var(--text);border-color:var(--line)}.light-mode .chip{color:#42506a}.light-mode .feature{background:#ffffff9c}.light-mode .status{color:#53627d;background:#ffffff99}.light-mode .side-icon,.light-mode .result-emoji{background:#f5f7ff}
    @media(max-width:820px){.workspace{grid-template-columns:1fr}.side{padding:20px}.step-list{grid-template-columns:repeat(3,1fr);gap:12px}.step-item{grid-template-columns:26px 1fr}.hero{padding-top:43px}}@media(max-width:560px){.shell{width:min(100% - 26px,1120px)}.topbar{padding-top:17px}.brand-name{font-size:14px}.status{font-size:10px;padding:7px 9px}.hero{padding:38px 0 25px}.hero h1{letter-spacing:-1.7px}.hero p{font-size:14px}.composer{padding:17px}.actions{align-items:stretch;flex-direction:column}.primary{width:100%}.privacy{text-align:center}.step-list{grid-template-columns:1fr}.features{grid-template-columns:1fr}.footer{flex-direction:column}.result{padding:17px}.result-label{font-size:23px}}
    @media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;transition-duration:.01ms!important}}
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="Sentiment Studio home"><span class="brand-mark">S</span><span class="brand-name">Sentiment Studio<span class="brand-sub">MACHINE LEARNING WORKSPACE</span></span></a>
      <button class="chip" id="themeToggle" type="button" aria-label="Toggle light and dark theme">☼ &nbsp; Appearance</button>
    </header>
    <main>
      <section class="hero">
        <div class="eyebrow"><span>✦</span> NLP · TEXT INTELLIGENCE</div>
        <h1>Read between<br>the <span class="gradient">sentiments.</span></h1>
        <p>Turn everyday feedback into a clearer signal. Enter a review, comment, or message and let your machine-learning model estimate the sentiment behind the words.</p>
      </section>
      <section class="workspace" aria-label="Sentiment analysis workspace">
        <div>
          <section class="panel composer">
            <div class="section-heading"><h2>Analyze your text</h2><span class="step">01 / INPUT</span></div>
            <p class="hint">Paste a message below. For the best results, use clear text in the language your model was trained on.</p>
            <div class="input-wrap"><textarea id="textInput" maxlength="5000" aria-label="Text to analyze" placeholder="For example: The quality is excellent and the delivery was faster than expected…"></textarea><span class="char-count" id="charCount">0 / 5000</span></div>
            <div class="examples-label">Try a sample</div>
            <div class="examples">
              <button class="chip" type="button" data-example="I absolutely love this product. The quality is excellent and it works perfectly!">😊 Positive example</button>
              <button class="chip" type="button" data-example="This was a disappointing experience. The product broke quickly and support was unhelpful.">☹ Negative example</button>
              <button class="chip" type="button" data-example="The package arrived today. The item matches the description.">◉ Everyday review</button>
            </div>
            <div class="actions"><span class="privacy">⌁ &nbsp; Your text is processed by this app's model.</span><button class="primary" id="analyzeBtn" type="button"><span id="buttonContent">Analyze sentiment&nbsp; ↗</span></button></div>
            <div class="notice" id="errorNotice" role="alert"></div>
          </section>
          <section class="panel result" id="result" aria-live="polite" aria-atomic="true">
            <div class="section-heading"><h2>Analysis result</h2><span class="step">02 / INSIGHT</span></div>
            <div class="result-head"><div class="result-emoji" id="resultIcon">✦</div><div><div class="result-kicker">Predicted sentiment</div><div class="result-label" id="sentiment">—</div><div class="result-description" id="description"></div></div></div>
            <div class="confidence" id="confidenceSection"><div class="confidence-line"><span>Model confidence estimate</span><strong id="confidence">—</strong></div><div class="track"><div class="bar" id="progressBar"></div></div></div>
          </section>
        </div>
        <aside class="panel side">
          <div class="side-head"><span class="side-icon">✧</span><h2 class="side-heading">From text to insight</h2></div>
          <p class="side-copy">A simple workflow built around your trained sentiment classifier. Results reflect the model's learned patterns, not a guarantee of how a person feels.</p>
          <div class="step-list">
            <div class="step-item"><span class="step-number">01</span><div><strong>Enter text</strong><span>Add a review, comment, or short message.</span></div></div>
            <div class="step-item"><span class="step-number">02</span><div><strong>Transform features</strong><span>TF-IDF converts text into features your model can process.</span></div></div>
            <div class="step-item"><span class="step-number">03</span><div><strong>Review prediction</strong><span>See the predicted label and confidence estimate when available.</span></div></div>
          </div>
          <div class="divider"></div>
          <div class="side-head"><span class="side-icon">◎</span><h2 class="side-heading">Designed for clarity</h2></div>
          <p class="side-copy" style="margin-bottom:0">A responsive interface, clear result states, and straightforward feedback make experimentation easy on desktop and mobile.</p>
        </aside>
      </section>
      <section class="features" aria-label="Features">
        <article class="feature"><div class="feature-symbol">↯</div><h3>Fast predictions</h3><p>Submit text to your local model inference endpoint and get a result without reloading the page.</p></article>
        <article class="feature"><div class="feature-symbol">⌘</div><h3>Machine-learning powered</h3><p>Uses your existing trained classifier and TF-IDF vectorizer rather than a placeholder prediction.</p></article>
        <article class="feature"><div class="feature-symbol">◔</div><h3>Confidence, when available</h3><p>Displays the model's maximum predicted probability when the estimator supports it.</p></article>
      </section>
    </main>
    <footer class="footer"><span><strong>Sentiment Studio</strong> · Text intelligence workspace</span><span>Machine learning · NLP · Built for the web</span></footer>
  </div>
<script>
(() => {
  const input = document.getElementById('textInput');
  const count = document.getElementById('charCount');
  const button = document.getElementById('analyzeBtn');
  const buttonContent = document.getElementById('buttonContent');
  const result = document.getElementById('result');
  const notice = document.getElementById('errorNotice');
  const themeToggle = document.getElementById('themeToggle');
  input.addEventListener('input', () => { count.textContent = `${input.value.length} / 5000`; });
  document.querySelectorAll('[data-example]').forEach(chip => chip.addEventListener('click', () => {
    input.value = chip.dataset.example; input.dispatchEvent(new Event('input')); input.focus();
  }));
  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    try { localStorage.setItem('sentiment-studio-theme', document.body.classList.contains('light-mode') ? 'light' : 'dark'); } catch (_) {}
  });
  try { if (localStorage.getItem('sentiment-studio-theme') === 'light') document.body.classList.add('light-mode'); } catch (_) {}

  button.addEventListener('click', async () => {
    const text = input.value.trim();
    notice.classList.remove('show'); notice.textContent = '';
    if (!text) { notice.textContent = 'Please enter some text before running the analysis.'; notice.classList.add('show'); input.focus(); return; }
    button.disabled = true; buttonContent.innerHTML = '<span class="spinner"></span>Analyzing…';
    try {
      const response = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json', 'Accept': 'application/json'}, body: JSON.stringify({text}) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || 'The analysis could not be completed. Please try again.');
      document.getElementById('resultIcon').textContent = data.icon || '✦';
      document.getElementById('sentiment').textContent = data.sentiment || 'Unknown';
      document.getElementById('description').textContent = data.description || '';
      const confidence = data.confidence;
      const confidenceSection = document.getElementById('confidenceSection');
      if (typeof confidence === 'number' && Number.isFinite(confidence)) {
        confidenceSection.style.display = '';
        document.getElementById('confidence').textContent = `${confidence.toFixed(2)}%`;
        document.getElementById('progressBar').style.width = `${Math.max(0, Math.min(100, confidence))}%`;
      } else {
        confidenceSection.style.display = 'none';
      }
      const lower = String(data.sentiment || '').toLowerCase();
      const emoji = document.getElementById('resultIcon');
      emoji.style.borderColor = lower.includes('positive') ? '#43d6a055' : lower.includes('negative') ? '#ff718555' : '#f6c96b55';
      result.classList.add('show');
      result.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest'});
    } catch (error) {
      notice.textContent = error.message || 'Something went wrong while analyzing your text.';
      notice.classList.add('show');
    } finally {
      button.disabled = false; buttonContent.textContent = 'Analyze sentiment ↗';
    }
  });
})();
</script>
</body>
</html>'''


@app.get("/")
def home():
    return render_template_string(HTML)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text must be 5,000 characters or fewer."}), 413
    try:
        return jsonify(predict_sentiment(text))
    except RuntimeError as exc:
        app.logger.error("Prediction unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The model could not analyze this text. Please try again."}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large. Please submit a shorter text."}), 413


# Vercel imports `app` as the WSGI application. Do not call app.run() in deployment.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
