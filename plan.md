# ClaimPilot — AI Co-Pilot for Insurance Claim Processing
### Full Implementation Plan (Hackathon Sprint, ~4-5 hours, solo, Google Colab + Streamlit + Gemini)

---

## 1. Concept

An insurance adjuster uploads: **(1)** a photo of the damaged vehicle, **(2)** a claim form (structured fields), **(3)** a free-text claim description. ClaimPilot:

1. Classifies damage severity from the photo (CNN) and highlights *where* it looked (Grad-CAM).
2. Scores fraud/approval risk from the claim form (ANN) and shows which features drove the score.
3. Uses Gemini to read the free-text description, summarize it, and flag if it's **inconsistent** with the photo/form (e.g., text says "minor scratch" but CNN says "severe").
4. Shows all of this to a human adjuster who **Approves / Rejects / Modifies** the recommendation.
5. Generates a downloadable **PDF report** of the whole decision (evidence + AI output + human verdict).

This single flow hits all 8 mandatory requirements and every rubric line — see §10.

---

## 2. Why Insurance Claim Processing (vs other handbook challenges)

- Image + tabular + text modalities arise **naturally** from one real claim — no forced/artificial 3rd modality.
- CNN task (damage severity) and ANN task (fraud/risk scoring) are both standard, well-documented Kaggle problems with labeled data — fast to train well in a few hours.
- The business pitch (insurers/TPAs pay per-claim to cut manual review time) is concrete and easy to defend live.
- Fits "Innovation" rubric line cleanly via the **cross-modal consistency check** (text vs image vs form) — most teams doing this challenge won't think to do this.

---

## 3. Architecture

```
                        ┌─────────────────────────┐
   Photo ───────────────▶   CNN (MobileNetV2 TL)   │──▶ severity + confidence + Grad-CAM
                        └─────────────────────────┘
                                                            │
   Claim form (tabular) ─▶ ANN (Keras dense net) ─▶ risk score + feature importance
                                                            │
   Claim description ─────▶ Gemini API ───────────▶ summary + consistency check + narrative
   (text)                                                    │
                                                            ▼
                                        ┌───────────────────────────┐
                                        │   Streamlit Web App        │
                                        │  (single page: upload →    │
                                        │   AI findings → HITL panel)│
                                        └───────────────────────────┘
                                                            │
                                     Adjuster: Approve / Reject / Modify
                                                            │
                                                            ▼
                                        PDF Report (fpdf2) — downloadable
```

All three model calls happen inside the same Streamlit process — no separate backend server needed. This keeps "System Architecture" simple and demoable within the time budget while still being a real multi-component pipeline.

---

## 4. Tech Stack

| Piece | Choice | Why |
|---|---|---|
| Model training | Google Colab (free GPU) | You already have access; fast enough for transfer learning in ~30-40 min |
| CNN | MobileNetV2 (ImageNet-pretrained) + fine-tuned head | Small, trains fast on CPU/GPU, good accuracy with few images |
| ANN | Keras Sequential dense network | Simple, fast to train on tabular data (seconds) |
| Explainability (image) | Grad-CAM | Standard, works directly on CNN's last conv layer |
| Explainability (tabular) | Permutation feature importance (SHAP as stretch goal) | Faster to compute than SHAP under time pressure, still legitimate XAI |
| LLM | Google Gemini API (free tier) | Generous free quota, good at structured summarization |
| Web app | Streamlit | Single Python file, no separate frontend/backend, minimal boilerplate — best fit for no prior coding experience |
| PDF report | fpdf2 | Lightweight, no system dependencies, easy to control layout |
| Hosting | Streamlit Community Cloud | Free, public URL, deploys straight from GitHub |
| Version control | GitHub (public or private repo) | Needed for Streamlit Cloud deployment |

---

## 5. Folder Structure

```
claimpilot/
├── notebook/
│   └── train_models.ipynb        # Colab notebook: data download + CNN + ANN training
├── models/
│   ├── damage_cnn.h5              # trained CNN (or .keras)
│   ├── risk_ann.h5                # trained ANN
│   └── risk_scaler.pkl            # feature scaler for ANN inputs
├── app.py                         # Streamlit web app (main entry point)
├── report_generator.py            # PDF report generation module
├── gemini_helper.py                # Gemini API wrapper (summary + consistency check)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml                # GEMINI_API_KEY (not committed — set in Streamlit Cloud dashboard)
└── sample_data/
    └── demo_claim.jpg              # a couple of sample images for live demo safety net
```

---

## 6. Step-by-Step Setup (before any code runs)

### 6.1 Kaggle API token (~2 min)
1. Go to kaggle.com → log in (or create free account).
2. Click your profile picture → **Settings** → scroll to **API** → **Create New Token**.
3. This downloads `kaggle.json`. Keep it — you'll upload it into the Colab notebook in step 1 of the notebook (I'll give you the exact cell for this).

### 6.2 Gemini API key (~2 min)
1. Go to Google AI Studio (aistudio.google.com) → **Get API key** → create a new key in a new/existing Google Cloud project.
2. Copy the key somewhere safe. You'll paste it into Colab as a variable and later into Streamlit Cloud's "Secrets" panel.

### 6.3 GitHub repo (~3 min)
1. Create a new **public** repo (Streamlit Community Cloud free tier deploys public repos most easily) named `claimpilot`.
2. We'll push the app files here after local testing.

### 6.4 Google Colab
1. Open a new Colab notebook, set Runtime → Change runtime type → **GPU (T4)**.
2. This is where all training happens (§8, §9).

---

## 7. Data Plan

### 7.1 Image data — car damage severity
- **Primary target:** Kaggle "Car Damage Detection / Severity" dataset (search terms: `car damage severity`, e.g. dataset by `anujms`), which typically provides folders like `01-minor`, `02-moderate`, `03-severe` (or `damage` vs `whole`).
- **Fallback:** if that exact dataset is unavailable/renamed, search `https://www.kaggle.com/search?q=car+damage` (per handbook) and pick any dataset with 2-3 severity/damage classes and at least ~50-100 images per class — that's enough for transfer learning in this time budget.
- Target task: **3-class classification** — `minor`, `moderate`, `severe` (or 2-class `damaged`/`not damaged` if a 3-class dataset isn't found in time — still valid, just simpler).

### 7.2 Tabular data — claim risk/fraud
- **Primary target:** Kaggle vehicle insurance claims fraud dataset (search terms: `insurance claims fraud`, e.g. dataset by `mastmustu` or `shivamb` — look for one with a binary `fraud_reported` or similar label and columns like claim amount, policy tenure, vehicle age, prior claims, incident severity).
- Target task: **binary classification** — fraud risk / needs-review flag, output as a probability (0-100% risk score).

### 7.3 Text data — claim description
- No dataset needed. At demo time, the adjuster (you) types/pastes a free-text claim description into a text box. Gemini processes this live — this is a genuine real-time modality, not a pre-trained one, which is explicitly allowed ("LLMs may be used for reasoning, summarization, explanations... not as the primary predictive engine").

---

## 8. Model 1 — CNN for Damage Severity

**Approach:** Transfer learning, not training from scratch (this is what makes it feasible in ~30-40 min).

- Base: `MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))`, base layers frozen.
- Head: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(num_classes, softmax)`.
- Data augmentation: horizontal flip, small rotation/zoom (Keras `ImageDataGenerator` or `tf.keras.layers` augmentation) — helps since damage datasets are usually small.
- Training: ~10-15 epochs frozen base, optionally 5 more epochs with the last ~20 layers unfrozen at low learning rate if time allows and accuracy is weak.
- Output per prediction: predicted class + softmax confidence for each class.
- **Grad-CAM:** computed on the last conv layer of MobileNetV2, overlaid as a heatmap on the uploaded image — this is your image-side Explainable AI requirement.
- Export: save as `damage_cnn.h5`.

---

## 9. Model 2 — ANN for Claim Risk Scoring

**Approach:** Small dense network on structured/tabular features — trains in seconds, so most time here goes to cleaning columns, not training.

- Preprocessing: select ~8-12 relevant columns (claim amount, vehicle age, policy tenure, prior claims count, incident severity category, etc.), one-hot encode categoricals, scale numerics with `StandardScaler` (save as `risk_scaler.pkl`).
- Architecture: `Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid)`.
- Loss: binary cross-entropy; metric: accuracy + AUC.
- Output: risk probability (0-1), shown as a percentage "fraud/needs-review risk."
- **Explainability:** permutation feature importance computed once after training (shuffle each feature, measure drop in validation AUC) → bar chart of top contributing features shown in the app. (Stretch: swap in `shap.Explainer` if there's spare time — same output slot in the UI either way.)
- Export: save as `risk_ann.h5` + `risk_scaler.pkl`.

---

## 10. LLM Layer — Gemini (reasoning only, not predictive)

`gemini_helper.py` will expose one function, e.g. `analyze_claim_text(description, cnn_result, ann_result)`, which sends a single prompt to Gemini asking it to:
1. Summarize the claim description in 2-3 sentences.
2. Compare the description against the CNN severity result and ANN risk score, and explicitly flag any inconsistency (e.g., text downplays damage the photo shows as severe).
3. Draft a short plain-English recommendation paragraph for the adjuster (not a decision — just an explanation), which also gets embedded in the final PDF report.

This satisfies "LLMs may be used for reasoning, summarization, explanations, or report generation" precisely, while the CNN/ANN remain the actual predictive engines.

---

## 11. Web App (Streamlit) — `app.py`

Single-page flow, top to bottom:

1. **Header** — "ClaimPilot: AI Co-Pilot for Insurance Claims"
2. **Input section:**
   - File uploader for the damage photo.
   - A small form for tabular fields (claim amount, vehicle age, prior claims, incident severity dropdown, etc. — matching whatever columns the ANN was trained on).
   - Text area for the free-text claim description.
   - "Analyze Claim" button.
3. **AI Findings section (after clicking Analyze):**
   - Left: uploaded photo + Grad-CAM overlay + predicted severity + confidence bar.
   - Right: risk score gauge/percentage + feature importance bar chart.
   - Below: Gemini's summary + consistency-check flag + recommendation paragraph.
4. **Human-in-the-Loop panel:**
   - Radio buttons / buttons: **Approve**, **Reject**, **Modify** (with a comment box if Modify/Reject chosen).
   - "Confirm Decision" button — stores the decision + comment + timestamp in session state (and appended to a local CSV log, showing a persistent audit trail).
5. **Report section:**
   - "Generate Report" button → calls `report_generator.py` → shows a "Download PDF" button.

This structure directly mirrors "Business Problem → Multimodal Integration → Deep Learning → XAI → HITL → UX" in the order a judge would want to see it.

---

## 12. PDF Report — `report_generator.py`

Using `fpdf2`, one function `build_report(...)` that lays out:
- Claim ID/timestamp, uploaded photo thumbnail.
- CNN result: severity class + confidence.
- ANN result: risk score + top 3 contributing features.
- Gemini summary + consistency flag + recommendation text.
- Human decision (Approve/Reject/Modify) + adjuster comment + decision timestamp.

Output: downloadable `.pdf` via Streamlit's `st.download_button`.

---

## 13. Deployment

1. Push `app.py`, `report_generator.py`, `gemini_helper.py`, `requirements.txt`, and the trained model files to the GitHub repo (model files must stay small — MobileNetV2 head-only fine-tune + small ANN should comfortably fit under GitHub's 100MB/file limit; if not, host models on Google Drive and download them at app startup with `gdown`).
2. Go to share.streamlit.io → **New app** → connect the GitHub repo → set main file to `app.py`.
3. In the app's **Settings → Secrets**, add `GEMINI_API_KEY = "..."`.
4. Deploy → get public URL.

---

## 14. Business Model & Commercialization (content I'll draft for you)

- **Target customer:** Mid-size insurers and Third-Party Administrators (TPAs) who currently do manual claim triage.
- **Value proposition:** Cuts first-pass claim review time from hours/days to minutes; consistent, auditable, explainable decisions; human stays in control (regulatory-friendly).
- **Revenue model:** SaaS, priced per-claim processed (e.g., tiered per-claim fee) or per-seat monthly subscription for adjuster teams, with an enterprise tier for API/on-prem integration.
- **Go-to-market:** Start with regional insurers/TPAs as design partners, expand via integration with existing claims-management software vendors.
- **Competitive edge:** Cross-modal consistency checking (text vs image vs form) is the differentiator most competitors' single-modality tools don't do.
- **Compliance angle:** HITL + explainability directly support insurance-industry auditability requirements (this doubles as a rubric point and a real sales argument).

I'll turn this into a one-pager/slide-ready text block as a separate file once you approve this plan.

---

## 15. Detailed Timeline (~4.5-5 hours)

| Block | Time | What happens |
|---|---|---|
| 1. Setup | 15 min | Kaggle token, Gemini key, GitHub repo, Colab GPU runtime |
| 2. Data download | 25 min | Run provided Colab cells to pull both Kaggle datasets |
| 3. Train CNN | 40 min | Run provided training cells, verify accuracy is reasonable, export `damage_cnn.h5` |
| 4. Train ANN | 20 min | Run provided cells, verify AUC, export `risk_ann.h5` + scaler |
| 5. Grad-CAM + feature importance | 15 min | Run provided cells, sanity-check outputs on a sample claim |
| 6. Streamlit app | 90 min | I provide complete `app.py`; you run locally (`streamlit run app.py`), we fix any issues together |
| 7. PDF report module | 20 min | I provide `report_generator.py`; test a generated PDF |
| 8. Gemini integration | 15 min | I provide `gemini_helper.py`; test with your API key |
| 9. Deploy to Streamlit Cloud | 20 min | Push to GitHub, connect, set secrets, verify public URL works |
| 10. Business model doc | 20 min | I generate the pitch content; you review |
| 11. End-to-end rehearsal + buffer | 20-30 min | Full run-through with 2-3 sample claims, fix anything broken |

If time gets tight, cut order: (a) drop CNN unfreezing/fine-tuning of extra layers, (b) drop SHAP in favor of permutation importance (already the default plan), (c) skip public deployment and demo locally as a last resort — public URL stays the goal but local demo is an acceptable fallback that still satisfies "working web application."

---

## 16. Rubric Coverage Checklist

- [x] 3 data modalities: image, tabular, text
- [x] Predictive deep learning model: CNN (MobileNetV2 transfer learning) + ANN
- [x] LLM used only for reasoning/summarization/report generation (Gemini)
- [x] Human-in-the-loop: Approve/Reject/Modify panel
- [x] Explainable AI: Grad-CAM (image) + permutation feature importance (tabular) + confidence scores
- [x] Working web application: Streamlit, deployed publicly
- [x] Downloadable report: PDF via fpdf2
- [x] Business model & commercialization strategy: drafted in §14

---

## 17. What I need from you before I start coding

1. Confirm this plan (or flag anything you want changed).
2. Get the Kaggle token and Gemini API key ready (steps in §6) — I'll tell you exactly where to paste them.
3. Tell me once your Colab notebook is open with GPU runtime selected, and I'll hand you the first notebook cell.

Once approved, build order will be: **Colab training notebook → app.py → report_generator.py → gemini_helper.py → deployment → business model doc.**
