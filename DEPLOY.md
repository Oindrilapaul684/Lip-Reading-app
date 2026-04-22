# 🚀 Deployment Guide — Lip Reading AI

## Project Structure
```
lip_reading_app/
├── app.py                  ← Flask backend (auto-downloads models at startup)
├── requirements.txt
├── Procfile                ← For Heroku / Railway
├── render.yaml             ← For Render
├── runtime.txt             ← Python version for Heroku
├── .gitignore
└── templates/
    └── index.html          ← Frontend (WebRTC camera)
```

> ✅ .pkl model files are NOT in this repo — they download automatically
>    from Google Drive when the app starts on the server.

---

## STEP 1 — Upload Your Models to Google Drive

1. Upload all your `.pkl` files to Google Drive.
2. For **each file**: right-click → **Share** → change to **"Anyone with the link"** → Copy link.
3. The shareable link looks like:
   ```
   https://drive.google.com/file/d/1ABCxyz_YOUR_FILE_ID_HERE/view?usp=sharing
   ```
   The part between `/d/` and `/view` is the **File ID** you need.

Collect the File ID for each model:

| Model file                | Environment variable name  |
|---------------------------|---------------------------|
| `random_forest.pkl`       | `GDRIVE_ALPHABET_RF`      |
| `random_forest_model.pkl` | `GDRIVE_DIGIT_RF`         |
| `pca_alphabet.pkl`        | `GDRIVE_PCA_ALPHABET`     |
| `pca_digit.pkl`           | `GDRIVE_PCA_DIGIT`        |
| `hmm_alphabet.pkl`        | `GDRIVE_HMM_ALPHABET`     |
| `hmm_digit.pkl`           | `GDRIVE_HMM_DIGIT`        |

---

## STEP 2 — Push to GitHub (no .pkl files needed)

```bash
git init
git add .
git commit -m "initial deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## STEP 3 — Deploy & Set Environment Variables

### 🔵 Render (Easiest — recommended)

1. Go to https://render.com → **New** → **Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --threads 2 --timeout 120`
4. Go to **Environment** tab and add these variables:

```
GDRIVE_ALPHABET_RF      = 1ABCxyz...
GDRIVE_DIGIT_RF         = 1DEFabc...
GDRIVE_PCA_ALPHABET     = 1GHIdef...
GDRIVE_PCA_DIGIT        = 1JKLghi...
GDRIVE_HMM_ALPHABET     = 1MNOjkl...
GDRIVE_HMM_DIGIT        = 1PQRmno...
```

5. Click **Deploy** — models download automatically on first boot.

---

### 🟣 Railway

1. Go to https://railway.app → **New Project** → **Deploy from GitHub**
2. Select your repo — Railway auto-detects `Procfile`
3. Go to **Variables** tab and add the same `GDRIVE_*` variables above
4. Click **Deploy**

---

### 🟠 Heroku

```bash
heroku login
heroku create your-app-name

heroku config:set GDRIVE_ALPHABET_RF=1ABCxyz...
heroku config:set GDRIVE_DIGIT_RF=1DEFabc...
heroku config:set GDRIVE_PCA_ALPHABET=1GHIdef...
heroku config:set GDRIVE_PCA_DIGIT=1JKLghi...
heroku config:set GDRIVE_HMM_ALPHABET=1MNOjkl...
heroku config:set GDRIVE_HMM_DIGIT=1PQRmno...

git push heroku main
heroku open
```

---

## How It Works at Startup

```
App boots
  │
  ├─ Checks for each .pkl file locally
  │     ├─ Already exists? → skip download
  │     └─ Missing? → download from Google Drive using GDRIVE_* env var
  │
  └─ Loads all models into memory → app is ready
```

Models are re-downloaded on every fresh deployment (cloud servers wipe their
filesystem on redeploy). Downloads typically take 10–60 seconds depending on
model size.

---

## 🌐 HTTPS Note
Browsers require HTTPS for webcam access. Render, Railway, and Heroku all
provide HTTPS automatically on their default domains. ✅

---

## 💡 Tips
- **Workers:** Keep `--workers 1` — MediaPipe's face_mesh is not fork-safe.
- **Timeout:** `--timeout 120` prevents gunicorn killing slow frame requests.
- **Slow first boot?** Large models take 1–2 min to download on first deploy.
- **Free tier sleep:** Render/Railway free tiers sleep after inactivity — models
  re-download on wakeup. Use a paid tier for production.
