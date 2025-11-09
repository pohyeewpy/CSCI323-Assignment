# Sentimental Analysis with Naive Bayes (CSCI323-Assignment)
# Group FT26 - Proj14

This is a group project for CSCI323, a local web app that analyzes restaurant reviews using a trained TF-IDF + Logistic Regression sentiment model.

---

## What’s Included
- Training model colab file: FTGroup26_Sentiment_Analysis.ipynb
- Website code folder: reviewlens_stack
- Models:
1. vectorizer.joblib
2. mnb.joblib
3. logistic.joblib
4. linearsvc.joblib
5. best_model.joblib (renamed linearsvc.joblib - same model)

This stack launches two services via Docker Compose:

| Service | Description | URL |
|----------|--------------|------|
| **api** | FastAPI backend (local sentiment model & insights) | [http://localhost:8000](http://localhost:8000) |
| **web** | React + Vite frontend dashboard | [http://localhost:5173](http://localhost:5173) |

---

## Prerequisites
- **Docker Desktop** installed and running  
  (Windows / macOS / Linux all supported)

- **Clone or unzip this folder**
Ensure this structure:
```
reviewlens_stack/
  docker-compose.yml
  backend/
    Dockerfile
    main.py
    requirements.txt
    reviews.csv
  frontend/
    Dockerfile
    package.json
    vite.config.js
    index.html
    .env.example
    postcss.config.cjs
    tailwind.config.cjs
    src/
      App.jsx
      main.jsx
      index.css
```

---

## Run the App
1. Open a terminal in the project root folder (where `docker-compose.yml` is located) or run it in VSCode.
2. Start docker  
3. Build and start all containers:
   ```bash
   docker compose up --build
   ```
4. Wait until you see:
   ```
   api-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
   web-1  | VITE v5.x  ready in ...ms
   ```
5. Then open your browser:
   - Frontend: [http://localhost:5173](http://localhost:5173)

---

## How to Use
1. In the web app, start typing a **restaurant name** (from the dataset).  
2. Select one from the suggestions: the search will auto-complete.  
3. Click **Analyze** to generate:
   - Overall sentiment and rating stats  
   - Top praises & top issues  
   - Sentiment breakdown pie chart  
   - Owner insight report with actionable suggestions  
   - Representative positive and negative quotes  

---

## Updating the App
If you make changes to the code or the dataset:
```bash
docker compose up --build
```
This will rebuild and restart the app with your latest updates.

---

## Useful Endpoints
| Endpoint | Description |
|-----------|--------------|
| `/api/places` | Lists all available restaurant names |
| `/api/ratings?place=<name>` | Returns rating and sentiment summary |
| `/api/report?place=<name>` | Full insight report (sentiment, praises, issues, suggestions) |
| `/healthz` | Health check for the backend |

Example:
```
http://localhost:8000/api/report?place=The%20Cow
```

---

## Tech Stack
- **Frontend:** React, TailwindCSS, Recharts, Vite  
- **Backend:** FastAPI, scikit-learn, pandas, joblib  
- **Deployment:** Docker Compose  

---

## Notes
- All processing is **offline**, using your local trained model.  
- Both frontend and backend run automatically through Docker; no manual setup needed.  
- To stop containers:  
  ```bash
  docker compose down
  ```

---

## Members
- Niruba Annriea Kichor Sagayaradje
- Joshe Chantiramugan
- Wong Poh Yee
- Vania Graciella Kwee
- Ngoc Minh Khanh Tran
- Angeles Micah Josephine Flores

---