---
## © Copyright
© 2026 Adssin (Aditya Gupta). All rights reserved.
This project is developed for academic and educational purposes.  
Unauthorized commercial use, distribution, or modification without permission is not allowed.
---

# HireSmart ATS - AI-Based Resume Screening & Feedback Platform (BY ADITYA GUPTA)

HireSmart ATS is a full-stack AI-powered resume screening platform that provides:
- Job Description (JD) based resume analysis
- ML-based General ATS Score (continuous percentage using regression)
- PDF resume & JD parsing
- Skill matching, missing skills, and AI suggestions
- Modern, responsive dashboard UI

---

## ✨ Features

- 📄 Upload Resume PDF
- 📎 Attach Job Description PDF or paste JD text
- 🎯 JD-Based Analysis:
  - ATS Match Score (%)
  - Resume Rating (out of 10)
  - Matched Skills
  - Missing Skills
  - AI Suggestions
- 🧠 ML-Based General ATS Score:
  - Uses a trained regression model
  - Outputs real continuous score (e.g., 59.29%, 72.45%)
  - Also shows quality label: Poor / Average / Good
- 🖥️ Clean, modern, responsive UI

---

## 🛠 Tech Stack

### Frontend
- React + TypeScript
- Tailwind CSS
- Vite
- Framer Motion

### Backend (ML API)
- Python
- FastAPI
- Scikit-learn
- Joblib
- TF-IDF Vectorizer

### Machine Learning
- Regression model trained on ATS score dataset
- Predicts continuous resume quality score (0–100)

---
