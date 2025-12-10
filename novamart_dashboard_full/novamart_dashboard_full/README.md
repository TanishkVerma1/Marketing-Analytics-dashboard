
# NovaMart Marketing Analytics Dashboard (Completed)

This repository contains a **ready-to-run Streamlit dashboard** for the
NovaMart Marketing Analytics dataset used in your Data Visualization & Analytics class.

## Contents

- `app.py` – Completed Streamlit app with all pages and charts.
- `data/` – All CSV files from the NovaMart dataset already placed correctly.
- `requirements.txt` – Minimal dependencies needed on Streamlit Cloud / locally.

## How to Run (Locally)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Use with GitHub + Streamlit Cloud

1. Create a new GitHub repository.
2. Upload **all files and folders** from this project, including the `data/` folder.
3. On Streamlit Community Cloud, create a new app and point it to:
   - **Repository:** your GitHub repo
   - **Main file:** `app.py`
4. The app will automatically use the CSV files from the `data/` folder and should run without any missing-file errors.

No manual data wiring is needed – everything is pre-configured to work out of the box.
