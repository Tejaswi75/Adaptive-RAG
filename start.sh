#!/bin/sh
# Start the API in the background, wait until it answers, then start the UI.
set -e

uvicorn src.main:app --host 127.0.0.1 --port 8000 &

python - <<'PY'
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/", timeout=2)
        print("Backend is up")
        break
    except Exception:
        time.sleep(1)
PY

# XSRF/CORS off: Spaces serves the app inside an iframe, which breaks Streamlit uploads otherwise
exec streamlit run streamlit_app/home.py \
    --server.port 7860 --server.address 0.0.0.0 \
    --server.enableXsrfProtection false --server.enableCORS false \
    --browser.gatherUsageStats false
