# backend/Dockerfile (Final Version)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python build_vector_store.py

EXPOSE 8000

# --- THE FIX IS HERE ---
# We change from the "exec" form to the "shell" form of CMD.
# The shell (sh -c) correctly interprets the $PORT environment variable
# provided by Render and passes the actual port number to gunicorn.
CMD sh -c "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT"
