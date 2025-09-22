# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the build script to create the chroma_db folder inside the final image.
# This makes our vector store "read-only" but compatible with free hosting.
RUN python build_vector_store.py

# Expose the port the app will run on
EXPOSE 8000

# The command to run the production server using Gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:${PORT}"]