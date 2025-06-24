FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependency manifests for cache
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Expose application port
EXPOSE 8000

# Run the FastAPI application
CMD ["poetry", "run", "python", "-m", "app.main"]