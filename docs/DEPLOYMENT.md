# Deployment

FUSED supports three practical deployment paths.

## 1. Local Windows

For a non-programmer:

```text
START_FUSED_UI.bat
```

The setup path creates `.venv`, installs the required UI dependency, and starts Streamlit.

## 2. Local Docker

Build:

```bash
docker build -t fused .
```

Run:

```bash
docker run --rm -p 8501:8501 fused
```

Open:

```text
http://localhost:8501
```

## 3. Streamlit hosting

The application entry point is:

```text
app/streamlit_app.py
```

The UI dependency is declared in:

```text
app/requirements.txt
```

Use a hosted environment's secret-management facility for `GEMINI_API_KEY`. Never commit the key.

## Local release gate

Before publishing:

```text
CHECK_FUSED_READY.bat
RUN_FUSED_RELEASE_CHECK.bat
```

The release check is intended to validate the repository locally without requiring a Gemini API call.

## Deployment boundary

The local research application is not the same thing as a scalable distributed production service. Large retrieval, concurrent users, authentication, monitoring, secrets management, and long-running model serving need additional infrastructure.
