# FUSED Web UI

Entrypoint: `streamlit_app.py`

Run from the repository root:

```bash
streamlit run app/streamlit_app.py
```

The UI intentionally keeps optional heavy dependencies out of the default FUSED core. The app itself requires Streamlit; FUSED's base backend remains dependency-free.

For a full local verification before publishing:

```bat
RUN_FUSED_RELEASE_CHECK.bat
```

## TB13 beginner UX

Launch the app with `START_FUSED_UI.bat`. The left navigation is the main workflow. The `Data` page contains a dedicated 100M-token section with buttons for audit, generation, and full ingestion.
