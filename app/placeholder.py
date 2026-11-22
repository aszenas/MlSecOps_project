"""Provide a temporary monitoring page for Compose integration."""

import os
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = Path(
    os.environ.get("REPORTS_DIR", str(PROJECT_ROOT / "reports"))
)

st.set_page_config(page_title="MLSecOps monitoring")
st.title("MLSecOps monitoring")
st.info("Monitoring dashboard setup is in progress.")

report_count = sum(path.is_file() for path in REPORTS_ROOT.rglob("*"))
st.metric("Available report files", report_count)
