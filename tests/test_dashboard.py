"""Smoke tests for the Streamlit dashboard.

Streamlit pages call `st.set_page_config()` etc. at import time, so we
can't import them outside a Streamlit runtime without errors. The test
strategy here is to validate syntax via py_compile, not actual rendering.

For interactive testing, run:
    streamlit run dashboard/Home.py
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"


@pytest.mark.skipif(
    not DASHBOARD_DIR.exists(),
    reason="dashboard/ directory not present (skip if dashboard wasn't installed)",
)
class TestDashboardSyntax:
    """Verify all dashboard .py files compile cleanly."""

    def test_home_compiles(self) -> None:
        py_compile.compile(str(DASHBOARD_DIR / "Home.py"), doraise=True)

    def test_benchmark_results_compiles(self) -> None:
        py_compile.compile(
            str(DASHBOARD_DIR / "pages" / "1_Benchmark_Results.py"),
            doraise=True,
        )

    def test_prediction_explorer_compiles(self) -> None:
        py_compile.compile(
            str(DASHBOARD_DIR / "pages" / "2_Prediction_Explorer.py"),
            doraise=True,
        )

    def test_all_pages_in_pages_dir_compile(self) -> None:
        """Auto-discovery: every .py file in dashboard/pages/ must compile."""
        pages_dir = DASHBOARD_DIR / "pages"
        if not pages_dir.exists():
            pytest.skip("No pages directory")
        for py_file in pages_dir.glob("*.py"):
            py_compile.compile(str(py_file), doraise=True)
