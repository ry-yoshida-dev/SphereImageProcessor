"""
Entry point for `streamlit run main.py`.
"""

from __future__ import annotations

from tools.viewer import SphereViewerApp


def main() -> None:
    """
    Launch the Streamlit sphere image viewer.
    """
    SphereViewerApp().run()


if __name__ == "__main__":
    main()
