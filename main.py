from __future__ import annotations

from tools.viewer import SphereViewerApp


def main() -> None:
    """
    Entry point for `streamlit run main.py`.
    """
    SphereViewerApp().run()


if __name__ == "__main__":
    main()
