# tools

## Overview

Standalone applications that use `sphere_image` but are not part of the
installable package itself. Code under this directory is run directly (e.g.
via `streamlit run main.py` from the repository root); it is not built,
packaged, or importable as `sphere_image.*`.

## Components

| Component | Description |
| --- | --- |
| [viewer/](./viewer/) | Streamlit app powering the interactive sphere image viewer launched from the repository's `main.py`. |
