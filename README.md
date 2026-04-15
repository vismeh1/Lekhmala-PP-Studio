# Lekhmala Photo Studio — Reflex Port

Converted from Streamlit (`pro_studio.py`) to [Reflex](https://reflex.dev).

## Project structure

```
lekhmala_reflex/
├── rxconfig.py
├── requirements.txt
├── README.md
└── lekhmala_studio/
    └── lekhmala_studio.py
```

## Setup

```bash
# 1. Clone / copy this folder
cd lekhmala_reflex

# 2. Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install PyTorch first (CPU example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining deps
pip install -r requirements.txt

# 5. Initialise Reflex (first run only — downloads Node.js frontend toolchain)
reflex init

# 6. Run in dev mode
reflex run
```

The app will be available at **http://localhost:3000**.

## Key differences vs Streamlit version

| Streamlit | Reflex |
|---|---|
| `st.session_state` | `rx.State` vars |
| `@st.cache_resource` | Module-level singleton `_get_enhancer()` |
| `st.tabs` | `rx.tabs.root / trigger / content` |
| `st.file_uploader` | `rx.upload` + `rx.upload_files` |
| `st.slider` | `rx.slider` with `on_value_commit` |
| `st.download_button` | Event handler returning `rx.download(...)` |
| `st.spinner` + `yield` | `is_processing` state var + `yield` in async handler |
| Custom CSS | Reflex theme + inline props |

## Production deployment

```bash
reflex export          # builds static frontend + API server
# Deploy the generated .web/ and run the Python API server
```

No Streamlit Community Cloud memory limits — Reflex runs your own backend process.
