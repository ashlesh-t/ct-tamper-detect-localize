# app.py
import streamlit as st

# === 1. Force page config FIRST ===
st.set_page_config(
    page_title="Detect-Locate CT Tamperings",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === 2. Import Components ===
from config.theme import load_css
from components.intro import show_intro
from components.controls import upload_section
from components.viewer import show_viewer
from components.results import show_results # <--- New Import
from utils.loader import load_medical_image

# === 3. Initialize Session State ===
# This ensures our variables exist before we try to use them
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'vol_data' not in st.session_state:
    st.session_state.vol_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# === 4. Inject CSS ===
st.markdown(load_css(), unsafe_allow_html=True)

# === 5. MAIN ROUTER LOGIC ===

if st.session_state.current_page == 'home':
    # --- HOME PAGE (Intro + Upload + Viewer) ---
    
    show_intro()
    
    # Upload Section
    uploaded_files = upload_section()

    # Logic: Load volume ONLY if files are uploaded AND we haven't loaded them yet
    # OR if the user uploaded a new set of files.
    if uploaded_files:
        # We use a simple check: if vol_data is None, we load.
        # (For production, you might want to check if filenames changed to force reload)
        if st.session_state.vol_data is None:
            with st.spinner("Loading volume... This may take a few seconds"):
                volume = load_medical_image(uploaded_files)
                st.session_state.vol_data = volume # Persist in state
                st.rerun() # Rerun to refresh UI with loaded volume

    # Show Viewer if data exists
    if st.session_state.vol_data is not None:
        # Optional: Add a "Clear" button to reset
        if st.sidebar.button("Clear / Upload New"):
            st.session_state.vol_data = None
            st.rerun()
            
        st.success(f"Ready for forensic analysis! Volume Shape: {st.session_state.vol_data.shape}")
        show_viewer(st.session_state.vol_data)

    elif not uploaded_files:
        st.info("Upload a volume or folder of slices to begin...")


elif st.session_state.current_page == 'results':
    # --- RESULTS PAGE ---
    show_results()