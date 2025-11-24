# components/intro.py
import streamlit as st
from streamlit_player import st_player

def show_intro():
    st.markdown("<h1 class='title'>Detect-Locate CT Tamperings</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced AI-Tampering Analysis of Medical Imaging Using Deep Learning</p>", unsafe_allow_html=True)

    # Optional: Add a forensic intro video (place in assets/background.mp4)
    # st_player("https://www.youtube.com/watch?v=4Q4YejlJjdM")  # Example cyber forensic video
    st.markdown("---")