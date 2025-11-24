# components/controls.py
import streamlit as st

def upload_section():
    st.markdown("### Upload Volume or Folder of Slices")
    uploaded = st.file_uploader(
        "Supports: .dcm | .npy | .nii.gz | .jpg | Folder of .npy or DICOM slices",
        type=['dcm', 'nii', 'gz', 'npy', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,   # THIS IS THE KEY
        key="master_uploader",
        help="Upload one file OR multiple .npy/DICOM slices from a folder"
    )
    if uploaded:
        st.session_state['uploaded_files'] = uploaded # <--- CRITICAL STEP
    return uploaded
