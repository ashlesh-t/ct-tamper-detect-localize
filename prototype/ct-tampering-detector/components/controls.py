# components/controls.py
import streamlit as st

def upload_section():
    st.markdown("### Upload Volume or Folder of Slices")
    
    # Clear button at the top
    if st.session_state.get('vol_data') is not None:
        if st.button("🗑️ Clear Current Scan", type="secondary"):
            st.session_state.vol_data = None
            st.session_state.uploaded_files = []
            st.rerun()
    
    uploaded = st.file_uploader(
        "Supports: .dcm | .npy | .nii.gz | .jpg | Folder of .npy or DICOM slices",
        type=['dcm', 'nii', 'gz', 'npy', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="master_uploader",
        help="Upload one file OR multiple .npy/DICOM slices from a folder"
    )
    
    if uploaded:
        st.session_state['uploaded_files'] = uploaded
    return uploaded