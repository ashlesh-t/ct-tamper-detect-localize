# components/results.py
import streamlit as st

def show_results():
    st.title("📋 Forensic Analysis Report")
    
    # 1. Retrieve Data from Session State
    data = st.session_state.get('analysis_results', {})
    
    if not data:
        st.error("No analysis data found. Please return to the viewer.")
        if st.button("⬅️ Back to Viewer"):
            st.session_state['current_page'] = 'home'
            st.rerun()
        return

    # 2. Display Top Metrics
    st.markdown("### Classification Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Dynamic color based on classification
        cls = data.get('classification', 'Unknown')
        color = "red" if cls == "Tampered" or cls == "Fake" else "green"
        st.markdown(f"**Verdict:** :{color}[{cls}]")
        
    with col2:
        st.metric("Confidence Score", f"{data.get('confidence', 0)*100:.1f}%")
        
    with col3:
        st.metric("Slices Affected", f"{len(data.get('affected_slices', []))}")

    st.divider()

    # 3. detailed Visualization
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Localized Tampering Mask")
        # In a real app, you would load the actual generated mask image here
        # st.image(data.get('mask_path')) 
        st.info("Visualization of Gradient-CAM / Localization Map would appear here.")
        
        # Mock visual for demo
        st.bar_chart({"Region A": 0.1, "Region B": 0.9, "Region C": 0.2})

    with c2:
        st.subheader("Affected Slices")
        st.write("The following slice indices showed high probability of manipulation:")
        st.dataframe(
            {"Slice Index": data.get('affected_slices', []), 
             "Risk Level": ["High"] * len(data.get('affected_slices', []))}
        )

    st.divider()

    # 4. Navigation Back
    if st.button("⬅️ Analyze New Scan (Return to Home)", use_container_width=True):
        # Clear results but keep volume? Or clear everything?
        # Usually better to keep volume so they can look again.
        st.session_state['current_page'] = 'home'
        # Optional: Clear results to prevent stale data
        # del st.session_state['analysis_results'] 
        st.rerun()