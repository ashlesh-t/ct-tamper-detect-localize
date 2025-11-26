# components/results.py
import streamlit as st
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from logs.logger import get_logger

logger = get_logger(__name__)

def apply_red_glow():
    """Apply red glow effect to the entire app"""
    st.markdown("""
    <style>
    .stApp {
        animation: redPulse 2s infinite;
        border: 2px solid #ff4444;
    }
    @keyframes redPulse {
        0% { box-shadow: 0 0 0px rgba(255, 68, 68, 0.4); }
        50% { box-shadow: 0 0 50px rgba(255, 68, 68, 0.8); }
        100% { box-shadow: 0 0 0px rgba(255, 68, 68, 0.4); }
    }
    </style>
    """, unsafe_allow_html=True)

def apply_green_glow():
    """Apply green glow effect for authentic scans"""
    st.markdown("""
    <style>
    .stApp {
        animation: greenPulse 3s infinite;
        border: 2px solid #44ff44;
    }
    @keyframes greenPulse {
        0% { box-shadow: 0 0 0px rgba(68, 255, 68, 0.3); }
        50% { box-shadow: 0 0 40px rgba(68, 255, 68, 0.6); }
        100% { box-shadow: 0 0 0px rgba(68, 255, 68, 0.3); }
    }
    </style>
    """, unsafe_allow_html=True)

def create_heatmap_overlay(original_slice, heatmap_b64, coords=None):
    """Create heatmap overlay on original CT slice"""
    try:
        # Decode base64 heatmap
        heatmap_data = base64.b64decode(heatmap_b64)
        heatmap_img = Image.open(BytesIO(heatmap_data))
        heatmap_array = np.array(heatmap_img)
        
        # Convert original slice to RGB for overlay
        original_normalized = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min() + 1e-8)
        original_uint8 = (original_normalized * 255).astype(np.uint8)
        original_rgb = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match original dimensions
        heatmap_resized = cv2.resize(heatmap_array, (original_rgb.shape[1], original_rgb.shape[0]))
        
        # Blend images (70% original, 30% heatmap)
        blended = cv2.addWeighted(original_rgb, 0.7, heatmap_resized, 0.3, 0)
        
        # Add bounding box if coordinates available
        if coords and len(coords) == 4:
            # Convert coordinates to integer tuples
            points = np.array(coords, dtype=np.int32)
            cv2.polylines(blended, [points], True, (255, 0, 0), 3)
            
            # Add X mark at center of bounding box
            center_x = sum(point[0] for point in coords) // 4
            center_y = sum(point[1] for point in coords) // 4
            marker_size = 32
            
            # Draw X mark
            cv2.line(blended, 
                    (center_x - marker_size, center_y - marker_size),
                    (center_x + marker_size, center_y + marker_size),
                    (0, 0, 255), 4)
            cv2.line(blended,
                    (center_x + marker_size, center_y - marker_size),
                    (center_x - marker_size, center_y + marker_size),
                    (0, 0, 255), 4)
        
        return blended
    except Exception as e:
        logger.error(f"Error creating heatmap overlay: {e}")
        # Return original image if heatmap fails
        original_normalized = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min() + 1e-8)
        original_uint8 = (original_normalized * 255).astype(np.uint8)
        return cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2RGB)

def show_results():
    st.markdown("""
    <style>
    .report-header {
        background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #444;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
        margin: 0.5rem 0;
    }
    .safe-card {
        border-left: 4px solid #44ff44;
    }
    .slice-viewer {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Retrieve data from session state
    data = st.session_state.get('analysis_results', {})
    volume_data = st.session_state.get('vol_data', None)
    
    if not data:
        st.error("No analysis data found. Please return to the viewer.")
        if st.button("⬅️ Back to Viewer"):
            st.session_state['current_page'] = 'home'
            st.rerun()
        return
    print("="*10, "DATA IN RESULTS", data, "\n", "="*10)
    classification = data.get('classification', 'Unknown')
    raw_conf = data.get('confidence', 0)

    # If dict → use overall confidence
    if isinstance(raw_conf, dict):
        confidence = raw_conf.get('overall', 0)
    else:
        confidence = raw_conf

        
    # Apply appropriate glow based on classification
    if classification in ['Fake', 'Tampered']:
        apply_red_glow()
    else:
        apply_green_glow()

    # Header Section
    st.markdown(f"""
    <div class="report-header">
        <h1 style="color: {'#ff4444' if classification in ['Fake', 'Tampered'] else '#44ff44'}; 
                   text-align: center; margin-bottom: 1rem;">
            🔍 FORENSIC ANALYSIS REPORT
        </h1>
        <h2 style="color: {'#ff4444' if classification in ['Fake', 'Tampered'] else '#44ff44'}; 
                   text-align: center; font-size: 2.5rem;">
            {classification.upper()}
        </h2>
        <p style="text-align: center; color: #ccc; font-size: 1.2rem;">
            Confidence: <strong>{confidence*100:.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back to Viewer", use_container_width=True):
            st.session_state['current_page'] = 'home'
            st.rerun()
    with col2:
        st.write("")  # Spacer
    with col3:
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.session_state['current_page'] = 'home'
            st.rerun()

    st.divider()

    # Detailed Metrics Section
    st.markdown("### 📊 Detailed Analysis Metrics")
    
    if classification in ['Fake', 'Tampered']:
        # Tampered case - show detailed breakdown
        detailed_data = data.get('data', {})
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Slices Analyzed</h4>
                <h2>{detailed_data.get('total_slices', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Tampered Slices</h4>
                <h2 style="color: #ff4444">{detailed_data.get('tampered_slices_count', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            sub_classification = data.get('sub_classification', 'Unknown')
            sub_confidence = data.get('confidence', {}).get('sub_classification', 0) if isinstance(data.get('confidence'), dict) else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>Tamper Type</h4>
                <h2>{sub_classification}</h2>
                <p>Confidence: {sub_confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Show injection/removal breakdown if available
        tamper_analysis = detailed_data.get('tamper_analysis', {})
        if tamper_analysis:
            inj_col, rem_col = st.columns(2)
            with inj_col:
                injected_data = tamper_analysis.get('injected', {})
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🔫 Injected Regions</h4>
                    <h3>{injected_data.get('count', 0)}</h3>
                    <p>Localized slices: {len(injected_data.get('localized_slices', []))}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with rem_col:
                removed_data = tamper_analysis.get('removed', {})
                st.markdown(f"""
                <div class="metric-card">
                    <h4>✂️ Removed Regions</h4>
                    <h3>{removed_data.get('count', 0)}</h3>
                    <p>Localized slices: {len(removed_data.get('localized_slices', []))}</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        # Real/authentic case
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card safe-card">
                <h4>✅ Authentication Verified</h4>
                <p>This CT scan appears to be authentic with no signs of tampering.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            slice_stats = data.get('slice_statistics', {})
            st.markdown(f"""
            <div class="metric-card safe-card">
                <h4>Slice Analysis</h4>
                <p>Real slices: {slice_stats.get('slices_predicted_real', 'N/A')}</p>
                <p>Fake slices: {slice_stats.get('slices_predicted_fake', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Slice Visualization Section
    if volume_data is not None and classification in ['Fake', 'Tampered']:
        st.markdown("### 🎯 Tampering Localization Viewer")
        
        # Prepare localization data
        localization_data = {}
        detailed_data = data.get('data', {})
        tamper_analysis = detailed_data.get('tamper_analysis', {})
        
        # Collect all localized slices
        for loc_type in ['injected', 'removed']:
            loc_slices = tamper_analysis.get(loc_type, {}).get('localized_slices', [])
            for slice_info in loc_slices:
                filename = slice_info.get('filename', '')
                localization_data[filename] = {
                    'heatmap': slice_info.get('heatmap', ''),
                    'coords': slice_info.get('coords', []),
                    'type': loc_type
                }
        
        # Create slice selector
        depth = volume_data.shape[0]
        
        # Find slices with localization data
        print("="*10, "localization_data: ", localization_data, "\n", "="*10)
        localized_indices = {i: filename for i, filename in enumerate(localization_data)}
        print(localization_data)
        print(localized_indices)
        
        if localized_indices:
            st.info(f"🔍 Found {len(localized_indices)} slices with localized tampering")
            
            # Slice navigation
            # localized_indices => dict: {slice_idx: filename}

            available_slice_indices = list(localized_indices.keys())

            # UI selection
            if len(available_slice_indices) > 1:
                selected_index = st.select_slider(
                    "Navigate to slice:",
                    options=available_slice_indices,
                    format_func=lambda x: f"Slice {x} - {localization_data[localized_indices[x]]['type'].title()}",
                    key="slice_navigator"
                )
            else:
                selected_index = available_slice_indices[0]
                st.info("Only one slice available for this patient.")

            # Retrieve filename + metadata
            selected_filename = localized_indices[selected_index]
            info = localization_data[selected_filename]

            st.markdown(f"**Slice {selected_index} - {info['type'].title()}**")

            if selected_index is not None:
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("#### Original CT Slice")
                    original_slice = volume_data[selected_index]
                    
                    # Normalize and display original
                    original_normalized = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min() + 1e-8)
                    original_uint8 = (original_normalized * 255).astype(np.uint8)
                    
                    st.image(original_uint8, use_column_width=True, caption=f"Slice {selected_index} - Original")
                
                with col_right:
                    st.markdown("#### Tampering Localization")
                    current_file = localized_indices[selected_index]
                    loc_info = localization_data.get(current_file, {})
                    
                    if loc_info.get('heatmap'):
                        overlay_img = create_heatmap_overlay(
                            volume_data[selected_index],
                            loc_info['heatmap']
                        )
                        st.image(overlay_img, use_column_width=True, 
                                caption=f"Slice {selected_index} - {loc_info.get('type', 'Unknown').title()} Detection")
                    else:
                        st.warning("No localization data available for this slice")
                
                # Show slice details
                st.markdown("##### 📋 Slice Details")
                detail_cols = st.columns(3)
                with detail_cols[0]:
                    st.metric("Slice Index", selected_index)
                with detail_cols[1]:
                    st.metric("Tamper Type", loc_info.get('type', 'Unknown').title())
                with detail_cols[2]:
                    if loc_info.get('coords'):
                        st.metric("Bounding Box", "Detected")
                    else:
                        st.metric("Bounding Box", "Not Available")
        else:
            st.warning("No localized tampering regions found in the volume")
    
    elif volume_data is not None:
        st.markdown("### 📄 Volume Summary")
        # For real volumes, just show a sample slice
        sample_slice = volume_data[depth // 2] if volume_data.shape[0] > 0 else volume_data[0]
        sample_normalized = (sample_slice - sample_slice.min()) / (sample_slice.max() - sample_slice.min() + 1e-8)
        sample_uint8 = (sample_normalized * 255).astype(np.uint8)
        
        st.image(sample_uint8, use_column_width=True, caption="Sample CT Slice - Authentic")
    
    # Footer with timestamp
    st.divider()
    from datetime import datetime
    st.caption(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")