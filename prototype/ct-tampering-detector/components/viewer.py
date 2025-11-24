import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from logs.logger import get_logger
import time
logger = get_logger(__name__)

class TamperPipeline:
    def __init__(self):
        pass

# from pipeline.get_results import TamperPipeline  # Import your pipeline

# --- HELPER: CSS GLOW EFFECT ---
def trigger_glow(color="red"):
    """Injects CSS to make the whole app screen pulse."""
    rgba = "255, 0, 0, 0.4" if color == "red" else "0, 255, 0, 0.4"
    st.markdown(f"""
    <style>
    .stApp {{
        animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: inset 0 0 0px rgba({rgba}); }}
        50% {{ box-shadow: inset 0 0 100px rgba({rgba}); }}
        100% {{ box-shadow: inset 0 0 0px rgba({rgba}); }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
def apply_windowing(image, level, window):
    """
    Apply medical windowing (contrast/brightness) efficiently.
    Input: 2D Numpy Array. Output: uint8 Image.
    """
    # Ensure we are working with native python floats for calculation
    level = float(level)
    window = float(window)
    
    min_val = level - window / 2.0
    max_val = level + window / 2.0
    
    img_windowed = np.clip(image, min_val, max_val)
    
    # Normalize to 0-255
    if max_val != min_val:
        img_windowed = ((img_windowed - min_val) / (window) * 255)
    else:
        img_windowed = img_windowed - min_val
        
    return img_windowed.astype(np.uint8)

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Best for forensic analysis of local artifacts.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def show_viewer(volume: np.ndarray):
    if volume is None or volume.size == 0:
        st.warning("No volume loaded.")
        return

    # Ensure standard layout (Depth, Height, Width)
    if volume.ndim == 2:
        volume = np.expand_dims(volume, 0)
    
    depth, height, width = volume.shape

    # --- STATE INITIALIZATION (With Strict Float Casting) ---
    # We use explicit keys in session_state to persist slider values
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = depth // 2
    
    # Initialize Window/Level with NATIVE Python floats (Crucial for Streamlit)
    if "val_level" not in st.session_state:
        st.session_state.val_level = float((volume.max() + volume.min()) / 2.0)
    if "val_width" not in st.session_state:
        st.session_state.val_width = float(volume.max() - volume.min())

    # Safe index clamping
    st.session_state.slice_idx = max(0, min(st.session_state.slice_idx, depth - 1))

    # --- STYLING ---
    st.markdown("""
    <style>
    .viewer-container { background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 5px; }
    .medical-text { font-family: 'Segoe UI', sans-serif; color: #E0E0E0; font-size: 0.9rem; letter-spacing: 0.5px; }
    .slice-indicator { font-family: 'Roboto Mono', monospace; color: #4FC3F7; font-weight: bold; font-size: 1.2rem; }
    .coords-box { background-color: #2D2D2D; border-left: 3px solid #4FC3F7; padding: 8px; margin-top: 10px; font-family: 'Roboto Mono', monospace; color: #ccc; font-size: 0.85rem; }
    
    /* Tweak slider color to look "Medical" */
    div.stSlider > div > div > div > div { background-color: #4FC3F7; }
    
    div.stButton > button { width: 100%; border-radius: 2px; border: 1px solid #444; background-color: #262626; color: white; }
    div.stButton > button:hover { border-color: #4FC3F7; color: #4FC3F7; }
    </style>
    """, unsafe_allow_html=True)

    col_main, col_sidebar = st.columns([3, 1])

    with col_main:
        # 1. Get Slice
        current_slice = volume[st.session_state.slice_idx].astype(np.float32)

        # 2. Apply Windowing
        # Note: We use the session state values directly
        display_img = apply_windowing(
            current_slice, 
            st.session_state.val_level, 
            st.session_state.val_width
        )
        
        # 3. Optional: CLAHE (Better HistEq)
        if st.session_state.get("enable_clahe", False):
            display_img = apply_clahe(display_img)

        # 4. Prepare for Canvas (PIL format)
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        display_pil = Image.fromarray(display_img_rgb)

        # Calculate aspect ratio
        max_canvas_size = 600
        img_h, img_w = display_img.shape
        scale_factor = min(max_canvas_size / img_h, max_canvas_size / img_w)
        canvas_height = int(img_h * scale_factor)
        canvas_width = int(img_w * scale_factor)

        # Header
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span class="medical-text">AXIAL VIEW</span>
            <span class="slice-indicator">SLICE: {st.session_state.slice_idx + 1} / {depth}</span>
        </div>
        """, unsafe_allow_html=True)

        # 5. Render Canvas
        # Note: We pass the PIL image to 'background_image'
        st_canvas(
            fill_color="rgba(255, 165, 0, 0.0)",
            stroke_width=1,
            background_color="#000000",
            background_image=display_pil,
            update_streamlit=False, # Set False for performance if you don't need click events
            height=canvas_height,
            width=canvas_width,
            drawing_mode="point",
            point_display_radius=0,
            key="medical_canvas",
            display_toolbar=False,
        )

        # 6. Zero-Lag Mouse Tracking (JavaScript)
        st.components.v1.html(f"""
        <script>
        const canvas = window.parent.document.querySelector('canvas');
        if (canvas) {{
            canvas.addEventListener('mousemove', function(e) {{
                const rect = canvas.getBoundingClientRect();
                const scaleX = {img_w} / rect.width;
                const scaleY = {img_h} / rect.height;
                
                const x = Math.floor((e.clientX - rect.left) * scaleX);
                const y = Math.floor((e.clientY - rect.top) * scaleY);
                
                // Clamp values
                const cx = Math.max(0, Math.min(x, {img_w - 1}));
                const cy = Math.max(0, Math.min(y, {img_h - 1}));
                
                const disp = window.parent.document.getElementById('live-coords');
                if (disp) {{
                    disp.innerHTML = `X: ${{cx}} | Y: ${{cy}}`;
                }}
            }});
        }}
        </script>
        """, height=0)

        st.markdown(f"""
        <div class="coords-box">
            <div style="display:flex; justify-content:space-between;">
                <span id="live-coords">Hover for coordinates...</span>
                <span>Matrix: {width}x{height}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sidebar:
        st.markdown("#### Navigation")
        
        # Slice Slider
        if depth > 1:
            # Define callbacks that mutate session_state BEFORE widgets are instantiated on rerun
            def _prev():
                cur = int(st.session_state.get("slice_idx", int(depth // 2)))
                st.session_state["slice_idx"] = max(0, cur - 1)

            def _next():
                cur = int(st.session_state.get("slice_idx", int(depth // 2)))
                st.session_state["slice_idx"] = min(depth - 1, cur + 1)

            c1, c2 = st.columns(2)
            with c1:
                st.button("PREV", on_click=_prev, key="btn_prev")
            with c2:
                st.button("NEXT", on_click=_next, key="btn_next")

            # Create slider using the current session value so it reflects any callback changes
            st.slider(
                "Slice Index",
                min_value=0,
                max_value=depth - 1,
                value=int(st.session_state.get("slice_idx", depth // 2)),
                key="slice_idx",
                label_visibility="collapsed"
            )

        st.markdown("---")
        st.markdown("#### Contrast / Filters")
        
        # Safe Float Conversions for Min/Max
        vol_min = float(volume.min())
        vol_max = float(volume.max())
        
        # Window Level Slider (Brightness)
        st.slider(
            "Level (Brightness)",
            min_value=vol_min,
            max_value=vol_max,
            key="val_level" # Auto-updates session_state.val_level
        )
        
        # Window Width Slider (Contrast)
        st.slider(
            "Width (Contrast)",
            min_value=1.0, # Explicit float
            max_value=float(vol_max - vol_min), # Explicit float cast
            key="val_width" # Auto-updates session_state.val_width
        )

        # CLAHE Toggle (Better HistEq)
        st.markdown("---")
        st.checkbox("🔍 Enhanced Contrast (CLAHE)", key="enable_clahe", help="Use adaptive histogram equalization to reveal hidden artifacts.")

        if st.button("Run Tamper Detection", type="primary"):
            pipeline = TamperPipeline()
            
            # 1. PREPARE DATA
            # We need both the 3D volume and the filenames.
            # Assuming 'volume' is your (D, H, W) numpy array available in this scope
            # and 'uploaded_files' (list of file objects) is in session_state.
            
            raw_files = st.session_state.get('uploaded_files', [])
            
            # Case A: User uploaded actual files
            if raw_files:
                # Create a list of dictionaries: [{'fname': 'slice01.dcm', 'data': <2D_Numpy_Array>}, ...]
                payload = []
                
                # Sort files by name first to match the volume order
                # (Assuming volume construction followed this sort order)
                sorted_raw_files = sorted(raw_files, key=lambda x: x.name)
                
                for idx, file_obj in enumerate(sorted_raw_files):
                    # Safety check: Ensure volume depth matches file count
                    if idx < depth: 
                        slice_data = volume[idx] # Extract the 2D slice from the 3D volume
                        
                        payload.append({
                            "fname": file_obj.name,
                            "data": slice_data
                        })
                
                sorted_payload = payload # Already sorted by virtue of sorted_raw_files

            # Case B: Fallback (e.g., Demo mode or simple NPY upload without filenames)
            else:
                # Create synthetic filenames if real ones aren't tracked
                sorted_payload = []
                for i in range(depth):
                    sorted_payload.append({
                        "fname": f"slice_{i:04d}.npy", # Standardized naming
                        "data": volume[i]
                    })

            # 2. THE ANIMATION CYCLE (st.status)
            try:
                with st.status("Initializing Forensics Pipeline...", expanded=True) as status:
                    
                    # Visual: Data Ingestion
                    st.write(f"📂 Packaging {len(sorted_payload)} slices for analysis...")
                    time.sleep(0.5) 
                    
                    # Visual: Preprocessing
                    st.write("⚙️ verifying numpy serialization...")
                    time.sleep(0.5)
                    
                    # Visual: Inference
                    st.write("🧠 Running EfficientNet-V2 Inference...")
                    
                    # --- CALL THE PIPELINE ---
                    # Now passing the list of dicts as requested
                    status_code, result_data = pipeline.analyze_volume(sorted_payload)
                    
                    # Visual: Completion
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # 3. HANDLE RESPONSES (Same as before)
                if status_code == 200:
                    st.success("Tampering Localized. Generating Report...")
                    time.sleep(0.5)
                    st.session_state['analysis_results'] = result_data
                    st.session_state['current_page'] = 'results'
                    st.rerun()

                elif status_code == 206:
                    cls = result_data.get('classification', 'Unknown')
                    if cls == "Fake":
                        trigger_glow("red")
                        st.error(f"⚠️ DETECTED: FAKE")
                        st.warning("Localization failed: Unable to pinpoint region.")
                    else:
                        trigger_glow("green")
                        st.success(f"✅ VERIFIED: REAL")
                        st.info("Authenticity verified. Localization skipped.")
                        
                else:
                    st.error(f"❌ Pipeline Error: {result_data.get('error')}")
                    time.sleep(3)
                    st.session_state.clear()
                    st.session_state['current_page'] = 'home'
                    st.rerun()

            except Exception as e:
                st.error(f"Critical System Failure: {str(e)}")
                # Log the full traceback in your logs for debugging
                logger.error(f"Pipeline crash: {e}", exc_info=True) 
                time.sleep(3)
                st.rerun()