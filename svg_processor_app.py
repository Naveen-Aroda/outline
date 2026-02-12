#!/usr/bin/env python3
"""
SVG Outline Processor - Streamlit Web Application
A modern, user-friendly web interface for processing SVG files to extract outlines.
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
import shutil
from process_svg_v2 import process_svg

# Page configuration
st.set_page_config(
    page_title="SVG Outline Processor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []


def main():
    # Header
    st.markdown('<div class="main-header">SVG Outline Processor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Extract smooth outlines from SVG files with customizable parameters</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Processing Parameters")
        
        outline_scale = st.slider(
            "Outline Scale Multiplier",
            min_value=1.0,
            max_value=2.0,
            value=1.4,
            step=0.05,
            help="How much bigger the outline should be (1.0 = same size)"
        )
        
        epsilon_factor = st.slider(
            "Epsilon Factor",
            min_value=0.0001,
            max_value=0.001,
            value=0.00015,
            step=0.00001,
            format="%.5f",
            help="Point reduction/smoothing (lower = smoother, more points)"
        )
        
        base_tension = st.slider(
            "Base Tension",
            min_value=0.3,
            max_value=3.0,
            value=0.6,
            step=0.1,
            help="Curve intensity (higher = more curved/flowing)"
        )
        
        angle_threshold = st.slider(
            "Angle Threshold (degrees)",
            min_value=90,
            max_value=180,
            value=160,
            step=5,
            help="Corner detection threshold (lower = only sharp corners)"
        )
        
        corner_tension_reduction = st.slider(
            "Corner Smoothing",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Corner smoothing (0.0 = maximum smoothing, 1.0 = no smoothing)"
        )
        
        st.markdown("---")
        
        if st.button("Reset to Defaults", use_container_width=True):
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📂 Input Selection")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Upload SVG File(s)", "Select Folder Path"],
            horizontal=True
        )
        
        uploaded_files = None
        folder_path = None
        
        if input_method == "Upload SVG File(s)":
            uploaded_files = st.file_uploader(
                "Choose SVG file(s)",
                type=['svg'],
                accept_multiple_files=True,
                help="Upload one or more SVG files to process"
            )
        else:
            folder_path = st.text_input(
                "Folder Path",
                placeholder="Enter path to folder containing SVG files",
                help="Enter the full path to a folder containing SVG files"
            )
            if folder_path and not os.path.exists(folder_path):
                st.error(f"Path does not exist: {folder_path}")
        
        # Output directory
        st.header("📁 Output Settings")
        output_dir = st.text_input(
            "Output Folder",
            value="output",
            help="Directory where processed files will be saved"
        )
    
    with col2:
        st.header("Configuration Summary")
        st.markdown(f"""
        **Outline Scale:** {outline_scale}x  
        **Epsilon Factor:** {epsilon_factor:.5f}  
        **Base Tension:** {base_tension}  
        **Angle Threshold:** {angle_threshold}°  
        **Corner Smoothing:** {corner_tension_reduction}
        """)
    
    # Process button
    st.markdown("---")
    
    # Determine if we have input to process
    has_input = False
    files_to_process = []
    
    if input_method == "Upload SVG File(s)":
        if uploaded_files:
            has_input = True
            files_to_process = uploaded_files
    else:
        if folder_path and os.path.exists(folder_path):
            svg_files = list(Path(folder_path).glob("*.svg"))
            if svg_files:
                has_input = True
                files_to_process = svg_files
    
    if has_input:
        if st.button("Process SVG(s)", type="primary", use_container_width=True, disabled=st.session_state.processing):
            process_files(
                files_to_process,
                output_dir,
                outline_scale,
                epsilon_factor,
                base_tension,
                angle_threshold,
                corner_tension_reduction,
                input_method == "Upload SVG File(s)"
            )
    else:
        st.info("Please select files or a folder to process")
    
    # Display results
    if st.session_state.processed_files:
        st.markdown("---")
        st.header("Processing Results")
        
        for result in st.session_state.processed_files:
            if result['success']:
                st.markdown(f"""
                <div class="success-box">
                    <strong>{result['name']}</strong><br>
                    Output saved to: {result['output_path']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <strong>{result['name']}</strong><br>
                    Error: {result['error']}
                </div>
                """, unsafe_allow_html=True)


def process_files(files_to_process, output_dir, outline_scale, epsilon_factor, 
                 base_tension, angle_threshold, corner_tension_reduction, is_uploaded):
    """Process the selected files"""
    st.session_state.processing = True
    st.session_state.processed_files = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()
    
    total_files = len(files_to_process)
    
    with log_container:
        st.markdown("### Processing Log")
        log_placeholder = st.empty()
        log_messages = []
        
        log_messages.append("=" * 60)
        log_messages.append("SVG Outline Processor - Starting")
        log_messages.append("=" * 60)
        log_messages.append(f"Output directory: {output_dir}")
        log_messages.append(f"Configuration:")
        log_messages.append(f"  - Outline scale: {outline_scale}x")
        log_messages.append(f"  - Epsilon factor: {epsilon_factor}")
        log_messages.append(f"  - Base tension: {base_tension}")
        log_messages.append(f"  - Angle threshold: {angle_threshold}°")
        log_messages.append(f"  - Corner smoothing: {corner_tension_reduction}")
        log_messages.append("")
        log_placeholder.code("\n".join(log_messages))
    
    success_count = 0
    
    for idx, file_item in enumerate(files_to_process):
        # Update progress
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        
        # Handle uploaded files vs folder files
        if is_uploaded:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='wb') as tmp_file:
                tmp_file.write(file_item.read())
                tmp_path = tmp_file.name
            file_name = file_item.name
        else:
            tmp_path = str(file_item)
            file_name = file_item.name
        
        status_text.info(f"Processing {idx + 1}/{total_files}: {file_name}")
        
        try:
            # Process the file
            result = process_svg(
                tmp_path,
                output_dir,
                outline_scale,
                angle_threshold,
                corner_tension_reduction,
                epsilon_factor,
                base_tension
            )
            
            if result:
                success_count += 1
                log_messages.append(f"[{idx + 1}/{total_files}] Successfully processed: {file_name}")
                
                st.session_state.processed_files.append({
                    'name': file_name,
                    'success': True,
                    'output_path': output_dir
                })
            else:
                log_messages.append(f"[{idx + 1}/{total_files}] Failed to process: {file_name}")
                
                st.session_state.processed_files.append({
                    'name': file_name,
                    'success': False,
                    'error': 'Processing returned None'
                })
            
            # Clean up temp file if uploaded
            if is_uploaded and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
        except Exception as e:
            log_messages.append(f"[{idx + 1}/{total_files}] Error processing {file_name}: {str(e)}")
            
            st.session_state.processed_files.append({
                'name': file_name,
                'success': False,
                'error': str(e)
            })
            
            # Clean up temp file if uploaded
            if is_uploaded and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Update log display
        log_placeholder.code("\n".join(log_messages))
    
    # Final summary
    log_messages.append("")
    log_messages.append("=" * 60)
    log_messages.append(f"Completed: {success_count}/{total_files} files processed successfully")
    log_messages.append("=" * 60)
    log_placeholder.code("\n".join(log_messages))
    
    progress_bar.progress(1.0)
    
    if success_count == total_files:
        status_text.success(f"All {total_files} file(s) processed successfully!")
    elif success_count > 0:
        status_text.warning(f"{success_count}/{total_files} file(s) processed successfully")
    else:
        status_text.error(f"All {total_files} file(s) failed to process")
    
    st.session_state.processing = False
    st.rerun()


if __name__ == "__main__":
    main()

