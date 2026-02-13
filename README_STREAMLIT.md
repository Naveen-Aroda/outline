# SVG Outline Processor - Streamlit Web App

A modern, web-based interface for processing SVG files to extract smooth outlines.

## Features

- 🌐 **Web-based UI** - Access from any browser
- 📤 **File Upload** - Upload SVG files directly or select a folder
- ⚙️ **Customizable Parameters** - Adjust all processing parameters with sliders
- 📊 **Real-time Progress** - See processing progress and logs in real-time
- ✅ **Results Display** - View success/failure status for each file

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app:
```bash
streamlit run svg_processor_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## Usage

1. **Select Input Method:**
   - **Upload SVG File(s)**: Click to upload one or more SVG files
   - **Select Folder Path**: Enter the path to a folder containing SVG files

2. **Configure Parameters** (in sidebar):
   - **Outline Scale Multiplier**: How much bigger the outline should be (1.0-2.0)
   - **Epsilon Factor**: Point reduction/smoothing (lower = smoother)
   - **Base Tension**: Curve intensity (higher = more curved)
   - **Angle Threshold**: Corner detection threshold (degrees)
   - **Corner Smoothing**: Corner smoothing level (0.0-1.0)

3. **Set Output Folder**: Enter the directory where processed files will be saved

4. **Click "Process SVG(s)"**: The app will process all files and display results

## Output Files

For each processed SVG file, the app generates 4 output files:
- `*_outline_only.svg` - Filled outline version
- `*_overlay.svg` - Overlay of original + outline (filled)
- `*_outline_only_lightburn.svg` - Single-line stroked version for LightBurn
- `*_overlay_lightburn.svg` - Overlay with single-line stroke for LightBurn

## Advantages over Tkinter Version

- ✅ No segmentation faults or GUI issues
- ✅ Modern, responsive web interface
- ✅ Works on any device with a browser
- ✅ Easy to deploy and share
- ✅ Better file upload handling
- ✅ Real-time progress updates
- ✅ Cleaner, more intuitive UI



