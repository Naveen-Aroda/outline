import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import tempfile

def svg_to_high_quality_image(svg_path, output_size=2048, padding=400):
    """Convert SVG to high-quality PNG using Inkscape with padding"""
    inkscape_exe = "inkscape"
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_png_path = temp_file.name
    
    try:
        subprocess.run([
            inkscape_exe,
            svg_path,
            "--export-type=png",
            f"--export-filename={temp_png_path}",
            f"--export-area-page",
            f"--export-dpi=150"
        ], check=True, capture_output=True)
        
        image = Image.open(temp_png_path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        
        # Resize to square while maintaining aspect ratio
        h, w = cv_image.shape[:2]
        if h != w:
            size = max(h, w)
            square = np.full((size, size, 3), 255, dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = cv_image
            cv_image = square
        
        # Resize to target size
        cv_image = cv2.resize(cv_image, (output_size, output_size))
        
        # Add padding around the image
        padded = cv2.copyMakeBorder(cv_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded
    
    finally:
        if os.path.exists(temp_png_path):
            os.unlink(temp_png_path)

def find_contours(image):
    """Find contours in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary

def heavy_dilate(binary_image, iterations=4):
    """Apply heavy dilation to merge shapes"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    dilated = cv2.dilate(binary_image, kernel, iterations=iterations)
    
    return dilated

def get_outline(dilated_image):
    """Extract outline directly from dilated image"""
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(dilated_image)
    cv2.drawContours(outline, contours, -1, 255, 2)
    return outline, contours



def trace_bitmap_to_svg(png_path, output_svg_path):
    """Use potrace to trace bitmap and create vector SVG"""
    potrace_exe = "potrace"
    
    try:
        # Convert PNG to BMP for potrace
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        bmp_path = png_path.replace('.png', '.bmp')
        cv2.imwrite(bmp_path, image)
        
        # Convert to absolute paths
        abs_bmp_path = os.path.abspath(bmp_path)
        abs_svg_path = os.path.abspath(output_svg_path)
        
        subprocess.run([
            potrace_exe,
            abs_bmp_path,
            "-s",  # SVG output
            "-o", abs_svg_path
        ], check=True, capture_output=True, text=True)
        
        # Clean up BMP file
        os.unlink(bmp_path)
        
        return output_svg_path
    except subprocess.CalledProcessError as e:
        print(f"Error tracing bitmap: {e}")
        return None

def process_svg_file(svg_path, output_dir="output"):
    """Process a single SVG file through all steps"""
    print(f"Processing: {svg_path}")
    
    # Step 1: Convert SVG to high-quality image
    image = svg_to_high_quality_image(svg_path)
    
    # Step 2: Find contours
    contours, binary = find_contours(image)
    
    # Step 3: Heavy dilation
    dilated = heavy_dilate(binary)
    
    # Step 4: Get outline directly from dilated
    outline, final_contours = get_outline(dilated)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(svg_path).stem
    
    # Overlay outline on original image (keep black outline)
    overlay_image = image.copy()
    cv2.drawContours(overlay_image, final_contours, -1, (0, 0, 0), 5)  # Black outline
    overlay_png_path = f"{output_dir}/{base_name}_overlay.png"
    cv2.imwrite(overlay_png_path, overlay_image)
    
    # Trace overlay bitmap to vector SVG using potrace
    traced_overlay_svg = f"{output_dir}/{base_name}_traced_overlay.svg"
    
    print(f"Tracing overlay bitmap to SVG...")
    trace_bitmap_to_svg(overlay_png_path, traced_overlay_svg)
    
    # Clean up temporary overlay PNG
    os.unlink(overlay_png_path)
    
    return outline, final_contours

def main():
    svg_folder = "svg"
    output_folder = "output"
    
    svg_files = list(Path(svg_folder).glob("*.svg"))
    
    for svg_file in svg_files:
        try:
            outline, contours = process_svg_file(str(svg_file), output_folder)
            print(f"✓ Processed {svg_file.name}")
        except Exception as e:
            print(f"✗ Error processing {svg_file.name}: {e}")

if __name__ == "__main__":
    main()