import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import re


def rasterize_svg(svg_path, output_size=2048, padding=400):
    """Convert SVG to high-quality PNG using Inkscape"""
    inkscape_exe = os.path.join(os.getcwd(), "inkscape", "bin", "inkscape.exe")
    
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
        
        # Load and process image
        image = Image.open(temp_png_path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        
        # Make square
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
        
        # Add padding
        padded = cv2.copyMakeBorder(
            cv_image, padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        return padded
    
    finally:
        if os.path.exists(temp_png_path):
            os.unlink(temp_png_path)


def extract_outline(image):
    """Extract outline from image using contours and dilation
    Returns both the thick band image and the outer contours for single-line export
    """
    # Convert to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Heavy dilation to merge shapes (same as original script)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    dilated = cv2.dilate(binary, kernel, iterations=4)
    
    # Get outline contours (outer edge for single-line export)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create thick outline by drawing filled contours then eroding
    # First, draw filled contours
    filled = np.zeros_like(dilated)
    cv2.drawContours(filled, contours, -1, 255, -1)  # -1 means filled
    
    # Erode to create inner boundary
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    eroded = cv2.erode(filled, erode_kernel, iterations=1)
    
    # Subtract eroded from filled to get thick outline band
    outline = cv2.subtract(filled, eroded)
    
    return outline, contours


def contours_to_svg_paths(contours, viewbox_size):
    """Convert OpenCV contours directly to SVG path strings (for single line outlines)"""
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Start path with first point
        path_data = f"M {contour[0][0][0]},{contour[0][0][1]}"
        
        # Add line segments to all other points
        for point in contour[1:]:
            path_data += f" L {point[0][0]},{point[0][1]}"
        
        # Close the path
        path_data += " Z"
        paths.append(path_data)
    
    return paths


def trace_to_svg(png_path, output_svg_path):
    """Trace bitmap to SVG using potrace"""
    potrace_exe = os.path.join(os.getcwd(), "potrace", "potrace.exe")
    
    try:
        # Convert PNG to BMP for potrace
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        # Invert so outline becomes black on white
        image = cv2.bitwise_not(image)
        
        bmp_path = png_path.replace('.png', '.bmp')
        cv2.imwrite(bmp_path, image)
        
        # Trace to SVG
        abs_bmp_path = os.path.abspath(bmp_path)
        abs_svg_path = os.path.abspath(output_svg_path)
        
        subprocess.run([
            potrace_exe,
            abs_bmp_path,
            "-s",  # SVG output
            "-o", abs_svg_path
        ], check=True, capture_output=True, text=True)
        
        # Clean up
        os.unlink(bmp_path)
        
        return output_svg_path
    except subprocess.CalledProcessError as e:
        print(f"Error tracing bitmap: {e}")
        return None


def get_svg_bbox(svg_path):
    """Get bounding box of SVG"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Try viewBox first
    viewbox = root.get('viewBox')
    if viewbox:
        parts = [float(x) for x in viewbox.split()]
        return {
            'x': parts[0],
            'y': parts[1],
            'width': parts[2],
            'height': parts[3]
        }
    
    # Fallback to width/height attributes
    width = float(root.get('width', 100).replace('px', ''))
    height = float(root.get('height', 100).replace('px', ''))
    
    return {
        'x': 0,
        'y': 0,
        'width': width,
        'height': height
    }


def align_outline_to_input(input_svg_path, outline_svg_path, output_overlay_path, output_outline_only_path, 
                          output_overlay_lightburn_path, output_outline_only_lightburn_path, 
                          outline_contours, image_size,
                          outline_scale_multiplier=1.25, lightburn_stroke_width=2):
    """
    Align traced outline SVG to match input SVG's bbox and create overlay.
    Makes outline bbox bigger than input bbox by outline_scale_multiplier.
    Creates 4 versions: filled (for viewing) and stroked single-line (for LightBurn).
    
    Args:
        outline_contours: OpenCV contours for creating single-line paths
        image_size: Size of the rasterized image (for contour scaling)
        outline_scale_multiplier: How much bigger the outline should be (1.25 = 25% bigger)
        lightburn_stroke_width: Stroke width for LightBurn-compatible SVGs
    """
    print(f"\n=== Aligning outline to input ===")
    
    # Get bounding boxes
    input_bbox = get_svg_bbox(input_svg_path)
    outline_bbox = get_svg_bbox(outline_svg_path)
    
    print(f"Original outline bbox: {outline_bbox['width']:.1f} x {outline_bbox['height']:.1f}")
    
    # Parse outline SVG
    outline_tree = ET.parse(outline_svg_path)
    outline_root = outline_tree.getroot()
    
    # Find the group/path that contains the traced paths
    # Potrace typically creates a single <g> element with transform
    outline_group = None
    for elem in outline_root:
        if elem.tag.endswith('g'):
            outline_group = elem
            break
    
    if outline_group is None:
        print("ERROR: No group found in outline SVG")
        return None, None
    
    # Calculate uniform scaling factor using max dimension
    # This ensures outline bbox is at least as large as input bbox in both dimensions
    # Then multiply by the scale multiplier to make it bigger
    scale_w = input_bbox['width'] / outline_bbox['width']
    scale_h = input_bbox['height'] / outline_bbox['height']
    scale = max(scale_w, scale_h) * outline_scale_multiplier
    
    print(f"Scale factor (uniform): {scale:.4f} (base * {outline_scale_multiplier} multiplier)")
    
    # Calculate rescaled outline dimensions
    rescaled_outline_width = outline_bbox['width'] * scale
    rescaled_outline_height = outline_bbox['height'] * scale
    
    print(f"Input bbox: {input_bbox['width']:.1f} x {input_bbox['height']:.1f}")
    print(f"Rescaled outline bbox: {rescaled_outline_width:.1f} x {rescaled_outline_height:.1f}")
    
    # Center the input bbox inside the rescaled outline bbox
    # Calculate offset to center input within the rescaled outline
    offset_x = (rescaled_outline_width - input_bbox['width']) / 2
    offset_y = (rescaled_outline_height - input_bbox['height']) / 2
    
    # Translation: move to input position, then subtract offset to center
    translate_x = input_bbox['x'] - offset_x
    translate_y = input_bbox['y'] - offset_y
    
    new_transform = f"translate({translate_x},{translate_y}) scale({scale},{scale})"
    
    # Parse existing transform from outline group
    existing_transform = outline_group.get('transform', '')
    if existing_transform:
        # Combine transforms: new transform first, then existing
        combined_transform = f"{new_transform} {existing_transform}"
    else:
        combined_transform = new_transform
    
    print(f"Applied transform: {combined_transform}")
    
    # Calculate expanded viewBox to fit both input and outline without clipping
    # The outline is bigger, so we need to expand the viewBox
    expanded_viewbox_x = input_bbox['x'] - offset_x
    expanded_viewbox_y = input_bbox['y'] - offset_y
    expanded_viewbox_width = rescaled_outline_width
    expanded_viewbox_height = rescaled_outline_height
    
    print(f"Expanded viewBox: {expanded_viewbox_x:.1f} {expanded_viewbox_y:.1f} {expanded_viewbox_width:.1f} x {expanded_viewbox_height:.1f}")
    
    # === Create outline-only SVG ===
    outline_only_root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f"{expanded_viewbox_x} {expanded_viewbox_y} {expanded_viewbox_width} {expanded_viewbox_height}",
        'width': str(expanded_viewbox_width),
        'height': str(expanded_viewbox_height)
    })
    
    # Copy outline group with new transform
    outline_only_group = ET.Element('g')
    outline_only_group.set('transform', combined_transform)
    outline_only_group.set('fill', '#000000')  # Fill the outline shape
    outline_only_group.set('stroke', 'none')
    
    # Copy all paths from original outline
    for path in outline_group.findall('.//{http://www.w3.org/2000/svg}path'):
        path_copy = ET.Element('path')
        path_copy.set('d', path.get('d', ''))
        path_copy.set('fill', '#000000')  # Fill the outline shape
        path_copy.set('stroke', 'none')
        outline_only_group.append(path_copy)
    
    outline_only_root.append(outline_only_group)
    
    # Save outline-only SVG (filled version)
    outline_only_tree = ET.ElementTree(outline_only_root)
    outline_only_tree.write(output_outline_only_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved outline-only SVG (filled): {output_outline_only_path}")
    
    # === Create outline-only SVG for LightBurn (single-line stroked version) ===
    outline_lightburn_root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f"{expanded_viewbox_x} {expanded_viewbox_y} {expanded_viewbox_width} {expanded_viewbox_height}",
        'width': str(expanded_viewbox_width),
        'height': str(expanded_viewbox_height)
    })
    
    # Convert contours to SVG paths (single lines, not filled shapes)
    contour_paths = contours_to_svg_paths(outline_contours, image_size)
    
    # Create group with transform to align contour paths
    outline_lightburn_group = ET.Element('g')
    # The contours are in image coordinates, so we need to transform them
    # from image space to the expanded viewbox space
    contour_scale = rescaled_outline_width / image_size
    contour_translate_x = expanded_viewbox_x
    contour_translate_y = expanded_viewbox_y
    lightburn_transform = f"translate({contour_translate_x},{contour_translate_y}) scale({contour_scale},{contour_scale})"
    outline_lightburn_group.set('transform', lightburn_transform)
    outline_lightburn_group.set('fill', 'none')  # No fill for LightBurn
    outline_lightburn_group.set('stroke', '#000000')  # Black stroke
    outline_lightburn_group.set('stroke-width', str(lightburn_stroke_width))
    
    # Add single-line paths from contours
    for path_data in contour_paths:
        path_elem = ET.Element('path')
        path_elem.set('d', path_data)
        path_elem.set('fill', 'none')  # No fill for LightBurn
        path_elem.set('stroke', '#000000')
        path_elem.set('stroke-width', str(lightburn_stroke_width))
        outline_lightburn_group.append(path_elem)
    
    outline_lightburn_root.append(outline_lightburn_group)
    
    # Save outline-only SVG for LightBurn
    outline_lightburn_tree = ET.ElementTree(outline_lightburn_root)
    outline_lightburn_tree.write(output_outline_only_lightburn_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved outline-only SVG (LightBurn - single line): {output_outline_only_lightburn_path}")
    
    # === Create overlay SVG (input + outline) ===
    input_tree = ET.parse(input_svg_path)
    input_root = input_tree.getroot()
    
    # Set expanded viewBox to accommodate both input and outline without clipping
    input_root.set('viewBox', f"{expanded_viewbox_x} {expanded_viewbox_y} {expanded_viewbox_width} {expanded_viewbox_height}")
    input_root.set('width', str(expanded_viewbox_width))
    input_root.set('height', str(expanded_viewbox_height))
    
    # Register and use SVG namespace
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg_ns = '{http://www.w3.org/2000/svg}'
    
    # Create a copy of the outline group for overlay with high visibility
    overlay_group = ET.Element(f'{svg_ns}g')
    overlay_group.set('transform', combined_transform)
    overlay_group.set('id', 'outline-layer')
    overlay_group.set('opacity', '1')
    
    # Copy all paths with explicit styling for maximum visibility
    path_count = 0
    for path in outline_group.findall('.//{http://www.w3.org/2000/svg}path'):
        path_copy = ET.Element(f'{svg_ns}path')
        path_copy.set('d', path.get('d', ''))
        # Use black color for outline
        path_copy.set('fill', '#000000')
        path_copy.set('fill-opacity', '0.7')
        path_copy.set('stroke', '#000000')
        path_copy.set('stroke-width', '2')
        path_copy.set('stroke-opacity', '1')
        overlay_group.append(path_copy)
        path_count += 1
    
    print(f"Added {path_count} outline paths to overlay")
    
    # Add outline to input SVG as the last element (on top)
    input_root.append(overlay_group)
    
    # Save overlay SVG (filled version)
    input_tree.write(output_overlay_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved overlay SVG (filled): {output_overlay_path}")
    
    # === Create overlay SVG for LightBurn (single-line stroked version) ===
    input_tree_lightburn = ET.parse(input_svg_path)
    input_root_lightburn = input_tree_lightburn.getroot()
    
    # Set expanded viewBox
    input_root_lightburn.set('viewBox', f"{expanded_viewbox_x} {expanded_viewbox_y} {expanded_viewbox_width} {expanded_viewbox_height}")
    input_root_lightburn.set('width', str(expanded_viewbox_width))
    input_root_lightburn.set('height', str(expanded_viewbox_height))
    
    # Create outline group with single-line stroke for LightBurn
    overlay_lightburn_group = ET.Element(f'{svg_ns}g')
    overlay_lightburn_group.set('transform', lightburn_transform)
    overlay_lightburn_group.set('id', 'outline-layer-lightburn')
    overlay_lightburn_group.set('fill', 'none')  # No fill for LightBurn
    overlay_lightburn_group.set('stroke', '#000000')  # Black stroke
    overlay_lightburn_group.set('stroke-width', str(lightburn_stroke_width))
    
    # Add single-line paths from contours
    for path_data in contour_paths:
        path_elem = ET.Element(f'{svg_ns}path')
        path_elem.set('d', path_data)
        path_elem.set('fill', 'none')  # No fill for LightBurn
        path_elem.set('stroke', '#000000')  # Black stroke
        path_elem.set('stroke-width', str(lightburn_stroke_width))
        overlay_lightburn_group.append(path_elem)
    
    # Add outline to input SVG
    input_root_lightburn.append(overlay_lightburn_group)
    
    # Save overlay SVG for LightBurn
    input_tree_lightburn.write(output_overlay_lightburn_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved overlay SVG (LightBurn - single line): {output_overlay_lightburn_path}")
    
    return output_overlay_path, output_outline_only_path, output_overlay_lightburn_path, output_outline_only_lightburn_path


def process_svg(svg_path, output_dir="output", outline_scale_multiplier=1.25):
    """Process a single SVG file
    
    Args:
        svg_path: Path to input SVG file
        output_dir: Directory to save outputs
        outline_scale_multiplier: How much bigger the outline should be (1.25 = 25% bigger)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {svg_path}")
    print(f"{'='*60}")
    
    base_name = Path(svg_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Rasterize SVG
    print("\n[1/5] Rasterizing SVG...")
    raster_image = rasterize_svg(svg_path)
    image_size = raster_image.shape[0]  # Square image, so width = height
    
    # Step 2: Extract outline and contours
    print("[2/5] Extracting outline...")
    outline_image, outline_contours = extract_outline(raster_image)
    
    # Step 3: Save outline as PNG
    print("[3/5] Saving outline image...")
    outline_png = f"{output_dir}/{base_name}_outline.png"
    cv2.imwrite(outline_png, outline_image)
    
    # Step 4: Trace outline to SVG
    print("[4/5] Tracing outline to SVG...")
    traced_outline_svg = f"{output_dir}/{base_name}_traced_outline.svg"
    trace_result = trace_to_svg(outline_png, traced_outline_svg)
    
    if not trace_result:
        print("ERROR: Failed to trace outline")
        return None
    
    # Step 5: Align and create output SVGs
    print("[5/5] Aligning and creating outputs...")
    overlay_svg = f"{output_dir}/{base_name}_overlay.svg"
    outline_only_svg = f"{output_dir}/{base_name}_outline_only.svg"
    overlay_lightburn_svg = f"{output_dir}/{base_name}_overlay_lightburn.svg"
    outline_only_lightburn_svg = f"{output_dir}/{base_name}_outline_only_lightburn.svg"
    
    result = align_outline_to_input(
        svg_path, 
        traced_outline_svg, 
        overlay_svg, 
        outline_only_svg,
        overlay_lightburn_svg,
        outline_only_lightburn_svg,
        outline_contours,
        image_size,
        outline_scale_multiplier,
        lightburn_stroke_width=2
    )
    
    # Clean up temporary files
    os.unlink(outline_png)
    os.unlink(traced_outline_svg)
    
    if result[0] and result[1] and result[2] and result[3]:
        print(f"\n[OK] Successfully processed {base_name}")
        print(f"  - Generated 4 SVG files:")
        print(f"    • {base_name}_outline_only.svg (filled)")
        print(f"    • {base_name}_overlay.svg (filled)")
        print(f"    • {base_name}_outline_only_lightburn.svg (stroked for LightBurn)")
        print(f"    • {base_name}_overlay_lightburn.svg (stroked for LightBurn)")
        return result
    else:
        print(f"\n[FAILED] Failed to process {base_name}")
        return None


def main():
    svg_folder = "svg"
    output_folder = "output"
    
    # How much bigger the outline should be compared to the input SVG (1.25 = 25% bigger)
    outline_scale_multiplier = 1.4
    
    svg_files = list(Path(svg_folder).glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {svg_folder}/")
        return
    
    print(f"Found {len(svg_files)} SVG file(s) to process")
    print(f"Outline will be {outline_scale_multiplier}x bigger than input SVG")
    
    # Process all SVG files
    success_count = 0
    for svg_file in svg_files:
        try:
            result = process_svg(str(svg_file), output_folder, outline_scale_multiplier)
            if result:
                success_count += 1
        except Exception as e:
            print(f"\n[ERROR] Error processing {svg_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(svg_files)} files processed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

