import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import re

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



def parse_svg_bbox(svg_path):
    """Extract bounding box from SVG"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get viewBox
    viewbox = root.get('viewBox')
    if viewbox:
        vb_parts = [float(x) for x in viewbox.split()]
        print(f"ViewBox: {vb_parts}")
        return {'viewbox': vb_parts, 'width': vb_parts[2], 'height': vb_parts[3]}
    
    # Fallback to width/height
    width = float(root.get('width', 100))
    height = float(root.get('height', 100))
    print(f"Using width/height: {width} x {height}")
    return {'viewbox': [0, 0, width, height], 'width': width, 'height': height}

def parse_transform(transform_str):
    """Parse SVG transform string"""
    if not transform_str:
        return {'translate': [0, 0], 'scale': [1, 1]}
    
    translate = [0, 0]
    scale = [1, 1]
    
    print(f"Parsing transform: {transform_str}")
    
    # Extract translate
    translate_match = re.search(r'translate\(([^)]+)\)', transform_str)
    if translate_match:
        parts = translate_match.group(1).replace(',', ' ').split()
        translate = [float(parts[0]), float(parts[1]) if len(parts) > 1 else 0]
    
    # Extract scale
    scale_match = re.search(r'scale\(([^)]+)\)', transform_str)
    if scale_match:
        parts = scale_match.group(1).replace(',', ' ').split()
        if len(parts) == 1:
            scale = [float(parts[0]), float(parts[0])]
        else:
            scale = [float(parts[0]), float(parts[1])]
    
    return {'translate': translate, 'scale': scale}

def get_svg_bbox_with_transforms(svg_path):
    """Get actual bounding box considering transforms"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    svg_info = parse_svg_bbox(svg_path)
    
    # Find first group with transform
    for elem in root.iter():
        if elem.get('transform'):
            transform = parse_transform(elem.get('transform'))
            print(f"Transform found: {transform}")
            
            # Find path coordinates - look in current element and children
            paths = []
            if elem.tag.endswith('path'):
                paths.append(elem)
            else:
                paths = elem.findall('.//{http://www.w3.org/2000/svg}path')
            
            if paths:
                all_coords = []
                for path in paths:  # Use ALL paths for accurate bbox
                    path_d = path.get('d')
                    coords = extract_path_coords(path_d)
                    if coords:
                        all_coords.extend(coords)
                
                if all_coords:
                    bbox = calculate_transformed_bbox(all_coords, transform, svg_info['viewbox'])
                    print(f"Calculated bbox: {bbox}")
                    return bbox
    
    print("No transform found, using viewbox")
    return svg_info

def extract_path_coords(path_d):
    """Extract coordinates from SVG path - get all coordinates for proper bbox"""
    if not path_d:
        return None
    
    coords = []
    
    # Extract ALL numbers from path - this captures all coordinate data
    numbers = re.findall(r'-?\d+(?:\.\d+)?', path_d)
    
    # Group numbers into x,y pairs
    for i in range(0, len(numbers)-1, 2):
        try:
            x, y = float(numbers[i]), float(numbers[i+1])
            coords.append([x, y])
        except (ValueError, IndexError):
            continue
    
    # Limit to reasonable number for performance
    if len(coords) > 200:
        coords = coords[:200]
    
    print(f"Extracted {len(coords)} coordinate pairs from path")
    if coords:
        print(f"Sample coords: {coords[:3]}...{coords[-3:] if len(coords) > 3 else ''}")
    
    return coords if coords else None

def calculate_transformed_bbox(coords, transform, viewbox):
    """Calculate bbox after applying transforms"""
    tx, ty = transform['translate']
    sx, sy = transform['scale']
    
    print(f"Transform: translate({tx}, {ty}) scale({sx}, {sy})")
    
    # Check for problematic transforms
    if abs(sx) < 0.001 or abs(sy) < 0.001:
        print(f"WARNING: Very small scale factors detected!")
    if abs(tx) > 10000 or abs(ty) > 10000:
        print(f"WARNING: Very large translation values detected!")
    
    transformed_coords = []
    for x, y in coords:
        # Apply transform: scale first, then translate
        new_x = x * sx + tx
        new_y = y * sy + ty
        transformed_coords.append([new_x, new_y])
    
    if not transformed_coords:
        return {'width': viewbox[2], 'height': viewbox[3]}
    
    xs = [coord[0] for coord in transformed_coords]
    ys = [coord[1] for coord in transformed_coords]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    print(f"Original coord range: X({min(x for x,y in coords):.0f} to {max(x for x,y in coords):.0f}), Y({min(y for x,y in coords):.0f} to {max(y for x,y in coords):.0f})")
    print(f"Transformed coord range: X({min_x:.2f} to {max_x:.2f}), Y({min_y:.2f} to {max_y:.2f})")
    print(f"Calculated size: {width:.2f} x {height:.2f}")
    
    # Check if coordinates are reasonable
    if min_x < -10000 or max_x > 10000 or min_y < -10000 or max_y > 10000:
        print(f"WARNING: Coordinates are outside reasonable range!")
    
    # If width or height is too small, use viewbox as fallback
    if width < 1 or height < 1:
        print(f"Size too small, using viewbox as fallback")
        return {'width': viewbox[2], 'height': viewbox[3]}
    
    return {'width': width, 'height': height, 'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

def validate_rescaled_svg(svg_path, expected_bbox):
    """Validate that rescaled SVG has correct bbox and is within viewBox"""
    print(f"\n=== Validating {svg_path} ===")
    
    # Get actual bbox of rescaled SVG
    actual_bbox = get_svg_bbox_with_transforms(svg_path)
    
    print(f"Expected size: {expected_bbox['width']} x {expected_bbox['height']}")
    print(f"Actual size: {actual_bbox['width']:.2f} x {actual_bbox['height']:.2f}")
    
    # Check if bbox is within viewBox
    if 'min_x' in actual_bbox and 'min_y' in actual_bbox:
        vb = expected_bbox['viewbox']
        within_x = vb[0] <= actual_bbox['min_x'] and actual_bbox['max_x'] <= vb[0] + vb[2]
        within_y = vb[1] <= actual_bbox['min_y'] and actual_bbox['max_y'] <= vb[1] + vb[3]
        
        print(f"ViewBox: {vb}")
        print(f"Content bounds: X({actual_bbox['min_x']:.1f} to {actual_bbox['max_x']:.1f}), Y({actual_bbox['min_y']:.1f} to {actual_bbox['max_y']:.1f})")
        print(f"Within viewBox: X={within_x}, Y={within_y}")
        
        if not within_x or not within_y:
            print(f"WARNING: Content is outside viewBox!")
            return False
    
    # Check size accuracy
    size_error_x = abs(actual_bbox['width'] - expected_bbox['width']) / expected_bbox['width']
    size_error_y = abs(actual_bbox['height'] - expected_bbox['height']) / expected_bbox['height']
    
    print(f"Size error: X={size_error_x:.1%}, Y={size_error_y:.1%}")
    
    if size_error_x > 0.1 or size_error_y > 0.1:  # More than 10% error
        print(f"WARNING: Size mismatch > 10%!")
        return False
    
    print(f"Validation passed")
    return True

def rescale_outline_svg(input_svg_path, outline_svg_path, rescaled_original_svg_path, output_svg_path):
    """Rescale outline SVG using exact same transform as rescaled original"""
    print(f"\n=== Rescaling outline {outline_svg_path} ===")
    
    # Get exact transform from rescaled original
    orig_tree = ET.parse(rescaled_original_svg_path)
    orig_root = orig_tree.getroot()
    
    exact_transform = None
    exact_viewbox = orig_root.get('viewBox')
    
    for elem in orig_root.iter():
        if elem.get('transform'):
            exact_transform = elem.get('transform')
            print(f"Using exact transform from original: {exact_transform}")
            break
    
    if not exact_transform:
        print("No transform found in rescaled original")
        return None
    
    # Apply to outline
    tree = ET.parse(outline_svg_path)
    root = tree.getroot()
    
    # Expand viewBox for outline to prevent cutting
    current_vb = exact_viewbox.split()
    expanded_width = float(current_vb[2]) + 100
    expanded_height = float(current_vb[3]) + 50
    expanded_viewbox = f"0 0 {expanded_width} {expanded_height}"
    
    root.set('viewBox', expanded_viewbox)
    root.set('width', f"{expanded_width}")
    root.set('height', f"{expanded_height}")
    print(f"Expanded outline viewBox to: {expanded_viewbox}")
    
    # Apply exact same transform
    for elem in root.iter():
        if elem.get('transform'):
            elem.set('transform', exact_transform)
            print(f"Applied exact transform to outline: {exact_transform}")
            break
    
    # Save result
    tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved rescaled outline SVG: {output_svg_path}")
    return output_svg_path

def combine_svgs(original_svg_path, outline_svg_path, output_svg_path):
    """Combine original and outline SVGs into single overlay SVG"""
    print(f"\n=== Combining {original_svg_path} and {outline_svg_path} ===")
    
    # Load original SVG as base
    tree = ET.parse(original_svg_path)
    root = tree.getroot()
    
    # Expand viewBox to prevent cutting - add 100px margin
    current_vb = root.get('viewBox').split()
    expanded_width = float(current_vb[2]) + 100
    expanded_height = float(current_vb[3]) + 50
    root.set('viewBox', f"0 0 {expanded_width} {expanded_height}")
    root.set('width', f"{expanded_width}")
    root.set('height', f"{expanded_height}")
    print(f"Expanded viewBox to: 0 0 {expanded_width} {expanded_height}")
    
    # Load outline SVG to get its paths
    outline_tree = ET.parse(outline_svg_path)
    outline_root = outline_tree.getroot()
    
    # Find the group with transform in outline SVG
    outline_group = None
    for elem in outline_root.iter():
        if elem.get('transform'):
            outline_group = elem
            break
    
    if outline_group is not None:
        # Create new group for outline paths with darker stroke
        outline_copy = ET.Element('{http://www.w3.org/2000/svg}g')
        outline_copy.set('transform', outline_group.get('transform'))
        outline_copy.set('fill', 'none')
        outline_copy.set('stroke', '#000000')
        outline_copy.set('stroke-width', '3')
        outline_copy.set('stroke-opacity', '1.0')
        
        # Copy all paths from outline group
        path_count = 0
        for path in outline_group.findall('.//{http://www.w3.org/2000/svg}path'):
            path_copy = ET.Element('{http://www.w3.org/2000/svg}path')
            path_copy.set('d', path.get('d'))
            path_copy.set('fill', 'none')
            path_copy.set('stroke', '#000000')
            path_copy.set('stroke-width', '3')
            path_copy.set('stroke-opacity', '1.0')
            outline_copy.append(path_copy)
            path_count += 1
        
        # Add outline group to combined SVG
        root.append(outline_copy)
        print(f"Added {path_count} outline paths with black stroke")
    else:
        print("No outline group found")
    
    # Save combined SVG
    tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved combined SVG: {output_svg_path}")
    return output_svg_path

def rescale_traced_svg(input_svg_path, traced_svg_path, output_svg_path):
    """Rescale traced SVG to match input SVG size"""
    print(f"\n=== Rescaling {traced_svg_path} ===")
    
    # Get bounding boxes
    input_bbox = parse_svg_bbox(input_svg_path)
    traced_bbox = get_svg_bbox_with_transforms(traced_svg_path)
    
    print(f"Input SVG size: {input_bbox['width']} x {input_bbox['height']}")
    print(f"Traced SVG size: {traced_bbox['width']:.2f} x {traced_bbox['height']:.2f}")
    
    # Handle zero division
    if traced_bbox['width'] == 0 or traced_bbox['height'] == 0:
        print("ERROR: Traced SVG has zero width or height - cannot rescale")
        return None
    
    # Calculate scale ratio - use minimum to ensure content fits
    scale_x = input_bbox['width'] / traced_bbox['width']
    scale_y = input_bbox['height'] / traced_bbox['height']
    scale_ratio = min(scale_x, scale_y) * 0.9  # 90% to ensure it fits
    
    print(f"Scale ratios: X={scale_x:.6f}, Y={scale_y:.6f}")
    print(f"Using conservative scale: {scale_ratio:.6f}")
    
    # Calculate centroid in original coordinate space
    if 'min_x' in traced_bbox:
        centroid_x = (traced_bbox['min_x'] + traced_bbox['max_x']) / 2
        centroid_y = (traced_bbox['min_y'] + traced_bbox['max_y']) / 2
    else:
        centroid_x = traced_bbox['width'] / 2
        centroid_y = traced_bbox['height'] / 2
    
    print(f"Original centroid: ({centroid_x:.2f}, {centroid_y:.2f})")
    
    # Calculate target center
    target_center_x = input_bbox['width'] / 2
    target_center_y = input_bbox['height'] / 2
    print(f"Target center: ({target_center_x:.2f}, {target_center_y:.2f})")
    
    # Apply transformation
    tree = ET.parse(traced_svg_path)
    root = tree.getroot()
    
    # Update viewBox to match input
    root.set('viewBox', f"0 0 {input_bbox['width']} {input_bbox['height']}")
    root.set('width', f"{input_bbox['width']}")
    root.set('height', f"{input_bbox['height']}")
    
    # FIXED: Apply correct transformation that keeps content in viewBox
    transform_applied = False
    for elem in root.iter():
        if elem.get('transform'):
            # Get original bbox without any transforms - use ALL paths for accurate bbox
            original_paths = elem.findall('.//{http://www.w3.org/2000/svg}path')
            if original_paths:
                all_original_coords = []
                for path in original_paths:  # Use ALL paths, not just first 5
                    path_d = path.get('d')
                    coords = extract_path_coords(path_d)
                    if coords:
                        all_original_coords.extend(coords)
                
                if all_original_coords:
                    # Calculate bbox in original coordinate space (before any transforms)
                    orig_xs = [x for x, y in all_original_coords]
                    orig_ys = [y for x, y in all_original_coords]
                    orig_min_x, orig_max_x = min(orig_xs), max(orig_xs)
                    orig_min_y, orig_max_y = min(orig_ys), max(orig_ys)
                    orig_width = orig_max_x - orig_min_x
                    orig_height = orig_max_y - orig_min_y
                    orig_center_x = (orig_min_x + orig_max_x) / 2
                    orig_center_y = (orig_min_y + orig_max_y) / 2
                    
                    print(f"Original coords center: ({orig_center_x:.1f}, {orig_center_y:.1f})")
                    print(f"Original coords size: {orig_width:.1f} x {orig_height:.1f}")
                    print(f"Original coord range: X({orig_min_x:.0f} to {orig_max_x:.0f}), Y({orig_min_y:.0f} to {orig_max_y:.0f})")
                    
                    # Calculate final scale to fit content in viewBox with margin
                    scale_x = input_bbox['width'] / orig_width
                    scale_y = input_bbox['height'] / orig_height
                    final_scale = min(scale_x, scale_y) * 0.85  # 85% margin
                    print(f"Final scale: {final_scale:.6f}")
                    
                    # FIX INVERSION: Check if original transform had negative Y scale
                    original_transform = parse_transform(elem.get('transform'))
                    has_negative_y = original_transform['scale'][1] < 0
                    
                    # FIXED: Ensure content stays within viewBox bounds
                    if has_negative_y:
                        print(f"Detected Y-axis inversion, applying correction")
                        # For Y-inversion, adjust centering to keep content in bounds
                        adjusted_center_y = target_center_y + 150  # Shift down more to avoid negative Y
                        new_transform = f"translate({target_center_x},{adjusted_center_y}) scale({final_scale},{-final_scale}) translate({-orig_center_x},{-orig_center_y})"
                    else:
                        new_transform = f"translate({target_center_x},{target_center_y}) scale({final_scale}) translate({-orig_center_x},{-orig_center_y})"
                    
                    elem.set('transform', new_transform)
                    print(f"Applied corrected transform: {new_transform}")
                    transform_applied = True
                    break
    
    if not transform_applied:
        print(f"WARNING: No transform element found!")
        return None
    
    # Save result
    tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)
    print(f"Saved rescaled SVG: {output_svg_path}")
    
    # Validate the result
    is_valid = validate_rescaled_svg(output_svg_path, input_bbox)
    
    return output_svg_path

def trace_bitmap_to_svg(png_path, output_svg_path):
    """Use potrace to trace bitmap and create vector SVG"""
    potrace_exe = "potrace"
    
    try:
        # Convert PNG to BMP for potrace
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        
        # For outline-only images, invert so outline becomes black on white background
        if "outline_only" in png_path:
            image = cv2.bitwise_not(image)
        
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
    print(f"\n{'='*50}")
    print(f"Processing: {svg_path}")
    print(f"{'='*50}")
    
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
    
    # Save outline only as PNG (white outline on black background for proper tracing)
    outline_only_png = f"{output_dir}/{base_name}_outline_only.png"
    cv2.imwrite(outline_only_png, outline)
    
    # Save original image only as PNG
    original_only_png = f"{output_dir}/{base_name}_original_only.png"
    cv2.imwrite(original_only_png, image)
    
    # Overlay outline on original image (keep black outline)
    overlay_image = image.copy()
    cv2.drawContours(overlay_image, final_contours, -1, (0, 0, 0), 5)  # Black outline
    overlay_png_path = f"{output_dir}/{base_name}_overlay.png"
    cv2.imwrite(overlay_png_path, overlay_image)
    
    # Trace all three versions to SVG
    traced_overlay_svg = f"{output_dir}/{base_name}_traced_overlay.svg"
    traced_outline_svg = f"{output_dir}/{base_name}_traced_outline_only.svg"
    traced_original_svg = f"{output_dir}/{base_name}_traced_original_only.svg"
    
    print(f"\nTracing bitmaps to SVG...")
    trace_bitmap_to_svg(overlay_png_path, traced_overlay_svg)
    trace_bitmap_to_svg(outline_only_png, traced_outline_svg)
    trace_bitmap_to_svg(original_only_png, traced_original_svg)
    
    # NEW: Rescale traced original to match input size
    rescaled_original_svg = f"{output_dir}/{base_name}_rescaled_original.svg"
    result = rescale_traced_svg(svg_path, traced_original_svg, rescaled_original_svg)
    
    if result:
        print(f"\nSuccessfully created rescaled original SVG: {result}")
        
        # NEW: Rescale outline using exact same transform
        rescaled_outline_svg = f"{output_dir}/{base_name}_rescaled_outline.svg"
        outline_result = rescale_outline_svg(svg_path, traced_outline_svg, rescaled_original_svg, rescaled_outline_svg)
        
        if outline_result:
            print(f"Successfully created rescaled outline SVG: {outline_result}")
            
            # NEW: Combine both rescaled SVGs into overlay
            rescaled_overlay_svg = f"{output_dir}/{base_name}_rescaled_overlay.svg"
            overlay_result = combine_svgs(rescaled_original_svg, rescaled_outline_svg, rescaled_overlay_svg)
            
            if overlay_result:
                print(f"Successfully created rescaled overlay SVG: {overlay_result}")
        
        # Validation of original rescaled SVG
        try:
            tree = ET.parse(result)
            root = tree.getroot()
            path_count = len(root.findall('.//{http://www.w3.org/2000/svg}path'))
            print(f"Rescaled original SVG contains {path_count} path elements")
        except Exception as e:
            print(f"Error validating rescaled SVG: {e}")
    else:
        print(f"Failed to create rescaled SVG")
    
    # Clean up temporary PNGs
    os.unlink(overlay_png_path)
    os.unlink(outline_only_png)
    os.unlink(original_only_png)
    
    return outline, final_contours

def main():
    svg_folder = "svg"
    output_folder = "output"
    
    svg_files = list(Path(svg_folder).glob("*.svg"))
    
    # Process all SVG files
    for svg_file in svg_files:
        try:
            outline, contours = process_svg_file(str(svg_file), output_folder)
            print(f"\n[OK] Processed {svg_file.name}")
        except Exception as e:
            print(f"\n[ERROR] Error processing {svg_file.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()