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
    
    # Heavy dilation with gaussian blur and thresholding in loop
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    dilated = binary
    for i in range(4):
        dilated = cv2.dilate(dilated, kernel, iterations=1)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        dilated = cv2.GaussianBlur(dilated, (19, 19), 0)
        _, dilated = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
    
    # One final dilation after the loop
    dilated = cv2.dilate(dilated, kernel, iterations=1)
    
    # Get outline contours (outer edge for single-line export)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filled outline
    filled = np.zeros_like(dilated)
    cv2.drawContours(filled, contours, -1, 255, -1)  # -1 means filled
    
    # Additional dilation step for centerline tracing
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_dilated = cv2.dilate(filled, dilate_kernel, iterations=1)
    
    return final_dilated, contours


def smooth_contours(contours, epsilon_factor=0.0002):
    """Smooth contours using polygon approximation - balanced point count for curvy smooth paths"""
    smoothed = []
    for contour in contours:
        # Calculate epsilon based on contour perimeter
        # 0.0009 gives a touch more definition with curvy smooth paths
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        # Approximate polygon to reduce points
        approx = cv2.approxPolyDP(contour, epsilon, True)
        smoothed.append(approx)
    return smoothed


def calculate_angle_at_point(p0, p1, p2):
    """Calculate the angle at point p1 formed by p0-p1-p2
    Returns angle in degrees (0-180, where 180 is straight, 0 is sharp)
    """
    # Vectors from p1 to p0 and p1 to p2
    v1 = np.array([p0[0] - p1[0], p0[1] - p1[1]])
    v2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 180.0  # No angle if vectors are zero
    
    # Normalize vectors
    v1_norm = v1 / mag1
    v2_norm = v2 / mag2
    
    # Calculate angle using dot product
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def detect_corners(points, angle_threshold=140):
    """Detect sharp corners in a sequence of points
    Returns array of booleans indicating which points are corners
    angle_threshold: angles below this value (in degrees) are considered corners
                     (180 = straight line, 0 = sharp corner)
    """
    if len(points) < 3:
        return [False] * len(points)
    
    is_corner = [False] * len(points)
    
    for i in range(len(points)):
        # Get three consecutive points (with wrapping for closed curves)
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        
        # Calculate angle at this point
        angle = calculate_angle_at_point(p0, p1, p2)
        
        # Mark as corner if angle is sharp
        is_corner[i] = angle < angle_threshold
    
    return is_corner


def fit_bezier_to_points(points, angle_threshold=140, corner_tension_reduction=0.6, base_tension=0.6):
    """Fit cubic Bezier curves to a sequence of points with corner detection
    Returns SVG path data with smooth, curvy curves and smoothed corners
    
    Args:
        points: Sequence of 2D points
        angle_threshold: Angles below this (in degrees) are considered corners (default 140)
        corner_tension_reduction: Multiply tension by this factor at corners to smooth them (default 0.6)
        base_tension: Base tension for Bezier curves - higher = more curved, lower = straighter (default 0.6)
    """
    if len(points) < 2:
        return ""
    
    # Start with move command
    path_data = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
    
    if len(points) == 2:
        # Just a line
        path_data += f" L {points[1][0]:.2f},{points[1][1]:.2f}"
        return path_data
    
    # Detect corners in the point sequence
    corners = detect_corners(points, angle_threshold)
    
    # For smooth curves, use Catmull-Rom to cubic Bezier conversion
    # This creates a smooth curve through all points
    # Higher tension = more curved, lower = more straight
    
    for i in range(len(points)):
        if i == 0:
            continue
            
        # Get four points for Catmull-Rom (with wrapping for closed curves)
        p0 = points[i - 1] if i > 0 else points[-2]
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % len(points)] if i < len(points) - 1 else points[1]
        
        # Adjust tension based on corner detection
        # Reduce tension at corners to make them smoother
        tension = base_tension
        if corners[i - 1] or corners[i]:
            # This segment has a corner, reduce tension to smooth it
            tension *= corner_tension_reduction
        
        # Convert Catmull-Rom to cubic Bezier control points
        cp1_x = p1[0] + (p2[0] - p0[0]) / 6.0 * tension
        cp1_y = p1[1] + (p2[1] - p0[1]) / 6.0 * tension
        cp2_x = p2[0] - (p3[0] - p1[0]) / 6.0 * tension
        cp2_y = p2[1] - (p3[1] - p1[1]) / 6.0 * tension
        
        # Add cubic Bezier curve command
        path_data += f" C {cp1_x:.2f},{cp1_y:.2f} {cp2_x:.2f},{cp2_y:.2f} {p2[0]:.2f},{p2[1]:.2f}"
    
    return path_data


def contours_to_smooth_svg_paths(contours, viewbox_size, epsilon_factor=0.0002, 
                                 angle_threshold=140, corner_tension_reduction=0.6, base_tension=0.6):
    """Convert OpenCV contours to smooth SVG path strings using Bezier curves with corner smoothing
    
    Args:
        epsilon_factor: Controls point reduction (lower = more points, smoother)
        angle_threshold: Angles below this (in degrees) are considered corners (default 140)
        corner_tension_reduction: Reduces bezier tension at corners for smoothing (default 0.6)
        base_tension: Base tension for Bezier curves - higher = more curved, lower = straighter (default 0.6)
    """
    paths = []
    
    # First smooth the contours (keep many points for smooth curves)
    smoothed_contours = smooth_contours(contours, epsilon_factor)
    
    for contour in smoothed_contours:
        if len(contour) < 3:
            continue
        
        # Extract points from contour
        points = [point[0] for point in contour]
        
        # Fit Bezier curves to create smooth path with corner detection
        path_data = fit_bezier_to_points(points, angle_threshold, corner_tension_reduction, base_tension)
        
        # Close the path
        path_data += " Z"
        paths.append(path_data)
    
    return paths


def contours_to_svg_paths(contours, viewbox_size):
    """Convert OpenCV contours directly to SVG path strings (for single line outlines)
    This is a fallback that creates straight lines - use contours_to_smooth_svg_paths for smooth curves
    """
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


def contours_to_svg_direct(contours, image_size, output_svg_path, epsilon_factor=0.0002,
                          angle_threshold=140, corner_tension_reduction=0, base_tension=0.6):
    """Create SVG directly from contours with smooth Bezier curves and corner smoothing
    
    Args:
        epsilon_factor: Controls point reduction (lower = more points)
        angle_threshold: Angles below this (in degrees) are considered corners (default 140)
        corner_tension_reduction: Reduces bezier tension at corners for smoothing (default 0.6)
        base_tension: Base tension for Bezier curves - higher = more curved, lower = straighter (default 0.6)
    """
    
    # Convert contours to smooth SVG paths with corner detection
    smooth_paths = contours_to_smooth_svg_paths(contours, image_size, epsilon_factor, 
                                                angle_threshold, corner_tension_reduction, base_tension)
    
    # Create SVG structure
    svg_root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f"0 0 {image_size} {image_size}",
        'width': str(image_size),
        'height': str(image_size)
    })
    
    # Add each path
    for path_data in smooth_paths:
        path_elem = ET.Element('path', {
            'd': path_data,
            'fill': '#000000',
            'stroke': 'none'
        })
        svg_root.append(path_elem)
    
    # Write SVG file
    tree = ET.ElementTree(svg_root)
    tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)
    
    return output_svg_path


def get_svg_bbox(svg_path):
    """Get bounding box of SVG, including original units if present"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get original width and height attributes with units
    width_str = root.get('width', '100')
    height_str = root.get('height', '100')
    
    # Extract units (e.g., 'mm', 'px', 'pt', etc.)
    width_unit_match = re.search(r'[a-zA-Z%]+$', width_str)
    height_unit_match = re.search(r'[a-zA-Z%]+$', height_str)
    width_unit = width_unit_match.group(0) if width_unit_match else ''
    height_unit = height_unit_match.group(0) if height_unit_match else ''
    
    # Try viewBox first
    viewbox = root.get('viewBox')
    if viewbox:
        parts = [float(x) for x in viewbox.split()]
        return {
            'x': parts[0],
            'y': parts[1],
            'width': parts[2],
            'height': parts[3],
            'width_unit': width_unit,
            'height_unit': height_unit
        }
    
    # Fallback to width/height attributes
    # Strip units and convert to float
    width = float(re.sub(r'[a-zA-Z%]+$', '', width_str))
    height = float(re.sub(r'[a-zA-Z%]+$', '', height_str))
    
    return {
        'x': 0,
        'y': 0,
        'width': width,
        'height': height,
        'width_unit': width_unit,
        'height_unit': height_unit
    }


def align_outline_to_input(input_svg_path, outline_svg_path, output_overlay_path, output_outline_only_path, 
                          output_overlay_lightburn_path, output_outline_only_lightburn_path, 
                          outline_contours, image_size,
                          outline_scale_multiplier=1.25, lightburn_stroke_width=2,
                          angle_threshold=140, corner_tension_reduction=0.6, epsilon_factor=0.0002, base_tension=0.6):
    """
    Align traced outline SVG to match input SVG's bbox and create overlay.
    Makes outline bbox bigger than input bbox by outline_scale_multiplier.
    Creates 4 versions: filled (for viewing) and stroked single-line (for LightBurn).
    
    Args:
        outline_contours: OpenCV contours for creating single-line paths
        image_size: Size of the rasterized image (for contour scaling)
        outline_scale_multiplier: How much bigger the outline should be (1.25 = 25% bigger)
        lightburn_stroke_width: Stroke width for LightBurn-compatible SVGs
        angle_threshold: Angles below this (in degrees) are considered corners (default 140)
        corner_tension_reduction: Reduces bezier tension at corners for smoothing (default 0.6)
        epsilon_factor: Controls point reduction/smoothing (lower = more points, smoother) (default 0.0002)
        base_tension: Base tension for Bezier curves - higher = more curved, lower = straighter (default 0.6)
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
    # Our direct SVG creation puts paths directly under root, no transform needed
    outline_group = None
    existing_transform = ''
    for elem in outline_root:
        if elem.tag.endswith('g'):
            outline_group = elem
            existing_transform = elem.get('transform', '')
            break
    
    # If no group found, create a virtual group with all path elements
    if outline_group is None:
        outline_group = outline_root  # Use root directly as it contains paths
        existing_transform = ''
    
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
    
    # Combine with existing transform if any
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
    
    # Use expanded viewBox dimensions to show both input and larger outline without clipping
    # Normalize to start at (0, 0) to prevent coordinate system issues
    output_viewbox_x = 0
    output_viewbox_y = 0
    output_viewbox_width = expanded_viewbox_width
    output_viewbox_height = expanded_viewbox_height
    
    # Adjust transforms to account for the normalized viewBox
    # We need to shift content from expanded viewBox's origin to (0, 0)
    viewbox_offset_x = -expanded_viewbox_x
    viewbox_offset_y = -expanded_viewbox_y
    
    print(f"Output viewBox (normalized): {output_viewbox_x:.1f} {output_viewbox_y:.1f} {output_viewbox_width:.1f} x {output_viewbox_height:.1f}")
    print(f"ViewBox offset: ({viewbox_offset_x:.1f}, {viewbox_offset_y:.1f})")
    
    # === Create outline-only SVG ===
    # Preserve original units from input SVG
    width_with_unit = f"{output_viewbox_width}{input_bbox.get('width_unit', '')}"
    height_with_unit = f"{output_viewbox_height}{input_bbox.get('height_unit', '')}"
    
    outline_only_root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f"{output_viewbox_x} {output_viewbox_y} {output_viewbox_width} {output_viewbox_height}",
        'width': width_with_unit,
        'height': height_with_unit
    })
    
    # Copy outline group with new transform (adjusted for normalized viewBox)
    outline_only_group = ET.Element('g')
    # Apply viewBox offset first, then the outline transform
    adjusted_transform = f"translate({viewbox_offset_x},{viewbox_offset_y}) {combined_transform}"
    outline_only_group.set('transform', adjusted_transform)
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
        'viewBox': f"{output_viewbox_x} {output_viewbox_y} {output_viewbox_width} {output_viewbox_height}",
        'width': width_with_unit,
        'height': height_with_unit
    })
    
    # Convert contours to smooth SVG paths with Bezier curves and corner smoothing
    contour_paths = contours_to_smooth_svg_paths(outline_contours, image_size, epsilon_factor=epsilon_factor,
                                                 angle_threshold=angle_threshold, 
                                                 corner_tension_reduction=corner_tension_reduction,
                                                 base_tension=base_tension)
    
    # Create group with transform to align contour paths
    outline_lightburn_group = ET.Element('g')
    # The contours are in image coordinates, so we need to transform them
    # First scale them to match the rescaled outline size, then translate to center
    contour_scale = rescaled_outline_width / image_size
    # Center the rescaled outline within the normalized output bbox (starting at 0,0)
    contour_translate_x = (output_viewbox_width - rescaled_outline_width) / 2
    contour_translate_y = (output_viewbox_height - rescaled_outline_height) / 2
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
    
    # Set normalized viewBox to maintain consistent size (preserve units)
    input_root.set('viewBox', f"{output_viewbox_x} {output_viewbox_y} {output_viewbox_width} {output_viewbox_height}")
    input_root.set('width', width_with_unit)
    input_root.set('height', height_with_unit)
    
    # Register and use SVG namespace
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg_ns = '{http://www.w3.org/2000/svg}'
    
    # Wrap all existing content in a group with viewBox offset to normalize coordinates
    content_wrapper = ET.Element(f'{svg_ns}g')
    content_wrapper.set('transform', f"translate({viewbox_offset_x},{viewbox_offset_y})")
    content_wrapper.set('id', 'input-content')
    
    # Move all existing children into the wrapper
    for child in list(input_root):
        input_root.remove(child)
        content_wrapper.append(child)
    
    # Add the wrapper back to root
    input_root.append(content_wrapper)
    
    # Create a copy of the outline group for overlay with high visibility
    overlay_group = ET.Element(f'{svg_ns}g')
    overlay_group.set('transform', adjusted_transform)
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
    
    # Set normalized viewBox to maintain consistent size (preserve units)
    input_root_lightburn.set('viewBox', f"{output_viewbox_x} {output_viewbox_y} {output_viewbox_width} {output_viewbox_height}")
    input_root_lightburn.set('width', width_with_unit)
    input_root_lightburn.set('height', height_with_unit)
    
    # Wrap all existing content in a group with viewBox offset to normalize coordinates
    content_wrapper_lb = ET.Element(f'{svg_ns}g')
    content_wrapper_lb.set('transform', f"translate({viewbox_offset_x},{viewbox_offset_y})")
    content_wrapper_lb.set('id', 'input-content')
    
    # Move all existing children into the wrapper
    for child in list(input_root_lightburn):
        input_root_lightburn.remove(child)
        content_wrapper_lb.append(child)
    
    # Add the wrapper back to root
    input_root_lightburn.append(content_wrapper_lb)
    
    # Create outline group with single-line stroke for LightBurn
    overlay_lightburn_group = ET.Element(f'{svg_ns}g')
    overlay_lightburn_group.set('transform', lightburn_transform)
    overlay_lightburn_group.set('id', 'outline-layer-lightburn')
    overlay_lightburn_group.set('fill', 'none')  # No fill for LightBurn
    overlay_lightburn_group.set('stroke', '#000000')  # Black stroke
    overlay_lightburn_group.set('stroke-width', str(lightburn_stroke_width))
    
    # Add smooth Bezier curve paths from contours
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


def process_svg(svg_path, output_dir="output", outline_scale_multiplier=1.25,
                angle_threshold=140, corner_tension_reduction=0.6, epsilon_factor=0.0002, base_tension=0.6):
    """Process a single SVG file
    
    Args:
        svg_path: Path to input SVG file
        output_dir: Directory to save outputs
        outline_scale_multiplier: How much bigger the outline should be (1.25 = 25% bigger)
        angle_threshold: Angles below this (in degrees) are considered corners (default 140)
        corner_tension_reduction: Reduces bezier tension at corners for smoothing (default 0.6)
        epsilon_factor: Controls point reduction/smoothing (lower = more points, smoother) (default 0.0002)
        base_tension: Base tension for Bezier curves - higher = more curved, lower = straighter (default 0.6)
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
    
    # Step 4: Create smooth SVG directly from contours (no bitmap tracing!)
    print("[4/5] Creating smooth SVG from contours with Bezier curves and corner smoothing...")
    traced_outline_svg = f"{output_dir}/{base_name}_traced_outline.svg"
    trace_result = contours_to_svg_direct(outline_contours, image_size, traced_outline_svg, 
                                         epsilon_factor=epsilon_factor, angle_threshold=angle_threshold,
                                         corner_tension_reduction=corner_tension_reduction, base_tension=base_tension)
    
    if not trace_result:
        print("ERROR: Failed to create outline SVG")
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
        lightburn_stroke_width=2,
        angle_threshold=angle_threshold,
        corner_tension_reduction=corner_tension_reduction,
        epsilon_factor=epsilon_factor,
        base_tension=base_tension
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
    
    # ===== CONFIGURATION PARAMETERS =====
    
    # How much bigger the outline should be compared to the input SVG (1.0 = same size)
    outline_scale_multiplier = 1.4
    
    # epsilon_factor: Controls point reduction/smoothing in contour approximation
    # Lower values = more points = smoother curves but larger file size
    # Higher values = fewer points = simpler paths but less detail
    # Typical range: 0.0001-0.001
    epsilon_factor = 0.00018  # Default 0.0002 gives smooth, detailed curves
    
    # base_tension: Controls how curved the Bezier curves are
    # Higher values = more curved/flowing paths, Lower values = straighter paths
    # Typical range: 0.3-3.0
    base_tension = 1  # Default 0.6, you have it at 2.0 for very curvy paths
    
    # Corner detection and smoothing parameters
    # angle_threshold: Angles below this (in degrees) are considered sharp corners
    # Lower values = only very sharp corners smoothed, Higher values = more corners smoothed
    # 180 = straight line, 90 = right angle
    angle_threshold = 160  # Default 140, higher = smoother
    
    # corner_tension_reduction: How much to reduce bezier tension at corners (0.0-1.0)
    # Lower values = smoother corners, Higher values = sharper corners
    # 0.0 = maximum smoothing, 1.0 = no smoothing
    corner_tension_reduction = 0  # Default 0.6, you have it at 0 for max smoothing
    
    # ===================================
    
    svg_files = list(Path(svg_folder).glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {svg_folder}/")
        return
    
    print(f"Found {len(svg_files)} SVG file(s) to process")
    print(f"Configuration:")
    print(f"  - Outline scale: {outline_scale_multiplier}x bigger than input SVG")
    print(f"  - Epsilon factor: {epsilon_factor} (point reduction/smoothing)")
    print(f"  - Base tension: {base_tension} (curve intensity)")
    print(f"  - Corner smoothing: angle_threshold={angle_threshold}°, tension_reduction={corner_tension_reduction}")
    
    # Process all SVG files
    success_count = 0
    for svg_file in svg_files:
        try:
            result = process_svg(str(svg_file), output_folder, outline_scale_multiplier,
                               angle_threshold, corner_tension_reduction, epsilon_factor, base_tension)
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

