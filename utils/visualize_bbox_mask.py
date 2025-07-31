from PIL import Image, ImageDraw
import numpy as np

def overlay_bbox_masks(image, bbox_mask_list, colors=None, fill_alpha=128, outline_only=False, outline_width=3):
    """
    Overlay bbox masks on image
    
    Args:
        image: PIL Image object
        bbox_mask_list: 2D list, each 1D list contains four coordinates [x1, y1, x2, y2]
        colors: color list, if None then auto-generate
        fill_alpha: fill transparency (0-255), 0 for completely transparent, 255 for completely opaque
        outline_only: whether to draw only outline, no fill
        outline_width: outline width
    
    Returns:
        PIL Image: image with overlaid bbox
    """
    # If no colors provided, auto-generate distinct colors
    if colors is None:
        colors = generate_distinct_colors(len(bbox_mask_list))
    
    # Ensure color count matches
    if len(colors) < len(bbox_mask_list):
        # If not enough colors, cycle through them
        colors = colors * (len(bbox_mask_list) // len(colors) + 1)
    colors = colors[:len(bbox_mask_list)]
    
    # Create image copy
    bboxed_image = image.copy()
    
    if outline_only:
        # Draw only outline
        draw = ImageDraw.Draw(bboxed_image)
        for i, bbox in enumerate(bbox_mask_list):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)
    else:
        # Semi-transparent fill + outline
        for i, bbox in enumerate(bbox_mask_list):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            
            # Create semi-transparent overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Semi-transparent fill
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(*color, fill_alpha))
            
            # Overlay onto original image
            bboxed_image = Image.alpha_composite(bboxed_image.convert('RGBA'), overlay)
        
        # Draw outline
        draw = ImageDraw.Draw(bboxed_image)
        for i, bbox in enumerate(bbox_mask_list):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)
    
    return bboxed_image

def generate_distinct_colors(n):
    """
    Generate n distinct colors
    
    Args:
        n: number of colors
    
    Returns:
        list: list of RGB color tuples
    """
    if n <= 8:
        # Use predefined base colors
        base_colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
        ]
        return base_colors[:n]
    else:
        # Use HSV color space to generate more colors
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

# Usage example
def example_usage():
    """Usage example"""
    # Create test image
    image = Image.new('RGB', (400, 300), (200, 200, 200))
    
    # Test bbox list
    bbox_mask_list = [
        [50, 50, 150, 150],   # First bbox
        [100, 100, 200, 200], # Second bbox (overlaps with first)
        [250, 50, 350, 150],  # Third bbox
    ]
    
    # Custom colors (optional)
    custom_colors = [
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (0, 255, 0),    # Green
    ]
    
    # 1. Semi-transparent fill + outline
    result1 = overlay_bbox_masks(
        image, 
        bbox_mask_list, 
        colors=custom_colors,
        fill_alpha=128,  # 50% transparency
        outline_only=False
    )
    
    # 2. Draw only outline
    result2 = overlay_bbox_masks(
        image, 
        bbox_mask_list, 
        colors=custom_colors,
        outline_only=True,
        outline_width=3
    )
    
    # 3. Use auto-generated colors
    result3 = overlay_bbox_masks(
        image, 
        bbox_mask_list, 
        colors=None,  # Auto-generate colors
        fill_alpha=100,  # 40% transparency
        outline_only=False
    )
    
    return [result1, result2, result3]

# Use in your code
def process_bbox_masks(image, processed_inputs):
    """
    Process bbox masks and return list of overlaid images
    """
    bboxed_image_list = []
    
    # Assume processed_inputs["bbox_masks_list"] format is:
    # [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    
    # Method 1: Semi-transparent fill
    bboxed_image = overlay_bbox_masks(
        image, 
        processed_inputs["bbox_masks_list"],
        fill_alpha=128,  # Can adjust transparency
        outline_only=False
    )
    bboxed_image_list.append(bboxed_image)
    
    # Method 2: Draw only outline
    bboxed_image_outline = overlay_bbox_masks(
        image, 
        processed_inputs["bbox_masks_list"],
        outline_only=True,
        outline_width=3
    )
    bboxed_image_list.append(bboxed_image_outline)
    
    return bboxed_image_list

# Advanced version with special handling for overlapping regions
def overlay_bbox_masks_advanced(image, bbox_mask_list, colors=None, fill_alpha=128, 
                               outline_only=False, outline_width=3, 
                               highlight_overlap=True):
    """
    Advanced bbox overlay with color blending for overlapping regions
    Color blending in overlapping regions, transparency also adjusted accordingly
    """
    if colors is None:
        colors = generate_distinct_colors(len(bbox_mask_list))
    
    if len(colors) < len(bbox_mask_list):
        colors = colors * (len(bbox_mask_list) // len(colors) + 1)
    colors = colors[:len(bbox_mask_list)]
    
    bboxed_image = image.copy()
    
    if highlight_overlap and not outline_only:
        # Create overlap count map and color accumulator
        overlap_map = np.zeros(image.size[::-1])  # height, width
        color_accumulator = np.zeros((image.size[1], image.size[0], 3))  # height, width, RGB
        
        # First pass: calculate overlapping regions and accumulate colors
        for i, bbox in enumerate(bbox_mask_list):
            x1, y1, x2, y2 = bbox
            color = np.array(colors[i])
            
            # Update overlap count and color accumulation
            overlap_map[y1:y2, x1:x2] += 1
            color_accumulator[y1:y2, x1:x2] += color
        
        # Second pass: draw blended colors
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                overlap_count = overlap_map[y, x]
                if overlap_count > 0:
                    # Use subtractive color blending (paint mixing)
                    mixed_color = blend_colors_subtractive(color_accumulator[y, x], overlap_count)
                    
                    # Adjust transparency: more overlap, less transparent
                    adjusted_alpha = adjust_alpha_for_overlap(fill_alpha, overlap_count)
                    
                    overlay.putpixel((x, y), (*mixed_color, adjusted_alpha))
        
        # Overlay onto original image
        bboxed_image = Image.alpha_composite(bboxed_image.convert('RGBA'), overlay)
        
        # Draw outline
        draw = ImageDraw.Draw(bboxed_image)
        for i, bbox in enumerate(bbox_mask_list):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)
            
    else:
        # Use standard method
        bboxed_image = overlay_bbox_masks(
            image, bbox_mask_list, colors, fill_alpha, outline_only, outline_width
        )
    
    return bboxed_image

def adjust_alpha_for_overlap(base_alpha, overlap_count):
    """
    Adjust transparency based on overlap count
    More overlap, less transparent, but change should not be too dramatic
    """
    if overlap_count == 1:
        return base_alpha
    
    # Use logarithmic function to smooth transparency changes
    # Avoid too dramatic transparency changes
    alpha_increase = min(50, int(base_alpha * 0.3 * (overlap_count - 1) / overlap_count))
    
    # Ensure transparency does not exceed 255
    new_alpha = min(255, base_alpha + alpha_increase)
    
    return int(new_alpha)

def blend_colors_subtractive(color_sum, count):
    """
    Use subtractive color blending (paint mixing) to simulate natural color mixing effects
    """
    if count == 1:
        return np.clip(color_sum, 0, 255).astype(int)
    
    # Calculate average color
    avg_color = color_sum / count
    avg_color = np.clip(avg_color, 0, 255)
    
    # Convert to 0-1 range
    r, g, b = avg_color / 255.0
    
    # Subtractive blending: simulate light absorption when mixing paints
    # When mixing, each paint absorbs light reflected by other paints
    absorption_factor = 0.25 * (count - 1) / count  # Moderate absorption
    
    # Apply absorption effect
    r = r * (1 - absorption_factor)
    g = g * (1 - absorption_factor) 
    b = b * (1 - absorption_factor)
    
    # Ensure minimum value, avoid complete black
    min_brightness = 0.4  # Maintain 40% minimum brightness
    r = max(r, min_brightness * 0.5)
    g = max(g, min_brightness * 0.5)
    b = max(b, min_brightness * 0.5)
    
    # Ensure values are within valid range
    r, g, b = np.clip([r, g, b], 0, 1)
    
    # Convert to 0-255 range integers
    result = (np.array([r, g, b]) * 255).astype(int)
    return result