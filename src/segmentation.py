import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image

def segment_characters(image_array: np.ndarray, shape: Tuple[int, int] = None) -> List[np.ndarray]:
    """
    Segment characters from a 2D or flattened image array.
    
    Args:
        image_array: Normalized image array (0-1).
        shape: Optional (height, width) tuple for reshaping flattened arrays.
    """
    # 1. Prepare image
    if image_array.ndim == 1:
        if shape:
            image_array = image_array.reshape(shape)
        else:
            # Assume square if no shape provided
            side = int(np.sqrt(image_array.size))
            if side * side != image_array.size:
                 raise ValueError(f"Image is not square (size={image_array.size}) and no shape provided.")
            image_array = image_array.reshape(side, side)
    
    # Convert to 8-bit for OpenCV (0-255)
    img_uint8 = (image_array * 255).astype(np.uint8)
    
    # 2. Find Contours
    # RETR_EXTERNAL: only outer contours
    # CHAIN_APPROX_SIMPLE: compress segments
    contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # 3. Filter and Sort Contours
    # Filter small noise
    min_area = 10
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return []
    
    # Sort left-to-right (based on bounding box x)
    bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
    # zip contours with boxes, sort by x, unzip
    sorted_pairs = sorted(zip(valid_contours, bounding_boxes), key=lambda b: b[1][0])
    
    segmented_chars = []
    
    for contour, (x, y, w, h) in sorted_pairs:
        # 4. Crop
        # Add small padding
        pad = 2
        y_min = max(0, y - pad)
        y_max = min(img_uint8.shape[0], y + h + pad)
        x_min = max(0, x - pad)
        x_max = min(img_uint8.shape[1], x + w + pad)
        
        crop = img_uint8[y_min:y_max, x_min:x_max]
        
        # 5. Preprocess (Resize to 20x20, center in 28x28) - Same as training
        # Convert to PIL for high-quality resize
        pil_crop = Image.fromarray(crop)
        
        # Aspect Ratio Resize
        target_size = 20
        width, height = pil_crop.size
        
        if height > width:
            new_height = target_size
            new_width = max(1, int(width * target_size / height))
        else:
            new_width = target_size
            new_height = max(1, int(height * target_size / width))
            
        pil_resized = pil_crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Paste into 28x28 black canvas
        final_img = Image.new('L', (28, 28), 0)
        offset_x = (28 - new_width) // 2
        offset_y = (28 - new_height) // 2
        final_img.paste(pil_resized, (offset_x, offset_y))
        
        # Normalize
        final_array = np.array(final_img).astype(np.float32) / 255.0
        segmented_chars.append(final_array.flatten())
        
    return segmented_chars
