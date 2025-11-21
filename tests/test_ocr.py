import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import time

API_URL = "http://127.0.0.1:8002/predict_text"

def create_synthetic_text_image(text: str) -> np.ndarray:
    """Create a synthetic image with text for testing."""
    # Create white text on black background
    img = Image.new('L', (28 * len(text) + 20, 50), 0)
    draw = ImageDraw.Draw(img)
    
    # Use default font or try to load one
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), text, fill=255, font=font)
    
    # Convert to numpy and normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    # Explicit cast to standard float to avoid JSON serialization issues with np.float32
    return [float(x) for x in img_np.flatten()], img.width, img.height

def test_ocr():
    print("🧪 Testing OCR Endpoint...")
    
    text_to_test = "ABC"
    print(f"   Input Text: '{text_to_test}'")
    
    image_data, width, height = create_synthetic_text_image(text_to_test)
    payload = {
        "image": image_data,
        "width": width,
        "height": height
    }
    
    try:
        start_time = time.time()
        resp = requests.post(API_URL, json=payload)
        latency = (time.time() - start_time) * 1000
        
        if resp.status_code != 200:
            print(f"❌ API Error: {resp.status_code}")
            print(f"Response: {resp.text}")
            return
            
        data = resp.json()
        predicted_text = data['text']
        segments = data['segments']
        
        print(f"   Predicted: '{predicted_text}'")
        print(f"   Latency: {latency:.2f}ms")
        print(f"   Segments found: {len(segments)}")
        
        if len(segments) == len(text_to_test):
            print("✅ Segmentation count matches!")
        else:
            print(f"⚠️ Segmentation count mismatch (Expected {len(text_to_test)}, got {len(segments)})")
            
        # Note: Accuracy might be low with default font, mainly testing the pipeline
        print("✅ OCR pipeline test completed.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_ocr()
