"""
TrOCR Predictor - Handwritten Text Recognition

Uses Microsoft's TrOCR model from HuggingFace for recognizing
complete lines of handwritten text.
"""
import torch
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.utils.logger import app_logger as logger


class TrOCRPredictor:
    """Predictor for handwritten text using TrOCR."""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        """
        Initialize TrOCR predictor.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "microsoft/trocr-base-handwritten" (default, faster)
                - "microsoft/trocr-large-handwritten" (more accurate, slower)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading TrOCR model: {model_name}")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("[OK] " + f"TrOCR loaded successfully on {self.device}")
    
    def predict(self, image: Image.Image) -> str:
        """
        Recognize text from an image.
        
        Args:
            image: PIL Image containing handwritten text
            
        Returns:
            Recognized text string
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image
        pixel_values = self.processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    def predict_with_confidence(
        self, 
        image: Image.Image
    ) -> Tuple[str, float]:
        """
        Recognize text with confidence score.
        
        Args:
            image: PIL Image containing handwritten text
            
        Returns:
            (text, confidence) tuple
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image
        pixel_values = self.processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode text
        generated_text = self.processor.batch_decode(
            outputs.sequences, 
            skip_special_tokens=True
        )[0]
        
        # Calculate confidence (average probability)
        if outputs.scores:
            probs = []
            for score in outputs.scores:
                prob = torch.softmax(score, dim=-1)
                max_prob = prob.max(dim=-1).values
                probs.append(max_prob.item())
            confidence = sum(probs) / len(probs) if probs else 0.0
        else:
            confidence = 0.0
        
        return generated_text, confidence
    
    def predict_from_array(self, img_array: np.ndarray) -> str:
        """
        Recognize text from numpy array.
        
        Args:
            img_array: Image as numpy array (H, W, C) or (H, W)
            
        Returns:
            Recognized text string
        """
        # Convert to PIL
        if len(img_array.shape) == 2:
            image = Image.fromarray(img_array.astype(np.uint8), mode='L')
        else:
            image = Image.fromarray(img_array.astype(np.uint8))
        
        return self.predict(image)


# Singleton instance
_trocr_instance: Optional[TrOCRPredictor] = None


def get_trocr_predictor() -> TrOCRPredictor:
    """Get or create TrOCR predictor singleton."""
    global _trocr_instance
    if _trocr_instance is None:
        _trocr_instance = TrOCRPredictor()
    return _trocr_instance


if __name__ == "__main__":
    # Test
    predictor = TrOCRPredictor()
    
    # Create test image with text
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', (400, 50), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello World", fill='black')
    
    result = predictor.predict(img)
    print(f"Recognized: {result}")
