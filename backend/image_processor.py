"""
Image preprocessing component for floor plan analysis
"""
import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    def preprocess_image(self, image_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Advanced image preprocessing with multiple techniques"""
        logger.info(f"Preprocessing image: {image_array.shape}")
        
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.copy()
            
            # Store original
            original = gray.copy()
            
            # Enhanced contrast with CLAHE
            enhanced = self.clahe.apply(gray)
            
            # Noise reduction with multiple filters
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            denoised = cv2.medianBlur(denoised, 5)
            
            # Multiple thresholding techniques
            thresh_otsu = self._apply_otsu_threshold(denoised)
            thresh_adaptive_mean = self._apply_adaptive_threshold(denoised, cv2.ADAPTIVE_THRESH_MEAN_C)
            thresh_adaptive_gaussian = self._apply_adaptive_threshold(denoised, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            
            # Choose best threshold
            candidates = {
                'otsu': thresh_otsu,
                'adaptive_mean': thresh_adaptive_mean,
                'adaptive_gaussian': thresh_adaptive_gaussian
            }
            
            best_threshold = self._select_best_threshold(candidates, original)
            
            # Morphological operations for cleaning
            cleaned = self._clean_binary_image(best_threshold)
            
            # Ensure walls are black and spaces are white
            if np.sum(cleaned == 0) > np.sum(cleaned == 255):
                cleaned = cv2.bitwise_not(cleaned)
            
            results = {
                'original': original,
                'enhanced': enhanced,
                'denoised': denoised,
                'binary': best_threshold,
                'cleaned': cleaned
            }
            
            logger.info("Image preprocessing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            # Fallback
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            return {'cleaned': binary, 'original': gray, 'binary': binary}
    
    def _apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu thresholding"""
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _apply_adaptive_threshold(self, image: np.ndarray, method: int) -> np.ndarray:
        """Apply adaptive thresholding"""
        return cv2.adaptiveThreshold(image, 255, method, cv2.THRESH_BINARY, 15, 10)
    
    def _select_best_threshold(self, candidates: Dict[str, np.ndarray], original: np.ndarray) -> np.ndarray:
        """Select the best thresholding result based on various metrics"""
        try:
            best_score = 0
            best_threshold = None
            
            for name, thresh in candidates.items():
                score = 0
                
                # Check black pixel ratio (walls should be 5-30% of image)
                black_ratio = np.sum(thresh == 0) / thresh.size
                if 0.05 <= black_ratio <= 0.30:
                    score += 3
                elif 0.03 <= black_ratio <= 0.40:
                    score += 2
                else:
                    score += 1
                
                # Check edge preservation
                edges_original = cv2.Canny(original, 50, 150)
                edges_thresh = cv2.Canny(thresh, 50, 150)
                edge_similarity = np.sum(edges_original & edges_thresh) / np.sum(edges_original) if np.sum(edges_original) > 0 else 0
                score += edge_similarity * 2
                
                # Check connectivity (walls should be well connected)
                num_labels, _ = cv2.connectedComponents(255 - thresh)
                if num_labels < 20:  # Not too fragmented
                    score += 1
                
                logger.debug(f"Threshold {name}: score={score:.2f}, black_ratio={black_ratio:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            
            return best_threshold if best_threshold is not None else candidates['otsu']
            
        except Exception as e:
            logger.error(f"Error selecting best threshold: {str(e)}")
            return candidates['otsu']
    
    def _clean_binary_image(self, binary: np.ndarray) -> np.ndarray:
        """Clean binary image using morphological operations"""
        kernel_small = np.ones((3,3), np.uint8)
        kernel_medium = np.ones((5,5), np.uint8)
        
        # Close small gaps
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
        # Final cleanup
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        return cleaned