import cv2
import numpy as np
from typing import Tuple, Dict, List
import logging
from skimage import measure, morphology, filters
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AdvancedImageProcessor:
    """Advanced image processing with multiple AI techniques for floor plan analysis"""
    
    def __init__(self, scale_factor=0.05):
        self.scale_factor = scale_factor
        self.debug_mode = False
        
    def preprocess_image(self, image_array: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Advanced preprocessing with multiple techniques"""
        logger.info(f"Starting advanced preprocessing for image: {image_array.shape}")
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        preprocessing_info = {
            "original_size": image_array.shape,
            "methods_used": []
        }
        
        # Method 1: Adaptive preprocessing
        enhanced = self._enhance_contrast(gray)
        preprocessing_info["methods_used"].append("contrast_enhancement")
        
        # Method 2: Multi-scale processing
        binary_multiscale = self._multiscale_binarization(enhanced)
        preprocessing_info["methods_used"].append("multiscale_binarization")
        
        # Method 3: Morphological operations
        cleaned = self._morphological_cleaning(binary_multiscale)
        preprocessing_info["methods_used"].append("morphological_cleaning")
        
        # Method 4: Connected component analysis for validation
        validated = self._validate_structure(cleaned)
        preprocessing_info["methods_used"].append("structure_validation")
        
        logger.info(f"Preprocessing complete using methods: {preprocessing_info['methods_used']}")
        
        return validated, preprocessing_info
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Advanced contrast enhancement using CLAHE and histogram equalization"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _multiscale_binarization(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale binarization using multiple thresholding techniques"""
        # Method 1: Otsu's thresholding
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Multi-level thresholding using scikit-image
        try:
            from skimage.filters import threshold_multiotsu
            thresholds = threshold_multiotsu(image, classes=3)
            multi_otsu = np.digitize(image, bins=thresholds)
            multi_otsu = ((multi_otsu == 0) * 255).astype(np.uint8)
        except:
            multi_otsu = otsu
        
        # Combine methods using voting
        combined = np.zeros_like(otsu)
        vote = (otsu == 0).astype(int) + (adaptive == 0).astype(int) + (multi_otsu == 255).astype(int)
        combined[vote >= 2] = 0  # Wall pixels (black)
        combined[vote < 2] = 255  # Room pixels (white)
        
        return combined
    
    def _morphological_cleaning(self, binary: np.ndarray) -> np.ndarray:
        """Advanced morphological operations for cleaning"""
        # Remove small noise
        kernel_small = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Fill gaps in walls
        kernel_medium = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove thin protrusions
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_medium)
        
        # Final cleaning with skimage
        try:
            cleaned = morphology.remove_small_objects(cleaned == 0, min_size=50)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)
            cleaned = (cleaned == 0).astype(np.uint8) * 255
        except:
            pass
        
        return cleaned
    
    def _validate_structure(self, binary: np.ndarray) -> np.ndarray:
        """Validate and fix structural issues"""
        # Ensure we have a reasonable wall-to-room ratio
        wall_pixels = np.sum(binary == 0)
        room_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        wall_ratio = wall_pixels / total_pixels
        
        # If wall ratio is too high or too low, invert
        if wall_ratio > 0.6 or wall_ratio < 0.05:
            binary = cv2.bitwise_not(binary)
            logger.info(f"Inverted binary image due to wall ratio: {wall_ratio:.3f}")
        
        return binary
    
    def detect_architectural_elements(self, binary_image: np.ndarray) -> Dict:
        """Detect architectural elements using advanced computer vision"""
        logger.info("Detecting architectural elements...")
        
        elements = {
            "walls": self._detect_walls_advanced(binary_image),
            "corners": self._detect_corners(binary_image),
            "openings": self._detect_openings(binary_image),
            "structural_lines": self._detect_structural_lines(binary_image)
        }
        
        return elements
    
    def _detect_walls_advanced(self, binary_image: np.ndarray) -> List[Dict]:
        """Advanced wall detection using multiple methods"""
        walls = []
        
        # Method 1: Hough Line Transform with multiple parameters
        edges = cv2.Canny(binary_image, 30, 100)
        
        # Standard Hough Lines
        lines_standard = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                       threshold=100, minLineLength=50, maxLineGap=10)
        
        # Probabilistic Hough Lines with different parameters
        lines_prob = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                   threshold=50, minLineLength=30, maxLineGap=5)
        
        # Combine and filter lines
        all_lines = []
        if lines_standard is not None:
            all_lines.extend(lines_standard)
        if lines_prob is not None:
            all_lines.extend(lines_prob)
        
        # Cluster similar lines
        if all_lines:
            walls = self._cluster_and_merge_lines(all_lines)
        
        # Method 2: Skeleton-based detection
        skeleton_walls = self._skeleton_based_wall_detection(binary_image)
        walls.extend(skeleton_walls)
        
        # Remove duplicates and merge nearby walls
        walls = self._merge_similar_walls(walls)
        
        logger.info(f"Detected {len(walls)} walls using advanced methods")
        return walls
    
    def _skeleton_based_wall_detection(self, binary_image: np.ndarray) -> List[Dict]:
        """Extract walls using morphological skeleton"""
        walls = []
        
        try:
            # Create skeleton of wall regions
            wall_mask = (binary_image == 0).astype(np.uint8)
            skeleton = morphology.skeletonize(wall_mask)
            
            # Find contours in skeleton
            contours, _ = cv2.findContours((skeleton * 255).astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) > 10:  # Sufficient points
                    # Fit line to contour
                    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate line endpoints
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((binary_image.shape[1] - x) * vy / vx) + y)
                    
                    if 0 <= lefty < binary_image.shape[0] and 0 <= righty < binary_image.shape[0]:
                        wall = {
                            "start": [0 * self.scale_factor, 0, lefty * self.scale_factor],
                            "end": [binary_image.shape[1] * self.scale_factor, 0, righty * self.scale_factor],
                            "height": 3.0,
                            "thickness": 0.2,
                            "confidence": len(contour) / 100.0,
                            "method": "skeleton"
                        }
                        walls.append(wall)
        
        except Exception as e:
            logger.warning(f"Skeleton-based detection failed: {e}")
        
        return walls
    
    def _detect_corners(self, binary_image: np.ndarray) -> List[Dict]:
        """Detect corners using Harris corner detection and other methods"""
        corners = []
        
        # Convert to float32
        gray_float = np.float32(binary_image)
        
        # Harris corner detection
        harris_corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        
        # Threshold for corner detection
        corner_threshold = 0.01 * harris_corners.max()
        corner_coords = np.where(harris_corners > corner_threshold)
        
        for y, x in zip(corner_coords[0], corner_coords[1]):
            corners.append({
                "position": [x * self.scale_factor, 0, y * self.scale_factor],
                "strength": float(harris_corners[y, x]),
                "type": "harris"
            })
        
        # FAST corner detection
        try:
            fast = cv2.FastFeatureDetector_create()
            keypoints = fast.detect(binary_image, None)
            
            for kp in keypoints:
                corners.append({
                    "position": [kp.pt[0] * self.scale_factor, 0, kp.pt[1] * self.scale_factor],
                    "strength": float(kp.response),
                    "type": "fast"
                })
        except:
            pass
        
        logger.info(f"Detected {len(corners)} corners")
        return corners
    
    def _detect_openings(self, binary_image: np.ndarray) -> Dict:
        """Detect doors and windows using gap analysis"""
        doors = []
        windows = []
        
        # Find gaps in walls by analyzing the boundary between wall and room regions
        wall_mask = (binary_image == 0)
        room_mask = (binary_image == 255)
        
        # Erosion and dilation to find gaps
        kernel = np.ones((3,3), np.uint8)
        eroded_walls = cv2.erode(wall_mask.astype(np.uint8), kernel)
        wall_gaps = wall_mask.astype(np.uint8) - eroded_walls
        
        # Find contours of gaps
        contours, _ = cv2.findContours(wall_gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Reasonable opening size
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify as door or window based on size and position
                aspect_ratio = max(w, h) / min(w, h)
                
                opening = {
                    "position": [(x + w/2) * self.scale_factor, 0, (y + h/2) * self.scale_factor],
                    "size": [w * self.scale_factor, h * self.scale_factor],
                    "area": area * (self.scale_factor ** 2)
                }
                
                if aspect_ratio > 2 and max(w, h) > 30:  # Likely a door
                    opening["type"] = "door"
                    doors.append(opening)
                elif 10 < min(w, h) < 50:  # Likely a window
                    opening["type"] = "window"
                    windows.append(opening)
        
        logger.info(f"Detected {len(doors)} doors and {len(windows)} windows")
        return {"doors": doors, "windows": windows}
    
    def _detect_structural_lines(self, binary_image: np.ndarray) -> List[Dict]:
        """Detect structural lines using advanced line detection"""
        structural_lines = []
        
        # Use multiple edge detection methods
        edges_canny = cv2.Canny(binary_image, 50, 150)
        edges_sobel = cv2.Sobel(binary_image, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        # Combine edge maps
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        
        # Line detection with different parameters
        lines = cv2.HoughLinesP(combined_edges, 1, np.pi/180, 
                              threshold=30, minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 15:  # Filter short lines
                    structural_lines.append({
                        "start": [x1 * self.scale_factor, 0, y1 * self.scale_factor],
                        "end": [x2 * self.scale_factor, 0, y2 * self.scale_factor],
                        "length": length * self.scale_factor,
                        "angle": np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    })
        
        return structural_lines
    
    def _cluster_and_merge_lines(self, lines: List) -> List[Dict]:
        """Cluster and merge similar lines using machine learning"""
        if not lines:
            return []
        
        # Extract features from lines
        features = []
        line_data = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Line features: midpoint, angle, length
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            features.append([mid_x, mid_y, angle * 100, length])  # Scale angle for clustering
            line_data.append([x1, y1, x2, y2, length])
        
        features = np.array(features)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=20, min_samples=1).fit(features)
        
        # Merge lines in each cluster
        merged_walls = []
        for cluster_id in set(clustering.labels_):
            cluster_lines = [line_data[i] for i in range(len(line_data)) 
                           if clustering.labels_[i] == cluster_id]
            
            if cluster_lines:
                merged_wall = self._merge_line_cluster(cluster_lines)
                merged_walls.append(merged_wall)
        
        return merged_walls
    
    def _merge_line_cluster(self, lines: List) -> Dict:
        """Merge a cluster of similar lines"""
        if len(lines) == 1:
            x1, y1, x2, y2, length = lines[0]
            return {
                "start": [x1 * self.scale_factor, 0, y1 * self.scale_factor],
                "end": [x2 * self.scale_factor, 0, y2 * self.scale_factor],
                "height": 3.0,
                "thickness": 0.2,
                "length": length * self.scale_factor
            }
        
        # Find the longest representative line
        longest_line = max(lines, key=lambda x: x[4])
        x1, y1, x2, y2, length = longest_line
        
        return {
            "start": [x1 * self.scale_factor, 0, y1 * self.scale_factor],
            "end": [x2 * self.scale_factor, 0, y2 * self.scale_factor],
            "height": 3.0,
            "thickness": 0.2,
            "length": length * self.scale_factor,
            "confidence": len(lines) / 10.0  # Confidence based on cluster size
        }
    
    def _merge_similar_walls(self, walls: List[Dict]) -> List[Dict]:
        """Enhanced wall merging with better algorithms"""
        if len(walls) <= 1:
            return walls
        
        merged = []
        used = set()
        
        for i, wall1 in enumerate(walls):
            if i in used:
                continue
            
            similar_walls = [wall1]
            used.add(i)
            
            for j, wall2 in enumerate(walls[i+1:], i+1):
                if j in used:
                    continue
                
                if self._are_walls_similar(wall1, wall2):
                    similar_walls.append(wall2)
                    used.add(j)
            
            # Merge similar walls
            if len(similar_walls) > 1:
                merged_wall = self._merge_wall_group_advanced(similar_walls)
            else:
                merged_wall = wall1
            
            merged.append(merged_wall)
        
        return merged
    
    def _are_walls_similar(self, wall1: Dict, wall2: Dict, 
                          angle_threshold: float = 15, distance_threshold: float = 1.0) -> bool:
        """Check if two walls are similar enough to merge"""
        # Calculate angles
        dx1 = wall1["end"][0] - wall1["start"][0]
        dz1 = wall1["end"][2] - wall1["start"][2]
        angle1 = np.arctan2(dz1, dx1) * 180 / np.pi
        
        dx2 = wall2["end"][0] - wall2["start"][0]
        dz2 = wall2["end"][2] - wall2["start"][2]
        angle2 = np.arctan2(dz2, dx2) * 180 / np.pi
        
        # Angle difference
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if angle_diff > angle_threshold:
            return False
        
        # Distance between walls
        points = [wall1["start"], wall1["end"], wall2["start"], wall2["end"]]
        min_distance = float('inf')
        
        for p1 in points[:2]:
            for p2 in points[2:]:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
                min_distance = min(min_distance, dist)
        
        return min_distance < distance_threshold
    
    def _merge_wall_group_advanced(self, walls: List[Dict]) -> Dict:
        """Advanced merging of wall groups"""
        # Collect all endpoints
        points = []
        for wall in walls:
            points.extend([wall["start"], wall["end"]])
        
        # Find the two points that are farthest apart
        max_dist = 0
        best_start, best_end = points[0], points[1]
        
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points[i+1:], i+1):
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_start, best_end = p1, p2
        
        # Calculate average properties
        avg_height = np.mean([wall.get("height", 3.0) for wall in walls])
        avg_thickness = np.mean([wall.get("thickness", 0.2) for wall in walls])
        total_confidence = sum([wall.get("confidence", 1.0) for wall in walls])
        
        return {
            "start": best_start,
            "end": best_end,
            "height": avg_height,
            "thickness": avg_thickness,
            "length": max_dist,
            "confidence": total_confidence / len(walls),
            "merged_from": len(walls)
        }