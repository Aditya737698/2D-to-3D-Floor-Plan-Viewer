"""
Wall detection component using multiple algorithms
"""
import cv2
import numpy as np
import logging
from typing import List, Dict
from sklearn.cluster import DBSCAN
from skimage import morphology

logger = logging.getLogger(__name__)

class WallDetector:
    def __init__(self, scale_factor: float = 0.05, wall_thickness: float = 0.15, 
                 wall_height: float = 2.8, min_wall_length: float = 0.3):
        self.scale_factor = scale_factor
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.min_wall_length = min_wall_length
    
    def detect_walls(self, preprocessed: Dict[str, np.ndarray]) -> List[Dict]:
        """Detect walls using multiple approaches"""
        logger.info("Starting wall detection...")
        
        try:
            binary = preprocessed['cleaned']
            height, width = binary.shape
            
            # Multiple wall detection approaches
            walls_hough = self._detect_walls_hough(binary)
            walls_contour = self._detect_walls_contour(binary)
            walls_skeleton = self._detect_walls_skeleton(binary)
            
            # Combine and filter results
            all_walls = walls_hough + walls_contour + walls_skeleton
            
            # Advanced filtering and merging
            filtered_walls = self._filter_and_merge_walls(all_walls)
            
            # Add architectural constraints
            constrained_walls = self._apply_architectural_constraints(filtered_walls, width, height)
            
            logger.info(f"Wall detection complete: {len(constrained_walls)} walls")
            return constrained_walls
            
        except Exception as e:
            logger.error(f"Error in wall detection: {str(e)}")
            return self._generate_fallback_walls(width, height)
    
    def _detect_walls_hough(self, binary: np.ndarray) -> List[Dict]:
        """Enhanced Hough line detection"""
        walls = []
        height, width = binary.shape
        
        try:
            # Multi-scale edge detection
            edges_30_100 = cv2.Canny(binary, 30, 100)
            edges_50_150 = cv2.Canny(binary, 50, 150)
            edges_combined = cv2.bitwise_or(edges_30_100, edges_50_150)
            
            # Multiple parameter sets for different wall types
            param_sets = [
                {'threshold': int(min(width, height) * 0.15), 'minLineLength': int(min(width, height) * 0.08), 'maxLineGap': int(min(width, height) * 0.02)},
                {'threshold': int(min(width, height) * 0.08), 'minLineLength': int(min(width, height) * 0.04), 'maxLineGap': int(min(width, height) * 0.03)},
                {'threshold': int(min(width, height) * 0.05), 'minLineLength': int(min(width, height) * 0.02), 'maxLineGap': int(min(width, height) * 0.015)}
            ]
            
            all_lines = []
            for params in param_sets:
                lines = cv2.HoughLinesP(
                    edges_combined,
                    rho=1,
                    theta=np.pi/180,
                    threshold=params['threshold'],
                    minLineLength=params['minLineLength'],
                    maxLineGap=params['maxLineGap']
                )
                
                if lines is not None:
                    all_lines.extend(lines)
            
            # Process lines
            for line in all_lines:
                x1, y1, x2, y2 = line[0]
                
                # Convert to 3D coordinates
                start = [x1 * self.scale_factor, 0, y1 * self.scale_factor]
                end = [x2 * self.scale_factor, 0, y2 * self.scale_factor]
                
                # Calculate length
                length = np.sqrt((end[0] - start[0])**2 + (end[2] - start[2])**2)
                
                if length >= self.min_wall_length:
                    walls.append({
                        "start": start,
                        "end": end,
                        "height": self.wall_height,
                        "thickness": self.wall_thickness,
                        "length": length,
                        "type": "hough"
                    })
            
            logger.debug(f"Hough detection found {len(walls)} walls")
            return walls
            
        except Exception as e:
            logger.error(f"Error in Hough wall detection: {str(e)}")
            return []
    
    def _detect_walls_contour(self, binary: np.ndarray) -> List[Dict]:
        """Detect walls using contour analysis"""
        walls = []
        
        try:
            # Find contours of wall regions
            wall_regions = 255 - binary  # Invert so walls are white
            contours, _ = cv2.findContours(wall_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Skip very small contours
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Extract line segments from polygon
                for i in range(len(approx)):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % len(approx)][0]
                    
                    # Convert to 3D coordinates
                    start = [p1[0] * self.scale_factor, 0, p1[1] * self.scale_factor]
                    end = [p2[0] * self.scale_factor, 0, p2[1] * self.scale_factor]
                    
                    # Calculate length
                    length = np.sqrt((end[0] - start[0])**2 + (end[2] - start[2])**2)
                    
                    if length >= self.min_wall_length:
                        walls.append({
                            "start": start,
                            "end": end,
                            "height": self.wall_height,
                            "thickness": self.wall_thickness,
                            "length": length,
                            "type": "contour"
                        })
            
            logger.debug(f"Contour detection found {len(walls)} walls")
            return walls
            
        except Exception as e:
            logger.error(f"Error in contour wall detection: {str(e)}")
            return []
    
    def _detect_walls_skeleton(self, binary: np.ndarray) -> List[Dict]:
        """Detect walls using morphological skeleton"""
        walls = []
        
        try:
            # Create wall mask
            wall_mask = 255 - binary
            
            # Morphological operations to clean up
            kernel = np.ones((5,5), np.uint8)
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)
            
            # Create skeleton
            skeleton = morphology.skeletonize(wall_mask > 0)
            skeleton = skeleton.astype(np.uint8) * 255
            
            # Find line segments in skeleton
            lines = cv2.HoughLinesP(
                skeleton,
                rho=1,
                theta=np.pi/180,
                threshold=20,
                minLineLength=30,
                maxLineGap=10
            )
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Convert to 3D coordinates
                    start = [x1 * self.scale_factor, 0, y1 * self.scale_factor]
                    end = [x2 * self.scale_factor, 0, y2 * self.scale_factor]
                    
                    # Calculate length
                    length = np.sqrt((end[0] - start[0])**2 + (end[2] - start[2])**2)
                    
                    if length >= self.min_wall_length:
                        walls.append({
                            "start": start,
                            "end": end,
                            "height": self.wall_height,
                            "thickness": self.wall_thickness,
                            "length": length,
                            "type": "skeleton"
                        })
            
            logger.debug(f"Skeleton detection found {len(walls)} walls")
            return walls
            
        except Exception as e:
            logger.error(f"Error in skeleton wall detection: {str(e)}")
            return []
    
    def _filter_and_merge_walls(self, walls: List[Dict]) -> List[Dict]:
        """Filter and merge similar walls using clustering"""
        if not walls:
            return []
        
        try:
            # Convert walls to feature vectors for clustering
            features = []
            for wall in walls:
                dx = wall['end'][0] - wall['start'][0]
                dz = wall['end'][2] - wall['start'][2]
                angle = np.arctan2(dz, dx)
                features.append([
                    wall['start'][0], wall['start'][2],
                    wall['end'][0], wall['end'][2],
                    angle, wall['length']
                ])
            
            features = np.array(features)
            
            # Normalize features
            features_norm = features.copy()
            if features_norm.shape[0] > 0:
                features_norm[:, :4] /= np.max(features_norm[:, :4]) if np.max(features_norm[:, :4]) > 0 else 1
                features_norm[:, 4] /= np.pi
                features_norm[:, 5] /= np.max(features_norm[:, 5]) if np.max(features_norm[:, 5]) > 0 else 1
            
            # Cluster similar walls
            clustering = DBSCAN(eps=0.1, min_samples=1).fit(features_norm)
            labels = clustering.labels_
            
            # Merge walls in each cluster
            merged_walls = []
            for label in np.unique(labels):
                if label == -1:  # Noise points
                    continue
                
                cluster_walls = [walls[i] for i in range(len(walls)) if labels[i] == label]
                merged_wall = self._merge_wall_cluster(cluster_walls)
                if merged_wall:
                    merged_walls.append(merged_wall)
            
            # Filter out very short or redundant walls
            filtered_walls = []
            for wall in merged_walls:
                if wall['length'] >= self.min_wall_length:
                    is_redundant = False
                    for existing_wall in filtered_walls:
                        if self._walls_are_similar(wall, existing_wall):
                            is_redundant = True
                            break
                    
                    if not is_redundant:
                        filtered_walls.append(wall)
            
            logger.debug(f"Filtered and merged: {len(walls)} -> {len(filtered_walls)} walls")
            return filtered_walls
            
        except Exception as e:
            logger.error(f"Error in wall filtering and merging: {str(e)}")
            return walls[:20]  # Return first 20 walls as fallback
    
    def _merge_wall_cluster(self, walls: List[Dict]) -> Dict:
        """Merge a cluster of similar walls"""
        if not walls:
            return None
        
        if len(walls) == 1:
            return walls[0]
        
        try:
            # Find the overall bounding line for all walls in cluster
            all_points = []
            for wall in walls:
                all_points.append([wall['start'][0], wall['start'][2]])
                all_points.append([wall['end'][0], wall['end'][2]])
            
            all_points = np.array(all_points)
            
            # Fit a line through all points using least squares
            if len(all_points) >= 2:
                # Calculate the direction vector
                centroid = np.mean(all_points, axis=0)
                centered_points = all_points - centroid
                
                # SVD to find principal direction
                _, _, V = np.linalg.svd(centered_points.T)
                direction = V[0]
                
                # Project all points onto the line
                projections = []
                for point in all_points:
                    point_centered = point - centroid
                    projection_length = np.dot(point_centered, direction)
                    projections.append(projection_length)
                
                # Find the extreme projections
                min_proj = min(projections)
                max_proj = max(projections)
                
                # Calculate the start and end points
                start_point = centroid + min_proj * direction
                end_point = centroid + max_proj * direction
                
                # Convert back to 3D coordinates
                start = [start_point[0], 0, start_point[1]]
                end = [end_point[0], 0, end_point[1]]
                
                length = np.sqrt((end[0] - start[0])**2 + (end[2] - start[2])**2)
                
                return {
                    "start": start,
                    "end": end,
                    "height": self.wall_height,
                    "thickness": self.wall_thickness,
                    "length": length,
                    "type": "merged"
                }
            
        except Exception as e:
            logger.error(f"Error merging wall cluster: {str(e)}")
        
        # Fallback: return the longest wall in the cluster
        return max(walls, key=lambda w: w['length'])
    
    def _walls_are_similar(self, wall1: Dict, wall2: Dict, distance_threshold: float = 0.5) -> bool:
        """Check if two walls are too similar"""
        try:
            points1 = [wall1['start'], wall1['end']]
            points2 = [wall2['start'], wall2['end']]
            
            min_distance = float('inf')
            for p1 in points1:
                for p2 in points2:
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
                    min_distance = min(min_distance, dist)
            
            return min_distance < distance_threshold
            
        except Exception as e:
            logger.error(f"Error checking wall similarity: {str(e)}")
            return False
    
    def _apply_architectural_constraints(self, walls: List[Dict], width: int, height: int) -> List[Dict]:
        """Apply architectural constraints to ensure realistic wall layouts"""
        try:
            constrained_walls = []
            
            # Ensure we have perimeter walls
            w_meters = width * self.scale_factor
            h_meters = height * self.scale_factor
            
            # Check for perimeter walls and add if missing
            perimeter_walls = [
                {"start": [0, 0, 0], "end": [w_meters, 0, 0], "type": "bottom"},
                {"start": [w_meters, 0, 0], "end": [w_meters, 0, h_meters], "type": "right"},
                {"start": [w_meters, 0, h_meters], "end": [0, 0, h_meters], "type": "top"},
                {"start": [0, 0, h_meters], "end": [0, 0, 0], "type": "left"}
            ]
            
            for perim_wall in perimeter_walls:
                # Check if we have a wall close to this perimeter
                has_close_wall = any(
                    self._is_wall_close_to_perimeter(wall, perim_wall, threshold=1.0) 
                    for wall in walls
                )
                
                # Add perimeter wall if missing
                if not has_close_wall:
                    perim_wall.update({
                        "height": self.wall_height,
                        "thickness": self.wall_thickness,
                        "length": np.sqrt((perim_wall["end"][0] - perim_wall["start"][0])**2 + 
                                        (perim_wall["end"][2] - perim_wall["start"][2])**2)
                    })
                    constrained_walls.append(perim_wall)
            
            # Add existing walls
            constrained_walls.extend(walls)
            
            return constrained_walls
            
        except Exception as e:
            logger.error(f"Error applying architectural constraints: {str(e)}")
            return walls
    
    def _is_wall_close_to_perimeter(self, wall: Dict, perimeter_wall: Dict, threshold: float) -> bool:
        """Check if a wall is close to a perimeter wall"""
        try:
            w1_start = np.array([wall['start'][0], wall['start'][2]])
            w1_end = np.array([wall['end'][0], wall['end'][2]])
            w2_start = np.array([perimeter_wall['start'][0], perimeter_wall['start'][2]])
            w2_end = np.array([perimeter_wall['end'][0], perimeter_wall['end'][2]])
            
            distances = [
                np.linalg.norm(w1_start - w2_start),
                np.linalg.norm(w1_start - w2_end),
                np.linalg.norm(w1_end - w2_start),
                np.linalg.norm(w1_end - w2_end)
            ]
            
            return min(distances) < threshold
            
        except Exception as e:
            logger.error(f"Error checking wall proximity to perimeter: {str(e)}")
            return False
    
    def _generate_fallback_walls(self, width: int, height: int) -> List[Dict]:
        """Generate basic perimeter walls as fallback"""
        logger.info("Generating fallback perimeter walls")
        
        w_meters = width * self.scale_factor
        h_meters = height * self.scale_factor
        
        return [
            {
                "start": [0, 0, 0],
                "end": [w_meters, 0, 0],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": w_meters,
                "type": "perimeter"
            },
            {
                "start": [w_meters, 0, 0],
                "end": [w_meters, 0, h_meters],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": h_meters,
                "type": "perimeter"
            },
            {
                "start": [w_meters, 0, h_meters],
                "end": [0, 0, h_meters],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": w_meters,
                "type": "perimeter"
            },
            {
                "start": [0, 0, h_meters],
                "end": [0, 0, 0],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": h_meters,
                "type": "perimeter"
            }
        ]