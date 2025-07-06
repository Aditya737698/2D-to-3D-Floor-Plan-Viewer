import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.cluster import KMeans, DBSCAN
from scipy import ndimage
from skimage import measure, segmentation, feature
import networkx as nx

logger = logging.getLogger(__name__)

class AdvancedRoomAnalyzer:
    """Advanced room analysis using AI and graph theory"""
    
    def __init__(self, scale_factor=0.05):
        self.scale_factor = scale_factor
        self.room_templates = self._load_room_templates()
        self.min_room_area = 2.0  # square meters
        
    def _load_room_templates(self) -> Dict:
        """Load room templates with enhanced features"""
        return {
            'bathroom': {
                'min_area': 2, 'max_area': 12, 
                'aspect_ratio': (1.0, 3.0),
                'typical_furniture': ['toilet', 'sink', 'shower', 'bathtub'],
                'wall_density': 'high',
                'shape_preference': 'rectangular'
            },
            'bedroom': {
                'min_area': 8, 'max_area': 40, 
                'aspect_ratio': (1.0, 2.2),
                'typical_furniture': ['bed', 'nightstand', 'wardrobe', 'desk'],
                'wall_density': 'medium',
                'shape_preference': 'rectangular'
            },
            'kitchen': {
                'min_area': 6, 'max_area': 25, 
                'aspect_ratio': (1.0, 3.5),
                'typical_furniture': ['counter', 'refrigerator', 'stove', 'sink'],
                'wall_density': 'high',
                'shape_preference': 'L-shaped_or_rectangular'
            },
            'living_room': {
                'min_area': 12, 'max_area': 60, 
                'aspect_ratio': (1.0, 2.8),
                'typical_furniture': ['sofa', 'tv', 'coffee_table', 'chairs'],
                'wall_density': 'low',
                'shape_preference': 'open'
            },
            'dining_room': {
                'min_area': 8, 'max_area': 30, 
                'aspect_ratio': (1.0, 2.0),
                'typical_furniture': ['dining_table', 'chairs'],
                'wall_density': 'medium',
                'shape_preference': 'rectangular'
            },
            'corridor': {
                'min_area': 1, 'max_area': 15, 
                'aspect_ratio': (3.0, 12.0),
                'typical_furniture': [],
                'wall_density': 'very_high',
                'shape_preference': 'linear'
            },
            'office': {
                'min_area': 6, 'max_area': 25, 
                'aspect_ratio': (1.0, 2.0),
                'typical_furniture': ['desk', 'chair', 'bookshelf'],
                'wall_density': 'medium',
                'shape_preference': 'rectangular'
            }
        }
    
    def analyze_rooms(self, binary_image: np.ndarray, walls: List[Dict]) -> List[Dict]:
        """Advanced room analysis using multiple AI techniques"""
        logger.info("Starting advanced room analysis...")
        
        # Method 1: Watershed segmentation
        watershed_rooms = self._watershed_segmentation(binary_image)
        
        # Method 2: Graph-based room detection
        graph_rooms = self._graph_based_room_detection(binary_image, walls)
        
        # Method 3: Connected component analysis with shape analysis
        component_rooms = self._enhanced_connected_components(binary_image)
        
        # Combine and validate results
        combined_rooms = self._combine_room_detections(
            watershed_rooms, graph_rooms, component_rooms
        )
        
        # Enhanced room classification
        classified_rooms = self._classify_rooms_advanced(combined_rooms, binary_image)
        
        # Validate and filter rooms
        final_rooms = self._validate_rooms(classified_rooms, binary_image)
        
        logger.info(f"Room analysis complete: {len(final_rooms)} rooms detected")
        return final_rooms
    
    def _watershed_segmentation(self, binary_image: np.ndarray) -> List[Dict]:
        """Use watershed algorithm for room segmentation"""
        rooms = []
        
        try:
            # Distance transform
            room_mask = (binary_image == 255).astype(np.uint8)
            dist_transform = cv2.distanceTransform(room_mask, cv2.DIST_L2, 5)
            
            # Find local maxima (room centers)
            local_maxima = feature.peak_local_maxima(
                dist_transform, min_distance=20, threshold_abs=5
            )
            
            # Create markers for watershed
            markers = np.zeros_like(binary_image, dtype=np.int32)
            for i, (y, x) in enumerate(zip(local_maxima[0], local_maxima[1])):
                markers[y, x] = i + 1
            
            # Apply watershed
            labels = segmentation.watershed(-dist_transform, markers, mask=room_mask)
            
            # Extract room regions
            for region_id in range(1, labels.max() + 1):
                region_mask = (labels == region_id)
                
                if np.sum(region_mask) > 100:  # Minimum size threshold
                    # Get bounding box
                    coords = np.where(region_mask)
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    room = {
                        'id': f'watershed_{region_id}',
                        'bounds': {
                            'x': x_min * self.scale_factor,
                            'y': y_min * self.scale_factor,
                            'width': (x_max - x_min) * self.scale_factor,
                            'height': (y_max - y_min) * self.scale_factor
                        },
                        'area': np.sum(region_mask) * (self.scale_factor ** 2),
                        'mask': region_mask,
                        'method': 'watershed',
                        'confidence': self._calculate_region_confidence(region_mask, dist_transform)
                    }
                    rooms.append(room)
        
        except Exception as e:
            logger.warning(f"Watershed segmentation failed: {e}")
        
        return rooms
    
    def _graph_based_room_detection(self, binary_image: np.ndarray, walls: List[Dict]) -> List[Dict]:
        """Use graph theory to detect rooms based on wall connectivity"""
        rooms = []
        
        try:
            # Create graph from walls
            G = nx.Graph()
            
            # Add wall endpoints as nodes
            for i, wall in enumerate(walls):
                start = tuple(wall['start'])
                end = tuple(wall['end'])
                G.add_edge(start, end, wall_id=i, length=wall.get('length', 0))
            
            # Find cycles in the graph (potential rooms)
            cycles = nx.cycle_basis(G)
            
            for i, cycle in enumerate(cycles):
                if len(cycle) >= 3:  # At least 3 walls to form a room
                    # Calculate room bounds from cycle
                    x_coords = [node[0] for node in cycle]
                    z_coords = [node[2] for node in cycle]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    z_min, z_max = min(z_coords), max(z_coords)
                    
                    width = x_max - x_min
                    height = z_max - z_min
                    area = width * height
                    
                    if area > self.min_room_area:
                        room = {
                            'id': f'graph_{i}',
                            'height': height,
                            'area': area,
                            'cycle_nodes': cycle,
                            'method': 'graph',
                            'confidence': len(cycle) / 10.0  # More walls = higher confidence
                        }
                        rooms.append(room)
        
        except Exception as e:
            logger.warning(f"Graph-based room detection failed: {e}")
        
        return rooms
    
    def _enhanced_connected_components(self, binary_image: np.ndarray) -> List[Dict]:
        """Enhanced connected component analysis with shape features"""
        rooms = []
        
        try:
            # Find connected components
            room_mask = (binary_image == 255).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                room_mask, connectivity=8
            )
            
            for i in range(1, num_labels):  # Skip background
                area_pixels = stats[i, cv2.CC_STAT_AREA]
                area_meters = area_pixels * (self.scale_factor ** 2)
                
                if area_meters < self.min_room_area:
                    continue
                
                # Extract region
                region_mask = (labels == i)
                
                # Calculate shape features
                shape_features = self._calculate_shape_features(region_mask)
                
                # Get bounding box
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                room = {
                    'id': f'component_{i}',
                    'bounds': {
                        'x': x * self.scale_factor,
                        'y': y * self.scale_factor,
                        'width': w * self.scale_factor,
                        'height': h * self.scale_factor
                    },
                    'area': area_meters,
                    'centroid': [centroids[i][0] * self.scale_factor, centroids[i][1] * self.scale_factor],
                    'mask': region_mask,
                    'shape_features': shape_features,
                    'method': 'connected_components'
                }
                rooms.append(room)
        
        except Exception as e:
            logger.warning(f"Connected components analysis failed: {e}")
        
        return rooms
    
    def _calculate_shape_features(self, mask: np.ndarray) -> Dict:
        """Calculate advanced shape features for room classification"""
        try:
            # Find contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {}
            
            contour = max(contours, key=cv2.contourArea)
            
            # Basic measurements
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Shape features
            features = {}
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            features['aspect_ratio'] = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            
            # Rectangularity (how close to rectangle)
            rect_area = w * h
            features['rectangularity'] = area / rect_area if rect_area > 0 else 0
            
            # Compactness
            features['compactness'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            features['convexity'] = area / hull_area if hull_area > 0 else 0
            
            # Solidity
            features['solidity'] = area / hull_area if hull_area > 0 else 0
            
            # Extent
            features['extent'] = area / rect_area if rect_area > 0 else 0
            
            # Moments for shape analysis
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                features['hu_moments'] = cv2.HuMoments(moments).flatten()
            
            return features
        
        except Exception as e:
            logger.warning(f"Shape feature calculation failed: {e}")
            return {}
    
    def _calculate_region_confidence(self, mask: np.ndarray, dist_transform: np.ndarray) -> float:
        """Calculate confidence score for a room region"""
        try:
            # Average distance from walls
            avg_distance = np.mean(dist_transform[mask])
            
            # Normalized by maximum possible distance
            max_distance = np.max(dist_transform)
            confidence = avg_distance / max_distance if max_distance > 0 else 0
            
            return min(confidence, 1.0)
        except:
            return 0.5
    
    def _combine_room_detections(self, *room_lists) -> List[Dict]:
        """Combine room detections from multiple methods"""
        all_rooms = []
        for room_list in room_lists:
            all_rooms.extend(room_list)
        
        if not all_rooms:
            return []
        
        # Remove overlapping rooms using spatial clustering
        combined_rooms = self._remove_overlapping_rooms(all_rooms)
        
        return combined_rooms
    
    def _remove_overlapping_rooms(self, rooms: List[Dict]) -> List[Dict]:
        """Remove overlapping rooms using IoU and confidence scores"""
        if len(rooms) <= 1:
            return rooms
        
        # Calculate IoU matrix
        n = len(rooms)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                iou = self._calculate_iou(rooms[i], rooms[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Remove highly overlapping rooms (IoU > 0.5)
        to_remove = set()
        for i in range(n):
            for j in range(i+1, n):
                if iou_matrix[i, j] > 0.5:
                    # Keep the room with higher confidence
                    conf_i = rooms[i].get('confidence', 0.5)
                    conf_j = rooms[j].get('confidence', 0.5)
                    
                    if conf_i >= conf_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
        
        # Return non-overlapping rooms
        return [room for i, room in enumerate(rooms) if i not in to_remove]
    
    def _calculate_iou(self, room1: Dict, room2: Dict) -> float:
        """Calculate Intersection over Union for two rooms"""
        try:
            bounds1 = room1['bounds']
            bounds2 = room2['bounds']
            
            # Calculate intersection
            x1 = max(bounds1['x'], bounds2['x'])
            y1 = max(bounds1['y'], bounds2['y'])
            x2 = min(bounds1['x'] + bounds1['width'], bounds2['x'] + bounds2['width'])
            y2 = min(bounds1['y'] + bounds1['height'], bounds2['y'] + bounds2['height'])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            area1 = bounds1['width'] * bounds1['height']
            area2 = bounds2['width'] * bounds2['height']
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _classify_rooms_advanced(self, rooms: List[Dict], binary_image: np.ndarray) -> List[Dict]:
        """Advanced room classification using multiple features"""
        classified_rooms = []
        
        for room in rooms:
            # Get room features
            area = room['area']
            bounds = room['bounds']
            aspect_ratio = bounds['height'] / bounds['width'] if bounds['width'] > 0 else 1.0
            
            # Shape features if available
            shape_features = room.get('shape_features', {})
            
            # Context features
            context_features = self._extract_context_features(room, binary_image)
            
            # Classify using template matching + ML features
            room_type, confidence = self._classify_room_type(
                area, aspect_ratio, shape_features, context_features
            )
            
            room['type'] = room_type
            room['classification_confidence'] = confidence
            room['context_features'] = context_features
            
            classified_rooms.append(room)
        
        return classified_rooms
    
    def _extract_context_features(self, room: Dict, binary_image: np.ndarray) -> Dict:
        """Extract contextual features for room classification"""
        features = {}
        
        try:
            bounds = room['bounds']
            
            # Position features
            height, width = binary_image.shape
            features['relative_x'] = bounds['x'] / (width * self.scale_factor)
            features['relative_y'] = bounds['y'] / (height * self.scale_factor)
            features['relative_area'] = room['area'] / (width * height * self.scale_factor ** 2)
            
            # Adjacency features (simplified)
            center_x = bounds['x'] + bounds['width'] / 2
            center_y = bounds['y'] + bounds['height'] / 2
            
            # Distance from edges
            features['distance_from_left'] = center_x / (width * self.scale_factor)
            features['distance_from_top'] = center_y / (height * self.scale_factor)
            
            # Wall density around room
            wall_density = self._calculate_local_wall_density(room, binary_image)
            features['wall_density'] = wall_density
            
        except Exception as e:
            logger.warning(f"Context feature extraction failed: {e}")
        
        return features
    
    def _calculate_local_wall_density(self, room: Dict, binary_image: np.ndarray) -> float:
        """Calculate wall density around a room"""
        try:
            bounds = room['bounds']
            
            # Define expanded region around room
            padding = 0.5  # meters
            x1 = max(0, int((bounds['x'] - padding) / self.scale_factor))
            y1 = max(0, int((bounds['y'] - padding) / self.scale_factor))
            x2 = min(binary_image.shape[1], int((bounds['x'] + bounds['width'] + padding) / self.scale_factor))
            y2 = min(binary_image.shape[0], int((bounds['y'] + bounds['height'] + padding) / self.scale_factor))
            
            # Extract region
            region = binary_image[y1:y2, x1:x2]
            
            # Calculate wall density
            total_pixels = region.size
            wall_pixels = np.sum(region == 0)
            
            return wall_pixels / total_pixels if total_pixels > 0 else 0
        
        except:
            return 0.5
    
    def _classify_room_type(self, area: float, aspect_ratio: float, 
                           shape_features: Dict, context_features: Dict) -> Tuple[str, float]:
        """Classify room type using template matching and features"""
        
        best_score = 0
        best_type = "Room"
        
        for room_type, template in self.room_templates.items():
            score = 0
            
            # Area matching (40% weight)
            if template['min_area'] <= area <= template['max_area']:
                score += 4
            else:
                # Penalize deviation from ideal range
                if area < template['min_area']:
                    penalty = (template['min_area'] - area) / template['min_area']
                    score += max(0, 4 - penalty * 2)
                else:
                    penalty = (area - template['max_area']) / template['max_area']
                    score += max(0, 4 - penalty * 2)
            
            # Aspect ratio matching (30% weight)
            min_ar, max_ar = template['aspect_ratio']
            if min_ar <= aspect_ratio <= max_ar:
                score += 3
            else:
                if aspect_ratio < min_ar:
                    penalty = (min_ar - aspect_ratio) / min_ar
                    score += max(0, 3 - penalty)
                else:
                    penalty = (aspect_ratio - max_ar) / max_ar
                    score += max(0, 3 - penalty)
            
            # Shape features matching (20% weight)
            if shape_features:
                shape_score = self._match_shape_features(shape_features, template)
                score += shape_score * 2
            
            # Context features matching (10% weight)
            if context_features:
                context_score = self._match_context_features(context_features, template)
                score += context_score * 1
            
            if score > best_score:
                best_score = score
                best_type = room_type
        
        # Normalize confidence score
        max_possible_score = 10
        confidence = min(best_score / max_possible_score, 1.0)
        
        # Format room type name
        formatted_type = best_type.replace('_', ' ').title()
        
        return formatted_type, confidence
    
    def _match_shape_features(self, features: Dict, template: Dict) -> float:
        """Match shape features against room template"""
        score = 0
        
        # Rectangularity preference
        rectangularity = features.get('rectangularity', 0.5)
        if template['shape_preference'] == 'rectangular' and rectangularity > 0.7:
            score += 0.5
        elif template['shape_preference'] == 'open' and rectangularity > 0.6:
            score += 0.3
        
        # Compactness
        compactness = features.get('compactness', 0.5)
        if compactness > 0.6:  # Well-formed rooms
            score += 0.3
        
        # Convexity
        convexity = features.get('convexity', 0.5)
        if convexity > 0.8:  # Simple, convex shapes
            score += 0.2
        
        return min(score, 1.0)
    
    def _match_context_features(self, features: Dict, template: Dict) -> float:
        """Match context features against room template"""
        score = 0
        
        # Wall density matching
        wall_density = features.get('wall_density', 0.5)
        template_density = template.get('wall_density', 'medium')
        
        if template_density == 'very_high' and wall_density > 0.7:
            score += 0.4
        elif template_density == 'high' and 0.5 < wall_density <= 0.7:
            score += 0.4
        elif template_density == 'medium' and 0.3 < wall_density <= 0.5:
            score += 0.4
        elif template_density == 'low' and wall_density <= 0.3:
            score += 0.4
        
        # Position preferences (bathrooms often smaller, kitchens often connected)
        relative_area = features.get('relative_area', 0.5)
        if template.get('min_area', 0) < 8 and relative_area < 0.15:  # Small rooms
            score += 0.3
        elif template.get('min_area', 0) >= 15 and relative_area > 0.2:  # Large rooms
            score += 0.3
        
        return min(score, 1.0)
    
    def _validate_rooms(self, rooms: List[Dict], binary_image: np.ndarray) -> List[Dict]:
        """Validate and filter room detections"""
        valid_rooms = []
        
        for room in rooms:
            # Basic validation
            if room['area'] < self.min_room_area:
                continue
            
            # Check bounds validity
            bounds = room['bounds']
            if bounds['width'] <= 0 or bounds['height'] <= 0:
                continue
            
            # Check if room is within image bounds
            height, width = binary_image.shape
            max_x = (bounds['x'] + bounds['width']) / self.scale_factor
            max_y = (bounds['y'] + bounds['height']) / self.scale_factor
            
            if max_x > width or max_y > height:
                continue
            
            # Minimum confidence threshold
            confidence = room.get('classification_confidence', 0.5)
            if confidence < 0.2:
                continue
            
            valid_rooms.append(room)
        
        # Sort by confidence and area
        valid_rooms.sort(key=lambda r: (r.get('classification_confidence', 0.5), r['area']), reverse=True)
        
        return valid_rooms