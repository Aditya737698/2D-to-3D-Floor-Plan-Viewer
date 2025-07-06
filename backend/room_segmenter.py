"""
Room segmentation and classification component
"""
import cv2
import numpy as np
import logging
from typing import List, Dict
from skimage import measure
from skimage.feature import corner_peaks

logger = logging.getLogger(__name__)

class RoomSegmenter:
    def __init__(self, scale_factor: float = 0.05, min_room_area: float = 1.5):
        self.scale_factor = scale_factor
        self.min_room_area = min_room_area
        
        # Enhanced room classification parameters
        self.room_templates = {
            'master_bedroom': {'min_area': 15, 'max_area': 35, 'aspect_ratio': (1.0, 2.2), 'keywords': ['master', 'main']},
            'bedroom': {'min_area': 8, 'max_area': 25, 'aspect_ratio': (1.0, 2.0), 'keywords': ['bed', 'room']},
            'living_room': {'min_area': 15, 'max_area': 60, 'aspect_ratio': (1.0, 2.5), 'keywords': ['living', 'lounge', 'family']},
            'kitchen': {'min_area': 6, 'max_area': 25, 'aspect_ratio': (1.0, 3.5), 'keywords': ['kitchen', 'pantry']},
            'dining_room': {'min_area': 8, 'max_area': 30, 'aspect_ratio': (1.0, 2.0), 'keywords': ['dining', 'meal']},
            'bathroom': {'min_area': 2, 'max_area': 12, 'aspect_ratio': (1.0, 3.0), 'keywords': ['bath', 'toilet', 'wc']},
            'garage': {'min_area': 12, 'max_area': 80, 'aspect_ratio': (1.0, 3.0), 'keywords': ['garage', 'car']},
            'corridor': {'min_area': 1, 'max_area': 15, 'aspect_ratio': (3.0, 20.0), 'keywords': ['corridor', 'hall', 'passage']},
            'closet': {'min_area': 1, 'max_area': 6, 'aspect_ratio': (1.0, 4.0), 'keywords': ['closet', 'wardrobe', 'storage']},
            'study': {'min_area': 6, 'max_area': 20, 'aspect_ratio': (1.0, 2.0), 'keywords': ['study', 'office', 'work']},
            'laundry': {'min_area': 3, 'max_area': 12, 'aspect_ratio': (1.0, 3.0), 'keywords': ['laundry', 'utility']},
            'balcony': {'min_area': 2, 'max_area': 15, 'aspect_ratio': (1.0, 5.0), 'keywords': ['balcony', 'terrace', 'patio']}
        }
    
    def segment_rooms(self, preprocessed: Dict[str, np.ndarray]) -> List[Dict]:
        """Advanced room segmentation with improved accuracy"""
        logger.info("Starting room segmentation...")
        
        try:
            binary = preprocessed['cleaned']
            height, width = binary.shape
            
            # Multi-step room segmentation approach
            rooms_watershed = self._segment_rooms_watershed(binary)
            rooms_connected = self._segment_rooms_connected_components(binary)
            
            # Combine and validate results
            all_rooms = rooms_watershed + rooms_connected
            
            # Filter and merge rooms
            filtered_rooms = self._filter_and_merge_rooms(all_rooms)
            
            # Classify room types with enhanced logic
            classified_rooms = []
            for room in filtered_rooms:
                room_type = self._classify_room_type(room)
                room.update({"type": room_type})
                classified_rooms.append(room)
            
            logger.info(f"Room segmentation complete: {len(classified_rooms)} rooms")
            return classified_rooms
            
        except Exception as e:
            logger.error(f"Error in room segmentation: {str(e)}")
            return self._create_fallback_rooms(width, height)
    
    def _segment_rooms_watershed(self, binary: np.ndarray) -> List[Dict]:
        """Segment rooms using watershed algorithm"""
        rooms = []
        
        try:
            # Distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find local maxima
            local_maxima = corner_peaks(dist_transform, min_distance=20, threshold_abs=0.3*dist_transform.max())
            
            # Create markers
            markers = np.zeros(binary.shape, dtype=np.int32)
            for i, peak in enumerate(local_maxima):
                markers[peak[0], peak[1]] = i + 1
            
            # Apply watershed
            labels = measure.watershed(-dist_transform, markers, mask=binary)
            
            # Extract room regions
            for region_id in np.unique(labels):
                if region_id == 0:  # Background
                    continue
                
                region_mask = (labels == region_id)
                region_props = measure.regionprops(region_mask.astype(int))
                
                if len(region_props) > 0:
                    prop = region_props[0]
                    area_pixels = prop.area
                    area_meters = area_pixels * (self.scale_factor ** 2)
                    
                    if area_meters >= self.min_room_area:
                        bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
                        
                        room = {
                            "id": f"room_watershed_{region_id}",
                            "bounds": {
                                "x": bbox[1] * self.scale_factor,
                                "y": bbox[0] * self.scale_factor,
                                "width": (bbox[3] - bbox[1]) * self.scale_factor,
                                "height": (bbox[2] - bbox[0]) * self.scale_factor
                            },
                            "area": area_meters,
                            "centroid": [prop.centroid[1] * self.scale_factor, prop.centroid[0] * self.scale_factor],
                            "aspect_ratio": prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0,
                            "method": "watershed"
                        }
                        rooms.append(room)
            
            logger.debug(f"Watershed segmentation found {len(rooms)} rooms")
            return rooms
            
        except Exception as e:
            logger.error(f"Error in watershed segmentation: {str(e)}")
            return []
    
    def _segment_rooms_connected_components(self, binary: np.ndarray) -> List[Dict]:
        """Segment rooms using connected components analysis"""
        rooms = []
        
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area_pixels = stats[i, cv2.CC_STAT_AREA]
                area_meters = area_pixels * (self.scale_factor ** 2)
                
                if area_meters >= self.min_room_area:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    
                    # Skip very thin regions
                    if w < 10 or h < 10:
                        continue
                    
                    room = {
                        "id": f"room_cc_{i}",
                        "bounds": {
                            "x": x * self.scale_factor,
                            "y": y * self.scale_factor,
                            "width": w * self.scale_factor,
                            "height": h * self.scale_factor
                        },
                        "area": area_meters,
                        "centroid": [centroids[i][0] * self.scale_factor, centroids[i][1] * self.scale_factor],
                        "aspect_ratio": max(w, h) / min(w, h) if min(w, h) > 0 else 1.0,
                        "method": "connected_components"
                    }
                    rooms.append(room)
            
            logger.debug(f"Connected components found {len(rooms)} rooms")
            return rooms
            
        except Exception as e:
            logger.error(f"Error in connected components segmentation: {str(e)}")
            return []
    
    def _filter_and_merge_rooms(self, rooms: List[Dict]) -> List[Dict]:
        """Filter and merge overlapping or duplicate rooms"""
        if not rooms:
            return []
        
        try:
            # Remove duplicate rooms based on overlap
            filtered_rooms = []
            
            for room in rooms:
                is_duplicate = False
                
                for existing_room in filtered_rooms:
                    overlap_ratio = self._calculate_room_overlap(room, existing_room)
                    
                    if overlap_ratio > 0.7:  # 70% overlap threshold
                        is_duplicate = True
                        # Keep the larger room
                        if room["area"] > existing_room["area"]:
                            filtered_rooms.remove(existing_room)
                            filtered_rooms.append(room)
                        break
                
                if not is_duplicate:
                    filtered_rooms.append(room)
            
            # Sort by area (largest first)
            filtered_rooms.sort(key=lambda r: r["area"], reverse=True)
            
            logger.debug(f"Filtered rooms: {len(rooms)} -> {len(filtered_rooms)}")
            return filtered_rooms
            
        except Exception as e:
            logger.error(f"Error filtering rooms: {str(e)}")
            return rooms
    
    def _calculate_room_overlap(self, room1: Dict, room2: Dict) -> float:
        """Calculate overlap ratio between two rooms"""
        try:
            bounds1 = room1["bounds"]
            bounds2 = room2["bounds"]
            
            # Calculate intersection rectangle
            x1 = max(bounds1["x"], bounds2["x"])
            y1 = max(bounds1["y"], bounds2["y"])
            x2 = min(bounds1["x"] + bounds1["width"], bounds2["x"] + bounds2["width"])
            y2 = min(bounds1["y"] + bounds1["height"], bounds2["y"] + bounds2["height"])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0  # No overlap
            
            intersection_area = (x2 - x1) * (y2 - y1)
            smaller_area = min(room1["area"], room2["area"])
            
            return intersection_area / smaller_area if smaller_area > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating room overlap: {str(e)}")
            return 0.0
    
    def _classify_room_type(self, room: Dict) -> str:
        """Advanced room type classification with context analysis"""
        try:
            area = room["area"]
            aspect_ratio = room["aspect_ratio"]
            bounds = room["bounds"]
            
            # Basic template matching
            best_score = 0
            best_type = "Room"
            
            for room_type, template in self.room_templates.items():
                score = 0
                
                # Area score (0-4 points)
                if template['min_area'] <= area <= template['max_area']:
                    score += 4
                else:
                    area_diff = min(abs(area - template['min_area']), abs(area - template['max_area']))
                    area_penalty = area_diff / max(template['max_area'], area)
                    score += max(0, 4 - area_penalty * 2)
                
                # Aspect ratio score (0-3 points)
                min_ar, max_ar = template['aspect_ratio']
                if min_ar <= aspect_ratio <= max_ar:
                    score += 3
                else:
                    ar_diff = min(abs(aspect_ratio - min_ar), abs(aspect_ratio - max_ar))
                    ar_penalty = ar_diff / max(max_ar, aspect_ratio)
                    score += max(0, 3 - ar_penalty)
                
                # Position-based heuristics (0-2 points)
                if room_type == 'garage':
                    # Garages are often at the edges
                    if bounds["x"] < 2.0 or bounds["y"] < 2.0:
                        score += 1
                elif room_type == 'corridor':
                    # Corridors often connect rooms
                    if aspect_ratio > 4.0:
                        score += 2
                elif room_type == 'bathroom':
                    # Bathrooms are often smaller and square-ish
                    if 1.0 <= aspect_ratio <= 2.0 and area < 10:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_type = room_type
            
            # Format the room type name
            formatted_type = best_type.replace('_', ' ').title()
            logger.debug(f"Classified room: {formatted_type} (score: {best_score:.2f}, area: {area:.2f}m², AR: {aspect_ratio:.2f})")
            
            return formatted_type
            
        except Exception as e:
            logger.error(f"Error classifying room: {str(e)}")
            return "Room"
    
    def _create_fallback_rooms(self, width: int, height: int) -> List[Dict]:
        """Create fallback rooms when segmentation fails"""
        w_meters = width * self.scale_factor
        h_meters = height * self.scale_factor
        
        return [{
            "id": "room_fallback",
            "type": "Living Room",
            "bounds": {
                "x": 1.0,
                "y": 1.0,
                "width": max(w_meters - 2.0, 3.0),
                "height": max(h_meters - 2.0, 3.0)
            },
            "area": max((w_meters - 2.0) * (h_meters - 2.0), 9.0),
            "centroid": [w_meters * 0.5, h_meters * 0.5],
            "aspect_ratio": max(width, height) / min(width, height)
        }]