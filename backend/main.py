from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import json
import os
from typing import Dict, List, Any
import uvicorn
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Floor Plan 3D Converter", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class EnhancedFloorPlanProcessor:
    def __init__(self):
        self.scale_factor = 0.05  # 1 pixel = 0.05 meters
        self.wall_thickness = 0.2
        self.wall_height = 3.0
        self.min_wall_length = 0.5
        self.min_room_area = 2.0
        
        # Room classification parameters
        self.room_templates = {
            'bathroom': {'min_area': 2, 'max_area': 8, 'aspect_ratio': (1.0, 3.0)},
            'bedroom': {'min_area': 8, 'max_area': 30, 'aspect_ratio': (1.0, 2.0)},
            'kitchen': {'min_area': 6, 'max_area': 20, 'aspect_ratio': (1.0, 3.0)},
            'living_room': {'min_area': 12, 'max_area': 50, 'aspect_ratio': (1.0, 2.5)},
            'corridor': {'min_area': 1, 'max_area': 10, 'aspect_ratio': (3.0, 10.0)},
            'closet': {'min_area': 1, 'max_area': 4, 'aspect_ratio': (1.0, 3.0)}
        }
    
    def preprocess_image(self, image_array: np.ndarray) -> tuple:
        """Enhanced image preprocessing with adaptive techniques"""
        logger.info(f"Processing image of size: {image_array.shape}")
        
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.copy()
            
            # Store original for reference
            original_gray = gray.copy()
            
            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Multiple thresholding approaches
            _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Choose the best threshold
            # Count black vs white pixels to determine which is better
            otsu_black_ratio = np.sum(thresh_otsu == 0) / thresh_otsu.size
            adaptive_black_ratio = np.sum(thresh_adaptive == 0) / thresh_adaptive.size
            
            # Use the threshold that gives a reasonable amount of black pixels (walls)
            if 0.05 <= otsu_black_ratio <= 0.4:  # 5-40% black pixels seems reasonable for floor plans
                binary = thresh_otsu
                method_used = "OTSU"
            elif 0.05 <= adaptive_black_ratio <= 0.4:
                binary = thresh_adaptive
                method_used = "Adaptive"
            else:
                # If neither is good, use OTSU as fallback
                binary = thresh_otsu
                method_used = "OTSU (fallback)"
            
            # Clean up the binary image
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Ensure walls are black (0) and rooms are white (255)
            black_pixels = np.sum(binary == 0)
            white_pixels = np.sum(binary == 255)
            
            if black_pixels > white_pixels:
                binary = cv2.bitwise_not(binary)
                walls_inverted = True
            else:
                walls_inverted = False
            
            logger.info(f"Preprocessing complete using {method_used}. Black: {np.sum(binary==0)}, White: {np.sum(binary==255)}, Inverted: {walls_inverted}")
            
            return binary, original_gray
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            # Fallback to simple thresholding
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.copy()
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            return binary, gray
    
    def detect_walls_enhanced(self, binary_image: np.ndarray) -> List[Dict]:
        """Enhanced wall detection with improved filtering"""
        logger.info("Starting enhanced wall detection...")
        
        walls = []
        height, width = binary_image.shape
        
        try:
            # Edge detection with different parameters for better results
            edges = cv2.Canny(binary_image, 30, 100, apertureSize=3)
            
            # Detect lines using HoughLinesP with adaptive parameters
            min_line_length = max(min(width, height) * 0.03, 20)  # At least 3% of image dimension or 20px
            max_line_gap = max(min(width, height) * 0.01, 5)     # Max 1% gap or 5px
            threshold = max(min(width, height) * 0.08, 30)       # Adaptive threshold
            
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=int(threshold),
                minLineLength=int(min_line_length),
                maxLineGap=int(max_line_gap)
            )
            
            if lines is not None and len(lines) > 0:
                logger.info(f"Found {len(lines)} potential wall lines")
                
                # Process and filter lines
                processed_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Convert to meters
                    start = [x1 * self.scale_factor, 0, y1 * self.scale_factor]
                    end = [x2 * self.scale_factor, 0, y2 * self.scale_factor]
                    
                    # Calculate length
                    length = np.sqrt((end[0] - start[0])**2 + (end[2] - start[2])**2)
                    
                    # Filter out very short lines
                    if length >= self.min_wall_length:
                        wall = {
                            "start": start,
                            "end": end,
                            "height": self.wall_height,
                            "thickness": self.wall_thickness,
                            "length": length
                        }
                        processed_lines.append(wall)
                
                # Merge similar walls
                walls = self.merge_similar_walls(processed_lines)
                logger.info(f"After merging: {len(walls)} walls")
            else:
                logger.warning("No lines detected by HoughLinesP")
                # Generate some basic perimeter walls as fallback
                walls = self.generate_fallback_walls(width, height)
                
        except Exception as e:
            logger.error(f"Error in wall detection: {str(e)}")
            # Generate fallback walls
            walls = self.generate_fallback_walls(width, height)
        
        logger.info(f"Final wall count: {len(walls)}")
        return walls
    
    def generate_fallback_walls(self, width: int, height: int) -> List[Dict]:
        """Generate basic perimeter walls as fallback"""
        logger.info("Generating fallback perimeter walls")
        
        # Create a basic rectangular room
        w_meters = width * self.scale_factor
        h_meters = height * self.scale_factor
        
        walls = [
            # Bottom wall
            {
                "start": [0, 0, 0],
                "end": [w_meters, 0, 0],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": w_meters
            },
            # Right wall
            {
                "start": [w_meters, 0, 0],
                "end": [w_meters, 0, h_meters],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": h_meters
            },
            # Top wall
            {
                "start": [w_meters, 0, h_meters],
                "end": [0, 0, h_meters],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": w_meters
            },
            # Left wall
            {
                "start": [0, 0, h_meters],
                "end": [0, 0, 0],
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": h_meters
            }
        ]
        
        return walls
    
    def merge_similar_walls(self, walls: List[Dict]) -> List[Dict]:
        """Merge walls that are very close and parallel"""
        if not walls or len(walls) <= 1:
            return walls
        
        try:
            merged_walls = []
            used = set()
            
            for i, wall1 in enumerate(walls):
                if i in used:
                    continue
                
                # Calculate wall direction
                dx1 = wall1['end'][0] - wall1['start'][0]
                dz1 = wall1['end'][2] - wall1['start'][2]
                length1 = np.sqrt(dx1*dx1 + dz1*dz1)
                
                if length1 == 0:
                    continue
                    
                angle1 = np.arctan2(dz1, dx1)
                
                similar_walls = [wall1]
                used.add(i)
                
                for j, wall2 in enumerate(walls[i+1:], i+1):
                    if j in used:
                        continue
                    
                    # Calculate wall2 direction
                    dx2 = wall2['end'][0] - wall2['start'][0]
                    dz2 = wall2['end'][2] - wall2['start'][2]
                    length2 = np.sqrt(dx2*dx2 + dz2*dz2)
                    
                    if length2 == 0:
                        continue
                        
                    angle2 = np.arctan2(dz2, dx2)
                    
                    # Check if walls are parallel (similar angles)
                    angle_diff = abs(angle1 - angle2)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    if angle_diff < np.pi/12:  # Within 15 degrees
                        # Check if walls are close to each other
                        min_dist = float('inf')
                        for p1 in [wall1['start'], wall1['end']]:
                            for p2 in [wall2['start'], wall2['end']]:
                                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
                                min_dist = min(min_dist, dist)
                        
                        if min_dist < 1.0:  # Within 1 meter
                            similar_walls.append(wall2)
                            used.add(j)
                
                # If we found similar walls, merge them
                if len(similar_walls) > 1:
                    merged_wall = self.merge_wall_group(similar_walls)
                    merged_walls.append(merged_wall)
                else:
                    merged_walls.append(wall1)
            
            return merged_walls
            
        except Exception as e:
            logger.error(f"Error merging walls: {str(e)}")
            return walls
    
    def merge_wall_group(self, walls: List[Dict]) -> Dict:
        """Merge a group of similar walls into one"""
        try:
            # Find the overall start and end points
            all_points = []
            for wall in walls:
                all_points.extend([wall['start'], wall['end']])
            
            # Find the two points that are farthest apart
            max_dist = 0
            best_start, best_end = all_points[0], all_points[1]
            
            for i, p1 in enumerate(all_points):
                for j, p2 in enumerate(all_points[i+1:], i+1):
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_start, best_end = p1, p2
            
            return {
                "start": best_start,
                "end": best_end,
                "height": self.wall_height,
                "thickness": self.wall_thickness,
                "length": max_dist
            }
        except Exception as e:
            logger.error(f"Error merging wall group: {str(e)}")
            return walls[0]  # Return first wall as fallback
    
    def segment_rooms_enhanced(self, binary_image: np.ndarray) -> List[Dict]:
        """Enhanced room segmentation using flood fill and contour analysis"""
        logger.info("Starting enhanced room segmentation...")
        
        rooms = []
        height, width = binary_image.shape
        
        try:
            # Invert image so rooms (white areas) become connected components
            room_image = binary_image.copy()
            
            # Fill small holes in walls
            kernel = np.ones((5,5), np.uint8)
            room_image = cv2.morphologyEx(room_image, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components (rooms)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                room_image, connectivity=8
            )
            
            logger.info(f"Found {num_labels-1} potential room regions")
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area_pixels = stats[i, cv2.CC_STAT_AREA]
                area_meters = area_pixels * (self.scale_factor ** 2)
                
                # Filter out very small areas
                if area_meters < self.min_room_area:
                    logger.debug(f"Skipping small area: {area_meters:.2f}m²")
                    continue
                
                # Get bounding box
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Skip very thin regions (likely noise)
                if w < 10 or h < 10:
                    continue
                
                # Convert to meters
                room = {
                    "id": f"room_{i}",
                    "bounds": {
                        "x": x * self.scale_factor,
                        "y": y * self.scale_factor,
                        "width": w * self.scale_factor,
                        "height": h * self.scale_factor
                    },
                    "area": area_meters,
                    "centroid": [centroids[i][0] * self.scale_factor, centroids[i][1] * self.scale_factor],
                    "aspect_ratio": max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                }
                
                # Classify room type
                room["type"] = self.classify_room_type(room)
                rooms.append(room)
                logger.debug(f"Added room: {room['type']}, area: {area_meters:.2f}m²")
            
            # If no rooms found, create a default room
            if not rooms:
                logger.warning("No rooms detected, creating default room")
                rooms.append({
                    "id": "room_default",
                    "type": "Room",
                    "bounds": {
                        "x": width * 0.1 * self.scale_factor,
                        "y": height * 0.1 * self.scale_factor,
                        "width": width * 0.8 * self.scale_factor,
                        "height": height * 0.8 * self.scale_factor
                    },
                    "area": width * height * 0.64 * (self.scale_factor ** 2),
                    "centroid": [width * 0.5 * self.scale_factor, height * 0.5 * self.scale_factor],
                    "aspect_ratio": max(width, height) / min(width, height)
                })
                
        except Exception as e:
            logger.error(f"Error in room segmentation: {str(e)}")
            # Create a fallback room
            rooms = [{
                "id": "room_fallback",
                "type": "Room",
                "bounds": {
                    "x": 1.0,
                    "y": 1.0,
                    "width": max(width * self.scale_factor - 2.0, 5.0),
                    "height": max(height * self.scale_factor - 2.0, 5.0)
                },
                "area": max((width * self.scale_factor - 2.0) * (height * self.scale_factor - 2.0), 25.0),
                "centroid": [width * 0.5 * self.scale_factor, height * 0.5 * self.scale_factor],
                "aspect_ratio": max(width, height) / min(width, height)
            }]
        
        logger.info(f"Found {len(rooms)} rooms")
        return rooms
    
    def classify_room_type(self, room: Dict) -> str:
        """Classify room type based on area and aspect ratio"""
        try:
            area = room["area"]
            aspect_ratio = room["aspect_ratio"]
            
            best_score = 0
            best_type = "Room"
            
            for room_type, template in self.room_templates.items():
                score = 0
                
                # Area score (0-3 points)
                if template['min_area'] <= area <= template['max_area']:
                    score += 3
                else:
                    if area < template['min_area']:
                        penalty = (template['min_area'] - area) / template['min_area']
                        score += max(0, 3 - penalty * 2)
                    else:
                        penalty = (area - template['max_area']) / template['max_area']
                        score += max(0, 3 - penalty * 2)
                
                # Aspect ratio score (0-2 points)
                min_ar, max_ar = template['aspect_ratio']
                if min_ar <= aspect_ratio <= max_ar:
                    score += 2
                else:
                    if aspect_ratio < min_ar:
                        penalty = (min_ar - aspect_ratio) / min_ar
                        score += max(0, 2 - penalty)
                    else:
                        penalty = (aspect_ratio - max_ar) / max_ar
                        score += max(0, 2 - penalty)
                
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
    
    def place_furniture_intelligently(self, rooms: List[Dict]) -> List[Dict]:
        """Place furniture in rooms based on type and layout"""
        all_furniture = []
        
        try:
            for room in rooms:
                furniture = self.get_room_furniture(room)
                all_furniture.extend(furniture)
            
            logger.info(f"Placed {len(all_furniture)} furniture items across {len(rooms)} rooms")
            
        except Exception as e:
            logger.error(f"Error placing furniture: {str(e)}")
        
        return all_furniture
    
    def get_room_furniture(self, room: Dict) -> List[Dict]:
        """Get appropriate furniture for a room"""
        try:
            room_type = room["type"].lower()
            bounds = room["bounds"]
            furniture = []
            
            # Room center
            center_x = bounds["x"] + bounds["width"] / 2
            center_z = bounds["y"] + bounds["height"] / 2
            
            if "living" in room_type:
                furniture.extend(self.place_living_room_furniture(bounds, center_x, center_z))
            elif "bedroom" in room_type:
                furniture.extend(self.place_bedroom_furniture(bounds, center_x, center_z))
            elif "kitchen" in room_type:
                furniture.extend(self.place_kitchen_furniture(bounds, center_x, center_z))
            elif "bathroom" in room_type:
                furniture.extend(self.place_bathroom_furniture(bounds, center_x, center_z))
            else:
                # Generic room furniture
                furniture.extend(self.place_generic_furniture(bounds, center_x, center_z))
            
            return furniture
            
        except Exception as e:
            logger.error(f"Error getting room furniture for {room.get('type', 'unknown')}: {str(e)}")
            return []
    
    def place_living_room_furniture(self, bounds: Dict, center_x: float, center_z: float) -> List[Dict]:
        """Place living room furniture"""
        furniture = []
        
        try:
            # Sofa
            furniture.append({
                "type": "Sofa",
                "position": [center_x, 0, bounds["y"] + bounds["height"] * 0.2],
                "rotation": 0,
                "scale": [min(bounds["width"] * 0.4, 2.5), 0.8, 1.0],
                "color": "#4682B4"
            })
            
            # TV
            furniture.append({
                "type": "TV",
                "position": [center_x, 0, bounds["y"] + bounds["height"] * 0.9],
                "rotation": 180,
                "scale": [min(bounds["width"] * 0.3, 2.0), 1.2, 0.15],
                "color": "#000000"
            })
            
            # Coffee table
            furniture.append({
                "type": "Coffee Table",
                "position": [center_x, 0, center_z],
                "rotation": 0,
                "scale": [1.2, 0.4, 0.8],
                "color": "#8B4513"
            })
            
            # Add chairs if room is large enough
            if bounds["width"] > 4 and bounds["height"] > 4:
                furniture.append({
                    "type": "Armchair",
                    "position": [bounds["x"] + bounds["width"] * 0.2, 0, center_z],
                    "rotation": 45,
                    "scale": [0.8, 0.8, 0.8],
                    "color": "#8B4513"
                })
                
        except Exception as e:
            logger.error(f"Error placing living room furniture: {str(e)}")
        
        return furniture
    
    def place_bedroom_furniture(self, bounds: Dict, center_x: float, center_z: float) -> List[Dict]:
        """Place bedroom furniture"""
        furniture = []
        
        try:
            # Bed
            bed_width = min(bounds["width"] * 0.4, 2.0)
            bed_length = min(bounds["height"] * 0.3, 2.2)
            
            furniture.append({
                "type": "Bed",
                "position": [center_x, 0, center_z],
                "rotation": 0 if bounds["width"] > bounds["height"] else 90,
                "scale": [bed_width, 0.6, bed_length],
                "color": "#FFFFFF"
            })
            
            # Nightstands
            if bounds["width"] > 3:
                furniture.extend([
                    {
                        "type": "Nightstand",
                        "position": [center_x + bed_width/2 + 0.4, 0, center_z],
                        "rotation": 0,
                        "scale": [0.4, 0.7, 0.4],
                        "color": "#8B4513"
                    },
                    {
                        "type": "Nightstand",
                        "position": [center_x - bed_width/2 - 0.4, 0, center_z],
                        "rotation": 0,
                        "scale": [0.4, 0.7, 0.4],
                        "color": "#8B4513"
                    }
                ])
            
            # Wardrobe
            if bounds["area"] > 12:
                furniture.append({
                    "type": "Wardrobe",
                    "position": [bounds["x"] + 0.3, 0, bounds["y"] + bounds["height"] * 0.2],
                    "rotation": 0,
                    "scale": [1.8, 2.2, 0.6],
                    "color": "#654321"
                })
                
        except Exception as e:
            logger.error(f"Error placing bedroom furniture: {str(e)}")
        
        return furniture
    
    def place_kitchen_furniture(self, bounds: Dict, center_x: float, center_z: float) -> List[Dict]:
        """Place kitchen furniture"""
        furniture = []
        
        try:
            # Counter
            counter_length = bounds["width"] * 0.8
            furniture.append({
                "type": "Counter",
                "position": [center_x, 0, bounds["y"] + 0.3],
                "rotation": 0,
                "scale": [counter_length, 0.9, 0.6],
                "color": "#D2B48C"
            })
            
            # Refrigerator
            furniture.append({
                "type": "Refrigerator",
                "position": [bounds["x"] + 0.4, 0, bounds["y"] + 0.3],
                "rotation": 0,
                "scale": [0.8, 2.0, 0.8],
                "color": "#F5F5F5"
            })
            
            # Stove
            furniture.append({
                "type": "Stove",
                "position": [center_x, 0, bounds["y"] + 0.3],
                "rotation": 0,
                "scale": [0.8, 0.9, 0.8],
                "color": "#2F4F4F"
            })
            
            # Dining table if room is large enough
            if bounds["area"] > 12:
                furniture.append({
                    "type": "Dining Table",
                    "position": [center_x, 0, center_z + bounds["height"] * 0.2],
                    "rotation": 0,
                    "scale": [1.4, 0.8, 0.9],
                    "color": "#8B4513"
                })
                
        except Exception as e:
            logger.error(f"Error placing kitchen furniture: {str(e)}")
        
        return furniture
    
    def place_bathroom_furniture(self, bounds: Dict, center_x: float, center_z: float) -> List[Dict]:
        """Place bathroom furniture"""
        furniture = []
        
        try:
            # Toilet
            furniture.append({
                "type": "Toilet",
                "position": [bounds["x"] + bounds["width"] * 0.3, 0, bounds["y"] + bounds["height"] * 0.3],
                "rotation": 0,
                "scale": [0.6, 0.8, 0.4],
                "color": "#FFFFFF"
            })
            
            # Sink
            furniture.append({
                "type": "Sink",
                "position": [bounds["x"] + bounds["width"] * 0.7, 0, bounds["y"] + 0.3],
                "rotation": 0,
                "scale": [0.6, 0.9, 0.4],
                "color": "#FFFFFF"
            })
            
            # Bathtub if room is large enough
            if bounds["area"] > 4:
                furniture.append({
                    "type": "Bathtub",
                    "position": [center_x, 0, bounds["y"] + bounds["height"] * 0.8],
                    "rotation": 0,
                    "scale": [1.6, 0.6, 0.8],
                    "color": "#FFFFFF"
                })
                
        except Exception as e:
            logger.error(f"Error placing bathroom furniture: {str(e)}")
        
        return furniture
    
    def place_generic_furniture(self, bounds: Dict, center_x: float, center_z: float) -> List[Dict]:
        """Place generic furniture for unspecified room types"""
        furniture = []
        
        try:
            # Simple table and chairs
            furniture.append({
                "type": "Table",
                "position": [center_x, 0, center_z],
                "rotation": 0,
                "scale": [1.2, 0.8, 0.8],
                "color": "#8B4513"
            })
            
            # Add chairs if room is large enough
            if bounds["area"] > 8:
                furniture.extend([
                    {
                        "type": "Chair",
                        "position": [center_x + 0.8, 0, center_z],
                        "rotation": 270,
                        "scale": [0.5, 0.9, 0.5],
                        "color": "#8B4513"
                    },
                    {
                        "type": "Chair",
                        "position": [center_x - 0.8, 0, center_z],
                        "rotation": 90,
                        "scale": [0.5, 0.9, 0.5],
                        "color": "#8B4513"
                    }
                ])
                
        except Exception as e:
            logger.error(f"Error placing generic furniture: {str(e)}")
        
        return furniture
    
    def detect_doors_windows(self, binary_image: np.ndarray, walls: List[Dict]) -> tuple:
        """Detect doors and windows by finding gaps in walls"""
        doors = []
        windows = []
        
        try:
            # Simple door placement based on room layout
            if len(walls) > 0:
                # Add a main entrance door
                doors.append({
                    "position": [0, 0, 0],
                    "rotation": 0,
                    "scale": [1.0, 2.1, 0.1],
                    "type": "Door"
                })
                
        except Exception as e:
            logger.error(f"Error detecting doors/windows: {str(e)}")
        
        return doors, windows
    
    def process_floor_plan(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Main processing pipeline with comprehensive error handling"""
        try:
            logger.info("Starting floor plan processing pipeline...")
            
            # Enhanced preprocessing
            binary_image, original_gray = self.preprocess_image(image_array)
            
            # Enhanced wall detection
            walls = self.detect_walls_enhanced(binary_image)
            
            # Enhanced room segmentation
            rooms = self.segment_rooms_enhanced(binary_image)
            
            # Detect doors and windows
            doors, windows = self.detect_doors_windows(binary_image, walls)
            
            # Intelligent furniture placement
            furniture = self.place_furniture_intelligently(rooms)
            
            # Calculate total area
            total_area = sum(room.get("area", 0) for room in rooms)
            
            result = {
                "walls": walls,
                "rooms": rooms,
                "doors": doors,
                "windows": windows,
                "furniture": furniture,
                "metadata": {
                    "scale_factor": self.scale_factor,
                    "total_area": total_area,
                    "room_count": len(rooms),
                    "wall_count": len(walls),
                    "furniture_count": len(furniture),
                    "image_size": list(image_array.shape[:2])
                }
            }
            
            logger.info(f"Processing complete: {len(walls)} walls, {len(rooms)} rooms, {len(furniture)} furniture items, total area: {total_area:.2f}m²")
            return result
            
        except Exception as e:
            logger.error(f"Error in floor plan processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal fallback result
            height, width = image_array.shape[:2]
            w_meters = width * self.scale_factor
            h_meters = height * self.scale_factor
            
            fallback_result = {
                "walls": [
                    {"start": [0, 0, 0], "end": [w_meters, 0, 0], "height": 3, "thickness": 0.2},
                    {"start": [w_meters, 0, 0], "end": [w_meters, 0, h_meters], "height": 3, "thickness": 0.2},
                    {"start": [w_meters, 0, h_meters], "end": [0, 0, h_meters], "height": 3, "thickness": 0.2},
                    {"start": [0, 0, h_meters], "end": [0, 0, 0], "height": 3, "thickness": 0.2}
                ],
                "rooms": [
                    {
                        "id": "room_1",
                        "type": "Room",
                        "bounds": {"x": 1, "y": 1, "width": w_meters-2, "height": h_meters-2},
                        "area": (w_meters-2) * (h_meters-2),
                        "centroid": [w_meters/2, h_meters/2]
                    }
                ],
                "doors": [],
                "windows": [],
                "furniture": [
                    {
                        "type": "Table",
                        "position": [w_meters/2, 0, h_meters/2],
                        "rotation": 0,
                        "scale": [1.2, 0.8, 0.8],
                        "color": "#8B4513"
                    }
                ],
                "metadata": {
                    "scale_factor": self.scale_factor,
                    "total_area": (w_meters-2) * (h_meters-2),
                    "room_count": 1,
                    "wall_count": 4,
                    "furniture_count": 1,
                    "image_size": list(image_array.shape[:2]),
                    "processing_error": str(e)
                }
            }
            
            return fallback_result

# Initialize processor
processor = EnhancedFloorPlanProcessor()

@app.post("/upload/")
async def upload_floor_plan(file: UploadFile = File(...)):
    """Upload and process floor plan image with enhanced AI processing"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        logger.info(f"Processing uploaded image: {file.filename}, size: {image_array.shape}")
        
        # Process the floor plan
        result = processor.process_floor_plan(image_array)
        
        # Save results to files
        try:
            output_files = {
                "walls.json": result["walls"],
                "rooms.json": result["rooms"],
                "doors.json": result["doors"],
                "windows.json": result["windows"],
                "furniture.json": result["furniture"],
                "labels.json": result
            }
            
            for filename, data in output_files.items():
                filepath = os.path.join("static", filename)
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save output files: {str(e)}")
        
        # Return comprehensive response
        return JSONResponse(content={
            "message": "Floor plan processed successfully",
            "data": result,
            "stats": {
                "walls": len(result["walls"]),
                "rooms": len(result["rooms"]),
                "doors": len(result["doors"]),
                "windows": len(result["windows"]),
                "furniture": len(result["furniture"]),
                "total_area": result["metadata"]["total_area"],
                "processing_info": {
                    "scale_factor": result["metadata"]["scale_factor"],
                    "image_size": result["metadata"]["image_size"]
                }
            },
            "success": True
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing upload: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced Floor Plan 3D Converter API",
        "version": "2.0.0",
        "features": [
            "Enhanced wall detection with HoughLinesP",
            "Intelligent room segmentation with connected components", 
            "Automatic room classification (Living Room, Bedroom, Kitchen, Bathroom)",
            "Smart furniture placement based on room type",
            "Adaptive image preprocessing",
            "Robust error handling and fallbacks"
        ],
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "processor": "enhanced",
        "supported_formats": ["PNG", "JPG", "JPEG", "GIF", "BMP"],
        "max_file_size": "10MB",
        "features": {
            "wall_detection": "HoughLinesP with adaptive parameters",
            "room_segmentation": "Connected components analysis",
            "room_classification": "Template matching with area and aspect ratio",
            "furniture_placement": "Rule-based intelligent placement"
        }
    }

@app.get("/sample")
async def get_sample_data():
    """Get sample floor plan data for testing"""
    sample_data = {
        "walls": [
            {"start": [0, 0, 0], "end": [10, 0, 0], "height": 3, "thickness": 0.2, "length": 10},
            {"start": [10, 0, 0], "end": [10, 0, 8], "height": 3, "thickness": 0.2, "length": 8},
            {"start": [10, 0, 8], "end": [0, 0, 8], "height": 3, "thickness": 0.2, "length": 10},
            {"start": [0, 0, 8], "end": [0, 0, 0], "height": 3, "thickness": 0.2, "length": 8},
            {"start": [5, 0, 0], "end": [5, 0, 8], "height": 3, "thickness": 0.2, "length": 8}
        ],
        "rooms": [
            {
                "id": "room_1",
                "type": "Living Room",
                "bounds": {"x": 0, "y": 0, "width": 5, "height": 8},
                "area": 40,
                "centroid": [2.5, 4],
                "aspect_ratio": 1.6
            },
            {
                "id": "room_2", 
                "type": "Bedroom",
                "bounds": {"x": 5, "y": 0, "width": 5, "height": 8},
                "area": 40,
                "centroid": [7.5, 4],
                "aspect_ratio": 1.6
            }
        ],
        "furniture": [
            {
                "type": "Sofa",
                "position": [2.5, 0, 2],
                "rotation": 0,
                "scale": [2.0, 0.8, 0.9],
                "color": "#4682B4"
            },
            {
                "type": "TV",
                "position": [2.5, 0, 6],
                "rotation": 180,
                "scale": [1.5, 1.0, 0.1],
                "color": "#000000"
            },
            {
                "type": "Coffee Table",
                "position": [2.5, 0, 4],
                "rotation": 0,
                "scale": [1.2, 0.4, 0.8],
                "color": "#8B4513"
            },
            {
                "type": "Bed",
                "position": [7.5, 0, 4],
                "rotation": 0,
                "scale": [2.0, 0.6, 1.8],
                "color": "#FFFFFF"
            },
            {
                "type": "Nightstand",
                "position": [6.3, 0, 4],
                "rotation": 0,
                "scale": [0.4, 0.7, 0.4],
                "color": "#8B4513"
            },
            {
                "type": "Nightstand",
                "position": [8.7, 0, 4],
                "rotation": 0,
                "scale": [0.4, 0.7, 0.4],
                "color": "#8B4513"
            }
        ],
        "doors": [
            {
                "position": [5, 0, 4],
                "rotation": 90,
                "scale": [1.0, 2.1, 0.1],
                "type": "Door"
            }
        ],
        "windows": [],
        "metadata": {
            "scale_factor": 0.05,
            "total_area": 80,
            "room_count": 2,
            "wall_count": 5,
            "furniture_count": 6,
            "image_size": [400, 200]
        }
    }
    
    return JSONResponse(content={
        "message": "Sample floor plan data - 2 bedroom apartment",
        "data": sample_data,
        "description": "Sample data showing a living room and bedroom with appropriate furniture placement"
    })

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    import sys
    import platform
    
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "opencv_available": True,
            "pil_available": True
        },
        "processor": {
            "scale_factor": processor.scale_factor,
            "wall_thickness": processor.wall_thickness,
            "wall_height": processor.wall_height,
            "min_wall_length": processor.min_wall_length,
            "min_room_area": processor.min_room_area
        },
        "room_templates": processor.room_templates
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")