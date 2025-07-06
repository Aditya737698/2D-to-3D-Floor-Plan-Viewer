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

app = FastAPI(title="Floor Plan 3D Converter", version="1.0.0")

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

class FloorPlanProcessor:
    def __init__(self):
        self.scale_factor = 0.1  # 1 pixel = 0.1 meters
        
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """Preprocess the floor plan image"""
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to binary
        _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def detect_walls(self, binary_image: np.ndarray) -> List[Dict]:
        """Detect walls using edge detection and line detection"""
        # Edge detection
        edges = cv2.Canny(binary_image, 50, 150)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=30, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Convert to 3D coordinates
                walls.append({
                    "start": [x1 * self.scale_factor, 0, y1 * self.scale_factor],
                    "end": [x2 * self.scale_factor, 0, y2 * self.scale_factor],
                    "height": 3.0,  # Standard wall height
                    "thickness": 0.2
                })
        
        return walls
    
    def segment_rooms(self, binary_image: np.ndarray) -> List[Dict]:
        """Segment rooms using contour detection"""
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rooms = []
        for i, contour in enumerate(contours):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum room area
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to 3D coordinates
            room = {
                "id": f"room_{i}",
                "bounds": {
                    "x": x * self.scale_factor,
                    "y": y * self.scale_factor,
                    "width": w * self.scale_factor,
                    "height": h * self.scale_factor
                },
                "area": area * (self.scale_factor ** 2),
                "contour": contour.tolist()
            }
            
            # Classify room type based on area and aspect ratio
            room["type"] = self.classify_room(room)
            rooms.append(room)
        
        return rooms
    
    def classify_room(self, room: Dict) -> str:
        """Classify room type based on area and dimensions"""
        area = room["area"]
        width = room["bounds"]["width"]
        height = room["bounds"]["height"]
        aspect_ratio = max(width, height) / min(width, height)
        
        if area < 8:  # Small room
            return "Bathroom"
        elif area < 15:
            return "Bedroom"
        elif area < 25:
            if aspect_ratio > 2:
                return "Corridor"
            else:
                return "Living Room"
        else:  # Large room
            return "Living Room"
    
    def place_furniture(self, room: Dict) -> List[Dict]:
        """Place furniture in room based on type and dimensions"""
        furniture = []
        room_type = room["type"]
        bounds = room["bounds"]
        
        # Room center
        center_x = bounds["x"] + bounds["width"] / 2
        center_z = bounds["y"] + bounds["height"] / 2
        
        if room_type == "Living Room":
            # Sofa
            furniture.append({
                "type": "Sofa",
                "position": [center_x, 0, center_z - bounds["height"] * 0.3],
                "rotation": 0,
                "scale": [2.0, 0.8, 0.9],
                "color": "#8B4513"
            })
            
            # TV
            furniture.append({
                "type": "TV",
                "position": [center_x, 1.2, center_z + bounds["height"] * 0.4],
                "rotation": 180,
                "scale": [1.2, 0.8, 0.1],
                "color": "#000000"
            })
            
            # Coffee Table
            furniture.append({
                "type": "Table",
                "position": [center_x, 0, center_z],
                "rotation": 0,
                "scale": [1.2, 0.4, 0.6],
                "color": "#8B4513"
            })
            
        elif room_type == "Bedroom":
            # Bed
            furniture.append({
                "type": "Bed",
                "position": [center_x, 0, center_z],
                "rotation": 0,
                "scale": [2.0, 0.6, 1.8],
                "color": "#FFFFFF"
            })
            
            # Nightstand
            furniture.append({
                "type": "Nightstand",
                "position": [center_x + 1.2, 0, center_z],
                "rotation": 0,
                "scale": [0.5, 0.7, 0.4],
                "color": "#8B4513"
            })
            
        elif room_type == "Bathroom":
            # Toilet
            furniture.append({
                "type": "Toilet",
                "position": [center_x - bounds["width"] * 0.3, 0, center_z],
                "rotation": 0,
                "scale": [0.6, 0.8, 0.4],
                "color": "#FFFFFF"
            })
            
            # Sink
            furniture.append({
                "type": "Sink",
                "position": [center_x + bounds["width"] * 0.3, 0, center_z],
                "rotation": 0,
                "scale": [0.6, 0.9, 0.4],
                "color": "#FFFFFF"
            })
        
        return furniture
    
    def detect_openings(self, binary_image: np.ndarray) -> tuple:
        """Detect doors and windows"""
        # This is a simplified version - in practice, you'd use more sophisticated detection
        doors = []
        windows = []
        
        # For demo purposes, add some doors based on room connections
        doors.append({
            "position": [5, 0, 3],
            "rotation": 0,
            "scale": [1.0, 2.1, 0.1],
            "type": "Door"
        })
        
        return doors, windows

# Initialize processor
processor = FloorPlanProcessor()

@app.post("/upload/")
async def upload_floor_plan(file: UploadFile = File(...)):
    """Upload and process floor plan image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Process image
        binary_image = processor.preprocess_image(image_array)
        
        # Extract layout components
        walls = processor.detect_walls(binary_image)
        rooms = processor.segment_rooms(binary_image)
        doors, windows = processor.detect_openings(binary_image)
        
        # Generate furniture for each room
        all_furniture = []
        for room in rooms:
            furniture = processor.place_furniture(room)
            all_furniture.extend(furniture)
        
        # Save JSON files
        output_data = {
            "walls": walls,
            "rooms": rooms,
            "doors": doors,
            "windows": windows,
            "furniture": all_furniture
        }
        
        # Save individual files
        with open("static/walls.json", "w") as f:
            json.dump(walls, f, indent=2)
        
        with open("static/rooms.json", "w") as f:
            json.dump(rooms, f, indent=2)
        
        with open("static/furniture.json", "w") as f:
            json.dump(all_furniture, f, indent=2)
        
        with open("static/labels.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        return JSONResponse(content={
            "message": "Floor plan processed successfully",
            "data": output_data,
            "stats": {
                "walls": len(walls),
                "rooms": len(rooms),
                "furniture": len(all_furniture)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Floor Plan 3D Converter API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)