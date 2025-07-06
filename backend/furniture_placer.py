"""
Intelligent furniture placement component with accurate positioning
"""
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FurniturePlacer:
    def __init__(self):
        # Standard furniture dimensions in meters
        self.furniture_catalog = {
            # Living Room
            'sofa': {'width': 2.2, 'depth': 0.9, 'height': 0.8, 'wall_clearance': 0.6},
            'coffee_table': {'width': 1.2, 'depth': 0.6, 'height': 0.4, 'wall_clearance': 0.8},
            'tv_stand': {'width': 1.5, 'depth': 0.4, 'height': 0.6, 'wall_clearance': 0.2},
            'armchair': {'width': 0.8, 'depth': 0.8, 'height': 0.9, 'wall_clearance': 0.5},
            
            # Bedroom
            'queen_bed': {'width': 1.6, 'depth': 2.0, 'height': 0.6, 'wall_clearance': 0.7},
            'king_bed': {'width': 1.9, 'depth': 2.1, 'height': 0.6, 'wall_clearance': 0.7},
            'nightstand': {'width': 0.5, 'depth': 0.4, 'height': 0.7, 'wall_clearance': 0.1},
            'dresser': {'width': 1.5, 'depth': 0.5, 'height': 0.9, 'wall_clearance': 0.3},
            'wardrobe': {'width': 1.2, 'depth': 0.6, 'height': 2.2, 'wall_clearance': 0.2},
            
            # Kitchen
            'refrigerator': {'width': 0.7, 'depth': 0.7, 'height': 1.8, 'wall_clearance': 0.1},
            'stove': {'width': 0.6, 'depth': 0.6, 'height': 0.9, 'wall_clearance': 0.1},
            'dishwasher': {'width': 0.6, 'depth': 0.6, 'height': 0.9, 'wall_clearance': 0.1},
            'kitchen_island': {'width': 1.5, 'depth': 0.9, 'height': 0.9, 'wall_clearance': 1.0},
            
            # Dining
            'dining_table_4': {'width': 1.2, 'depth': 0.8, 'height': 0.8, 'wall_clearance': 1.0},
            'dining_table_6': {'width': 1.8, 'depth': 0.9, 'height': 0.8, 'wall_clearance': 1.0},
            'dining_chair': {'width': 0.5, 'depth': 0.5, 'height': 0.9, 'wall_clearance': 0.4},
            
            # Bathroom
            'bathtub': {'width': 1.7, 'depth': 0.8, 'height': 0.6, 'wall_clearance': 0.1},
            'shower': {'width': 0.9, 'depth': 0.9, 'height': 2.2, 'wall_clearance': 0.1},
            'toilet': {'width': 0.6, 'depth': 0.4, 'height': 0.8, 'wall_clearance': 0.3},
            'vanity': {'width': 1.2, 'depth': 0.5, 'height': 0.9, 'wall_clearance': 0.1},
            'pedestal_sink': {'width': 0.6, 'depth': 0.4, 'height': 0.9, 'wall_clearance': 0.2}
        }
    
    def place_furniture(self, rooms: List[Dict], walls: List[Dict]) -> List[Dict]:
        """Place furniture intelligently in all rooms"""
        logger.info("Starting intelligent furniture placement...")
        
        all_furniture = []
        
        try:
            # Analyze room adjacencies and traffic flow
            room_adjacencies = self._analyze_room_adjacencies(rooms, walls)
            
            for room in rooms:
                # Calculate usable space considering wall clearances
                usable_area = self._calculate_usable_area(room, walls)
                
                # Place furniture based on room type
                room_furniture = self._place_room_furniture(room, usable_area, room_adjacencies)
                all_furniture.extend(room_furniture)
            
            # Validate and adjust furniture placement
            validated_furniture = self._validate_and_adjust_placement(all_furniture, rooms)
            
            logger.info(f"Furniture placement complete: {len(validated_furniture)} items")
            return validated_furniture
            
        except Exception as e:
            logger.error(f"Error in furniture placement: {str(e)}")
            return []
    
    def _analyze_room_adjacencies(self, rooms: List[Dict], walls: List[Dict]) -> Dict:
        """Analyze which rooms are adjacent to each other"""
        adjacencies = {}
        
        try:
            for i, room1 in enumerate(rooms):
                adjacencies[room1['id']] = []
                
                for j, room2 in enumerate(rooms):
                    if i != j and self._rooms_are_adjacent(room1, room2):
                        adjacencies[room1['id']].append(room2['id'])
            
            return adjacencies
            
        except Exception as e:
            logger.error(f"Error analyzing room adjacencies: {str(e)}")
            return {}
    
    def _rooms_are_adjacent(self, room1: Dict, room2: Dict) -> bool:
        """Check if two rooms share a wall"""
        try:
            bounds1 = room1['bounds']
            bounds2 = room2['bounds']
            
            # Check if rooms share a vertical wall
            if (abs(bounds1['x'] + bounds1['width'] - bounds2['x']) < 0.5 or 
                abs(bounds2['x'] + bounds2['width'] - bounds1['x']) < 0.5):
                # Check if there's vertical overlap
                y1_start, y1_end = bounds1['y'], bounds1['y'] + bounds1['height']
                y2_start, y2_end = bounds2['y'], bounds2['y'] + bounds2['height']
                return not (y1_end <= y2_start or y2_end <= y1_start)
            
            # Check if rooms share a horizontal wall
            if (abs(bounds1['y'] + bounds1['height'] - bounds2['y']) < 0.5 or 
                abs(bounds2['y'] + bounds2['height'] - bounds1['y']) < 0.5):
                # Check if there's horizontal overlap
                x1_start, x1_end = bounds1['x'], bounds1['x'] + bounds1['width']
                x2_start, x2_end = bounds2['x'], bounds2['x'] + bounds2['width']
                return not (x1_end <= x2_start or x2_end <= x1_start)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking room adjacency: {str(e)}")
            return False
    
    def _calculate_usable_area(self, room: Dict, walls: List[Dict]) -> Dict:
        """Calculate usable area within a room considering wall clearances"""
        try:
            bounds = room['bounds']
            
            # Standard clearance from walls
            clearance = 0.4
            
            usable_area = {
                'x': bounds['x'] + clearance,
                'y': bounds['y'] + clearance,
                'width': max(bounds['width'] - 2 * clearance, 1.0),
                'height': max(bounds['height'] - 2 * clearance, 1.0),
                'center_x': bounds['x'] + bounds['width'] / 2,
                'center_y': bounds['y'] + bounds['height'] / 2
            }
            
            return usable_area
            
        except Exception as e:
            logger.error(f"Error calculating usable area: {str(e)}")
            return room['bounds'].copy()
    
    def _place_room_furniture(self, room: Dict, usable_area: Dict, adjacencies: Dict) -> List[Dict]:
        """Place furniture in a specific room based on type and layout"""
        try:
            room_type = room['type'].lower()
            area = room['area']
            furniture = []
            
            if 'living' in room_type:
                furniture = self._place_living_room_furniture(usable_area, area)
            elif 'master' in room_type:
                furniture = self._place_master_bedroom_furniture(usable_area, area)
            elif 'bedroom' in room_type:
                furniture = self._place_bedroom_furniture(usable_area, area)
            elif 'kitchen' in room_type:
                furniture = self._place_kitchen_furniture(usable_area, area)
            elif 'dining' in room_type:
                furniture = self._place_dining_room_furniture(usable_area, area)
            elif 'bathroom' in room_type:
                furniture = self._place_bathroom_furniture(usable_area, area)
            elif 'study' in room_type or 'office' in room_type:
                furniture = self._place_study_furniture(usable_area, area)
            elif 'garage' in room_type:
                furniture = self._place_garage_furniture(usable_area, area)
            else:
                furniture = self._place_generic_furniture(usable_area, area)
            
            return furniture
            
        except Exception as e:
            logger.error(f"Error placing furniture in {room.get('type', 'unknown')}: {str(e)}")
            return []
    
    def _place_living_room_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place living room furniture with proper spacing"""
        furniture = []
        
        try:
            # Main seating arrangement
            sofa_dims = self.furniture_catalog['sofa']
            
            # Place sofa against the longest wall
            if usable_area['width'] > usable_area['height']:
                # Horizontal orientation
                sofa_x = usable_area['center_x']
                sofa_y = usable_area['y'] + sofa_dims['depth'] / 2 + 0.2
                sofa_rotation = 0
            else:
                # Vertical orientation
                sofa_x = usable_area['x'] + sofa_dims['depth'] / 2 + 0.2
                sofa_y = usable_area['center_y']
                sofa_rotation = 90
            
            # Ensure sofa fits
            if self._furniture_fits(sofa_x, sofa_y, sofa_dims, usable_area, sofa_rotation):
                furniture.append({
                    "type": "Sofa",
                    "position": [sofa_x, 0, sofa_y],
                    "rotation": sofa_rotation,
                    "scale": [sofa_dims['width'], sofa_dims['height'], sofa_dims['depth']],
                    "color": "#4682B4"
                })
                
                # Coffee table in front of sofa
                coffee_dims = self.furniture_catalog['coffee_table']
                if sofa_rotation == 0:
                    coffee_x = sofa_x
                    coffee_y = sofa_y + sofa_dims['depth'] / 2 + coffee_dims['depth'] / 2 + 0.8
                else:
                    coffee_x = sofa_x + sofa_dims['depth'] / 2 + coffee_dims['depth'] / 2 + 0.8
                    coffee_y = sofa_y
                
                if self._furniture_fits(coffee_x, coffee_y, coffee_dims, usable_area):
                    furniture.append({
                        "type": "Coffee Table",
                        "position": [coffee_x, 0, coffee_y],
                        "rotation": 0,
                        "scale": [coffee_dims['width'], coffee_dims['height'], coffee_dims['depth']],
                        "color": "#8B4513"
                    })
                
                # TV opposite the sofa
                tv_dims = self.furniture_catalog['tv_stand']
                if sofa_rotation == 0:
                    tv_x = sofa_x
                    tv_y = usable_area['y'] + usable_area['height'] - tv_dims['depth'] / 2 - 0.2
                    tv_rotation = 180
                else:
                    tv_x = usable_area['x'] + usable_area['width'] - tv_dims['depth'] / 2 - 0.2
                    tv_y = sofa_y
                    tv_rotation = 270
                
                if self._furniture_fits(tv_x, tv_y, tv_dims, usable_area):
                    furniture.append({
                        "type": "TV Stand",
                        "position": [tv_x, 0, tv_y],
                        "rotation": tv_rotation,
                        "scale": [tv_dims['width'], tv_dims['height'], tv_dims['depth']],
                        "color": "#654321"
                    })
                    
                    # TV on top of stand
                    furniture.append({
                        "type": "TV",
                        "position": [tv_x, tv_dims['height'], tv_y],
                        "rotation": tv_rotation,
                        "scale": [tv_dims['width'] * 0.8, 0.8, 0.1],
                        "color": "#000000"
                    })
            
            # Add armchair if space permits
            if area > 20:
                chair_dims = self.furniture_catalog['armchair']
                chair_x = usable_area['x'] + chair_dims['width'] / 2 + 0.3
                chair_y = usable_area['center_y']
                
                if self._furniture_fits(chair_x, chair_y, chair_dims, usable_area):
                    furniture.append({
                        "type": "Armchair",
                        "position": [chair_x, 0, chair_y],
                        "rotation": 45,
                        "scale": [chair_dims['width'], chair_dims['height'], chair_dims['depth']],
                        "color": "#CD853F"
                    })
            
        except Exception as e:
            logger.error(f"Error placing living room furniture: {str(e)}")
        
        return furniture
    
    def _place_bedroom_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place bedroom furniture with proper positioning"""
        furniture = []
        
        try:
            # Choose bed size based on room area
            if area > 20:
                bed_dims = self.furniture_catalog['king_bed']
                bed_type = "King Bed"
            else:
                bed_dims = self.furniture_catalog['queen_bed']
                bed_type = "Queen Bed"
            
            # Place bed in center, oriented based on room shape
            bed_x = usable_area['center_x']
            bed_y = usable_area['center_y']
            
            # Orient bed based on room proportions
            if usable_area['width'] > usable_area['height']:
                bed_rotation = 0  # Bed width along room width
            else:
                bed_rotation = 90  # Bed width along room height
            
            if self._furniture_fits(bed_x, bed_y, bed_dims, usable_area, bed_rotation):
                furniture.append({
                    "type": bed_type,
                    "position": [bed_x, 0, bed_y],
                    "rotation": bed_rotation,
                    "scale": [bed_dims['width'], bed_dims['height'], bed_dims['depth']],
                    "color": "#FFFFFF"
                })
                
                # Place nightstands on both sides
                nightstand_dims = self.furniture_catalog['nightstand']
                
                if bed_rotation == 0:
                    # Nightstands on left and right
                    ns_left_x = bed_x - bed_dims['width'] / 2 - nightstand_dims['width'] / 2 - 0.1
                    ns_right_x = bed_x + bed_dims['width'] / 2 + nightstand_dims['width'] / 2 + 0.1
                    ns_y = bed_y
                else:
                    # Nightstands on top and bottom
                    ns_left_x = bed_x
                    ns_right_x = bed_x
                    ns_y_1 = bed_y - bed_dims['width'] / 2 - nightstand_dims['depth'] / 2 - 0.1
                    ns_y_2 = bed_y + bed_dims['width'] / 2 + nightstand_dims['depth'] / 2 + 0.1
                
                if bed_rotation == 0:
                    positions = [(ns_left_x, ns_y), (ns_right_x, ns_y)]
                else:
                    positions = [(ns_left_x, ns_y_1), (ns_right_x, ns_y_2)]
                
                for i, (ns_x, ns_y_pos) in enumerate(positions):
                    if self._furniture_fits(ns_x, ns_y_pos, nightstand_dims, usable_area):
                        furniture.append({
                            "type": "Nightstand",
                            "position": [ns_x, 0, ns_y_pos],
                            "rotation": 0,
                            "scale": [nightstand_dims['width'], nightstand_dims['height'], nightstand_dims['depth']],
                            "color": "#8B4513"
                        })
            
            # Add dresser against a wall
            if area > 15:
                dresser_dims = self.furniture_catalog['dresser']
                dresser_x = usable_area['x'] + dresser_dims['depth'] / 2 + 0.2
                dresser_y = usable_area['y'] + usable_area['height'] - dresser_dims['width'] / 2 - 0.3
                
                if self._furniture_fits(dresser_x, dresser_y, dresser_dims, usable_area):
                    furniture.append({
                        "type": "Dresser",
                        "position": [dresser_x, 0, dresser_y],
                        "rotation": 90,
                        "scale": [dresser_dims['width'], dresser_dims['height'], dresser_dims['depth']],
                        "color": "#654321"
                    })
            
        except Exception as e:
            logger.error(f"Error placing bedroom furniture: {str(e)}")
        
        return furniture
    
    def _place_master_bedroom_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place master bedroom furniture with luxury items"""
        furniture = self._place_bedroom_furniture(usable_area, area)
        
        try:
            # Add seating area if room is large enough
            if area > 25:
                chair_dims = self.furniture_catalog['armchair']
                chair_x = usable_area['x'] + usable_area['width'] - chair_dims['width'] / 2 - 0.3
                chair_y = usable_area['y'] + chair_dims['depth'] / 2 + 0.3
                
                if self._furniture_fits(chair_x, chair_y, chair_dims, usable_area):
                    furniture.append({
                        "type": "Reading Chair",
                        "position": [chair_x, 0, chair_y],
                        "rotation": 225,
                        "scale": [chair_dims['width'], chair_dims['height'], chair_dims['depth']],
                        "color": "#8B4513"
                    })
            
            # Add walk-in closet area if very large
            if area > 30:
                wardrobe_dims = self.furniture_catalog['wardrobe']
                wardrobe_x = usable_area['x'] + wardrobe_dims['depth'] / 2 + 0.2
                wardrobe_y = usable_area['y'] + wardrobe_dims['width'] / 2 + 0.3
                
                if self._furniture_fits(wardrobe_x, wardrobe_y, wardrobe_dims, usable_area):
                    furniture.append({
                        "type": "Walk-in Closet",
                        "position": [wardrobe_x, 0, wardrobe_y],
                        "rotation": 90,
                        "scale": [wardrobe_dims['width'], wardrobe_dims['height'], wardrobe_dims['depth']],
                        "color": "#DEB887"
                    })
            
        except Exception as e:
            logger.error(f"Error placing master bedroom furniture: {str(e)}")
        
        return furniture
    
    def _place_kitchen_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place kitchen furniture with proper work triangle"""
        furniture = []
        
        try:
            # Determine kitchen layout based on room shape
            aspect_ratio = usable_area['width'] / usable_area['height']
            
            if aspect_ratio > 2.0:
                # Galley kitchen
                furniture.extend(self._place_galley_kitchen(usable_area))
            elif aspect_ratio < 0.5:
                # Corridor kitchen
                furniture.extend(self._place_corridor_kitchen(usable_area))
            else:
                # L-shaped or linear kitchen
                furniture.extend(self._place_linear_kitchen(usable_area))
            
            # Add island if space permits
            if area > 15:
                island_dims = self.furniture_catalog['kitchen_island']
                island_x = usable_area['center_x']
                island_y = usable_area['center_y']
                
                if self._furniture_fits(island_x, island_y, island_dims, usable_area):
                    furniture.append({
                        "type": "Kitchen Island",
                        "position": [island_x, 0, island_y],
                        "rotation": 0,
                        "scale": [island_dims['width'], island_dims['height'], island_dims['depth']],
                        "color": "#F5F5DC"
                    })
            
        except Exception as e:
            logger.error(f"Error placing kitchen furniture: {str(e)}")
        
        return furniture
    
    def _place_galley_kitchen(self, usable_area: Dict) -> List[Dict]:
        """Place furniture for galley kitchen layout"""
        furniture = []
        
        # Counter on one side
        counter_length = usable_area['width'] * 0.8
        counter_y1 = usable_area['y'] + 0.3
        counter_y2 = usable_area['y'] + usable_area['height'] - 0.3
        
        # Appliances on both sides
        fridge_dims = self.furniture_catalog['refrigerator']
        stove_dims = self.furniture_catalog['stove']
        
        # Side 1: Refrigerator and counter
        furniture.extend([
            {
                "type": "Refrigerator",
                "position": [usable_area['x'] + 0.4, 0, counter_y1],
                "rotation": 0,
                "scale": [fridge_dims['width'], fridge_dims['height'], fridge_dims['depth']],
                "color": "#F5F5F5"
            },
            {
                "type": "Kitchen Counter",
                "position": [usable_area['center_x'], 0, counter_y1],
                "rotation": 0,
                "scale": [counter_length, 0.9, 0.6],
                "color": "#DEB887"
            }
        ])
        
        # Side 2: Stove and counter
        furniture.extend([
            {
                "type": "Stove",
                "position": [usable_area['center_x'], 0, counter_y2],
                "rotation": 0,
                "scale": [stove_dims['width'], stove_dims['height'], stove_dims['depth']],
                "color": "#2F4F4F"
            },
            {
                "type": "Kitchen Counter",
                "position": [usable_area['center_x'] + 1.0, 0, counter_y2],
                "rotation": 0,
                "scale": [counter_length - 1.0, 0.9, 0.6],
                "color": "#DEB887"
            }
        ])
        
        return furniture
    
    def _place_corridor_kitchen(self, usable_area: Dict) -> List[Dict]:
        """Place furniture for corridor kitchen layout"""
        furniture = []
        
        # Single wall layout
        counter_length = usable_area['height'] * 0.8
        
        fridge_dims = self.furniture_catalog['refrigerator']
        stove_dims = self.furniture_catalog['stove']
        
        furniture.extend([
            {
                "type": "Kitchen Counter",
                "position": [usable_area['x'] + 0.3, 0, usable_area['center_y']],
                "rotation": 90,
                "scale": [counter_length, 0.9, 0.6],
                "color": "#DEB887"
            },
            {
                "type": "Refrigerator",
                "position": [usable_area['x'] + 0.3, 0, usable_area['y'] + 0.4],
                "rotation": 0,
                "scale": [fridge_dims['width'], fridge_dims['height'], fridge_dims['depth']],
                "color": "#F5F5F5"
            },
            {
                "type": "Stove",
                "position": [usable_area['x'] + 0.3, 0, usable_area['center_y']],
                "rotation": 0,
                "scale": [stove_dims['width'], stove_dims['height'], stove_dims['depth']],
                "color": "#2F4F4F"
            }
        ])
        
        return furniture
    
    def _place_linear_kitchen(self, usable_area: Dict) -> List[Dict]:
        """Place furniture for linear kitchen layout"""
        furniture = []
        
        # Linear counter along one wall
        counter_length = usable_area['width'] * 0.8
        
        fridge_dims = self.furniture_catalog['refrigerator']
        stove_dims = self.furniture_catalog['stove']
        
        furniture.extend([
            {
                "type": "Kitchen Counter",
                "position": [usable_area['center_x'], 0, usable_area['y'] + 0.3],
                "rotation": 0,
                "scale": [counter_length, 0.9, 0.6],
                "color": "#DEB887"
            },
            {
                "type": "Refrigerator",
                "position": [usable_area['x'] + 0.4, 0, usable_area['y'] + 0.3],
                "rotation": 0,
                "scale": [fridge_dims['width'], fridge_dims['height'], fridge_dims['depth']],
                "color": "#F5F5F5"
            },
            {
                "type": "Stove",
                "position": [usable_area['center_x'], 0, usable_area['y'] + 0.3],
                "rotation": 0,
                "scale": [stove_dims['width'], stove_dims['height'], stove_dims['depth']],
                "color": "#2F4F4F"
            }
        ])
        
        return furniture
    
    def _place_dining_room_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place dining room furniture"""
        furniture = []
        
        try:
            # Choose table size based on area
            if area > 20:
                table_dims = self.furniture_catalog['dining_table_6']
                table_type = "Dining Table (6-seat)"
                chair_count = 6
            else:
                table_dims = self.furniture_catalog['dining_table_4']
                table_type = "Dining Table (4-seat)"
                chair_count = 4
            
            # Place table in center
            table_x = usable_area['center_x']
            table_y = usable_area['center_y']
            
            if self._furniture_fits(table_x, table_y, table_dims, usable_area):
                furniture.append({
                    "type": table_type,
                    "position": [table_x, 0, table_y],
                    "rotation": 0,
                    "scale": [table_dims['width'], table_dims['height'], table_dims['depth']],
                    "color": "#8B4513"
                })
                
                # Place chairs around table
                chair_dims = self.furniture_catalog['dining_chair']
                chair_spacing = 0.6
                
                # Calculate chair positions
                chair_positions = []
                
                if chair_count == 4:
                    chair_positions = [
                        (table_x - table_dims['width']/2 - chair_spacing, table_y, 90),  # Left
                        (table_x + table_dims['width']/2 + chair_spacing, table_y, 270), # Right
                        (table_x, table_y - table_dims['depth']/2 - chair_spacing, 0),   # Front
                        (table_x, table_y + table_dims['depth']/2 + chair_spacing, 180)  # Back
                    ]
                else:  # 6 chairs
                    chair_positions = [
                        (table_x - table_dims['width']/2 - chair_spacing, table_y - table_dims['depth']/4, 90),
                        (table_x - table_dims['width']/2 - chair_spacing, table_y + table_dims['depth']/4, 90),
                        (table_x + table_dims['width']/2 + chair_spacing, table_y - table_dims['depth']/4, 270),
                        (table_x + table_dims['width']/2 + chair_spacing, table_y + table_dims['depth']/4, 270),
                        (table_x, table_y - table_dims['depth']/2 - chair_spacing, 0),
                        (table_x, table_y + table_dims['depth']/2 + chair_spacing, 180)
                    ]
                
                for i, (chair_x, chair_y, rotation) in enumerate(chair_positions):
                    if self._furniture_fits(chair_x, chair_y, chair_dims, usable_area):
                        furniture.append({
                            "type": "Dining Chair",
                            "position": [chair_x, 0, chair_y],
                            "rotation": rotation,
                            "scale": [chair_dims['width'], chair_dims['height'], chair_dims['depth']],
                            "color": "#654321"
                        })
            
        except Exception as e:
            logger.error(f"Error placing dining room furniture: {str(e)}")
        
        return furniture
    
    def _place_bathroom_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place bathroom furniture based on size"""
        furniture = []
        
        try:
            if area > 8:  # Large bathroom
                furniture.extend(self._place_large_bathroom_furniture(usable_area))
            elif area > 4:  # Medium bathroom
                furniture.extend(self._place_medium_bathroom_furniture(usable_area))
            else:  # Small bathroom
                furniture.extend(self._place_small_bathroom_furniture(usable_area))
            
        except Exception as e:
            logger.error(f"Error placing bathroom furniture: {str(e)}")
        
        return furniture
    
    def _place_large_bathroom_furniture(self, usable_area: Dict) -> List[Dict]:
        """Place furniture for large bathroom"""
        furniture = []
        
        bathtub_dims = self.furniture_catalog['bathtub']
        vanity_dims = self.furniture_catalog['vanity']
        toilet_dims = self.furniture_catalog['toilet']
        shower_dims = self.furniture_catalog['shower']
        
        # Bathtub along one wall
        bathtub_x = usable_area['x'] + usable_area['width'] - bathtub_dims['width']/2 - 0.1
        bathtub_y = usable_area['y'] + bathtub_dims['depth']/2 + 0.1
        
        furniture.extend([
            {
                "type": "Bathtub",
                "position": [bathtub_x, 0, bathtub_y],
                "rotation": 0,
                "scale": [bathtub_dims['width'], bathtub_dims['height'], bathtub_dims['depth']],
                "color": "#FFFFFF"
            },
            {
                "type": "Shower",
                "position": [usable_area['x'] + shower_dims['width']/2 + 0.1, 0, usable_area['y'] + shower_dims['depth']/2 + 0.1],
                "rotation": 0,
                "scale": [shower_dims['width'], shower_dims['height'], shower_dims['depth']],
                "color": "#F8F8FF"
            },
            {
                "type": "Double Vanity",
                "position": [usable_area['center_x'], 0, usable_area['y'] + usable_area['height'] - vanity_dims['depth']/2 - 0.1],
                "rotation": 0,
                "scale": [vanity_dims['width'] * 1.5, vanity_dims['height'], vanity_dims['depth']],
                "color": "#DEB887"
            },
            {
                "type": "Toilet",
                "position": [usable_area['center_x'], 0, usable_area['center_y']],
                "rotation": 0,
                "scale": [toilet_dims['width'], toilet_dims['height'], toilet_dims['depth']],
                "color": "#FFFFFF"
            }
        ])
        
        return furniture
    
    def _place_medium_bathroom_furniture(self, usable_area: Dict) -> List[Dict]:
        """Place furniture for medium bathroom"""
        furniture = []
        
        bathtub_dims = self.furniture_catalog['bathtub']
        vanity_dims = self.furniture_catalog['vanity']
        toilet_dims = self.furniture_catalog['toilet']
        
        furniture.extend([
            {
                "type": "Bathtub",
                "position": [usable_area['center_x'], 0, usable_area['y'] + bathtub_dims['depth']/2 + 0.1],
                "rotation": 0,
                "scale": [bathtub_dims['width'], bathtub_dims['height'], bathtub_dims['depth']],
                "color": "#FFFFFF"
            },
            {
                "type": "Vanity",
                "position": [usable_area['x'] + vanity_dims['depth']/2 + 0.1, 0, usable_area['y'] + usable_area['height'] - vanity_dims['width']/2 - 0.1],
                "rotation": 90,
                "scale": [vanity_dims['width'], vanity_dims['height'], vanity_dims['depth']],
                "color": "#DEB887"
            },
            {
                "type": "Toilet",
                "position": [usable_area['x'] + usable_area['width'] - toilet_dims['width']/2 - 0.1, 0, usable_area['center_y']],
                "rotation": 0,
                "scale": [toilet_dims['width'], toilet_dims['height'], toilet_dims['depth']],
                "color": "#FFFFFF"
            }
        ])
        
        return furniture
    
    def _place_small_bathroom_furniture(self, usable_area: Dict) -> List[Dict]:
        """Place furniture for small bathroom"""
        furniture = []
        
        shower_dims = self.furniture_catalog['shower']
        pedestal_dims = self.furniture_catalog['pedestal_sink']
        toilet_dims = self.furniture_catalog['toilet']
        
        furniture.extend([
            {
                "type": "Shower",
                "position": [usable_area['x'] + shower_dims['width']/2 + 0.1, 0, usable_area['y'] + shower_dims['depth']/2 + 0.1],
                "rotation": 0,
                "scale": [shower_dims['width'], shower_dims['height'], shower_dims['depth']],
                "color": "#F8F8FF"
            },
            {
                "type": "Pedestal Sink",
                "position": [usable_area['x'] + usable_area['width'] - pedestal_dims['width']/2 - 0.1, 0, usable_area['y'] + pedestal_dims['depth']/2 + 0.1],
                "rotation": 0,
                "scale": [pedestal_dims['width'], pedestal_dims['height'], pedestal_dims['depth']],
                "color": "#FFFFFF"
            },
            {
                "type": "Toilet",
                "position": [usable_area['center_x'], 0, usable_area['y'] + usable_area['height'] - toilet_dims['depth']/2 - 0.1],
                "rotation": 0,
                "scale": [toilet_dims['width'], toilet_dims['height'], toilet_dims['depth']],
                "color": "#FFFFFF"
            }
        ])
        
        return furniture
    
    def _place_study_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place study/office furniture"""
        furniture = []
        
        try:
            # Desk against wall
            desk_x = usable_area['x'] + usable_area['width'] - 0.3
            desk_y = usable_area['center_y']
            
            furniture.extend([
                {
                    "type": "Desk",
                    "position": [desk_x, 0, desk_y],
                    "rotation": 90,
                    "scale": [1.2, 0.8, 0.6],
                    "color": "#8B4513"
                },
                {
                    "type": "Office Chair",
                    "position": [desk_x - 0.6, 0, desk_y],
                    "rotation": 270,
                    "scale": [0.6, 0.9, 0.6],
                    "color": "#000000"
                },
                {
                    "type": "Bookshelf",
                    "position": [usable_area['x'] + 0.2, 0, usable_area['y'] + 1.0],
                    "rotation": 0,
                    "scale": [2.0, 2.2, 0.3],
                    "color": "#654321"
                }
            ])
            
        except Exception as e:
            logger.error(f"Error placing study furniture: {str(e)}")
        
        return furniture
    
    def _place_garage_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place garage furniture and vehicles"""
        furniture = []
        
        try:
            # Cars
            if area > 25:  # Two-car garage
                furniture.extend([
                    {
                        "type": "Car",
                        "position": [usable_area['x'] + usable_area['width'] * 0.25, 0, usable_area['center_y']],
                        "rotation": 0,
                        "scale": [1.8, 1.5, 4.5],
                        "color": "#4169E1"
                    },
                    {
                        "type": "Car",
                        "position": [usable_area['x'] + usable_area['width'] * 0.75, 0, usable_area['center_y']],
                        "rotation": 0,
                        "scale": [1.8, 1.5, 4.5],
                        "color": "#DC143C"
                    }
                ])
            else:  # Single-car garage
                furniture.append({
                    "type": "Car",
                    "position": [usable_area['center_x'], 0, usable_area['center_y']],
                    "rotation": 0,
                    "scale": [1.8, 1.5, 4.5],
                    "color": "#4169E1"
                })
            
            # Storage shelving
            furniture.append({
                "type": "Storage Shelves",
                "position": [usable_area['x'] + 0.3, 0, usable_area['y'] + usable_area['height'] - 0.3],
                "rotation": 0,
                "scale": [2.0, 2.0, 0.4],
                "color": "#708090"
            })
            
        except Exception as e:
            logger.error(f"Error placing garage furniture: {str(e)}")
        
        return furniture
    
    def _place_generic_furniture(self, usable_area: Dict, area: float) -> List[Dict]:
        """Place generic furniture for unspecified room types"""
        furniture = []
        
        try:
            # Simple table and chairs
            table_x = usable_area['center_x']
            table_y = usable_area['center_y']
            
            furniture.append({
                "type": "Table",
                "position": [table_x, 0, table_y],
                "rotation": 0,
                "scale": [1.2, 0.8, 0.8],
                "color": "#8B4513"
            })
            
            # Add chairs around table
            chair_positions = [
                (table_x - 0.8, table_y, 90),
                (table_x + 0.8, table_y, 270),
                (table_x, table_y - 0.6, 0),
                (table_x, table_y + 0.6, 180)
            ]
            
            for chair_x, chair_y, rotation in chair_positions:
                if self._point_in_usable_area(chair_x, chair_y, usable_area):
                    furniture.append({
                        "type": "Chair",
                        "position": [chair_x, 0, chair_y],
                        "rotation": rotation,
                        "scale": [0.5, 0.9, 0.5],
                        "color": "#654321"
                    })
            
        except Exception as e:
            logger.error(f"Error placing generic furniture: {str(e)}")
        
        return furniture
    
    def _furniture_fits(self, x: float, y: float, dims: Dict, usable_area: Dict, rotation: int = 0) -> bool:
        """Check if furniture fits in the usable area"""
        try:
            # Adjust dimensions for rotation
            if rotation == 90 or rotation == 270:
                width, depth = dims['depth'], dims['width']
            else:
                width, depth = dims['width'], dims['depth']
            
            # Check boundaries
            left = x - width / 2
            right = x + width / 2
            top = y - depth / 2
            bottom = y + depth / 2
            
            return (left >= usable_area['x'] and 
                    right <= usable_area['x'] + usable_area['width'] and
                    top >= usable_area['y'] and 
                    bottom <= usable_area['y'] + usable_area['height'])
            
        except Exception as e:
            logger.error(f"Error checking furniture fit: {str(e)}")
            return False
    
    def _point_in_usable_area(self, x: float, y: float, usable_area: Dict) -> bool:
        """Check if a point is within the usable area"""
        return (usable_area['x'] <= x <= usable_area['x'] + usable_area['width'] and
                usable_area['y'] <= y <= usable_area['y'] + usable_area['height'])
    
    def _validate_and_adjust_placement(self, furniture: List[Dict], rooms: List[Dict]) -> List[Dict]:
        """Validate furniture placement and remove overlapping items"""
        if not furniture:
            return furniture
        
        try:
            validated_furniture = []
            
            for item in furniture:
                is_valid = True
                
                # Check for overlaps with existing furniture
                for existing_item in validated_furniture:
                    if self._furniture_overlaps(item, existing_item):
                        is_valid = False
                        logger.debug(f"Removed overlapping furniture: {item['type']}")
                        break
                
                # Check if furniture is within room bounds
                if is_valid and not self._furniture_in_rooms(item, rooms):
                    is_valid = False
                    logger.debug(f"Removed out-of-bounds furniture: {item['type']}")
                
                if is_valid:
                    validated_furniture.append(item)
            
            logger.debug(f"Furniture validation: {len(furniture)} -> {len(validated_furniture)} items")
            return validated_furniture
            
        except Exception as e:
            logger.error(f"Error validating furniture placement: {str(e)}")
            return furniture
    
    def _furniture_overlaps(self, item1: Dict, item2: Dict, threshold: float = 0.3) -> bool:
        """Check if two furniture items overlap"""
        try:
            pos1 = item1["position"]
            scale1 = item1["scale"]
            pos2 = item2["position"]
            scale2 = item2["scale"]
            
            # Calculate bounding boxes
            bbox1 = {
                "x_min": pos1[0] - scale1[0] / 2,
                "x_max": pos1[0] + scale1[0] / 2,
                "z_min": pos1[2] - scale1[2] / 2,
                "z_max": pos1[2] + scale1[2] / 2
            }
            
            bbox2 = {
                "x_min": pos2[0] - scale2[0] / 2,
                "x_max": pos2[0] + scale2[0] / 2,
                "z_min": pos2[2] - scale2[2] / 2,
                "z_max": pos2[2] + scale2[2] / 2
            }
            
            # Check for overlap
            overlap_x = max(0, min(bbox1["x_max"], bbox2["x_max"]) - max(bbox1["x_min"], bbox2["x_min"]))
            overlap_z = max(0, min(bbox1["z_max"], bbox2["z_max"]) - max(bbox1["z_min"], bbox2["z_min"]))
            
            return overlap_x > threshold and overlap_z > threshold
            
        except Exception as e:
            logger.error(f"Error checking furniture overlap: {str(e)}")
            return False
    
    def _furniture_in_rooms(self, item: Dict, rooms: List[Dict]) -> bool:
        """Check if furniture is within any room bounds"""
        try:
            pos = item["position"]
            
            for room in rooms:
                bounds = room["bounds"]
                if (bounds["x"] <= pos[0] <= bounds["x"] + bounds["width"] and
                    bounds["y"] <= pos[2] <= bounds["y"] + bounds["height"]):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking furniture room bounds: {str(e)}")
            return True  # Default to valid if error occurs