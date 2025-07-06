import * as THREE from 'three';

export class RoomBuilder {
  constructor(materials) {
    this.materials = materials;
  }
  
  createRoom(room) {
    if (!room.bounds) {
      console.warn('Invalid room data:', room);
      return new THREE.Group();
    }
    
    const roomGroup = new THREE.Group();
    const bounds = room.bounds;
    
    // Validate bounds
    if (!bounds.width || !bounds.height || bounds.width <= 0 || bounds.height <= 0) {
      console.warn('Invalid room bounds:', bounds);
      return roomGroup;
    }
    
    // Create room floor
    const floorGeometry = new THREE.PlaneGeometry(bounds.width, bounds.height);
    const roomType = this.normalizeRoomType(room.type);
    const floorMaterial = this.materials.getFloorMaterial(roomType);
    
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(bounds.x + bounds.width/2, 0, bounds.y + bounds.height/2);
    floor.receiveShadow = true;
    
    roomGroup.add(floor);
    
    // Add room border
    this.addRoomBorder(roomGroup, bounds);
    
    // Add room label
    this.addRoomLabel(roomGroup, room);
    
    // Add room-specific details
    this.addRoomDetails(roomGroup, room);
    
    roomGroup.userData = { roomType: roomType, roomId: room.id };
    
    return roomGroup;
  }
  
  normalizeRoomType(type) {
    if (!type) return 'default';
    return type.toLowerCase().replace(/\s+/g, '_');
  }
  
  addRoomBorder(roomGroup, bounds) {
    const borderThickness = 0.02;
    const borderHeight = 0.005;
    
    // Create border segments
    const segments = [
      // Top
      { width: bounds.width, height: borderThickness, x: bounds.x + bounds.width/2, z: bounds.y + bounds.height },
      // Bottom
      { width: bounds.width, height: borderThickness, x: bounds.x + bounds.width/2, z: bounds.y },
      // Left
      { width: borderThickness, height: bounds.height, x: bounds.x, z: bounds.y + bounds.height/2 },
      // Right
      { width: borderThickness, height: bounds.height, x: bounds.x + bounds.width, z: bounds.y + bounds.height/2 }
    ];
    
    segments.forEach(segment => {
      const borderGeometry = new THREE.PlaneGeometry(segment.width, segment.height);
      const border = new THREE.Mesh(borderGeometry, this.materials.furniture.wood);
      border.rotation.x = -Math.PI / 2;
      border.position.set(segment.x, borderHeight, segment.z);
      roomGroup.add(border);
    });
  }
  
  addRoomLabel(roomGroup, room) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 128;
    
    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Create gradient background
    const gradient = context.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, 'rgba(255,255,255,0.9)');
    gradient.addColorStop(1, 'rgba(240,240,240,0.9)');
    
    context.fillStyle = gradient;
    context.fillRect(10, 10, canvas.width - 20, canvas.height - 20);
    
    // Add border
    context.strokeStyle = '#cccccc';
    context.lineWidth = 2;
    context.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
    
    // Add text
    context.font = 'bold 36px Arial';
    context.fillStyle = '#333333';
    context.textAlign = 'center';
    const roomName = room.type || 'Room';
    context.fillText(roomName, canvas.width/2, canvas.height/2 + 5);
    
    // Add area info if available
    if (room.area && room.area > 0) {
      context.font = '24px Arial';
      context.fillStyle = '#666666';
      const area = room.area.toFixed(1) + ' m²';
      context.fillText(area, canvas.width/2, canvas.height/2 + 35);
    }
    
    const texture = new THREE.CanvasTexture(canvas);
    const labelMaterial = new THREE.MeshBasicMaterial({ 
      map: texture, 
      transparent: true,
      alphaTest: 0.1
    });
    
    const labelSize = Math.min(room.bounds.width, room.bounds.height) * 0.5;
    const labelGeometry = new THREE.PlaneGeometry(labelSize, labelSize * 0.25);
    const label = new THREE.Mesh(labelGeometry, labelMaterial);
    label.rotation.x = -Math.PI / 2;
    label.position.set(
      room.bounds.x + room.bounds.width/2, 
      0.02, 
      room.bounds.y + room.bounds.height/2
    );
    
    roomGroup.add(label);
  }
  
  addRoomDetails(roomGroup, room) {
    const roomType = this.normalizeRoomType(room.type);
    const bounds = room.bounds;
    
    // Add room-specific architectural details
    switch (roomType) {
      case 'bathroom':
        this.addBathroomDetails(roomGroup, bounds);
        break;
      case 'kitchen':
        this.addKitchenDetails(roomGroup, bounds);
        break;
      case 'living_room':
      case 'livingroom':
        this.addLivingRoomDetails(roomGroup, bounds);
        break;
      case 'bedroom':
        this.addBedroomDetails(roomGroup, bounds);
        break;
      default:
        this.addDefaultRoomDetails(roomGroup, bounds);
    }
  }
  
  addBathroomDetails(roomGroup, bounds) {
    // Add tile pattern
    const tileSize = 0.3;
    const tilesX = Math.floor(bounds.width / tileSize);
    const tilesZ = Math.floor(bounds.height / tileSize);
    
    for (let x = 0; x < tilesX; x++) {
      for (let z = 0; z < tilesZ; z++) {
        if ((x + z) % 2 === 0) {
          const tileGeometry = new THREE.PlaneGeometry(tileSize * 0.9, tileSize * 0.9);
          const tile = new THREE.Mesh(tileGeometry, this.materials.furniture.white);
          tile.rotation.x = -Math.PI / 2;
          tile.position.set(
            bounds.x + x * tileSize + tileSize/2,
            0.005,
            bounds.y + z * tileSize + tileSize/2
          );
          roomGroup.add(tile);
        }
      }
    }
    
    // Add drain in center
    const drainGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.02, 8);
    const drain = new THREE.Mesh(drainGeometry, this.materials.furniture.metal);
    drain.position.set(
      bounds.x + bounds.width/2,
      0.01,
      bounds.y + bounds.height/2
    );
    roomGroup.add(drain);
  }
  
  addKitchenDetails(roomGroup, bounds) {
    // Add backsplash along one wall
    const backsplashHeight = 0.6;
    const backsplashGeometry = new THREE.PlaneGeometry(bounds.width * 0.8, backsplashHeight);
    const backsplash = new THREE.Mesh(backsplashGeometry, this.materials.furniture.white);
    backsplash.position.set(
      bounds.x + bounds.width/2,
      backsplashHeight/2,
      bounds.y + 0.05
    );
    roomGroup.add(backsplash);
    
    // Add kitchen island outline if room is large enough
    if (bounds.width > 4 && bounds.height > 4) {
      const islandWidth = bounds.width * 0.3;
      const islandHeight = bounds.height * 0.2;
      
      const islandGeometry = new THREE.PlaneGeometry(islandWidth, islandHeight);
      const island = new THREE.Mesh(islandGeometry, this.materials.furniture.wood);
      island.rotation.x = -Math.PI / 2;
      island.position.set(
        bounds.x + bounds.width/2,
        0.01,
        bounds.y + bounds.height/2
      );
      roomGroup.add(island);
    }
  }
  
  addLivingRoomDetails(roomGroup, bounds) {
    // Add hardwood plank pattern
    const plankWidth = 0.15;
    const plankLength = 1.2;
    const planksX = Math.floor(bounds.width / plankLength);
    const planksZ = Math.floor(bounds.height / plankWidth);
    
    for (let x = 0; x < planksX; x++) {
      for (let z = 0; z < planksZ; z++) {
        // Offset every other row for realistic look
        const offsetX = (z % 2) * plankLength * 0.5;
        
        const plankGeometry = new THREE.PlaneGeometry(plankLength * 0.95, plankWidth * 0.9);
        const plank = new THREE.Mesh(plankGeometry, this.materials.furniture.wood);
        plank.rotation.x = -Math.PI / 2;
        plank.position.set(
          bounds.x + x * plankLength + plankLength/2 + offsetX,
          0.003,
          bounds.y + z * plankWidth + plankWidth/2
        );
        roomGroup.add(plank);
      }
    }
    
    // Add area rug in center
    if (bounds.width > 3 && bounds.height > 3) {
      const rugWidth = bounds.width * 0.6;
      const rugHeight = bounds.height * 0.6;
      
      const rugGeometry = new THREE.PlaneGeometry(rugWidth, rugHeight);
      const rug = new THREE.Mesh(rugGeometry, this.materials.furniture.fabric);
      rug.rotation.x = -Math.PI / 2;
      rug.position.set(
        bounds.x + bounds.width/2,
        0.006,
        bounds.y + bounds.height/2
      );
      roomGroup.add(rug);
    }
  }
  
  addBedroomDetails(roomGroup, bounds) {
    // Add carpet/rug
    const carpetWidth = bounds.width * 0.8;
    const carpetHeight = bounds.height * 0.8;
    
    const carpetGeometry = new THREE.PlaneGeometry(carpetWidth, carpetHeight);
    const carpet = new THREE.Mesh(carpetGeometry, this.materials.furniture.fabric);
    carpet.rotation.x = -Math.PI / 2;
    carpet.position.set(
      bounds.x + bounds.width/2,
      0.004,
      bounds.y + bounds.height/2
    );
    roomGroup.add(carpet);
  }
  
  addDefaultRoomDetails(roomGroup, bounds) {
    // Add simple border pattern
    const borderWidth = 0.1;
    const borderGeometry = new THREE.RingGeometry(
      Math.min(bounds.width, bounds.height) * 0.4,
      Math.min(bounds.width, bounds.height) * 0.4 + borderWidth,
      16
    );
    const border = new THREE.Mesh(borderGeometry, this.materials.furniture.wood);
    border.rotation.x = -Math.PI / 2;
    border.position.set(
      bounds.x + bounds.width/2,
      0.002,
      bounds.y + bounds.height/2
    );
    roomGroup.add(border);
  }
  
  // Calculate room center point
  getRoomCenter(room) {
    const bounds = room.bounds;
    return {
      x: bounds.x + bounds.width / 2,
      z: bounds.y + bounds.height / 2
    };
  }
  
  // Check if a point is inside a room
  isPointInRoom(room, x, z) {
    const bounds = room.bounds;
    return x >= bounds.x && 
           x <= bounds.x + bounds.width &&
           z >= bounds.y && 
           z <= bounds.y + bounds.height;
  }
  
  // Get room area
  getRoomArea(room) {
    const bounds = room.bounds;
    return bounds.width * bounds.height;
  }
}