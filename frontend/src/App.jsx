import React, { useState, useEffect, useRef } from 'react';
import { Upload, Home, Eye, RotateCcw, ZoomIn, ZoomOut, Settings } from 'lucide-react';
import * as THREE from 'three';

// 3D Scene Manager Class with Enhanced Components
class FloorPlan3DScene {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    
    // Scene groups for better organization
    this.wallGroup = new THREE.Group();
    this.roomGroup = new THREE.Group();
    this.furnitureGroup = new THREE.Group();
    this.doorGroup = new THREE.Group();
    this.windowGroup = new THREE.Group();
    
    // Materials library
    this.materials = this.createMaterials();
    
    this.init();
  }
  
  init() {
    // Setup renderer
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setClearColor(0xf8f9fa);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.gammaOutput = true;
    this.renderer.gammaFactor = 2.2;
    this.container.appendChild(this.renderer.domElement);
    
    // Setup camera
    this.camera.position.set(20, 25, 20);
    this.camera.lookAt(0, 0, 0);
    
    // Add scene groups
    this.scene.add(this.wallGroup);
    this.scene.add(this.roomGroup);
    this.scene.add(this.furnitureGroup);
    this.scene.add(this.doorGroup);
    this.scene.add(this.windowGroup);
    
    // Setup lighting
    this.setupAdvancedLighting();
    
    // Setup controls
    this.setupControls();
    
    // Add environment
    this.addEnvironment();
    
    // Start render loop
    this.animate();
    
    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
  }
  
  createMaterials() {
    return {
      wall: new THREE.MeshLambertMaterial({ 
        color: 0xe8e8e8,
        transparent: true,
        opacity: 0.9
      }),
      floor: {
        living_room: new THREE.MeshLambertMaterial({ color: 0xf5f5dc }),
        bedroom: new THREE.MeshLambertMaterial({ color: 0xffe4e1 }),
        kitchen: new THREE.MeshLambertMaterial({ color: 0xf0f8ff }),
        bathroom: new THREE.MeshLambertMaterial({ color: 0xe0ffff }),
        corridor: new THREE.MeshLambertMaterial({ color: 0xf5f5f5 }),
        default: new THREE.MeshLambertMaterial({ color: 0xffffff })
      },
      furniture: {
        wood: new THREE.MeshLambertMaterial({ color: 0x8b4513 }),
        fabric: new THREE.MeshLambertMaterial({ color: 0x4682b4 }),
        metal: new THREE.MeshLambertMaterial({ color: 0x696969 }),
        white: new THREE.MeshLambertMaterial({ color: 0xffffff }),
        black: new THREE.MeshLambertMaterial({ color: 0x2f2f2f })
      },
      door: new THREE.MeshLambertMaterial({ color: 0x8b4513 }),
      window: new THREE.MeshLambertMaterial({ 
        color: 0x87ceeb, 
        transparent: true, 
        opacity: 0.6 
      }),
      ground: new THREE.MeshLambertMaterial({ color: 0xcccccc })
    };
  }
  
  setupAdvancedLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);
    
    // Main directional light (sun)
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(20, 30, 10);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 100;
    mainLight.shadow.camera.left = -50;
    mainLight.shadow.camera.right = 50;
    mainLight.shadow.camera.top = 50;
    mainLight.shadow.camera.bottom = -50;
    this.scene.add(mainLight);
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-10, 20, -10);
    this.scene.add(fillLight);
    
    // Point lights for rooms
    const roomLight1 = new THREE.PointLight(0xffffff, 0.5, 15);
    roomLight1.position.set(5, 8, 5);
    this.scene.add(roomLight1);
    
    const roomLight2 = new THREE.PointLight(0xffffff, 0.5, 15);
    roomLight2.position.set(-5, 8, -5);
    this.scene.add(roomLight2);
  }
  
  setupControls() {
    let isMouseDown = false;
    let isPanning = false;
    let mouseX = 0;
    let mouseY = 0;
    
    const canvas = this.renderer.domElement;
    
    canvas.addEventListener('mousedown', (event) => {
      isMouseDown = true;
      isPanning = event.button === 2; // Right mouse button for panning
      mouseX = event.clientX;
      mouseY = event.clientY;
      event.preventDefault();
    });
    
    canvas.addEventListener('mousemove', (event) => {
      if (!isMouseDown) return;
      
      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;
      
      if (isPanning) {
        // Pan camera
        const panSpeed = 0.02;
        const right = new THREE.Vector3();
        const up = new THREE.Vector3(0, 1, 0);
        
        this.camera.getWorldDirection(right);
        right.cross(up).normalize();
        
        this.camera.position.add(right.multiplyScalar(-deltaX * panSpeed));
        this.camera.position.add(up.multiplyScalar(deltaY * panSpeed));
      } else {
        // Orbit camera
        const spherical = new THREE.Spherical();
        spherical.setFromVector3(this.camera.position);
        spherical.theta -= deltaX * 0.01;
        spherical.phi += deltaY * 0.01;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
        
        this.camera.position.setFromSpherical(spherical);
        this.camera.lookAt(0, 0, 0);
      }
      
      mouseX = event.clientX;
      mouseY = event.clientY;
    });
    
    canvas.addEventListener('mouseup', () => {
      isMouseDown = false;
      isPanning = false;
    });
    
    canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
    });
    
    // Zoom with mouse wheel
    canvas.addEventListener('wheel', (event) => {
      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      this.camera.position.multiplyScalar(scale);
      event.preventDefault();
    });
  }
  
  addEnvironment() {
    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const ground = new THREE.Mesh(groundGeometry, this.materials.ground);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.5;
    ground.receiveShadow = true;
    this.scene.add(ground);
    
    // Sky color
    this.scene.background = new THREE.Color(0xf0f8ff);
  }
  
  createWall(wall) {
    const start = new THREE.Vector3(wall.start[0], wall.start[1], wall.start[2]);
    const end = new THREE.Vector3(wall.end[0], wall.end[1], wall.end[2]);
    
    const length = start.distanceTo(end);
    const height = wall.height || 3.0;
    const thickness = wall.thickness || 0.2;
    
    // Create wall geometry
    const geometry = new THREE.BoxGeometry(length, height, thickness);
    const wallMesh = new THREE.Mesh(geometry, this.materials.wall);
    
    // Position wall
    const center = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    center.y = height / 2; // Proper wall positioning
    wallMesh.position.copy(center);
    
    // Rotate wall to match direction
    const direction = new THREE.Vector3().subVectors(end, start).normalize();
    const angle = Math.atan2(direction.z, direction.x);
    wallMesh.rotation.y = angle;
    
    wallMesh.castShadow = true;
    wallMesh.receiveShadow = true;
    
    // Add wall details
    this.addWallDetails(wallMesh, length, height);
    
    return wallMesh;
  }
  
  addWallDetails(wall, length, height) {
    // Add baseboard
    const baseboardGeometry = new THREE.BoxGeometry(length * 0.98, 0.1, 0.05);
    const baseboard = new THREE.Mesh(baseboardGeometry, this.materials.furniture.wood);
    baseboard.position.set(0, -height/2 + 0.05, 0.125);
    wall.add(baseboard);
    
    // Add wall texture variation
    if (Math.random() > 0.7) {
      const frameGeometry = new THREE.BoxGeometry(0.8, 1.2, 0.02);
      const frame = new THREE.Mesh(frameGeometry, this.materials.furniture.wood);
      frame.position.set((Math.random() - 0.5) * length * 0.5, 0.5, 0.125);
      wall.add(frame);
    }
  }
  
  createRoom(room) {
    const roomGroup = new THREE.Group();
    const bounds = room.bounds;
    
    // Create room floor
    const floorGeometry = new THREE.PlaneGeometry(bounds.width, bounds.height);
    const roomType = room.type.toLowerCase().replace(' ', '_');
    const floorMaterial = this.materials.floor[roomType] || this.materials.floor.default;
    
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(bounds.x + bounds.width/2, 0, bounds.y + bounds.height/2);
    floor.receiveShadow = true;
    
    // Add floor border
    const borderGeometry = new THREE.RingGeometry(
      Math.min(bounds.width, bounds.height) / 2 - 0.1,
      Math.min(bounds.width, bounds.height) / 2,
      8
    );
    const border = new THREE.Mesh(borderGeometry, this.materials.furniture.wood);
    border.rotation.x = -Math.PI / 2;
    border.position.set(bounds.x + bounds.width/2, 0.01, bounds.y + bounds.height/2);
    
    roomGroup.add(floor);
    roomGroup.add(border);
    
    // Add room label with better styling
    this.addRoomLabel(roomGroup, room);
    
    // Add room-specific details
    this.addRoomDetails(roomGroup, room);
    
    return roomGroup;
  }
  
  addRoomLabel(roomGroup, room) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 128;
    
    // Create gradient background
    const gradient = context.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, 'rgba(255,255,255,0.9)');
    gradient.addColorStop(1, 'rgba(240,240,240,0.9)');
    
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add border
    context.strokeStyle = '#cccccc';
    context.lineWidth = 2;
    context.strokeRect(0, 0, canvas.width, canvas.height);
    
    // Add text
    context.font = 'bold 36px Arial';
    context.fillStyle = '#333333';
    context.textAlign = 'center';
    context.fillText(room.type, canvas.width/2, canvas.height/2 + 12);
    
    // Add area info
    context.font = '24px Arial';
    context.fillStyle = '#666666';
    const area = room.area ? room.area.toFixed(1) + ' m²' : '';
    context.fillText(area, canvas.width/2, canvas.height/2 + 45);
    
    const texture = new THREE.CanvasTexture(canvas);
    const labelMaterial = new THREE.MeshBasicMaterial({ 
      map: texture, 
      transparent: true,
      alphaTest: 0.1
    });
    
    const labelGeometry = new THREE.PlaneGeometry(3, 0.75);
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
    const roomType = room.type.toLowerCase();
    const bounds = room.bounds;
    const centerX = bounds.x + bounds.width/2;
    const centerZ = bounds.y + bounds.height/2;
    
    // Add room-specific architectural details
    if (roomType.includes('bathroom')) {
      // Add tile pattern
      this.addTilePattern(roomGroup, bounds);
    } else if (roomType.includes('kitchen')) {
      // Add kitchen backsplash
      this.addKitchenDetails(roomGroup, bounds);
    } else if (roomType.includes('living')) {
      // Add hardwood pattern
      this.addHardwoodPattern(roomGroup, bounds);
    }
  }
  
  addTilePattern(roomGroup, bounds) {
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
  }
  
  addHardwoodPattern(roomGroup, bounds) {
    const plankWidth = 0.15;
    const plankLength = 1.2;
    const planksX = Math.floor(bounds.width / plankLength);
    const planksZ = Math.floor(bounds.height / plankWidth);
    
    for (let x = 0; x < planksX; x++) {
      for (let z = 0; z < planksZ; z++) {
        const plankGeometry = new THREE.PlaneGeometry(plankLength * 0.95, plankWidth * 0.9);
        const plank = new THREE.Mesh(plankGeometry, this.materials.furniture.wood);
        plank.rotation.x = -Math.PI / 2;
        plank.position.set(
          bounds.x + x * plankLength + plankLength/2,
          0.003,
          bounds.y + z * plankWidth + plankWidth/2
        );
        roomGroup.add(plank);
      }
    }
  }
  
  addKitchenDetails(roomGroup, bounds) {
    // Add a simple backsplash effect
    const backsplashGeometry = new THREE.PlaneGeometry(bounds.width * 0.8, 0.6);
    const backsplash = new THREE.Mesh(backsplashGeometry, this.materials.furniture.white);
    backsplash.position.set(
      bounds.x + bounds.width/2,
      1.5,
      bounds.y + 0.1
    );
    roomGroup.add(backsplash);
  }
  
  createDetailedFurniture(item) {
    const group = new THREE.Group();
    const type = item.type.toLowerCase().replace(/[^a-z]/g, '');
    
    // Create furniture based on type with detailed models
    if (type.includes('sofa')) {
      group.add(this.createSofa(item));
    } else if (type.includes('bed')) {
      group.add(this.createBed(item));
    } else if (type.includes('coffeetable') || type.includes('table')) {
      group.add(this.createTable(item));
    } else if (type.includes('diningtable')) {
      group.add(this.createTable(item));
    } else if (type.includes('tv')) {
      group.add(this.createTV(item));
    } else if (type.includes('chair') || type.includes('armchair')) {
      group.add(this.createChair(item));
    } else if (type.includes('toilet')) {
      group.add(this.createToilet(item));
    } else if (type.includes('sink')) {
      group.add(this.createSink(item));
    } else if (type.includes('refrigerator')) {
      group.add(this.createRefrigerator(item));
    } else if (type.includes('desk')) {
      group.add(this.createDesk(item));
    } else if (type.includes('dresser')) {
      group.add(this.createDresser(item));
    } else if (type.includes('bathtub')) {
      group.add(this.createBathtub(item));
    } else if (type.includes('counter')) {
      group.add(this.createCounter(item));
    } else if (type.includes('stove')) {
      group.add(this.createStove(item));
    } else if (type.includes('wardrobe')) {
      group.add(this.createWardrobe(item));
    } else if (type.includes('nightstand')) {
      group.add(this.createNightstand(item));
    } else {
      // Default furniture
      group.add(this.createDefaultFurniture(item));
    }
    
    // Position and rotate
    group.position.set(item.position[0], item.position[1], item.position[2]);
    group.rotation.y = (item.rotation * Math.PI) / 180;
    
    // Add label
    this.addFurnitureLabel(group, item);
    
    return group;
  }
  
  createSofa(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Main body
    const bodyGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.6, scale[2]);
    const body = new THREE.Mesh(bodyGeometry, this.materials.furniture.fabric);
    body.position.y = scale[1] * 0.3;
    body.castShadow = true;
    group.add(body);
    
    // Back cushions
    const backGeometry = new THREE.BoxGeometry(scale[0] * 0.9, scale[1] * 0.8, scale[2] * 0.2);
    const back = new THREE.Mesh(backGeometry, this.materials.furniture.fabric);
    back.position.set(0, scale[1] * 0.7, scale[2] * 0.3);
    back.castShadow = true;
    group.add(back);
    
    // Arms
    const armGeometry = new THREE.BoxGeometry(scale[0] * 0.15, scale[1] * 0.6, scale[2]);
    const leftArm = new THREE.Mesh(armGeometry, this.materials.furniture.fabric);
    leftArm.position.set(-scale[0] * 0.425, scale[1] * 0.5, 0);
    leftArm.castShadow = true;
    group.add(leftArm);
    
    const rightArm = leftArm.clone();
    rightArm.position.x = scale[0] * 0.425;
    group.add(rightArm);
    
    return group;
  }
  
  createBed(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Mattress
    const mattressGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const mattress = new THREE.Mesh(mattressGeometry, this.materials.furniture.white);
    mattress.position.y = scale[1] / 2;
    mattress.castShadow = true;
    group.add(mattress);
    
    // Bed frame
    const frameGeometry = new THREE.BoxGeometry(scale[0] * 1.1, scale[1] * 0.3, scale[2] * 1.1);
    const frame = new THREE.Mesh(frameGeometry, this.materials.furniture.wood);
    frame.position.y = scale[1] * 0.15;
    frame.castShadow = true;
    group.add(frame);
    
    // Headboard
    const headboardGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 2, scale[2] * 0.1);
    const headboard = new THREE.Mesh(headboardGeometry, this.materials.furniture.wood);
    headboard.position.set(0, scale[1] * 1.2, scale[2] * 0.55);
    headboard.castShadow = true;
    group.add(headboard);
    
    // Pillows
    const pillowGeometry = new THREE.BoxGeometry(scale[0] * 0.3, scale[1] * 0.2, scale[2] * 0.2);
    const pillow1 = new THREE.Mesh(pillowGeometry, this.materials.furniture.white);
    pillow1.position.set(-scale[0] * 0.25, scale[1] * 1.1, scale[2] * 0.3);
    group.add(pillow1);
    
    const pillow2 = pillow1.clone();
    pillow2.position.x = scale[0] * 0.25;
    group.add(pillow2);
    
    return group;
  }
  
  createTable(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Table top
    const topGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.1, scale[2]);
    const top = new THREE.Mesh(topGeometry, this.materials.furniture.wood);
    top.position.y = scale[1] * 0.95;
    top.castShadow = true;
    group.add(top);
    
    // Legs
    const legGeometry = new THREE.BoxGeometry(0.08, scale[1] * 0.9, 0.08);
    const legPositions = [
      [-scale[0] * 0.4, scale[1] * 0.45, -scale[2] * 0.4],
      [scale[0] * 0.4, scale[1] * 0.45, -scale[2] * 0.4],
      [-scale[0] * 0.4, scale[1] * 0.45, scale[2] * 0.4],
      [scale[0] * 0.4, scale[1] * 0.45, scale[2] * 0.4]
    ];
    
    legPositions.forEach(pos => {
      const leg = new THREE.Mesh(legGeometry, this.materials.furniture.wood);
      leg.position.set(pos[0], pos[1], pos[2]);
      leg.castShadow = true;
      group.add(leg);
    });
    
    return group;
  }
  
  createTV(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Screen
    const screenGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const screen = new THREE.Mesh(screenGeometry, this.materials.furniture.black);
    screen.position.y = scale[1] / 2;
    screen.castShadow = true;
    group.add(screen);
    
    // Stand
    const standGeometry = new THREE.BoxGeometry(scale[0] * 0.3, scale[1] * 0.2, scale[2] * 2);
    const stand = new THREE.Mesh(standGeometry, this.materials.furniture.black);
    stand.position.y = scale[1] * 0.1;
    stand.castShadow = true;
    group.add(stand);
    
    return group;
  }
  
  createChair(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Seat
    const seatGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.1, scale[2]);
    const seat = new THREE.Mesh(seatGeometry, this.materials.furniture.wood);
    seat.position.y = scale[1] * 0.5;
    seat.castShadow = true;
    group.add(seat);
    
    // Backrest
    const backGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.8, scale[2] * 0.1);
    const back = new THREE.Mesh(backGeometry, this.materials.furniture.wood);
    back.position.set(0, scale[1] * 0.8, scale[2] * 0.4);
    back.castShadow = true;
    group.add(back);
    
    // Legs
    const legGeometry = new THREE.BoxGeometry(0.05, scale[1] * 0.5, 0.05);
    const legPositions = [
      [-scale[0] * 0.4, scale[1] * 0.25, -scale[2] * 0.4],
      [scale[0] * 0.4, scale[1] * 0.25, -scale[2] * 0.4],
      [-scale[0] * 0.4, scale[1] * 0.25, scale[2] * 0.4],
      [scale[0] * 0.4, scale[1] * 0.25, scale[2] * 0.4]
    ];
    
    legPositions.forEach(pos => {
      const leg = new THREE.Mesh(legGeometry, this.materials.furniture.wood);
      leg.position.set(pos[0], pos[1], pos[2]);
      leg.castShadow = true;
      group.add(leg);
    });
    
    return group;
  }
  
  createToilet(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Base
    const baseGeometry = new THREE.CylinderGeometry(scale[0] * 0.4, scale[0] * 0.5, scale[1] * 0.3, 12);
    const base = new THREE.Mesh(baseGeometry, this.materials.furniture.white);
    base.position.y = scale[1] * 0.15;
    base.castShadow = true;
    group.add(base);
    
    // Bowl
    const bowlGeometry = new THREE.CylinderGeometry(scale[0] * 0.3, scale[0] * 0.35, scale[1] * 0.2, 12);
    const bowl = new THREE.Mesh(bowlGeometry, this.materials.furniture.white);
    bowl.position.y = scale[1] * 0.4;
    bowl.castShadow = true;
    group.add(bowl);
    
    // Tank
    const tankGeometry = new THREE.BoxGeometry(scale[0] * 0.6, scale[1] * 0.4, scale[2] * 0.2);
    const tank = new THREE.Mesh(tankGeometry, this.materials.furniture.white);
    tank.position.set(0, scale[1] * 0.7, scale[2] * 0.3);
    tank.castShadow = true;
    group.add(tank);
    
    // Seat
    const seatGeometry = new THREE.CylinderGeometry(scale[0] * 0.35, scale[0] * 0.35, 0.05, 12);
    const seat = new THREE.Mesh(seatGeometry, this.materials.furniture.white);
    seat.position.y = scale[1] * 0.5;
    group.add(seat);
    
    return group;
  }
  
  createSink(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Basin
    const basinGeometry = new THREE.CylinderGeometry(scale[0] * 0.4, scale[0] * 0.3, scale[1] * 0.2, 16);
    const basin = new THREE.Mesh(basinGeometry, this.materials.furniture.white);
    basin.position.y = scale[1] * 0.8;
    basin.castShadow = true;
    group.add(basin);
    
    // Counter/Vanity
    const counterGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.1, scale[2]);
    const counter = new THREE.Mesh(counterGeometry, this.materials.furniture.wood);
    counter.position.y = scale[1] * 0.75;
    counter.castShadow = true;
    group.add(counter);
    
    // Cabinet
    const cabinetGeometry = new THREE.BoxGeometry(scale[0] * 0.9, scale[1] * 0.6, scale[2] * 0.9);
    const cabinet = new THREE.Mesh(cabinetGeometry, this.materials.furniture.wood);
    cabinet.position.y = scale[1] * 0.3;
    cabinet.castShadow = true;
    group.add(cabinet);
    
    // Faucet
    const faucetGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 8);
    const faucet = new THREE.Mesh(faucetGeometry, this.materials.furniture.metal);
    faucet.position.set(0, scale[1] * 1.0, -scale[2] * 0.2);
    group.add(faucet);
    
    return group;
  }
  
  createRefrigerator(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Main body
    const bodyGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const body = new THREE.Mesh(bodyGeometry, this.materials.furniture.white);
    body.position.y = scale[1] / 2;
    body.castShadow = true;
    group.add(body);
    
    // Door handles
    const handleGeometry = new THREE.BoxGeometry(0.02, 0.2, 0.05);
    const handle1 = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    handle1.position.set(scale[0] * 0.4, scale[1] * 0.7, scale[2] * 0.51);
    group.add(handle1);
    
    const handle2 = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    handle2.position.set(scale[0] * 0.4, scale[1] * 0.3, scale[2] * 0.51);
    group.add(handle2);
    
    // Door separation line
    const lineGeometry = new THREE.BoxGeometry(scale[0], 0.02, scale[2] * 0.02);
    const line = new THREE.Mesh(lineGeometry, this.materials.furniture.metal);
    line.position.set(0, scale[1] * 0.5, scale[2] * 0.51);
    group.add(line);
    
    return group;
  }
  
  createDesk(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Desktop
    const topGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.08, scale[2]);
    const top = new THREE.Mesh(topGeometry, this.materials.furniture.wood);
    top.position.y = scale[1] * 0.92;
    top.castShadow = true;
    group.add(top);
    
    // Legs/Pedestals
    const leftPedestalGeometry = new THREE.BoxGeometry(scale[0] * 0.15, scale[1] * 0.8, scale[2] * 0.9);
    const leftPedestal = new THREE.Mesh(leftPedestalGeometry, this.materials.furniture.wood);
    leftPedestal.position.set(-scale[0] * 0.35, scale[1] * 0.4, 0);
    leftPedestal.castShadow = true;
    group.add(leftPedestal);
    
    const rightPedestalGeometry = new THREE.BoxGeometry(scale[0] * 0.15, scale[1] * 0.8, scale[2] * 0.9);
    const rightPedestal = new THREE.Mesh(rightPedestalGeometry, this.materials.furniture.wood);
    rightPedestal.position.set(scale[0] * 0.35, scale[1] * 0.4, 0);
    rightPedestal.castShadow = true;
    group.add(rightPedestal);
    
    return group;
  }
  
  createDresser(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Main body
    const bodyGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const body = new THREE.Mesh(bodyGeometry, this.materials.furniture.wood);
    body.position.y = scale[1] / 2;
    body.castShadow = true;
    group.add(body);
    
    // Drawers (visual lines)
    for (let i = 0; i < 3; i++) {
      const drawerLineGeometry = new THREE.BoxGeometry(scale[0] * 0.9, 0.02, scale[2] * 0.02);
      const drawerLine = new THREE.Mesh(drawerLineGeometry, this.materials.furniture.metal);
      drawerLine.position.set(0, scale[1] * (0.25 + i * 0.25), scale[2] * 0.51);
      group.add(drawerLine);
      
      // Drawer handles
      const handleGeometry = new THREE.BoxGeometry(0.08, 0.02, 0.03);
      const handle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
      handle.position.set(scale[0] * 0.3, scale[1] * (0.25 + i * 0.25), scale[2] * 0.52);
      group.add(handle);
    }
    
    return group;
  }
  
  createBathtub(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Tub body
    const tubGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const tub = new THREE.Mesh(tubGeometry, this.materials.furniture.white);
    tub.position.y = scale[1] / 2;
    tub.castShadow = true;
    group.add(tub);
    
    // Inner basin
    const basinGeometry = new THREE.BoxGeometry(scale[0] * 0.85, scale[1] * 0.6, scale[2] * 0.85);
    const basin = new THREE.Mesh(basinGeometry, this.materials.furniture.white);
    basin.position.y = scale[1] * 0.7;
    group.add(basin);
    
    // Faucet
    const faucetGeometry = new THREE.CylinderGeometry(0.03, 0.03, 0.2, 8);
    const faucet = new THREE.Mesh(faucetGeometry, this.materials.furniture.metal);
    faucet.position.set(0, scale[1] * 1.1, -scale[2] * 0.4);
    faucet.rotation.x = Math.PI / 2;
    group.add(faucet);
    
    return group;
  }
  
  createCounter(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Countertop
    const topGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.1, scale[2]);
    const top = new THREE.Mesh(topGeometry, this.materials.furniture.wood);
    top.position.y = scale[1] * 0.95;
    top.castShadow = true;
    group.add(top);
    
    // Cabinet base
    const cabinetGeometry = new THREE.BoxGeometry(scale[0], scale[1] * 0.8, scale[2] * 0.9);
    const cabinet = new THREE.Mesh(cabinetGeometry, this.materials.furniture.wood);
    cabinet.position.y = scale[1] * 0.4;
    cabinet.castShadow = true;
    group.add(cabinet);
    
    // Cabinet doors (visual lines)
    const numDoors = Math.floor(scale[0] / 0.6);
    for (let i = 0; i < numDoors; i++) {
      const doorLineGeometry = new THREE.BoxGeometry(0.02, scale[1] * 0.7, scale[2] * 0.02);
      const doorLine = new THREE.Mesh(doorLineGeometry, this.materials.furniture.metal);
      doorLine.position.set(-scale[0] * 0.4 + i * (scale[0] * 0.8 / numDoors), scale[1] * 0.4, scale[2] * 0.46);
      group.add(doorLine);
    }
    
    return group;
  }
  
  createStove(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Main body
    const bodyGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const body = new THREE.Mesh(bodyGeometry, this.materials.furniture.metal);
    body.position.y = scale[1] / 2;
    body.castShadow = true;
    group.add(body);
    
    // Cooktop
    const cooktopGeometry = new THREE.BoxGeometry(scale[0] * 0.9, 0.05, scale[2] * 0.9);
    const cooktop = new THREE.Mesh(cooktopGeometry, this.materials.furniture.black);
    cooktop.position.y = scale[1] * 1.02;
    group.add(cooktop);
    
    // Burners
    const burnerGeometry = new THREE.CylinderGeometry(0.08, 0.08, 0.02, 16);
    const burnerPositions = [
      [-scale[0] * 0.25, scale[1] * 1.03, -scale[2] * 0.25],
      [scale[0] * 0.25, scale[1] * 1.03, -scale[2] * 0.25],
      [-scale[0] * 0.25, scale[1] * 1.03, scale[2] * 0.25],
      [scale[0] * 0.25, scale[1] * 1.03, scale[2] * 0.25]
    ];
    
    burnerPositions.forEach(pos => {
      const burner = new THREE.Mesh(burnerGeometry, this.materials.furniture.black);
      burner.position.set(pos[0], pos[1], pos[2]);
      group.add(burner);
    });
    
    // Oven door handle
    const handleGeometry = new THREE.BoxGeometry(scale[0] * 0.6, 0.03, 0.05);
    const handle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    handle.position.set(0, scale[1] * 0.6, scale[2] * 0.51);
    group.add(handle);
    
    return group;
  }
  
  createWardrobe(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Main body
    const bodyGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const body = new THREE.Mesh(bodyGeometry, this.materials.furniture.wood);
    body.position.y = scale[1] / 2;
    body.castShadow = true;
    group.add(body);
    
    // Door separation
    const separationGeometry = new THREE.BoxGeometry(0.02, scale[1] * 0.9, scale[2] * 0.02);
    const separation = new THREE.Mesh(separationGeometry, this.materials.furniture.metal);
    separation.position.set(0, scale[1] / 2, scale[2] * 0.51);
    group.add(separation);
    
    // Door handles
    const handleGeometry = new THREE.BoxGeometry(0.02, 0.15, 0.05);
    const leftHandle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    leftHandle.position.set(-scale[0] * 0.15, scale[1] * 0.6, scale[2] * 0.51);
    group.add(leftHandle);
    
    const rightHandle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    rightHandle.position.set(scale[0] * 0.15, scale[1] * 0.6, scale[2] * 0.51);
    group.add(rightHandle);
    
    return group;
  }
  
  createNightstand(item) {
    const group = new THREE.Group();
    const scale = item.scale;
    
    // Main body
    const bodyGeometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const body = new THREE.Mesh(bodyGeometry, this.materials.furniture.wood);
    body.position.y = scale[1] / 2;
    body.castShadow = true;
    group.add(body);
    
    // Drawer
    const drawerLineGeometry = new THREE.BoxGeometry(scale[0] * 0.9, 0.02, scale[2] * 0.02);
    const drawerLine = new THREE.Mesh(drawerLineGeometry, this.materials.furniture.metal);
    drawerLine.position.set(0, scale[1] * 0.7, scale[2] * 0.51);
    group.add(drawerLine);
    
    // Handle
    const handleGeometry = new THREE.BoxGeometry(0.06, 0.02, 0.03);
    const handle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    handle.position.set(0, scale[1] * 0.7, scale[2] * 0.52);
    group.add(handle);
    
    return group;
  }
  
  createDefaultFurniture(item) {
    const geometry = new THREE.BoxGeometry(item.scale[0], item.scale[1], item.scale[2]);
    const color = item.color || '#8B4513';
    const material = new THREE.MeshLambertMaterial({ color: color });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.y = item.scale[1] / 2;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }
  
  addFurnitureLabel(group, item) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    
    context.fillStyle = 'rgba(255,255,255,0.9)';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#cccccc';
    context.strokeRect(0, 0, canvas.width, canvas.height);
    
    context.font = '18px Arial';
    context.fillStyle = 'black';
    context.textAlign = 'center';
    context.fillText(item.type, canvas.width/2, canvas.height/2 + 6);
    
    const texture = new THREE.CanvasTexture(canvas);
    const labelMaterial = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
    const labelGeometry = new THREE.PlaneGeometry(1, 0.25);
    const label = new THREE.Mesh(labelGeometry, labelMaterial);
    label.position.set(0, Math.max(item.scale[1], 2) + 0.5, 0);
    
    group.add(label);
  }
  
  createDoor(door) {
    const group = new THREE.Group();
    const scale = door.scale;
    
    // Door frame
    const frameGeometry = new THREE.BoxGeometry(scale[0] * 1.1, scale[1], 0.15);
    const frame = new THREE.Mesh(frameGeometry, this.materials.furniture.wood);
    frame.position.y = scale[1] / 2;
    group.add(frame);
    
    // Door panel
    const doorGeometry = new THREE.BoxGeometry(scale[0] * 0.9, scale[1] * 0.9, 0.08);
    const doorPanel = new THREE.Mesh(doorGeometry, this.materials.door);
    doorPanel.position.set(0, scale[1] / 2, 0);
    doorPanel.castShadow = true;
    group.add(doorPanel);
    
    // Door handle
    const handleGeometry = new THREE.SphereGeometry(0.05, 8, 6);
    const handle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    handle.position.set(scale[0] * 0.3, scale[1] * 0.5, 0.1);
    group.add(handle);
    
    group.position.set(door.position[0], door.position[1], door.position[2]);
    group.rotation.y = (door.rotation * Math.PI) / 180;
    
    return group;
  }
  
  loadScene(sceneData) {
    // Clear existing scene
    this.clearScene();
    
    if (!sceneData) return;
    
    // Add walls
    if (sceneData.walls) {
      sceneData.walls.forEach(wall => {
        const wallMesh = this.createWall(wall);
        this.wallGroup.add(wallMesh);
      });
    }
    
    // Add rooms
    if (sceneData.rooms) {
      sceneData.rooms.forEach(room => {
        const roomGroup = this.createRoom(room);
        this.roomGroup.add(roomGroup);
      });
    }
    
    // Add doors
    if (sceneData.doors) {
      sceneData.doors.forEach(door => {
        const doorMesh = this.createDoor(door);
        this.doorGroup.add(doorMesh);
      });
    }
    
    // Add furniture
    if (sceneData.furniture) {
      sceneData.furniture.forEach(item => {
        const furnitureMesh = this.createDetailedFurniture(item);
        this.furnitureGroup.add(furnitureMesh);
      });
    }
  }
  
  clearScene() {
    this.wallGroup.clear();
    this.roomGroup.clear();
    this.furnitureGroup.clear();
    this.doorGroup.clear();
    this.windowGroup.clear();
  }
  
  animate() {
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene, this.camera);
  }
  
  resetCamera() {
    this.camera.position.set(20, 25, 20);
    this.camera.lookAt(0, 0, 0);
  }
  
  onWindowResize() {
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }
  
  dispose() {
    this.clearScene();
    this.container.removeChild(this.renderer.domElement);
    this.renderer.dispose();
  }
  
  toggleGroup(groupName) {
    const groups = {
      walls: this.wallGroup,
      rooms: this.roomGroup,
      furniture: this.furnitureGroup,
      doors: this.doorGroup,
      windows: this.windowGroup
    };
    
    if (groups[groupName]) {
      groups[groupName].visible = !groups[groupName].visible;
    }
  }
}

// Main React Component
export default function FloorPlan3DViewer() {
  const [sceneData, setSceneData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [showControls, setShowControls] = useState(false);
  const [visibility, setVisibility] = useState({
    walls: true,
    rooms: true,
    furniture: true,
    doors: true,
    windows: true
  });
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // Enhanced sample data with detailed room information
  const sampleData = {
    walls: [
      // Outer walls
      { start: [0, 0, 0], end: [15, 0, 0], height: 3, thickness: 0.3 },
      { start: [15, 0, 0], end: [15, 0, 12], height: 3, thickness: 0.3 },
      { start: [15, 0, 12], end: [0, 0, 12], height: 3, thickness: 0.3 },
      { start: [0, 0, 12], end: [0, 0, 0], height: 3, thickness: 0.3 },
      
      // Interior walls
      { start: [8, 0, 0], end: [8, 0, 8], height: 3, thickness: 0.2 },
      { start: [0, 0, 8], end: [8, 0, 8], height: 3, thickness: 0.2 },
      { start: [12, 0, 8], end: [15, 0, 8], height: 3, thickness: 0.2 },
      
      // Bathroom wall
      { start: [12, 0, 0], end: [12, 0, 4], height: 3, thickness: 0.2 }
    ],
    rooms: [
      {
        id: "living_room",
        type: "Living Room",
        bounds: { x: 0, y: 0, width: 8, height: 8 },
        area: 64,
        centroid: [4, 4]
      },
      {
        id: "kitchen",
        type: "Kitchen", 
        bounds: { x: 8, y: 0, width: 4, height: 8 },
        area: 32,
        centroid: [10, 4]
      },
      {
        id: "bedroom_1",
        type: "Bedroom",
        bounds: { x: 0, y: 8, width: 8, height: 4 },
        area: 32,
        centroid: [4, 10]
      },
      {
        id: "bedroom_2", 
        type: "Bedroom",
        bounds: { x: 8, y: 8, width: 4, height: 4 },
        area: 16,
        centroid: [10, 10]
      },
      {
        id: "bathroom",
        type: "Bathroom",
        bounds: { x: 12, y: 0, width: 3, height: 4 },
        area: 12,
        centroid: [13.5, 2]
      },
      {
        id: "corridor",
        type: "Corridor",
        bounds: { x: 12, y: 8, width: 3, height: 4 },
        area: 12,
        centroid: [13.5, 10]
      }
    ],
    doors: [
      {
        position: [7, 0, 8],
        rotation: 90,
        scale: [1.0, 2.1, 0.1],
        type: "Door"
      },
      {
        position: [12, 0, 6],
        rotation: 0,
        scale: [1.0, 2.1, 0.1],
        type: "Door"
      },
      {
        position: [12, 0, 2],
        rotation: 90,
        scale: [0.8, 2.1, 0.1],
        type: "Door"
      }
    ],
    furniture: [
      // Living Room Furniture
      {
        type: "Sofa",
        position: [4, 0, 3],
        rotation: 0,
        scale: [3.0, 0.8, 1.2],
        color: "#4682B4",
        room: "living_room"
      },
      {
        type: "Coffee Table",
        position: [4, 0, 5],
        rotation: 0,
        scale: [1.5, 0.4, 0.8],
        color: "#8B4513",
        room: "living_room"
      },
      {
        type: "TV",
        position: [4, 0, 7.5],
        rotation: 180,
        scale: [2.0, 1.2, 0.15],
        color: "#000000",
        room: "living_room"
      },
      {
        type: "Armchair",
        position: [1.5, 0, 4],
        rotation: 45,
        scale: [1.0, 0.8, 1.0],
        color: "#8B4513",
        room: "living_room"
      },
      {
        type: "Armchair",
        position: [6.5, 0, 4],
        rotation: -45,
        scale: [1.0, 0.8, 1.0],
        color: "#8B4513",
        room: "living_room"
      },
      
      // Kitchen Furniture
      {
        type: "Refrigerator",
        position: [9, 0, 1],
        rotation: 0,
        scale: [0.8, 2.0, 0.8],
        color: "#F5F5F5",
        room: "kitchen"
      },
      {
        type: "Counter",
        position: [10, 0, 0.5],
        rotation: 0,
        scale: [3.5, 0.9, 0.6],
        color: "#D2B48C",
        room: "kitchen"
      },
      {
        type: "Stove",
        position: [11, 0, 0.5],
        rotation: 0,
        scale: [0.8, 0.9, 0.8],
        color: "#2F4F4F",
        room: "kitchen"
      },
      {
        type: "Sink",
        position: [9, 0, 0.5],
        rotation: 0,
        scale: [0.6, 0.9, 0.5],
        color: "#E6E6FA",
        room: "kitchen"
      },
      {
        type: "Dining Table",
        position: [10, 0, 5],
        rotation: 0,
        scale: [1.8, 0.8, 1.0],
        color: "#8B4513",
        room: "kitchen"
      },
      {
        type: "Chair",
        position: [9.5, 0, 5.8],
        rotation: 180,
        scale: [0.5, 0.9, 0.5],
        color: "#8B4513",
        room: "kitchen"
      },
      {
        type: "Chair",
        position: [10.5, 0, 5.8],
        rotation: 180,
        scale: [0.5, 0.9, 0.5],
        color: "#8B4513",
        room: "kitchen"
      },
      {
        type: "Chair",
        position: [9.5, 0, 4.2],
        rotation: 0,
        scale: [0.5, 0.9, 0.5],
        color: "#8B4513",
        room: "kitchen"
      },
      {
        type: "Chair",
        position: [10.5, 0, 4.2],
        rotation: 0,
        scale: [0.5, 0.9, 0.5],
        color: "#8B4513",
        room: "kitchen"
      },
      
      // Bedroom 1 Furniture
      {
        type: "Bed",
        position: [4, 0, 10],
        rotation: 0,
        scale: [2.2, 0.6, 2.0],
        color: "#FFFFFF",
        room: "bedroom_1"
      },
      {
        type: "Nightstand",
        position: [5.5, 0, 10],
        rotation: 0,
        scale: [0.5, 0.7, 0.4],
        color: "#8B4513",
        room: "bedroom_1"
      },
      {
        type: "Nightstand",
        position: [2.5, 0, 10],
        rotation: 0,
        scale: [0.5, 0.7, 0.4],
        color: "#8B4513",
        room: "bedroom_1"
      },
      {
        type: "Wardrobe",
        position: [1, 0, 11.5],
        rotation: 0,
        scale: [2.0, 2.2, 0.6],
        color: "#654321",
        room: "bedroom_1"
      },
      {
        type: "Desk",
        position: [6.5, 0, 8.5],
        rotation: 0,
        scale: [1.5, 0.8, 0.7],
        color: "#8B4513",
        room: "bedroom_1"
      },
      {
        type: "Chair",
        position: [6.5, 0, 9],
        rotation: 180,
        scale: [0.6, 0.9, 0.6],
        color: "#8B4513",
        room: "bedroom_1"
      },
      
      // Bedroom 2 Furniture  
      {
        type: "Bed",
        position: [10, 0, 10],
        rotation: 90,
        scale: [1.8, 0.6, 1.4],
        color: "#FFFFFF",
        room: "bedroom_2"
      },
      {
        type: "Nightstand",
        position: [10, 0, 11.2],
        rotation: 0,
        scale: [0.4, 0.7, 0.4],
        color: "#8B4513",
        room: "bedroom_2"
      },
      {
        type: "Dresser",
        position: [8.5, 0, 9],
        rotation: 0,
        scale: [1.2, 1.0, 0.5],
        color: "#8B4513",
        room: "bedroom_2"
      },
      
      // Bathroom Furniture
      {
        type: "Toilet",
        position: [13, 0, 1],
        rotation: 0,
        scale: [0.6, 0.8, 0.4],
        color: "#FFFFFF",
        room: "bathroom"
      },
      {
        type: "Sink",
        position: [14.5, 0, 2.5],
        rotation: 270,
        scale: [0.6, 0.9, 0.4],
        color: "#FFFFFF",
        room: "bathroom"
      },
      {
        type: "Bathtub",
        position: [13.5, 0, 3.5],
        rotation: 0,
        scale: [1.8, 0.6, 0.8],
        color: "#FFFFFF",
        room: "bathroom"
      }
    ]
  };
  
  useEffect(() => {
    if (containerRef.current && !sceneRef.current) {
      sceneRef.current = new FloorPlan3DScene(containerRef.current);
      sceneRef.current.loadScene(sampleData);
      setSceneData(sampleData);
      setUploadStatus('Sample 3D floor plan loaded with detailed rooms and furniture');
    }
    
    return () => {
      if (sceneRef.current) {
        sceneRef.current.dispose();
        sceneRef.current = null;
      }
    };
  }, []);
  
  useEffect(() => {
    if (sceneRef.current && sceneData) {
      sceneRef.current.loadScene(sceneData);
    }
  }, [sceneData]);
  
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setIsLoading(true);
    setUploadStatus('Processing floor plan image...');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }
      
      const result = await response.json();
      setSceneData(result.data);
      setUploadStatus(`✅ Successfully processed: ${result.stats.walls} walls, ${result.stats.rooms} rooms, ${result.stats.furniture} furniture items`);
    } catch (error) {
      setUploadStatus(`❌ Error: ${error.message}. Make sure the backend server is running on port 8000.`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const loadSampleData = () => {
    setSceneData(sampleData);
    setUploadStatus('✅ Sample 3D floor plan loaded with 6 rooms and 25+ furniture pieces');
  };
  
  const resetCamera = () => {
    if (sceneRef.current) {
      sceneRef.current.resetCamera();
    }
  };
  
  const toggleVisibility = (groupName) => {
    if (sceneRef.current) {
      sceneRef.current.toggleGroup(groupName);
      setVisibility(prev => ({
        ...prev,
        [groupName]: !prev[groupName]
      }));
    }
  };
  
  const getRoomStats = () => {
    if (!sceneData) return {};
    
    const roomTypes = {};
    sceneData.rooms?.forEach(room => {
      const type = room.type;
      if (!roomTypes[type]) {
        roomTypes[type] = { count: 0, totalArea: 0 };
      }
      roomTypes[type].count++;
      roomTypes[type].totalArea += room.area || 0;
    });
    
    return roomTypes;
  };
  
  const roomStats = getRoomStats();
  
  return (
    <div className="w-full h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-lg p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Home className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">
                Advanced 3D Floor Plan Viewer
              </h1>
              <p className="text-sm text-gray-600">
                Interactive architectural visualization with detailed furniture
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>Upload Floor Plan</span>
            </button>
            
            <button
              onClick={loadSampleData}
              className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
            >
              <Eye className="w-4 h-4" />
              <span>Load Sample</span>
            </button>
            
            <button
              onClick={resetCamera}
              className="flex items-center space-x-2 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset View</span>
            </button>
            
            <button
              onClick={() => setShowControls(!showControls)}
              className="flex items-center space-x-2 bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
            >
              <Settings className="w-4 h-4" />
              <span>Controls</span>
            </button>
          </div>
        </div>
        
        {uploadStatus && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">{uploadStatus}</p>
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className="flex-1 relative">
        {isLoading ? (
          <div className="flex items-center justify-center h-full bg-gray-100">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
              <p className="text-lg text-gray-700 mb-2">Processing floor plan...</p>
              <p className="text-sm text-gray-500">Analyzing walls, rooms, and placing furniture</p>
            </div>
          </div>
        ) : (
          <div 
            ref={containerRef} 
            className="w-full h-full"
            style={{ height: 'calc(100vh - 140px)' }}
          />
        )}
        
        {/* Enhanced Controls Panel */}
        <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 max-w-xs">
          <div className="flex flex-col space-y-2">
            <div className="text-sm text-gray-700 font-semibold border-b pb-1">Navigation</div>
            <div className="text-xs text-gray-600 space-y-1">
              <div>🖱️ <strong>Left drag:</strong> Rotate view</div>
              <div>🖱️ <strong>Right drag:</strong> Pan view</div>
              <div>🎪 <strong>Mouse wheel:</strong> Zoom in/out</div>
              <div>🎯 <strong>Reset View:</strong> Return to default</div>
            </div>
          </div>
        </div>
        
        {/* Visibility Controls */}
        {showControls && (
          <div className="absolute top-4 left-4 bg-white rounded-lg shadow-lg p-4 min-w-48">
            <h3 className="text-sm font-semibold text-gray-800 mb-3 border-b pb-1">
              Layer Visibility
            </h3>
            <div className="space-y-2">
              {Object.entries(visibility).map(([key, visible]) => (
                <label key={key} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={visible}
                    onChange={() => toggleVisibility(key)}
                    className="rounded text-blue-600"
                  />
                  <span className="text-sm text-gray-700 capitalize">
                    {key.replace('_', ' ')}
                  </span>
                </label>
              ))}
            </div>
          </div>
        )}
        
        {/* Enhanced Stats Panel */}
        {sceneData && (
          <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg p-4 min-w-64">
            <h3 className="text-sm font-semibold text-gray-800 mb-3 border-b pb-1">
              Floor Plan Statistics
            </h3>
            <div className="space-y-3">
              {/* Basic Stats */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-blue-50 p-2 rounded">
                  <div className="font-medium text-blue-800">Walls</div>
                  <div className="text-lg font-bold text-blue-600">
                    {sceneData.walls?.length || 0}
                  </div>
                </div>
                <div className="bg-green-50 p-2 rounded">
                  <div className="font-medium text-green-800">Rooms</div>
                  <div className="text-lg font-bold text-green-600">
                    {sceneData.rooms?.length || 0}
                  </div>
                </div>
                <div className="bg-purple-50 p-2 rounded">
                  <div className="font-medium text-purple-800">Furniture</div>
                  <div className="text-lg font-bold text-purple-600">
                    {sceneData.furniture?.length || 0}
                  </div>
                </div>
                <div className="bg-orange-50 p-2 rounded">
                  <div className="font-medium text-orange-800">Doors</div>
                  <div className="text-lg font-bold text-orange-600">
                    {sceneData.doors?.length || 0}
                  </div>
                </div>
              </div>
              
              {/* Room Breakdown */}
              {Object.keys(roomStats).length > 0 && (
                <div className="border-t pt-2">
                  <div className="text-xs font-medium text-gray-700 mb-1">Room Types</div>
                  <div className="space-y-1">
                    {Object.entries(roomStats).map(([type, data]) => (
                      <div key={type} className="flex justify-between text-xs">
                        <span className="text-gray-600">{type}</span>
                        <span className="font-medium">
                          {data.count} ({data.totalArea.toFixed(0)}m²)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Total Area */}
              <div className="border-t pt-2 text-xs">
                <div className="flex justify-between font-medium">
                  <span className="text-gray-700">Total Area:</span>
                  <span className="text-gray-900">
                    {Object.values(roomStats).reduce((sum, data) => sum + data.totalArea, 0).toFixed(0)}m²
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
}