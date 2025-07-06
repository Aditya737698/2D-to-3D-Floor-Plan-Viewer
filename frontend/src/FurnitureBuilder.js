import * as THREE from 'three';

export class FurnitureBuilder {
  constructor(materials) {
    this.materials = materials;
  }
  
  createFurniture(item) {
    if (!item.type || !item.position || !item.scale) {
      console.warn('Invalid furniture data:', item);
      return new THREE.Group();
    }
    
    const group = new THREE.Group();
    const type = this.normalizeFurnitureType(item.type);
    
    try {
      // Create furniture based on type
      const furnitureObject = this.createFurnitureByType(type, item);
      if (furnitureObject) {
        group.add(furnitureObject);
      }
      
      // Position and rotate
      group.position.set(item.position[0], item.position[1], item.position[2]);
      group.rotation.y = (item.rotation || 0) * Math.PI / 180;
      
      // Add label
      this.addFurnitureLabel(group, item);
      
      group.userData = { furnitureType: type, originalItem: item };
      
    } catch (error) {
      console.error(`Error creating furniture ${item.type}:`, error);
      // Fallback to default furniture
      group.add(this.createDefaultFurniture(item));
    }
    
    return group;
  }
  
  normalizeFurnitureType(type) {
    return type.toLowerCase().replace(/[^a-z]/g, '');
  }
  
  createFurnitureByType(type, item) {
    const furnitureMap = {
      'sofa': () => this.createSofa(item),
      'bed': () => this.createBed(item),
      'coffeetable': () => this.createTable(item),
      'diningtable': () => this.createTable(item),
      'table': () => this.createTable(item),
      'tv': () => this.createTV(item),
      'chair': () => this.createChair(item),
      'armchair': () => this.createChair(item),
      'toilet': () => this.createToilet(item),
      'sink': () => this.createSink(item),
      'refrigerator': () => this.createRefrigerator(item),
      'desk': () => this.createDesk(item),
      'dresser': () => this.createDresser(item),
      'bathtub': () => this.createBathtub(item),
      'counter': () => this.createCounter(item),
      'stove': () => this.createStove(item),
      'wardrobe': () => this.createWardrobe(item),
      'nightstand': () => this.createNightstand(item)
    };
    
    const createFunc = furnitureMap[type];
    if (createFunc) {
      return createFunc();
    } else {
      console.warn(`Unknown furniture type: ${type}, using default`);
      return this.createDefaultFurniture(item);
    }
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
    for (let i = 0; i < 2; i++) {
      const pillow = new THREE.Mesh(pillowGeometry, this.materials.furniture.white);
      pillow.position.set((i - 0.5) * scale[0] * 0.4, scale[1] * 1.1, scale[2] * 0.3);
      group.add(pillow);
    }
    
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
    
    // Counter
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
    
    const handle2 = handle1.clone();
    handle2.position.y = scale[1] * 0.3;
    group.add(handle2);
    
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
    
    // Pedestals
    const pedestalGeometry = new THREE.BoxGeometry(scale[0] * 0.15, scale[1] * 0.8, scale[2] * 0.9);
    const leftPedestal = new THREE.Mesh(pedestalGeometry, this.materials.furniture.wood);
    leftPedestal.position.set(-scale[0] * 0.35, scale[1] * 0.4, 0);
    leftPedestal.castShadow = true;
    group.add(leftPedestal);
    
    const rightPedestal = leftPedestal.clone();
    rightPedestal.position.x = scale[0] * 0.35;
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
    
    // Drawer handles
    for (let i = 0; i < 3; i++) {
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
    
    // Door handles
    const handleGeometry = new THREE.BoxGeometry(0.02, 0.15, 0.05);
    const leftHandle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    leftHandle.position.set(-scale[0] * 0.15, scale[1] * 0.6, scale[2] * 0.51);
    group.add(leftHandle);
    
    const rightHandle = leftHandle.clone();
    rightHandle.position.x = scale[0] * 0.15;
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
    
    // Handle
    const handleGeometry = new THREE.BoxGeometry(0.06, 0.02, 0.03);
    const handle = new THREE.Mesh(handleGeometry, this.materials.furniture.metal);
    handle.position.set(0, scale[1] * 0.7, scale[2] * 0.52);
    group.add(handle);
    
    return group;
  }
  
  createDefaultFurniture(item) {
    const geometry = new THREE.BoxGeometry(item.scale[0], item.scale[1], item.scale[2]);
    const color = item.color ? new THREE.Color(item.color) : new THREE.Color(0x8B4513);
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
    label.position.set(0, Math.max(...item.scale) + 0.5, 0);
    
    group.add(label);
  }
  
  // Helper method to get furniture color
  getFurnitureColor(item) {
    if (item.color) {
      return new THREE.Color(item.color);
    }
    
    // Default colors based on furniture type
    const type = this.normalizeFurnitureType(item.type);
    const colorMap = {
      'sofa': 0x4682b4,
      'bed': 0xffffff,
      'table': 0x8b4513,
      'chair': 0x8b4513,
      'tv': 0x000000,
      'refrigerator': 0xf5f5f5,
      'stove': 0x2f4f4f,
      'toilet': 0xffffff,
      'sink': 0xffffff,
      'bathtub': 0xffffff
    };
    
    return new THREE.Color(colorMap[type] || 0x8b4513);
  }
}