import * as THREE from 'three';
import { MaterialLibrary } from './MaterialLibrary.js';
import { WallBuilder } from './WallBuilder.js';
import { RoomBuilder } from './RoomBuilder.js';
import { FurnitureBuilder } from './FurnitureBuilder.js';

export class SceneManager {
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
    
    // Builders
    this.materials = new MaterialLibrary();
    this.wallBuilder = new WallBuilder(this.materials);
    this.roomBuilder = new RoomBuilder(this.materials);
    this.furnitureBuilder = new FurnitureBuilder(this.materials);
    
    // Scene state
    this.sceneData = null;
    this.isAnimating = false;
    
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
    this.setupLighting();
    
    // Setup controls
    this.setupControls();
    
    // Add environment
    this.addEnvironment();
    
    // Start render loop
    this.startAnimation();
    
    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
  }
  
  setupLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);
    
    // Main directional light
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
  }
  
  setupControls() {
    let isMouseDown = false;
    let isPanning = false;
    let mouseX = 0;
    let mouseY = 0;
    
    const canvas = this.renderer.domElement;
    
    canvas.addEventListener('mousedown', (event) => {
      isMouseDown = true;
      isPanning = event.button === 2;
      mouseX = event.clientX;
      mouseY = event.clientY;
      event.preventDefault();
    });
    
    canvas.addEventListener('mousemove', (event) => {
      if (!isMouseDown) return;
      
      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;
      
      if (isPanning) {
        const panSpeed = 0.02;
        const right = new THREE.Vector3();
        const up = new THREE.Vector3(0, 1, 0);
        
        this.camera.getWorldDirection(right);
        right.cross(up).normalize();
        
        this.camera.position.add(right.multiplyScalar(-deltaX * panSpeed));
        this.camera.position.add(up.multiplyScalar(deltaY * panSpeed));
      } else {
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
  
  loadScene(sceneData) {
    console.log('Loading scene data:', sceneData);
    
    // Clear existing scene
    this.clearScene();
    
    if (!sceneData) {
      console.warn('No scene data provided');
      return;
    }
    
    this.sceneData = sceneData;
    
    try {
      // Build walls
      if (sceneData.walls && sceneData.walls.length > 0) {
        console.log(`Building ${sceneData.walls.length} walls`);
        sceneData.walls.forEach((wall, index) => {
          try {
            const wallMesh = this.wallBuilder.createWall(wall);
            this.wallGroup.add(wallMesh);
          } catch (error) {
            console.error(`Error creating wall ${index}:`, error);
          }
        });
      }
      
      // Build rooms
      if (sceneData.rooms && sceneData.rooms.length > 0) {
        console.log(`Building ${sceneData.rooms.length} rooms`);
        sceneData.rooms.forEach((room, index) => {
          try {
            const roomGroup = this.roomBuilder.createRoom(room);
            this.roomGroup.add(roomGroup);
          } catch (error) {
            console.error(`Error creating room ${index}:`, error);
          }
        });
      }
      
      // Build doors
      if (sceneData.doors && sceneData.doors.length > 0) {
        console.log(`Building ${sceneData.doors.length} doors`);
        sceneData.doors.forEach((door, index) => {
          try {
            const doorMesh = this.createDoor(door);
            this.doorGroup.add(doorMesh);
          } catch (error) {
            console.error(`Error creating door ${index}:`, error);
          }
        });
      }
      
      // Build furniture
      if (sceneData.furniture && sceneData.furniture.length > 0) {
        console.log(`Building ${sceneData.furniture.length} furniture pieces`);
        sceneData.furniture.forEach((item, index) => {
          try {
            const furnitureMesh = this.furnitureBuilder.createFurniture(item);
            this.furnitureGroup.add(furnitureMesh);
          } catch (error) {
            console.error(`Error creating furniture ${index}:`, error);
          }
        });
      }
      
      // Auto-fit camera to scene
      this.fitCameraToScene();
      
    } catch (error) {
      console.error('Error loading scene:', error);
    }
  }
  
  createDoor(door) {
    const group = new THREE.Group();
    const scale = door.scale || [1.0, 2.1, 0.1];
    
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
  
  fitCameraToScene() {
    if (!this.sceneData) return;
    
    // Calculate scene bounds
    let minX = Infinity, maxX = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    // Check walls
    if (this.sceneData.walls) {
      this.sceneData.walls.forEach(wall => {
        const positions = [wall.start, wall.end];
        positions.forEach(pos => {
          minX = Math.min(minX, pos[0]);
          maxX = Math.max(maxX, pos[0]);
          minZ = Math.min(minZ, pos[2]);
          maxZ = Math.max(maxZ, pos[2]);
        });
      });
    }
    
    // Check rooms
    if (this.sceneData.rooms) {
      this.sceneData.rooms.forEach(room => {
        const bounds = room.bounds;
        minX = Math.min(minX, bounds.x);
        maxX = Math.max(maxX, bounds.x + bounds.width);
        minZ = Math.min(minZ, bounds.y);
        maxZ = Math.max(maxZ, bounds.y + bounds.height);
      });
    }
    
    // Calculate center and size
    const centerX = (minX + maxX) / 2;
    const centerZ = (minZ + maxZ) / 2;
    const sizeX = Math.max(maxX - minX, 5);
    const sizeZ = Math.max(maxZ - minZ, 5);
    const maxSize = Math.max(sizeX, sizeZ);
    
    // Position camera
    const distance = maxSize * 1.5;
    this.camera.position.set(centerX + distance, distance, centerZ + distance);
    this.camera.lookAt(centerX, 0, centerZ);
    
    console.log(`Scene bounds: X(${minX.toFixed(1)}, ${maxX.toFixed(1)}) Z(${minZ.toFixed(1)}, ${maxZ.toFixed(1)})`);
    console.log(`Camera positioned at: (${this.camera.position.x.toFixed(1)}, ${this.camera.position.y.toFixed(1)}, ${this.camera.position.z.toFixed(1)})`);
  }
  
  clearScene() {
    this.wallGroup.clear();
    this.roomGroup.clear();
    this.furnitureGroup.clear();
    this.doorGroup.clear();
    this.windowGroup.clear();
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
  
  startAnimation() {
    if (this.isAnimating) return;
    this.isAnimating = true;
    this.animate();
  }
  
  animate() {
    if (!this.isAnimating) return;
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene, this.camera);
  }
  
  resetCamera() {
    if (this.sceneData) {
      this.fitCameraToScene();
    } else {
      this.camera.position.set(20, 25, 20);
      this.camera.lookAt(0, 0, 0);
    }
  }
  
  onWindowResize() {
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }
  
  dispose() {
    this.isAnimating = false;
    this.clearScene();
    if (this.container.contains(this.renderer.domElement)) {
      this.container.removeChild(this.renderer.domElement);
    }
    this.renderer.dispose();
  }
}