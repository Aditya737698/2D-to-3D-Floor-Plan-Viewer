import * as THREE from 'three';

export class MaterialLibrary {
  constructor() {
    this.materials = this.createMaterials();
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
        black: new THREE.MeshLambertMaterial({ color: 0x2f2f2f }),
        glass: new THREE.MeshLambertMaterial({ 
          color: 0x87ceeb, 
          transparent: true, 
          opacity: 0.3 
        })
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
  
  // Getter methods for easy access
  get wall() { return this.materials.wall; }
  get floor() { return this.materials.floor; }
  get furniture() { return this.materials.furniture; }
  get door() { return this.materials.door; }
  get window() { return this.materials.window; }
  get ground() { return this.materials.ground; }
  
  // Get floor material by room type
  getFloorMaterial(roomType) {
    const type = roomType.toLowerCase().replace(/\s+/g, '_');
    return this.materials.floor[type] || this.materials.floor.default;
  }
  
  // Get furniture material by type
  getFurnitureMaterial(materialType) {
    return this.materials.furniture[materialType] || this.materials.furniture.wood;
  }
  
  // Create custom colored material
  createColoredMaterial(color, transparent = false, opacity = 1.0) {
    return new THREE.MeshLambertMaterial({
      color: color,
      transparent: transparent,
      opacity: opacity
    });
  }
  
  // Dispose all materials
  dispose() {
    Object.values(this.materials).forEach(material => {
      if (material.dispose) {
        material.dispose();
      } else if (typeof material === 'object') {
        Object.values(material).forEach(mat => {
          if (mat.dispose) mat.dispose();
        });
      }
    });
  }
}