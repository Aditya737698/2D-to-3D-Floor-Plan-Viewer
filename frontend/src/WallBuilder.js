import * as THREE from 'three';

export class WallBuilder {
  constructor(materials) {
    this.materials = materials;
  }
  
  createWall(wall) {
    if (!wall.start || !wall.end) {
      console.warn('Invalid wall data:', wall);
      return new THREE.Group();
    }
    
    const start = new THREE.Vector3(wall.start[0], wall.start[1], wall.start[2]);
    const end = new THREE.Vector3(wall.end[0], wall.end[1], wall.end[2]);
    
    const length = start.distanceTo(end);
    const height = wall.height || 3.0;
    const thickness = wall.thickness || 0.2;
    
    if (length < 0.1) {
      console.warn('Wall too short:', length);
      return new THREE.Group();
    }
    
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
    
    // Create wall group with details
    const wallGroup = new THREE.Group();
    wallGroup.add(wallMesh);
    
    // Add wall details
    this.addWallDetails(wallGroup, wallMesh, length, height, thickness);
    
    return wallGroup;
  }
  
  addWallDetails(wallGroup, wallMesh, length, height, thickness) {
    // Add baseboard
    const baseboardGeometry = new THREE.BoxGeometry(length * 0.98, 0.1, 0.05);
    const baseboard = new THREE.Mesh(baseboardGeometry, this.materials.furniture.wood);
    baseboard.position.set(0, -height/2 + 0.05, thickness/2 + 0.025);
    baseboard.rotation.copy(wallMesh.rotation);
    baseboard.position.add(wallMesh.position);
    baseboard.position.y = 0.05;
    wallGroup.add(baseboard);
    
    // Add crown molding
    const crownGeometry = new THREE.BoxGeometry(length * 0.98, 0.08, 0.04);
    const crown = new THREE.Mesh(crownGeometry, this.materials.furniture.white);
    crown.position.set(0, height/2 - 0.04, thickness/2 + 0.02);
    crown.rotation.copy(wallMesh.rotation);
    crown.position.add(wallMesh.position);
    crown.position.y = height - 0.04;
    wallGroup.add(crown);
    
    // Occasionally add wall decorations
    if (length > 2 && Math.random() > 0.7) {
      this.addWallDecoration(wallGroup, wallMesh, length, height, thickness);
    }
  }
  
  addWallDecoration(wallGroup, wallMesh, length, height, thickness) {
    // Add a simple wall frame/artwork
    const frameWidth = Math.min(length * 0.3, 1.2);
    const frameHeight = frameWidth * 0.7;
    
    const frameGeometry = new THREE.BoxGeometry(frameWidth, frameHeight, 0.03);
    const frame = new THREE.Mesh(frameGeometry, this.materials.furniture.wood);
    
    // Position frame on wall
    frame.position.copy(wallMesh.position);
    frame.position.y = height * 0.6;
    frame.position.z += thickness/2 + 0.015;
    frame.rotation.copy(wallMesh.rotation);
    
    wallGroup.add(frame);
    
    // Add picture inside frame
    const pictureGeometry = new THREE.BoxGeometry(frameWidth * 0.8, frameHeight * 0.8, 0.01);
    const picture = new THREE.Mesh(pictureGeometry, this.materials.furniture.white);
    picture.position.copy(frame.position);
    picture.position.z += 0.01;
    
    wallGroup.add(picture);
  }
  
  // Create wall with openings (doors/windows)
  createWallWithOpenings(wall, openings = []) {
    const wallGroup = this.createWall(wall);
    
    openings.forEach(opening => {
      this.addOpening(wallGroup, wall, opening);
    });
    
    return wallGroup;
  }
  
  addOpening(wallGroup, wall, opening) {
    // This would cut openings in walls for doors/windows
    // For now, just mark the opening positions
    const openingGeometry = new THREE.BoxGeometry(
      opening.width || 1.0,
      opening.height || 2.1,
      0.1
    );
    
    const openingMaterial = new THREE.MeshLambertMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 0.3
    });
    
    const openingMesh = new THREE.Mesh(openingGeometry, openingMaterial);
    openingMesh.position.set(
      opening.position[0],
      opening.position[1] + (opening.height || 2.1) / 2,
      opening.position[2]
    );
    
    wallGroup.add(openingMesh);
  }
}