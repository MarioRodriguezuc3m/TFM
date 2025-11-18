AFRAME.registerComponent('boundary-checker', {
  
  init: function () {
    // Los límites de tu isla (¡estaban correctos!)
    this.bounds = {
      x_min: -15, x_max: 15,
      z_min: -22.5, z_max: 2.5
    };
    
    // Vector para reutilizar
    this.lastGoodPosition = new THREE.Vector3();

    // Guardamos la posición inicial del 'rig' como la primera posición válida
    // 'this.el' es el <a-entity id="cameraRig">
    this.lastGoodPosition.copy(this.el.object3D.position);
  },

  tick: function () {
    // 1. OBTENEMOS LA POSICIÓN ACTUAL DEL RIG
    const currentPosition = this.el.object3D.position;
    console.log(`Posición actual del rig: x=${currentPosition.x.toFixed(2)}, z=${currentPosition.z.toFixed(2)}`);

    // 2. COMPROBAMOS SI ESA POSICIÓN ESTÁ FUERA DE LOS LÍMITES
    const isOutOfBounds = 
          currentPosition.x < this.bounds.x_min || 
          currentPosition.x > this.bounds.x_max ||
          currentPosition.z < this.bounds.z_min ||
          currentPosition.z > this.bounds.z_max;

    // 3. ACTUAMOS EN CONSECUENCIA
    if (isOutOfBounds) {
      // Si está fuera, devolvemos el 'rig' (currentPosition) a su última posición buena
      currentPosition.copy(this.lastGoodPosition);
    } else {
      // Si está dentro, actualizamos la última posición buena con la posición actual del 'rig'
      this.lastGoodPosition.copy(currentPosition);
    }
  }
});