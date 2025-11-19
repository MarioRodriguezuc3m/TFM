AFRAME.registerComponent('boundary-checker', {
  
  init: function () {
    // Los límites de tu isla (¡estaban correctos!)
    this.bounds = {
      x_min: -15, x_max: 15,
      z_min: -22.5, z_max: 2.5
    };
    
    // --- NUEVO: DETECCIÓN DE OBSTÁCULOS ---
    this.obstacles = [];
    // Seleccionamos solo los elementos que marcamos como "obstaculo" en el HTML
    // (Las palmeras, rocas, cajas... pero NO los arbustos)
    const els = document.querySelectorAll('.obstaculo');
    
    els.forEach(el => {
      this.obstacles.push({
        // Guardamos el objeto 3D para leer su posición
        object3D: el.object3D, 
        // Leemos el radio personalizado (o usamos 1m por defecto)
        radius: parseFloat(el.getAttribute('data-radio')) || 1.0 
      });
    });
    // --------------------------------------

    // Vector para reutilizar
    this.lastGoodPosition = new THREE.Vector3();

    // Guardamos la posición inicial del 'rig' como la primera posición válida
    // 'this.el' es el <a-entity id="cameraRig">
    this.lastGoodPosition.copy(this.el.object3D.position);
  },

  tick: function () {
    // 1. OBTENEMOS LA POSICIÓN ACTUAL DEL RIG
    const currentPosition = this.el.object3D.position;
    // console.log(`Posición actual: x=${currentPosition.x.toFixed(2)}, z=${currentPosition.z.toFixed(2)}`);

    // 2. COMPROBAMOS SI ESA POSICIÓN ESTÁ FUERA DE LOS LÍMITES
    const isOutOfBounds = 
          currentPosition.x < this.bounds.x_min || 
          currentPosition.x > this.bounds.x_max ||
          currentPosition.z < this.bounds.z_min ||
          currentPosition.z > this.bounds.z_max;

    // Si ya está fuera de la isla, lo devolvemos y terminamos
    if (isOutOfBounds) {
      currentPosition.copy(this.lastGoodPosition);
      return;
    }

    // 3. NUEVO: COMPROBAMOS CHOQUE CON OBSTÁCULOS
    let hitObstacle = false;

    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i];
      
      // Obtenemos posición del obstáculo en el mundo
      // (Usamos getWorldPosition por si el obstáculo está dentro de un grupo)
      const obsPos = new THREE.Vector3();
      obs.object3D.getWorldPosition(obsPos);

      // Distancia simple en 2D (ignoramos altura Y)
      const dx = currentPosition.x - obsPos.x;
      const dz = currentPosition.z - obsPos.z;
      const distance = Math.sqrt(dx*dx + dz*dz);

      // Si la distancia es menor que el radio del objeto + un margen del cuerpo (0.3m)
      if (distance < (obs.radius + 0.3)) {
        hitObstacle = true;
        break; // ¡Choque detectado! No hace falta seguir mirando
      }
    }

    // 4. ACTUAMOS EN CONSECUENCIA
    if (hitObstacle) {
      // Si chocas con algo, te quedas donde estabas antes (efecto muro)
      currentPosition.copy(this.lastGoodPosition);
    } else {
      // Si estás dentro de la isla y sin chocar, esta posición es válida
      this.lastGoodPosition.copy(currentPosition);
    }
  }
});