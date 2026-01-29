AFRAME.registerComponent('boundary-checker', {
  
  init: function () {
    // Límites de la isla
    this.bounds = {
      x_min: -15, x_max: 15,
      z_min: -22.5, z_max: 2.5
    };
    
    // DETECCIÓN DE OBSTÁCULOS
    this.obstacles = [];
    // Se seleccionan todos los elementos con clase 'obstaculo'
    const els = document.querySelectorAll('.obstaculo');
    
    els.forEach(el => {
      this.obstacles.push({
        object3D: el.object3D, 
        radius: parseFloat(el.getAttribute('data-radio')) || 1.0 
      });
    });

    this.lastGoodPosition = new THREE.Vector3();

    // Inicialmente, guardamos la posición de inicio como válida
    this.lastGoodPosition.copy(this.el.object3D.position);
  },

  tick: function () {
    // 1. OBTENEMOS LA POSICIÓN ACTUAL DEL RIG
    const currentPosition = this.el.object3D.position;

    // 2. COMPROBAMOS SI ESA POSICIÓN ESTÁ FUERA DE LOS LÍMITES
    const isOutOfBounds = 
          currentPosition.x < this.bounds.x_min || 
          currentPosition.x > this.bounds.x_max ||
          currentPosition.z < this.bounds.z_min ||
          currentPosition.z > this.bounds.z_max;

    // Si está fuera de la isla, se regresa a la última posición válida
    if (isOutOfBounds) {
      currentPosition.copy(this.lastGoodPosition);
      return;
    }

    // 3. COMPROBCIÓN CHOQUE CON OBSTÁCULOS
    let hitObstacle = false;

    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i];
      
      // se obtiene la posición del obstáculo
      const obsPos = new THREE.Vector3();
      obs.object3D.getWorldPosition(obsPos);

      const dx = currentPosition.x - obsPos.x;
      const dz = currentPosition.z - obsPos.z;
      const distance = Math.sqrt(dx*dx + dz*dz);

      // Si la distancia es menor que el radio del objeto + un margen se considera un choque
      if (distance < (obs.radius + 0.3)) {
        hitObstacle = true;
        break; // Choque detectado
      }
    }

    if (hitObstacle) {
      // Si el usuario choca con algo, el usuario no avanza
      currentPosition.copy(this.lastGoodPosition);
    } else {
      // Si todo está bien, se actualiza la última posición válida
      this.lastGoodPosition.copy(currentPosition);
    }
  }
});

AFRAME.registerComponent('captura-escena', {
  init: function () {
    this.scene = this.el.sceneEl;
    
    const btn = document.getElementById('btn-captura');
    if (btn) {
      btn.addEventListener('click', () => {
        this.tomarFoto("manual");
      });
    }
  },

  remove: function() {
    if (this.intervalo) clearInterval(this.intervalo);
  },

  tomarFoto: function (origen) {
    const screenshotComponent = this.scene.components.screenshot;
    if (!screenshotComponent) return;

    // 1. Obtener caputra de la escena
    const canvas = screenshotComponent.getCanvas('perspective');

    // 2. Convertir a Base64
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    
    // 3. Generar nombre de archivo
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const nombreArchivo = `captura_${origen}_${timestamp}.jpg`;

    // 4. Se envía imagen al backend de python que se encargará de procesarlo
    fetch('http://localhost:3000/api/guardar-captura', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            imagen: dataURL,
            nombre: nombreArchivo
        })
    })
    .then(response => {
        if (response.ok) {
            console.log(`📡 Enviada al servidor: ${nombreArchivo}`);
        } else {
            console.error("Error al guardar en servidor");
        }
    })
    .catch(error => console.error("Error de conexión:", error));
  }
});