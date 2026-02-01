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
        radius: parseFloat(el.getAttribute('data-radio')) || 1.0  // Radio por defecto: 1.0
      });
    });

    this.lastGoodPosition = new THREE.Vector3();
    this.lastGoodPosition.copy(this.el.object3D.position);
  },

  // Cada frame: valida límites y colisiones, restaura posición si falla alguna
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

// Captura la escena, detecta objetos visibles y envía imagen + metadatos al servidor.
AFRAME.registerComponent('captura-escena', {
  init: function () {
    this.scene = this.el.sceneEl;
    
    // Enlaza el botón de captura con tomarFoto()
    const btn = document.getElementById('btn-captura');
    if (btn) {
      btn.addEventListener('click', () => {
        this.tomarFoto();
      });
    }
  },

  remove: function() {
    if (this.intervalo) clearInterval(this.intervalo);
  },

  tomarFoto: function () {
    const screenshotComponent = this.scene.components.screenshot;
    if (!screenshotComponent) return;

    const camera = this.scene.camera;
    
    camera.updateMatrix();
    camera.updateMatrixWorld();

    const frustum = new THREE.Frustum();
    
    const projScreenMatrix = new THREE.Matrix4();
    projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    
    frustum.setFromProjectionMatrix(projScreenMatrix);


    // A continuación se filtran los objetos etiquetados que están visibles en la imagen con un bucle for ---
    const elementosEtiquetados = document.querySelectorAll('[data-label]');
    const infoEscena = [];

    const cameraPosWorld = new THREE.Vector3();
    camera.getWorldPosition(cameraPosWorld);

    elementosEtiquetados.forEach(el => {
        const object3D = el.object3D;
        const worldPos = new THREE.Vector3();
        object3D.getWorldPosition(worldPos);

        // Verifica si el objeto está dentro del angulo de visión de la cámara
        const estaEnVision = frustum.containsPoint(worldPos);

        // Distancia entre la cámara y el objeto
        const distancia = worldPos.distanceTo(cameraPosWorld);
        
        // Umbral de distancia por defecto (25 m)
        let distanciaMaxima = 25;
        
        // Se permite una mayor distancia al barco ya que es el elemento más grande de la escena
        if (el.dataset.label === "Barco Pirata") {
            distanciaMaxima = 40;
        }

        // Se incluye el bojeto en los metadatos solo si está en visión y dentro de la distancia máxima
        if (estaEnVision && distancia < distanciaMaxima) {
            
            infoEscena.push({
                etiqueta: el.dataset.label,
                descripcion: el.dataset.desc,
                posicion: {
                    x: worldPos.x.toFixed(2),
                    y: worldPos.y.toFixed(2),
                    z: worldPos.z.toFixed(2)
                },
                distancia: distancia.toFixed(1) + "m"
            });
        }
    });

    console.log(`👁️ Objetos visibles detectados: ${infoEscena.length}`);


    // Se envía al backend la imagen en base64 + metadatos de objetos visibles
    const canvas = screenshotComponent.getCanvas('perspective');
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    const nombreArchivo = `captura_escena.jpg`;
    console.log('Nombre archivo:', nombreArchivo);

    
    fetch('http://localhost:3000/api/guardar-captura', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            imagen: dataURL,
            nombre: nombreArchivo,
            metadatos: infoEscena
        })
    })
    .then(res => {
        if(res.ok) console.log("✅ Imagen y datos filtrados enviados.");
        else console.error("❌ Error en servidor.");
    })
    .catch(err => console.error("❌ Error conexión:", err));
  }
});