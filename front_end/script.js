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

    // Obtener posición mundial de la cámara
    const cameraPosWorld = new THREE.Vector3();
    camera.getWorldPosition(cameraPosWorld);

    // Obtener rotación y dirección mundial
    const worldQuaternion = new THREE.Quaternion();
    camera.getWorldQuaternion(worldQuaternion);
    
    const worldRotation = new THREE.Euler();
    worldRotation.setFromQuaternion(worldQuaternion, 'YXZ');
    
    const cameraDirection = new THREE.Vector3(0, 0, -1);
    cameraDirection.applyQuaternion(worldQuaternion);
    cameraDirection.normalize();
    
    // Lista de objetos visibles
    const objetosVisibles = [];

    // 1. PROCESAR GRUPOS (entidades con data-tipo="grupo")
    const gruposEtiquetados = document.querySelectorAll('[data-tipo="grupo"]');
    
    gruposEtiquetados.forEach(grupo => {
        const objWorldPos = new THREE.Vector3();
        grupo.object3D.getWorldPosition(objWorldPos);

        // Transformar posición del mundo a posición LOCAL respecto a la cámara
        const localPos = objWorldPos.clone();
        camera.worldToLocal(localPos);

        const estaEnVision = frustum.containsPoint(objWorldPos);
        const distancia = objWorldPos.distanceTo(cameraPosWorld);
        
        // Umbral de distancia para grupos
        let distanciaMaxima = 25;

        if (estaEnVision && distancia < distanciaMaxima) {
            // Procesar sub-objetos
            const subObjetos = [];
            const hijosConSubLabel = grupo.querySelectorAll('[data-sublabel]');
            
            hijosConSubLabel.forEach(hijo => {
                subObjetos.push({
                    etiqueta: hijo.dataset.sublabel,
                    descripcion: hijo.dataset.subdesc
                });
            });

            objetosVisibles.push({
                etiqueta: grupo.dataset.label,
                descripcion: grupo.dataset.desc,
                posicion_relativa: {
                    x: parseFloat(localPos.x.toFixed(2)),
                    y: parseFloat(localPos.y.toFixed(2)),
                    z: parseFloat(localPos.z.toFixed(2)) 
                },
                objetos_contenidos: subObjetos.length > 0 ? subObjetos : undefined
            });
        }
    });

    // 2. PROCESAR OBJETOS INDIVIDUALES (Decorados, elementos sueltos)
    const objetosIndividuales = document.querySelectorAll('[data-label]:not([data-tipo="grupo"])');

    objetosIndividuales.forEach(obj => {
        const objWorldPos = new THREE.Vector3();
        obj.object3D.getWorldPosition(objWorldPos);

        const localPos = objWorldPos.clone();
        camera.worldToLocal(localPos);

        const estaEnVision = frustum.containsPoint(objWorldPos);
        const distancia = objWorldPos.distanceTo(cameraPosWorld);
        
        // Criterio de distancia para objetos pequeños (Arbustos/Rocas)
        let distanciaMaxima = 15; // Más pequeño para no saturar con decorados innecesarios
        if (obj.dataset.label === "Barco Pirata") {
            distanciaMaxima = 40; 
        }

        // Se incluye el bojeto en los metadatos solo si está en visión y dentro de la distancia máxima
        if (estaEnVision && distancia < distanciaMaxima) {
            objetosVisibles.push({
                etiqueta: obj.dataset.label,
                descripcion: obj.dataset.desc,
                posicion_relativa: {
                    x: parseFloat(localPos.x.toFixed(2)),
                    y: parseFloat(localPos.y.toFixed(2)),
                    z: parseFloat(localPos.z.toFixed(2))
                }
            });
        }
    });

    // Logs de debug
    console.log(`👁️ Objetos visibles detectados: ${objetosVisibles.length}`);
    console.log('📍 Posición cámara:', {
        x: cameraPosWorld.x.toFixed(2),
        y: cameraPosWorld.y.toFixed(2),
        z: cameraPosWorld.z.toFixed(2)
    });
    console.log('🔄 Rotación (grados):', {
        x: THREE.MathUtils.radToDeg(worldRotation.x).toFixed(1),
        y: THREE.MathUtils.radToDeg(worldRotation.y).toFixed(1),
        z: THREE.MathUtils.radToDeg(worldRotation.z).toFixed(1)
    });
    
    const yRotation = THREE.MathUtils.radToDeg(worldRotation.y);
    let orientacion = '';
    if (yRotation > -45 && yRotation <= 45) orientacion = 'Norte';
    else if (yRotation > 45 && yRotation <= 135) orientacion = 'Este';
    else if (yRotation > 135 || yRotation <= -135) orientacion = 'Sur';
    else orientacion = 'Oeste';
    console.log(`🧭 Mirando hacia: ${orientacion} (${yRotation.toFixed(1)}°)`);

    console.log('📦 Objetos detectados:', JSON.stringify(objetosVisibles, null, 2));

    // Enviar al backend
    const canvas = screenshotComponent.getCanvas('perspective');
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    const nombreArchivo = `captura_escena.jpg`;

    fetch('http://localhost:3000/api/guardar-captura', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            imagen: dataURL,
            nombre: nombreArchivo,
            objetos_visibles: objetosVisibles
        })
    })
    .then(res => {
        if(res.ok) {
            console.log("✅ Imagen y datos enviados.");
            return res.json();
        }
        else console.error("❌ Error en servidor.");
    })
    .then(data => {
        if(data && data.descripcion) {
            console.log("📢 Descripción generada:");
            console.log(data.descripcion);
        }
    })
    .catch(err => console.error("❌ Error conexión:", err));
  }
});