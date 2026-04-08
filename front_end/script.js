AFRAME.registerComponent('boundary-checker', {
  
  init: function () {
    // Límites de la isla
    this.bounds = {
      x_min: -15, x_max: 15,
      z_min: -22.5, z_max: 2.5
    };
    
    // DETECCIÓN DE OBSTÁCULOS
    this.obstacles =[];
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
    // (Mantenemos el comentario original, pero el código del botón manual ha sido retirado 
    // para que se ejecute automáticamente desde la voz)
  },

  remove: function() {
    if (this.intervalo) clearInterval(this.intervalo);
  },

  procesar: function (textoUsuario) {
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
    const objetosVisibles =[];

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
                    label: hijo.dataset.sublabel,
                    description: hijo.dataset.subdesc
                });
            });

            objetosVisibles.push({
                label: grupo.dataset.label,
                description: grupo.dataset.desc,
                relative_position: {
                    x: parseFloat(localPos.x.toFixed(2)),
                    y: parseFloat(localPos.y.toFixed(2)),
                    z: parseFloat(localPos.z.toFixed(2)) 
                },
                contained_objects: subObjetos.length > 0 ? subObjetos : undefined
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
        let distanciaMaxima = 15; // Smaller to avoid unnecessary decoration clutter
        if (obj.dataset.label === "Pirate Ship") {
            distanciaMaxima = 40; 
        }

        // Include object in metadata only if in vision and within max distance
        if (estaEnVision && distancia < distanciaMaxima) {
            objetosVisibles.push({
                label: obj.dataset.label,
                description: obj.dataset.desc,
                relative_position: {
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

    fetch('http://localhost:3000/api/procesar-consulta', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            texto: textoUsuario,
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
            
            // Leemos la respuesta con TTS
            const utterance = new SpeechSynthesisUtterance(data.descripcion);
            utterance.lang = 'es-ES';
            window.speechSynthesis.speak(utterance);
        }
    })
    .catch(err => console.error("❌ Error conexión:", err));
  }
});

// --- FUNCIONALIDAD DE STT CON WEB SPEECH API (PUSH-TO-TALK) ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
  console.warn("⚠️ Este navegador no soporta la Web Speech API.");
} else {
  const recognition = new SpeechRecognition();
  recognition.lang = 'es-ES';        
  
  // CRÍTICO PARA PTT: continuous = true. 
  // Esto evita que el navegador corte el micro si el usuario hace una pausa mientras mantiene el botón pulsado.
  recognition.continuous = true;     
  recognition.interimResults = true; 

  let isListening = false;
  let finalTranscript = ''; 

  const btn = document.getElementById('btn-stt');
  
  // Función para la síntesis de voz (TTS)
  const speak = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-ES'; 
    window.speechSynthesis.speak(utterance);
  };

  // --- LÓGICA DE INICIO (PULSAR) ---
  const startListening = (e) => {
    // Evitar múltiples activaciones si ya está pulsado
    if (e) e.preventDefault(); 
    if (isListening) return;

    finalTranscript = ''; // Limpiamos la frase anterior
    try {
      recognition.start();
    } catch(err) {
      // Ignorar error si el reconocimiento ya estaba iniciado internamente
    }
    
    btn.innerHTML = "🎙️ ESCUCHANDO... (Suelta para enviar)";
    btn.style.backgroundColor = "#ffcccc";
    btn.style.transform = "scale(0.95)"; // Efecto visual de pulsado
  };

  // --- LÓGICA DE FIN (SOLTAR) ---
  const stopListening = (e) => {
    if (e) e.preventDefault();
    if (!isListening) return;

    recognition.stop(); // Detenemos el micro al soltar
    btn.innerHTML = "🎙️ Mantén pulsado para hablar";
    btn.style.backgroundColor = "#ffffff";
    btn.style.transform = "scale(1)"; // Restaurar tamaño original
  };

  // 1. Eventos de Ratón
  btn.addEventListener('mousedown', startListening);
  btn.addEventListener('mouseup', stopListening);
  btn.addEventListener('mouseleave', stopListening); // Por si el ratón sale del botón mientras pulsa

  // 2. Eventos Táctiles (Móviles / Tablets)
  btn.addEventListener('touchstart', startListening, {passive: false});
  btn.addEventListener('touchend', stopListening);

  // 3. Evento de Teclado (Barra Espaciadora) para accesibilidad web
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat) {
      startListening();
    }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
      stopListening();
    }
  });

  // --- EVENTOS DEL RECONOCIMIENTO ---
  recognition.onstart = () => {
    isListening = true;
    console.log("✅ Micrófono abierto. Mantén pulsado para hablar.");
    // Opcional: speak("Escuchando"); // Podría pisar la voz del usuario, mejor un sonido corto (beep) si lo tienes
  };

  recognition.onend = () => {
    isListening = false;
    console.log("🛑 Micrófono cerrado.");
    
    // Al soltar el botón y cerrarse el micro, procesamos todo el texto acumulado
    let textoProcesado = finalTranscript.trim();
    
    if (textoProcesado.length > 0) {
      console.log(`%c🗣️ Consulta finalizada: "${textoProcesado}"`,
        'color: #0088cc; font-size: 16px; font-weight: bold; background-color: #f0f8ff; padding: 4px; border-radius: 4px;'
      );
      
      speak("Procesando tu consulta...");
      
      // AQUÍ ES DONDE LLAMARÍAS A TU BACKEND (ROUTER DE INTENCIONES)
      // fetch('http://localhost:3000/api/intencion', { ... body: { texto: textoProcesado } })
      
      // Capturamos y procesamos automáticamente usando la frase dicha
      const camara = document.querySelector('[captura-escena]');
      if (camara && camara.components['captura-escena']) {
        camara.components['captura-escena'].procesar(textoProcesado);
      }
      
    } else {
      console.log("🔇 No se detectó ninguna palabra.");
    }
  };

  recognition.onresult = (event) => {
    let interimTranscript = '';
    
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript + ' ';
      } else {
        interimTranscript += transcript;
      }
    }
    
    if (interimTranscript) {
      console.log(`💬 (parcial): "${interimTranscript.trim()}"`);
    }
  };

  recognition.onerror = (event) => {
    console.error("❌ Error STT:", event.error);
    if (event.error !== 'no-speech' && event.error !== 'aborted') {
      speak("Error al escuchar. Por favor, intenta de nuevo.");
    }
    stopListening(); // Resetea el botón en caso de error
  };
  
  // Ajuste inicial del texto del botón
  btn.innerHTML = "🎙️ Mantén pulsado para hablar";
}