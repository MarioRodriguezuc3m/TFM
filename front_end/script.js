// ===================================================================
// ESTADO GLOBAL Y HELPERS COMPARTIDOS
// ===================================================================

// Flags del flujo de carga / audio. El navegador bloquea el audio hasta el
// primer gesto del usuario, así que se coordina la bienvenida con estos flags.
let escenaCargada = false;     // true cuando <a-scene> dispara 'loaded'
let audioDesbloqueado = false; // true tras la primera pulsación de tecla
let bienvenidaDicha = false;   // el mensaje de cargado+instrucciones ya se dijo
let ultimaDescripcion = '';    // última respuesta del backend (para repetir con R)
let ultimoAvisoCargando = 0;   // throttle del aviso "cargando escenario"
let vozPausada = false;        // estado propio de pausa del TTS (toggle con la tecla E)

const MENSAJE_BIENVENIDA =
  'Escenario cargado. Ya puedes explorar. Usa las flechas para mirar alrededor, ' +
  'las teclas W A S D para moverte, mantén pulsada la tecla Q para hablar, ' +
  'pulsa R para repetir la última descripción, ' +
  'pulsa la barra espaciadora para que te describa la escena que estás viendo ' +
  'y pulsa la tecla E para pausar o reanudar la descripción.';

// --- Síntesis de voz (TTS) para avisos cortos (no pausables) ---
function hablar(texto, cancelar) {
  lectura.detener(); // un aviso corto supersede a la descripción pausable
  const u = new SpeechSynthesisUtterance(texto);
  u.lang = 'es-ES';
  if (cancelar) window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

// --- Lectura de la DESCRIPCIÓN, pausable de forma INSTANTÁNEA ---
// La pausa nativa (speechSynthesis.pause) tiene un retardo perceptible porque deja
// terminar el audio ya almacenado en buffer. Para que la pausa sea inmediata usamos
// cancel() (corte al instante) y reanudamos desde la última palabra leída, que
// conocemos por el evento 'boundary'.
const lectura = {
  texto: '', pos: 0, activa: false, gen: 0,
  hablar(texto) {
    this.texto = texto; this.pos = 0; this.activa = true; vozPausada = false;
    this._speak(0);
  },
  _speak(idx) {
    const g = ++this.gen; // token: ignora callbacks de utterances ya superadas
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(this.texto.slice(idx));
    u.lang = 'es-ES';
    u.onboundary = (e) => { if (g === this.gen) this.pos = idx + (e.charIndex || 0); };
    u.onend = () => { if (g === this.gen && !vozPausada) this.activa = false; };
    window.speechSynthesis.speak(u);
  },
  pausar() {
    if (!this.activa || vozPausada) return;
    vozPausada = true;
    this.gen++;                       // invalida el onend de la utterance que cortamos
    window.speechSynthesis.cancel();  // corte instantáneo
  },
  reanudar() {
    if (!this.activa || !vozPausada) return;
    vozPausada = false;
    this._speak(this.pos);            // continúa desde la última palabra leída
  },
  detener() { this.activa = false; vozPausada = false; this.gen++; }
};

// ===================================================================
// MOTOR DE AUDIO (efectos de pasos y cámara, con paneo estéreo)
// ===================================================================
let audioCtx = null;
const buffers = { hierba: null, arena: null, camara: null };

function getAudioCtx() {
  if (!audioCtx) {
    // Compartir el MISMO contexto que A-Frame/THREE (el de los sonidos
    // posicionales). Si hay dos contextos distintos, reanudar uno no reanuda el
    // otro y los ambientales se quedan mudos.
    try {
      audioCtx = THREE.AudioContext.getContext();
    } catch (e) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) audioCtx = new AC();
    }
  }
  return audioCtx;
}

// Carga y decodifica un fichero de sonido en un AudioBuffer. Degrada sin error:
// si el fichero no está, ese efecto simplemente no suena.
async function cargarSonido(clave, url) {
  const ctx = getAudioCtx();
  if (!ctx) return;
  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    buffers[clave] = await ctx.decodeAudioData(await resp.arrayBuffer());
  } catch (e) {
    console.warn(`🔇 No se pudo cargar ${url}: ${e.message}`);
  }
}

let sonidosCargados = false;
function cargarSonidos() {
  if (sonidosCargados) return;
  sonidosCargados = true;
  cargarSonido('hierba', 'sounds/pasos_hierba.mp3');
  cargarSonido('arena',  'sounds/pasos_arena.mp3');
  cargarSonido('camara', 'sounds/camara.mp3');
}

// Bucle de audio con volumen y paneo controlables. El AudioBufferSourceNode no se
// puede reiniciar tras stop(), así que se recrea en cada start().
function crearBucle(volumen) {
  return {
    source: null, gain: null, panner: null, bufferActual: null,
    start(buffer) {
      const ctx = getAudioCtx();
      if (!ctx || !buffer) return;
      if (this.source && this.bufferActual === buffer) return; // ya suena ese buffer
      this.stop();
      this.gain = ctx.createGain();
      this.gain.gain.value = volumen;
      this.panner = ctx.createStereoPanner();
      this.source = ctx.createBufferSource();
      this.source.buffer = buffer;
      this.source.loop = true;
      this.source.connect(this.panner);
      this.panner.connect(this.gain);
      this.gain.connect(ctx.destination);
      this.source.start();
      this.bufferActual = buffer;
    },
    stop() {
      if (this.source) {
        try { this.source.stop(); } catch (e) {}
        this.source.disconnect();
        this.source = null;
      }
      this.bufferActual = null;
    },
    setPan(p) {
      if (this.panner) this.panner.pan.value = Math.max(-1, Math.min(1, p));
    }
  };
}

const sonidoMovimiento = crearBucle(0.4); // pasos (hierba/arena)
const sonidoCamara = crearBucle(0.3);     // ajuste de cámara al girar

// Arranca los sonidos ambientales posicionales (hoguera, palmeras) anclados a
// objetos con el componente `sound` de A-Frame. Necesita un gesto del usuario
// (autoplay) y que la escena esté cargada; si aún no hay emisores, no marca como
// iniciado y se reintenta en el siguiente gesto.
let ambientesIniciados = false;
function iniciarAmbientes() {
  if (ambientesIniciados) return;
  // Reanudar el contexto de audio compartido (el mismo de A-Frame/THREE).
  const ctx = getAudioCtx();
  if (ctx && ctx.state === 'suspended') ctx.resume();
  const emisores = document.querySelectorAll('[sound]');
  if (!emisores.length) return; // escena aún sin emisores: se reintenta luego
  ambientesIniciados = true;
  emisores.forEach(el => {
    const s = el.components && el.components.sound;
    if (s) { try { s.playSound(); } catch (e) {} }
  });
}

// Marca que ya ha habido un gesto del usuario (necesario por la política de
// autoplay), crea/reanuda el AudioContext, carga los efectos y arranca ambientes.
function desbloquearAudio() {
  audioDesbloqueado = true;
  const ctx = getAudioCtx();
  if (ctx && ctx.state === 'suspended') ctx.resume();
  cargarSonidos();
  iniciarAmbientes();
}

// --- Orientación: helpers reutilizados por la captura y por el giro de cámara ---
// Devuelve el punto cardinal o intercardinal (Norte/Noreste/Este/Sureste/Sur/
// Suroeste/Oeste/Noroeste) a partir del ángulo de guiñada (yaw) en grados,
// en sectores de 45°. Normaliza el ángulo al rango (-180, 180].
function orientacionCardinal(yawDeg) {
  let y = yawDeg % 360;
  if (y > 180) y -= 360;
  if (y <= -180) y += 360;
  if (y > -22.5 && y <= 22.5)   return 'Norte';
  if (y > 22.5 && y <= 67.5)    return 'Noreste';
  if (y > 67.5 && y <= 112.5)   return 'Este';
  if (y > 112.5 && y <= 157.5)  return 'Sureste';
  if (y > 157.5 || y <= -157.5) return 'Sur';
  if (y > -157.5 && y <= -112.5) return 'Suroeste';
  if (y > -112.5 && y <= -67.5)  return 'Oeste';
  return 'Noroeste'; // (-67.5, -22.5]
}

// Descripción cualitativa de la vertical (pitch). Vacío si mira al frente.
function verticalCualitativa(pitchDeg) {
  if (pitchDeg > 15) return 'hacia arriba';
  if (pitchDeg < -15) return 'hacia abajo';
  return '';
}

// --- Flujo de bienvenida / aviso de carga ---
function reproducirBienvenida() {
  if (bienvenidaDicha) return;
  bienvenidaDicha = true;
  hablar(MENSAJE_BIENVENIDA, true);
}

function avisarCargando() {
  const now = Date.now();
  if (now - ultimoAvisoCargando > 3000) {
    ultimoAvisoCargando = now;
    hablar('Cargando escenario, espere unos segundos.', true);
  }
}

// Listener "portero": se ejecuta en el primer gesto de teclado de cada pulsación,
// en fase de captura sobre window, ANTES que el resto de manejadores. Desbloquea
// el audio y coordina los avisos hablados de carga/bienvenida.
function onPrimerGesto(e) {
  desbloquearAudio();
  if (!escenaCargada) {
    // Mientras carga, las teclas funcionales están inertes (cada manejador
    // comprueba 'escenaCargada' y se anula); solo disparan este aviso.
    avisarCargando();
    return;
  }
  if (!bienvenidaDicha) {
    // Primera pulsación tras la carga: se consume para reproducir la bienvenida
    // y que no dispare además una acción funcional.
    reproducirBienvenida();
    e.preventDefault();
    e.stopImmediatePropagation();
  }
}
window.addEventListener('keydown', onPrimerGesto, true);

// Aviso de escenario cargado. Si el audio ya está desbloqueado (el usuario pulsó
// alguna tecla durante la carga), se anuncia en el momento; si no, queda pendiente
// hasta la primera pulsación posterior (gestionada por onPrimerGesto).
function onEscenaCargada() {
  escenaCargada = true;
  // Enfocar el elemento oculto dentro de la región role="application": los lectores
  // de pantalla (NVDA/JAWS) entran en modo foco y dejan pasar las teclas a los
  // listeners en lugar de interceptarlas, sin anunciar nada (no tiene nombre).
  const foco = document.getElementById('foco-teclado');
  if (foco) foco.focus();
  if (audioDesbloqueado && !bienvenidaDicha) reproducirBienvenida();
  // Si el usuario ya interactuó durante la carga, arrancar los ambientes ahora
  // que la escena (y sus emisores) ya existen.
  if (audioDesbloqueado) iniciarAmbientes();
}
(function registrarCargaEscena() {
  const sceneEl = document.querySelector('a-scene');
  if (!sceneEl) return;
  if (sceneEl.hasLoaded) onEscenaCargada();
  else sceneEl.addEventListener('loaded', onEscenaCargada);
})();


// ===================================================================
// COMPONENTE: LÍMITES Y COLISIONES
// ===================================================================
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
        radius: parseFloat(el.getAttribute('data-radio')) || 1.0,
        el: el  // Guardamos referencia al elemento para acceder a data-label
      });
    });

    this.lastGoodPosition = new THREE.Vector3();
    this.lastGoodPosition.copy(this.el.object3D.position);

    // Control de cooldown para no repetir el aviso de colisión constantemente
    this.lastCollisionTime = 0;
    this.collisionCooldown = 2000; // ms entre avisos de colisión
    this.lastCollidedLabel = '';   // Último label anunciado
  },

  // Anuncia un aviso por TTS con cooldown para no repetirlo constantemente.
  // 'key' identifica el aviso: si cambia respecto al último, se anuncia aunque
  // no haya pasado el cooldown.
  announce: function (message, key) {
    const now = Date.now();
    if (now - this.lastCollisionTime > this.collisionCooldown || key !== this.lastCollidedLabel) {
      this.lastCollisionTime = now;
      this.lastCollidedLabel = key;

      const utterance = new SpeechSynthesisUtterance(message);
      utterance.lang = 'es-ES';
      utterance.rate = 1.1;
      // Un aviso supersede la descripción pausable y corta el TTS en curso.
      lectura.detener();
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
    }
  },

  // Anuncia por TTS el nombre del objeto con el que se colisiona
  announceCollision: function (label) {
    this.announce(`Colisión con ${label}`, label);
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

    // Si está fuera de la isla, se avisa por voz y se regresa a la última posición válida
    if (isOutOfBounds) {
      this.announce('Has llegado al límite de la isla', '__limite_isla__');
      currentPosition.copy(this.lastGoodPosition);
      return;
    }

    // 3. COMPROBCIÓN CHOQUE CON OBSTÁCULOS
    let hitObstacle = false;
    let collidedLabel = '';

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

        // Obtener el label en español: se busca primero data-sublabel-es / data-label-es
        // (atributos en español definidos en el HTML), con fallback al inglés si no existen.
        // También se sube al padre por si el obstáculo es hijo de un grupo etiquetado.
        const el = obs.el;
        const parent = el.parentElement;
        collidedLabel =
          el.dataset.sublabelEs ||                               // sub-label ES del propio elemento
          el.dataset.labelEs    ||                               // label ES del propio elemento
          (parent && parent.dataset.labelEs) ||                  // label ES del grupo padre
          el.dataset.sublabel   ||                               // fallback: sub-label EN
          el.dataset.label      ||                               // fallback: label EN
          (parent && parent.dataset.label)   ||                  // fallback: label EN del grupo padre
          'objeto desconocido';

        break; // Choque detectado
      }
    }

    if (hitObstacle) {
      // Anunciamos la colisión por voz
      this.announceCollision(collidedLabel);
      // Si el usuario choca con algo, el usuario no avanza
      currentPosition.copy(this.lastGoodPosition);
    } else {
      // Si todo está bien, se actualiza la última posición válida
      this.lastGoodPosition.copy(currentPosition);
    }
  }
});


// ===================================================================
// COMPONENTE: CONTROL DE CÁMARA POR TECLADO (GIRO + MOVIMIENTO)
// ===================================================================
// Flechas izq/dcha → guiñada (yaw) sobre el rig; flechas arriba/abajo → cabeceo
// (pitch) sobre la cámara, con tope. W/A/S/D → mover el rig en la dirección de
// mirada. Todo se maneja con UN listener en fase de captura sobre window, así que
// es independiente del foco del DOM (a diferencia del keyboard-controls de
// aframe-extras, que se descuelga cuando el micrófono mueve el foco y deja al
// lector de pantalla interceptando las teclas). Por eso este componente sustituye
// por completo a movement-controls para el teclado.
AFRAME.registerComponent('keyboard-camera-controls', {

  init: function () {
    this.camera = this.el.querySelector('[camera]');
    this.yawSpeed = THREE.MathUtils.degToRad(60);   // rad/s
    this.pitchSpeed = THREE.MathUtils.degToRad(45);  // rad/s
    this.pitchLimit = THREE.MathUtils.degToRad(50);  // tope de cabeceo
    this.moveSpeed = 4;                              // m/s
    this.keys = {
      left: false, right: false, up: false, down: false,   // giro (flechas)
      fwd: false, back: false, strafeL: false, strafeR: false // movimiento (WASD)
    };
    this.anuncioTimeout = null;

    // Mapa de tecla → acción. Solo estas teclas se gestionan/bloquean.
    this.mapa = {
      ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down',
      KeyW: 'fwd', KeyS: 'back', KeyA: 'strafeL', KeyD: 'strafeR'
    };
    this.esGiro = (code) =>
      code === 'ArrowLeft' || code === 'ArrowRight' ||
      code === 'ArrowUp' || code === 'ArrowDown';

    this.onKeyDown = (e) => {
      const accion = this.mapa[e.code];
      if (!accion) return;
      e.preventDefault();   // evita scroll de página
      e.stopPropagation();  // evita que otros manejadores procesen la tecla
      if (!escenaCargada) return;
      this.keys[accion] = true;
    };

    this.onKeyUp = (e) => {
      const accion = this.mapa[e.code];
      if (!accion) return;
      e.preventDefault();
      e.stopPropagation();
      this.keys[accion] = false;
      if (this.esGiro(e.code)) this.programarAnuncio();
    };

    window.addEventListener('keydown', this.onKeyDown, true);
    window.addEventListener('keyup', this.onKeyUp, true);
  },

  remove: function () {
    window.removeEventListener('keydown', this.onKeyDown, true);
    window.removeEventListener('keyup', this.onKeyUp, true);
    if (this.anuncioTimeout) clearTimeout(this.anuncioTimeout);
  },

  // Anuncia la orientación cuando el usuario deja de girar (con un pequeño
  // debounce para no anunciar a mitad de un giro encadenado).
  programarAnuncio: function () {
    if (this.anuncioTimeout) clearTimeout(this.anuncioTimeout);
    this.anuncioTimeout = setTimeout(() => {
      if (this.keys.left || this.keys.right || this.keys.up || this.keys.down) return;
      const yawDeg = THREE.MathUtils.radToDeg(this.el.object3D.rotation.y);
      const pitchDeg = this.camera ? THREE.MathUtils.radToDeg(this.camera.object3D.rotation.x) : 0;
      const card = orientacionCardinal(yawDeg);
      const vert = verticalCualitativa(pitchDeg);
      hablar(vert ? `Mirando al ${card}, ${vert}` : `Mirando al ${card}`, true);
    }, 150);
  },

  tick: function (t, dt) {
    if (!escenaCargada || !dt) return;
    const seg = dt / 1000;

    // --- GIRO ---
    if (this.keys.left)  this.el.object3D.rotation.y += this.yawSpeed * seg;
    if (this.keys.right) this.el.object3D.rotation.y -= this.yawSpeed * seg;

    if (this.camera) {
      const camRot = this.camera.object3D.rotation;
      if (this.keys.up)   camRot.x = Math.min(this.pitchLimit, camRot.x + this.pitchSpeed * seg);
      if (this.keys.down) camRot.x = Math.max(-this.pitchLimit, camRot.x - this.pitchSpeed * seg);
    }

    // --- MOVIMIENTO (en el plano, según la guiñada actual) ---
    // Con yaw=0 se mira hacia -Z: adelante=(-sinY,0,-cosY), derecha=(cosY,0,-sinY).
    const yaw = this.el.object3D.rotation.y;
    const sinY = Math.sin(yaw), cosY = Math.cos(yaw);
    let dx = 0, dz = 0;
    if (this.keys.fwd)     { dx += -sinY; dz += -cosY; }
    if (this.keys.back)    { dx +=  sinY; dz +=  cosY; }
    if (this.keys.strafeR) { dx +=  cosY; dz += -sinY; }
    if (this.keys.strafeL) { dx += -cosY; dz +=  sinY; }

    if (dx !== 0 || dz !== 0) {
      const len = Math.hypot(dx, dz);
      const pos = this.el.object3D.position;
      pos.x += (dx / len) * this.moveSpeed * seg;
      pos.z += (dz / len) * this.moveSpeed * seg;
    }

    // --- SONIDO DE PASOS (terreno por la z del rig, paneado por A/D) ---
    const moviendo = this.keys.fwd || this.keys.back || this.keys.strafeL || this.keys.strafeR;
    if (moviendo) {
      // Límite hierba/arena en z=-10 (planos suelo-cesped y suelo-arena de index.html)
      const terreno = this.el.object3D.position.z >= -10 ? 'hierba' : 'arena';
      if (buffers[terreno]) sonidoMovimiento.start(buffers[terreno]);
      sonidoMovimiento.setPan((this.keys.strafeR ? 1 : 0) - (this.keys.strafeL ? 1 : 0));
    } else {
      sonidoMovimiento.stop();
    }

    // --- SONIDO DE CÁMARA (mientras se gira, paneado por ←/→) ---
    const girando = this.keys.left || this.keys.right || this.keys.up || this.keys.down;
    if (girando) {
      if (buffers.camara) sonidoCamara.start(buffers.camara);
      sonidoCamara.setPan((this.keys.right ? 1 : 0) - (this.keys.left ? 1 : 0));
    } else {
      sonidoCamara.stop();
    }
  }
});


// ===================================================================
// COMPONENTE: CAPTURA DE ESCENA
// ===================================================================
// Visibilidad de GRUPOS por una caja AJUSTADA A LOS ORÍGENES del grupo y de sus
// sub-objetos (NO a su geometría completa). Un grupo cuenta como visible si esa
// caja entra en el frustum, y la distancia se mide a su punto más cercano. Usar
// los orígenes (y no la geometría) evita que un objeto alto/ancho como la palmera
// del campamento infle la caja y cuele el grupo en cámara aunque sus objetos
// reales queden muy a un lado (a ~77º del centro y a ~19 m). A la vez, la caja
// sigue cubriendo el hueco ENTRE piezas, así que un grupo cercano pero lateral
// (p. ej. el almacén, pegado a un costado) se mantiene si alguna de sus piezas
// entra en el encuadre. 'local' se deja como el origen del grupo en coords de
// cámara para no alterar los campos enriquecidos (C4).
function analizarVisibilidad(el, frustum, camera, cameraPosWorld) {
    const origen = new THREE.Vector3();
    el.object3D.getWorldPosition(origen);
    const local = origen.clone();
    camera.worldToLocal(local);

    // Caja que abarca el origen del grupo y el de cada sub-objeto etiquetado.
    const box = new THREE.Box3();
    box.expandByPoint(origen);
    el.querySelectorAll('[data-sublabel]').forEach(hijo => {
        const p = new THREE.Vector3();
        hijo.object3D.getWorldPosition(p);
        box.expandByPoint(p);
    });

    return {
        enVision: frustum.intersectsBox(box),
        distancia: box.distanceToPoint(cameraPosWorld), // 0 si la cámara está dentro
        local,
    };
}

// Captura la escena, detecta objetos visibles y envía imagen + metadatos al servidor.
AFRAME.registerComponent('captura-escena', {
  init: function () {
    this.scene = this.el.sceneEl;
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

    // Objeto más cercano de toda la escena, BAJANDO al sub-objeto concreto dentro
    // de los grupos (para poder señalar "la pila de cajas" en vez de "el almacén").
    // Se calcula aquí porque el navegador es quien tiene las posiciones de los
    // sub-objetos; al backend se le envía solo el resultado (campo nearest_object),
    // sin ensuciar contained_objects con posiciones por pieza.
    let nearestObject = null;
    let nearestDist = Infinity;
    const considerarNearest = (label, description, group, localPos) => {
        const d = Math.hypot(localPos.x, localPos.z);
        if (d < nearestDist) {
            nearestDist = d;
            nearestObject = {
                label,
                description,
                relative_position: {
                    x: parseFloat(localPos.x.toFixed(2)),
                    y: parseFloat(localPos.y.toFixed(2)),
                    z: parseFloat(localPos.z.toFixed(2))
                }
            };
            if (group) nearestObject.group = group;
        }
    };

    // 1. PROCESAR GRUPOS (entidades con data-tipo="grupo")
    const gruposEtiquetados = document.querySelectorAll('[data-tipo="grupo"]');

    gruposEtiquetados.forEach(grupo => {
        // Visibilidad por caja envolvente: el grupo cuenta si CUALQUIER parte de
        // su volumen entra en cámara (p. ej. un barril del almacén, aunque el
        // origen del grupo quede fuera del encuadre).
        const vis = analizarVisibilidad(grupo, frustum, camera, cameraPosWorld);
        const localPos = vis.local;
        const estaEnVision = vis.enVision;
        const distancia = vis.distancia;

        // Umbral de distancia para grupos
        let distanciaMaxima = 25;

        if (estaEnVision && distancia < distanciaMaxima) {
            // Procesar sub-objetos
            const subObjetos = [];
            const hijosConSubLabel = grupo.querySelectorAll('[data-sublabel]');

            hijosConSubLabel.forEach(hijo => {
                subObjetos.push({
                    // Nombre en español del HTML (data-sublabel-es), con respaldo
                    // al inglés, para que el modelo no tenga que traducir.
                    label: hijo.dataset.sublabelEs || hijo.dataset.sublabel,
                    description: hijo.dataset.subdesc
                });

                // Candidato a "objeto más cercano": cada sub-objeto compite con su
                // PROPIA posición, para poder señalar la pieza concreta del grupo y
                // no el grupo entero. Excepciones pedidas: el grupo mesa no participa,
                // y en el almacén las botellas se omiten. La posición NO se envía por
                // sub-objeto; solo se usa aquí para elegir el más cercano.
                const esMesa = grupo.id === 'grupo-mesa';
                const esBotellaAlmacen =
                    grupo.id === 'grupo-almacen' && hijo.dataset.sublabel === 'Bottle';

                if (!esMesa && !esBotellaAlmacen) {
                    const subWorldPos = new THREE.Vector3();
                    hijo.object3D.getWorldPosition(subWorldPos);
                    const subLocal = subWorldPos.clone();
                    camera.worldToLocal(subLocal);
                    considerarNearest(
                        hijo.dataset.sublabelEs || hijo.dataset.sublabel,
                        hijo.dataset.subdesc,
                        grupo.dataset.labelEs || grupo.dataset.label,
                        subLocal
                    );
                }
            });

            objetosVisibles.push({
                label: grupo.dataset.labelEs || grupo.dataset.label,
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
    // Se excluyen los elementos con data-no-enviar: detalles puramente decorativos
    // (p. ej. la piedra pequeña) que no aportan a la descripción de la escena.
    const objetosIndividuales = document.querySelectorAll('[data-label]:not([data-tipo="grupo"]):not([data-no-enviar])');

    objetosIndividuales.forEach(obj => {
        // Para objetos SUELTOS se usa el test por ORIGEN (como antes): su caja
        // envolvente (p. ej. palmeras altas/anchas o el barco) es tan grande que
        // intersectaría el frustum aunque el objeto esté en la periferia, colando
        // objetos que no se ven realmente. La caja solo se usa para grupos.
        const objWorldPos = new THREE.Vector3();
        obj.object3D.getWorldPosition(objWorldPos);

        const localPos = objWorldPos.clone();
        camera.worldToLocal(localPos);

        let estaEnVision = frustum.containsPoint(objWorldPos);
        const distancia = objWorldPos.distanceTo(cameraPosWorld);

        // Criterio de distancia para objetos pequeños (Arbustos/Rocas)
        let distanciaMaxima = 15; // Smaller to avoid unnecessary decoration clutter
        if (obj.dataset.label === "Pirate Ship") {
            distanciaMaxima = 40;
            // El barco es un hito grande y lejano: más permisivo con el centrado
            // (caja envolvente, lo capta aunque su pivote no esté centrado), pero
            // conservando su criterio de lejanía (distancia por origen, máx 40).
            const box = new THREE.Box3().setFromObject(obj.object3D);
            if (!box.isEmpty()) estaEnVision = frustum.intersectsBox(box);
        }

        // Include object in metadata only if in vision and within max distance
        if (estaEnVision && distancia < distanciaMaxima) {
            objetosVisibles.push({
                label: obj.dataset.labelEs || obj.dataset.label,
                description: obj.dataset.desc,
                relative_position: {
                    x: parseFloat(localPos.x.toFixed(2)),
                    y: parseFloat(localPos.y.toFixed(2)),
                    z: parseFloat(localPos.z.toFixed(2))
                }
            });
            // Objeto suelto: también compite por ser el más cercano (sin grupo).
            considerarNearest(obj.dataset.labelEs || obj.dataset.label, obj.dataset.desc, null, localPos);
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
    const orientacion = orientacionCardinal(yRotation);
    console.log(`🧭 Mirando hacia: ${orientacion} (${yRotation.toFixed(1)}°)`);

    console.log('📦 Objetos detectados:', JSON.stringify(objetosVisibles, null, 2));

    // Enviar al backend
    const canvas = screenshotComponent.getCanvas('perspective');
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    const nombreArchivo = `captura_escena.jpg`;

    fetch('/api/procesar-consulta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            texto: textoUsuario,
            imagen: dataURL,
            nombre: nombreArchivo,
            objetos_visibles: objetosVisibles,
            nearest_object: nearestObject
        })
    })
    .then(res => {
        if(res.ok) {
            console.log("✅ Imagen y datos enviados.");
            return res.json();
        }
        console.error("❌ Error en servidor.");
        hablar("Ha ocurrido un error al procesar la consulta. Inténtalo de nuevo.");
    })
    .then(data => {
        if(data && data.descripcion) {
            console.log("📢 Descripción generada:");
            console.log(data.descripcion);

            // Guardamos la última descripción para poder repetirla (tecla R)
            ultimaDescripcion = data.descripcion;

            // Leemos la respuesta con TTS (pausable con la barra espaciadora)
            lectura.hablar(data.descripcion);
        }
    })
    .catch(err => {
        console.error("❌ Error conexión:", err);
        hablar("No se ha podido conectar con el servidor. Inténtalo de nuevo.");
    });
  }
});


// ===================================================================
// RECONOCIMIENTO DE VOZ (STT) — PUSH-TO-TALK CON BARRA ESPACIADORA
// ===================================================================
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
  let focoRafId = null;

  // Devuelve el foco a la escena (región role="application"). Al arrancar el
  // micrófono el foco puede irse de la escena; si eso pasa, el lector de pantalla
  // sale del modo foco, empieza a leer la página y deja de pasar las teclas.
  const reenfocarEscena = () => {
    const foco = document.getElementById('foco-teclado');
    if (foco && document.activeElement !== foco) foco.focus();
  };

  // Reclama el foco frame a frame durante 'durMs'. El foco se pierde un instante
  // DESPUÉS de arrancar el micro, así que lo recuperamos en cuanto pase, antes de
  // que el lector de pantalla llegue a leer nada.
  const mantenerFocoEscena = (durMs) => {
    const t0 = performance.now();
    const paso = () => {
      reenfocarEscena();
      if (performance.now() - t0 < durMs) focoRafId = requestAnimationFrame(paso);
      else focoRafId = null;
    };
    if (focoRafId) cancelAnimationFrame(focoRafId);
    paso();
  };

  // --- LÓGICA DE INICIO (PULSAR) ---
  const startListening = (e) => {
    if (e) e.preventDefault();
    if (isListening) return;

    finalTranscript = ''; // Limpiamos la frase anterior
    try {
      recognition.start();
    } catch(err) {
      // Ignorar error si el reconocimiento ya estaba iniciado internamente
    }
    // Mantener el foco en la escena mientras arranca el micro: el cambio de foco
    // que dispararía al lector de pantalla ocurre en estos primeros instantes.
    mantenerFocoEscena(700);
  };

  // --- LÓGICA DE FIN (SOLTAR) ---
  const stopListening = (e) => {
    if (e) e.preventDefault();
    if (!isListening) return;

    recognition.stop(); // Detenemos el micro al soltar
  };

  // Tecla Q = pulsar para hablar. Inerte hasta que la escena cargue.
  // (Se usa Q en vez de la barra espaciadora porque el navegador / el lector de
  // pantalla suelen interceptar el Espacio sobre el elemento enfocado.)
  // Se maneja en fase de CAPTURA sobre window con preventDefault + stopPropagation,
  // igual que las flechas de giro, para que el lector de pantalla no la procese.
  window.addEventListener('keydown', (e) => {
    if (e.code !== 'KeyQ') return;
    e.preventDefault();
    e.stopPropagation();
    if (e.repeat || !escenaCargada) return;
    startListening();
  }, true);
  window.addEventListener('keyup', (e) => {
    if (e.code !== 'KeyQ') return;
    e.preventDefault();
    e.stopPropagation();
    stopListening();
  }, true);

  // Tecla E = pausar / reanudar la descripción en curso.
  // En captura sobre window (como Q) para que el lector de pantalla no la procese.
  window.addEventListener('keydown', (e) => {
    if (e.code !== 'KeyE') return;
    e.preventDefault();
    e.stopPropagation();
    if (e.repeat || !escenaCargada) return;
    if (vozPausada) lectura.reanudar();
    else lectura.pausar();
  }, true);

  // Barra espaciadora = lanzar una consulta predefinida de descripción de la
  // escena (equivale a decir por voz "Descríbeme la escena que estoy viendo").
  // En captura sobre window (como Q) para que el lector de pantalla no la procese.
  window.addEventListener('keydown', (e) => {
    if (e.code !== 'Space') return;
    e.preventDefault();
    e.stopPropagation();
    if (e.repeat || !escenaCargada || isListening) return;

    hablar("Procesando su consulta de descripción general");
    const camara = document.querySelector('[captura-escena]');
    if (camara && camara.components['captura-escena']) {
      camara.components['captura-escena'].procesar("Descríbeme la escena que estoy viendo.");
    }
  }, true);

  // Tecla R = repetir la última descripción generada por el modelo.
  document.addEventListener('keydown', (e) => {
    if (e.code === 'KeyR' && !e.repeat) {
      if (!escenaCargada || isListening) return;
      if (ultimaDescripcion) {
        lectura.hablar(ultimaDescripcion); // pausable con la barra espaciadora
      } else {
        hablar("Todavía no hay ninguna descripción que repetir.", true);
      }
    }
  });

  // --- EVENTOS DEL RECONOCIMIENTO ---
  recognition.onstart = () => {
    isListening = true;
    console.log("✅ Micrófono abierto. Mantén pulsado para hablar.");
    reenfocarEscena();
    hablar("Comience a hablar", true); // confirma por voz que ya puede hablar
  };

  recognition.onend = () => {
    isListening = false;
    console.log("🛑 Micrófono cerrado.");
    mantenerFocoEscena(700); // al cerrar el micro el foco también puede moverse

    // Al soltar el botón y cerrarse el micro, procesamos todo el texto acumulado
    let textoProcesado = finalTranscript.trim();

    if (textoProcesado.length > 0) {
      console.log(`%c🗣️ Consulta finalizada: "${textoProcesado}"`,
        'color: #0088cc; font-size: 16px; font-weight: bold; background-color: #f0f8ff; padding: 4px; border-radius: 4px;'
      );

      hablar("Procesando tu consulta...");

      // Capturamos y procesamos automáticamente usando la frase dicha
      const camara = document.querySelector('[captura-escena]');
      if (camara && camara.components['captura-escena']) {
        camara.components['captura-escena'].procesar(textoProcesado);
      }

    } else {
      console.log("🔇 No se detectó ninguna palabra.");
      hablar("No se ha detectado ninguna consulta. Mantén pulsado el botón y habla.");
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
    if (event.error === 'network') {
      // Brave/Firefox/Safari exponen la API pero no tienen el servicio de
      // reconocimiento en la nube: start() siempre falla con 'network'.
      hablar("El reconocimiento de voz no está disponible en este navegador. Por favor, abre este enlace en Google Chrome o Microsoft Edge.");
    } else if (event.error === 'not-allowed') {
      hablar("No hay permiso para usar el micrófono. Actívalo en el navegador y recarga la página.");
    } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
      hablar("Error al escuchar. Por favor, intenta de nuevo.");
    }
    stopListening(); // Resetea el estado en caso de error
  };
}
