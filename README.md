# Ejecución

El backend (FastAPI) sirve también el frontend (escena A-Frame), por lo que solo hace falta un proceso. Requisito previo: Ollama corriendo con el modelo `qwen2.5vl:latest` descargado.

```powershell
cd .\back_end\
python -m core.backend
```

Abrir la aplicación en: http://localhost:3000

## Modo prueba remota (enlace público)

Para que una persona externa acceda al escenario virtual mediante un enlace (necesario para las pruebas de usuario), se expone el servidor local con un túnel de Cloudflare. El túnel proporciona HTTPS, imprescindible para que el navegador permita el micrófono y la Web Speech API en accesos remotos.

Instalación (una sola vez):

```powershell
winget install --id Cloudflare.cloudflared
```

Con el backend ya arrancado, en otra terminal:

```powershell
cloudflared tunnel --url http://localhost:3000
```

El comando imprime un enlace del tipo `https://<aleatorio>.trycloudflare.com`: ese es el enlace que se envía a la persona que hace la prueba. El enlace cambia en cada ejecución del túnel y muere al cerrarlo (Ctrl+C). Si el túnel no llega a conectar (redes que bloquean QUIC), usar:

```powershell
cloudflared tunnel --protocol http2 --url http://localhost:3000
```

Indicaciones para la persona que realiza la prueba:

- Usar **Chrome o Edge** (el reconocimiento de voz de la Web Speech API no funciona en Firefox/Safari).
- Conceder el permiso de micrófono cuando el navegador lo pida.
- La primera carga puede tardar: los modelos 3D de la escena se descargan a través del túnel.

Durante la sesión, este PC debe permanecer encendido, sin suspensión y con Ollama, el backend y el túnel en ejecución.
