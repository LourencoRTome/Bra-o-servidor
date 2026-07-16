"""
=========================================================
SERVIDOR - Fase 1: apenas o "relay" WebSocket
=========================================================
Este ficheiro é o ponto de partida. Ainda NÃO tem:
  - gravações (guardar/listar CSVs)
  - análise de dados (suavização, extremos, etc.)

Tem só o essencial para o Recetor e a app comunicarem através
do servidor, tal como hoje comunicam por porta série - mas
agora pela internet.

Como correr isto no teu computador (opcional, para testar
antes de publicar no Render):
  1. pip install fastapi "uvicorn[standard]"
  2. uvicorn main:app --reload
  3. O servidor fica disponível em http://127.0.0.1:8000
=========================================================
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Esta é a "aplicação" - o objeto principal que representa o nosso servidor.
app = FastAPI()

# Guardamos aqui a ligação atual do Recetor e da app, para sabermos para
# onde reenviar as mensagens. Começam a None (ninguém ligado ainda).
ligacao_esp32: WebSocket | None = None
ligacao_app: WebSocket | None = None


@app.get("/")
def raiz():
    """Uma rota simples só para confirmar que o servidor está no ar.
    Se abrires o endereço do servidor num browser, é isto que vês."""
    return {
        "estado": "servidor no ar",
        "esp32_ligado": ligacao_esp32 is not None,
        "app_ligada": ligacao_app is not None,
    }


@app.websocket("/esp32")
async def websocket_esp32(websocket: WebSocket):
    """O Recetor liga-se aqui. Tudo o que o Recetor enviar (as linhas
    'DATA,...' e as respostas 'OK:...') é reenviado para a app, se estiver
    ligada."""
    global ligacao_esp32
    await websocket.accept()
    ligacao_esp32 = websocket
    print("Recetor ligado.")
    try:
        while True:
            mensagem = await websocket.receive_text()
            if ligacao_app is not None:
                await ligacao_app.send_text(mensagem)
    except WebSocketDisconnect:
        print("Recetor desligou-se.")
        ligacao_esp32 = None


@app.websocket("/app")
async def websocket_app(websocket: WebSocket):
    """A app Flutter liga-se aqui. Tudo o que a app enviar (comandos como
    'MAP_ON', 'EIXO,YAW', 'SERVO_SET,120') é reenviado para o Recetor, se
    estiver ligado."""
    global ligacao_app
    await websocket.accept()
    ligacao_app = websocket
    print("App ligada.")
    try:
        while True:
            mensagem = await websocket.receive_text()
            if ligacao_esp32 is not None:
                await ligacao_esp32.send_text(mensagem)
    except WebSocketDisconnect:
        print("App desligou-se.")
        ligacao_app = None
