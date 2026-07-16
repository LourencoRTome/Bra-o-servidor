# Servidor do Braço Robótico (Fase 1)

Servidor FastAPI que faz de "estafeta" (relay) entre o ESP32 Recetor e a
app Flutter, usando WebSocket em vez da porta série.

## Rotas

- `GET /` — confirma que o servidor está no ar.
- `WS /esp32` — o Recetor liga-se aqui.
- `WS /app` — a app liga-se aqui.

Tudo o que chega de um lado é reenviado para o outro, tal como hoje o
Python fala com o Recetor por porta série.

## Correr localmente (opcional)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Depois abre http://127.0.0.1:8000 no browser.
