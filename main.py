"""
=========================================================
SERVIDOR - Fase 1 (relay WebSocket) + Fase 2 (gravações/análise REST)
=========================================================
Fase 1 (mantida tal como estava): o Recetor liga-se a /esp32, a app
liga-se a /app, e tudo o que um lado envia é reencaminhado para o outro.

Fase 2 (novo): guardar gravações (CSV: tempo_ms,angulo_servo) e correr
sobre elas o mesmo tratamento de dados e análise de extremos que a app
Tkinter local fazia - portado para numpy/scipy puro em tratamento.py e
analise.py (sem GPR nem PID, por decisão do projeto - ver PROGRESSO.md).

Armazenamento: em ficheiros (pasta 'dados_gravacoes/'), um .csv com os
pontos e um .json com os metadados por gravação. Isto sobrevive a
"adormecer/acordar" no Render (spin down por inatividade - a mesma
instância só hiberna, o disco mantém-se). Só se perde se o Render criar
uma instância nova a sério (ex.: redeploy) e o disco não for persistente
(no plano Free, o disco NÃO é persistente entre deploys - só entre
sleep/wake da mesma instância). Para garantias reais entre deploys seria
preciso um "Render Disk" (pago) ou uma base de dados externa.

Como correr isto no teu computador:
  1. pip install fastapi "uvicorn[standard]" numpy scipy
  2. uvicorn main:app --reload
  3. O servidor fica disponível em http://127.0.0.1:8000
=========================================================
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tratamento
import analise

app = FastAPI()

# Permite pedidos de qualquer origem (necessário para a app Flutter Web
# correr em localhost e falar com o servidor noutro domínio - sem isto o
# browser bloqueia os pedidos REST com "Failed to fetch"/CORS, mesmo que
# o WebSocket funcione à mesma). Para produção a sério, isto pode ser
# restringido a domínios específicos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Persistência em disco das gravações
# ---------------------------------------------------------------

PASTA_DADOS = Path(__file__).parent / "dados_gravacoes"
PASTA_DADOS.mkdir(exist_ok=True)


def _caminho_csv(id_gravacao: str) -> Path:
    return PASTA_DADOS / f"{id_gravacao}.csv"


def _caminho_meta(id_gravacao: str) -> Path:
    return PASTA_DADOS / f"{id_gravacao}.json"


def _guardar_em_disco(id_gravacao: str, nome: str, criado_em: str, pontos: list[tuple[float, float]]):
    _caminho_csv(id_gravacao).write_text(
        "\n".join(f"{ts},{ang}" for ts, ang in pontos), encoding="utf-8"
    )
    _caminho_meta(id_gravacao).write_text(
        json.dumps({"id": id_gravacao, "nome": nome, "criado_em": criado_em, "n_pontos": len(pontos)}),
        encoding="utf-8"
    )


def _carregar_do_disco() -> dict[str, dict]:
    """Reconstrói o dicionário em memória a partir dos ficheiros já
    guardados, para as gravações sobreviverem a um reinício do processo."""
    carregadas = {}
    for caminho_meta in PASTA_DADOS.glob("*.json"):
        id_gravacao = caminho_meta.stem
        caminho_csv = _caminho_csv(id_gravacao)
        if not caminho_csv.exists():
            continue
        try:
            meta = json.loads(caminho_meta.read_text(encoding="utf-8"))
            pontos = _parse_csv(caminho_csv.read_text(encoding="utf-8"))
            carregadas[id_gravacao] = {
                "id": id_gravacao, "nome": meta["nome"], "criado_em": meta["criado_em"],
                "n_pontos": len(pontos), "pontos": pontos,
            }
        except (ValueError, KeyError, json.JSONDecodeError):
            # Ficheiro corrompido ou incompleto: ignora-o em vez de impedir o arranque do servidor.
            continue
    return carregadas

# ---------------------------------------------------------------
# Fase 1: relay WebSocket (inalterado)
# ---------------------------------------------------------------

ligacao_esp32: WebSocket | None = None
ligacao_app: WebSocket | None = None


@app.get("/")
def raiz():
    """Uma rota simples só para confirmar que o servidor está no ar."""
    return {
        "estado": "servidor no ar",
        "esp32_ligado": ligacao_esp32 is not None,
        "app_ligada": ligacao_app is not None,
        "gravacoes_guardadas": len(gravacoes),
    }


@app.websocket("/esp32")
async def websocket_esp32(websocket: WebSocket):
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


# ---------------------------------------------------------------
# Fase 2: gravações + análise (REST)
# ---------------------------------------------------------------

# Guarda cada gravação em memória: {id: {"nome":..., "criado_em":..., "pontos": [(ts_ms, angulo_servo), ...]}}
# Ao arrancar, recupera-se tudo o que já estiver na pasta 'dados_gravacoes/'.
gravacoes: dict[str, dict] = {}


class GravacaoEntrada(BaseModel):
    """Corpo esperado em POST /gravacoes. 'csv' é o conteúdo do ficheiro
    tal como gravado pela app local: linhas 'tempo_ms,angulo_servo' (sem
    cabeçalho), exatamente como em gravacoes/*.csv."""
    nome: str
    csv: str


class ParametrosAnalise(BaseModel):
    """Todos os parâmetros opcionais, com os mesmos valores por omissão
    usados na janela Tkinter (JanelaAnaliseExtremos / aba Tratamento de Dados)."""
    # tratamento (suavização)
    janela_suavizacao: int = 1
    # deteção de extremos
    modo: str = "filtrado"  # "filtrado" ou "todos"
    prominencia: float = 2.0
    distancia_min_s: float = 0.1
    # limpeza de micro-extremos
    limpar_micro: bool = True
    micro_diferenca_angular: float = 2.0
    micro_intervalo_s: float = 0.20
    micro_velocidade_max: float = 15.0
    # estagnações
    detetar_estagnacao: bool = True
    estagnacao_tolerancia: float = 1.0
    estagnacao_duracao_min: float = 0.30
    estagnacao_diferenca_minima: float = 3.0
    estagnacao_intervalo_fusao: float = 0.30
    # agrupamento
    tolerancia_angular: float = 2.0
    tolerancia_continuidade: float = 10.0
    tolerancia_temporal: float = 5.0


def _parse_csv(texto: str) -> list[tuple[float, float]]:
    """Lê 'tempo_ms,angulo_servo' por linha, tal como _carregar_gravacao_analise
    fazia na app local. Levanta ValueError se algo não for interpretável."""
    pontos = []
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        partes = linha.split(",")
        pontos.append((float(partes[0]), float(partes[1])))
    if len(pontos) < 2:
        raise ValueError("A gravação tem poucos pontos para ser analisada.")
    return pontos


# Carrega o que já estiver em disco (sobrevive a sleep/wake da mesma instância).
gravacoes.update(_carregar_do_disco())
print(f"{len(gravacoes)} gravação(ões) recuperada(s) do disco.")


@app.post("/gravacoes")
def criar_gravacao(entrada: GravacaoEntrada):
    """Guarda uma gravação (CSV tempo_ms,angulo_servo) e devolve o seu id."""
    try:
        pontos = _parse_csv(entrada.csv)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    id_gravacao = uuid.uuid4().hex[:12]
    criado_em = datetime.now(timezone.utc).isoformat()
    gravacoes[id_gravacao] = {
        "id": id_gravacao,
        "nome": entrada.nome,
        "criado_em": criado_em,
        "n_pontos": len(pontos),
        "pontos": pontos,
    }
    _guardar_em_disco(id_gravacao, entrada.nome, criado_em, pontos)
    return {"id": id_gravacao, "n_pontos": len(pontos)}


@app.get("/gravacoes")
def listar_gravacoes():
    """Lista as gravações guardadas (sem os pontos, só metadados)."""
    return [
        {"id": g["id"], "nome": g["nome"], "criado_em": g["criado_em"], "n_pontos": g["n_pontos"]}
        for g in gravacoes.values()
    ]


@app.get("/gravacoes/{id_gravacao}")
def obter_gravacao(id_gravacao: str):
    """Devolve o CSV completo de uma gravação (tempo_ms,angulo_servo)."""
    gravacao = gravacoes.get(id_gravacao)
    if gravacao is None:
        raise HTTPException(status_code=404, detail="Gravação não encontrada.")
    csv = "\n".join(f"{ts},{ang}" for ts, ang in gravacao["pontos"])
    return {
        "id": gravacao["id"], "nome": gravacao["nome"],
        "criado_em": gravacao["criado_em"], "csv": csv,
    }


@app.post("/gravacoes/{id_gravacao}/analise")
def analisar_gravacao(id_gravacao: str, parametros: Optional[ParametrosAnalise] = None):
    """Corre o tratamento de dados (suavização/derivadas/métricas) e a
    análise de extremos sobre uma gravação já guardada, devolvendo tudo
    em JSON. Equivalente a: 'Recalcular' + 'Análise de Extremos' na app local."""
    gravacao = gravacoes.get(id_gravacao)
    if gravacao is None:
        raise HTTPException(status_code=404, detail="Gravação não encontrada.")
    p = parametros or ParametrosAnalise()

    ts_ms = [ponto[0] for ponto in gravacao["pontos"]]
    # dados_gravados guarda o ângulo do servo (0-180); converte para a escala do eixo (-90..90)
    angulos_eixo = [ponto[1] - 90.0 for ponto in gravacao["pontos"]]

    resultado_tratamento = tratamento.processar_gravacao(ts_ms, angulos_eixo, janela=p.janela_suavizacao)

    try:
        resultado_analise = analise.analisar_extremos(
            resultado_tratamento["ts_s"], resultado_tratamento["angulos_suaves"],
            modo=p.modo, prominencia=p.prominencia, distancia_min_s=p.distancia_min_s,
            limpar_micro=p.limpar_micro, micro_diferenca_angular=p.micro_diferenca_angular,
            micro_intervalo_s=p.micro_intervalo_s, micro_velocidade_max=p.micro_velocidade_max,
            detetar_estagnacao=p.detetar_estagnacao, estagnacao_tolerancia=p.estagnacao_tolerancia,
            estagnacao_duracao_min=p.estagnacao_duracao_min,
            estagnacao_diferenca_minima=p.estagnacao_diferenca_minima,
            estagnacao_intervalo_fusao=p.estagnacao_intervalo_fusao,
            tolerancia_angular=p.tolerancia_angular, tolerancia_continuidade=p.tolerancia_continuidade,
            tolerancia_temporal=p.tolerancia_temporal,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "id": id_gravacao,
        "tratamento": resultado_tratamento,
        "analise_extremos": resultado_analise,
    }


class ParametrosTrajetoria(BaseModel):
    """Parâmetros da trajetória suave (velocidade tipo logística), mesmos
    valores por omissão do 'Trajetória com velocidade exponencial para o
    servo' na janela Tkinter. Aceita também os parâmetros de análise
    (herdados), para poder recalcular tudo numa só chamada."""
    analise: ParametrosAnalise = ParametrosAnalise()
    fator_velocidade: float = 1.0
    passo_ms: float = 20.0
    intensidade: float = 8.0


@app.post("/gravacoes/{id_gravacao}/trajetoria")
def gerar_trajetoria(id_gravacao: str, parametros: Optional[ParametrosTrajetoria] = None):
    """Corre a análise de extremos e gera a partir dela a trajetória suave
    (perfil de velocidade tipo logística), devolvendo pontos prontos a
    reproduzir no servo: (tempo_ms, angulo_servo 0-180). Equivalente ao
    botão 'Exportar trajetória suavizada' / 'Reproduzir no servo ao vivo'
    da janela de Análise de Extremos na app Tkinter."""
    gravacao = gravacoes.get(id_gravacao)
    if gravacao is None:
        raise HTTPException(status_code=404, detail="Gravação não encontrada.")
    p = parametros or ParametrosTrajetoria()

    ts_ms = [ponto[0] for ponto in gravacao["pontos"]]
    angulos_eixo = [ponto[1] - 90.0 for ponto in gravacao["pontos"]]
    resultado_tratamento = tratamento.processar_gravacao(ts_ms, angulos_eixo, janela=p.analise.janela_suavizacao)

    try:
        resultado_analise = analise.analisar_extremos(
            resultado_tratamento["ts_s"], resultado_tratamento["angulos_suaves"],
            modo=p.analise.modo, prominencia=p.analise.prominencia, distancia_min_s=p.analise.distancia_min_s,
            limpar_micro=p.analise.limpar_micro, micro_diferenca_angular=p.analise.micro_diferenca_angular,
            micro_intervalo_s=p.analise.micro_intervalo_s, micro_velocidade_max=p.analise.micro_velocidade_max,
            detetar_estagnacao=p.analise.detetar_estagnacao, estagnacao_tolerancia=p.analise.estagnacao_tolerancia,
            estagnacao_duracao_min=p.analise.estagnacao_duracao_min,
            estagnacao_diferenca_minima=p.analise.estagnacao_diferenca_minima,
            estagnacao_intervalo_fusao=p.analise.estagnacao_intervalo_fusao,
            tolerancia_angular=p.analise.tolerancia_angular, tolerancia_continuidade=p.analise.tolerancia_continuidade,
            tolerancia_temporal=p.analise.tolerancia_temporal,
        )
        trajetoria = analise.gerar_trajetoria_exponencial(
            resultado_analise["extremos"],
            fator_velocidade=p.fator_velocidade,
            passo_s=p.passo_ms / 1000.0,
            intensidade=p.intensidade,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(trajetoria) < 2:
        raise HTTPException(status_code=400, detail="São precisos pelo menos dois extremos para gerar a trajetória.")

    pontos_servo = []
    for tempo_s, angulo_eixo in trajetoria:
        tempo_ms = int(round(tempo_s * 1000.0))
        angulo_servo = int(round(min(180.0, max(0.0, angulo_eixo + 90.0))))
        if pontos_servo and tempo_ms == pontos_servo[-1][0]:
            pontos_servo[-1] = (tempo_ms, angulo_servo)
        else:
            pontos_servo.append((tempo_ms, angulo_servo))

    return {
        "id": id_gravacao,
        "n_pontos": len(pontos_servo),
        "pontos": pontos_servo,
        "analise_extremos": resultado_analise,
    }
