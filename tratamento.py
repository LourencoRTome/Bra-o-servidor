"""
=========================================================
TRATAMENTO DE DADOS - versão para servidor (sem Tkinter/matplotlib)
=========================================================
Portado de so_para_ed.py: só a suavização por média móvel, o cálculo de
derivadas (velocidade/aceleração/jerk) e as métricas. Excluído por
decisão do projeto (ver PROGRESSO.md): o módulo experimental de IA/GPR
e o módulo de PID + otimização automática.
=========================================================
"""

import numpy as np


def suavizar(valores, janela):
    """Média móvel simples. 'janela' é o número de amostras (deve ser
    ímpar, >=1)."""
    valores = np.array(valores, dtype=float)
    if janela <= 1 or len(valores) < janela:
        return valores
    kernel = np.ones(janela) / janela
    suave = np.convolve(valores, kernel, mode="same")
    metade = janela // 2
    for i in range(metade):
        suave[i] = np.mean(valores[:i + metade + 1])
        suave[-(i + 1)] = np.mean(valores[-(i + metade + 1):])
    return suave


def calcular_derivadas(ts_ms, angulos):
    """A partir de (tempo_ms, ângulo) calcula velocidade, aceleração e
    jerk angulares."""
    ts_s = np.array(ts_ms, dtype=float) / 1000.0
    angs = np.array(angulos, dtype=float)

    dt = np.diff(ts_s)
    dt[dt <= 0] = 1e-6

    vel = np.diff(angs) / dt
    vel_t = ts_s[:-1] + dt / 2

    if len(vel) > 1:
        dt_v = np.diff(vel_t)
        dt_v[dt_v <= 0] = 1e-6
        acel = np.diff(vel) / dt_v
        acel_t = vel_t[:-1] + dt_v / 2
    else:
        acel, acel_t = np.array([]), np.array([])

    if len(acel) > 1:
        dt_a = np.diff(acel_t)
        dt_a[dt_a <= 0] = 1e-6
        jerk = np.diff(acel) / dt_a
        jerk_t = acel_t[:-1] + dt_a / 2
    else:
        jerk, jerk_t = np.array([]), np.array([])

    return ts_s, angs, vel_t, vel, acel_t, acel, jerk_t, jerk


def calcular_metricas(ts_s, angs, vel, acel, jerk):
    """Devolve um dicionário de métricas numéricas (para serializar em
    JSON), em vez das strings já formatadas usadas na tabela Tkinter."""
    n_pontos = len(ts_s)
    duracao = float(ts_s[-1] - ts_s[0]) if n_pontos > 1 else 0.0
    dt_medio = float(np.mean(np.diff(ts_s))) if n_pontos > 1 else 0.0
    dt_jitter = float(np.std(np.diff(ts_s))) if n_pontos > 1 else 0.0

    metricas = {
        "n_pontos": n_pontos,
        "duracao_s": duracao,
        "taxa_amostragem_hz": (1 / dt_medio if dt_medio > 0 else 0.0),
        "jitter_amostragem_ms": dt_jitter * 1000,
        "angulo_minimo": float(np.min(angs)),
        "angulo_maximo": float(np.max(angs)),
        "amplitude_movimento": float(np.max(angs) - np.min(angs)),
        "angulo_medio": float(np.mean(angs)),
    }
    if len(vel):
        metricas["velocidade_max_dps"] = float(np.max(np.abs(vel)))
        metricas["velocidade_rms_dps"] = float(np.sqrt(np.mean(vel ** 2)))
    if len(acel):
        metricas["aceleracao_max_dps2"] = float(np.max(np.abs(acel)))
        metricas["aceleracao_rms_dps2"] = float(np.sqrt(np.mean(acel ** 2)))
    if len(jerk):
        metricas["jerk_max_dps3"] = float(np.max(np.abs(jerk)))
    return metricas


def processar_gravacao(ts_ms, angulos_eixo, janela=1):
    """Pipeline completo: suaviza, calcula derivadas e métricas, e
    reamostra vel/acel para a base temporal original (alinhado
    ponto-a-ponto com o ângulo, tal como na exportação da app local).

    'angulos_eixo' já deve estar na escala do eixo (-90..90), tal como
    convertido a partir do ângulo do servo (0-180) antes de chamar isto.
    """
    janela = max(1, int(janela))
    if janela % 2 == 0:
        janela += 1

    angulos_suaves = suavizar(angulos_eixo, janela)
    ts_s, angs, vel_t, vel, acel_t, acel, jerk_t, jerk = calcular_derivadas(ts_ms, angulos_suaves)

    vel_interp = np.interp(ts_s, vel_t, vel) if len(vel_t) >= 2 else np.zeros_like(ts_s)
    acel_interp = np.interp(ts_s, acel_t, acel) if len(acel_t) >= 2 else np.zeros_like(ts_s)

    metricas = calcular_metricas(ts_s, angs, vel, acel, jerk)

    return {
        "janela_usada": janela,
        "metricas": metricas,
        "ts_ms": list(ts_ms),
        "ts_s": ts_s.tolist(),
        "angulos": angs.tolist(),
        "angulos_suaves": angulos_suaves.tolist(),
        "velocidade": vel_interp.tolist(),
        "aceleracao": acel_interp.tolist(),
    }
