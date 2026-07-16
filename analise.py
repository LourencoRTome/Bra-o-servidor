"""
=========================================================
ANÁLISE DE EXTREMOS - versão para servidor (sem Tkinter/matplotlib)
=========================================================
Portado de analise_de_extremos.py: toda a lógica de deteção, filtragem,
agrupamento e geração de trajetória. Excluído por decisão do projeto
(ver PROGRESSO.md): tudo o que dependia da janela Tkinter (JanelaAnaliseExtremos)
e dos gráficos matplotlib - aqui só interessam as funções puras que
devolvem estruturas de dados (arrays/dicts), prontas a serializar em JSON.
=========================================================
"""

import numpy as np


def detectar_extremos_todos(t, y):
    """Devolve (indices_maximos, indices_minimos) considerando TODA e
    qualquer mudança de direção da curva como um extremo - nenhum filtro."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    dy = np.diff(y)
    sinal = np.sign(dy)

    sinal_prop = np.empty_like(sinal)
    ultimo = 0.0
    for i, s in enumerate(sinal):
        if s == 0:
            sinal_prop[i] = ultimo
        else:
            sinal_prop[i] = s
            ultimo = s

    mudanca = np.diff(sinal_prop)
    maximos = np.where(mudanca < 0)[0] + 1
    minimos = np.where(mudanca > 0)[0] + 1
    return maximos, minimos


def detectar_extremos_filtrados(t, y, prominencia=1.0, distancia_min_s=0.1):
    """Deteção via scipy.signal.find_peaks, ignorando oscilações com
    amplitude (proeminência) menor que 'prominencia'. Devolve
    (indices_maximos, indices_minimos, erro)."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return None, None, "A biblioteca 'scipy' não está instalada neste ambiente."

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int), np.array([], dtype=int), None

    dt_medio = np.mean(np.diff(t)) if len(t) > 1 else 0.02
    dt_medio = dt_medio if dt_medio > 0 else 0.02
    distancia_amostras = max(1, int(round(distancia_min_s / dt_medio)))

    maximos, _ = find_peaks(y, prominence=prominencia, distance=distancia_amostras)
    minimos, _ = find_peaks(-y, prominence=prominencia, distance=distancia_amostras)
    return maximos, minimos, None


def detectar_estagnacoes(t, y, tolerancia_angular=1.0, duracao_min_s=0.30,
                          diferenca_angular_minima_entre_estagnacoes=0.0,
                          intervalo_max_s_entre_estagnacoes=0.0,
                          indices_extremos=None):
    """Deteta troços quase horizontais e devolve início/fim e ordenada média."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if (len(t) < 2 or tolerancia_angular < 0 or duracao_min_s < 0
            or diferenca_angular_minima_entre_estagnacoes < 0
            or intervalo_max_s_entre_estagnacoes < 0):
        return []
    estagnacoes = []
    inicio = 0
    while inicio < len(y) - 1:
        fim = inicio
        minimo = maximo = y[inicio]
        while fim + 1 < len(y):
            candidato = y[fim + 1]
            novo_minimo, novo_maximo = min(minimo, candidato), max(maximo, candidato)
            if novo_maximo - novo_minimo > tolerancia_angular:
                break
            fim += 1
            minimo, maximo = novo_minimo, novo_maximo
        if fim > inicio and t[fim] - t[inicio] >= duracao_min_s:
            estagnacoes.append({
                "inicio": inicio, "fim": fim,
                "valor": float(np.mean(y[inicio:fim + 1]))
            })
            inicio = fim + 1
        else:
            inicio += 1

    if diferenca_angular_minima_entre_estagnacoes > 0:
        estagnacoes = _fundir_estagnacoes_proximas(
            estagnacoes, t, diferenca_angular_minima_entre_estagnacoes,
            intervalo_max_s_entre_estagnacoes, indices_extremos
        )

    return estagnacoes


def _fundir_estagnacoes_proximas(estagnacoes, t, diferenca_angular_minima,
                                  intervalo_max_s=0.0, indices_extremos=None):
    """Funde estagnações vizinhas cuja diferença de ordenada média seja
    demasiado pequena, quando estão próximas no tempo e sem extremo real
    entre elas."""
    if not estagnacoes:
        return estagnacoes

    conjunto_extremos = set(
        int(i) for i in (indices_extremos if indices_extremos is not None else [])
    )

    fundidas = [dict(estagnacoes[0])]
    for atual in estagnacoes[1:]:
        anterior = fundidas[-1]

        diferenca_valor = abs(atual["valor"] - anterior["valor"])
        intervalo_tempo = float(t[atual["inicio"]] - t[anterior["fim"]])
        tem_extremo_no_meio = any(
            anterior["fim"] < indice < atual["inicio"] for indice in conjunto_extremos
        )

        pode_fundir = (
            diferenca_valor < diferenca_angular_minima
            and not tem_extremo_no_meio
            and (intervalo_max_s <= 0 or intervalo_tempo <= intervalo_max_s)
        )

        if pode_fundir:
            n_anterior = anterior["fim"] - anterior["inicio"] + 1
            n_atual = atual["fim"] - atual["inicio"] + 1
            anterior["valor"] = (
                (anterior["valor"] * n_anterior + atual["valor"] * n_atual)
                / (n_anterior + n_atual)
            )
            anterior["fim"] = atual["fim"]
        else:
            fundidas.append(dict(atual))
    return fundidas


def filtrar_indices_fora_de_estagnacoes(indices, estagnacoes):
    """Remove índices de extremos que caiam dentro de alguma estagnação."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or not estagnacoes:
        return indices
    intervalos = [(e["inicio"], e["fim"]) for e in estagnacoes]

    def dentro_de_estagnacao(indice):
        return any(inicio <= indice <= fim for inicio, fim in intervalos)

    return np.array([i for i in indices if not dentro_de_estagnacao(i)], dtype=int)


def montar_lista_extremos(t, y, indices_maximos, indices_minimos, estagnacoes=None):
    """Junta máximos e mínimos numa única lista de dicts, ordenada no tempo."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    itens = []
    for i in indices_maximos:
        itens.append({"tipo": "Máximo", "indice": int(i), "tempo": float(t[i]), "valor": float(y[i])})
    for i in indices_minimos:
        itens.append({"tipo": "Mínimo", "indice": int(i), "tempo": float(t[i]), "valor": float(y[i])})
    for numero, estagnacao in enumerate(estagnacoes or [], start=1):
        for extremo, indice in (("Início", estagnacao["inicio"]), ("Fim", estagnacao["fim"])):
            itens.append({
                "tipo": "Estagnação", "extremo_estagnacao": extremo,
                "estagnacao_id": numero, "indice": int(indice),
                "tempo": float(t[indice]), "valor": float(estagnacao["valor"]),
                "estagnacao": True,
            })
    itens.sort(key=lambda d: d["tempo"])
    return itens


def limpar_micro_extremos(extremos, diferenca_angular_max=2.0,
                           intervalo_max_s=0.20,
                           velocidade_estavel_max=15.0):
    """Limpa pares máximo/mínimo próximos que não representam uma inversão real."""
    if diferenca_angular_max < 0 or intervalo_max_s < 0 or velocidade_estavel_max < 0:
        raise ValueError("Os limites de limpeza de micro-extremos não podem ser negativos.")

    itens = [dict(item) for item in sorted(extremos, key=lambda item: item["tempo"])]
    resumo = {"patamares": 0, "removidos": 0}
    indice = 0
    while indice < len(itens) - 1:
        primeiro, segundo = itens[indice], itens[indice + 1]
        tipos_opostos = {primeiro["tipo"], segundo["tipo"]} == {"Máximo", "Mínimo"}
        intervalo = float(segundo["tempo"] - primeiro["tempo"])
        amplitude = abs(float(segundo["valor"]) - float(primeiro["valor"]))
        if not tipos_opostos or intervalo <= 0 or intervalo > intervalo_max_s or amplitude > diferenca_angular_max:
            indice += 1
            continue

        e_patamar = False
        if 0 < indice and indice + 2 < len(itens):
            anterior, seguinte = itens[indice - 1], itens[indice + 2]
            dt_antes = float(primeiro["tempo"] - anterior["tempo"])
            dt_depois = float(seguinte["tempo"] - segundo["tempo"])
            if dt_antes > 0 and dt_depois > 0:
                vel_antes = (float(primeiro["valor"]) - float(anterior["valor"])) / dt_antes
                vel_depois = (float(seguinte["valor"]) - float(segundo["valor"])) / dt_depois
                e_patamar = (
                    vel_antes * vel_depois > 0
                    and max(abs(vel_antes), abs(vel_depois)) <= velocidade_estavel_max
                )

        if e_patamar:
            valor_estavel = (float(primeiro["valor"]) + float(segundo["valor"])) / 2.0
            for item in (primeiro, segundo):
                item["valor"] = valor_estavel
                item["micro_estavel"] = True
            resumo["patamares"] += 1
            indice += 2
        else:
            del itens[indice:indice + 2]
            resumo["removidos"] += 1
            indice = max(0, indice - 1)

    return itens, resumo


def agrupar_extremos_por_angulo(extremos, tolerancia_angular=2.0,
                                tolerancia_continuidade=10.0):
    """Agrupa extremos pelo tipo, proximidade angular e continuidade temporal."""
    if tolerancia_angular < 0 or tolerancia_continuidade < 0:
        raise ValueError("As tolerâncias angulares não podem ser negativas.")
    if not extremos:
        return []

    itens = [dict(item) for item in extremos]
    n = len(itens)
    pai = list(range(n))

    def encontrar(indice):
        while pai[indice] != indice:
            pai[indice] = pai[pai[indice]]
            indice = pai[indice]
        return indice

    def unir(a, b):
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a != raiz_b:
            pai[raiz_b] = raiz_a

    for tipo in ("Máximo", "Mínimo"):
        indices_tipo = [i for i, item in enumerate(itens) if item["tipo"] == tipo]

        por_valor = sorted(indices_tipo, key=lambda i: itens[i]["valor"])
        for anterior, atual in zip(por_valor, por_valor[1:]):
            if itens[atual]["valor"] - itens[anterior]["valor"] <= tolerancia_angular:
                unir(anterior, atual)

        por_tempo = sorted(indices_tipo, key=lambda i: itens[i]["tempo"])
        for anterior, atual in zip(por_tempo, por_tempo[1:]):
            if abs(itens[atual]["valor"] - itens[anterior]["valor"]) <= tolerancia_continuidade:
                unir(anterior, atual)

    por_estagnacao = {}
    for i, item in enumerate(itens):
        if item.get("estagnacao"):
            por_estagnacao.setdefault(item["estagnacao_id"], []).append(i)
    for indices in por_estagnacao.values():
        for indice in indices[1:]:
            unir(indices[0], indice)

    componentes = {}
    for i, item in enumerate(itens):
        componentes.setdefault(encontrar(i), []).append(i)

    componentes_ordenados = sorted(
        componentes.values(),
        key=lambda indices: (
            np.mean([itens[i]["valor"] for i in indices]), itens[indices[0]]["tipo"]
        )
    )
    for grupo, indices in enumerate(componentes_ordenados, start=1):
        referencia = float(np.mean([itens[i]["valor"] for i in indices]))
        for i in indices:
            itens[i]["grupo"] = grupo
            itens[i]["angulo_referencia_grupo"] = referencia
            itens[i]["valor_ajustado"] = (
                itens[i]["valor"] if itens[i].get("micro_estavel") else referencia
            )

    return sorted(itens, key=lambda item: item["tempo"])


def agrupar_extremos_temporalmente(extremos, tolerancia_padrao=5.0,
                                   chave_valor="valor_ajustado"):
    """Agrupa no tempo ciclos consecutivos de máximo/mínimo."""
    if tolerancia_padrao < 0:
        raise ValueError("A tolerância do padrão temporal não pode ser negativa.")

    if not extremos:
        return []

    ordenados = sorted(extremos, key=lambda item: item["tempo"])
    pares = [ordenados[inicio:inicio + 2] for inicio in range(0, len(ordenados), 2)]
    resumos = []
    grupo_atual = 0
    valores_maximos, valores_minimos = [], []

    for par in pares:
        por_tipo = {item["tipo"]: item[chave_valor] for item in par}
        par_completo = "Máximo" in por_tipo and "Mínimo" in por_tipo
        par_estagnacao = (
            len(par) == 2
            and all(item.get("estagnacao") for item in par)
            and par[0].get("estagnacao_id") == par[1].get("estagnacao_id")
        )

        novo_grupo = par_estagnacao or not par_completo or grupo_atual == 0
        if not novo_grupo:
            referencia_max = float(np.mean(valores_maximos))
            referencia_min = float(np.mean(valores_minimos))
            distancia_padrao = max(
                abs(por_tipo["Máximo"] - referencia_max),
                abs(por_tipo["Mínimo"] - referencia_min)
            )
            novo_grupo = distancia_padrao > tolerancia_padrao

        if novo_grupo:
            grupo_atual += 1
            valores_maximos = []
            valores_minimos = []
            resumos.append({"grupo": grupo_atual, "itens": [], "pares_completos": 0})

        for item in par:
            item["grupo_temporal"] = grupo_atual
            resumos[-1]["itens"].append(item)
        if par_completo:
            valores_maximos.append(por_tipo["Máximo"])
            valores_minimos.append(por_tipo["Mínimo"])
            resumos[-1]["pares_completos"] += 1

    def e_transicao(resumo):
        return (
            resumo["pares_completos"] == 1 and len(resumo["itens"]) == 2
            and not any(item.get("estagnacao") for item in resumo["itens"])
        )

    def linhas(resumo):
        return {item["grupo"] for item in resumo["itens"]}

    def padrao_compativel(resumo_a, resumo_b):
        if linhas(resumo_a) & linhas(resumo_b):
            return True
        valores_a = {item["tipo"]: item[chave_valor] for item in resumo_a["itens"]}
        valores_b = {item["tipo"]: item[chave_valor] for item in resumo_b["itens"]}
        if "Máximo" not in valores_a or "Mínimo" not in valores_a:
            return False
        if "Máximo" not in valores_b or "Mínimo" not in valores_b:
            return False
        distancia = max(
            abs(valores_a["Máximo"] - valores_b["Máximo"]),
            abs(valores_a["Mínimo"] - valores_b["Mínimo"])
        )
        return distancia <= tolerancia_padrao

    blocos_transicao = []
    indice = 1
    while indice < len(resumos) - 1:
        if not e_transicao(resumos[indice]):
            indice += 1
            continue
        bloco = [indice]
        seguinte = indice + 1
        while (seguinte < len(resumos) - 1 and e_transicao(resumos[seguinte])
               and padrao_compativel(resumos[bloco[-1]], resumos[seguinte])):
            bloco.append(seguinte)
            seguinte += 1
        blocos_transicao.append(bloco)
        indice = seguinte

    reatribuicoes = {}
    for bloco in blocos_transicao:
        anterior = resumos[bloco[0] - 1]
        seguinte = resumos[bloco[-1] + 1]
        linhas_anteriores = linhas(anterior)
        linhas_seguintes = linhas(seguinte)
        grupo_transicao = resumos[bloco[0]]["grupo"]

        for indice_bloco in bloco:
            for item in resumos[indice_bloco]["itens"]:
                reatribuicoes[id(item)] = grupo_transicao
                pertence_anterior = item["grupo"] in linhas_anteriores
                pertence_seguinte = item["grupo"] in linhas_seguintes
                if pertence_anterior and not pertence_seguinte:
                    reatribuicoes[id(item)] = anterior["grupo"]
                elif pertence_seguinte and not pertence_anterior:
                    reatribuicoes[id(item)] = seguinte["grupo"]

    for item in ordenados:
        if id(item) in reatribuicoes:
            item["grupo_temporal"] = reatribuicoes[id(item)]

    grupos_ativos = sorted({item["grupo_temporal"] for item in ordenados})
    nova_numeracao = {grupo_antigo: novo for novo, grupo_antigo in enumerate(grupos_ativos, start=1)}
    for item in ordenados:
        item["grupo_temporal"] = nova_numeracao[item["grupo_temporal"]]

    resumos_finais = []
    for grupo in range(1, len(grupos_ativos) + 1):
        itens = [item for item in ordenados if item["grupo_temporal"] == grupo]
        maximos = [item[chave_valor] for item in itens if item["tipo"] == "Máximo"]
        minimos = [item[chave_valor] for item in itens if item["tipo"] == "Mínimo"]
        pares_completos = sum(
            1 for par in pares
            if len(par) == 2 and {item["grupo_temporal"] for item in par} == {grupo}
        )
        resumos_finais.append({
            "grupo": grupo,
            "itens": itens,
            "pares_completos": pares_completos,
            "inicio": min(item["tempo"] for item in itens),
            "fim": max(item["tempo"] for item in itens),
            "media_maximo": float(np.mean(maximos)) if maximos else None,
            "media_minimo": float(np.mean(minimos)) if minimos else None,
        })

    return resumos_finais


def redistribuir_tempos_por_grupo(resumos):
    """Distribui uniformemente o tempo de cada grupo temporal."""
    if not resumos:
        return []

    ordenados = sorted(resumos, key=lambda resumo: resumo["inicio"])
    proximo_inicio = float(ordenados[0]["inicio"])

    for indice_resumo, resumo in enumerate(ordenados):
        itens = sorted(resumo["itens"], key=lambda item: item["tempo"])
        duracao = float(resumo["fim"] - resumo["inicio"])
        if len(itens) == 1:
            itens[0]["tempo_redistribuido"] = proximo_inicio
        else:
            passo = duracao / (len(itens) - 1)
            for indice, item in enumerate(itens):
                item["tempo_redistribuido"] = proximo_inicio + indice * passo

        resumo["duracao"] = duracao
        resumo["inicio_redistribuido"] = proximo_inicio
        resumo["fim_redistribuido"] = proximo_inicio + duracao
        if indice_resumo < len(ordenados) - 1:
            proximo_resumo = ordenados[indice_resumo + 1]
            pausa = max(0.0, float(proximo_resumo["inicio"] - resumo["fim"]))
            resumo["intervalo_ate_proximo"] = pausa
            proximo_inicio = resumo["fim_redistribuido"] + pausa
        else:
            resumo["intervalo_ate_proximo"] = None

    return ordenados


def gerar_trajetoria_exponencial(extremos, fator_velocidade=1.0, passo_s=0.02,
                                 intensidade=8.0):
    """Gera uma trajetória suave entre extremos ajustados, com velocidade
    tipo logística (lenta perto dos extremos, rápida a meio)."""
    if fator_velocidade <= 0:
        raise ValueError("O fator de velocidade tem de ser superior a zero.")
    if passo_s <= 0:
        raise ValueError("O passo temporal tem de ser superior a zero.")
    if intensidade <= 0:
        raise ValueError("A intensidade exponencial tem de ser superior a zero.")
    if not extremos:
        return []
    if any("tempo_redistribuido" not in item for item in extremos):
        raise ValueError("Calcule primeiro os grupos temporais e os tempos distribuídos.")

    ordenados = sorted(extremos, key=lambda item: item["tempo_redistribuido"])
    inicio_original = float(ordenados[0]["tempo_redistribuido"])
    trajetoria = [
        (inicio_original, float(ordenados[0]["valor_ajustado"]))
    ]

    for anterior, atual in zip(ordenados, ordenados[1:]):
        t0 = inicio_original + (float(anterior["tempo_redistribuido"]) - inicio_original) / fator_velocidade
        t1 = inicio_original + (float(atual["tempo_redistribuido"]) - inicio_original) / fator_velocidade
        y0 = float(anterior["valor_ajustado"])
        y1 = float(atual["valor_ajustado"])
        duracao = t1 - t0
        if duracao <= 0:
            continue
        n_passos = max(1, int(np.ceil(duracao / passo_s)))
        for indice in range(1, n_passos + 1):
            fracao = indice / n_passos
            limite_inferior = 1.0 / (1.0 + np.exp(intensidade))
            limite_superior = 1.0 / (1.0 + np.exp(-intensidade))
            logistica = 1.0 / (1.0 + np.exp(-intensidade * (2.0 * fracao - 1.0)))
            fracao_suave = (logistica - limite_inferior) / (limite_superior - limite_inferior)
            trajetoria.append((t0 + fracao * duracao, y0 + fracao_suave * (y1 - y0)))

    return trajetoria


def analisar_extremos(ts_s, valores, modo="filtrado", prominencia=2.0, distancia_min_s=0.1,
                       limpar_micro=True, micro_diferenca_angular=2.0, micro_intervalo_s=0.20,
                       micro_velocidade_max=15.0, detetar_estagnacao=True,
                       estagnacao_tolerancia=1.0, estagnacao_duracao_min=0.30,
                       estagnacao_diferenca_minima=3.0, estagnacao_intervalo_fusao=0.30,
                       tolerancia_angular=2.0, tolerancia_continuidade=10.0,
                       tolerancia_temporal=5.0, excluir=None):
    """Pipeline completo de análise de extremos, equivalente ao que a
    janela Tkinter fazia botão-a-botão, mas tudo numa só chamada,
    devolvendo um dict pronto a serializar em JSON. 'modo' é 'todos' ou
    'filtrado'."""
    ts_s = np.asarray(ts_s, dtype=float)
    valores = np.asarray(valores, dtype=float)
    if len(valores) < 3:
        raise ValueError("Dados insuficientes para procurar extremos (mínimo 3 pontos).")

    if modo == "todos":
        maximos, minimos = detectar_extremos_todos(ts_s, valores)
    else:
        maximos, minimos, erro = detectar_extremos_filtrados(
            ts_s, valores, prominencia=prominencia, distancia_min_s=distancia_min_s
        )
        if erro:
            raise RuntimeError(erro)

    estagnacoes = []
    if detetar_estagnacao:
        estagnacoes = detectar_estagnacoes(
            ts_s, valores,
            tolerancia_angular=estagnacao_tolerancia,
            duracao_min_s=estagnacao_duracao_min,
            diferenca_angular_minima_entre_estagnacoes=estagnacao_diferenca_minima,
            intervalo_max_s_entre_estagnacoes=estagnacao_intervalo_fusao,
            indices_extremos=np.concatenate([maximos, minimos])
        )
        maximos = filtrar_indices_fora_de_estagnacoes(maximos, estagnacoes)
        minimos = filtrar_indices_fora_de_estagnacoes(minimos, estagnacoes)

    extremos = montar_lista_extremos(ts_s, valores, maximos, minimos, estagnacoes)

    # Filtro de exclusão manual: remove os extremos que o utilizador marcou
    # como ignorados na app. Aplicado ANTES do agrupamento angular e temporal,
    # para que toda a redistribuição de tempos seja recalculada sem eles.
    # Cada entrada de 'excluir' é um dict com:
    #   - extremo normal:  {"tipo": "Máximo"|"Mínimo", "tempo": float}
    #   - estagnação:      {"tipo": "Estagnação", "estagnacao_id": int}
    # Para estagnações excluímos sempre o par completo (início + fim).
    if excluir:
        ids_estagnacao_excluir = {
            int(e["estagnacao_id"]) for e in excluir
            if e.get("tipo") == "Estagnação" and "estagnacao_id" in e
        }
        tempos_excluir = {
            (e["tipo"], round(float(e["tempo"]), 6)) for e in excluir
            if e.get("tipo") in ("Máximo", "Mínimo") and "tempo" in e
        }

        def _nao_excluido(item):
            if item.get("estagnacao") and item.get("estagnacao_id") in ids_estagnacao_excluir:
                return False
            chave = (item["tipo"], round(float(item["tempo"]), 6))
            if chave in tempos_excluir:
                return False
            return True

        extremos = [e for e in extremos if _nao_excluido(e)]

    resumo_micro = {"patamares": 0, "removidos": 0}
    if limpar_micro:
        extremos, resumo_micro = limpar_micro_extremos(
            extremos, diferenca_angular_max=micro_diferenca_angular,
            intervalo_max_s=micro_intervalo_s, velocidade_estavel_max=micro_velocidade_max
        )

    lista_extremos = agrupar_extremos_por_angulo(extremos, tolerancia_angular, tolerancia_continuidade)
    resumos_temporais = agrupar_extremos_temporalmente(lista_extremos, tolerancia_temporal)
    resumos_temporais = redistribuir_tempos_por_grupo(resumos_temporais)

    return {
        "extremos": lista_extremos,
        "grupos_temporais": resumos_temporais,
        "resumo_micro_extremos": resumo_micro,
        "n_maximos": sum(1 for i in lista_extremos if i["tipo"] == "Máximo"),
        "n_minimos": sum(1 for i in lista_extremos if i["tipo"] == "Mínimo"),
        "n_grupos_angulares": len({i["grupo"] for i in lista_extremos}) if lista_extremos else 0,
        "n_grupos_temporais": len(resumos_temporais),
    }
