# -*- coding: utf-8 -*-
"""
medistar_features.py
--------------------
Fonte UNICA da engenharia de atributos do Medistar.

E importado tanto pelo script de treino (medistar_modelo.py) quanto pelo app de
deploy (app.py). Manter a derivacao das features em um so lugar garante que o
paciente avaliado em producao receba EXATAMENTE a mesma transformacao usada no
treino - eliminando o risco de "training/serving skew" (divergencia silenciosa
entre treino e inferencia).

Todas as features derivadas usam SOMENTE variaveis de entrada observaveis no
momento da triagem. Nenhuma usa o rotulo nem o score_risco_total -> sem vazamento.
"""

import numpy as np
import pandas as pd

# Ordem ordinal das classes (gravidade crescente)
ORDEM = ["baixo_risco", "atencao", "alta_prioridade", "critico_territorial"]
TARGET = "prioridade_atendimento"

# Removidas por VAZAMENTO (usada para criar o rotulo)
LEAKAGE = ["score_risco_total"]
# Removidas por serem IDENTIFICADORES / texto livre sem poder preditivo
IDS = ["patient_id", "community_id", "data_atendimento",
       "municipio", "hospital_referencia", "codigo_ibge"]

# Features categoricas (entram no OneHotEncoder)
CAT_FEATURES = ["uf", "sexo", "cluster_sintomas"]

# Os 9 sintomas binarios
SINTOMAS = ["sintoma_febre", "sintoma_tosse", "sintoma_dor_garganta",
            "sintoma_dispneia", "sintoma_dor_corpo", "sintoma_cefaleia",
            "sintoma_manchas_pele", "sintoma_diarreia", "sintoma_vomito"]

# Nomes "bonitos" das features derivadas (para graficos/relatorios)
FEATURES_DERIVADAS = [
    "escore_alerta_vitais",
    "hipoxemia",
    "carga_sintomatica",
    "indice_acesso",
    "indice_ambiental",
    "indice_conectividade_ruim",
    "paciente_vulneravel",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe um DataFrame com as colunas BRUTAS e devolve uma copia com as
    7 features derivadas adicionadas. Vetorizado; funciona para 1 ou N linhas.

    Fundamentacao clinica/territorial de cada feature:

    - escore_alerta_vitais (0-5): contagem de sistemas fisiologicos alterados,
      inspirado em escores de alerta precoce (tipo NEWS simplificado). Cada
      sistema (temperatura, FC, FR, PA, SpO2) conta 1 se estiver fora da faixa.
    - hipoxemia (0/1): SpO2 < 92%, marcador isolado de gravidade respiratoria.
    - carga_sintomatica (0-9): numero total de sintomas relatados.
    - indice_acesso: dificuldade de chegar ao atendimento (distancia + tempo de
      deslocamento + isolamento geografico).
    - indice_ambiental: severidade do contexto climatico/ambiental (enchente +
      chuva + queimadas).
    - indice_conectividade_ruim: precariedade da comunicacao com a central
      (offline + perda de pacotes + latencia + baixa cobertura 4G).
    - paciente_vulneravel (0/1): idade extrema, gestante ou comorbidade.
    """
    out = df.copy()

    # --- 1) Escore de alerta de sinais vitais (0 a 5) ---
    febre_ou_hipotermia = (out["temperatura_c"] >= 38.0) | (out["temperatura_c"] < 35.0)
    fc_alterada = (out["freq_card_bpm"] > 100) | (out["freq_card_bpm"] < 50)
    fr_alterada = (out["freq_resp_irpm"] > 22) | (out["freq_resp_irpm"] < 10)
    pa_alterada = (out["pa_sistolica"] < 90) | (out["pa_sistolica"] > 180)
    spo2_baixa = out["spo2_pct"] < 92
    out["escore_alerta_vitais"] = (
        febre_ou_hipotermia.astype(int)
        + fc_alterada.astype(int)
        + fr_alterada.astype(int)
        + pa_alterada.astype(int)
        + spo2_baixa.astype(int)
    )

    # --- 2) Hipoxemia (flag isolada) ---
    out["hipoxemia"] = (out["spo2_pct"] < 92).astype(int)

    # --- 3) Carga sintomatica (0 a 9) ---
    out["carga_sintomatica"] = out[SINTOMAS].sum(axis=1)

    # --- 4) Indice de dificuldade de acesso ---
    out["indice_acesso"] = (
        out["distancia_hospital_km"] / 100.0
        + out["tempo_deslocamento_min"] / 120.0
        + out["isolamento_geografico_0a1"]
    )

    # --- 5) Indice de severidade ambiental ---
    out["indice_ambiental"] = (
        out["risco_enchente_0a1"]
        + out["chuva_7d_mm"] / 200.0
        + out["focos_queimada_30d"] / 50.0
    )

    # --- 6) Indice de conectividade ruim (maior = pior) ---
    out["indice_conectividade_ruim"] = (
        out["offline_ultimas_24h"]
        + out["perda_pacotes_pct"] / 100.0
        + out["latencia_ms"] / 1000.0
        + (1.0 - out["cobertura_4g_pct_area_local"] / 100.0)
    )

    # --- 7) Paciente vulneravel ---
    out["paciente_vulneravel"] = (
        (out["idade"] >= 65)
        | (out["idade"] <= 5)
        | (out["gestante"] == 1)
        | (out["comorbidade"] == 1)
    ).astype(int)

    return out


def get_feature_lists(df_columns):
    """Dada a lista de colunas do X final (ja com derivadas), separa em
    numericas e categoricas para o ColumnTransformer."""
    cat = [c for c in CAT_FEATURES if c in df_columns]
    num = [c for c in df_columns if c not in cat]
    return num, cat
