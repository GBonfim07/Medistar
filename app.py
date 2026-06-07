# -*- coding: utf-8 -*-
"""
Medistar - App de Triagem Inteligente (deploy do modelo de ML)
==============================================================
Telemedicina DOMICILIAR para regioes isoladas - sem deslocamento de equipe.

O proprio paciente (ou um cuidador) registra os sintomas e as medicoes obtidas
por DISPOSITIVOS DOMESTICOS / SENSORES IoT (oximetro de dedo, termometro e,
quando disponivel, medidor de pressao). Os dados de territorio e conectividade
sao fornecidos automaticamente pela plataforma (satelite/sensores). O modelo
GERA a prioridade de atendimento, com confianca e explicacao SHAP.

Como rodar localmente:
    pip install -r requirements.txt
    python medistar_modelo.py        # gera medistar_modelo.joblib (1a vez)
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

from medistar_features import engineer_features

# ---------------------------------------------------------------------------
# Config + estilo
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Medistar | Triagem Inteligente",
                   page_icon="\U0001F6F0\uFE0F", layout="wide")

CORES = {
    "baixo_risco": "#16a34a",
    "atencao": "#f59e0b",
    "alta_prioridade": "#ea580c",
    "critico_territorial": "#dc2626",
}
ROTULO = {
    "baixo_risco": "Baixo risco",
    "atencao": "Atencao",
    "alta_prioridade": "Alta prioridade",
    "critico_territorial": "Critico territorial",
}

st.markdown("""
<style>
.main-header{
  background:linear-gradient(110deg,#0d9488 0%,#0e7490 60%,#1e3a8a 100%);
  padding:1.4rem 1.8rem;border-radius:16px;color:#fff;margin-bottom:1.2rem;}
.main-header h1{margin:0;font-size:1.7rem;}
.main-header p{margin:.35rem 0 0;opacity:.92;font-size:.95rem;}
.result-card{padding:1.4rem 1.6rem;border-radius:16px;color:#fff;text-align:center;}
.result-card .lvl{font-size:1.9rem;font-weight:800;letter-spacing:.3px;}
.result-card .conf{font-size:1rem;opacity:.95;margin-top:.2rem;}
.disclaimer{background:#fff7ed;border-left:5px solid #ea580c;padding:.8rem 1rem;
  border-radius:8px;font-size:.86rem;color:#7c2d12;}
.metric-pill{background:#f1f5f9;border-radius:10px;padding:.6rem .8rem;text-align:center;margin-bottom:.6rem;}
.metric-pill .v{font-size:1.25rem;font-weight:700;color:#0f172a;}
.metric-pill .k{font-size:.72rem;color:#475569;text-transform:uppercase;letter-spacing:.4px;}
small.muted{color:#64748b;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carregamento do modelo (cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carregar_bundle(caminho="medistar_modelo.joblib"):
    bundle = joblib.load(caminho)
    explainer = shap.TreeExplainer(bundle["pipeline"].named_steps["clf"])
    return bundle, explainer


try:
    BUNDLE, EXPLAINER = carregar_bundle()
except FileNotFoundError:
    st.error("Arquivo medistar_modelo.joblib nao encontrado. "
             "Rode antes:  python medistar_modelo.py")
    st.stop()

PIPE = BUNDLE["pipeline"]
ORDEM = BUNDLE["ordem"]
CLASSES = BUNDLE["classes"]
RAW_COLS = BUNDLE["raw_input_cols"]
DEFAULTS = BUNDLE["defaults"]
SCAT = BUNDLE["schema_cat"]
FEAT_OUT = BUNDLE["feature_names_out"]
MET = BUNDLE["metricas_teste"]

# Valores fisiologicos de REFERENCIA (usados quando o dispositivo nao esta disponivel)
VITAIS_REFERENCIA = {
    "temperatura_c": 36.5, "freq_card_bpm": 75, "freq_resp_irpm": 16,
    "pa_sistolica": 120, "pa_diastolica": 80, "spo2_pct": 98,
}

# ---------------------------------------------------------------------------
# Cabecalho
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>\U0001F6F0\uFE0F Medistar &mdash; Triagem Inteligente de Atendimento</h1>
  <p>Telemedicina domiciliar para regioes isoladas. O paciente (ou cuidador)
  registra sintomas e medicoes de <b>dispositivos domesticos / sensores IoT</b>;
  o modelo avalia o paciente <b>dentro do seu territorio</b> e gera a priorizacao.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("\U0001F4CA Sobre o modelo")
    st.caption("Random Forest otimizado - classificacao multiclasse")
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='metric-pill'><div class='v'>{MET['acuracia']:.0%}</div>"
                f"<div class='k'>Acuracia</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-pill'><div class='v'>{MET['roc_auc_macro']:.2f}</div>"
                f"<div class='k'>ROC-AUC</div></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown(f"<div class='metric-pill'><div class='v'>{MET['f1_macro']:.2f}</div>"
                f"<div class='k'>F1-macro</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-pill'><div class='v'>{MET['f1_weighted']:.2f}</div>"
                f"<div class='k'>F1-ponder.</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**F1 por classe (teste)**")
    for c in ORDEM:
        st.markdown(
            f"<span style='color:{CORES[c]};font-weight:700'>&#9679;</span> "
            f"{ROTULO[c]}: <b>{MET['f1_por_classe'][c]:.2f}</b>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='disclaimer'>\u26A0\uFE0F Ferramenta de <b>apoio a decisao</b>. "
                "Nao realiza diagnostico nem substitui o julgamento clinico.</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formulario de entrada (sem st.form para os toggles funcionarem dinamicamente)
# ---------------------------------------------------------------------------
st.subheader("Dados do atendimento")
st.caption("Preenchido pelo paciente ou cuidador, a distancia. Os campos de "
           "territorio e conectividade sao normalmente fornecidos pela plataforma "
           "(satelite/sensores) - aqui ficam editaveis para simulacao.")

cA, cB = st.columns(2)
with cA:
    st.markdown("##### \U0001F464 Paciente")
    idade = st.number_input("Idade", 0, 110, 40, 1)
    sexo = st.selectbox("Sexo", SCAT["sexo"])
    gestante = 1 if st.checkbox("Gestante") else 0
    comorbidade = 1 if st.checkbox(
        "Possui comorbidade",
        help="Doenca cronica preexistente (diabetes, hipertensao, asma, etc.).") else 0

    st.markdown("##### \U0001FA7A Medicoes do paciente (dispositivos / IoT)")
    st.caption("Cada grupo corresponde a um aparelho. Informe os que o paciente "
               "tiver - os demais ficam como nao informados.")

    # Termometro (praticamente universal) - ligado por padrao
    tem_termometro = st.toggle(
        "Termometro disponivel", value=True,
        help="Termometro domestico - praticamente universal.")
    if tem_termometro:
        temperatura_c = st.slider(
            "Temperatura (C)", 34.0, 42.0, 36.8, 0.1,
            help="Medida com termometro domestico.")
    else:
        temperatura_c = VITAIS_REFERENCIA["temperatura_c"]
        st.caption("Temperatura nao informada.")

    # Oximetro de dedo -> SpO2 + FC (mesmo aparelho). Opcional, mas LIGADO por padrao:
    # a SpO2 e a 2a variavel mais importante do modelo (tira-la custa ~7,5 p.p. de
    # acuracia), entao fica disponivel por padrao - diferente da pressao arterial.
    tem_oximetro = st.toggle(
        "Oximetro de dedo disponivel", value=True,
        help="Mede a SpO2. Opcional: se o paciente nao tiver, desligue. "
             "Recomendamos manter ligado quando possivel - a SpO2 e um dos sinais "
             "mais importantes para detectar gravidade respiratoria, e o oximetro "
             "de dedo e barato.")
    if tem_oximetro:
        spo2_pct = st.slider(
            "Saturacao de oxigenio - SpO2 (%)", 70, 100, 97, 1,
            help="Leitura do oximetro de dedo (nao e estimativa do paciente). "
                 "Abaixo de 92% indica risco respiratorio.")
    else:
        spo2_pct = VITAIS_REFERENCIA["spo2_pct"]
        st.caption("Sem oximetro: SpO2 nao informada - priorizacao com "
                   "confiabilidade reduzida para quadros respiratorios.")

    # Frequencia cardiaca: pode vir do oximetro, de smartwatch/pulseira ou da
    # contagem do pulso. Impacto moderado no modelo (~2 p.p.); opcional, ligado
    # por padrao por ser de facil obtencao.
    tem_fc = st.toggle(
        "Frequencia cardiaca disponivel", value=True,
        help="Pode ser obtida pelo oximetro de dedo, por smartwatch/pulseira ou "
             "contando o pulso por 1 minuto. Opcional - desligue se nao houver "
             "como medir.")
    if tem_fc:
        freq_card_bpm = st.number_input(
            "Frequencia cardiaca (bpm)", 30, 220, 80, 1,
            help="Batimentos por minuto. Normal em repouso: 60-100.")
    else:
        freq_card_bpm = VITAIS_REFERENCIA["freq_card_bpm"]
        st.caption("Frequencia cardiaca nao informada.")

    tem_pa_fr = st.toggle(
        "Medidor de pressao / medicao de respiracao disponiveis", value=False,
        help="Menos comum em casa. Medidor automatico de pressao e contagem "
             "orientada da respiracao. Deixe desligado se indisponivel.")
    if tem_pa_fr:
        cpa1, cpa2 = st.columns(2)
        pa_sistolica = cpa1.number_input(
            "PA sistolica (mmHg)", 60, 240, 120, 1,
            help="Pressao 'maxima', do medidor automatico. Ex.: 120 em 120/80.")
        pa_diastolica = cpa2.number_input(
            "PA diastolica (mmHg)", 30, 150, 80, 1,
            help="Pressao 'minima', do medidor automatico. Ex.: 80 em 120/80.")
        freq_resp_irpm = st.number_input(
            "Frequencia respiratoria (irpm)", 6, 60, 16, 1,
            help="Respiracoes por minuto: conte os movimentos do torax por 1 min "
                 "(o app pode orientar o paciente). Normal adulto: 12-20.")
    else:
        pa_sistolica = VITAIS_REFERENCIA["pa_sistolica"]
        pa_diastolica = VITAIS_REFERENCIA["pa_diastolica"]
        freq_resp_irpm = VITAIS_REFERENCIA["freq_resp_irpm"]
        st.caption("Sem medidor de pressao/respiracao: PA e FR nao informadas.")

    st.markdown("##### \U0001F912 Sintomas relatados")
    cluster_sintomas = st.selectbox(
        "Cluster de sintomas predominante", SCAT["cluster_sintomas"],
        help="Agrupamento do conjunto de sintomas predominante (ex.: respiratorio, "
             "gastrointestinal).")
    duracao_sintomas_dias = st.number_input(
        "Duracao dos sintomas (dias)", 0, 60, 2, 1,
        help="Ha quantos dias o paciente apresenta os sintomas.")
    sint_cols = st.columns(3)
    nomes_sint = {
        "sintoma_febre": "Febre",
        "sintoma_tosse": "Tosse",
        "sintoma_dor_garganta": "Dor de garganta",
        "sintoma_dispneia": "Falta de ar",
        "sintoma_dor_corpo": "Dor no corpo",
        "sintoma_cefaleia": "Dor de cabeca",
        "sintoma_manchas_pele": "Manchas na pele",
        "sintoma_diarreia": "Diarreia",
        "sintoma_vomito": "Vomito",
    }
    sintomas_val = {}
    for i, (k, lab) in enumerate(nomes_sint.items()):
        sintomas_val[k] = 1 if sint_cols[i % 3].checkbox(lab) else 0

with cB:
    st.markdown("##### \U0001F5FA\uFE0F Territorio e ambiente")
    st.caption("Dados geoespaciais/ambientais da comunidade (origem: satelite e sensores).")
    uf = st.selectbox("UF", SCAT["uf"])
    distancia_hospital_km = st.number_input(
        "Distancia ao hospital (km)", 0, 600, 120, 5,
        help="Distancia ate o hospital de referencia mais proximo.")
    tempo_deslocamento_min = st.number_input(
        "Tempo de deslocamento (min)", 0, 1200, 180, 10,
        help="Tempo estimado para levar o paciente ate o atendimento, se preciso.")
    isolamento_geografico_0a1 = st.slider(
        "Isolamento geografico (0-1)", 0.0, 1.0, 0.4, 0.05,
        help="Quao isolada e a comunidade: 0 = bem conectada por estrada/rio; "
             "1 = muito isolada. Derivado da analise territorial.")
    risco_enchente_0a1 = st.slider(
        "Risco de enchente (0-1)", 0.0, 1.0, 0.2, 0.05,
        help="Probabilidade de enchente na area (dados ambientais/satelite).")
    chuva_7d_mm = st.number_input(
        "Chuva acumulada em 7 dias (mm)", 0.0, 600.0, 40.0, 5.0,
        help="Total de chuva dos ultimos 7 dias (dados meteorologicos).")
    focos_queimada_30d = st.number_input(
        "Focos de queimada (30 dias)", 0, 300, 5, 1,
        help="Numero de focos de queimada detectados por satelite em 30 dias.")

    st.markdown("##### \U0001F4E1 Conectividade")
    st.caption("Qualidade da comunicacao da comunidade com a central medica.")
    cobertura_4g_pct_area_local = st.slider(
        "Cobertura 4G na area (%)", 0.0, 100.0, 60.0, 1.0,
        help="Percentual da area local com sinal de celular 4G.")
    internet_satelite_disponivel = 1 if st.checkbox(
        "Internet via satelite disponivel", value=True,
        help="Se a comunidade tem acesso a internet por satelite.") else 0
    latencia_ms = st.number_input(
        "Latencia (ms)", 0, 5000, 400, 10,
        help="Atraso da conexao em milissegundos. Conexoes via satelite "
             "costumam ter latencia alta (500 ms ou mais).")
    perda_pacotes_pct = st.slider(
        "Perda de pacotes de rede (%)", 0.0, 100.0, 5.0, 0.5,
        help="Percentual dos dados enviados que se PERDEM na transmissao. "
             "Quanto maior, mais instavel a conexao. Acima de ~5% ja prejudica "
             "videochamada e envio de dados.")
    offline_ultimas_24h = 1 if st.checkbox(
        "Ficou offline nas ultimas 24h",
        help="Se a comunidade ficou sem comunicacao em algum momento nas 24h.") else 0

    st.markdown("##### \U0001F465 Contexto coletivo (vigilancia)")
    casos_mesmo_cluster_24h = st.number_input(
        "Casos semelhantes na comunidade (24h)", 1, 30, 1, 1,
        help="Quantos pacientes da mesma comunidade tiveram sintomas parecidos "
             "nas ultimas 24h. 5 ou mais dispara alerta de possivel surto.")
    alerta_comunitario = int(casos_mesmo_cluster_24h >= 5)
    if alerta_comunitario:
        st.warning(f"\U0001F6A8 Alerta comunitario ativo - {casos_mesmo_cluster_24h} "
                   "casos semelhantes em 24h (possivel surto).")
    else:
        st.caption("Sem alerta comunitario (gatilho: 5 ou mais casos em 24h).")

st.markdown("")
enviado = st.button("\U0001F50E Gerar prioridade", width="stretch",
                    type="primary")


# ---------------------------------------------------------------------------
# Inferencia + explicabilidade
# ---------------------------------------------------------------------------
def montar_linha():
    """Monta 1 linha com TODAS as colunas brutas esperadas, partindo dos
    defaults e sobrescrevendo com o que foi informado."""
    linha = dict(DEFAULTS)  # cobre lat/long e quaisquer campos nao expostos
    linha.update({
        "idade": idade, "sexo": sexo, "gestante": gestante, "comorbidade": comorbidade,
        "temperatura_c": temperatura_c, "freq_card_bpm": freq_card_bpm,
        "freq_resp_irpm": freq_resp_irpm, "pa_sistolica": pa_sistolica,
        "pa_diastolica": pa_diastolica, "spo2_pct": spo2_pct,
        "duracao_sintomas_dias": duracao_sintomas_dias,
        "cluster_sintomas": cluster_sintomas, "uf": uf,
        "distancia_hospital_km": distancia_hospital_km,
        "tempo_deslocamento_min": tempo_deslocamento_min,
        "isolamento_geografico_0a1": isolamento_geografico_0a1,
        "risco_enchente_0a1": risco_enchente_0a1, "chuva_7d_mm": chuva_7d_mm,
        "focos_queimada_30d": focos_queimada_30d,
        "cobertura_4g_pct_area_local": cobertura_4g_pct_area_local,
        "internet_satelite_disponivel": internet_satelite_disponivel,
        "latencia_ms": latencia_ms, "perda_pacotes_pct": perda_pacotes_pct,
        "offline_ultimas_24h": offline_ultimas_24h,
        "casos_mesmo_cluster_24h": casos_mesmo_cluster_24h,
        "alerta_comunitario": alerta_comunitario,
    })
    linha.update(sintomas_val)
    return pd.DataFrame([linha])[RAW_COLS]


def shap_local(X_eng, classe_idx):
    """Valores SHAP da instancia para a classe predita (trata os formatos
    possiveis da saida multiclasse)."""
    Xt = PIPE.named_steps["pre"].transform(X_eng)
    sv = EXPLAINER.shap_values(Xt)
    if isinstance(sv, list):
        vals = np.asarray(sv[classe_idx])[0]
    else:
        arr = np.asarray(sv)
        vals = arr[0, :, classe_idx] if arr.ndim == 3 else arr[0]
    return pd.DataFrame({"feature": FEAT_OUT, "shap": vals})


if enviado:
    X_raw = montar_linha()
    X_eng = engineer_features(X_raw)
    pred = PIPE.predict(X_eng)[0]
    proba = PIPE.predict_proba(X_eng)[0]
    conf = float(proba.max())
    classe_idx = list(CLASSES).index(pred)

    st.markdown("## Resultado")
    faltantes = []
    if not tem_termometro:
        faltantes.append("temperatura")
    if not tem_oximetro:
        faltantes.append("SpO2")
    if not tem_fc:
        faltantes.append("frequencia cardiaca")
    if not tem_pa_fr:
        faltantes.append("pressao arterial e respiracao")
    if faltantes:
        st.caption("\u26A0\uFE0F Gerado sem: " + "; ".join(faltantes)
                   + " - interpretar com cautela.")
    col_res, col_prob = st.columns([1, 1.25])

    with col_res:
        st.markdown(
            f"<div class='result-card' style='background:{CORES[pred]}'>"
            f"<div style='font-size:.8rem;opacity:.9;text-transform:uppercase;"
            f"letter-spacing:1px'>Prioridade gerada</div>"
            f"<div class='lvl'>{ROTULO[pred]}</div>"
            f"<div class='conf'>Confianca do modelo: {conf:.0%}</div></div>",
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        d = X_eng.iloc[0]
        st.markdown(
            f"<small class='muted'>Escore de alerta vital: "
            f"<b>{int(d['escore_alerta_vitais'])}/5</b> &middot; "
            f"Carga sintomatica: <b>{int(d['carga_sintomatica'])}/9</b> &middot; "
            f"Hipoxemia: <b>{'sim' if d['hipoxemia'] else 'nao'}</b></small>",
            unsafe_allow_html=True)

    with col_prob:
        st.markdown("**Probabilidade por classe**")
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ordem_plot = [c for c in ORDEM if c in CLASSES]
        vals = [proba[list(CLASSES).index(c)] for c in ordem_plot]
        ax.barh([ROTULO[c] for c in ordem_plot], vals,
                color=[CORES[c] for c in ordem_plot])
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
        ax.set_xlim(0, 1); ax.invert_yaxis()
        ax.set_xlabel("Probabilidade"); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")

    st.markdown("### \U0001F9E0 Por que essa classificacao? (SHAP)")
    st.caption(f"Fatores que mais empurraram a decisao para {ROTULO[pred]} neste "
               f"paciente. Barras vermelhas (direita) aumentam; azuis (esquerda) reduzem.")
    sh = shap_local(X_eng, classe_idx)
    sh = sh.reindex(sh["shap"].abs().sort_values(ascending=False).index).head(8)[::-1]
    fig2, ax2 = plt.subplots(figsize=(8, 3.6))
    cols = ["#dc2626" if v > 0 else "#2563eb" for v in sh["shap"]]
    ax2.barh(sh["feature"], sh["shap"], color=cols)
    ax2.axvline(0, color="#334155", lw=.8)
    ax2.set_xlabel("Contribuicao SHAP para a classe prevista")
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2, width="stretch")

    with st.expander("Ver linha enviada ao modelo (com features derivadas)"):
        # valores convertidos para texto para evitar coluna de tipos mistos
        # (texto + numero) que o serializador Arrow do Streamlit nao aceita
        linha_view = pd.DataFrame({
            "campo": list(X_eng.columns),
            "valor": [str(v) for v in X_eng.iloc[0].values],
        })
        st.dataframe(linha_view, width="stretch", hide_index=True)

    st.markdown("<div class='disclaimer'>\u26A0\uFE0F Resultado de apoio a decisao. "
                "A conduta final e sempre do profissional de saude responsavel."
                "</div>", unsafe_allow_html=True)
else:
    st.info("Preencha os dados do atendimento e clique em Gerar prioridade.")