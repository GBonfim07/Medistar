# -*- coding: utf-8 -*-
"""
Medistar - Modelo de IA/ML para Priorizacao Inteligente de Atendimento
Plataforma de Telemedicina e Vigilancia em Saude para Regioes Isoladas

Tarefa: classificacao supervisionada MULTICLASSE.
Prever `prioridade_atendimento` (4 niveis ordinais de gravidade) a partir do
quadro clinico do paciente somado ao contexto territorial, ambiental e de
conectividade da comunidade.

Cobre os itens da rubrica:
  1. Definicao do problema
  2. Dataset (sintetico)
  3. Tratamento, preparacao e ENGENHARIA DE ATRIBUTOS
  4. Tecnicas de treinamento + validacao e comparacao
  5. Avaliacao por metricas de desempenho
  6. Avaliacao visual do modelo
  7. Interpretabilidade com SHAP
  8. SERIALIZACAO do modelo para deploy (joblib)  <-- novo
  9. Inferencia (o modelo gerando a prioridade)
 10. Limitacoes e consideracoes honestas

Como rodar:
    pip install -r requirements.txt
    python medistar_modelo.py
Gera:
    - medistar_modelo.joblib   (pipeline treinado + schema de inferencia)
    - figuras_medistar/*.png   (graficos da avaliacao)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
import shap

# Engenharia de atributos compartilhada com o app de deploy
from src.medistar_features import (engineer_features, get_feature_lists, ORDEM,
                               TARGET, LEAKAGE, IDS, CAT_FEATURES, SINTOMAS,
                               FEATURES_DERIVADAS)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

FIGDIR = "figuras_medistar"
os.makedirs(FIGDIR, exist_ok=True)
SHOW = os.environ.get("MEDISTAR_SHOW", "0") == "1"  # exporte =1 p/ abrir janelas


def finish(fig, nome):
    """Salva a figura em PNG e (opcionalmente) exibe."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, nome), bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)


print("=" * 70)
print("1. DEFINICAO DO PROBLEMA")
print("=" * 70)
print("""
Em comunidades isoladas (ribeirinhas, rurais, indigenas, de floresta) o risco
real de um caso NAO depende so dos sintomas, mas tambem do contexto: distancia
ate o hospital, isolamento geografico, clima severo e qualidade da conexao.

Formulacao: classificacao supervisionada multiclasse (4 classes ordinais)
Alvo `prioridade_atendimento`:
  1. baixo_risco         - sinais estaveis e facil acesso
  2. atencao             - sintomas moderados ou algum fator de risco
  3. alta_prioridade     - sinais relevantes + dificuldade de acesso
  4. critico_territorial - gravidade clinica agravada pelo contexto territorial
""")

# ============================================================================
# 2. DATASET (SINTETICO)
# ============================================================================
print("=" * 70)
print("2. DATASET")
print("=" * 70)
DATA = "medistar_pacientes_telemedicina_sintetico.csv"
df = pd.read_csv(DATA)
print(f"Dimensoes: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"UFs: {df['uf'].nunique()} | Comunidades: {df['community_id'].nunique()} "
      f"| Municipios: {df['municipio'].nunique()}")
print(f"Valores faltantes: {int(df.isnull().sum().sum())} | "
      f"Duplicatas: {int(df.duplicated().sum())}")

dist = df[TARGET].value_counts().reindex(ORDEM)
print("\nDistribuicao da variavel-alvo:")
for k, v in dist.items():
    print(f"  {k:22s}: {v:4d}  ({v/len(df)*100:4.1f}%)")

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=dist.values, y=dist.index, hue=dist.index,
            palette="YlOrRd", legend=False, ax=ax)
for i, v in enumerate(dist.values):
    ax.text(v + 4, i, str(v), va="center")
ax.set_title("Distribuicao da variavel-alvo (prioridade_atendimento)")
ax.set_xlabel("No de atendimentos"); ax.set_ylabel("")
finish(fig, "01_distribuicao_alvo.png")

# ============================================================================
# 3. TRATAMENTO, PREPARACAO E ENGENHARIA DE ATRIBUTOS
# ============================================================================
print("\n" + "=" * 70)
print("3. TRATAMENTO, PREPARACAO E ENGENHARIA DE ATRIBUTOS")
print("=" * 70)
print("""Decisao sobre VAZAMENTO (data leakage), apos investigacao:

 - score_risco_total: e a SOMA ponderada de fatores de risco e foi usada para
   CRIAR o rotulo (faixas de corte fixas, sem sobreposicao entre classes).
   Mante-lo daria ~100% artificial -> REMOVIDO.
 - casos_mesmo_cluster_24h e alerta_comunitario: correlacionam ~0,34 com o
   score, ou seja, sao PARCELAS do score - exatamente como temperatura,
   distancia e isolamento tambem sao. Como removemos a SOMA (o score), manter
   as parcelas individuais NAO e vazamento; e o problema legitimo (o modelo
   precisa reaprender os pesos). Por isso sao MANTIDAS. Obs.: alerta_comunitario
   e perfeitamente derivavel de casos_mesmo_cluster_24h (alerta=1 <=> casos>=5);
   a redundancia e absorvida pelo Random Forest sem prejuizo.
""")

# 3.1 Remocao de vazamento e identificadores
X_raw = df.drop(columns=LEAKAGE + IDS + [TARGET])
y = df[TARGET].astype("category").cat.set_categories(ORDEM, ordered=True)
raw_input_cols = list(X_raw.columns)  # colunas BRUTAS que o app deve fornecer
print(f"Removidas (vazamento)      : {LEAKAGE}")
print(f"Removidas (identificadores): {IDS}")

# 3.2 ENGENHARIA DE ATRIBUTOS (7 novas features derivadas)
X = engineer_features(X_raw)
print(f"\nFeatures derivadas criadas ({len(FEATURES_DERIVADAS)}): {FEATURES_DERIVADAS}")

num_features, cat_features = get_feature_lists(X.columns)
print(f"Total de features -> numericas: {len(num_features)} | "
      f"categoricas: {len(cat_features)}")

# 3.3 Split estratificado 75/25
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
print(f"Treino: {X_tr.shape[0]} | Teste: {X_te.shape[0]}")

# 3.4 Pre-processador (fit so no treino -> sem vazamento de escala)
pre = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
])

# ============================================================================
# 4. TREINAMENTO + VALIDACAO CRUZADA + COMPARACAO
# ============================================================================
print("\n" + "=" * 70)
print("4. TREINAMENTO (5 modelos + validacao cruzada 5-fold)")
print("=" * 70)
modelos = {
    "Regressao Logistica": LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=RANDOM_STATE),
    "Arvore de Decisao": DecisionTreeClassifier(class_weight="balanced",
                                                random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

resultados = []
for nome, clf in modelos.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    resultados.append({"modelo": nome, "f1_macro_cv": scores.mean(), "std": scores.std()})
    print(f"  {nome:22s} F1-macro CV = {scores.mean():.3f} +/- {scores.std():.3f}")

res_df = (pd.DataFrame(resultados)
          .sort_values("f1_macro_cv", ascending=False).reset_index(drop=True))
print("\nRanking (validacao cruzada):\n", res_df.to_string(index=False))

# Grafico comparativo dos modelos
fig, ax = plt.subplots(figsize=(7.5, 4))
sns.barplot(data=res_df, x="f1_macro_cv", y="modelo", hue="modelo",
            palette="crest", legend=False, ax=ax)
ax.errorbar(res_df["f1_macro_cv"], range(len(res_df)),
            xerr=res_df["std"], fmt="none", ecolor="black", capsize=4)
for i, v in enumerate(res_df["f1_macro_cv"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center")
ax.set_xlim(0, 1); ax.set_title("Comparacao de modelos (F1-macro, 5-fold CV)")
ax.set_xlabel("F1-macro"); ax.set_ylabel("")
finish(fig, "02_comparacao_modelos.png")

# 4.1 Escolha do modelo final.
# OBS HONESTA: na CV, Regressao Logistica (~0.76) e Random Forest empatam no topo
# - esperado, pois o rotulo sintetico segue uma regra quase-LINEAR (soma + cortes),
# cenario em que um modelo linear ja vai muito bem. Entre os dois empatados,
# escolhemos o Random Forest por: (a) desempenho equivalente apos tuning;
# (b) robustez a relacoes NAO-lineares, que e o que importa em dados REAIS (ver
# limitacao 2); (c) interpretabilidade nativa (importancias + SHAP TreeExplainer).
print("\nOtimizando Random Forest com GridSearchCV...")
rf_pipe = Pipeline([("pre", pre),
                    ("clf", RandomForestClassifier(class_weight="balanced",
                                                   random_state=RANDOM_STATE, n_jobs=-1))])
grid = {
    "clf__n_estimators": [300, 500],
    "clf__max_depth": [None, 20],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", 0.5],
}
gs = GridSearchCV(rf_pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
gs.fit(X_tr, y_tr)
print("Melhores hiperparametros:", gs.best_params_)
print("F1-macro CV (otimizado):", round(gs.best_score_, 3))
modelo_final = gs.best_estimator_

# ============================================================================
# 5. METRICAS DE DESEMPENHO (conjunto de teste)
# ============================================================================
print("\n" + "=" * 70)
print("5. METRICAS DE DESEMPENHO (conjunto de teste)")
print("=" * 70)
y_pred = modelo_final.predict(X_te)
y_te_str = y_te.astype(str)
y_pred_str = pd.Series(y_pred, index=y_te.index).astype(str)

acc = accuracy_score(y_te_str, y_pred_str)
p_m, r_m, f_m, _ = precision_recall_fscore_support(y_te_str, y_pred_str, average="macro")
p_w, r_w, f_w, _ = precision_recall_fscore_support(y_te_str, y_pred_str, average="weighted")
print(f"Acuracia : {acc:.3f}")
print(f"Macro     -> Precisao {p_m:.3f} | Revocacao {r_m:.3f} | F1 {f_m:.3f}")
print(f"Ponderada -> Precisao {p_w:.3f} | Revocacao {r_w:.3f} | F1 {f_w:.3f}")
print("\nclassification_report:\n", classification_report(y_te_str, y_pred_str))

classes = list(modelo_final.classes_)
y_proba = modelo_final.predict_proba(X_te)
roc_auc_macro = roc_auc_score(y_te_str, y_proba, multi_class="ovr",
                              average="macro", labels=classes)
print(f"ROC-AUC (macro, One-vs-Rest): {roc_auc_macro:.3f}")

# ============================================================================
# 6. AVALIACAO VISUAL DO MODELO
# ============================================================================
print("\n" + "=" * 70)
print("6. AVALIACAO VISUAL")
print("=" * 70)

# 6.1 Matriz de confusao (cor = % da linha, numero = contagem)
cm = confusion_matrix(y_te_str, y_pred_str, labels=ORDEM)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=ORDEM, yticklabels=ORDEM, ax=ax,
            cbar_kws={"label": "proporcao por linha"})
ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
ax.set_title("Matriz de Confusao")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
finish(fig, "03_matriz_confusao.png")

# 6.2 Curvas ROC por classe (OvR)
y_te_bin = label_binarize(y_te_str, classes=classes)
fig, ax = plt.subplots(figsize=(7, 5.5))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Taxa de Falsos Positivos"); ax.set_ylabel("Taxa de Verdadeiros Positivos")
ax.set_title("Curvas ROC (One-vs-Rest)"); ax.legend(loc="lower right")
finish(fig, "04_curvas_roc.png")

# 6.3 F1 por classe
_, _, f1_classe, _ = precision_recall_fscore_support(y_te_str, y_pred_str, labels=ORDEM)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=list(f1_classe), y=ORDEM, hue=ORDEM, palette="viridis", legend=False, ax=ax)
for i, v in enumerate(f1_classe):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center")
ax.set_xlim(0, 1); ax.set_title("F1-score por classe"); ax.set_xlabel("F1")
finish(fig, "05_f1_por_classe.png")

# 6.4 Importancia das variaveis (Random Forest)
ohe = modelo_final.named_steps["pre"].named_transformers_["cat"]
feat_names = num_features + list(ohe.get_feature_names_out(cat_features))
importances = modelo_final.named_steps["clf"].feature_importances_
imp_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
          .sort_values("importance", ascending=False).head(15))
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.barplot(x="importance", y="feature", data=imp_df, hue="feature",
            palette="rocket", legend=False, ax=ax)
ax.set_title("Top 15 variaveis mais importantes (Random Forest)")
ax.set_xlabel("Importancia"); ax.set_ylabel("")
finish(fig, "06_importancia_variaveis.png")

# 6.5 SHAP - explicabilidade para a classe critica
print("Calculando valores SHAP (pode levar alguns segundos)...")
X_te_t = modelo_final.named_steps["pre"].transform(X_te)
explainer = shap.TreeExplainer(modelo_final.named_steps["clf"])
sv = explainer.shap_values(X_te_t[:200])
idx_crit = classes.index("critico_territorial")
shap_vals = sv[idx_crit] if isinstance(sv, list) else sv[:, :, idx_crit]
fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_vals, X_te_t[:200], feature_names=feat_names,
                  show=False, max_display=12)
plt.title("SHAP - contribuicao das features para 'critico_territorial'")
finish(fig, "07_shap_summary.png")

# ============================================================================
# 8. SERIALIZACAO DO MODELO PARA DEPLOY
# ============================================================================
print("\n" + "=" * 70)
print("8. SERIALIZACAO DO MODELO (joblib) PARA O APP DE DEPLOY")
print("=" * 70)

# Schema de inferencia: tudo que o app precisa para montar uma linha valida,
# preencher defaults e configurar widgets - SEM depender do CSV.
num_brutas = [c for c in raw_input_cols if c not in CAT_FEATURES]
schema_num = {}
for c in num_brutas:
    s = df[c]
    schema_num[c] = {
        "min": float(s.min()), "max": float(s.max()),
        "median": float(s.median()), "mean": float(s.mean()),
        "is_binary": bool(set(s.unique()) <= {0, 1}),
    }
schema_cat = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_FEATURES}
defaults = {}
for c in raw_input_cols:
    defaults[c] = (df[c].median() if c in num_brutas else df[c].mode().iloc[0])

bundle = {
    "pipeline": modelo_final,
    "ordem": ORDEM,
    "classes": classes,
    "raw_input_cols": raw_input_cols,     # colunas brutas esperadas
    "num_features": num_features,         # pos-engenharia (ColumnTransformer)
    "cat_features": cat_features,
    "feature_names_out": feat_names,      # nomes pos-transform (para SHAP no app)
    "schema_num": schema_num,             # min/max/median/binary por num bruta
    "schema_cat": schema_cat,             # categorias possiveis
    "defaults": defaults,                 # default de cada coluna bruta
    "sintomas": SINTOMAS,
    "metricas_teste": {                   # para exibir no app/README
        "acuracia": round(float(acc), 3),
        "f1_macro": round(float(f_m), 3),
        "f1_weighted": round(float(f_w), 3),
        "roc_auc_macro": round(float(roc_auc_macro), 3),
        "f1_por_classe": {c: round(float(v), 3) for c, v in zip(ORDEM, f1_classe)},
    },
    "sklearn_version": __import__("sklearn").__version__,
}
joblib.dump(bundle, "medistar_modelo.joblib")
print("Modelo + schema salvos em: medistar_modelo.joblib")
print("Metricas embutidas:", bundle["metricas_teste"])

# ============================================================================
# 9. INFERENCIA - O MODELO GERANDO A PRIORIDADE
# ============================================================================
print("\n" + "=" * 70)
print("9. INFERENCIA: o modelo PRODUZINDO a coluna prioridade_atendimento")
print("=" * 70)
print("""No dataset a coluna ja vem preenchida porque ela e o GABARITO usado para
treinar. Na pratica, para um paciente NOVO (sem prioridade definida), e o modelo
quem gera essa coluna. Demonstracao com pacientes do teste (nunca vistos no treino):
""")
n = 8
amostra = X_te.head(n).copy()
reais = y_te.head(n).astype(str).values
preds = modelo_final.predict(amostra)
conf = modelo_final.predict_proba(amostra).max(axis=1)
demo = pd.DataFrame({
    "idade": amostra["idade"].values,
    "spo2_pct": amostra["spo2_pct"].values,
    "dist_hosp_km": amostra["distancia_hospital_km"].values,
    "escore_vitais": amostra["escore_alerta_vitais"].values,
    "prioridade_REAL": reais,
    "prioridade_GERADA": preds,
    "confianca_%": (conf * 100).round(1),
})
print(demo.to_string(index=False))

# ============================================================================
# 10. LIMITACOES E CONSIDERACOES HONESTAS
# ============================================================================
print("\n" + "=" * 70)
print("10. LIMITACOES E CONSIDERACOES HONESTAS")
print("=" * 70)
print("""
1) Dado sintetico com rotulo derivado de regra: a prioridade foi construida a
   partir do score_risco_total (cortes fixos). Para evitar vazamento, o score foi
   REMOVIDO. O modelo "redescobre" a logica a partir das parcelas e nao chega a
   100% - o que e saudavel e honesto para dados simulados.

2) Em um Medistar real nao existiria essa formula limpa: a prioridade viria de
   decisoes medicas, desfechos e historico - com muitos fatores interagindo. E
   ai que capturar relacoes nao-lineares agrega valor sobre uma regra manual.

3) Os erros se concentram entre classes VIZINHAS na escala de gravidade (ex.:
   'atencao' x 'alta_prioridade'), o tipo de erro menos perigoso.

4) Ferramenta de APOIO a decisao clinica, NUNCA substituta do julgamento humano.
   Antes de uso real exige validacao com dados reais e revisao etica/clinica.
""")
print("Concluido. Rode o app com:  streamlit run app.py")
