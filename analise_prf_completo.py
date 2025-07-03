import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use('Agg')

# Cria pasta para salvar imagens
output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set(style="whitegrid", palette="Set2")
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# === Captura de texto para relatório ===
relatorio_texto = {}

# === Parte 1: Coleta, Limpeza e Pré-processamento ===
caminho = "dados"
arquivos = [arq for arq in os.listdir(caminho) if arq.endswith(".csv") and "Radares" not in arq]
df_lista = []

for arq in arquivos:
    ano = arq[-8:-4]
    df_temp = pd.read_csv(os.path.join(caminho, arq), sep=';', encoding='latin1', low_memory=False)
    df_temp["ano"] = int(ano)
    df_lista.append(df_temp)

df = pd.concat(df_lista, ignore_index=True)
df.replace("(null)", np.nan, inplace=True)

if 'condicao_metereologica' in df.columns:
    df["condicao_metereologica"] = df["condicao_metereologica"].fillna("Ignorado")

for col in ["latitude", "longitude"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

for col in ["mortos", "feridos_graves"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

df["gravidade"] = df["mortos"] + df["feridos_graves"]

if 'data_inversa' in df.columns:
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
    df["mes"] = df["data_inversa"].dt.month
    df["dia_semana"] = df["data_inversa"].dt.day_name()

# Exporta base limpa
df.to_csv("df_limpo.csv", index=False)
relatorio_texto["limpeza"] = "\u2705 Dataset limpo salvo como 'df_limpo.csv'"
print(relatorio_texto["limpeza"])

# === Parte 2: Análise Estatística e Visualização ===
relatorio_texto["normalidade"] = "\U0001F9EA Teste de Normalidade:\n"
for col in ["mortos", "feridos_graves", "gravidade"]:
    stat, p = stats.shapiro(df[col].sample(n=500, random_state=42))
    resultado = "Não normal" if p < 0.05 else "Normal"
    relatorio_texto["normalidade"] += f"{col}: stat={stat:.4f}, p={p:.4f} → {resultado}\n"
print("\n" + relatorio_texto["normalidade"])

corr = df[["mortos", "feridos_graves", "gravidade"]].corr()
sns.heatmap(corr, annot=True, cmap="Reds", fmt=".2f")
plt.title("Mapa de Calor")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlacao_heatmap.png"), dpi=300)
plt.clf()

df["fim_de_semana"] = df["dia_semana"].isin(["Saturday", "Sunday"])
t_stat, p_valor = stats.ttest_ind(
    df[df["fim_de_semana"]]["mortos"],
    df[~df["fim_de_semana"]]["mortos"],
    equal_var=False
)
relatorio_texto["ttest"] = f"\U0001F4CA Teste t:\nT={t_stat:.4f}, p={p_valor:.4f}"
print("\n" + relatorio_texto["ttest"])

sns.histplot(df["gravidade"], bins=20, kde=True, color="purple")
plt.title("Distribuição da Gravidade")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "histograma_gravidade.png"), dpi=300)
plt.clf()

df["tipo_dia"] = df["fim_de_semana"].map({True: "Fim de Semana", False: "Dia Útil"})
sns.boxplot(data=df, x="tipo_dia", y="gravidade", color="#4A90E2")
plt.title("Gravidade por Tipo de Dia")
plt.ylim(0, 30)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_gravidade_fds.png"), dpi=300)
plt.clf()

sns.scatterplot(x="feridos_graves", y="mortos", data=df, alpha=0.5)
plt.title("Feridos Graves vs Mortos")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dispersao_feridos_mortos.png"), dpi=300)
plt.clf()

df_mensal = df.groupby(["ano", "mes"])["gravidade"].sum().reset_index()
df_mensal["data"] = pd.to_datetime(df_mensal.rename(columns={"ano": "year", "mes": "month"}).assign(day=1)[["year", "month", "day"]])
plt.plot(df_mensal["data"], df_mensal["gravidade"], marker='o', color='green')
plt.title("Gravidade Mensal")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "serie_temporal_gravidade.png"), dpi=300)
plt.clf()

mortos = df["mortos"].gt(0).sum()
sem_mortos = df["mortos"].eq(0).sum()
plt.pie([mortos, sem_mortos], labels=["Com Mortos", "Sem Mortos"],
        colors=["red", "lightgray"], autopct='%1.1f%%', startangle=90)
plt.title("Acidentes com Mortes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pizza_mortos.png"), dpi=300)
plt.clf()

# === Parte 3: Modelagem e Machine Learning ===
df = pd.read_csv("df_limpo.csv")
df["gravidade_alta"] = (df["gravidade"] >= 2).astype(int)

features = ["mortos", "feridos_graves", "latitude", "longitude", "mes"]
relatorio_texto["ausentes_antes"] = "\U0001F50D Verificando valores ausentes nas features:\n"
relatorio_texto["ausentes_antes"] += df[features + ["gravidade_alta"]].isna().sum().to_string()
print("\n" + relatorio_texto["ausentes_antes"])

for col in ["latitude", "longitude", "mes"]:
    df[col] = df[col].fillna(df[col].median())

relatorio_texto["ausentes_depois"] = "\n✅ Valores ausentes após preenchimento:\n"
relatorio_texto["ausentes_depois"] += df[features + ["gravidade_alta"]].isna().sum().to_string()
print(relatorio_texto["ausentes_depois"])

df_final = df.dropna(subset=features)
relatorio_texto["linhas_restantes"] = f"\n\U0001F50E Linhas restantes após dropna: {len(df_final)}"
print(relatorio_texto["linhas_restantes"])

X = df_final[features]
y = df_final["gravidade_alta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(random_state=42)

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)

relatorio_texto["logistica"] = "\U0001F4CA Regressão Logística:\n" + classification_report(y_test, y_pred_log)
relatorio_texto["rf"] = "\U0001F332 Random Forest:\n" + classification_report(y_test, y_pred_rf)
print("\n" + relatorio_texto["logistica"])
print("\n" + relatorio_texto["rf"])

log_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

relatorio_texto["validacao"] = (
    f"\n\U0001F4C8 Validação Cruzada:\n"
    f"  • Logística: {log_scores.mean():.3f} ± {log_scores.std():.3f}\n"
    f"  • Random Forest: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}"
)
print(relatorio_texto["validacao"])

param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)
relatorio_texto["melhor_modelo"] = f"\n\U0001F50D Melhor Random Forest: {grid.best_params_}"
print(relatorio_texto["melhor_modelo"])

importancia_df = pd.DataFrame({
    'feature': X.columns,
    'importancia': rf_model.feature_importances_
}).sort_values(by='importancia', ascending=True)

sns.barplot(data=importancia_df, x='importancia', y='feature', palette='viridis')
plt.title("Importância das Variáveis")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "importancia_features_rf.png"), dpi=300)
plt.clf()

relatorio_texto["interpretacao"] = (
    f"\n\U0001F4CC Interpretação dos Resultados:\n"
    f"Acurácia média: Logística = {log_scores.mean():.3f}, RF = {rf_scores.mean():.3f}\n"
    f"Variáveis mais importantes: feridos_graves, mortos"
)
print(relatorio_texto["interpretacao"])

relatorio_texto["limitacoes"] = (
    "\n⚠️ Limitações:\n"
    "- Não usamos variáveis categóricas (ex: tipo_acidente)\n"
    "- Classes desbalanceadas"
)
print(relatorio_texto["limitacoes"])

relatorio_texto["melhorias"] = (
    "\n\U0001F4A1 Melhorias:\n"
    "- Incluir mais variáveis (clima, tipo acidente)\n"
    "- Testar SMOTE, XGBoost, LightGBM"
)
print(relatorio_texto["melhorias"])

print("\n✅ Script finalizado com sucesso!")

# === Salvar textos para PDF ===
with open("texto_analise.txt", "w", encoding="utf-8") as f:
    for secao, conteudo in relatorio_texto.items():
        f.write(f"=== {secao.upper()} ===\n{conteudo}\n\n")
print("\n📝 Texto de análise salvo como 'texto_analise.txt'")
