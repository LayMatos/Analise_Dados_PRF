import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import plotly.io as pio
pio.renderers.default = 'browser'  # força abrir no navegador

# === CONFIGURAÇÃO ===
# Caminho onde estão os arquivos .csv (todos os anos)
caminho = "dados"

# === 1. CARREGAR DADOS DE TODOS OS ANOS ===
arquivos = [arq for arq in os.listdir(caminho) if arq.endswith(".csv") and "Radares" not in arq]
todos_anos = []

for arquivo in arquivos:
    ano = arquivo[-8:-4]
    df = pd.read_csv(os.path.join(caminho, arquivo), sep=';', encoding='latin1', low_memory=False)
    df["ano"] = int(ano)
    todos_anos.append(df)

df_todos = pd.concat(todos_anos, ignore_index=True)
print(f"Total de registros: {df_todos.shape[0]}")

# === 2. LIMPEZA DE DADOS ===
df_todos["mortos"] = pd.to_numeric(df_todos["mortos"], errors="coerce").fillna(0).astype(int)
df_todos["feridos_graves"] = pd.to_numeric(df_todos["feridos_graves"], errors="coerce").fillna(0).astype(int)
df_todos["gravidade"] = df_todos["mortos"] + df_todos["feridos_graves"]
df_todos["br"] = pd.to_numeric(df_todos["br"], errors="coerce")
df_todos["km"] = df_todos["km"].astype(str).str.replace(",", ".", regex=False).replace("(null)", np.nan).astype(float)
df_todos["latitude"] = df_todos["latitude"].astype(str).str.replace(",", ".").astype(float)
df_todos["longitude"] = df_todos["longitude"].astype(str).str.replace(",", ".").astype(float)

# === 3. ANÁLISES EXPLORATÓRIAS ===
print("\nTop 10 Rodovias mais perigosas:")
print(df_todos.groupby("br")["gravidade"].sum().sort_values(ascending=False).head(10))

print("\nTop 10 Municípios mais perigosos:")
print(df_todos.groupby("municipio")["gravidade"].sum().sort_values(ascending=False).head(10))

print("\nTop 10 Tipos de Acidente:")
print(df_todos["tipo_acidente"].value_counts().head(10))

print("\nTop 10 Causas de Acidente:")
print(df_todos["causa_acidente"].value_counts().head(10))

# === 4. GRÁFICO DE LINHA - GRAVIDADE POR ANO ===
acidentes_por_ano = df_todos.groupby("ano")["gravidade"].sum().reset_index()

fig = px.line(acidentes_por_ano, x="ano", y="gravidade",
              title="Gravidade dos Acidentes por Ano (2007–2023)",
              labels={"gravidade": "Mortos + Feridos Graves", "ano": "Ano"})
fig.show()

# === 5. MAPA DE CALOR DOS ACIDENTES ===
df_geo = df_todos.dropna(subset=["latitude", "longitude"])

plt.figure(figsize=(10, 6))
sns.kdeplot(
    x=df_geo["longitude"],
    y=df_geo["latitude"],
    fill=True,
    cmap="Reds",
    bw_adjust=0.3,
    alpha=0.5
)
plt.title("Mapa de Calor dos Acidentes no Brasil (2007–2023)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
