# dashboard_prf.py

import dash                             # Framework para criar dashboards web em Python
from dash import html, dcc, Input, Output  # Componentes do Dash
import pandas as pd                    # Manipulação de dados
import plotly.express as px            # Gráficos interativos
import os                              # Acesso a arquivos e diretórios

# === Carregamento e união dos dados de todos os anos ===
caminho = "dados"  # Pasta onde estão os arquivos .csv
arquivos = [arq for arq in os.listdir(caminho) if arq.endswith(".csv") and "Radares" not in arq]
dfs = []

for arq in arquivos:
    ano = arq[-8:-4]  # Extrai o ano do nome do arquivo
    df = pd.read_csv(os.path.join(caminho, arq), sep=';', encoding='latin1', low_memory=False)
    
    df["ano"] = int(ano)
    df["mortos"] = pd.to_numeric(df["mortos"], errors="coerce").fillna(0).astype(int)
    df["feridos_graves"] = pd.to_numeric(df["feridos_graves"], errors="coerce").fillna(0).astype(int)
    df["gravidade"] = df["mortos"] + df["feridos_graves"]
    
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)  # Une todos os anos em um único DataFrame
anos = sorted(df["ano"].unique())       # Lista de anos únicos para o dropdown

# === Iniciar o app Dash ===
app = dash.Dash(__name__)
app.title = "Dashboard PRF"

# === Layout do Dashboard ===
app.layout = html.Div([
    html.H1("📊 Acidentes Rodoviários PRF (2007–2023)", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Selecione o Ano:"),
        dcc.Dropdown(
            id='dropdown-ano',
            options=[{"label": str(ano), "value": ano} for ano in anos],
            value=anos[-1],  # último ano como padrão
            clearable=False
        )
    ], style={'width': '30%', 'margin': 'auto'}),

    html.Div(id="indicadores", style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': 30}),

    dcc.Graph(id="grafico_linha"),
    dcc.Graph(id="mapa_calor")
])

# === Callback para atualizar os componentes interativos ===
@app.callback(
    Output("indicadores", "children"),
    Output("grafico_linha", "figure"),
    Output("mapa_calor", "figure"),
    Input("dropdown-ano", "value")
)
def atualizar_dashboard(ano_selecionado):
    df_ano = df[df["ano"] == ano_selecionado]

    total_acidentes = len(df_ano)
    total_mortos = df_ano["mortos"].sum()
    total_feridos = df_ano["feridos_graves"].sum()

    indicadores = [
        html.Div([
            html.H3("Total de Acidentes"),
            html.P(f"{total_acidentes:,}".replace(",", ".")),
        ], style={"textAlign": "center"}),

        html.Div([
            html.H3("Total de Mortos"),
            html.P(f"{total_mortos:,}".replace(",", ".")),
        ], style={"textAlign": "center"}),

        html.Div([
            html.H3("Feridos Graves"),
            html.P(f"{total_feridos:,}".replace(",", ".")),
        ], style={"textAlign": "center"}),
    ]

    # Gráfico de linha: gravidade por ano
    df_agrupado = df.groupby("ano")["gravidade"].sum().reset_index()
    fig_linha = px.line(
        df_agrupado, x="ano", y="gravidade",
        title="Gravidade Total por Ano",
        labels={"gravidade": "Mortos + Feridos Graves", "ano": "Ano"}
    )

    # Mapa de calor: localização dos acidentes
    df_mapa = df_ano.dropna(subset=["latitude", "longitude"])
    fig_mapa = px.density_mapbox(
        df_mapa, lat="latitude", lon="longitude", z="gravidade",
        radius=5,
        center=dict(lat=-15.5, lon=-47.5), zoom=4,
        mapbox_style="open-street-map",
        title="Mapa de Calor de Acidentes"
    )

    return indicadores, fig_linha, fig_mapa

# === Rodar o app ===
if __name__ == "__main__":
    app.run(debug=True)
