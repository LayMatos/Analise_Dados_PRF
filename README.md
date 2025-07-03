# 🚨 Análise de Acidentes da PRF com Machine Learning

Este projeto realiza uma análise exploratória e modelagem preditiva usando dados de acidentes da Polícia Rodoviária Federal (PRF). Ele inclui desde a limpeza de dados até a aplicação de algoritmos de Machine Learning e geração de relatórios visuais e textuais.

Link dos dados 
https://www.kaggle.com/datasets/tgomesjuliana/police-traffic-incidents

---

## 📁 Estrutura do Projeto

```text
📦 prf-acidentes/
├── dados/                  # Arquivos CSV originais
├── resultados/             # Gráficos gerados automaticamente
├── df_limpo.csv            # Base de dados limpa e unificada
├── texto_analise.txt       # Relatório textual final
├── analise_prf_completo.py # Script principal (análise + modelagem)
├── gerar_relatorio.py      # (opcional) Geração de PDF com fpdf2
└── README.md               # Este arquivo (documentação do projeto)

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Pandas / NumPy** – Manipulação de dados
- **Seaborn / Matplotlib** – Visualização de dados
- **SciPy** – Testes estatísticos
- **Scikit-learn** – Modelagem e validação
- **FPDF / fpdf2** – (opcional) Geração de relatório PDF

---

## 🔍 Etapas da Análise

### 1. 📂 Coleta e Limpeza de Dados

- Leitura de múltiplos arquivos `.csv`
- Conversão de tipos (`latitude`, `longitude`, `data`)
- Tratamento de valores nulos
- Criação da variável `gravidade` = `mortos + feridos_graves`
- Extração de colunas como `mês` e `dia da semana`

---

### 2. 📊 Análise Estatística e Visualizações

- **Teste de Normalidade** com Shapiro-Wilk
- **Correlação** entre mortos, feridos e gravidade
- **Teste T** para comparar gravidade entre fins de semana e dias úteis
- **Gráficos gerados automaticamente**:
  - Heatmap de correlação
  - Histograma da gravidade
  - Boxplot por tipo de dia
  - Dispersão feridos × mortos
  - Série temporal da gravidade mensal
  - Gráfico de pizza de acidentes com mortos

---

### 3. ⚙️ Machine Learning

#### Variável Alvo:
- `gravidade_alta` → binária: 0 (baixa) ou 1 (alta)

#### Modelos Utilizados:
- 🔵 **Regressão Logística**
- 🌲 **Random Forest**

#### Pipeline:
- Separação treino/teste (70% / 30%)
- Padronização com `StandardScaler`
- Avaliação com `classification_report` e `cross_val_score`
- Otimização com `GridSearchCV`

#### Resultados:
- Avaliação de acurácia dos modelos
- Gráfico de importância das variáveis (feature importance)

---

### 4. 📝 Relatórios Gerados

- ✅ `texto_analise.txt`: contém todas as análises, testes e interpretações
- 🖼️ Gráficos salvos em `resultados/`:
  - `correlacao_heatmap.png`
  - `histograma_gravidade.png`
  - `boxplot_gravidade_fds.png`
  - `dispersao_feridos_mortos.png`
  - `serie_temporal_gravidade.png`
  - `pizza_mortos.png`
  - `importancia_features_rf.png`

---

## 📌 Interpretações e Conclusões

- A variável `feridos_graves` foi a mais relevante na previsão da gravidade
- Os dados **não seguem distribuição normal**
- Fins de semana tendem a ter maior gravidade média
- O modelo Random Forest teve melhor desempenho geral

---

## ⚠️ Limitações

- Não foram incluídas variáveis categóricas (ex: tipo de acidente, clima)
- Classes desbalanceadas
- Dados incompletos em algumas colunas

---

## 💡 Possíveis Melhorias

- Aplicar técnicas de balanceamento como **SMOTE**
- Incluir variáveis como:
  - Tipo de acidente
  - Condição climática
- Testar outros modelos: **XGBoost**, **LightGBM**
- Geração automática de relatório em **PDF com gráficos e textos**

---

## 👩‍💻 Como Executar

```bash
# 1. Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn scipy fpdf2

# 2. Coloque os arquivos CSV na pasta 'dados/'

# 3. Execute o script principal
python analise_prf_completo.py

# 4. Veja os resultados em:
#    - resultados/ (gráficos)
#    - texto_analise.txt (relatório textual)


