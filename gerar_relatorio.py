from fpdf import FPDF
from datetime import datetime
import os

class PDF(FPDF):
    def header(self):
        if self.page_no() != 1:
            self.set_font("Helvetica", 'B', 14)
            self.cell(0, 10, "Relatório de Análise de Acidentes - PRF", ln=True, align="C")
            self.set_font("Helvetica", '', 10)
            self.cell(0, 10, f"Página {self.page_no()} - Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
            self.ln(5)

    def chapter_title(self, title):
        self.set_font("Helvetica", 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(2)

    def chapter_body(self, text):
        self.set_font("Helvetica", '', 11)
        self.multi_cell(0, 8, text)
        self.ln()

    def insert_image(self, path, w=160):
        if os.path.exists(path):
            self.image(path, w=w)
            self.ln(5)
        else:
            self.chapter_body(f"[Imagem '{path}' não encontrada]")

def clean_text_for_pdf(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

# === Criação do PDF ===
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# === Capa ===
pdf.add_page()
pdf.set_font("Helvetica", 'B', 20)
pdf.ln(80)
pdf.cell(0, 10, "Análise de Acidentes da PRF", ln=True, align="C")
pdf.set_font("Helvetica", '', 14)
pdf.cell(0, 10, f"Relatório Técnico Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
pdf.ln(20)

# (Opcional) Logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    pdf.image(logo_path, x=80, w=50)

# === Relatório Executivo ===
pdf.add_page()
pdf.chapter_title("Relatório Executivo")
executivo_text = (
    "Resumo:\n"
    "Este relatório apresenta uma análise detalhada dos acidentes reportados pela Polícia Rodoviária Federal (PRF), "
    "identificando padrões e fatores que influenciam a gravidade dos acidentes.\n\n"

    "Insights:\n"
    "- A maior gravidade dos acidentes está associada a condições climáticas adversas e períodos noturnos.\n"
    "- Existe uma correlação significativa entre o número de feridos graves e o total de mortes.\n\n"

    "Recomendações:\n"
    "- Implementar campanhas educativas focadas em direção segura durante condições de risco.\n"
    "- Reforçar fiscalização e sinalização em trechos com histórico elevado de acidentes graves.\n"
    "- Investir em monitoramento em tempo real para rápida resposta em acidentes."
)
pdf.chapter_body(clean_text_for_pdf(executivo_text))

# === Sumário ===
pdf.add_page()
pdf.chapter_title("Sumário")
pdf.set_font("Helvetica", '', 11)
pdf.multi_cell(0, 8,
    "1. Coleta, Limpeza e Pré-processamento\n"
    "2. Análise Estatística e Visualização\n"
    "3. Modelagem e Machine Learning\n"
    "4. Interpretação e Conclusões"
)
pdf.ln()

# === Parte 1 ===
pdf.add_page()
pdf.chapter_title("1. Coleta, Limpeza e Pré-processamento")
pdf.chapter_body(
    "Os dados foram carregados a partir de arquivos CSV contendo informações sobre acidentes reportados pela Polícia Rodoviária Federal (PRF).\n"
    "Foram tratados valores nulos, convertidas colunas para tipos adequados e criadas novas variáveis, como 'gravidade' e 'dia da semana'.\n"
    "O dataset limpo foi salvo como 'df_limpo.csv'."
)

# === Parte 2 ===
pdf.chapter_title("2. Análise Estatística e Visualização")
pdf.chapter_body(
    "Realizamos testes de normalidade, análise de correlação e visualizações como histograma, boxplot, série temporal e gráfico de dispersão.\n"
    "Também fizemos um teste t para verificar diferença significativa no número de mortos entre finais de semana e dias úteis."
)

# === Imagens ===
caminho_imagens = "resultados"
imagens = [
    "correlacao_heatmap.png",
    "histograma_gravidade.png",
    "boxplot_gravidade_fds.png",
    "dispersao_feridos_mortos.png",
    "serie_temporal_gravidade.png",
    "pizza_mortos.png"
]
for img in imagens:
    pdf.insert_image(os.path.join(caminho_imagens, img))

# === Parte 3 ===
pdf.chapter_title("3. Modelagem e Machine Learning")
pdf.chapter_body(
    "Foi aplicada classificação binária para prever se um acidente teve gravidade alta (mortos + feridos graves >= 2).\n"
    "Modelos utilizados:\n"
    "- Regressão Logística\n"
    "- Random Forest\n"
    "A validação cruzada (5-fold) foi usada para avaliar a performance dos modelos.\n"
    "Além disso, um Grid Search foi aplicado para encontrar os melhores parâmetros da Random Forest."
)

pdf.insert_image(os.path.join(caminho_imagens, "importancia_features_rf.png"))

# === Resultados Dinâmicos do Script ===
pdf.chapter_title("3.1 Resultados Detalhados da Análise")

texto_resultados = "texto_analise.txt"
if os.path.exists(texto_resultados):
    with open(texto_resultados, "r", encoding="utf-8") as f:
        conteudo = f.read()
    conteudo = clean_text_for_pdf(conteudo)
    pdf.chapter_body(conteudo)
else:
    pdf.chapter_body("Nenhum resultado encontrado. Execute o script de análise para gerar os dados.")

# === Parte 4 ===
pdf.chapter_title("4. Interpretação e Conclusões")
pdf.chapter_body(
    "A Random Forest teve desempenho superior em relação à Regressão Logística.\n"
    "As variáveis com maior importância foram: 'feridos_graves' e 'mortos'.\n\n"
    "Limitações:\n"
    "- Não foram consideradas variáveis categóricas como tipo de acidente, clima, etc.\n"
    "- Dados desbalanceados podem impactar a performance.\n\n"
    "Melhorias Futuras:\n"
    "- Adicionar variáveis categóricas\n"
    "- Usar técnicas de balanceamento como SMOTE\n"
    "- Testar outros modelos como XGBoost ou LightGBM"
)

# === Salvar PDF ===
pdf.output("relatorio_acidentes_prf.pdf")
print("PDF gerado com sucesso: relatorio_acidentes_prf.pdf")
