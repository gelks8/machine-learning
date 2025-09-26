# Machine Learning Text Classification

Este projeto é um classificador de sentimentos desenvolvido em Python que realiza a análise de reviews de texto, identificando se o sentimento é positivo ou negativo.  
O objetivo é servir como uma aplicação de aprendizado e prática em Machine Learning.

---

📝 **Funcionalidades**

- Treinamento e teste de modelo de classificação de texto
- Suporte a CSV de teste para experimentos rápidos
- Suporte ao IMDB Movie Reviews Dataset para análise de grandes volumes de dados
- Pré-processamento de texto em inglês
- Resultados de previsão de sentimento para cada review

---

⚠️ **Dificuldades encontradas**

- Ajustar o script para aceitar diferentes nomes de arquivos CSV, inclusive com espaços no nome
- Garantir compatibilidade com datasets grandes (como o IMDB Dataset)
- Configurar `.gitignore` para ignorar `venv` e evitar subir dependências para o GitHub
- Configurar caminhos relativos e absolutos para leitura dos arquivos de forma segura

---

💻 **Pré-requisitos**

Para rodar o projeto, você precisa ter instalado no seu computador:

- Python 3.10 ou superior
- Pip
- Editor de código (opcional, ex.: VSCode, PyCharm, Sublime)
- Git (para clonar o repositório)
- Conta no Kaggle (para baixar o IMDB Dataset)

---

🚀 **Como executar**

Clone o repositório:

```bash
git clone <https://github.com/gelks8/machine-learning>
cd machine-learning

Crie e ative um ambiente virtual:
python -m venv venv

Instale as dependências:
pip install -r requirements.txt

Teste com o IMDB Dataset

Baixe o dataset completo do IMDB Movie Reviews aqui:
[IMDB Dataset - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)

Coloque o arquivo baixado na pasta data/ do projeto e renomeie se necessário para:
data/IMDB Dataset.csv

Execute o script:
python main.py --data "data/IMDB Dataset.csv"
