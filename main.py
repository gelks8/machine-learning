import os
import re
import unicodedata
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, classification_report, ConfusionMatrixDisplay, make_scorer)
import matplotlib.pyplot as plt
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
tqdm.pandas()

# ----------------------------
# Pré-processamento
# ----------------------------
def normalize_text(text):
  if pd.isna(text):
      return ""
  # remove URLs and emails
  text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
  text = re.sub(r"\S+@\S+", "", text)
  # normalize unicode (remove accents)
  text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
  text = text.lower()
  # keep letters and spaces
  text = re.sub(r"[^a-z\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

def preprocess_text(text, stop_words_set, stemmer):
  text = normalize_text(text)
  tokens = nltk.word_tokenize(text)
  tokens = [t for t in tokens if t not in stop_words_set and len(t) > 1]
  tokens = [stemmer.stem(t) for t in tokens]
  return " ".join(tokens)

# ----------------------------
# Carregar dados (detecção de colunas comuns)
# ----------------------------
def load_dataset(filepath):
  df = pd.read_csv(filepath)
  # detectar coluna de texto
  text_cols = [c for c in df.columns if c.lower() in ('review_text','review','text','content')]
  label_cols = [c for c in df.columns if c.lower() in ('sentiment','label','rating','target')]
  if not text_cols:
    # pega a primeira coluna por ausência de alternativa
    text_col = df.columns[0]
  else:
    text_col = text_cols[0]
  if not label_cols:
    raise ValueError("Não encontrei coluna de rótulo. Renomeie a coluna de sentimento para 'sentiment' ou 'label'.")
  else:
    label_col = label_cols[0]
  df = df[[text_col, label_col]].rename(columns={text_col: 'review_text', label_col: 'sentiment'})
  df = df.dropna(subset=['review_text', 'sentiment']).reset_index(drop=True)
  return df

# ----------------------------
# Pipeline e treino/avaliação
# ----------------------------
def build_pipeline(clf):
  pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', clf)
  ])
  return pipe

def evaluate_cv(pipe, X, y, k=5):
  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
  scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label='positivo'),
    'recall': make_scorer(recall_score, pos_label='positivo'),
    'f1': make_scorer(f1_score, pos_label='positivo')
  }
  scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
  # calcula média e std
  results = {metric: (scores[f'test_{metric}'].mean(), scores[f'test_{metric}'].std()) for metric in scoring}
  return results

def holdout_evaluate(pipe, X_train, X_test, y_train, y_test):
  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4)
  cm = confusion_matrix(y_test, y_pred)
  return report, cm

# ----------------------------
# Utilitários
# ----------------------------
def print_cv_results(name, results):
  print(f"\n--- CV results for {name} ---")
  for metric, (mean, std) in results.items():
      print(f"{metric}: {mean:.4f} ± {std:.4f}")

def plot_and_save_cm(cm, labels, outpath):
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot(cmap=plt.cm.Blues)
  plt.title("Matriz de Confusão")
  plt.savefig(outpath, bbox_inches='tight')
  plt.close()

# ----------------------------
# Main
# ----------------------------
def main(args):
  nltk.download('punkt', quiet=True)
  nltk.download('stopwords', quiet=True)
  # Detectar idioma automaticamente
  sample_text = Path(args.data).stem.lower()  # pega o nome do arquivo
  if "imdb" in sample_text or "amazon" in sample_text or "english" in sample_text:
    lang = "english"
  else:
    lang = "portuguese"

  print(f"Usando idioma de pré-processamento: {lang}")

  stop_words = set(stopwords.words(lang))
  stemmer = SnowballStemmer(lang)

  print("Carregando dataset...")
  df = load_dataset(args.data)
  print(f"Dataset carregado: {len(df)} linhas")

  print("Pré-processando textos (pode demorar dependendo do dataset)...")
  df['processed_review'] = df['review_text'].progress_apply(lambda t: preprocess_text(t, stop_words, stemmer))

  # mapear labels para strings padronizadas (ex.: positivo/negativo)
  df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
  # se numeric (ex: 1-5), mapear para positivo/negativo (exemplo)
  if pd.api.types.is_numeric_dtype(df['sentiment']):
      # simples: >=4 positivo, <=2 negativo, 3 neutro (remover)
      df = df[df['sentiment'] != 3]
      df['sentiment'] = df['sentiment'].apply(lambda s: 'positivo' if s >= 4 else 'negativo')

  # opcional: filtrar apenas positivo/negativo
  df = df[df['sentiment'].isin(['positivo','negativo','positive','negative','pos','neg','0','1'])]
  # uniformizar labels
  df['sentiment'] = df['sentiment'].replace({'positive':'positivo','negative':'negativo','pos':'positivo','neg':'negativo','1':'positivo','0':'negativo'})

  X = df['processed_review']
  y = df['sentiment']

  # modelos
  nb = MultinomialNB()
  lr = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=42)

  pipe_nb = build_pipeline(nb)
  pipe_lr = build_pipeline(lr)

  # CV
  print("Rodando validação cruzada (5-fold) — Naive Bayes")
  res_nb = evaluate_cv(pipe_nb, X, y, k=args.k)
  print_cv_results("NaiveBayes", res_nb)

  print("Rodando validação cruzada (5-fold) — Logistic Regression")
  res_lr = evaluate_cv(pipe_lr, X, y, k=args.k)
  print_cv_results("LogisticRegression", res_lr)

  # Hold-out para matriz de confusão e relatório
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
  print("\nTreinando em hold-out e avaliando Naive Bayes...")
  report_nb, cm_nb = holdout_evaluate(pipe_nb, X_train, X_test, y_train, y_test)
  print("Relatório Naive Bayes:\n", report_nb)
  plot_and_save_cm(cm_nb, labels=['negativo','positivo'], outpath='models/cm_nb.png')
  print("Matriz de confusão salva em models/cm_nb.png")

  print("\nTreinando em hold-out e avaliando Logistic Regression...")
  report_lr, cm_lr = holdout_evaluate(pipe_lr, X_train, X_test, y_train, y_test)
  print("Relatório Logistic Regression:\n", report_lr)
  plot_and_save_cm(cm_lr, labels=['negativo','positivo'], outpath='models/cm_lr.png')
  print("Matriz de confusão salva em models/cm_lr.png")

  # Grid Search opcional para LR (exemplo simples)
  print("\nExecutando GridSearchCV simples para Logistic Regression (opcional)...")
  param_grid = {
      'clf__C': [0.01, 0.1, 1, 10],
  }
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
  scorer = make_scorer(f1_score, pos_label="positivo")
  gs = GridSearchCV(pipe_lr, param_grid, scoring=scorer, cv=skf, n_jobs=-1)
  gs.fit(X_train, y_train)
  print("Best params:", gs.best_params_, " Best score (cv):", gs.best_score_)
  best_lr = gs.best_estimator_

  # escolher melhor entre NB e LR por F1 na validação CV (métrica exemplo)
  mean_f1_nb = res_nb['f1'][0]  # média armazenada
  mean_f1_lr = res_lr['f1'][0]
  # aqui usamos o resultado da CV (média)
  chosen = ('nb', pipe_nb) if mean_f1_nb >= mean_f1_lr else ('lr', best_lr)
  print(f"\nModelo escolhido: {chosen[0]}")

  # salvar modelo escolhido
  os.makedirs('models', exist_ok=True)
  joblib.dump(chosen[1], 'models/best_model.joblib')
  print("Modelo salvo em models/best_model.joblib")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='data/reviews.csv', help='Caminho para CSV com reviews')
  parser.add_argument('--test_size', type=float, default=0.2, help='Proporção do teste (hold-out)')
  parser.add_argument('--k', type=int, default=5, help='Número de folds para validação cruzada (k)')
  args = parser.parse_args()
  main(args)