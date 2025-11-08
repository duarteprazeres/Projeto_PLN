import pandas as pd
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from utils import preprocess_text, calculate_metrics, OFFENSIVE_TERMS, get_sentiment_score, extract_manual_features

def load_and_preprocess_data(df):
    print("Carregando e pré-processando os dados...")
    df = df[['comentario', 'label_final']].copy()
    df.rename(columns={'comentario': 'text', 'label_final': 'label'}, inplace=True)
    df['processed_tokens_with_punct'], df['original_text'] = zip(*df['text'].apply(preprocess_text, remove_punctuation=False, apply_stemming=False))
    df['processed_tokens_no_punct'], _ = zip(*df['text'].apply(preprocess_text, remove_punctuation=True, apply_stemming=False))
    df['exclamation_count'] = df['original_text'].apply(lambda x: x.count('!'))
    df['question_mark_count'] = df['original_text'].apply(lambda x: x.count('?'))
    df['repeated_punct'] = df['original_text'].apply(lambda x: bool(re.search(r'([!?.]){2,}', x)))
    df['sentiment_score'] = df['processed_tokens_no_punct'].apply(get_sentiment_score)
    print(f"Total de {len(df)} amostras.")
    return df

def classify_with_rules(row):
    tokens_no_punct = row['processed_tokens_no_punct']
    sentiment_score = row['sentiment_score']
    exclamation_count = row['exclamation_count']
    repeated_punct = row['repeated_punct']
    if any(term in tokens_no_punct for term in OFFENSIVE_TERMS):
        return 1
    if sentiment_score < -0.6:
        return 1
    if exclamation_count >= 2:
        return 1
    if repeated_punct:
        return 1
    if any(term in tokens_no_punct for term in OFFENSIVE_TERMS) and repeated_punct:
        return 1
    return 0

def run_rules_baseline(test_df):
    print("\n========== Baseline: Classificador por Regras ==========")
    test_df['predicted_label_regras'] = test_df.apply(classify_with_rules, axis=1)
    y_true = test_df['label']
    y_pred = test_df['predicted_label_regras']
    accuracy, precision, recall, f1, cm, report = calculate_metrics(y_true, y_pred)
    print("=== Resultados Classificador por Regras ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ofensivo): {precision:.4f}")
    print(f"Recall (Ofensivo): {recall:.4f}")
    print(f"F1-Score (Ofensivo): {f1:.4f}")
    print("Matriz de Confusão:\n", cm)
    print("Relatório de Classificação:\n", report)
    return accuracy, precision, recall, f1

def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n========== Modelo: Regressão Logística ==========")
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1, cm, report = calculate_metrics(y_test, y_pred)
    print("=== Resultados Regressão Logística ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ofensivo): {precision:.4f}")
    print(f"Recall (Ofensivo): {recall:.4f}")
    print(f"F1-Score (Ofensivo): {f1:.4f}")
    print("Matriz de Confusão:\n", cm)
    print("Relatório de Classificação:\n", report)
    return accuracy, precision, recall, f1

def run_random_forest(X_train, X_test, y_train, y_test):
    print("\n========== Modelo: Random Forest ==========")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1, cm, report = calculate_metrics(y_test, y_pred)
    print("=== Resultados Random Forest ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ofensivo): {precision:.4f}")
    print(f"Recall (Ofensivo): {recall:.4f}")
    print(f"F1-Score (Ofensivo): {f1:.4f}")
    print("Matriz de Confusão:\n", cm)
    print("Relatório de Classificação:\n", report)
    return accuracy, precision, recall, f1

def run_svm(X_train, X_test, y_train, y_test):
    print("\n========== Modelo: SVM (Linear) ==========")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1, cm, report = calculate_metrics(y_test, y_pred)
    print("=== Resultados SVM Linear ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ofensivo): {precision:.4f}")
    print(f"Recall (Ofensivo): {recall:.4f}")
    print(f"F1-Score (Ofensivo): {f1:.4f}")
    print("Matriz de Confusão:\n", cm)
    print("Relatório de Classificação:\n", report)
    return accuracy, precision, recall, f1

def main(df):
    df = load_and_preprocess_data(df)

    from utils import gerar_lexico_ofensivo, STOPWORDS_PT, OFFENSIVE_TERMS

    # Geração automática do léxico a partir do dataset preprocessado
    lexico_automatico = gerar_lexico_ofensivo(df, min_ratio=5, min_freq=3)

    OFFENSIVE_TERMS = OFFENSIVE_TERMS.union(lexico_automatico)
    print("Conjunto final de termos ofensivos (manual + automático):")
    print(OFFENSIVE_TERMS)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    print(f"\n=== Divisão dos dados ===")
    print(f"Treino={len(train_df)} | Teste={len(test_df)}")
    
    # Baseline por regras
    acc_rules, prec_rules, rec_rules, f1_rules = run_rules_baseline(test_df)

    # Features para ML
    X_train_manual = extract_manual_features(train_df)
    X_test_manual = extract_manual_features(test_df)
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
    X_train_tfidf = tfidf.fit_transform(train_df['text'])
    X_test_tfidf = tfidf.transform(test_df['text'])
    X_train = np.hstack([X_train_tfidf.toarray(), X_train_manual])
    X_test = np.hstack([X_test_tfidf.toarray(), X_test_manual])
    y_train = train_df['label']
    y_test = test_df['label']

    # Regressão Logística
    acc_lr, prec_lr, rec_lr, f1_lr = run_logistic_regression(X_train, X_test, y_train, y_test)

    # Random Forest
    acc_rf, prec_rf, rec_rf, f1_rf = run_random_forest(X_train, X_test, y_train, y_test)

    # SVM Linear
    acc_svm, prec_svm, rec_svm, f1_svm = run_svm(X_train, X_test, y_train, y_test)

    print("\n\n========== Resumo dos Modelos ==========")
    print("Modelo                Acurácia   Precisão   Recall     F1")
    print("------------------------------------------------------------")
    print(f"Regras               {acc_rules:.4f}    {prec_rules:.4f}    {rec_rules:.4f}    {f1_rules:.4f}")
    print(f"Reg. Logística       {acc_lr:.4f}    {prec_lr:.4f}    {rec_lr:.4f}    {f1_lr:.4f}")
    print(f"Random Forest        {acc_rf:.4f}    {prec_rf:.4f}    {rec_rf:.4f}    {f1_rf:.4f}")
    print(f"SVM Linear           {acc_svm:.4f}    {prec_svm:.4f}    {rec_svm:.4f}    {f1_svm:.4f}")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    df = pd.read_csv("hf://datasets/franciellevargas/HateBR/HateBR.csv")
    main(df)
