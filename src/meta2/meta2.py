import pandas as pd
import os
import re
import numpy as np

# Imports de ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imports do teu ficheiro utils.py
try:
    from meta1.utils import (
        preprocess_text,
        calculate_metrics,
        gerar_lexico_ofensivo,
        get_sentiment_score,
        OFFENSIVE_TERMS,
        STOPWORDS_PT
    )
except ImportError:
    print("Erro: Não foi possível encontrar o ficheiro 'utils.py'.")
    print("Certifica-te de que 'utils.py' está na mesma pasta que 'meta2.py'.")
    exit()

# --- Configurações ---
OUTPUT_DIR = 'output_meta2'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_feature_engineer(df):
    """
    Carrega os dados e aplica a engenharia de features linguísticas.
    Esta função prepara o DataFrame para ser usado pelos Pipelines.
    """
    print("Carregando e pré-processando os dados...")
    df = df[['comentario', 'label_final']].copy()
    df.rename(columns={'comentario': 'text', 'label_final': 'label'}, inplace=True)
    
    # 1. Pré-processar o texto para obter tokens (para features linguísticas)
    # Usamos apply_stemming=False para léxicos e sentimento, como no teu utils
    processed_data = df['text'].apply(preprocess_text, remove_punctuation=True, apply_stemming=False)
    df['tokens_no_punct'], df['original_text'] = zip(*processed_data)

    print("Extraindo features linguísticas...")
    
    # 2. Extrair Features Linguísticas
    df['exclamation_count'] = df['original_text'].apply(lambda x: x.count('!'))
    df['question_mark_count'] = df['original_text'].apply(lambda x: x.count('?'))
    df['repeated_punct'] = df['original_text'].apply(lambda x: bool(re.search(r'([!?.]){2,}', x))).astype(int)
    df['caps_ratio'] = df['original_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-6))
    df['sentiment_score'] = df['tokens_no_punct'].apply(get_sentiment_score)
    
     # 3. Dividir os dados (o DataFrame completo)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    y_train = train_df['label']
    y_test = test_df['label']
    print(f"\n=== Divisão dos dados ===")
    print(f"Treino={len(train_df)} | Teste={len(test_df)}")
    
    all_results = []
    
    # 4. Gerar Léxico Ofensivo (com base nos dados de treino)
    # Nota: Idealmente, isto seria feito *após* o split, apenas no train_df.
    # Mas para simplificar, geramos com todos os dados como no teu script.
    lexico_automatico = gerar_lexico_ofensivo(train_df, min_ratio=5, min_freq=3)
    COMBINED_LEXICON = OFFENSIVE_TERMS.union(lexico_automatico)
    
    # 5. Extrair Feature de Contagem Ofensiva
    df['offensive_term_count'] = df['tokens_no_punct'].apply(lambda tokens: sum(1 for token in tokens if token in COMBINED_LEXICON))
    
    print(f"Total de {len(df)} amostras processadas.")
    return df

def main():
    """Função principal para executar a Meta 2 com Pipelines e GridSearchCV."""
    
    print("A carregar o dataset HateBR do Hugging Face...")
    df = pd.read_csv("hf://datasets/franciellevargas/HateBR/HateBR.csv")
    
    # 1. Processar dados e extrair features
    df = load_and_feature_engineer(df)
    
    # 2. Dividir os dados (o DataFrame completo)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    y_train = train_df['label']
    y_test = test_df['label']
    print(f"\n=== Divisão dos dados ===")
    print(f"Treino={len(train_df)} | Teste={len(test_df)}")
    
    all_results = []

    # --- META 2: Engenharia de Features com Pipelines ---
    
    # Definir as colunas para cada transformador
    LINGUISTIC_FEATURES = [
        'exclamation_count', 'question_mark_count', 'repeated_punct',
        'caps_ratio', 'offensive_term_count', 'sentiment_score'
    ]
    TEXT_FEATURE = 'text' # Coluna de texto original para o TF-IDF

    # Abordagem 1: TF-IDF (Conteúdo do Texto)
    preprocessor_tfidf = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words=list(STOPWORDS_PT)), TEXT_FEATURE)
        ],
        remainder='drop') # Ignora as colunas linguísticas

    # Abordagem 2: Linguística (Conhecimento Linguístico)
    preprocessor_ling = ColumnTransformer(
        transformers=[
            ('linguistic', StandardScaler(), LINGUISTIC_FEATURES)
        ],
        remainder='drop') # Ignora a coluna de texto

    # Abordagem 3: Híbrida (TF-IDF + Linguística)
    preprocessor_hybrid = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words=list(STOPWORDS_PT)), TEXT_FEATURE),
            ('linguistic', StandardScaler(), LINGUISTIC_FEATURES)
        ],
        remainder='drop') # Ignora outras colunas

    # Dicionário de conjuntos de features (pré-processadores)
    feature_sets = {
        "Ab. 1: TF-IDF": preprocessor_tfidf,
        "Ab. 2: Linguística": preprocessor_ling,
        "Ab. 3: Híbrida": preprocessor_hybrid    }

    # --- META 2: Matriz de Experimentação com GridSearchCV ---
    
    # Modelos para testar
    models_to_test = {
        "Reg. Logística": LogisticRegression(max_iter=300, random_state=42),
        "SVM Linear": LinearSVC(random_state=42, dual=True, max_iter=5000)
        # RandomForest é omitido por ser muito lento com GridSearchCV
    }
    
    # Hiperparâmetros para otimizar
    param_grids = {
        "Reg. Logística": {'clf__C': [0.1, 1, 10]}, # Procura o melhor C
        "SVM Linear": {'clf__C': [0.1, 1, 10]}      # Procura o melhor C
    }

    print("\n=== META 2: Iniciando Matriz de Experimentação (com GridSearchCV) ===")

    for model_name, model in models_to_test.items():
        for feature_name, preprocessor in feature_sets.items():
            
            print(f"\n--- Testando: {model_name} | Features: {feature_name} ---")
            
            # 1. Criar o Pipeline completo
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('clf', model)
            ])
            
            # 2. Definir o Grid de Parâmetros
            # (Ajusta os parâmetros do grid para este pipeline específico)
            current_param_grid = {k: v for k, v in param_grids[model_name].items()}

            # 3. Executar o GridSearchCV
            # cv=3 faz 3-fold cross-validation
            # scoring='f1' foca na métrica F1 para a classe positiva (Ofensivo)
            grid_search = GridSearchCV(pipeline, current_param_grid, cv=3, scoring='f1', n_jobs=-1)
            
            # O fit é feito no DataFrame 'cru'
            # O pipeline trata de aplicar o ColumnTransformer
            grid_search.fit(train_df, y_train) 
            
            print(f"Melhor pontuação F1 (em validação cruzada): {grid_search.best_score_:.4f}")
            print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

            # 4. Avaliar o melhor modelo no conjunto de teste
            y_pred = grid_search.predict(test_df)
            
            # 5. Calcular métricas
            accuracy, precision, recall, f1, cm, report = calculate_metrics(y_test, y_pred)
            
            # 6. Guardar resultados
            all_results.append({
                "Modelo": model_name,
                "Features": feature_name,
                "Melhor F1 (CV)": grid_search.best_score_,
                "F1 (Teste)": f1,
                "Precisão (Teste)": precision,
                "Recall (Teste)": recall,
                "Melhores Parâmetros": str(grid_search.best_params_)
            })

            # Salvar relatório individual
            report_path = os.path.join(OUTPUT_DIR, f'report__{model_name}__{feature_name.replace(":", "")}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Modelo: {model_name} com Features: {feature_name}\n")
                f.write(f"Melhores Parâmetros: {grid_search.best_params_}\n\n")
                f.write(report)
                f.write("\nMatriz de Confusão:\n")
                f.write(str(cm))

    # --- Apresentação Final dos Resultados ---
    print("\n\n========== Resumo Comparativo da Meta 2 (Otimizado) ==========")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="F1 (Teste)", ascending=False)
    
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    # Mostrar colunas principais
    print(results_df[['Modelo', 'Features', 'F1 (Teste)', 'Precisão (Teste)', 'Recall (Teste)', 'Melhores Parâmetros']].to_string(index=False))
    
    # Salvar tabela resumo
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'results_summary_meta2.csv'), index=False)
    
    print("---------------------------------------------------------------")
    print(f"\nAnálise concluída. Relatórios individuais e resumo salvos em '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()