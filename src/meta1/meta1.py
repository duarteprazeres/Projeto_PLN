
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import re # Importar regex
from meta1.utils import preprocess_text, calculate_metrics, OFFENSIVE_TERMS, ATTACK_INTENT_TERMS, get_sentiment_score
import nltk
import spacy 
# data required for the examples and exercises in the book
nltk.download("book")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('rslp') # Necessário para o Stemmer de Português [6]
nltk.download('stopwords') # Necessário para a lista de stopwords [6]
nltk.download('punkt')

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/franciellevargas/HateBR/HateBR.csv")

# Exibir as 5 primeiras linhas do dataframe
print(df.head())


# --- Configurações ---

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Carregamento e Pré-processamento dos Dados ---
def load_and_preprocess_data(df):
    print("Carregando e pré-processando os dados...")
    try:
        
        df = df[['comentario', 'label_final']].copy()
        df.rename(columns={'comentario': 'text', 'label_final': 'label'}, inplace=True)

        # Aplicar pré-processamento. processed_tokens conterá a pontuação como tokens.
        # Guardamos o texto original para features de pontuação.
        df['processed_tokens_with_punct'], df['original_text'] = zip(*df['text'].apply(preprocess_text, remove_punctuation=False, apply_stemming=False))
        # Para léxicos de sentimento, podemos querer uma versão sem pontuação para evitar '!!!ódio'
        df['processed_tokens_no_punct'], _ = zip(*df['text'].apply(preprocess_text, remove_punctuation=True, apply_stemming=False))


        # Extrair features de pontuação aqui, do 'original_text'
        df['exclamation_count'] = df['original_text'].apply(lambda x: x.count('!'))
        df['question_mark_count'] = df['original_text'].apply(lambda x: x.count('?'))
        df['repeated_punct'] = df['original_text'].apply(lambda x: bool(re.search(r'([!?.]){2,}', x))) # Pelo menos 2 pontuações seguidas

        print(f"Dados carregados e pré-processados. Total de {len(df)} amostras.")
        return df
    except FileNotFoundError:
        print(f"Erro: O ficheiro '{df}' não foi encontrado. Verifique o caminho.")
        exit()
    except Exception as e:
        print(f"Erro ao carregar ou processar dados: {e}")
        exit()

# --- 2. Análise de Dados e Extração de Conhecimento Linguístico ---
def analyze_linguistic_knowledge(df):
    print("\nRealizando análise de conhecimento linguístico...")

    # Usar tokens sem pontuação para unigramas para evitar 'palavra!!!' como token separado
    offensive_words = df[df['label'] == 1]['processed_tokens_no_punct'].explode()
    non_offensive_words = df[df['label'] == 0]['processed_tokens_no_punct'].explode()

    print("\n--- Unigramas mais frequentes em textos ofensivos (sem pontuação) ---")
    offensive_unigrams = Counter(offensive_words).most_common(20)
    for word, count in offensive_unigrams:
        print(f"{word}: {count}")

    print("\n--- Unigramas mais frequentes em textos não ofensivos (sem pontuação) ---")
    non_offensive_unigrams = Counter(non_offensive_words).most_common(20)
    for word, count in non_offensive_unigrams:
        print(f"{word}: {count}")
    
    # Análise de Sentimento (usando tokens sem pontuação para melhor correspondência com léxico)
    df['sentiment_score'] = df['processed_tokens_no_punct'].apply(get_sentiment_score)
    print(f"\nScore médio de sentimento (Ofensivo): {df[df['label'] == 1]['sentiment_score'].mean():.2f}")
    print(f"Score médio de sentimento (Não Ofensivo): {df[df['label'] == 0]['sentiment_score'].mean():.2f}")

    # Análise das features de pontuação
    print(f"\nScore médio de exclamações (Ofensivo): {df[df['label'] == 1]['exclamation_count'].mean():.2f}")
    print(f"Score médio de exclamações (Não Ofensivo): {df[df['label'] == 0]['exclamation_count'].mean():.2f}")
    print(f"Proporção com pontuação repetida (Ofensivo): {df[df['label'] == 1]['repeated_punct'].mean():.2f}")
    print(f"Proporção com pontuação repetida (Não Ofensivo): {df[df['label'] == 0]['repeated_punct'].mean():.2f}")


# --- 3. Definição e Justificação de Regras ---
def classify_with_rules(row):
    """
    Classifica um texto como ofensivo (1) ou não ofensivo (0) baseado em regras,
    utilizando tokens (com/sem pontuação), score de sentimento e features de pontuação.
    Retorna 1 para ofensivo, 0 para não ofensivo.
    """
    tokens_no_punct = row['processed_tokens_no_punct'] # Para léxico
    tokens_with_punct = row['processed_tokens_with_punct'] # Se quiser tokens como '!!!'
    sentiment_score = row['sentiment_score']
    exclamation_count = row['exclamation_count']
    question_mark_count = row['question_mark_count']
    repeated_punct = row['repeated_punct']

    # REGRA 1: Presença de termos explicitamente ofensivos (sem pontuação para match exato com léxico)
    if any(term in tokens_no_punct for term in OFFENSIVE_TERMS):
        return 1

    # REGRA 2: Presença de termos de intenção de ataque (sem pontuação)
    if any(term in tokens_no_punct for term in ATTACK_INTENT_TERMS):
        return 1

    # REGRA 3: Score de sentimento muito negativo
    if sentiment_score < -0.6: # Ajuste este limiar
        return 1
    
    # REGRA 4: Alta contagem de pontos de exclamação
    if exclamation_count >= 2: # Se há 2 ou mais exclamações
        return 1

    # REGRA 5: Presença de pontuação repetida (!!!, ???, !?, etc.)
    if repeated_punct:
        return 1

    # REGRA 6: Combinação de regras (ex: termo ofensivo E alta pontuação)
    if (any(term in tokens_no_punct for term in OFFENSIVE_TERMS) or sentiment_score < -0.4) and repeated_punct:
        return 1

    # Se nenhuma regra de ofensividade for acionada, classifica como não ofensivo
    return 0

# --- 4. Avaliação do Sistema ---
def main(df):
    df = load_and_preprocess_data(df)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])

    print(f"\nDados divididos: Treino={len(train_df)} amostras, Teste={len(test_df)} amostras.")
    print("--------------------------------------------------")

    # Realizar a análise de conhecimento linguístico nos dados de treino
    analyze_linguistic_knowledge(train_df)

    # Aplicar o classificador baseado em regras ao conjunto de teste
    print("\nAplicando classificador baseado em regras ao conjunto de teste...")

    test_df['sentiment_score'] = test_df['processed_tokens_no_punct'].apply(get_sentiment_score) # Calcular o sentiment_score também no conjunto de teste
    test_df['predicted_label'] = test_df.apply(classify_with_rules, axis=1) # Passa a linha inteira para a função de regras

    # Calcular as métricas de avaliação
    y_true = test_df['label']
    y_pred = test_df['predicted_label']

    accuracy, precision, recall, f1, cm, report = calculate_metrics(y_true, y_pred)

    print("\n--- Resultados da Meta 1 (Classificador Baseado em Regras) ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Classe Ofensivo): {precision:.4f}")
    print(f"Recall (Classe Ofensivo): {recall:.4f}")
    print(f"F1-Score (Classe Ofensivo): {f1:.4f}")
    print("\nMatriz de Confusão:\n", cm)
    print("\nRelatório de Classificação:\n", report)

    # Salvar resultados e análise de erros (igual à versão anterior)
    with open(os.path.join(OUTPUT_DIR, 'results_meta1.txt'), 'w', encoding='utf-8') as f:
        f.write("--- Resultados da Meta 1 (Classificador Baseado em Regras) ---\n")
        f.write(f"Acurácia: {accuracy:.4f}\n")
        f.write(f"Precisão (Classe Ofensivo): {precision:.4f}\n")
        f.write(f"Recall (Classe Ofensivo): {recall:.4f}\n")
        f.write(f"F1-Score (Classe Ofensivo): {f1:.4f}\n")
        f.write("\nMatriz de Confusão:\n")
        f.write(str(cm) + "\n")
        f.write("\nRelatório de Classificação:\n")
        f.write(report + "\n")
        f.write("\nExemplos de Falsos Positivos:\n")
        fp = test_df[(test_df['label'] == 0) & (test_df['predicted_label'] == 1)]
        if not fp.empty:
            for i, row in fp.head(5).iterrows():
                f.write(f"- Texto: {row['text']} (Previsão: Ofensivo, Real: Não Ofensivo)\n")
        else:
            f.write("- Nenhum Falso Positivo encontrado nos 5 primeiros exemplos.\n")

        f.write("\nExemplos de Falsos Negativos:\n")
        fn = test_df[(test_df['label'] == 1) & (test_df['predicted_label'] == 0)]
        if not fn.empty:
            for i, row in fn.head(5).iterrows():
                f.write(f"- Texto: {row['text']} (Previsão: Não Ofensivo, Real: Ofensivo)\n")
        else:
            f.write("- Nenhum Falso Negativo encontrado nos 5 primeiros exemplos.\n")

    print(f"\nResultados salvos em {os.path.join(OUTPUT_DIR, 'results_meta1.txt')}")

    print("\n--- Análise de Erros (Exemplos) ---")
    print("\nFalsos Positivos (Predito Ofensivo, Real Não Ofensivo):")
    fp = test_df[(test_df['label'] == 0) & (test_df['predicted_label'] == 1)]
    if not fp.empty:
        for i, row in fp.head(3).iterrows():
            print(f"- Texto: {row['text']}\n  Tokens (sem pont.): {row['processed_tokens_no_punct']}\n  Score Sentimento: {row['sentiment_score']:.2f}\n  Exclamações: {row['exclamation_count']}\n  Pontuação Repetida: {row['repeated_punct']}\n")
    else:
        print("- Nenhum Falso Positivo encontrado nos 3 primeiros exemplos.")


    print("\nFalsos Negativos (Predito Não Ofensivo, Real Ofensivo):")
    fn = test_df[(test_df['label'] == 1) & (test_df['predicted_label'] == 0)]
    if not fn.empty:
        for i, row in fn.head(3).iterrows():
            print(f"- Texto: {row['text']}\n  Tokens (sem pont.): {row['processed_tokens_no_punct']}\n  Score Sentimento: {row['sentiment_score']:.2f}\n  Exclamações: {row['exclamation_count']}\n  Pontuação Repetida: {row['repeated_punct']}\n")
    else:
        print("- Nenhum Falso Negativo encontrado nos 3 primeiros exemplos.")


if __name__ == "__main__":
    main(df)
