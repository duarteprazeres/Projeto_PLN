import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer # Para stemming em português

# Definir as stopwords em português
STOPWORDS_PT = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess_text(text, remove_punctuation=False, apply_stemming=True):
    """
    Realiza o pré-processamento do texto:
    - Minúsculas
    - Remove links, menções (@) e hashtags (#)
    - Remove números
    - Remove espaços extras
    - Tokenização
    - Remove stopwords
    - Opcionalmente remove pontuação e aplica stemming.
    """
    original_text_for_features = text # Guarda o texto original para extrair features de pontuação depois

    # minúsculas
    text = text.lower()
    # Remover links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remover menções de usuários
    text = re.sub(r'@\w+', '', text)
    # Remover hashtags (manter a palavra, mas remover o #
    text = re.sub(r'#\w+', '', text)
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Decisão sobre a pontuação ---
    # Para capturar a intensidade, não vamos remover a pontuação AQUI
    # Ela será tokenizada pelo word_tokenize.
    # Se 'remove_punctuation' for True, faremos isso APÓS a tokenização
    # ou antes se decidirmos que queremos palavras puras para o léxico.
    # Por agora, vamos manter para a tokenização.

    tokens = word_tokenize(text, language='portuguese')

    processed_tokens = []
    for word in tokens:
        if remove_punctuation:
            word = word.translate(str.maketrans('', '', string.punctuation))
        
        if word and word not in STOPWORDS_PT: # Garante que a palavra não está vazia após remover pontuação
            if apply_stemming:
                processed_tokens.append(stemmer.stem(word))
            else:
                processed_tokens.append(word)

    return processed_tokens, original_text_for_features # Retorna também o texto original para features de pontuação

def calculate_metrics(y_true, y_pred, labels=[0, 1]):
    """
    Calcula e retorna as métricas de avaliação para classificação binária.
    labels: [0=Não Ofensivo, 1=Ofensivo]
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=['Não Ofensivo', 'Ofensivo'], zero_division=0)

    return accuracy, precision, recall, f1, cm, report

# --- Termos ofensivos (exemplo; atualiza conforme extração dos dados) ---
OFFENSIVE_TERMS = {
    'idiota', 'imbecil', 'merda', 'bosta', 'vagabundo', 'corrupto',
    'lixo', 'nojento', 'ridículo', 'burro', 'desgraçado', 'fdp',
    'vergonha', 'pirralha', 'destruir', 'matar', 'atacar',
    'caluniar', 'difamar', 'xingar', 'processar'
}

def gerar_lexico_ofensivo(df, n=40, stopwords=None, extra_filtrar=None):
    """
    Gera os n termos mais comuns em textos ofensivos do dataset.
    """
    ofensivos = df[df['label'] == 1]['processed_tokens_no_punct'].explode()
    freq = Counter(ofensivos)
    termos = [w for w, _ in freq.most_common(n*2)]
    if stopwords is not None:
        termos = [w for w in termos if w not in stopwords]
    if extra_filtrar is not None:
        termos = [w for w in termos if w not in extra_filtrar]
    return set(termos[:n])


# --- Léxicos de Sentimento (Exemplo Simplificado) ---
SENTIMENT_LEXICON = {
    'ótimo': 1, 'bom': 0.8, 'feliz': 0.7, 'gostei': 0.6,
    'ruim': -1, 'péssimo': -0.8, 'triste': -0.7, 'ódio': -0.9, 'horrível': -1,
    'mau': -0.8, 'ofensivo': -1, 'abominável': -1, 'vergonha': -0.8,
    'nojo': -0.9, 'mentira': -0.7, 'fraco': -0.5,
}

def get_sentiment_score(tokens):
    """
    Calcula um score de sentimento básico para uma lista de tokens.
    """
    score = 0
    for token in tokens:
        score += SENTIMENT_LEXICON.get(token, 0) # 0 se a palavra não estiver no léxico
    return score

def extract_manual_features(df):
    """
    Retorna array de features linguísticas manuais para o modelo (pode ser expandido).
    """
    # Converte boolean repeated_punct para int, para ML
    features = df[['sentiment_score', 'exclamation_count', 'question_mark_count', 'repeated_punct']].copy()
    features['repeated_punct'] = features['repeated_punct'].astype(int)
    return features.values