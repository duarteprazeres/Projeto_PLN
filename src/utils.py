import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer # Para stemming em português

# Definir as stopwords em português
STOPWORDS_PT = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess_text(text, remove_punctuation=False, apply_stemming=True):
    """
    Realiza o pré-processamento de um texto:
    - Minúsculas
    - Remove links, menções (@) e hashtags (#)
    - Remove números
    - Remove espaços extras
    - Tokenização
    - Remove stopwords
    - Opcionalmente remove pontuação e aplica stemming.
    """
    original_text_for_features = text # Guardar o texto original para extrair features de pontuação depois

    text = text.lower() # Minúsculas

    # Remover links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remover menções de usuários
    text = re.sub(r'@\w+', '', text)
    # Remover hashtags (manter a palavra, mas remover o #, ou remover tudo dependendo da estratégia)
    # Por agora, vamos remover tudo, mas pode-se decidir manter a palavra sem o #
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
        # Se decidirmos remover pontuação para a análise de palavras (e.g., léxicos),
        # podemos fazê-lo aqui por token.
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

# --- Dicionários de Termos Ofensivos (Exemplos - você deve expandir e refinar) ---
# Estes são exemplos, você precisará construir dicionários mais abrangentes
# baseados na análise do seu dataset HateBR.
OFFENSIVE_TERMS = {
    'idiota', 'imbecil', 'merda', 'bosta', 'vagabundo', 'corrupto',
    'lixo', 'nojento', 'ridículo', 'burro', 'desgraçado', 'fdp','vergonha', 'pirralha'
    # Adicione mais termos que encontrar no HateBR
}

# Termos que podem indicar intenção de ódio ou ataque
ATTACK_INTENT_TERMS = {
    'destruir', 'matar', 'atacar', 'caluniar', 'difamar', 'xingar', 'processar',
    # Expanda com base na sua análise
}

# --- Léxicos de Sentimento (Exemplo Simplificado) ---
# Para uma análise de sentimento mais robusta, você precisaria de um léxico
# em português com scores para muitas palavras, ou usar uma biblioteca NLP.
# Este é um exemplo bem simplificado para demonstrar a ideia.
SENTIMENT_LEXICON = {
    'ótimo': 1, 'bom': 0.8, 'feliz': 0.7, 'gostei': 0.6,
    'ruim': -1, 'péssimo': -0.8, 'triste': -0.7, 'ódio': -0.9, 'horrível': -1,
    'mau': -0.8, 'ofensivo': -1, 'abominável': -1, 'vergonha': -0.8,
    'nojo': -0.9, 'mentira': -0.7, 'fraco': -0.5,
    # Adicione mais palavras com polaridade
}

def get_sentiment_score(tokens):
    """
    Calcula um score de sentimento básico para uma lista de tokens.
    """
    score = 0
    for token in tokens:
        score += SENTIMENT_LEXICON.get(token, 0) # 0 se a palavra não estiver no léxico
    return score