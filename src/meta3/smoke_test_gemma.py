import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, accuracy_score
import time
import platform

# CONFIGURAÇÕES
# Link direto do Hugging Face (Requer login via terminal)
DATASET_URL = "hf://datasets/franciellevargas/HateBR/HateBR.csv" 

# Configurações do teste
AMOSTRA_TAMANHO = 20  # None = corre o dataset todo (demorado!)
MODEL_ID = "google/gemma-2-2b-it" 
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"--> A iniciar pipeline com o modelo: {MODEL_ID}")
print(f"--> Sistema detetado: {platform.system()} | Processador: {platform.processor()}")

# DETEÇÃO DE HARDWARE (Universal Mac/Windows)
model_kwargs = {}

if torch.cuda.is_available():
    # MODO WINDOWS (Colega com NVIDIA)
    print("✅ Hardware: NVIDIA GPU detetada (Modo CUDA).")
    print("--> Ativando compressão 4-bit para máxima eficiência.")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs = {"quantization_config": nf4_config, "low_cpu_mem_usage": True}

elif torch.backends.mps.is_available():
    # MODO MAC
    print("🍎 Hardware: Apple Silicon detetado (Modo MPS/Metal).")
    print("--> Ativando modo nativo float16.")
    model_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}

else:
    # MODO CPU (Emergência)
    print("⚠️ Hardware: Apenas CPU detetado. Vai ser lento!")
    model_kwargs = {"low_cpu_mem_usage": True}

# --- 2. CARREGAR DADOS (Direto do Hugging Face) ---
print(f"--> A descarregar dataset de: {DATASET_URL}...")
try:
    # O pandas lê direto do URL usando a autenticação do teu 'huggingface-cli login'
    df = pd.read_csv(DATASET_URL)
    
    # Seleção das colunas
    df = df[['comentario', 'label_final']].rename(
        columns={'comentario': 'texto', 'label_final': 'label_real'}
    )
    
    # Mapeamento para texto (facilita a comparação com o LLM)
    df['label_texto'] = df['label_real'].map({0: 'Não-ofensivo', 1: 'Ofensivo'})
    
    if AMOSTRA_TAMANHO:
        df = df.sample(n=AMOSTRA_TAMANHO, random_state=42)
        
        # Mostra quantas de cada tipo apanhámos para garantir que está equilibrado
        print(f"--> Distribuição da amostra:\n{df['label_texto'].value_counts()}")
    print(f"--> Dados carregados: {len(df)} linhas.")
    
except Exception as e:
    print(f"❌ Erro ao ler dataset: {e}")
    print("Dica: Verifica se fizeste 'huggingface-cli login' no terminal.")
    exit()

#3. CARREGAR MODELO
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs=model_kwargs,
        device_map="auto", 
    )
    print("--> Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro crítico ao carregar modelo: {e}")
    exit()

#4. ENGENHARIA DE PROMPT
def gerar_prompt(comentario):
    # Prompt Few-Shot (Com exemplos)
    messages = [
        {"role": "user", "content": f"""
        Analisa o comentário abaixo e classifica como 'Ofensivo' ou 'Não-ofensivo'.
        
        Exemplos:
        Comentário: "Adorei a foto, estás linda!"
        Classificação: Não-ofensivo

        Comentário: "Que burra, nem devias estar aqui."
        Classificação: Ofensivo
        
        Comentário: "{comentario}"
        Classificação:"""}
    ]
    return messages

def limpar_resposta(texto_gerado):
    # Função auxiliar para limpar a resposta do modelo
    texto = texto_gerado.lower()
    if "não-ofensivo" in texto or "não ofensivo" in texto:
        return "Não-ofensivo"
    elif "ofensivo" in texto:
        return "Ofensivo"
    else:
        return "Erro"

# 5. EXECUÇÃO DO TESTE
print("\n--> A classificar comentários (isto pode demorar um pouco)...")
start_time = time.time()

predicoes = []

for i, texto in enumerate(df['texto']):
    # Mostra progresso a cada 5 linhas
    if i % 5 == 0: print(f"Processando linha {i}/{len(df)}...")
    
    # 1. Gerar prompt
    prompt = gerar_prompt(texto)
    
    # 2. Chamar modelo
    saida = pipe(prompt, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # 3. Processar resposta
    resposta_crua = saida[0]["generated_text"][-1]["content"]
    resposta_limpa = limpar_resposta(resposta_crua)
    
    predicoes.append(resposta_limpa)

df['predicao'] = predicoes
tempo_total = time.time() - start_time

# 6. RESULTADOS
print("\n" + "="*30)
print(f"MODELO: {MODEL_ID}")
print(f"TEMPO: {tempo_total:.2f} segundos")
print("="*30)

# Filtrar erros para cálculo de métricas
df_validos = df[df['predicao'] != "Erro"]
erros_count = len(df) - len(df_validos)

if len(df_validos) > 0:
    print(f"Acurácia: {accuracy_score(df_validos['label_texto'], df_validos['predicao']):.2f}")
    print(f"Erros de formatação (respostas inválidas): {erros_count}")
    print("\nRelatório Detalhado:")
    print(classification_report(df_validos['label_texto'], df_validos['predicao']))
else:
    print("⚠️ O modelo não gerou nenhuma resposta válida (verificar Prompt).")

# Guardar resultados
nome_csv = f"resultados_{MODEL_ID.split('/')[-1]}.csv"
df.to_csv(nome_csv, index=False)
print(f"--> Resultados guardados em: {nome_csv}")