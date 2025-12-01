import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, accuracy_score
import time
import platform

# --- CONFIGURAÇÕES GERAIS ---
DATASET_URL = "hf://datasets/franciellevargas/HateBR/HateBR.csv" 
AMOSTRA_TAMANHO = 20  # Mantém 20 para testes rápidos. Põe 50 ou 100 para o relatório final.
MODEL_ID = "google/gemma-2-2b-it" 

print(f"--> A iniciar pipeline com o modelo: {MODEL_ID}")
print(f"--> Sistema: {platform.system()} | Processador: {platform.processor()}")

# --- 1. CONFIGURAÇÃO DE HARDWARE (Universal) ---
model_kwargs = {}
if torch.cuda.is_available():
    print("✅ Hardware: NVIDIA GPU (CUDA). Modo 4-bit ativado.")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs = {"quantization_config": nf4_config, "low_cpu_mem_usage": True}
elif torch.backends.mps.is_available():
    print("🍎 Hardware: Apple Silicon (MPS). Modo float16 ativado.")
    model_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
else:
    print("⚠️ Hardware: CPU. Vai ser lento.")
    model_kwargs = {"low_cpu_mem_usage": True}

# --- 2. CARREGAR DADOS ---
print(f"--> A carregar dados...")
try:
    df_original = pd.read_csv(DATASET_URL)
    # Ajuste das colunas conforme verificámos antes
    df_original = df_original[['comentario', 'label_final']].rename(
        columns={'comentario': 'texto', 'label_final': 'label_real'}
    )
    df_original['label_texto'] = df_original['label_real'].map({0: 'Não-ofensivo', 1: 'Ofensivo'})
    
    if AMOSTRA_TAMANHO:
        # Importante: Amostragem aleatória para evitar viés
        df = df_original.sample(n=AMOSTRA_TAMANHO, random_state=42).copy()
        print(f"--> Amostra aleatória carregada: {len(df)} linhas.")
        print(f"--> Distribuição: {df['label_texto'].value_counts().to_dict()}")
    else:
        df = df_original.copy()
        
except Exception as e:
    print(f"❌ Erro ao ler dataset: {e}")
    exit()

# --- 3. CARREGAR MODELO ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = pipeline("text-generation", model=MODEL_ID, model_kwargs=model_kwargs, device_map="auto")
    print("--> Modelo pronto!")
except Exception as e:
    print(f"❌ Erro no modelo: {e}")
    exit()

# --- 4. DEFINIÇÃO DAS 4 PROMPTS (A Evolução) ---

def get_prompt_1_zeroshot(texto):
    # Prompt 1: Básica e direta (Baseline)
    return [{"role": "user", "content": f"""
    Classifica o comentário abaixo como 'Ofensivo' ou 'Não-ofensivo'.
    Responde apenas com a etiqueta.
    
    Comentário: "{texto}"
    Classificação:"""}]

def get_prompt_2_fewshot_basic(texto):
    # Prompt 2: Few-Shot (Ensina o formato)
    return [{"role": "user", "content": f"""
    Classifica o comentário como 'Ofensivo' ou 'Não-ofensivo'.
    
    Exemplos:
    Comentário: "Adoro este lugar!" -> Classificação: Não-ofensivo
    Comentário: "Seu idiota inútil." -> Classificação: Ofensivo
    
    Comentário: "{texto}"
    Classificação:"""}]

def get_prompt_3_fewshot_advanced(texto):
    # Prompt 3: Few-Shot Focado em Casos Difíceis (Ironia/Crítica vs Ódio)
    return [{"role": "user", "content": f"""
    Analisa o comentário para detetar discurso de ódio.
    Distingue entre crítica válida (Não-ofensivo) e insultos (Ofensivo).
    
    Exemplos:
    "Não concordo nada contigo." -> Não-ofensivo
    "És uma vergonha para a humanidade." -> Ofensivo
    "O serviço foi péssimo." -> Não-ofensivo
    
    Comentário: "{texto}"
    Classificação:"""}]

def get_prompt_4_persona(texto):
    # Prompt 4: Persona (Content Moderator) + Regras Estritas
    return [{"role": "user", "content": f"""
    Tu és um Moderador de Conteúdo de uma rede social.
    A tua tarefa é identificar toxicidade.
    - Se for apenas opinião negativa, marca 'Não-ofensivo'.
    - Se tiver insultos ou ataques pessoais, marca 'Ofensivo'.
    
    Comentário: "{texto}"
    Classificação (Ofensivo/Não-ofensivo):"""}]

# Lista de prompts para o loop
prompts_list = [
    (1, get_prompt_1_zeroshot),
    (2, get_prompt_2_fewshot_basic),
    (3, get_prompt_3_fewshot_advanced),
    (4, get_prompt_4_persona)
]

def limpar_resposta(texto_gerado):
    t = texto_gerado.lower()
    if "não-ofensivo" in t or "não ofensivo" in t: return "Não-ofensivo"
    if "ofensivo" in t: return "Ofensivo"
    return "Erro"

# --- 5. EXECUÇÃO EM LOOP ---

for num_prompt, func_prompt in prompts_list:
    print(f"\n🚀 A executar PROMPT {num_prompt}...")
    
    predicoes = []
    start_time = time.time()
    
    for i, row in df.iterrows():
        # Gerar a prompt específica desta rodada
        messages = func_prompt(row['texto'])
        
        # Inferência
        saida = pipe(messages, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        res_limpa = limpar_resposta(saida[0]["generated_text"][-1]["content"])
        predicoes.append(res_limpa)
        
        # Feedback visual simples (ponto a cada inferência)
        print(".", end="", flush=True)
    
    # Guardar resultados desta prompt numa coluna temporária para cálculo
    col_name = f"pred_prompt_{num_prompt}"
    df[col_name] = predicoes
    
    # Métricas
    # 1. Calcular acurácia REAL (considerando "Erro" como falha)
    # Usamos o df completo, não o filtrado
    acc = accuracy_score(df['label_texto'], df[col_name])
    
    # 2. Contar quantos erros de formatação ocorreram
    num_erros_formato = len(df[df[col_name] == "Erro"])
    
    print(f"\n✅ Prompt {num_prompt} terminada!")
    print(f"   Acurácia Real: {acc:.2f}")
    print(f"   Respostas Inválidas (Erro): {num_erros_formato}/{len(df)}")
    
    # --- GUARDAR CSV INDIVIDUAL ---
    nome_modelo_limpo = MODEL_ID.split('/')[-1]
    nome_ficheiro = f"resultados_{num_prompt}_{nome_modelo_limpo}.csv"
    
    # Guardamos apenas as colunas relevantes para este ficheiro
    df_final = df[['texto', 'label_real', 'label_texto', col_name]].copy()
    df_final.rename(columns={col_name: 'predicao'}, inplace=True)
    
    path_guardar = f"src/meta3/{nome_ficheiro}"
    df_final.to_csv(path_guardar, index=False)
    print(f"💾 Ficheiro guardado: {path_guardar}")

print("\n🏁 Processo concluído para todas as 4 prompts!")