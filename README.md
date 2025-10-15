# Projeto PLN

Este repositório contém o desenvolvimento do nosso projeto de Processamento de Linguagem Natural (PLN).

## O que já foi feito

- **Configuração do ambiente:**  
    - Estrutura inicial do repositório criada.
    - Criei um ambiente virtual onde estão lá todos os pacotes(dependencias) a utilizar no projeto
    - Dependências principais instaladas diretamente no ambiente virtual.

- **Coleta de dados:**  
    - Os dados são carregados diretamente de uma fonte online (Hugging Face), não há pasta `/data` local (o data frame nao está na pasta do projeto, é carregado automaticamente do hugging face).

- **Pré-processamento:**  
    - Scripts para limpeza e normalização dos textos implementados.
    - Decidi não remover numa fase inicial a pontuação, criando tokens para ela. Decidi tomar esta decisão para experimentar a ver se tem alguma relevância dado que a pontuação costuma ser utilizada para dar mais intensidade à raiva num insulto/discurso de ódio.
    - Remoção de stopwords e pontuação.

- **Exploração dos dados:**  
    - Análise exploratória básica realizada (frequência de palavras, tamanho dos textos).

- **Documentação:**  
    - README inicial criado para orientar o desenvolvimento.

## Próximos passos

- Implementar modelos de PLN.
- Avaliar resultados e ajustar pré-processamento.
- Documentar experimentos e resultados.

Qualquer dúvida, avisa!