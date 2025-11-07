<div align="center">

<picture>
  <img alt="Adaptive Extractor Logo" src="docs/assets/logo.svg" width="25%" height="25%">
</picture>

**An PDF information extraction tool powered by LLM feedback optimization via caching.**

<h3>

[Documentation](docs/) • [Experiments](examples/) • [Report](reports/)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

</div>

# TODO - Melhorar, deixar mais direto e curto, apenas as porcentagens, redirecionar para o relatório completo com todos os detalhes.

## 🔥 **Resultados Principais**

💡 **Desempenho de Referência (1.000 documentos sintéticos):**  

- 🧠 **Precisão média:** `91.38%`  
- ⚡ **Tempo médio de processamento:** `3.28s`  
- 💰 **Redução progressiva de custo:** via **cache adaptativo de regras**

📊 **Comparativo:**  

- Esta implementação supera a extração base (**LLM puro**) com:  
  - ➡️ **–X%** de tempo de processamento  
  - ➡️ **–Y%** de custo total  
  - sem comprometer a **alta precisão**.

## Visão Geral

Este projeto apresenta um **pipeline inteligente de extração de dados** que aprende com o feedback de um LLM para reduzir progressivamente custos e tempo de processamento, mantendo uma alta precisão. Em vez de chamar LLMs caros para cada documento, resumidamente, o sistema:

1. **Extrai dados estruturados** de PDFs (com OCR) usando uma primeira LLM (*Extractor*) (gpt-5-mini).
2. **Gera regras de extração reutilizáveis** usando uma segunda LLM (*Rule Generator*), em padrões regex, a partir de extrações bem-sucedidas.
3. **Armazena e valida** essas regras geradas em um loop de feedback adaptativo, ajustando os prompts de geração por um número definido de iterações.
4. **Melhora progressivamente** a eficiência ao reutilizar regras validadas em documentos similares.

### A Pipeline "de cima"

```text
┌─────────────┐
│  PDF Texto  │
│   (OCR)     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐    Cache Hit?
│  Tentar Regras      │────────Sim────▶ ✓ Extração Rápida
│  em Cache           │             (Sem chamada ao LLM Extractor)
└──────┬──────────────┘
       │ Não
       ▼
┌─────────────────────┐
│  LLM Extractor      │  ◀── Extração estruturada
│  (gpt-5-mini)       │       com schema Pydantic
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  LLM Rule Generator │  ◀── Gera regras regex
│  (gpt-5-mini)       │        com validação
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Loop de Feedback   │  ◀── Valida e refina as regras geradas
│  (Validação)        │           (max N tentativas)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Cache Adaptativo   │  ◀── Armazena regras
│  (LRU + Pesos)      │      validadas para uso futuro
└─────────────────────┘
```

**A Otimização**: Ao gerar e armazenar regras de extração, cada extração bem-sucedida torna o sistema mais rápido e barato para os próximos documentos **similares**. A cache se adapta usando um LRU (Least Recently Used) ponderado, priorizando regras frequentemente bem-sucedidas.

## Experimentos e Resultados

Acesse o **[Relatório do Weights & Biases](https://wandb.ai/your_username/your_project)** para a visualização completa dos experimentos.

Veja a [Documentação de Experimentos](https://www.google.com/search?q=./docs/experiments.md) para a análise detalhada.

## Documentação Extendida

### Conceitos Principais

- **[Arquitetura do Pipeline](https://www.google.com/search?q=./docs/pipeline.md)** - Como o sistema dual-LLM funciona
- **[Sistema de Cache Adaptativo](https://www.google.com/search?q=./docs/cache.md)** - Cache LRU com priorização ponderada de regras
- **[Geração e Validação de Regras](https://www.google.com/search?q=./docs/rules.md)** - Como as regras são criadas, validadas e refinadas
- **[Geração de Dados Sintéticos](https://www.google.com/search?q=./docs/synthetic_data.md)** - Simulando a imprecisão de documentos OCR

## Como Rodar

```bash
# Clone esse repositório e entre no seu diretório
git clone https://github.com/GustavoZiel/adaptive-PDF-extractor.git
cd adaptive-PDF-extractor

# Instale as dependências
uv sync

# Ative o ambiente virtual
source .venv/bin/activate

# Crie um arquivo .env na raiz do projeto (Seguindo exemplo em .env.example)
cp .env.example .env

# Configure a API key do OpenAI no .env
echo 'OPENAI_API_KEY="sua_api_key_aqui"' >> .env

# Veja todas as opções de configuração da pipeline
uv run src/main.py --help

# Veja todas as opções de configuração para geração de dados sintéticos
python3 -m scripts.generate_fake_data --help

# Gere dados sintéticos de exemplo (1.000 documentos)
python3 -m scripts.generate_fake_data \
  --save-path data/fake \
  --dataset-filename dataset.json \
  --num-samples 1000 \
  --seed 1

# Rode o pipeline nos dados de exemplo
uv run src/main.py \
  --data-folder data/fake \
  --dataset-filename dataset.json \
  --cache-filename cache.json \
  --max-attempts 5 \
  --use-wandb
```

## Estrutura do Projeto

```text
enter_ai_fellowship/
├── src/
│   ├── main.py           # Orquestração principal do pipeline
│   ├── cache.py          # Implementação do cache adaptativo
│   ├── rule.py           # Geração e execução de regras
│   ├── models.py         # Inicialização dos LLMs e prompts
│   └── utils.py          # Utilitários de processamento de dados
├── data/
│   ├── dataset.json      # Documentos de entrada
│   ├── processed/        # Resultados processados
│   └── cache/            # Regras de extração em cache
├── docs/                 # Documentação detalhada
├── notebooks/            # Jupyter notebooks para análise
└── Experiments/          # Resultados e métricas de experimentos
```

## Tecnologias Utilizadas

- **LLM**: OpenAI gpt-5-mini (configurável)
- **Framework**: LangChain para orquestração
- **Validação**: Pydantic para saídas estruturadas
- **Tracking**: Weights & Biases + Weave para log de experimentos
- **Linguagem**: Python 3.11+

## Agradecimentos

Agradeço a oportunidade de realizar esse projeto, me diverti bastante e aprendi muito também! 🚀

> Gustavo
