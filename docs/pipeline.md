### Pipeline de Extração Adaptativa

O sistema processa cada documento PDF através de um pipeline de **três etapas** que otimiza progressivamente a extração de dados estruturados via **aprendizado de regras** e **cache adaptativo**.

---

### Fluxo Principal

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRADA: PDF → OCR → Texto bruto                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  ETAPA 1: Fast Path           │
         │  Extração via Cache           │
         └───────────┬───────────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Todos os campos extraídos?
        └────┬─────────────────┬───┘
             │ SIM             │ NÃO
             │                 │
             ▼                 ▼
    ┌────────────────┐  ┌──────────────────────┐
    │ ✓ Retorna      │  │  ETAPA 2: Slow Path  │
    │   Resposta     │  │  Extração via LLM     │
    └────────────────┘  └──────┬───────────────┘
                               │
                   ┌───────────▼────────────────┐
                   │  ETAPA 3: Rule Learning    │
                   │  Geração + Validação       │
                   └───────────┬────────────────┘
                               │
                   ┌───────────▼────────────────┐
                   │  Adiciona Regras ao Cache  │
                   │  (LRU + Peso)              │
                   └────────────────────────────┘
```

---

### ETAPA 1: Extração via Cache (Fast Path)

Tenta extrair **todos os campos** usando regras previamente aprendidas armazenadas no **cache adaptativo**.

**Entrada:**
* Texto bruto do documento
* Lista de campos a extrair
* Cache com regras (organizado por tipo de documento)

**Processo:**

```python
for campo in campos_necessarios:
    valor = cache.try_extract(campo, texto)
    
    if valor != None:
        # ✓ Cache Hit - Regra encontrada e aplicada
        campos_sucesso.append(campo)
    else:
        # ✗ Cache Miss - Precisa extração via LLM
        campos_falhados.append(campo)
```

**Saída:**
* **Campos extraídos com sucesso** → Retornados imediatamente
* **Campos falhados** → Passam para ETAPA 2

**Otimização:**

Quando uma regra extrai um campo com sucesso:
1. Seu **peso é incrementado** (+1)
2. É **reposicionada na lista** (regras mais usadas ficam no início)
3. Próximas buscas são **mais rápidas** (verificação sequencial otimizada)

---

### ETAPA 2: Extração via LLM (Slow Path)

Campos que falharam no cache são extraídos usando a **Extractor LLM** (gpt-5-mini).

**Entrada:**
* Texto bruto do documento
* **Apenas** os campos que falharam no cache
* Schema de extração com descrições dos campos

**Processo:**

```python
# Cria modelo Pydantic dinâmico apenas para campos falhados
schema_falhados = {campo: descricao for campo in campos_falhados}
modelo_pydantic = create_pydantic_model(schema_falhados)

# Configura agente LangChain com saída estruturada
agent = create_extraction_agent(llm, modelo_pydantic)

# Extrai campos usando prompt de extração
resposta = agent.invoke({
    "messages": [{
        "role": "user",
        "content": EXTRACTION_PROMPT.format(
            text=texto,
            schema=schema_completo
        )
    }]
})

valores_extraidos = resposta["structured_response"]
```

**Características da Extração:**

* **Extração literal** - captura texto exatamente como aparece
* **Validação por descrição** - usa descrições do schema para validar formato
* **Política NULL estrita** - retorna `None` quando campo está ausente/inválido
* **Saída estruturada** - garante JSON válido via Pydantic

**Saída:**
* Dicionário com valores extraídos (normalizados)
* Passa para ETAPA 3 para aprendizado de regras

---

### ETAPA 3: Geração e Validação de Regras (Rule Learning)

Para cada campo extraído com sucesso pela LLM, o sistema tenta **gerar uma regra reutilizável** usando a **Rule Generator LLM**.

**Entrada:**
* Texto bruto do documento
* Campo e valor extraído
* Descrição do campo
* Lista de outros campos (para evitar contaminação)

**Processo - Loop de Validação (até N tentativas):**

```python
for tentativa in range(max_tentativas):
    # 1. Gera regra usando LLM
    rule_response = agent_rule.invoke({
        "messages": [{
            "role": "user", 
            "content": RULE_GENERATION_PROMPT + feedback_anterior
        }]
    })
    
    # 2. Valida sintaxe da regra
    rule = Rule.model_validate(rule_response)
    
    # 3. Testa se regra extrai valor correto
    valor_extraido = execute_rule(rule, texto)
    if valor_extraido != valor_esperado:
        feedback = "Regra extraiu valor errado..."
        continue  # Tenta novamente com feedback
    
    # 4. Valida formato com validation_regex
    if not re.match(rule.validation_regex, valor_esperado):
        feedback = "Validation_regex não valida o valor..."
        continue
    
    # 5. Verifica contaminação de keywords
    if contains_other_keywords(valor_extraido, outros_campos):
        feedback = "Valor contém keywords de outros campos..."
        continue
    
    # ✓ Todas validações passaram!
    return rule
```

**Validações Aplicadas:**

1. **Sintaxe JSON** - verifica estrutura da resposta
2. **Extração correta** - regra deve extrair valor esperado do texto
3. **Validação de formato** - `validation_regex` deve aceitar o valor
4. **Sem contaminação** - valor não pode conter keywords de outros campos

**Feedback Adaptativo:**

A cada falha, o sistema:
* **Registra o erro** com detalhes específicos
* **Adiciona ao histórico de feedback**
* **Reinvoca a LLM** com o feedback acumulado
* Processo continua até **sucesso** ou **limite de tentativas**

**Saída:**
* **Regra validada** → Adicionada ao cache
* **Falha após N tentativas** → Campo não gera regra (mas valor extraído é usado)

---

### Salvamento e Persistência

**Cache:**
* Salvo em disco após **cada regra gerada** (salvamento incremental)
* Formato JSON com estrutura de listas duplamente encadeadas
* Carregado no início do processamento
* Preserva aprendizado entre execuções

**Respostas:**
* Todas extrações salvas ao final do processamento
* Incluem valores esperados (ground truth) quando disponível
* Formato JSON com metadados (índice, label, acurácia)

---

### Métricas Rastreadas

Para cada documento processado:

* **Tempo de processamento** - tempo total (s)
* **Acurácia** - % de campos corretos vs. ground truth
* **Campos extraídos** - sucesso vs. falha
* **Tokens consumidos** - prompt + completion (LLM1 e LLM2)
* **Custo** - baseado em preços do modelo
* **Regras geradas** - novas regras adicionadas ao cache
* **Taxa de cache hit** - % de campos extraídos via fast path
* **Chamadas LLM** - Extractor (LLM1) e Rule Generator (LLM2)

**Integração WandB (opcional):**
* Log de métricas em tempo real
* Upload de cache e respostas finais
* Tracking de experimentos completo

---

### Exemplo de Execução Completa

**Documento 1 (Cache vazio):**

```
Texto: "Nome: João Silva\nInscricao: 123456\nCategoria: ADVOGADO"
Campos: ["nome", "inscricao", "categoria"]

┌─ ETAPA 1: Cache vazio
│  ✗ nome: Cache miss
│  ✗ inscricao: Cache miss  
│  ✗ categoria: Cache miss
│  → Todos campos para LLM
│
┌─ ETAPA 2: Extração LLM
│  ✓ Extraído: {"nome": "João Silva", "inscricao": "123456", "categoria": "ADVOGADO"}
│
┌─ ETAPA 3: Geração de Regras
│  ✓ Regra para "nome" gerada e validada
│  ✓ Regra para "inscricao" gerada e validada
│  ✓ Regra para "categoria" gerada e validada
│  → 3 regras adicionadas ao cache

Resultado: 3 campos extraídos | 3 regras novas | 1 chamada LLM1 | 3 chamadas LLM2
```

**Documento 2 (Cache com 3 regras):**

```
Texto: "Nome: Maria Santos\nInscricao: 654321\nCategoria\nEndereco: Rua ABC"
Campos: ["nome", "inscricao", "categoria"]

┌─ ETAPA 1: Usando cache
│  ✓ nome: "Maria Santos" (regra aplicada, peso++, repositioned)
│  ✓ inscricao: "654321" (regra aplicada, peso++, repositioned)
│  ✓ categoria: None (regra para campo vazio aplicada, peso++, repositioned)
│  → Todos campos extraídos!

Resultado: 3 campos extraídos | 0 regras novas | 0 chamadas LLM
```

**Documento 3 (Cache otimizado, campo novo):**

```
Texto: "Nome: Pedro Costa\nInscricao: 789012\nTelefone: (11) 99999-9999"
Campos: ["nome", "inscricao", "telefone"]

┌─ ETAPA 1: Usando cache
│  ✓ nome: "Pedro Costa" (peso++, já no topo da lista)
│  ✓ inscricao: "789012" (peso++, já no topo da lista)
│  ✗ telefone: Cache miss (campo novo)
│
┌─ ETAPA 2: Extração LLM (só telefone)
│  ✓ Extraído: {"telefone": "(11) 99999-9999"}
│
┌─ ETAPA 3: Geração de Regra
│  ✓ Regra para "telefone" gerada e validada
│  → 1 regra nova adicionada ao cache

Resultado: 3 campos extraídos | 1 regra nova | 1 chamada LLM1 | 1 chamada LLM2
```

---

### Observações Importantes

**Normalização de Texto:**

Antes de qualquer processamento, o texto é normalizado:
* Separação de letras/números concatenados (`"Nome123"` → `"Nome 123"`)
* Separação de palavras concatenadas (`"NomeInscricao"` → `"Nome Inscricao"`)
* Colapso de múltiplos espaços/tabs → espaço único
* Colapso de múltiplas quebras de linha → quebra única
* Remoção de espaços em branco nas extremidades

**Modo LLM-Only:**

O sistema pode ser executado **sem cache** (`--no-use-cache`):
* Pula ETAPA 1 (fast path)
* Todos campos vão direto para ETAPA 2 (LLM)
* **Não gera regras** (pula ETAPA 3)
* Útil para baseline de comparação de desempenho

**Adaptação Progressiva:**

A cada documento processado, o cache:
1. **Aprende** novas regras para campos inéditos
2. **Otimiza** a ordem de regras existentes por frequência de uso
3. **Acelera** extrações futuras em documentos similares
4. **Reduz custo** ao diminuir chamadas à LLM

---
