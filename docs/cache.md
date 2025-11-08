### Sistema de Cache Adaptativo

Para cada tipo de documento (ex.: **carteira_oab**, **tela_sistema**, **identidade**, etc.), existe uma estrutura de **Cache** ([`cache.py`](../src/cache.py)) que armazena, para cada *field* daquele tipo (por exemplo, para *carteira_oab*: **nome**, **inscrição**, **subseção**, **seccional**, etc.), uma **lista duplamente encadeada** que funciona como uma **LRU Cache** (*Least Recently Used Cache*).

Cada nó dessa lista contém uma estrutura de **Regra** ([`rule.py`](../src/rule.py)), que representa a regra de extração aprendida e gerada pela *Rule Generator LLM*.

Sempre que uma regra é utilizada com sucesso para extrair um campo no *fast path*, o seu **peso é incrementado em 1**, e a posição da regra na lista é atualizada de forma que as regras com maior peso fiquem mais próximas do início, acelerando futuras buscas.

Quando uma nova regra é gerada no *slow path* para um determinado campo, ela é adicionada **ao final da lista** correspondente, com **peso inicial igual a 1**.

Essa estrutura de cache é **persistida em disco** em formato **JSON**, sendo carregada no início do processamento dos documentos. Ao término do processamento, a cache atualizada é novamente **salva em disco**, preservando o aprendizado incremental das regras.

---

### Exemplo

Dicionário de caches para diferentes tipos de documentos:

```python
dict_caches = {
    "carteira_oab": Cache1,
    "tela_sistema": Cache2,
    "identidade": Cache3,
    ...
}
```

Para um documento do tipo `'carteira_oab'`, o `Cache1` é utilizado para extração rápida (*fast path*):

```python
Cache1 = {
    "nome": DoublyLinkedList1,       # Lista de regras para o campo 'nome'
    "inscrição": DoublyLinkedList2,  # Lista de regras para o campo 'inscrição'
    ...
}
```

Cada `DoublyLinkedList` armazena nós com regras de extração.
Exemplo de lista encadeada para o campo `'nome'`:

#### Após a geração de 3 novas regras

```
+-------------------+    +-------------------+    +-------------------+
|      Node1        |<-->|      Node2        |<-->|      Node3        |
|   RuleA | Peso1   |    |   RuleB | Peso1   |    |   RuleC | Peso1   |
+-------------------+    +-------------------+    +-------------------+
```

#### Após o uso bem-sucedido da **RuleC**

```
+-------------------+    +-------------------+    +-------------------+
|      Node3        |<-->|      Node2        |<-->|      Node1        |
|   RuleC | Peso2   |    |   RuleB | Peso1   |    |   RuleA | Peso1   |
+-------------------+    +-------------------+    +-------------------+
```

#### Após inserir uma nova regra **RuleD**

```
+-------------------+    +-------------------+    +-------------------+    +-------------------+
|      Node3        |<-->|      Node2        |<-->|      Node1        |<-->|      Node4        |
|   RuleC | Peso2   |    |   RuleB | Peso1   |    |   RuleA | Peso1   |    |   RuleD | Peso1   |
+-------------------+    +-------------------+    +-------------------+    +-------------------+
```
