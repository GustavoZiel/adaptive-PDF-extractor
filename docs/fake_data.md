### Geração de Dados Sintéticos para Testes

Para testar e avaliar o sistema de extração de dados, foi desenvolvido o script [`generate_fake_data.py`](../scripts/generate_fake_data.py), responsável por **simular textos de documentos** do tipo `"carteira_oab"`.

O script gera amostras contendo campos como `nome`, `inscricao`, `endereco_profissional`, `subsecao`, `seccional`, entre outros, incorporando **ruídos**, como:

* quebras de linha e espaços irregulares,
* variações de formatação,
* e **valores nulos** (*missing fields*) em alguns campos.

---

### Estrutura do Output

O script gera uma lista de dicionários no seguinte formato:

```python
[
  {
    "label": "carteira_oab",
    "extraction_schema": {
      "nome": "Nome do profissional, normalmente no canto superior esquerdo da imagem",
      "inscricao": "Número de inscrição do profissional",
      "seccional": "Seccional do profissional",
      "subsecao": "Subseção à qual o profissional faz parte",
      "categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA",
      "endereco_profissional": "Endereço do profissional",
      "telefone_profissional": "Telefone do profissional",
      "situacao": "Situação do profissional, normalmente no canto inferior direito."
    },
    "pdf_text": "Nome Benício da CunhaInscricao 176354025Seccional\tSubsecao\nCategoria Endereco ProfissionalLadeira Raul Pastor, 6, Ambrosina, 08063608 Teixeira / PBTelefone Profissional\n11 8353-3740Situacao\tSituação Regular",
    "expected_answer": {
      "nome": "Benício da Cunha",
      "inscricao": "176354025",
      "seccional": null,
      "subsecao": null,
      "categoria": null,
      "endereco_profissional": "Ladeira Raul Pastor, 6, Ambrosina, 08063608 Teixeira / PB",
      "telefone_profissional": "11 8353-3740",
      "situacao": "Situação Regular"
    }
  },
  ...
]
```

---

### Campos Principais

* **`pdf_text`** → Simula o texto extraído de um documento PDF, com ruídos e variações de formatação.
* **`expected_answer`** → Contém os valores *corretos* esperados para cada campo, utilizados na **validação e cálculo de acurácia** do sistema de extração.

---

### Exemplos de Textos Gerados

```
Nome Benício da CunhaInscricao 176354025Seccional       Subsecao
Categoria Endereco ProfissionalLadeira Raul Pastor, 6, Ambrosina, 08063608 Teixeira / PBTelefone Profissional
11 8353-3740Situacao    Situação Regular
```

```
Nome   Beatriz CirinoInscricao
415283607Seccional MASubsecao Categoria   ADVOGADOEndereco Profissional
Recanto da Cruz, 36, Marieta 3ª Seção, 17812-865 Siqueira de Alves / PRTelefone Profissional +55 (061) 4999 6228Situacao
Situação Irregular
```

```
Nome   Anthony Gabriel VargasInscricao 387425160Seccional   ROSubsecao
CategoriaADVOGADAEndereco ProfissionalFavela de Lima, 8, Santa Rosa, 98386-756 Abreu da Prata / PITelefone Profissional 0900-957-9032Situacao Situação Irregular
```

```
Situacao
Situação RegularCategoriaADVOGADATelefone Profissional +55 (031) 0896 5260Endereco Profissional
Subsecao        Nome   Seccional
BAInscricao
```

---

### Como Usar o Script

Execute o script diretamente pela linha de comando para gerar os dados sintéticos:

```bash
# Exibe todas as opções de configuração
python3 -m scripts.generate_fake_data --help

# Gera um conjunto de 1.000 documentos sintéticos
python3 -m scripts.generate_fake_data \
  --save-path data/fake \
  --dataset-filename dataset \
  --num-samples 1000 \
  --seed 1
```
