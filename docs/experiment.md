### Experimentos Realizados

Seis experimentos foram conduzidos para avaliar a eficácia da pipeline.

~~Acesse todos os resultados no relatório de experimentos do Weights & Biases~~ — *to come*

Todos os datasets foram gerados sinteticamente utilizando o script [`generate_fake_data.py`](../scripts/generate_fake_data.py):

```bash
python3 -m scripts.generate_fake_data \
  --save-path data/fake \
  --dataset-filename X \
  --num-samples X000 \
  --seed 1
```

No primeiro experimento, utilizou-se o dataset `fake_dataset_1000samples_seed_1` tanto com otimização por cache quanto sem otimização por cache.

Em seguida, utilizando a cache gerada no experimento com otimização por cache, foram realizados testes em datasets maiores: `fake_dataset_2000samples_seed_1` e `fake_dataset_3000samples_seed_1`, mantendo a mesma seed.

Por fim, foram realizados experimentos com datasets diferentes — `fake_dataset_1000samples_seed_2` e `fake_dataset_1000samples_seed_3` — com o objetivo de avaliar a capacidade de generalização das regras armazenadas na cache do primeiro experimento.


Os detalhes de cada experimento são:

| Experimento | Nº de Samples | Seed | Otimização por Cache | Dataset | Resultados | Cache |
| ----------- | ------------- | ---- | -------------------- | ------- | ---------- | ----- |
| 1           | 1000          | 1    | ❌ Sem cache          | [dataset](../data/fake/datasets/fake_dataset_1000samples_seed_1.json) | *to come* | — |
| 2           | 1000          | 1    | ✅ Com cache          | [dataset](../data/fake/datasets/fake_dataset_1000samples_seed_1.json) | [results](../data/fake/results/fake_dataset_1000samples_seed_1_with_cache_result.json) | [cache](../data/fake/caches/fake_dataset_1000samples_seed_1_cache.json) |
| 3           | 2000          | 1    | ✅ Com cache          | [dataset](../data/fake/datasets/fake_dataset_2000samples_seed_1.json) | [results](../data/fake/results/fake_dataset_2000samples_seed_1_with_cache_result.json) | [cache](../data/fake/caches/fake_dataset_1000samples_seed_1_cache.json) |
| 4           | 3000          | 1    | ✅ Com cache          | [dataset](../data/fake/datasets/fake_dataset_3000samples_seed_1.json) | [results](../data/fake/results/fake_dataset_3000samples_seed_1_with_cache_result.json) | [cache](../data/fake/caches/fake_dataset_1000samples_seed_1_cache.json) |
| 5           | 1000          | 2    | ✅ Com cache          | [dataset](../data/fake/datasets/fake_dataset_1000samples_seed_2.json) | [results](../data/fake/results/fake_dataset_1000samples_seed_2_with_cache_result.json) | [cache](../data/fake/caches/fake_dataset_1000samples_seed_1_cache.json) |
| 6           | 1000          | 3    | ✅ Com cache          | [dataset](../data/fake/datasets/fake_dataset_1000samples_seed_3.json) | [results](../data/fake/results/fake_dataset_1000samples_seed_3_with_cache_result.json) | [cache](../data/fake/caches/fake_dataset_1000samples_seed_1_cache.json) |
