# metrics e model_selection

## Objetivo e dependências

Fornecer avaliação e divisão dos dados com nomes e semântica familiares ao scikit-learn. Depende da fundação; validação cruzada depende também de uma forma de criar estimadores novos.

## metrics

- Primeira entrega: `mean_absolute_error`, `mean_squared_error` e `r2_score`, com equivalentes Go exportados. Definir política de opções e valores padrão conforme a versão de referência; documentar limitações como pesos e múltiplas saídas.
- Classificação posterior: `accuracy_score`, `precision_score`, `recall_score` e `confusion_matrix` para rótulos binários, incluindo ordenação, classe positiva e divisões por zero.
- Testar valores calculados à mão, vetores inválidos, dados não finitos, previsão perfeita, alvo constante e R² com uma amostra, incluindo comportamento explícito de `force_finite` se suportado. Confrontar casos equivalentes com fixtures Python.

## model_selection

- Primeira entrega: equivalente a `train_test_split` para matrizes e alvos, preservando alinhamento e ordem conforme as opções implementadas. Validar tamanhos e documentar arredondamento, padrão de embaralhamento, semente e limites.
- A sequência de embaralhamento Go pode diferir de NumPy mesmo com semente igual. Demonstrar reprodutibilidade dentro do Go; só declarar paridade exata de índices quando algoritmo e gerador coincidirem. Para testes comparativos de modelos usar índices pré-calculados comuns.
- Posteriormente: `KFold` e `cross_val_score`, com instância não treinada independente por partição e pipeline ajustado somente no treino. Definir como reproduzir opções do estimador sem compartilhar estado; prevenir vazamento entre folds. Estratificação e busca de hiperparâmetros ficam para specs posteriores.
- Testar partições sem sobreposição, cobertura, tamanhos, alinhamento, semente, casos-limite e ausência de vazamento. Comparar índices apenas nos modos cuja paridade seja explicitamente garantida.
