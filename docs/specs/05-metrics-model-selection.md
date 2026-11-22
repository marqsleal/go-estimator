# metrics e model_selection

## Objetivo e dependências

Fornecer avaliação e divisão dos dados com nomes e semântica familiares ao scikit-learn. Depende da fundação; validação cruzada depende também de uma forma de criar estimadores novos.

## metrics

- Primeira entrega: `mean_absolute_error`, `mean_squared_error` e `r2_score`, com equivalentes Go exportados. Definir política de opções e valores padrão conforme a versão de referência; documentar limitações como pesos e múltiplas saídas.
- Classificação: `AccuracyScore(yTrue, yPred []int)`, `PrecisionScore(yTrue, yPred []int, positiveLabel int)`, `RecallScore(yTrue, yPred []int, positiveLabel int)` e `ConfusionMatrix(yTrue, yPred []int) ([][]int, []int, error)`. A matriz inclui as classes observadas em `yTrue` ou `yPred`, ordenadas em ordem crescente; linhas são classes verdadeiras e colunas classes previstas. Precision/recall exigem rótulo positivo explícito. Se o denominador for zero, retornam zero, equivalente a configurar `zero_division=0` explicitamente na referência. As três métricas são binárias quando usadas com `LogisticRegression`, enquanto a matriz aceita quaisquer rótulos inteiros observados.
- Todas as métricas de classificação rejeitam vetores vazios ou comprimentos diferentes. Precision e recall também rejeitam `positiveLabel` ausente de ambos os vetores, para evitar um resultado silencioso. `AccuracyScore` é proporção correta em `[0,1]`.
- Testar valores calculados à mão, vetores inválidos, dados não finitos, previsão perfeita, alvo constante e R² com uma amostra, incluindo comportamento explícito de `force_finite` se suportado. Para classificação, cobrir ordem dos rótulos, positivo explícito, denominadores zero e labels observadas só na predição. Confrontar casos equivalentes com fixtures Python.

## model_selection

- Primeira entrega: equivalente a `train_test_split` para matrizes e alvos, preservando alinhamento e ordem conforme as opções implementadas. Validar tamanhos e documentar arredondamento, padrão de embaralhamento, semente e limites.
- A sequência de embaralhamento Go pode diferir de NumPy mesmo com semente igual. Demonstrar reprodutibilidade dentro do Go; só declarar paridade exata de índices quando algoritmo e gerador coincidirem. Para testes comparativos de modelos usar índices pré-calculados comuns.
- Validação cruzada: `KFold(n, k, shuffle, randomState)` retorna índices por fold. Os primeiros `n % k` folds recebem uma amostra extra. A ordem é preservada sem embaralhamento; com embaralhamento, a semente é local e reproduzível no Go. `CrossValScore` usa uma fábrica de pipelines de regressão e retorna R² por fold. `CrossValClassificationScore` faz o mesmo para classificação e retorna acurácia. Cada fold cria uma instância independente e ajusta o pipeline somente nas linhas de treino. Estratificação e busca de hiperparâmetros ficam para specs posteriores.
- Testar partições sem sobreposição, cobertura, tamanhos, alinhamento, semente, casos-limite e ausência de vazamento. Comparar índices apenas nos modos cuja paridade seja explicitamente garantida.
