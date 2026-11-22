# pipeline

## Objetivo e dependências

Compor transformadores e estimador final em uma interface familiar a `sklearn.pipeline.Pipeline`. Depende de contratos concretos dos primeiros transformadores e estimadores; não antecipar uma hierarquia universal.

## Contrato

- Uma sequência nomeada de etapas, com transformadores intermediários e estimador final. `Fit` ajusta cada transformador com dados de treino, transforma esses dados e então ajusta o estimador; `Predict` aplica apenas `Transform` antes da predição.
- Regressão e classificação permanecem interfaces separadas, sem `any` nos métodos de ajuste ou previsão. A etapa seguinte à regressão adiciona `ClassificationPipeline`, `Classifier` e `ClassificationFactory` paralelos aos tipos de regressão existentes. O alvo é `[]int`; o estimador final é binário e retorna `[]int`. Transformadores continuam operando apenas sobre `*matrix.Dense`.
- Validar nomes duplicados, etapas vazias, incompatibilidade de dimensões e uso antes de ajuste. Propagar erros identificando a etapa. Documentar posse, reutilização, estado após falha e comportamento de ajuste repetido.
- Parâmetros aninhados, cache, passthrough, serialização e equivalência completa de `get_params`/`set_params` ficam fora do primeiro recorte até serem especificados.

## Testes e aceite

Comparar pipeline com aplicação manual de `StandardScaler` e `LinearRegression`, incluindo predições e parâmetros ajustados. Usar dados de teste com média diferente para detectar reajuste indevido; testar falha por etapa, treino repetido e dimensões inválidas. Comparar a saída com pipeline Python usando exatamente os mesmos dados e partições. Benchmark mede pipeline inteiro e etapas separadas.

Para `ClassificationPipeline`, repetir a comparação manual com `StandardScaler` e `LogisticRegression`; verificar que os transformadores são ajustados só no treino, `PredictProba` mantém ordem de classes, erros identificam a etapa e um `Fit` que falha preserva o pipeline ajustado anteriormente. Persistência de classificadores e de `ClassificationPipeline` fica fora deste incremento; requer extensão versionada da spec de persistência.
