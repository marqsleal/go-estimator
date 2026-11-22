# pipeline

## Objetivo e dependências

Compor transformadores e estimador final em uma interface familiar a `sklearn.pipeline.Pipeline`. Depende de contratos concretos dos primeiros transformadores e estimadores; não antecipar uma hierarquia universal.

## Contrato

- Uma sequência nomeada de etapas, com transformadores intermediários e estimador final. `Fit` ajusta cada transformador com dados de treino, transforma esses dados e então ajusta o estimador; `Predict` aplica apenas `Transform` antes da predição.
- Definir como tipos diferentes de alvo e saída são representados em Go sem conversões inseguras. A primeira versão pode ser restrita a regressão de alvo único; classificação requer contrato próprio ou extensão verificada.
- Validar nomes duplicados, etapas vazias, incompatibilidade de dimensões e uso antes de ajuste. Propagar erros identificando a etapa. Documentar posse, reutilização, estado após falha e comportamento de ajuste repetido.
- Parâmetros aninhados, cache, passthrough, serialização e equivalência completa de `get_params`/`set_params` ficam fora do primeiro recorte até serem especificados.

## Testes e aceite

Comparar pipeline com aplicação manual de `StandardScaler` e `LinearRegression`, incluindo predições e parâmetros ajustados. Usar dados de teste com média diferente para detectar reajuste indevido; testar falha por etapa, treino repetido e dimensões inválidas. Comparar a saída com pipeline Python usando exatamente os mesmos dados e partições. Benchmark mede pipeline inteiro e etapas separadas.
