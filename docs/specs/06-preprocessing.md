# preprocessing

## Objetivo e dependências

Implementar primeiro o equivalente a `sklearn.preprocessing.StandardScaler`, sobre a matriz da fundação. Expandir para outros transformadores apenas após o fluxo inicial funcionar.

## Contrato

- `Fit`, `Transform` e `FitTransform`; parâmetros equivalentes a `copy`, `with_mean` e `with_std`, com defaults da versão de referência. Preservar a distinção entre opção omitida e `false` explícito.
- Ajustar estatísticas por característica usando apenas amostras recebidas por `Fit`; `Transform` reutiliza estatísticas e valida número de colunas. Expor atributos ajustados equivalentes quando aplicável, sem permitir corromper estado interno.
- Seguir a convenção de variância e o tratamento de colunas constantes ou quase constantes da referência; verificar com fixtures. Documentar precisamente `copy=false`, propriedade da matriz e condições em que o compartilhamento é possível.
- Primeiro recorte: matrizes densas finitas e sem pesos. A referência pode aceitar NaN, pesos ou dados esparsos; registrar essas diferenças explicitamente até implementá-las.

## Testes e aceite

Casos com médias e variâncias conhecidas; coluna constante e quase constante; uma amostra; combinações de opções; `FitTransform` equivalente a `Fit` seguido de `Transform`; dados de teste com distribuição diferente para provar que não são reajustados; estado não ajustado, forma incorreta e não finitos. Comparar estatísticas e saídas com fixtures Python. Benchmark separa `Fit`, `Transform` e `FitTransform`, incluindo alocações.
