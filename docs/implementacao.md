# Implementação atual

Este documento descreve o que existe no código hoje. A compatibilidade é por recurso e pela versão de referência indicada, não uma promessa de reproduzir toda a API do scikit-learn.

## Escopo entregue

O primeiro fluxo cobre matrizes densas `float64`, divisão treino/teste, padronização, regressão de alvo único, métricas de regressão, pipeline e persistência para inferência. O código de produção, os testes Go e os benchmarks Go usam apenas a biblioteca padrão. Python e scikit-learn foram usados externamente para conferir a referência e criar dados estáticos; o projeto não os executa nem distribui.

O módulo declara Go 1.22 como mínimo. A implementação foi validada nesta máquina com Go 1.26.1; a execução em Go 1.22 ainda precisa ser confirmada. O caminho `go-learn` em `go.mod` é local de desenvolvimento, não o caminho público de importação. Licença e caminho público devem ser definidos antes da publicação.

## Pacotes e contratos

### `matrix`

`Dense` armazena valores `float64` contíguos em ordem row-major. `NewDense` e `NewDenseFromRows` copiam a entrada. `Values`, `Row` e `Clone` também devolvem cópias. `At` faz panic para índices fora da forma; `Set` e `Row` devolvem erro. A matriz aceita dimensões zero, mas os estimadores exigem pelo menos uma amostra e uma característica.

### `preprocessing`

`StandardScaler` oferece `Fit`, `Transform`, `FitTransform`, `Mean`, `Var` e `Scale`. Centralização e escala têm padrão `true`; a variância usa divisor `n`. Colunas constantes usam escala 1. Transformações devolvem uma matriz nova e não alteram a entrada. A implementação rejeita valores não finitos e preserva o ajuste anterior se `Fit` falhar.

### `model_selection`

`TrainTestSplit` mantém alinhados X, y e índices. `test_size` padrão é 0,25; o tamanho do conjunto de teste arredonda para cima. `shuffle` padrão é verdadeiro. `WithRandomState` aceita semente inteira de 0 a 2³²−1, incluindo zero. Cada chamada usa um gerador local e é reprodutível dentro da implementação Go; a sequência não é compatível com NumPy. Sem semente, o código obtém entropia do sistema.

`KFold` retorna índices de treino e teste, distribui as amostras restantes nos primeiros folds e permite embaralhamento reprodutível. `CrossValScore` calcula R² por fold para fábricas de pipelines de regressão. `CrossValClassificationScore` calcula acurácia para fábricas de pipelines de classificação. Cada fold recebe instâncias novas e ajusta transformadores somente nos dados de treino.

### `metrics`

As funções `MeanAbsoluteError`, `MeanSquaredError` e `R2Score` recebem `yTrue` antes de `yPred`, validam comprimentos e finitude e retornam `(valor, error)`. O R² de uma amostra é NaN. Para alvo constante, segue o comportamento finito padrão do scikit-learn: 1 para predição perfeita e 0 para predição imperfeita.

### `linear_model`

`LinearRegression` implementa `Fit`, `Predict` e `Score`, com `fit_intercept=true` por padrão, alvo único e acesso por cópia aos coeficientes. O ajuste usa QR de Householder com pivoteamento de colunas, implementado em `internal/linalg`; não forma `XᵀX`. O limiar relativo de posto é `1e-12`. Posto deficiente retorna `ErrRankDeficient`, em vez de produzir a solução de norma mínima por SVD do scikit-learn. Um ajuste inválido preserva o último modelo válido.

### `pipeline`

`RegressionPipeline` compõe transformadores intermediários e um regressor. Cada `Fit` cria instâncias novas a partir das fábricas, treina as etapas em sequência e só publica o estado quando todas concluem. Erros incluem o nome da etapa. O contrato implementado é de regressão; a composição é restrita às interfaces Go declaradas no pacote.

### `modelio`

`Save(io.Writer, model)` e `LoadRegressor(io.Reader)` usam JSON próprio versionado (`format: go-learn`, `version: 1`). O formato aceita `LinearRegression` isolada ou pipeline com etapas `StandardScaler` e regressão linear. O carregamento valida formato, tipos, dimensões, finitude e escalas antes de retornar um preditor. Não há importação de pickle ou joblib.

O chamador controla arquivos e substituições. Para substituir um artefato em disco com segurança, escreva em um temporário no mesmo diretório, verifique escrita e fechamento e renomeie depois. O pacote trabalha com readers e writers e não escolhe caminhos de arquivo.

## Exemplo: treinar, salvar e carregar para inferência

```go
var artifact bytes.Buffer
if err := modelio.Save(&artifact, model); err != nil {
	return err
}

loaded, err := modelio.LoadRegressor(&artifact)
if err != nil {
	return err
}
predictions, err := loaded.Predict(xTest)
if err != nil {
	return err
}
```

O artefato serializa os parâmetros ajustados. A semente e os dados de treino não são necessários para predição.

## Exemplos executáveis

Os fluxos completos de regressão, classificação binária e validação cruzada estão em [`examples/`](../examples/README.md). Esse catálogo deve ser revisado e iterado a cada feature, conforme os [critérios de validação](specs/02-testes-validacao.md#exemplos-de-uso).

## Validação e comparação

- `go test ./...` cobre contratos públicos, erros, cálculo manual, atomicidade de ajuste, integração do pipeline, persistência e leitura de fixture.
- `go vet ./...` e `go test -race ./...` também passam no ambiente Go 1.26.1.
- [Fixture de compatibilidade](../testdata/scikit-learn-1.9.1-regression.json) registra Python 3.11.17, scikit-learn 1.9.1, NumPy 2.4.6 e SciPy 1.17.1. Ele compara split sem shuffle, estatísticas, coeficientes, intercepto e predições.
- Os benchmarks em [`benchmarks/`](../benchmarks/README.md) medem ajuste, predição, scaler, métricas e pipeline. A [baseline inicial](../benchmarks/results/initial-baseline.md) mostra QR como gargalo na carga `10.000 × 32`; os números são específicos ao host e não são uma meta de desempenho.

## Fora do escopo atual

Regressão com múltiplas saídas ou pesos, classificação multiclasse, persistência de classificadores, estratificação em folds, dados esparsos, imputação, NaN em estimadores, otimização por BLAS, importação de formatos Python e API pública de busca/tuning ainda não estão implementados. As specs em [`docs/specs/`](README.md) definem a sequência proposta para essas extensões.


## Classificação binária

`linear_model.NewLogisticRegression` aceita `WithC`, `WithLogisticFitIntercept`, `WithTolerance` e `WithMaxIterations`. Os defaults são C=1, intercepto habilitado, tolerância de gradiente 1e-6 e até 100 passos Newton. `Fit` recebe `[]int` com duas classes. `Classes` devolve os rótulos ordenados e `PredictProba` devolve duas colunas nessa ordem. `Score` calcula acurácia. Erros de convergência preservam o ajuste anterior e permitem `errors.Is(err, linear_model.ErrConvergence)`.

O solver usa Hessiana densa e Cholesky com busca Armijo. A regularização L2 penaliza os coeficientes e o intercepto é livre. O custo por iteração é quadrático no número de características para montar a Hessiana e cúbico para fatorá-la. A tolerância de gradiente e o solver seguem a spec própria.

`metrics` oferece `AccuracyScore`, `PrecisionScore`, `RecallScore` e `ConfusionMatrix`. Precision e recall recebem o rótulo positivo como terceiro argumento. A matriz retorna contagens e os rótulos ordenados dos eixos.

`pipeline.NewClassificationPipeline` recebe as mesmas etapas transformadoras da regressão e uma fábrica `func() pipeline.Classifier`. Oferece `Fit`, `Predict`, `PredictProba`, `Classes`, `Score` e `Fitted`. Cada ajuste usa instâncias novas e publica o estado após concluir todas as etapas. O exemplo executável em `pipeline/classification_test.go` mostra treino e predição em dados distintos.

A fixture estática de classificação usa o scikit-learn 1.9.1 local como referência externa. A suíte Go lê o JSON. Os testes cobrem derivadas numéricas, integração com scaler, falha de ajuste e dados inválidos.
