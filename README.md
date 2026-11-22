# go-learn

Biblioteca Go de aprendizado de máquina clássico com uma API inspirada no scikit-learn. A primeira versão cobre matrizes densas `float64`, `StandardScaler`, divisão treino/teste, métricas de regressão, `LinearRegression`, pipeline e persistência JSON para inferência de regressão. Também oferece classificação binária com `LogisticRegression`, métricas e pipeline de classificação. Produção e validação obrigatória dependem apenas da biblioteca padrão do Go.

Python e scikit-learn são referências externas para compatibilidade e fixtures. O projeto não incorpora nem executa Python.

## Exemplo

```go
x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}})
if err != nil {
	return err
}
y := []float64{1, 3, 5, 7, 9, 11}

split, err := model_selection.TrainTestSplit(x, y,
	model_selection.WithTestSize(0.25),
	model_selection.WithShuffle(false),
)
if err != nil {
	return err
}
model, err := pipeline.NewRegressionPipeline(
	[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer {
		return preprocessing.NewStandardScaler()
	}}},
	func() pipeline.Regressor { return linear_model.NewLinearRegression() },
)
if err != nil {
	return err
}
if err := model.Fit(split.XTrain, split.YTrain); err != nil {
	return err
}
predictions, err := model.Predict(split.XTest)
if err != nil {
	return err
}
mse, err := metrics.MeanSquaredError(split.YTest, predictions)
if err != nil {
	return err
}
fmt.Println(mse)
```

Imports usados no exemplo:

```go
import (
	"fmt"
	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/metrics"
	"go-learn/model_selection"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)
```

Consulte [a documentação da implementação](docs/implementacao.md), [as specs](docs/README.md) e a [matriz de compatibilidade](docs/compatibilidade.md) para contratos, limites, persistência e benchmarks.

## Verificação

```sh
go test ./...
go vet ./...
go test -race ./...
```

O módulo Go está como `go-learn` durante o desenvolvimento. O caminho público importável e a licença precisam ser definidos antes de uma versão publicada.
