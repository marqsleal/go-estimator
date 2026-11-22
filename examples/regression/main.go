// Este exemplo divide os dados antes de ajustar o scaler, evitando que as
// estatísticas do conjunto de teste influenciem o treinamento.
package main

import (
	"fmt"
	"log"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/metrics"
	"go-learn/model_selection"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

// randomState fixa a partição deste exemplo para que a execução seja repetível.
const randomState uint64 = 7

func main() {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}})
	if err != nil {
		log.Fatal(err)
	}
	y := []float64{1, 3, 5, 7, 9, 11, 13, 15}

	// A partição é feita antes do pipeline para manter os dados de teste fora do ajuste.
	split, err := model_selection.TrainTestSplit(x, y,
		model_selection.WithTestSize(0.25), model_selection.WithRandomState(randomState))
	if err != nil {
		log.Fatal(err)
	}
	// O scaler é ajustado dentro do Fit, portanto aprende estatísticas só do treino.
	model, err := pipeline.NewRegressionPipeline(
		[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
		func() pipeline.Regressor { return linear_model.NewLinearRegression() },
	)
	if err != nil {
		log.Fatal(err)
	}
	if err := model.Fit(split.XTrain, split.YTrain); err != nil {
		log.Fatal(err)
	}
	// O conjunto de teste é usado apenas para medir o modelo já ajustado.
	predictions, err := model.Predict(split.XTest)
	if err != nil {
		log.Fatal(err)
	}
	mse, err := metrics.MeanSquaredError(split.YTest, predictions)
	if err != nil {
		log.Fatal(err)
	}
	r2, err := metrics.R2Score(split.YTest, predictions)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("MSE=%.4f R²=%.4f\n", mse, r2)
}
