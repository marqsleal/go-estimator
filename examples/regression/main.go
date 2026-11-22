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

func main() {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}})
	if err != nil {
		log.Fatal(err)
	}
	y := []float64{1, 3, 5, 7, 9, 11, 13, 15}

	// A semente fixa torna a partição repetível nesta implementação Go.
	split, err := model_selection.TrainTestSplit(x, y,
		model_selection.WithTestSize(0.25), model_selection.WithRandomState(7))
	if err != nil {
		log.Fatal(err)
	}
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
