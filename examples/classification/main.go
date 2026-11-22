// Este exemplo ajusta um pipeline binário e consulta rótulos, probabilidades
// e métricas no conjunto de teste.
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
const randomState uint64 = 3

func main() {
	x, err := matrix.NewDenseFromRows([][]float64{{-4}, {-3}, {-2}, {-1}, {1}, {2}, {3}, {4}})
	if err != nil {
		log.Fatal(err)
	}
	y := []int{0, 0, 0, 0, 1, 1, 1, 1}
	// TrainTestSplit recebe alvos float64. Mantemos os rótulos inteiros originais
	// e usamos os índices retornados para reconstruir y de treino e teste.
	split, err := model_selection.TrainTestSplit(x, toFloatTargets(y),
		model_selection.WithTestSize(0.25), model_selection.WithRandomState(randomState))
	if err != nil {
		log.Fatal(err)
	}
	trainLabels, err := selectIntTargets(y, split.TrainIndices)
	if err != nil {
		log.Fatal(err)
	}
	testLabels, err := selectIntTargets(y, split.TestIndices)
	if err != nil {
		log.Fatal(err)
	}
	// O scaler e o classificador são ajustados apenas com a partição de treino.
	model, err := pipeline.NewClassificationPipeline(
		[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
		func() pipeline.Classifier { return linear_model.NewLogisticRegression() },
	)
	if err != nil {
		log.Fatal(err)
	}
	if err := model.Fit(split.XTrain, trainLabels); err != nil {
		log.Fatal(err)
	}
	predictions, err := model.Predict(split.XTest)
	if err != nil {
		log.Fatal(err)
	}
	probabilities, err := model.PredictProba(split.XTest)
	if err != nil {
		log.Fatal(err)
	}
	// As métricas comparam os rótulos de teste às previsões do modelo.
	accuracy, err := metrics.AccuracyScore(testLabels, predictions)
	if err != nil {
		log.Fatal(err)
	}
	precision, err := metrics.PrecisionScore(testLabels, predictions, 1)
	if err != nil {
		log.Fatal(err)
	}
	recall, err := metrics.RecallScore(testLabels, predictions, 1)
	if err != nil {
		log.Fatal(err)
	}
	confusion, labels, err := metrics.ConfusionMatrix(testLabels, predictions)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("classes=%v predictions=%v probabilities=%v\n", model.Classes(), predictions, probabilities.Values())
	fmt.Printf("accuracy=%.2f precision=%.2f recall=%.2f confusion(%v)=%v\n", accuracy, precision, recall, labels, confusion)
}

func toFloatTargets(values []int) []float64 {
	// A conversão existe só para obter a mesma permutação de índices do split.
	converted := make([]float64, len(values))
	for i, value := range values {
		converted[i] = float64(value)
	}
	return converted
}

func selectIntTargets(values []int, indices []int) ([]int, error) {
	// Seleciona os rótulos usando os índices aplicados às linhas de X.
	selected := make([]int, len(indices))
	for i, index := range indices {
		if index < 0 || index >= len(values) {
			return nil, fmt.Errorf("target index %d is out of range", index)
		}
		selected[i] = values[index]
	}
	return selected, nil
}
