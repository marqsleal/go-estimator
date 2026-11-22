// Este exemplo compara os fluxos de validação cruzada disponíveis. Cada
// fábrica cria um pipeline novo, e cada pipeline ajusta seu scaler só no fold
// de treino.
package main

import (
	"fmt"
	"log"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/model_selection"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func main() {
	xRegression, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}})
	if err != nil {
		log.Fatal(err)
	}
	yRegression := []float64{1, 3, 5, 7, 9, 11, 13, 15}
	seed := uint64(11)
	regressionScores, err := model_selection.CrossValScore(xRegression, yRegression, 4,
		func() pipeline.Regressor {
			model, err := pipeline.NewRegressionPipeline(
				[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
				func() pipeline.Regressor { return linear_model.NewLinearRegression() },
			)
			if err != nil {
				log.Fatal(err)
			}
			return model
		}, true, &seed)
	if err != nil {
		log.Fatal(err)
	}

	xClassification, err := matrix.NewDenseFromRows([][]float64{{-4}, {-3}, {-2}, {-1}, {1}, {2}, {3}, {4}})
	if err != nil {
		log.Fatal(err)
	}
	yClassification := []int{0, 0, 0, 0, 1, 1, 1, 1}
	classificationScores, err := model_selection.CrossValClassificationScore(xClassification, yClassification, 4,
		func() pipeline.Classifier {
			model, err := pipeline.NewClassificationPipeline(
				[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
				func() pipeline.Classifier { return linear_model.NewLogisticRegression() },
			)
			if err != nil {
				log.Fatal(err)
			}
			return model
		}, true, &seed)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("regression R² by fold: %v\n", regressionScores)
	fmt.Printf("classification accuracy by fold: %v\n", classificationScores)
}
