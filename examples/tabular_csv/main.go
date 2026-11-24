// Demonstra carregar uma matriz numérica CSV e gravar previsões em CSV.
package main

import (
	"bytes"
	"fmt"
	"log"
	"strings"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/pipeline"
	"go-learn/tabular"
)

func main() {
	x, err := tabular.ReadCSV(strings.NewReader("0\n1\n2\n3\n"), tabular.CSVOptions{})
	if err != nil {
		log.Fatal(err)
	}
	y := []float64{1, 3, 5, 7}
	model, err := pipeline.NewRegressionPipeline(nil, func() pipeline.Regressor { return linear_model.NewLinearRegression() })
	if err != nil {
		log.Fatal(err)
	}
	if err := model.Fit(x, y); err != nil {
		log.Fatal(err)
	}
	predictions, err := model.Predict(x)
	if err != nil {
		log.Fatal(err)
	}
	output, err := matrix.NewDenseFromRows([][]float64{predictions})
	if err != nil {
		log.Fatal(err)
	}
	var csv bytes.Buffer
	if err := tabular.WriteCSV(&csv, output, ','); err != nil {
		log.Fatal(err)
	}
	fmt.Print(csv.String())
	var jsonOutput bytes.Buffer
	if err := tabular.WriteJSON(&jsonOutput, output); err != nil {
		log.Fatal(err)
	}
	fmt.Print(jsonOutput.String())
}
