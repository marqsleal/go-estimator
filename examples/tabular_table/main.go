// Demonstra leitura de tabela nomeada, seleção de colunas, treino e gravação.
package main

import (
	"bytes"
	"fmt"
	"log"
	"os"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/pipeline"
	"go-learn/tabular"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatal("usage: tabular_table input.csv predictions.csv")
	}
	input, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	table, err := tabular.ReadCSVTable(input, ',')
	closeErr := input.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		log.Fatal(err)
	}
	x, y, err := table.Split([]string{"area", "rooms"}, "price")
	if err != nil {
		log.Fatal(err)
	}
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
	output, err := matrix.NewDense(len(predictions), 1, predictions)
	if err != nil {
		log.Fatal(err)
	}
	var csv bytes.Buffer
	if err := tabular.WriteCSV(&csv, output, ','); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("predictions CSV: %s", csv.String())
	resultTable, err := tabular.NewTable([]string{"prediction"}, output)
	if err != nil {
		log.Fatal(err)
	}
	outputFile, err := os.Create(os.Args[2])
	if err != nil {
		log.Fatal(err)
	}
	writeErr := tabular.WriteCSVTable(outputFile, resultTable, ',')
	closeErr = outputFile.Close()
	if writeErr != nil {
		log.Fatal(writeErr)
	}
	if closeErr != nil {
		log.Fatal(closeErr)
	}
	var encoded bytes.Buffer
	if err := tabular.WriteJSONTable(&encoded, resultTable); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("predictions JSON: %s", encoded.String())
	var parquet bytes.Buffer
	if err := tabular.WriteParquet(&parquet, resultTable); err != nil {
		log.Fatal(err)
	}
	loaded, err := tabular.ReadParquet(bytes.NewReader(parquet.Bytes()))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Parquet round-trip columns: %v\n", loaded.Columns)
}
