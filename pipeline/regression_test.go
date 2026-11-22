package pipeline_test

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func TestRegressionPipelineFitPredict(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}})
	if err != nil {
		t.Fatal(err)
	}
	p, err := pipeline.NewRegressionPipeline(
		[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
		func() pipeline.Regressor { return linear_model.NewLinearRegression() },
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Predict(x); !errors.Is(err, pipeline.ErrNotFitted) {
		t.Fatalf("Predict before Fit error = %v", err)
	}
	if err := p.Fit(x, []float64{1, 3, 5, 7, 9, 11}); err != nil {
		t.Fatal(err)
	}
	got, err := p.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 3, 5, 7, 9, 11}
	for i := range got {
		if got[i]-want[i] > 1e-12 || want[i]-got[i] > 1e-12 {
			t.Fatalf("predictions = %v, want %v", got, want)
		}
	}
	steps := p.FittedTransformers()
	if len(steps) != 1 || steps[0].Name != "scale" {
		t.Fatalf("fitted steps = %#v", steps)
	}
}

func TestRegressionPipelineAtomicFitAndValidation(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}})
	if err != nil {
		t.Fatal(err)
	}
	factory := func() pipeline.Transformer { return preprocessing.NewStandardScaler() }
	if _, err := pipeline.NewRegressionPipeline([]pipeline.Step{{Name: "same", New: factory}, {Name: "same", New: factory}}, func() pipeline.Regressor { return linear_model.NewLinearRegression() }); err == nil {
		t.Fatal("duplicate step names were accepted")
	}
	p, err := pipeline.NewRegressionPipeline([]pipeline.Step{{Name: "scale", New: factory}}, func() pipeline.Regressor { return linear_model.NewLinearRegression() })
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Fit(x, []float64{1, 3, 5}); err != nil {
		t.Fatal(err)
	}
	want, err := p.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Fit(x, []float64{1}); err == nil {
		t.Fatal("Fit with invalid target length succeeded")
	}
	got, err := p.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("failed Fit changed predictions: got %v want %v", got, want)
	}
}

func ExampleRegressionPipeline() {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}})
	if err != nil {
		panic(err)
	}
	model, err := pipeline.NewRegressionPipeline(
		[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
		func() pipeline.Regressor { return linear_model.NewLinearRegression() },
	)
	if err != nil {
		panic(err)
	}
	if err := model.Fit(x, []float64{1, 3, 5, 7}); err != nil {
		panic(err)
	}
	predictions, err := model.Predict(x)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.0f %.0f %.0f %.0f\n", predictions[0], predictions[1], predictions[2], predictions[3])
	// Output: 1 3 5 7
}
