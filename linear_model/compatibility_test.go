package linear_model_test

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/metrics"
	"go-learn/model_selection"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func TestScikitLearn191RegressionFixture(t *testing.T) {
	type fixture struct {
		X            [][]float64 `json:"X"`
		Y            []float64   `json:"y"`
		TrainIndices []int       `json:"train_indices"`
		TestIndices  []int       `json:"test_indices"`
		Options      struct {
			TestSize float64 `json:"test_size"`
			Shuffle  bool    `json:"shuffle"`
		} `json:"options"`
		Scaler struct {
			Mean     []float64 `json:"mean"`
			Variance []float64 `json:"variance"`
			Scale    []float64 `json:"scale"`
		} `json:"scaler"`
		Regression struct {
			Coefficients []float64 `json:"coefficients"`
			Intercept    float64   `json:"intercept"`
		} `json:"regression"`
		Predictions []float64 `json:"predictions"`
	}

	data, err := os.ReadFile("../testdata/scikit-learn-1.9.1-regression.json")
	if err != nil {
		t.Fatal(err)
	}
	var expected fixture
	if err := json.Unmarshal(data, &expected); err != nil {
		t.Fatal(err)
	}
	x, err := matrix.NewDenseFromRows(expected.X)
	if err != nil {
		t.Fatal(err)
	}
	split, err := model_selection.TrainTestSplit(x, expected.Y,
		model_selection.WithTestSize(expected.Options.TestSize),
		model_selection.WithShuffle(expected.Options.Shuffle),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(split.TrainIndices, expected.TrainIndices) || !reflect.DeepEqual(split.TestIndices, expected.TestIndices) {
		t.Fatalf("split indices train=%v test=%v", split.TrainIndices, split.TestIndices)
	}
	model, err := pipeline.NewRegressionPipeline(
		[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
		func() pipeline.Regressor { return linear_model.NewLinearRegression() },
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.Fit(split.XTrain, split.YTrain); err != nil {
		t.Fatal(err)
	}
	steps := model.FittedTransformers()
	scaler := steps[0].Transformer.(*preprocessing.StandardScaler)
	assertClose(t, scaler.Mean(), expected.Scaler.Mean, 1e-12)
	assertClose(t, scaler.Var(), expected.Scaler.Variance, 1e-12)
	assertClose(t, scaler.Scale(), expected.Scaler.Scale, 1e-12)
	regressor := model.FittedRegressor().(*linear_model.LinearRegression)
	assertClose(t, regressor.Coef(), expected.Regression.Coefficients, 1e-12)
	if math.Abs(regressor.Intercept()-expected.Regression.Intercept) > 1e-12 {
		t.Fatalf("intercept = %.16g, want %.16g", regressor.Intercept(), expected.Regression.Intercept)
	}
	predictions, err := model.Predict(split.XTest)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, predictions, expected.Predictions, 1e-12)
	if _, err := metrics.MeanSquaredError(split.YTest, predictions); err != nil {
		t.Fatal(err)
	}
}

func assertClose(t *testing.T, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tolerance {
			t.Fatalf("value[%d] = %.16g, want %.16g (tolerance %g)", i, got[i], want[i], tolerance)
		}
	}
}
