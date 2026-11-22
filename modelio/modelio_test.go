package modelio_test

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/modelio"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func TestSaveLoadLinearRegression(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}})
	if err != nil {
		t.Fatal(err)
	}
	model := linear_model.NewLinearRegression()
	if err := model.Fit(x, []float64{1, 3, 5, 7}); err != nil {
		t.Fatal(err)
	}
	var artifact bytes.Buffer
	if err := modelio.Save(&artifact, model); err != nil {
		t.Fatal(err)
	}
	loaded, err := modelio.LoadRegressor(&artifact)
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{1, 3, 5, 7} {
		if got[i]-want > 1e-12 || want-got[i] > 1e-12 {
			t.Fatalf("prediction[%d] = %g, want %g", i, got[i], want)
		}
	}
}

func TestSaveLoadPipelineAndFixture(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}})
	if err != nil {
		t.Fatal(err)
	}
	newPipeline := func() *pipeline.RegressionPipeline {
		model, err := pipeline.NewRegressionPipeline(
			[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
			func() pipeline.Regressor { return linear_model.NewLinearRegression() },
		)
		if err != nil {
			t.Fatal(err)
		}
		return model
	}
	model := newPipeline()
	if err := model.Fit(x, []float64{1, 3, 5, 7}); err != nil {
		t.Fatal(err)
	}
	var artifact bytes.Buffer
	if err := modelio.Save(&artifact, model); err != nil {
		t.Fatal(err)
	}
	loaded, err := modelio.LoadRegressor(&artifact)
	if err != nil {
		t.Fatal(err)
	}
	checkPredictions(t, loaded, x, []float64{1, 3, 5, 7})

	fixture, err := os.Open("testdata/pipeline-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	defer fixture.Close()
	fromFixture, err := modelio.LoadRegressor(fixture)
	if err != nil {
		t.Fatal(err)
	}
	checkPredictions(t, fromFixture, x, []float64{1, 3, 5, 7})
}

func TestLoadRegressorRejectsInvalidArtifacts(t *testing.T) {
	tests := []struct {
		name string
		json string
	}{
		{name: "truncated", json: `{"format":"go-learn"`},
		{name: "unknown version", json: `{"format":"go-learn","version":2,"kind":"linear_regression"}`},
		{name: "unknown field", json: `{"format":"go-learn","version":1,"kind":"linear_regression","other":true}`},
		{name: "bad scale", json: `{"format":"go-learn","version":1,"kind":"regression_pipeline","n_features":1,"steps":[{"name":"scale","type":"standard_scaler","with_mean":true,"with_std":true,"mean":[0],"variance":[1],"scale":[0]}],"regressor":{"type":"linear_regression","fit_intercept":true,"coefficients":[1],"intercept":0}}`},
		{name: "unsupported step", json: `{"format":"go-learn","version":1,"kind":"regression_pipeline","n_features":1,"steps":[{"name":"scale","type":"minmax"}],"regressor":{"type":"linear_regression","fit_intercept":true,"coefficients":[1],"intercept":0}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := modelio.LoadRegressor(strings.NewReader(tt.json)); err == nil {
				t.Fatal("LoadRegressor succeeded for invalid artifact")
			}
		})
	}

	if err := modelio.Save(&bytes.Buffer{}, linear_model.NewLinearRegression()); err == nil {
		t.Fatal("Save succeeded for unfitted estimator")
	}
}

func checkPredictions(t *testing.T, model pipeline.Regressor, x *matrix.Dense, want []float64) {
	t.Helper()
	got, err := model.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i]-want[i] > 1e-12 || want[i]-got[i] > 1e-12 {
			t.Fatalf("prediction[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}
