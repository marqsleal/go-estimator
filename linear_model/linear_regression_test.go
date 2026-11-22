package linear_model_test

import (
	"errors"
	"math"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
)

func TestLinearRegressionFitPredict(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}})
	if err != nil {
		t.Fatal(err)
	}
	model := linear_model.NewLinearRegression()
	if err := model.Fit(x, []float64{1, 3, 5, 7}); err != nil {
		t.Fatal(err)
	}
	prediction, err := model.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(model.Coef()[0]-2) > 1e-12 || math.Abs(model.Intercept()-1) > 1e-12 {
		t.Fatalf("coef=%v intercept=%g", model.Coef(), model.Intercept())
	}
	for i, want := range []float64{1, 3, 5, 7} {
		if math.Abs(prediction[i]-want) > 1e-12 {
			t.Fatalf("prediction[%d]=%g want %g", i, prediction[i], want)
		}
	}
}

func TestLinearRegressionFitInterceptDisabled(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{1}, {2}, {3}})
	if err != nil {
		t.Fatal(err)
	}
	model := linear_model.NewLinearRegression(linear_model.WithFitIntercept(false))
	if err := model.Fit(x, []float64{2, 4, 6}); err != nil {
		t.Fatal(err)
	}
	if math.Abs(model.Coef()[0]-2) > 1e-12 || model.Intercept() != 0 {
		t.Fatalf("coef=%v intercept=%g", model.Coef(), model.Intercept())
	}
}

func TestLinearRegressionErrorsPreservePreviousFit(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}})
	if err != nil {
		t.Fatal(err)
	}
	model := linear_model.NewLinearRegression()
	if err := model.Fit(x, []float64{1, 3, 5}); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Predict(x); err != nil {
		t.Fatal(err)
	}
	if err := model.Fit(x, []float64{1}); err == nil {
		t.Fatal("Fit with mismatched target length succeeded")
	}
	if got := model.Coef()[0]; math.Abs(got-2) > 1e-12 {
		t.Fatalf("failed Fit changed coefficients: %g", got)
	}
	duplicate, err := matrix.NewDenseFromRows([][]float64{{1, 1}, {2, 2}, {3, 3}})
	if err != nil {
		t.Fatal(err)
	}
	if err := model.Fit(duplicate, []float64{1, 2, 3}); !errors.Is(err, linear_model.ErrRankDeficient) {
		t.Fatalf("rank deficient fit error = %v, want ErrRankDeficient", err)
	}
}
