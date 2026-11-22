package linear_model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"go-learn/matrix"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func TestLogisticReference(t *testing.T) {
	var fixture struct {
		X, Query [][]float64
		Y        []int
		Cases    []struct {
			Scale         bool
			Objective     float64
			FitIntercept  bool `json:"fit_intercept"`
			Coef          []float64
			Intercept     float64
			Probabilities [][]float64
			Predictions   []int
		}
	}
	data, err := os.ReadFile("../testdata/scikit-learn-1.9.1-classification.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, tc := range fixture.Cases {
		name := "raw"
		if tc.Scale {
			name = "scaled"
		}
		if tc.FitIntercept {
			name += "_intercept"
		}
		t.Run(name, func(t *testing.T) {
			x, _ := matrix.NewDenseFromRows(fixture.X)
			query, _ := matrix.NewDenseFromRows(fixture.Query)
			newModel := func() *LogisticRegression {
				return NewLogisticRegression(WithTolerance(1e-10), WithLogisticFitIntercept(tc.FitIntercept))
			}
			model := newModel()
			if tc.Scale {
				scaler := preprocessing.NewStandardScaler()
				if err := scaler.Fit(x); err != nil {
					t.Fatal(err)
				}
				x, _ = scaler.Transform(x)
				query, _ = scaler.Transform(query)
			}
			if err := model.Fit(x, fixture.Y); err != nil {
				t.Fatal(err)
			}
			theta := model.Coef()
			if tc.FitIntercept {
				theta = append(theta, model.Intercept())
			}
			objective, _, _ := model.objective(x, fixture.Y, 9, theta, false)
			if math.Abs(objective-tc.Objective) > 1e-10 {
				t.Fatalf("objective %g vs %g", objective, tc.Objective)
			}
			for i, c := range model.Coef() {
				if math.Abs(c-tc.Coef[i]) > 1e-7 {
					t.Fatalf("coefficient %d: %g vs %g", i, c, tc.Coef[i])
				}
			}
			if math.Abs(model.Intercept()-tc.Intercept) > 1e-7 {
				t.Fatal("intercept mismatch")
			}
			proba, err := model.PredictProba(query)
			if err != nil {
				t.Fatal(err)
			}
			for i, row := range tc.Probabilities {
				for j, want := range row {
					if math.Abs(proba.At(i, j)-want) > 1e-7 {
						t.Fatal("probability mismatch")
					}
				}
			}
			pred, _ := model.Predict(query)
			if !reflect.DeepEqual(pred, tc.Predictions) {
				t.Fatal(pred)
			}
			if tc.Scale {
				steps := []pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}}
				p, err := pipeline.NewClassificationPipeline(steps, func() pipeline.Classifier { return newModel() })
				if err != nil {
					t.Fatal(err)
				}
				raw, _ := matrix.NewDenseFromRows(fixture.X)
				q, _ := matrix.NewDenseFromRows(fixture.Query)
				if err = p.Fit(raw, fixture.Y); err != nil {
					t.Fatal(err)
				}
				got, err := p.PredictProba(q)
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range got.Values() {
					if math.Abs(v-proba.Values()[i]) > 1e-12 {
						t.Fatal("pipeline mismatch")
					}
				}
				if err = p.Fit(raw, []int{1}); err == nil {
					t.Fatal("expected fit error")
				}
				after, _ := p.Predict(q)
				if !reflect.DeepEqual(after, pred) {
					t.Fatal("failed fit changed pipeline")
				}
			}
		})
	}
}

func TestLogisticContracts(t *testing.T) {
	x, _ := matrix.NewDenseFromRows([][]float64{{-2}, {-1}, {1}, {2}})
	y := []int{-3, -3, 8, 8}
	m := NewLogisticRegression(WithLogisticFitIntercept(false))
	if _, err := m.Predict(x); !errors.Is(err, ErrNotFitted) {
		t.Fatal(err)
	}
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	score, err := m.Score(x, y)
	if err != nil || score != 1 {
		t.Fatal(score, err)
	}
	zero, _ := matrix.NewDenseFromRows([][]float64{{0}, {1e300}, {-1e300}})
	pred, err := m.Predict(zero)
	if err != nil || !reflect.DeepEqual(pred, []int{-3, 8, -3}) {
		t.Fatal(pred, err)
	}
	proba, err := m.PredictProba(zero)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(proba.At(i, 0)+proba.At(i, 1)-1) > 1e-15 {
			t.Fatal("invalid probabilities")
		}
	}
	coef := m.Coef()
	classes := m.Classes()
	coef[0] = 999
	classes[0] = 999
	if m.Coef()[0] == 999 || m.Classes()[0] == 999 {
		t.Fatal("aliased accessors")
	}
	before := m.Coef()
	m.maxIterations = 1
	m.tolerance = 1e-16
	if err := m.Fit(x, y); !errors.Is(err, ErrConvergence) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, m.Coef()) {
		t.Fatal("failed fit changed model")
	}
	for _, tc := range []struct {
		name    string
		options []LogisticRegressionOption
	}{
		{"C", []LogisticRegressionOption{WithC(0)}}, {"tol", []LogisticRegressionOption{WithTolerance(math.NaN())}}, {"iterations", []LogisticRegressionOption{WithMaxIterations(0)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := NewLogisticRegression(tc.options...).Fit(x, y); err == nil {
				t.Fatal("invalid option accepted")
			}
		})
	}
	if err := NewLogisticRegression().Fit(x, []int{1, 1, 1, 1}); err == nil {
		t.Fatal("single class accepted")
	}
}

// Finite differences constrain the analytic derivatives independently of the solver.
func TestLogisticDerivatives(t *testing.T) {
	x, _ := matrix.NewDenseFromRows([][]float64{{-2, 1}, {1, 3}, {0, -1}})
	y := []int{0, 5, 0}
	m := NewLogisticRegression(WithC(0.7))
	theta := []float64{0.3, -0.7, 0.2}
	_, gradient, hessian := m.objective(x, y, 5, theta, true)
	const epsilon = 1e-5
	for j := range theta {
		plus := append([]float64(nil), theta...)
		minus := append([]float64(nil), theta...)
		plus[j] += epsilon
		minus[j] -= epsilon
		lp, gp, _ := m.objective(x, y, 5, plus, true)
		lm, gm, _ := m.objective(x, y, 5, minus, true)
		if math.Abs((lp-lm)/(2*epsilon)-gradient[j]) > 1e-8 {
			t.Fatal("gradient mismatch")
		}
		for k := range theta {
			if math.Abs((gp[k]-gm[k])/(2*epsilon)-hessian[k*len(theta)+j]) > 1e-8 {
				t.Fatal("Hessian mismatch")
			}
		}
	}
}

func TestLogisticInvalidInputs(t *testing.T) {
	valid, _ := matrix.NewDenseFromRows([][]float64{{0}, {1}})
	bad, _ := matrix.NewDenseFromRows([][]float64{{math.Inf(1)}, {1}})
	for _, tc := range []struct {
		name string
		x    *matrix.Dense
		y    []int
	}{
		{"nil", nil, []int{0, 1}}, {"length", valid, []int{0}}, {"nonfinite", bad, []int{0, 1}}, {"oneclass", valid, []int{1, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := NewLogisticRegression().Fit(tc.x, tc.y); err == nil {
				t.Fatal("invalid input accepted")
			}
		})
	}
	x, _ := matrix.NewDenseFromRows([][]float64{{0, 0, 0}, {1, 1, 1}})
	if err := NewLogisticRegression().Fit(x, []int{0, 1}); err != nil {
		t.Fatal("regularized wide data:", err)
	}
}

func ExampleLogisticRegression() {
	x, _ := matrix.NewDenseFromRows([][]float64{{-2}, {-1}, {1}, {2}})
	model := NewLogisticRegression(WithLogisticFitIntercept(false))
	if err := model.Fit(x, []int{0, 0, 1, 1}); err != nil {
		panic(err)
	}
	predictions, err := model.Predict(x)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)
	// Output: [0 0 1 1]
}
