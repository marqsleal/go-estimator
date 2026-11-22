package metrics_test

import (
	"errors"
	"math"
	"testing"

	"go-learn/metrics"
)

func TestRegressionMetrics(t *testing.T) {
	tests := []struct {
		name string
		fn   func([]float64, []float64) (float64, error)
		true []float64
		pred []float64
		want float64
	}{
		{name: "mean absolute error", fn: metrics.MeanAbsoluteError, true: []float64{1, 3}, pred: []float64{2, 1}, want: 1.5},
		{name: "mean squared error", fn: metrics.MeanSquaredError, true: []float64{1, 3}, pred: []float64{2, 1}, want: 2.5},
		{name: "r2 perfect", fn: metrics.R2Score, true: []float64{1, 3}, pred: []float64{1, 3}, want: 1},
		{name: "r2 constant perfect", fn: metrics.R2Score, true: []float64{2, 2}, pred: []float64{2, 2}, want: 1},
		{name: "r2 constant imperfect", fn: metrics.R2Score, true: []float64{2, 2}, pred: []float64{1, 2}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.fn(tt.true, tt.pred)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(got-tt.want) > 1e-12 {
				t.Fatalf("metric = %g, want %g", got, tt.want)
			}
		})
	}
}

func TestRegressionMetricsInvalidAndSingleSample(t *testing.T) {
	if _, err := metrics.MeanSquaredError([]float64{1}, nil); !errors.Is(err, metrics.ErrInvalidTargets) {
		t.Fatalf("invalid length error = %v", err)
	}
	if _, err := metrics.MeanAbsoluteError([]float64{math.NaN()}, []float64{0}); !errors.Is(err, metrics.ErrInvalidTargets) {
		t.Fatalf("non-finite error = %v", err)
	}
	got, err := metrics.R2Score([]float64{1}, []float64{1})
	if err != nil || !math.IsNaN(got) {
		t.Fatalf("single sample R2 = %g, %v, want NaN", got, err)
	}
}
