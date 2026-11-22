package linalg

import (
	"errors"
	"math"
	"testing"
)

func TestLeastSquares(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		rows int
		cols int
		b    []float64
		want []float64
		err  error
	}{
		{name: "overdetermined", a: []float64{1, 0, 1, 1, 1, 2}, rows: 3, cols: 2, b: []float64{1, 3, 5}, want: []float64{1, 2}},
		{name: "column pivoting", a: []float64{1e-5, 1, 0, 2, 0, 3}, rows: 3, cols: 2, b: []float64{3.00003, 6, 9}, want: []float64{3, 3}},
		{name: "rank deficient", a: []float64{1, 1, 2, 2, 3, 3}, rows: 3, cols: 2, b: []float64{1, 2, 3}, err: ErrRankDeficient},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LeastSquares(tt.a, tt.rows, tt.cols, tt.b)
			if !errors.Is(err, tt.err) {
				t.Fatalf("LeastSquares() error = %v, want %v", err, tt.err)
			}
			if err != nil {
				return
			}
			for i := range tt.want {
				if math.Abs(got[i]-tt.want[i]) > 1e-8 {
					t.Fatalf("coefficient[%d] = %.12g, want %.12g", i, got[i], tt.want[i])
				}
			}
		})
	}
}
