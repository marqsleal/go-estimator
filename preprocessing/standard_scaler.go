// Package preprocessing contains data transformations used before estimation.
package preprocessing

import (
	"errors"
	"fmt"
	"math"

	"go-learn/matrix"
)

var (
	ErrNotFitted = errors.New("preprocessing: scaler is not fitted")
	ErrNonFinite = errors.New("preprocessing: input contains a non-finite value")
)

// ScalerOption configures a StandardScaler.
type ScalerOption func(*StandardScaler)

// WithMean controls feature centering. The default is true.
func WithMean(enabled bool) ScalerOption {
	return func(s *StandardScaler) { s.withMean = enabled }
}

// WithStd controls feature scaling to unit variance. The default is true.
func WithStd(enabled bool) ScalerOption {
	return func(s *StandardScaler) { s.withStd = enabled }
}

// StandardScaler centers features and scales them to unit variance.
// It is not safe to call Fit concurrently with other methods.
type StandardScaler struct {
	withMean bool
	withStd  bool
	fitted   bool
	mean     []float64
	variance []float64
	scale    []float64
}

// NewStandardScaler creates a scaler with centering and unit variance enabled.
func NewStandardScaler(options ...ScalerOption) *StandardScaler {
	s := &StandardScaler{withMean: true, withStd: true}
	for _, option := range options {
		if option != nil {
			option(s)
		}
	}
	return s
}

// Fit estimates per-feature statistics and preserves the previous state if it
// fails.
func (s *StandardScaler) Fit(x *matrix.Dense) error {
	if x == nil || x.Rows() == 0 || x.Cols() == 0 {
		return errors.New("preprocessing: Fit requires at least one row and one column")
	}
	rows, cols := x.Shape()
	mean := make([]float64, cols)
	variance := make([]float64, cols)
	scale := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := x.At(i, j)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("%w at (%d, %d)", ErrNonFinite, i, j)
			}
			mean[j] += value / float64(rows)
		}
	}
	for j, value := range mean {
		if !isFinite(value) {
			return fmt.Errorf("preprocessing: mean overflow for feature %d", j)
		}
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			delta := x.At(i, j) - mean[j]
			variance[j] += delta * delta / float64(rows)
		}
	}
	for j := range scale {
		if !isFinite(variance[j]) {
			return fmt.Errorf("preprocessing: variance overflow for feature %d", j)
		}
		scale[j] = math.Sqrt(variance[j])
		if scale[j] == 0 {
			scale[j] = 1
		}
	}
	if !s.withMean {
		mean = nil
	}
	if !s.withStd {
		variance = nil
		for j := range scale {
			scale[j] = 1
		}
	}
	s.mean, s.variance, s.scale, s.fitted = mean, variance, scale, true
	return nil
}

// Transform applies the fitted statistics and returns a new matrix.
func (s *StandardScaler) Transform(x *matrix.Dense) (*matrix.Dense, error) {
	if !s.fitted {
		return nil, ErrNotFitted
	}
	if x == nil || x.Cols() != len(s.scale) {
		return nil, fmt.Errorf("preprocessing: got %d features, want %d", columns(x), len(s.scale))
	}
	values := x.Values()
	for i := 0; i < x.Rows(); i++ {
		for j := 0; j < x.Cols(); j++ {
			value := values[i*x.Cols()+j]
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("%w at (%d, %d)", ErrNonFinite, i, j)
			}
			if s.withMean {
				value -= s.mean[j]
			}
			if s.withStd {
				value /= s.scale[j]
			}
			if !isFinite(value) {
				return nil, fmt.Errorf("preprocessing: transformed value is non-finite at (%d, %d)", i, j)
			}
			values[i*x.Cols()+j] = value
		}
	}
	return matrix.NewDense(x.Rows(), x.Cols(), values)
}

// FitTransform fits the scaler and transforms x.
func (s *StandardScaler) FitTransform(x *matrix.Dense) (*matrix.Dense, error) {
	if err := s.Fit(x); err != nil {
		return nil, err
	}
	return s.Transform(x)
}

// Mean returns a copy of the fitted feature means, or nil when centering is off.
func (s *StandardScaler) Mean() []float64 { return clone(s.mean) }

// Var returns a copy of the fitted population variances, or nil when scaling is off.
func (s *StandardScaler) Var() []float64 { return clone(s.variance) }

// Scale returns a copy of the divisors used by Transform.
func (s *StandardScaler) Scale() []float64 { return clone(s.scale) }

// Fitted reports whether Fit has completed successfully.
func (s *StandardScaler) Fitted() bool { return s.fitted }

// State returns the fitted scaler parameters and configuration as independent
// copies. It is intended for inspection and model persistence.
func (s *StandardScaler) State() (withMean, withStd bool, mean, variance, scale []float64, fitted bool) {
	return s.withMean, s.withStd, clone(s.mean), clone(s.variance), clone(s.scale), s.fitted
}

// RestoreStandardScaler constructs a fitted scaler from persisted parameters.
// It rejects inconsistent dimensions, non-finite values, and non-positive
// scales.
func RestoreStandardScaler(withMean, withStd bool, mean, variance, scale []float64) (*StandardScaler, error) {
	if len(scale) == 0 || (withMean && len(mean) != len(scale)) || (!withMean && len(mean) != 0) || (withStd && len(variance) != len(scale)) || (!withStd && len(variance) != 0) {
		return nil, errors.New("preprocessing: invalid persisted scaler dimensions")
	}
	for i, value := range scale {
		if !isFinite(value) || value <= 0 {
			return nil, fmt.Errorf("preprocessing: invalid persisted scale at index %d", i)
		}
	}
	for i, value := range mean {
		if !isFinite(value) {
			return nil, fmt.Errorf("preprocessing: invalid persisted mean at index %d", i)
		}
	}
	for i, value := range variance {
		if !isFinite(value) || value < 0 {
			return nil, fmt.Errorf("preprocessing: invalid persisted variance at index %d", i)
		}
	}
	return &StandardScaler{withMean: withMean, withStd: withStd, mean: clone(mean), variance: clone(variance), scale: clone(scale), fitted: true}, nil
}

func columns(x *matrix.Dense) int {
	if x == nil {
		return 0
	}
	return x.Cols()
}

func clone(values []float64) []float64 {
	if values == nil {
		return nil
	}
	out := make([]float64, len(values))
	copy(out, values)
	return out
}

func isFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
