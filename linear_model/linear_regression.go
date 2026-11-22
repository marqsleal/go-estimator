// Package linear_model contains linear estimators.
package linear_model

import (
	"errors"
	"fmt"
	"math"

	"go-learn/internal/linalg"
	"go-learn/matrix"
	"go-learn/metrics"
)

var ErrNotFitted = errors.New("linear_model: estimator is not fitted")
var ErrRankDeficient = linalg.ErrRankDeficient

// RegressionOption configures LinearRegression.
type RegressionOption func(*LinearRegression)

// WithFitIntercept controls whether a constant term is fitted. It defaults to true.
func WithFitIntercept(enabled bool) RegressionOption {
	return func(model *LinearRegression) { model.fitIntercept = enabled }
}

// LinearRegression fits a dense least-squares regression with one target.
// Fit is not safe to call concurrently with Predict or another Fit.
type LinearRegression struct {
	fitIntercept bool
	fitted       bool
	coefficients []float64
	intercept    float64
}

// NewLinearRegression creates a least-squares estimator with an intercept.
func NewLinearRegression(options ...RegressionOption) *LinearRegression {
	model := &LinearRegression{fitIntercept: true}
	for _, option := range options {
		if option != nil {
			option(model)
		}
	}
	return model
}

// Fit estimates coefficients and preserves the previous model if fitting fails.
func (model *LinearRegression) Fit(x *matrix.Dense, y []float64) error {
	if x == nil || x.Rows() == 0 || x.Cols() == 0 || x.Rows() != len(y) {
		return fmt.Errorf("linear_model: X and y dimensions are incompatible")
	}
	rows, cols := x.Shape()
	if rows < cols {
		return fmt.Errorf("linear_model: need at least %d samples, got %d", cols, rows)
	}
	for i, value := range y {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("linear_model: y contains a non-finite value at index %d", i)
		}
	}
	features := x.Values()
	for i, value := range features {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("linear_model: X contains a non-finite value at flat index %d", i)
		}
	}
	fitFeatures := append([]float64(nil), features...)
	fitTargets := append([]float64(nil), y...)
	intercept := 0.0
	if model.fitIntercept {
		means := make([]float64, cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				means[j] += features[i*cols+j] / float64(rows)
			}
			intercept += y[i] / float64(rows)
		}
		for j := 0; j < cols; j++ {
			for i := 0; i < rows; i++ {
				fitFeatures[i*cols+j] -= means[j]
			}
		}
		for i := range fitTargets {
			fitTargets[i] -= intercept
		}
	}
	coefficients, err := linalg.LeastSquares(fitFeatures, rows, cols, fitTargets)
	if err != nil {
		return fmt.Errorf("linear_model: fit least-squares model: %w", err)
	}
	if model.fitIntercept {
		for j, coefficient := range coefficients {
			mean := 0.0
			for i := 0; i < rows; i++ {
				mean += features[i*cols+j] / float64(rows)
			}
			intercept -= coefficient * mean
		}
	}
	if math.IsNaN(intercept) || math.IsInf(intercept, 0) {
		return errors.New("linear_model: fitted intercept is non-finite")
	}
	for i, coefficient := range coefficients {
		if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) {
			return fmt.Errorf("linear_model: fitted coefficient %d is non-finite", i)
		}
	}
	model.coefficients, model.intercept, model.fitted = coefficients, intercept, true
	return nil
}

// Predict returns one prediction per row.
func (model *LinearRegression) Predict(x *matrix.Dense) ([]float64, error) {
	if !model.fitted {
		return nil, ErrNotFitted
	}
	if x == nil || x.Cols() != len(model.coefficients) {
		return nil, fmt.Errorf("linear_model: got %d features, want %d", columns(x), len(model.coefficients))
	}
	predictions := make([]float64, x.Rows())
	for i := 0; i < x.Rows(); i++ {
		value := model.intercept
		for j, coefficient := range model.coefficients {
			feature := x.At(i, j)
			if math.IsNaN(feature) || math.IsInf(feature, 0) {
				return nil, fmt.Errorf("linear_model: X contains a non-finite value at (%d, %d)", i, j)
			}
			value += coefficient * feature
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("linear_model: prediction for row %d is non-finite", i)
		}
		predictions[i] = value
	}
	return predictions, nil
}

// Score returns the R² coefficient of determination.
func (model *LinearRegression) Score(x *matrix.Dense, y []float64) (float64, error) {
	predictions, err := model.Predict(x)
	if err != nil {
		return 0, err
	}
	return metrics.R2Score(y, predictions)
}

// Coef returns a copy of the fitted feature coefficients.
func (model *LinearRegression) Coef() []float64 {
	coefficients := make([]float64, len(model.coefficients))
	copy(coefficients, model.coefficients)
	return coefficients
}

// Intercept returns the fitted constant term.
func (model *LinearRegression) Intercept() float64 { return model.intercept }

// NFeaturesIn returns the number of fitted features.
func (model *LinearRegression) NFeaturesIn() int { return len(model.coefficients) }

// Fitted reports whether Fit has completed successfully.
func (model *LinearRegression) Fitted() bool { return model.fitted }

// State returns independent copies of the fitted regression parameters and
// its fit_intercept setting.
func (model *LinearRegression) State() (fitIntercept bool, coefficients []float64, intercept float64, fitted bool) {
	return model.fitIntercept, model.Coef(), model.intercept, model.fitted
}

// RestoreLinearRegression constructs a fitted estimator from persisted state.
func RestoreLinearRegression(fitIntercept bool, coefficients []float64, intercept float64) (*LinearRegression, error) {
	if len(coefficients) == 0 || math.IsNaN(intercept) || math.IsInf(intercept, 0) {
		return nil, errors.New("linear_model: invalid persisted regression state")
	}
	for i, value := range coefficients {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("linear_model: invalid persisted coefficient at index %d", i)
		}
	}
	return &LinearRegression{fitIntercept: fitIntercept, fitted: true, coefficients: append([]float64(nil), coefficients...), intercept: intercept}, nil
}

func columns(x *matrix.Dense) int {
	if x == nil {
		return 0
	}
	return x.Cols()
}
