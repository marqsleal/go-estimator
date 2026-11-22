// Package metrics provides regression evaluation functions.
package metrics

import (
	"errors"
	"fmt"
	"math"
)

var ErrInvalidTargets = errors.New("metrics: targets must be non-empty, equal-length, and finite")

// MeanAbsoluteError computes the mean absolute error between yTrue and yPred.
func MeanAbsoluteError(yTrue, yPred []float64) (float64, error) {
	if err := validate(yTrue, yPred); err != nil {
		return 0, err
	}
	total := 0.0
	for i := range yTrue {
		total += math.Abs(yTrue[i] - yPred[i])
	}
	return total / float64(len(yTrue)), nil
}

// MeanSquaredError computes the mean squared error between yTrue and yPred.
func MeanSquaredError(yTrue, yPred []float64) (float64, error) {
	if err := validate(yTrue, yPred); err != nil {
		return 0, err
	}
	total := 0.0
	for i := range yTrue {
		delta := yTrue[i] - yPred[i]
		total += delta * delta
	}
	return total / float64(len(yTrue)), nil
}

// R2Score computes the coefficient of determination. It uses force-finite
// behavior for constant targets, matching the default of scikit-learn 1.9.1.
// A single sample returns NaN, matching scikit-learn's undefined score.
func R2Score(yTrue, yPred []float64) (float64, error) {
	if err := validate(yTrue, yPred); err != nil {
		return 0, err
	}
	if len(yTrue) < 2 {
		return math.NaN(), nil
	}
	mean := 0.0
	for _, value := range yTrue {
		mean += value / float64(len(yTrue))
	}
	residual, total := 0.0, 0.0
	for i, value := range yTrue {
		delta := value - yPred[i]
		residual += delta * delta
		centered := value - mean
		total += centered * centered
	}
	if total == 0 {
		if residual == 0 {
			return 1, nil
		}
		return 0, nil
	}
	return 1 - residual/total, nil
}

func validate(yTrue, yPred []float64) error {
	if len(yTrue) == 0 || len(yTrue) != len(yPred) {
		return fmt.Errorf("%w: got lengths %d and %d", ErrInvalidTargets, len(yTrue), len(yPred))
	}
	for i := range yTrue {
		if math.IsNaN(yTrue[i]) || math.IsInf(yTrue[i], 0) || math.IsNaN(yPred[i]) || math.IsInf(yPred[i], 0) {
			return fmt.Errorf("%w: non-finite value at index %d", ErrInvalidTargets, i)
		}
	}
	return nil
}
