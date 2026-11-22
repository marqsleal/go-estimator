package linear_model

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"go-learn/matrix"
	"go-learn/metrics"
)

// ErrConvergence indicates a failed logistic optimization.
var ErrConvergence = errors.New("linear_model: logistic optimization did not converge")

// LogisticRegressionOption configures the binary logistic estimator.
type LogisticRegressionOption func(*LogisticRegression)

// WithC sets inverse L2 regularization strength (default 1).
func WithC(c float64) LogisticRegressionOption { return func(m *LogisticRegression) { m.c = c } }

// WithLogisticFitIntercept controls the unpenalized intercept (default true).
func WithLogisticFitIntercept(enabled bool) LogisticRegressionOption {
	return func(m *LogisticRegression) { m.fitIntercept = enabled }
}

// WithTolerance sets the maximum absolute gradient at convergence (default 1e-6).
func WithTolerance(tol float64) LogisticRegressionOption {
	return func(m *LogisticRegression) { m.tolerance = tol }
}

// WithMaxIterations sets the maximum Newton steps (default 100).
func WithMaxIterations(n int) LogisticRegressionOption {
	return func(m *LogisticRegression) { m.maxIterations = n }
}

// LogisticRegression fits binary integer labels with L2-regularized Newton
// optimization. Fit must not run concurrently with reads or another Fit.
type LogisticRegression struct {
	c, tolerance         float64
	maxIterations        int
	fitIntercept, fitted bool
	coefficients         []float64
	intercept            float64
	classes              []int
	iterations           int
}

// NewLogisticRegression constructs a binary estimator. Fit validates options.
func NewLogisticRegression(options ...LogisticRegressionOption) *LogisticRegression {
	m := &LogisticRegression{c: 1, tolerance: 1e-6, maxIterations: 100, fitIntercept: true}
	for _, option := range options {
		if option != nil {
			option(m)
		}
	}
	return m
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}

// Fit publishes a fresh adjustment only after convergence.
func (m *LogisticRegression) Fit(x *matrix.Dense, y []int) error {
	if !finite(m.c) || m.c <= 0 || !finite(m.tolerance) || m.tolerance <= 0 || m.maxIterations <= 0 {
		return errors.New("linear_model: invalid logistic options")
	}
	if x == nil || x.Rows() == 0 || x.Cols() == 0 || x.Rows() != len(y) {
		return errors.New("linear_model: incompatible X and y dimensions")
	}
	labels := make(map[int]bool)
	for i, label := range y {
		labels[label] = true
		for j := 0; j < x.Cols(); j++ {
			if !finite(x.At(i, j)) {
				return errors.New("linear_model: non-finite X")
			}
		}
	}
	if len(labels) != 2 {
		return errors.New("linear_model: exactly two classes required")
	}
	classes := make([]int, 0, 2)
	for label := range labels {
		classes = append(classes, label)
	}
	sort.Ints(classes)
	q := x.Cols()
	if m.fitIntercept {
		q++
	}
	theta := make([]float64, q)
	for iteration := 0; iteration <= m.maxIterations; iteration++ {
		loss, gradient, hessian := m.objective(x, y, classes[1], theta, true)
		norm := 0.0
		for _, g := range gradient {
			norm = math.Max(norm, math.Abs(g))
		}
		if finite(loss) && norm <= m.tolerance {
			m.coefficients = append([]float64(nil), theta[:x.Cols()]...)
			m.intercept = 0
			if m.fitIntercept {
				m.intercept = theta[q-1]
			}
			m.classes, m.iterations, m.fitted = classes, iteration, true
			return nil
		}
		if iteration == m.maxIterations || !finite(loss) {
			return fmt.Errorf("%w after %d iterations", ErrConvergence, iteration)
		}
		direction, ok := newtonDirection(hessian, gradient)
		if !ok {
			return fmt.Errorf("%w: Hessian at iteration %d", ErrConvergence, iteration)
		}
		slope := 0.0
		for j := range theta {
			slope += gradient[j] * direction[j]
		}
		candidate := make([]float64, q)
		step := 1.0
		accepted := false
		for attempt := 0; attempt < 50; attempt++ {
			for j := range theta {
				candidate[j] = theta[j] + step*direction[j]
			}
			next, _, _ := m.objective(x, y, classes[1], candidate, false)
			if finite(next) && next <= loss+1e-4*step*slope {
				accepted = true
				break
			}
			step *= 0.5
		}
		if !accepted {
			return fmt.Errorf("%w: line search at iteration %d", ErrConvergence, iteration)
		}
		theta = candidate
	}
	return ErrConvergence
}

// objective uses a stable softplus loss and sigmoid curvature.
func (m *LogisticRegression) objective(x *matrix.Dense, y []int, positive int, theta []float64, derivatives bool) (float64, []float64, []float64) {
	q := len(theta)
	var gradient, hessian []float64
	if derivatives {
		gradient = make([]float64, q)
		hessian = make([]float64, q*q)
	}
	row := make([]float64, q)
	loss := 0.0
	n := float64(x.Rows())
	for i, label := range y {
		z := 0.0
		for j := 0; j < x.Cols(); j++ {
			row[j] = x.At(i, j)
			z += row[j] * theta[j]
		}
		if m.fitIntercept {
			row[q-1] = 1
			z += theta[q-1]
		}
		if !finite(z) {
			return math.Inf(1), gradient, hessian
		}
		target := 0.0
		if label == positive {
			target = 1
		}
		signed := z
		if target == 1 {
			signed = -z
		}
		loss += (math.Max(signed, 0) + math.Log1p(math.Exp(-math.Abs(signed)))) / n
		if derivatives {
			residual := sigmoid(z) - target
			curvature := sigmoid(z) * sigmoid(-z)
			for j := range row {
				gradient[j] += row[j] * residual / n
				for k := range row {
					hessian[j*q+k] += (row[j] * curvature / n) * row[k]
				}
			}
		}
	}
	penalty := (1 / m.c) / n
	for j := 0; j < x.Cols(); j++ {
		loss += 0.5 * (theta[j] * penalty) * theta[j]
		if derivatives {
			gradient[j] += penalty * theta[j]
			hessian[j*q+j] += penalty
		}
	}
	return loss, gradient, hessian
}

func newtonDirection(hessian, gradient []float64) ([]float64, bool) {
	n := len(gradient)
	lower := make([]float64, len(hessian))
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			value := hessian[i*n+j]
			for k := 0; k < j; k++ {
				value -= lower[i*n+k] * lower[j*n+k]
			}
			if i == j {
				if !finite(value) || value <= 0 {
					return nil, false
				}
				lower[i*n+j] = math.Sqrt(value)
			} else {
				lower[i*n+j] = value / lower[j*n+j]
			}
		}
	}
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		value := -gradient[i]
		for j := 0; j < i; j++ {
			value -= lower[i*n+j] * result[j]
		}
		result[i] = value / lower[i*n+i]
	}
	for i := n - 1; i >= 0; i-- {
		value := result[i]
		for j := i + 1; j < n; j++ {
			value -= lower[j*n+i] * result[j]
		}
		result[i] = value / lower[i*n+i]
		if !finite(result[i]) {
			return nil, false
		}
	}
	return result, true
}

func (m *LogisticRegression) logits(x *matrix.Dense) ([]float64, error) {
	if !m.fitted {
		return nil, ErrNotFitted
	}
	if x == nil || x.Cols() != len(m.coefficients) {
		return nil, errors.New("linear_model: incompatible features")
	}
	result := make([]float64, x.Rows())
	for i := range result {
		z := m.intercept
		for j, w := range m.coefficients {
			v := x.At(i, j)
			if !finite(v) {
				return nil, errors.New("linear_model: non-finite X")
			}
			z += v * w
		}
		if !finite(z) {
			return nil, errors.New("linear_model: non-finite logit")
		}
		result[i] = z
	}
	return result, nil
}

// Predict returns original integer labels. Zero logits choose the smaller class.
func (m *LogisticRegression) Predict(x *matrix.Dense) ([]int, error) {
	z, err := m.logits(x)
	if err != nil {
		return nil, err
	}
	result := make([]int, len(z))
	for i, v := range z {
		result[i] = m.classes[0]
		if v > 0 {
			result[i] = m.classes[1]
		}
	}
	return result, nil
}

// PredictProba returns two columns in Classes order.
func (m *LogisticRegression) PredictProba(x *matrix.Dense) (*matrix.Dense, error) {
	z, err := m.logits(x)
	if err != nil {
		return nil, err
	}
	values := make([]float64, 2*len(z))
	for i, v := range z {
		values[2*i] = sigmoid(-v)
		values[2*i+1] = sigmoid(v)
	}
	return matrix.NewDense(len(z), 2, values)
}

// Score returns classification accuracy.
func (m *LogisticRegression) Score(x *matrix.Dense, y []int) (float64, error) {
	pred, err := m.Predict(x)
	if err != nil {
		return 0, err
	}
	return metrics.AccuracyScore(y, pred)
}

// Coef returns a copy of the fitted coefficients.
func (m *LogisticRegression) Coef() []float64 { return append([]float64(nil), m.coefficients...) }

// Classes returns the sorted labels as a copy.
func (m *LogisticRegression) Classes() []int { return append([]int(nil), m.classes...) }

// Intercept returns the fitted intercept, or zero before fitting.
func (m *LogisticRegression) Intercept() float64 { return m.intercept }

// NFeaturesIn returns the fitted feature count.
func (m *LogisticRegression) NFeaturesIn() int { return len(m.coefficients) }

// NIter returns the Newton steps in the last successful adjustment.
func (m *LogisticRegression) NIter() int { return m.iterations }

// Fitted reports whether a successful Fit has completed.
func (m *LogisticRegression) Fitted() bool { return m.fitted }
