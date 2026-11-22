// Package pipeline composes preprocessing steps and a regression estimator.
package pipeline

import (
	"errors"
	"fmt"

	"go-learn/matrix"
)

var ErrNotFitted = errors.New("pipeline: estimator is not fitted")

// Transformer is the interface required for an intermediate pipeline step.
type Transformer interface {
	Fit(*matrix.Dense) error
	Transform(*matrix.Dense) (*matrix.Dense, error)
}

// Regressor is the interface required by a regression pipeline.
type Regressor interface {
	Fit(*matrix.Dense, []float64) error
	Predict(*matrix.Dense) ([]float64, error)
}

// Step describes a named transformer factory. Each successful Fit creates new
// transformers, so repeated fits do not share fitted state.
type Step struct {
	Name string
	New  func() Transformer
}

// RegressionFactory creates a new regression estimator.
type RegressionFactory func() Regressor

type fittedStep struct {
	name        string
	transformer Transformer
}

// RegressionPipeline transforms input through named steps before regression.
// Fit is not safe to call concurrently with Predict or another Fit.
type RegressionPipeline struct {
	steps        []Step
	newRegressor RegressionFactory
	fittedSteps  []fittedStep
	regressor    Regressor
	fitted       bool
}

// NewRegressionPipeline validates and constructs a regression pipeline.
func NewRegressionPipeline(steps []Step, newRegressor RegressionFactory) (*RegressionPipeline, error) {
	if newRegressor == nil {
		return nil, errors.New("pipeline: regressor factory is nil")
	}
	seen := make(map[string]struct{}, len(steps))
	stepsCopy := make([]Step, len(steps))
	for i, step := range steps {
		if step.Name == "" || step.New == nil {
			return nil, fmt.Errorf("pipeline: step %d must have a name and factory", i)
		}
		if _, exists := seen[step.Name]; exists {
			return nil, fmt.Errorf("pipeline: duplicate step name %q", step.Name)
		}
		seen[step.Name] = struct{}{}
		stepsCopy[i] = step
	}
	return &RegressionPipeline{steps: stepsCopy, newRegressor: newRegressor}, nil
}

// Fit creates and fits fresh steps and a fresh regressor. It publishes the new
// fitted state only after every stage succeeds.
func (p *RegressionPipeline) Fit(x *matrix.Dense, y []float64) error {
	if x == nil {
		return errors.New("pipeline: input matrix is nil")
	}
	current := x
	candidateSteps := make([]fittedStep, len(p.steps))
	for i, step := range p.steps {
		if step.New == nil {
			return fmt.Errorf("pipeline: step %q cannot be refitted after loading", step.Name)
		}
		transformer := step.New()
		if transformer == nil {
			return fmt.Errorf("pipeline: step %q factory returned nil", step.Name)
		}
		if err := transformer.Fit(current); err != nil {
			return fmt.Errorf("pipeline: fit step %q: %w", step.Name, err)
		}
		transformed, err := transformer.Transform(current)
		if err != nil {
			return fmt.Errorf("pipeline: transform step %q after fit: %w", step.Name, err)
		}
		if transformed == nil {
			return fmt.Errorf("pipeline: step %q returned nil transformed data", step.Name)
		}
		candidateSteps[i] = fittedStep{name: step.Name, transformer: transformer}
		current = transformed
	}
	candidateRegressor := p.newRegressor()
	if candidateRegressor == nil {
		return errors.New("pipeline: regressor factory returned nil")
	}
	if err := candidateRegressor.Fit(current, y); err != nil {
		return fmt.Errorf("pipeline: fit regressor: %w", err)
	}
	p.fittedSteps, p.regressor, p.fitted = candidateSteps, candidateRegressor, true
	return nil
}

// Predict applies fitted transformations and returns regression predictions.
func (p *RegressionPipeline) Predict(x *matrix.Dense) ([]float64, error) {
	if !p.fitted {
		return nil, ErrNotFitted
	}
	if x == nil {
		return nil, errors.New("pipeline: input matrix is nil")
	}
	current := x
	for _, step := range p.fittedSteps {
		transformed, err := step.transformer.Transform(current)
		if err != nil {
			return nil, fmt.Errorf("pipeline: transform step %q: %w", step.name, err)
		}
		if transformed == nil {
			return nil, fmt.Errorf("pipeline: step %q returned nil transformed data", step.name)
		}
		current = transformed
	}
	predictions, err := p.regressor.Predict(current)
	if err != nil {
		return nil, fmt.Errorf("pipeline: predict regressor: %w", err)
	}
	return predictions, nil
}

// Fitted reports whether Fit has completed successfully.
func (p *RegressionPipeline) Fitted() bool { return p.fitted }

// FittedTransformers returns the fitted transformers in pipeline order.
// It is intended for inspection and persistence. Returned objects remain owned
// by the pipeline and must not be modified while predictions are in progress.
func (p *RegressionPipeline) FittedTransformers() []NamedTransformer {
	steps := make([]NamedTransformer, len(p.fittedSteps))
	for i, step := range p.fittedSteps {
		steps[i] = NamedTransformer{Name: step.name, Transformer: step.transformer}
	}
	return steps
}

// FittedRegressor returns the fitted estimator, or nil before a successful Fit.
// It is intended for inspection and persistence.
func (p *RegressionPipeline) FittedRegressor() Regressor { return p.regressor }

// RestoreRegressionPipeline constructs a fitted pipeline from fitted components.
// It is intended for loading a previously serialized pipeline.
func RestoreRegressionPipeline(steps []NamedTransformer, regressor Regressor) (*RegressionPipeline, error) {
	if regressor == nil {
		return nil, errors.New("pipeline: restored regressor is nil")
	}
	seen := make(map[string]struct{}, len(steps))
	fittedSteps := make([]fittedStep, len(steps))
	factories := make([]Step, len(steps))
	for i, step := range steps {
		if step.Name == "" || step.Transformer == nil {
			return nil, fmt.Errorf("pipeline: restored step %d is incomplete", i)
		}
		if _, exists := seen[step.Name]; exists {
			return nil, fmt.Errorf("pipeline: duplicate restored step name %q", step.Name)
		}
		seen[step.Name] = struct{}{}
		fittedSteps[i] = fittedStep{name: step.Name, transformer: step.Transformer}
		factories[i] = Step{Name: step.Name}
	}
	return &RegressionPipeline{steps: factories, fittedSteps: fittedSteps, regressor: regressor, fitted: true}, nil
}

// NamedTransformer pairs a fitted transformation with its pipeline name.
type NamedTransformer struct {
	Name        string
	Transformer Transformer
}
