package pipeline

import (
	"errors"
	"fmt"

	"go-learn/matrix"
	"go-learn/metrics"
)

// Classifier is the contract for a binary classification pipeline.
type Classifier interface {
	Fit(*matrix.Dense, []int) error
	Predict(*matrix.Dense) ([]int, error)
	PredictProba(*matrix.Dense) (*matrix.Dense, error)
	Classes() []int
}

// ClassificationFactory creates an independent, untrained classifier.
type ClassificationFactory func() Classifier

// ClassificationPipeline transforms input through named steps before classification.
// Fit is not safe to call concurrently with Predict or another Fit.
type ClassificationPipeline struct {
	steps         []Step
	newClassifier ClassificationFactory
	fittedSteps   []fittedStep
	classifier    Classifier
	fitted        bool
}

// NewClassificationPipeline validates and constructs a classification pipeline.
func NewClassificationPipeline(steps []Step, newClassifier ClassificationFactory) (*ClassificationPipeline, error) {
	if newClassifier == nil {
		return nil, errors.New("pipeline: classifier factory is nil")
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
	return &ClassificationPipeline{steps: stepsCopy, newClassifier: newClassifier}, nil
}

// Fit creates and fits fresh steps and a fresh classifier. It publishes the new
// fitted state only after every stage succeeds.
func (p *ClassificationPipeline) Fit(x *matrix.Dense, y []int) error {
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
	candidateClassifier := p.newClassifier()
	if candidateClassifier == nil {
		return errors.New("pipeline: classifier factory returned nil")
	}
	if err := candidateClassifier.Fit(current, y); err != nil {
		return fmt.Errorf("pipeline: fit classifier: %w", err)
	}
	p.fittedSteps, p.classifier, p.fitted = candidateSteps, candidateClassifier, true
	return nil
}

// Predict applies fitted transformations and returns classification predictions.
func (p *ClassificationPipeline) Predict(x *matrix.Dense) ([]int, error) {
	if !p.fitted {
		return nil, ErrNotFitted
	}
	current, err := p.transform(x)
	if err != nil {
		return nil, err
	}
	predictions, err := p.classifier.Predict(current)
	if err != nil {
		return nil, fmt.Errorf("pipeline: predict classifier: %w", err)
	}
	return predictions, nil
}

// Fitted reports whether Fit has completed successfully.
func (p *ClassificationPipeline) Fitted() bool { return p.fitted }

// Classes returns the fitted classifier labels in probability-column order.
func (p *ClassificationPipeline) Classes() []int {
	if !p.fitted {
		return nil
	}
	return append([]int(nil), p.classifier.Classes()...)
}

// PredictProba applies fitted transformations and returns class probabilities.
func (p *ClassificationPipeline) PredictProba(x *matrix.Dense) (*matrix.Dense, error) {
	if !p.fitted {
		return nil, ErrNotFitted
	}
	current, err := p.transform(x)
	if err != nil {
		return nil, err
	}
	probabilities, err := p.classifier.PredictProba(current)
	if err != nil {
		return nil, fmt.Errorf("pipeline: predict probabilities: %w", err)
	}
	return probabilities, nil
}

func (p *ClassificationPipeline) transform(x *matrix.Dense) (*matrix.Dense, error) {
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
	return current, nil
}

// Score returns accuracy after applying fitted transformations.
func (p *ClassificationPipeline) Score(x *matrix.Dense, y []int) (float64, error) {
	predictions, err := p.Predict(x)
	if err != nil {
		return 0, err
	}
	return metrics.AccuracyScore(y, predictions)
}
