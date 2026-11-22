// Package modelio saves and loads fitted go-learn regression models.
package modelio

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

const formatName = "go-learn"
const formatVersion = 1

var ErrUnsupportedModel = errors.New("modelio: unsupported model type")

type artifact struct {
	Format    string            `json:"format"`
	Version   int               `json:"version"`
	Kind      string            `json:"kind"`
	NFeatures int               `json:"n_features"`
	Steps     []stepArtifact    `json:"steps,omitempty"`
	Regressor regressorArtifact `json:"regressor"`
}

type stepArtifact struct {
	Name     string    `json:"name"`
	Type     string    `json:"type"`
	WithMean bool      `json:"with_mean,omitempty"`
	WithStd  bool      `json:"with_std,omitempty"`
	Mean     []float64 `json:"mean,omitempty"`
	Variance []float64 `json:"variance,omitempty"`
	Scale    []float64 `json:"scale,omitempty"`
}

type regressorArtifact struct {
	Type         string    `json:"type"`
	FitIntercept bool      `json:"fit_intercept"`
	Coefficients []float64 `json:"coefficients"`
	Intercept    float64   `json:"intercept"`
}

// Save writes a versioned JSON representation of a fitted LinearRegression or
// a RegressionPipeline containing only StandardScaler steps and a
// LinearRegression regressor.
func Save(w io.Writer, model any) error {
	if w == nil {
		return errors.New("modelio: writer is nil")
	}
	data, err := encode(model)
	if err != nil {
		return err
	}
	if err := json.NewEncoder(w).Encode(data); err != nil {
		return fmt.Errorf("modelio: encode model: %w", err)
	}
	return nil
}

// LoadRegressor loads a saved LinearRegression or regression pipeline.
func LoadRegressor(r io.Reader) (pipeline.Regressor, error) {
	if r == nil {
		return nil, errors.New("modelio: reader is nil")
	}
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()
	var data artifact
	if err := decoder.Decode(&data); err != nil {
		return nil, fmt.Errorf("modelio: decode artifact: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, errors.New("modelio: artifact contains multiple JSON values")
		}
		return nil, fmt.Errorf("modelio: trailing data: %w", err)
	}
	if data.Format != formatName || data.Version != formatVersion {
		return nil, fmt.Errorf("modelio: unsupported format/version %q/%d", data.Format, data.Version)
	}
	regressor, err := restoreRegressor(data.Regressor)
	if err != nil {
		return nil, err
	}
	if data.NFeatures <= 0 || regressor.NFeaturesIn() != data.NFeatures {
		return nil, errors.New("modelio: artifact feature count is inconsistent")
	}
	switch data.Kind {
	case "linear_regression":
		if len(data.Steps) != 0 {
			return nil, errors.New("modelio: standalone regressor cannot contain steps")
		}
		return regressor, nil
	case "regression_pipeline":
		steps := make([]pipeline.NamedTransformer, len(data.Steps))
		for i, step := range data.Steps {
			if step.Type != "standard_scaler" {
				return nil, fmt.Errorf("modelio: unsupported pipeline step type %q", step.Type)
			}
			scaler, err := preprocessing.RestoreStandardScaler(step.WithMean, step.WithStd, step.Mean, step.Variance, step.Scale)
			if err != nil {
				return nil, fmt.Errorf("modelio: restore step %q: %w", step.Name, err)
			}
			if len(step.Scale) != data.NFeatures {
				return nil, fmt.Errorf("modelio: step %q feature count is inconsistent", step.Name)
			}
			steps[i] = pipeline.NamedTransformer{Name: step.Name, Transformer: scaler}
		}
		return pipeline.RestoreRegressionPipeline(steps, regressor)
	default:
		return nil, fmt.Errorf("modelio: unsupported model kind %q", data.Kind)
	}
}

func encode(model any) (artifact, error) {
	data := artifact{Format: formatName, Version: formatVersion}
	switch value := model.(type) {
	case *linear_model.LinearRegression:
		state, err := encodeRegressor(value)
		if err != nil {
			return artifact{}, err
		}
		data.Kind, data.Regressor, data.NFeatures = "linear_regression", state, len(state.Coefficients)
	case *pipeline.RegressionPipeline:
		if value == nil || !value.Fitted() {
			return artifact{}, errors.New("modelio: cannot save an unfitted pipeline")
		}
		regressor, ok := value.FittedRegressor().(*linear_model.LinearRegression)
		if !ok {
			return artifact{}, ErrUnsupportedModel
		}
		state, err := encodeRegressor(regressor)
		if err != nil {
			return artifact{}, err
		}
		data.Kind, data.Regressor, data.NFeatures = "regression_pipeline", state, len(state.Coefficients)
		for _, fitted := range value.FittedTransformers() {
			scaler, ok := fitted.Transformer.(*preprocessing.StandardScaler)
			if !ok {
				return artifact{}, fmt.Errorf("%w: pipeline step %q", ErrUnsupportedModel, fitted.Name)
			}
			withMean, withStd, mean, variance, scale, isFitted := scaler.State()
			if !isFitted || len(scale) != data.NFeatures {
				return artifact{}, fmt.Errorf("modelio: pipeline step %q is unfitted or has inconsistent dimensions", fitted.Name)
			}
			data.Steps = append(data.Steps, stepArtifact{Name: fitted.Name, Type: "standard_scaler", WithMean: withMean, WithStd: withStd, Mean: mean, Variance: variance, Scale: scale})
		}
	default:
		return artifact{}, fmt.Errorf("%w: %T", ErrUnsupportedModel, model)
	}
	return data, nil
}

func encodeRegressor(model *linear_model.LinearRegression) (regressorArtifact, error) {
	if model == nil {
		return regressorArtifact{}, errors.New("modelio: regressor is nil")
	}
	fitIntercept, coefficients, intercept, fitted := model.State()
	if !fitted {
		return regressorArtifact{}, errors.New("modelio: cannot save an unfitted regressor")
	}
	return regressorArtifact{Type: "linear_regression", FitIntercept: fitIntercept, Coefficients: coefficients, Intercept: intercept}, nil
}

func restoreRegressor(data regressorArtifact) (*linear_model.LinearRegression, error) {
	if data.Type != "linear_regression" {
		return nil, fmt.Errorf("modelio: unsupported regressor type %q", data.Type)
	}
	model, err := linear_model.RestoreLinearRegression(data.FitIntercept, data.Coefficients, data.Intercept)
	if err != nil {
		return nil, fmt.Errorf("modelio: restore regressor: %w", err)
	}
	return model, nil
}

// Ensure the returned estimators satisfy the common prediction contract.
var _ interface {
	Predict(*matrix.Dense) ([]float64, error)
} = (*linear_model.LinearRegression)(nil)
