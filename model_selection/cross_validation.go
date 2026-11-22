package model_selection

import (
	"fmt"

	"go-learn/matrix"
	"go-learn/metrics"
	"go-learn/pipeline"
)

// CrossValScore fits a fresh regression pipeline for every fold and returns
// the R² score for each validation partition.
func CrossValScore(x *matrix.Dense, y []float64, k int, newEstimator pipeline.RegressionFactory, shuffle bool, randomState *uint64) ([]float64, error) {
	if x == nil || x.Rows() != len(y) || newEstimator == nil {
		return nil, fmt.Errorf("%w: invalid regression data or nil estimator factory", ErrInvalidSplit)
	}
	folds, err := KFold(x.Rows(), k, shuffle, randomState)
	if err != nil {
		return nil, err
	}
	scores := make([]float64, len(folds))
	for i, fold := range folds {
		trainX, trainY, err := selectRows(x, y, fold.TrainIndices)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d select training rows: %w", i, err)
		}
		testX, testY, err := selectRows(x, y, fold.TestIndices)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d select validation rows: %w", i, err)
		}
		model := newEstimator()
		if model == nil {
			return nil, fmt.Errorf("model_selection: fold %d estimator factory returned nil", i)
		}
		if err := model.Fit(trainX, trainY); err != nil {
			return nil, fmt.Errorf("model_selection: fold %d fit: %w", i, err)
		}
		predictions, err := model.Predict(testX)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d predict: %w", i, err)
		}
		scores[i], err = metrics.R2Score(testY, predictions)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d score: %w", i, err)
		}
	}
	return scores, nil
}

// CrossValClassificationScore fits a fresh classification pipeline per fold
// and returns accuracy for each validation partition.
func CrossValClassificationScore(x *matrix.Dense, y []int, k int, newEstimator pipeline.ClassificationFactory, shuffle bool, randomState *uint64) ([]float64, error) {
	if x == nil || x.Rows() != len(y) || newEstimator == nil {
		return nil, fmt.Errorf("%w: invalid classification data or nil estimator factory", ErrInvalidSplit)
	}
	folds, err := KFold(x.Rows(), k, shuffle, randomState)
	if err != nil {
		return nil, err
	}
	scores := make([]float64, len(folds))
	for i, fold := range folds {
		trainX, trainY, err := selectRowsInt(x, y, fold.TrainIndices)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d select training rows: %w", i, err)
		}
		testX, testY, err := selectRowsInt(x, y, fold.TestIndices)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d select validation rows: %w", i, err)
		}
		model := newEstimator()
		if model == nil {
			return nil, fmt.Errorf("model_selection: fold %d estimator factory returned nil", i)
		}
		if err := model.Fit(trainX, trainY); err != nil {
			return nil, fmt.Errorf("model_selection: fold %d fit: %w", i, err)
		}
		predictions, err := model.Predict(testX)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d predict: %w", i, err)
		}
		scores[i], err = metrics.AccuracyScore(testY, predictions)
		if err != nil {
			return nil, fmt.Errorf("model_selection: fold %d score: %w", i, err)
		}
	}
	return scores, nil
}

func selectRowsInt(x *matrix.Dense, y []int, indices []int) (*matrix.Dense, []int, error) {
	values := make([]float64, 0, len(indices)*x.Cols())
	targets := make([]int, len(indices))
	for i, index := range indices {
		row, err := x.Row(index)
		if err != nil {
			return nil, nil, err
		}
		values = append(values, row...)
		targets[i] = y[index]
	}
	selected, err := matrix.NewDense(len(indices), x.Cols(), values)
	return selected, targets, err
}
