package metrics

import (
	"errors"
	"sort"
)

// ErrInvalidLabels indicates empty or mismatched classification targets.
var ErrInvalidLabels = errors.New("metrics: labels must be non-empty and equal-length")

func validateLabels(yTrue, yPred []int) error {
	if len(yTrue) == 0 || len(yTrue) != len(yPred) {
		return ErrInvalidLabels
	}
	return nil
}

// AccuracyScore returns the fraction of correctly predicted labels.
func AccuracyScore(yTrue, yPred []int) (float64, error) {
	if err := validateLabels(yTrue, yPred); err != nil {
		return 0, err
	}
	correct := 0
	for i, label := range yTrue {
		if label == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue)), nil
}

// PrecisionScore measures correctness among predictions of positiveLabel.
func PrecisionScore(yTrue, yPred []int, positiveLabel int) (float64, error) {
	return positiveScore(yTrue, yPred, positiveLabel, true)
}

// RecallScore measures coverage of true positiveLabel samples.
func RecallScore(yTrue, yPred []int, positiveLabel int) (float64, error) {
	return positiveScore(yTrue, yPred, positiveLabel, false)
}

func positiveScore(yTrue, yPred []int, positive int, precision bool) (float64, error) {
	if err := validateLabels(yTrue, yPred); err != nil {
		return 0, err
	}
	found := false
	correct, total := 0, 0
	for i, label := range yTrue {
		if label == positive || yPred[i] == positive {
			found = true
		}
		if label == positive && yPred[i] == positive {
			correct++
		}
		if (precision && yPred[i] == positive) || (!precision && label == positive) {
			total++
		}
	}
	if !found {
		return 0, errors.New("metrics: positive label is absent")
	}
	if total == 0 {
		return 0, nil
	}
	return float64(correct) / float64(total), nil
}

// ConfusionMatrix returns counts with true labels in rows and predictions in
// columns, followed by the sorted labels defining both axes.
func ConfusionMatrix(yTrue, yPred []int) ([][]int, []int, error) {
	if err := validateLabels(yTrue, yPred); err != nil {
		return nil, nil, err
	}
	indices := make(map[int]int)
	for i, label := range yTrue {
		indices[label] = 0
		indices[yPred[i]] = 0
	}
	labels := make([]int, 0, len(indices))
	for label := range indices {
		labels = append(labels, label)
	}
	sort.Ints(labels)
	counts := make([][]int, len(labels))
	for i, label := range labels {
		indices[label] = i
		counts[i] = make([]int, len(labels))
	}
	for i, label := range yTrue {
		counts[indices[label]][indices[yPred[i]]]++
	}
	return counts, labels, nil
}
