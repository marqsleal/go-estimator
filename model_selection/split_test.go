package model_selection_test

import (
	"errors"
	"reflect"
	"testing"

	"go-learn/matrix"
	"go-learn/model_selection"
)

func TestTrainTestSplit(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}})
	if err != nil {
		t.Fatal(err)
	}
	y := []float64{0, 1, 2, 3, 4, 5}
	got, err := model_selection.TrainTestSplit(x, y, model_selection.WithTestSize(0.25), model_selection.WithShuffle(false))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.TrainIndices, []int{0, 1, 2, 3}) || !reflect.DeepEqual(got.TestIndices, []int{4, 5}) {
		t.Fatalf("indices train=%v test=%v", got.TrainIndices, got.TestIndices)
	}
	if !reflect.DeepEqual(got.YTrain, []float64{0, 1, 2, 3}) || !reflect.DeepEqual(got.YTest, []float64{4, 5}) {
		t.Fatalf("targets train=%v test=%v", got.YTrain, got.YTest)
	}
}

func TestTrainTestSplitSeedAndValidation(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}})
	if err != nil {
		t.Fatal(err)
	}
	y := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	first, err := model_selection.TrainTestSplit(x, y, model_selection.WithRandomState(0))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model_selection.TrainTestSplit(x, y, model_selection.WithRandomState(3)); err != nil {
		t.Fatal(err)
	}
	second, err := model_selection.TrainTestSplit(x, y, model_selection.WithRandomState(0))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.TestIndices, second.TestIndices) {
		t.Fatalf("seed is not repeatable: %v != %v", first.TestIndices, second.TestIndices)
	}
	if want := []int{1, 7}; !reflect.DeepEqual(first.TestIndices, want) {
		t.Fatalf("seed 0 test indices = %v, want %v", first.TestIndices, want)
	}
	if _, err := model_selection.TrainTestSplit(x, y, model_selection.WithRandomState(1<<32)); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("out-of-range seed error = %v", err)
	}
	if _, err := model_selection.TrainTestSplit(x, y[:2]); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("target length error = %v", err)
	}
}
