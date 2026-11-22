package model_selection_test

import (
	"errors"
	"reflect"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/model_selection"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func TestKFoldPartitionsAndRemainder(t *testing.T) {
	folds, err := model_selection.KFold(10, 3, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantTests := [][]int{{0, 1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	seen := make([]int, 10)
	for i, fold := range folds {
		if !reflect.DeepEqual(fold.TestIndices, wantTests[i]) {
			t.Fatalf("fold %d test indices=%v want %v", i, fold.TestIndices, wantTests[i])
		}
		if len(fold.TrainIndices)+len(fold.TestIndices) != 10 {
			t.Fatalf("fold %d sizes do not cover all samples", i)
		}
		train := make(map[int]bool, len(fold.TrainIndices))
		for _, index := range fold.TrainIndices {
			train[index] = true
		}
		for _, index := range fold.TestIndices {
			if train[index] {
				t.Fatalf("sample %d occurs in train and test for fold %d", index, i)
			}
			seen[index]++
		}
	}
	for index, count := range seen {
		if count != 1 {
			t.Fatalf("sample %d appears in test %d times", index, count)
		}
	}
}

func TestKFoldShuffleReproducibilityAndValidation(t *testing.T) {
	seed := uint64(42)
	first, err := model_selection.KFold(12, 5, true, &seed)
	if err != nil {
		t.Fatal(err)
	}
	second, err := model_selection.KFold(12, 5, true, &seed)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same seed produced different folds\n%v\n%v", first, second)
	}
	if _, err := model_selection.KFold(1, 2, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("invalid n error=%v", err)
	}
	if _, err := model_selection.KFold(5, 1, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("invalid k error=%v", err)
	}
	if _, err := model_selection.KFold(5, 6, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("k > n error=%v", err)
	}
	seed = 1 << 32
	if _, err := model_selection.KFold(5, 2, true, &seed); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("invalid seed error=%v", err)
	}
}

func TestCrossValScoreRegressionCreatesFreshPipelines(t *testing.T) {
	x, _ := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}})
	y := []float64{1, 3, 5, 7, 9, 11, 13, 15}
	created := 0
	scores, err := model_selection.CrossValScore(x, y, 4, func() pipeline.Regressor {
		created++
		p, err := pipeline.NewRegressionPipeline(
			[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
			func() pipeline.Regressor { return linear_model.NewLinearRegression() },
		)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if created != 4 || len(scores) != 4 {
		t.Fatalf("created=%d scores=%v", created, scores)
	}
	for i, score := range scores {
		if score < 1-1e-12 {
			t.Fatalf("fold %d R²=%g, want 1", i, score)
		}
	}
}

func TestCrossValClassificationScoreAndFoldError(t *testing.T) {
	x, _ := matrix.NewDenseFromRows([][]float64{{-4}, {-3}, {-2}, {-1}, {1}, {2}, {3}, {4}})
	y := []int{0, 1, 0, 1, 0, 1, 0, 1}
	created := 0
	scores, err := model_selection.CrossValClassificationScore(x, y, 4, func() pipeline.Classifier {
		created++
		p, err := pipeline.NewClassificationPipeline(
			[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
			func() pipeline.Classifier { return linear_model.NewLogisticRegression() },
		)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if created != 4 || len(scores) != 4 {
		t.Fatalf("created=%d scores=%v", created, scores)
	}
	for i, score := range scores {
		if score < 0 || score > 1 {
			t.Fatalf("fold %d accuracy=%g outside [0,1]", i, score)
		}
	}

	groupedX, _ := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}})
	if _, err := model_selection.CrossValClassificationScore(groupedX, []int{0, 0, 1, 1}, 2,
		func() pipeline.Classifier { return linear_model.NewLogisticRegression() }, false, nil); err == nil {
		t.Fatal("expected fold training error when training data contains one class")
	}
}

func TestCrossValScoreRejectsInvalidInputs(t *testing.T) {
	x, _ := matrix.NewDenseFromRows([][]float64{{0}, {1}, {2}, {3}})
	if _, err := model_selection.CrossValScore(x, []float64{0}, 2, func() pipeline.Regressor { return linear_model.NewLinearRegression() }, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("regression target mismatch error=%v", err)
	}
	if _, err := model_selection.CrossValScore(x, []float64{0, 1, 2, 3}, 2, nil, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("nil regression factory error=%v", err)
	}
	if _, err := model_selection.CrossValClassificationScore(x, []int{0, 1}, 2, func() pipeline.Classifier { return linear_model.NewLogisticRegression() }, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("classification target mismatch error=%v", err)
	}
	if _, err := model_selection.CrossValClassificationScore(x, []int{0, 1, 0, 1}, 2, nil, false, nil); !errors.Is(err, model_selection.ErrInvalidSplit) {
		t.Fatalf("nil classifier factory error=%v", err)
	}
}
