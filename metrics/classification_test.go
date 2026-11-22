package metrics

import (
	"reflect"
	"testing"
)

func TestClassificationMetrics(t *testing.T) {
	truth := []int{-2, -2, 9, 9}
	pred := []int{-2, 9, 9, 9}
	for _, tc := range []struct {
		name  string
		score func() (float64, error)
		want  float64
	}{
		{"accuracy", func() (float64, error) { return AccuracyScore(truth, pred) }, 0.75},
		{"precision", func() (float64, error) { return PrecisionScore(truth, pred, 9) }, 2.0 / 3},
		{"recall", func() (float64, error) { return RecallScore(truth, pred, 9) }, 1},
		{"zero_precision", func() (float64, error) { return PrecisionScore([]int{9}, []int{-2}, 9) }, 0},
		{"zero_recall", func() (float64, error) { return RecallScore([]int{-2}, []int{9}, 9) }, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.score()
			if err != nil || got != tc.want {
				t.Fatal(got, err)
			}
		})
	}
	counts, labels, err := ConfusionMatrix([]int{9, -2}, []int{20, -2})
	if err != nil || !reflect.DeepEqual(labels, []int{-2, 9, 20}) || !reflect.DeepEqual(counts, [][]int{{1, 0, 0}, {0, 0, 1}, {0, 0, 0}}) {
		t.Fatal(counts, labels, err)
	}
	if _, err := AccuracyScore(nil, nil); err == nil {
		t.Fatal("empty labels")
	}
	if _, err := RecallScore([]int{0}, []int{0}, 1); err == nil {
		t.Fatal("absent positive label")
	}
	if _, err := PrecisionScore([]int{0}, nil, 0); err == nil {
		t.Fatal("length mismatch")
	}
	if _, _, err := ConfusionMatrix(nil, nil); err == nil {
		t.Fatal("empty matrix")
	}
}
