package preprocessing_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"go-learn/matrix"
	"go-learn/preprocessing"
)

func TestStandardScalerFitTransform(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{1, 4}, {3, 4}, {5, 4}})
	if err != nil {
		t.Fatal(err)
	}
	scaler := preprocessing.NewStandardScaler()
	got, err := scaler.FitTransform(x)
	if err != nil {
		t.Fatal(err)
	}
	if want := []float64{3, 4}; !reflect.DeepEqual(scaler.Mean(), want) {
		t.Fatalf("Mean() = %v, want %v", scaler.Mean(), want)
	}
	if want := []float64{8.0 / 3, 0}; !closeSlice(scaler.Var(), want, 1e-14) {
		t.Fatalf("Var() = %v, want %v", scaler.Var(), want)
	}
	if want := []float64{-math.Sqrt(1.5), 0, 0, 0, math.Sqrt(1.5), 0}; !closeSlice(got.Values(), want, 1e-14) {
		t.Fatalf("Transform() = %v, want %v", got.Values(), want)
	}
	if !reflect.DeepEqual(x.Values(), []float64{1, 4, 3, 4, 5, 4}) {
		t.Fatalf("input was modified: %v", x.Values())
	}
}

func TestStandardScalerOptionsAndErrors(t *testing.T) {
	x, err := matrix.NewDenseFromRows([][]float64{{1, 3}, {3, 5}})
	if err != nil {
		t.Fatal(err)
	}
	scaler := preprocessing.NewStandardScaler(preprocessing.WithMean(false), preprocessing.WithStd(false))
	if _, err := scaler.Transform(x); !errors.Is(err, preprocessing.ErrNotFitted) {
		t.Fatalf("Transform before Fit error = %v", err)
	}
	if err := scaler.Fit(x); err != nil {
		t.Fatal(err)
	}
	if scaler.Mean() != nil || scaler.Var() != nil {
		t.Fatalf("disabled statistics should be nil: mean=%v var=%v", scaler.Mean(), scaler.Var())
	}
	got, err := scaler.Transform(x)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Values(), x.Values()) {
		t.Fatalf("disabled transform = %v, want %v", got.Values(), x.Values())
	}

	bad, err := matrix.NewDenseFromRows([][]float64{{math.NaN()}})
	if err != nil {
		t.Fatal(err)
	}
	if err := scaler.Fit(bad); !errors.Is(err, preprocessing.ErrNonFinite) {
		t.Fatalf("Fit non-finite error = %v", err)
	}
	if !scaler.Fitted() {
		t.Fatal("failed Fit should preserve prior fitted state")
	}
}

func closeSlice(got, want []float64, tolerance float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tolerance {
			return false
		}
	}
	return true
}
