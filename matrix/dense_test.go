package matrix_test

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"go-learn/matrix"
)

func TestNewDense(t *testing.T) {
	tests := []struct {
		name    string
		rows    int
		cols    int
		values  []float64
		wantErr error
	}{
		{name: "row major shape", rows: 2, cols: 2, values: []float64{1, 2, 3, 4}},
		{name: "zero sized", rows: 0, cols: 4, values: []float64{}},
		{name: "negative rows", rows: -1, cols: 2, values: []float64{}, wantErr: matrix.ErrInvalidDimensions},
		{name: "size mismatch", rows: 2, cols: 2, values: []float64{1}, wantErr: matrix.ErrDataSize},
		{name: "overflow", rows: int(^uint(0) >> 1), cols: 2, values: []float64{}, wantErr: matrix.ErrInvalidDimensions},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := matrix.NewDense(tt.rows, tt.cols, tt.values)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("NewDense() error = %v, want %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}
			if rows, cols := got.Shape(); rows != tt.rows || cols != tt.cols {
				t.Fatalf("Shape() = (%d, %d), want (%d, %d)", rows, cols, tt.rows, tt.cols)
			}
			if !reflect.DeepEqual(got.Values(), tt.values) {
				t.Fatalf("Values() = %v, want %v", got.Values(), tt.values)
			}
		})
	}
}

func TestNewDenseCopiesInputAndValues(t *testing.T) {
	input := []float64{1, 2, 3, 4}
	got, err := matrix.NewDense(2, 2, input)
	if err != nil {
		t.Fatal(err)
	}
	input[0] = 99
	values := got.Values()
	values[1] = 99
	if want := []float64{1, 2, 3, 4}; !reflect.DeepEqual(got.Values(), want) {
		t.Fatalf("matrix changed through input/output slices: got %v, want %v", got.Values(), want)
	}
}

func TestNewDenseFromRows(t *testing.T) {
	tests := []struct {
		name    string
		rows    [][]float64
		want    []float64
		wantErr error
	}{
		{name: "rectangular", rows: [][]float64{{1, 2}, {3, 4}}, want: []float64{1, 2, 3, 4}},
		{name: "empty", rows: [][]float64{}, want: []float64{}},
		{name: "ragged", rows: [][]float64{{1, 2}, {3}}, wantErr: matrix.ErrRaggedRows},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := matrix.NewDenseFromRows(tt.rows)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("NewDenseFromRows() error = %v, want %v", err, tt.wantErr)
			}
			if err == nil && !reflect.DeepEqual(got.Values(), tt.want) {
				t.Fatalf("Values() = %v, want %v", got.Values(), tt.want)
			}
		})
	}
}

func TestSetRowAndClone(t *testing.T) {
	got, err := matrix.NewDenseFromRows([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}
	if err := got.Set(1, 0, 8); err != nil {
		t.Fatal(err)
	}
	row, err := got.Row(1)
	if err != nil {
		t.Fatal(err)
	}
	if want := []float64{8, 4}; !reflect.DeepEqual(row, want) {
		t.Fatalf("Row(1) = %v, want %v", row, want)
	}
	row[0] = 10
	clone := got.Clone()
	if err := clone.Set(1, 1, 9); err != nil {
		t.Fatal(err)
	}
	if got.At(1, 0) != 8 || got.At(1, 1) != 4 {
		t.Fatalf("original changed through row or clone: %v", got.Values())
	}
	if !errors.Is(got.Set(-1, 0, 0), matrix.ErrIndexOutOfRange) {
		t.Fatal("Set() with invalid index did not return ErrIndexOutOfRange")
	}
}

func TestDenseAtOutOfRangePanics(t *testing.T) {
	got, err := matrix.NewDense(1, 1, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if recovered := recover(); recovered == nil {
			t.Fatal("At with an invalid index did not panic")
		}
	}()
	got.At(1, 0)
}

func ExampleNewDense() {
	values := []float64{1, 2, 3, 4}
	m, err := matrix.NewDense(2, 2, values)
	if err != nil {
		panic(err)
	}
	rows, cols := m.Shape()
	fmt.Println(rows, cols)
	fmt.Println(m.At(1, 0))
	// Output:
	// 2 2
	// 3
}
