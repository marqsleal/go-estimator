package tabular

import (
	"bytes"
	"go-learn/matrix"
	"math"
	"strings"
	"testing"
)

func TestNumericJSONRejectsInvalidCells(t *testing.T) {
	for _, input := range []string{`[[null]]`, `[["12"]]`, `[null]`, `[[1]] false`} {
		if data, err := ReadJSON(strings.NewReader(input)); err == nil || data != nil {
			t.Fatalf("accepted %s", input)
		}
		document := `{"columns":["x"],"data":` + input + `}`
		if table, err := ReadJSONTable(strings.NewReader(document)); err == nil || table != nil {
			t.Fatalf("accepted table %s", document)
		}
	}
	if _, err := ReadJSONTable(strings.NewReader(`{"columns":["x"],"data":[[0]]} false`)); err == nil {
		t.Fatal("accepted trailing document")
	}
}

func TestParquetPreservesUnsortedColumns(t *testing.T) {
	data, _ := matrix.NewDense(1, 2, []float64{17, 2})
	table, _ := NewTable([]string{"z", "a"}, data)
	var buffer bytes.Buffer
	if err := WriteParquet(&buffer, table); err != nil {
		t.Fatal(err)
	}
	loaded, err := ReadParquet(&buffer)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Columns[0] != "z" || loaded.Data.At(0, 0) != 17 || loaded.Columns[1] != "a" || loaded.Data.At(0, 1) != 2 {
		t.Fatalf("corrupt round trip: %v", loaded)
	}
}

func TestEmptyNamedTables(t *testing.T) {
	csv, err := ReadCSVTable(strings.NewReader("x,y\n"), ',')
	if err != nil {
		t.Fatal(err)
	}
	json, err := ReadJSONTable(strings.NewReader(`{"columns":["x","y"],"data":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []*Table{csv, json} {
		if table.Data.Rows() != 0 || table.Data.Cols() != 2 {
			t.Fatal("lost shape")
		}
	}
}

func TestTypedSelectionAndIntegerBounds(t *testing.T) {
	table, _ := NewTypedTable([]string{"name", "x", "y"}, [][]any{{"Ana", int64(30), float64(2)}})
	x, y, err := table.SplitTargets([]string{"x"}, []string{"y"})
	if err != nil || x.At(0, 0) != 30 || y.At(0, 0) != 2 {
		t.Fatalf("selection: %v", err)
	}
	data, _ := matrix.NewDense(1, 1, []float64{-float64(math.MinInt)})
	labels, _ := NewTable([]string{"y"}, data)
	if _, _, err := labels.SplitInt(nil, "y"); err == nil {
		t.Fatal("accepted out-of-range integer")
	}
}
