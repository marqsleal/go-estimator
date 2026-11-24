package tabular

import (
	"go-learn/matrix"
	"testing"
)

func TestTableSelectAndSplitPreserveColumnAndRowOrder(t *testing.T) {
	data, _ := matrix.NewDenseFromRows([][]float64{{1, 10, 100}, {2, 20, 200}})
	table, err := NewTable([]string{"a", "b", "label"}, data)
	if err != nil {
		t.Fatal(err)
	}
	x, y, err := table.Split([]string{"b", "a"}, "label")
	if err != nil {
		t.Fatal(err)
	}
	if x.At(0, 0) != 10 || x.At(0, 1) != 1 || x.At(1, 0) != 20 || y[0] != 100 || y[1] != 200 {
		t.Fatalf("bad aligned selection X=%v y=%v", x.Values(), y)
	}
}

func TestNewTableRejectsInvalidNamesAndSelectUnknownColumn(t *testing.T) {
	data, _ := matrix.NewDense(1, 2, []float64{1, 2})
	for _, names := range [][]string{{"x"}, {"x", "x"}, {"x", ""}} {
		if _, err := NewTable(names, data); err == nil {
			t.Errorf("expected invalid names error for %v", names)
		}
	}
	table, _ := NewTable([]string{"x", "y"}, data)
	if _, err := table.Select("missing"); err == nil {
		t.Fatal("expected unknown column error")
	}
}

func TestNewTypedTableSupportsHeterogeneousAndNullCells(t *testing.T) {
	table, err := NewTypedTable([]string{"name", "score", "active"}, [][]any{{"ana", float64(9.5), true}, {"bia", nil, false}})
	if err != nil {
		t.Fatal(err)
	}
	if table.Rows[0][0] != "ana" || table.Rows[1][1] != nil {
		t.Fatalf("unexpected typed table %#v", table.Rows)
	}
}

func TestSplitIntPreservesRowsAndRejectsFractionalLabels(t *testing.T) {
	data, _ := matrix.NewDenseFromRows([][]float64{{10, 0}, {20, 1}, {30, 2}})
	table, err := NewTable([]string{"feature", "class"}, data)
	if err != nil {
		t.Fatal(err)
	}
	x, y, err := table.SplitInt([]string{"feature"}, "class")
	if err != nil {
		t.Fatal(err)
	}
	if x.At(2, 0) != 30 || y[0] != 0 || y[1] != 1 || y[2] != 2 {
		t.Fatalf("misaligned split: X=%v y=%v", x.Values(), y)
	}
	fractional, _ := matrix.NewDenseFromRows([][]float64{{10, 0.5}})
	bad, _ := NewTable([]string{"feature", "class"}, fractional)
	if _, _, err := bad.SplitInt([]string{"feature"}, "class"); err == nil {
		t.Fatal("expected fractional class error")
	}
}
