package tabular

import (
	"bytes"
	"strings"
	"testing"

	"go-learn/matrix"
)

func TestReadJSONNumericMatrix(t *testing.T) {
	got, err := ReadJSON(strings.NewReader(`[[1,2.5],[-3,4]]`))
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows() != 2 || got.Cols() != 2 || got.At(0, 1) != 2.5 || got.At(1, 0) != -3 {
		t.Fatalf("unexpected matrix %v", got.Values())
	}
}

func TestReadJSONRejectsInvalidOrRaggedData(t *testing.T) {
	for _, input := range []string{`null`, `[[1],[2,3]]`, `[[1,null]]`, `[[1,"x"]]`, `[[1]] false`} {
		if got, err := ReadJSON(strings.NewReader(input)); err == nil || got != nil {
			t.Errorf("ReadJSON(%s)=(%v,%v), want error", input, got, err)
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	data, err := matrix.NewDense(2, 2, []float64{1.25, -2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := WriteJSON(&encoded, data); err != nil {
		t.Fatal(err)
	}
	got, err := ReadJSON(&encoded)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range data.Values() {
		if got.Values()[i] != v {
			t.Errorf("value %d=%g want %g", i, got.Values()[i], v)
		}
	}
}

func TestJSONTableRoundTripPreservesColumnNames(t *testing.T) {
	data, _ := matrix.NewDenseFromRows([][]float64{{1, 0}, {2, 1}})
	table, err := NewTable([]string{"feature", "label"}, data)
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := WriteJSONTable(&encoded, table); err != nil {
		t.Fatal(err)
	}
	got, err := ReadJSONTable(&encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got.Columns[0] != "feature" || got.Columns[1] != "label" || got.Data.At(1, 0) != 2 {
		t.Fatalf("unexpected decoded table: %#v %v", got.Columns, got.Data.Values())
	}
}

func TestReadJSONTableRejectsInvalidColumnsAndRows(t *testing.T) {
	for _, input := range []string{`{"columns":["x","x"],"data":[[1,2]]}`, `{"columns":["x"],"data":[[1,2]]}`} {
		if table, err := ReadJSONTable(strings.NewReader(input)); err == nil || table != nil {
			t.Errorf("ReadJSONTable(%s)=(%v,%v), want error", input, table, err)
		}
	}
}
