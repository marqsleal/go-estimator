package tabular

import (
	"bytes"
	"testing"

	"go-learn/matrix"
)

func TestParquetTableRoundTrip(t *testing.T) {
	data, _ := matrix.NewDenseFromRows([][]float64{{1.25, -2}, {3, 4}})
	table, err := NewTable([]string{"feature", "label"}, data)
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := WriteParquet(&encoded, table); err != nil {
		t.Fatal(err)
	}
	got, err := ReadParquet(bytes.NewReader(encoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if got.Columns[0] != "feature" || got.Columns[1] != "label" {
		t.Fatalf("columns=%v", got.Columns)
	}
	for i, value := range data.Values() {
		if got.Data.Values()[i] != value {
			t.Errorf("value %d=%g want %g", i, got.Data.Values()[i], value)
		}
	}
}

func TestReadParquetRejectsInvalidInput(t *testing.T) {
	if table, err := ReadParquet(bytes.NewReader([]byte("not parquet"))); err == nil || table != nil {
		t.Fatalf("ReadParquet()=(%v,%v), want error", table, err)
	}
}
