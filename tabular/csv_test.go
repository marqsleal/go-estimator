package tabular

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"go-learn/matrix"
)

func TestReadCSVNumericMatrix(t *testing.T) {
	got, err := ReadCSV(strings.NewReader("1,2\n3.5,-4\n"), CSVOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows() != 2 || got.Cols() != 2 {
		t.Fatalf("shape=(%d,%d), want (2,2)", got.Rows(), got.Cols())
	}
	want := []float64{1, 2, 3.5, -4}
	for i, value := range got.Values() {
		if value != want[i] {
			t.Errorf("value %d=%v, want %v", i, value, want[i])
		}
	}
}

func TestReadCSVHeaderAndDelimiter(t *testing.T) {
	got, err := ReadCSV(strings.NewReader("a;b\n1;2\n3;4\n"), CSVOptions{Header: true, Delimiter: ';'})
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows() != 2 || got.Cols() != 2 || got.At(1, 1) != 4 {
		t.Fatalf("unexpected matrix: %#v", got.Values())
	}
}

func TestReadCSVReportsMalformedRowsAndValues(t *testing.T) {
	for _, input := range []string{"1,2\n3\n", "1,2\nx,4\n"} {
		if got, err := ReadCSV(strings.NewReader(input), CSVOptions{}); err == nil || got != nil {
			t.Errorf("ReadCSV(%q)=(%v,%v), want nil matrix and error", input, got, err)
		}
	}
}

func TestWriteAndReadCSVRoundTrip(t *testing.T) {
	values := []float64{math.Pi, -0.0, 1e-200, 1e200}
	data, err := matrix.NewDense(2, 2, values)
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := WriteCSV(&encoded, data, ';'); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadCSV(&encoded, CSVOptions{Delimiter: ';'})
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range decoded.Values() {
		if value != values[i] {
			t.Errorf("value %d=%v, want %v", i, value, values[i])
		}
	}
}

func TestForEachCSVRowStreamsRecords(t *testing.T) {
	count := 0
	err := ForEachCSVRow(strings.NewReader("name,value\na,1\nb,2\n"), ',', func(columns, row []string) error {
		if columns[0] != "name" || len(row) != 2 {
			t.Fatalf("unexpected record columns=%v row=%v", columns, row)
		}
		count++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("visited %d rows, want 2", count)
	}
}
