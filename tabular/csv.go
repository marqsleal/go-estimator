// Package tabular reads and writes numeric tabular data.
package tabular

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	"go-learn/matrix"
)

// CSVOptions configures CSV headers and delimiters. A zero Rune uses comma.
type CSVOptions struct {
	Header    bool
	Delimiter rune
}

// ReadCSV reads a rectangular CSV of numeric values into a dense matrix.
// When Header is true, the first record is skipped.
func ReadCSV(reader io.Reader, options CSVOptions) (*matrix.Dense, error) {
	csvReader := csv.NewReader(reader)
	if options.Delimiter != 0 {
		csvReader.Comma = options.Delimiter
	}
	csvReader.FieldsPerRecord = -1
	var rows [][]float64
	width := -1
	for record := 0; ; record++ {
		fields, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("tabular: CSV record %d: %w", record+1, err)
		}
		if record == 0 && options.Header {
			continue
		}
		if width < 0 {
			width = len(fields)
		}
		if len(fields) != width {
			return nil, fmt.Errorf("tabular: CSV record %d has %d columns, want %d", record+1, len(fields), width)
		}
		row := make([]float64, len(fields))
		for col, field := range fields {
			value, parseErr := strconv.ParseFloat(field, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("tabular: CSV record %d column %d: %w", record+1, col+1, parseErr)
			}
			row[col] = value
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return matrix.NewDense(0, 0, nil)
	}
	return matrix.NewDenseFromRows(rows)
}

// ReadCSVTable reads a CSV whose first record contains unique column names.
func ReadCSVTable(reader io.Reader, delimiter rune) (*Table, error) {
	csvReader := csv.NewReader(reader)
	if delimiter != 0 {
		csvReader.Comma = delimiter
	}
	csvReader.FieldsPerRecord = -1
	header, err := csvReader.Read()
	if err != nil {
		return nil, fmt.Errorf("tabular: read CSV header: %w", err)
	}
	rows := make([][]float64, 0)
	for record := 2; ; record++ {
		fields, readErr := csvReader.Read()
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, fmt.Errorf("tabular: CSV record %d: %w", record, readErr)
		}
		if len(fields) != len(header) {
			return nil, fmt.Errorf("tabular: CSV record %d has %d columns, want %d", record, len(fields), len(header))
		}
		row := make([]float64, len(fields))
		for col, field := range fields {
			value, parseErr := strconv.ParseFloat(field, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("tabular: CSV record %d column %d: %w", record, col+1, parseErr)
			}
			row[col] = value
		}
		rows = append(rows, row)
	}
	body, err := matrix.NewDenseFromRows(rows)
	if len(rows) == 0 {
		body, err = matrix.NewDense(0, len(header), nil)
	}
	if err != nil {
		return nil, err
	}
	return NewTable(header, body)
}

// ForEachCSVRow streams CSV records to visit without retaining previous rows.
// It passes the header and fields to visit once per data record.
func ForEachCSVRow(reader io.Reader, delimiter rune, visit func(columns []string, row []string) error) error {
	if visit == nil {
		return fmt.Errorf("tabular: CSV row visitor is nil")
	}
	csvReader := csv.NewReader(reader)
	if delimiter != 0 {
		csvReader.Comma = delimiter
	}
	csvReader.FieldsPerRecord = -1
	header, err := csvReader.Read()
	if err != nil {
		return fmt.Errorf("tabular: read CSV header: %w", err)
	}
	if err := validateColumnNames(header); err != nil {
		return err
	}
	for record := 2; ; record++ {
		row, readErr := csvReader.Read()
		if readErr == io.EOF {
			return nil
		}
		if readErr != nil {
			return fmt.Errorf("tabular: CSV record %d: %w", record, readErr)
		}
		if len(row) != len(header) {
			return fmt.Errorf("tabular: CSV record %d has %d columns, want %d", record, len(row), len(header))
		}
		if err := visit(append([]string(nil), header...), row); err != nil {
			return fmt.Errorf("tabular: visit CSV record %d: %w", record, err)
		}
	}
}

// WriteCSVTable writes a table with its column names as the first record.
func WriteCSVTable(writer io.Writer, table *Table, delimiter rune) error {
	if table == nil || table.Data == nil {
		return fmt.Errorf("tabular: table is nil")
	}
	csvWriter := csv.NewWriter(writer)
	if delimiter != 0 {
		csvWriter.Comma = delimiter
	}
	if err := csvWriter.Write(table.Columns); err != nil {
		return fmt.Errorf("tabular: write CSV header: %w", err)
	}
	for row := 0; row < table.Data.Rows(); row++ {
		fields := make([]string, table.Data.Cols())
		for col := range fields {
			fields[col] = strconv.FormatFloat(table.Data.At(row, col), 'g', -1, 64)
		}
		if err := csvWriter.Write(fields); err != nil {
			return fmt.Errorf("tabular: write CSV row %d: %w", row+1, err)
		}
	}
	csvWriter.Flush()
	return csvWriter.Error()
}

// WriteCSV writes matrix values as decimal CSV records, without a header.
func WriteCSV(writer io.Writer, data *matrix.Dense, delimiter rune) error {
	csvWriter := csv.NewWriter(writer)
	if delimiter != 0 {
		csvWriter.Comma = delimiter
	}
	for row := 0; row < data.Rows(); row++ {
		fields := make([]string, data.Cols())
		for col := range fields {
			fields[col] = strconv.FormatFloat(data.At(row, col), 'g', -1, 64)
		}
		if err := csvWriter.Write(fields); err != nil {
			return fmt.Errorf("tabular: write CSV row %d: %w", row+1, err)
		}
	}
	csvWriter.Flush()
	return csvWriter.Error()
}
