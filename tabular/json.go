package tabular

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"go-learn/matrix"
)

// ReadJSON reads a JSON array of numeric row arrays into a dense matrix.
// Empty input arrays produce a (0, 0) matrix. Nulls and heterogeneous rows are errors.
func ReadJSON(reader io.Reader) (*matrix.Dense, error) {
	decoder := json.NewDecoder(reader)
	decoder.UseNumber()
	var rows [][]any
	if err := decoder.Decode(&rows); err != nil {
		return nil, fmt.Errorf("tabular: decode JSON matrix: %w", err)
	}
	if rows == nil {
		return nil, fmt.Errorf("tabular: JSON matrix must be an array")
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("tabular: JSON contains trailing data")
		}
		return nil, fmt.Errorf("tabular: trailing JSON data: %w", err)
	}
	values := make([][]float64, len(rows))
	for rowIndex, row := range rows {
		if row == nil {
			return nil, fmt.Errorf("tabular: JSON row %d must be an array", rowIndex+1)
		}
		values[rowIndex] = make([]float64, len(row))
		for colIndex, number := range row {
			numeric, ok := number.(json.Number)
			if !ok {
				return nil, fmt.Errorf("tabular: JSON row %d column %d must be numeric", rowIndex+1, colIndex+1)
			}
			value, err := numeric.Float64()
			if err != nil {
				return nil, fmt.Errorf("tabular: JSON row %d column %d: %w", rowIndex+1, colIndex+1, err)
			}
			values[rowIndex][colIndex] = value
		}
	}
	data, err := matrix.NewDenseFromRows(values)
	if err != nil {
		return nil, fmt.Errorf("tabular: JSON matrix: %w", err)
	}
	return data, nil
}

// WriteJSON writes a matrix as a JSON array of numeric row arrays.
func WriteJSON(writer io.Writer, data *matrix.Dense) error {
	rows := make([][]float64, data.Rows())
	for row := range rows {
		rows[row] = make([]float64, data.Cols())
		for col := range rows[row] {
			rows[row][col] = data.At(row, col)
		}
	}
	encoder := json.NewEncoder(writer)
	if err := encoder.Encode(rows); err != nil {
		return fmt.Errorf("tabular: encode JSON matrix: %w", err)
	}
	return nil
}

// ReadJSONTable reads {"columns":[...],"data":[[...],...]} numeric tables.
func ReadJSONTable(reader io.Reader) (*Table, error) {
	var document struct {
		Columns []string        `json:"columns"`
		Data    json.RawMessage `json:"data"`
	}
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("tabular: decode JSON table: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return nil, fmt.Errorf("tabular: JSON table contains trailing data")
	}
	values, err := ReadJSON(bytes.NewReader(document.Data))
	if err != nil {
		return nil, fmt.Errorf("tabular: JSON table data: %w", err)
	}
	if values.Rows() == 0 {
		values, err = matrix.NewDense(0, len(document.Columns), nil)
	}
	if err != nil {
		return nil, err
	}
	return NewTable(document.Columns, values)
}

// WriteJSONTable writes a table as {"columns":[...],"data":[[...],...]}.
func WriteJSONTable(writer io.Writer, table *Table) error {
	if table == nil || table.Data == nil {
		return fmt.Errorf("tabular: table is nil")
	}
	rows := make([][]float64, table.Data.Rows())
	for row := range rows {
		rows[row] = make([]float64, table.Data.Cols())
		for col := range rows[row] {
			rows[row][col] = table.Data.At(row, col)
		}
	}
	if err := json.NewEncoder(writer).Encode(struct {
		Columns []string    `json:"columns"`
		Data    [][]float64 `json:"data"`
	}{table.Columns, rows}); err != nil {
		return fmt.Errorf("tabular: encode JSON table: %w", err)
	}
	return nil
}
