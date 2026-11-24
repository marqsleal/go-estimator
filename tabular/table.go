package tabular

import (
	"fmt"
	"math"

	"go-learn/matrix"
)

// Table stores numeric columns with stable names and row order.
type Table struct {
	Columns []string
	Data    *matrix.Dense
	Rows    [][]any
}

// NewTypedTable constructs a table with string, float64, int64, bool, or nil cells.
func NewTypedTable(columns []string, rows [][]any) (*Table, error) {
	if err := validateColumnNames(columns); err != nil {
		return nil, err
	}
	copyRows := make([][]any, len(rows))
	for i, row := range rows {
		if len(row) != len(columns) {
			return nil, fmt.Errorf("tabular: row %d has %d cells, want %d", i+1, len(row), len(columns))
		}
		copyRows[i] = append([]any(nil), row...)
		for j, value := range row {
			switch value.(type) {
			case nil, string, float64, int64, bool:
			default:
				return nil, fmt.Errorf("tabular: unsupported value type %T at row %d column %q", value, i+1, columns[j])
			}
		}
	}
	return &Table{Columns: append([]string(nil), columns...), Rows: copyRows}, nil
}

func validateColumnNames(columns []string) error {
	seen := make(map[string]struct{}, len(columns))
	for i, name := range columns {
		if name == "" {
			return fmt.Errorf("tabular: column %d has an empty name", i+1)
		}
		if _, ok := seen[name]; ok {
			return fmt.Errorf("tabular: duplicate column name %q", name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

// SplitInt returns features and a classification target, requiring integral values in int range.
func (t *Table) SplitInt(features []string, target string) (*matrix.Dense, []int, error) {
	x, err := t.Select(features...)
	if err != nil {
		return nil, nil, err
	}
	targets, err := t.Select(target)
	if err != nil {
		return nil, nil, err
	}
	y := make([]int, targets.Rows())
	for row := range y {
		value := targets.At(row, 0)
		if math.Trunc(value) != value || value < float64(math.MinInt) || value >= -float64(math.MinInt) {
			return nil, nil, fmt.Errorf("tabular: target %q row %d is not an integer", target, row+1)
		}
		y[row] = int(value)
	}
	return x, y, nil
}

// NewTable validates and copies the column names. Matrix values are already owned by Dense.
func NewTable(columns []string, data *matrix.Dense) (*Table, error) {
	if data == nil || len(columns) != data.Cols() {
		return nil, fmt.Errorf("tabular: %d column names for matrix with %d columns", len(columns), cols(data))
	}
	if err := validateColumnNames(columns); err != nil {
		return nil, err
	}
	return &Table{Columns: append([]string(nil), columns...), Data: data.Clone()}, nil
}

func cols(data *matrix.Dense) int {
	if data == nil {
		return 0
	}
	return data.Cols()
}

// Select returns a matrix whose columns follow the requested name order.
func (t *Table) Select(names ...string) (*matrix.Dense, error) {
	if t == nil {
		return nil, fmt.Errorf("tabular: table is nil")
	}
	indices := make([]int, len(names))
	for i, name := range names {
		indices[i] = -1
		for j, column := range t.Columns {
			if column == name {
				indices[i] = j
				break
			}
		}
		if indices[i] < 0 {
			return nil, fmt.Errorf("tabular: unknown column %q", name)
		}
	}
	rowCount := len(t.Rows)
	if t.Data != nil {
		rowCount = t.Data.Rows()
	}
	values := make([]float64, 0, rowCount*len(names))
	for row := 0; row < rowCount; row++ {
		for _, col := range indices {
			if t.Data != nil {
				values = append(values, t.Data.At(row, col))
				continue
			}
			if len(t.Rows[row]) != len(t.Columns) {
				return nil, fmt.Errorf("tabular: row %d has invalid width", row+1)
			}
			var value float64
			switch cell := t.Rows[row][col].(type) {
			case float64:
				value = cell
			case int64:
				value = float64(cell)
				if value >= 9223372036854775808.0 || int64(value) != cell {
					return nil, fmt.Errorf("tabular: row %d column %q loses integer precision", row+1, t.Columns[col])
				}
			default:
				return nil, fmt.Errorf("tabular: row %d column %q must be numeric", row+1, t.Columns[col])
			}
			values = append(values, value)
		}
	}
	return matrix.NewDense(rowCount, len(names), values)
}

// Split returns feature columns and one float target column, preserving row alignment.
func (t *Table) Split(features []string, target string) (*matrix.Dense, []float64, error) {
	x, err := t.Select(features...)
	if err != nil {
		return nil, nil, err
	}
	yMatrix, err := t.Select(target)
	if err != nil {
		return nil, nil, err
	}
	y := make([]float64, yMatrix.Rows())
	for row := range y {
		y[row] = yMatrix.At(row, 0)
	}
	return x, y, nil
}

// SplitTargets selects feature and target matrices in the requested column order.
func (t *Table) SplitTargets(features, targets []string) (*matrix.Dense, *matrix.Dense, error) {
	x, err := t.Select(features...)
	if err != nil {
		return nil, nil, err
	}
	y, err := t.Select(targets...)
	if err != nil {
		return nil, nil, err
	}
	return x, y, nil
}
