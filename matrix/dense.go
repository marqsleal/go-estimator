// Package matrix provides dense, row-major matrices of float64 values.
package matrix

import (
	"errors"
	"fmt"
)

var (
	// ErrInvalidDimensions indicates that a matrix dimension is negative or
	// that the requested shape cannot be represented by a Go slice.
	ErrInvalidDimensions = errors.New("matrix: invalid dimensions")
	// ErrDataSize indicates that the number of values does not match the shape.
	ErrDataSize = errors.New("matrix: data size does not match dimensions")
	// ErrRaggedRows indicates that rows have different lengths.
	ErrRaggedRows = errors.New("matrix: rows have different lengths")
	// ErrIndexOutOfRange indicates that a requested matrix index is invalid.
	ErrIndexOutOfRange = errors.New("matrix: index out of range")
)

// Dense stores a two-dimensional matrix in row-major order. Its zero value is
// a valid matrix with shape (0, 0). Values supplied to constructors are copied.
type Dense struct {
	rows int
	cols int
	data []float64
}

// NewDense creates a rows-by-cols matrix and copies values in row-major order.
// Zero-sized dimensions are allowed. Negative dimensions, integer overflow,
// and a mismatched number of values return an error.
func NewDense(rows, cols int, values []float64) (*Dense, error) {
	size, err := elementCount(rows, cols)
	if err != nil {
		return nil, err
	}
	if len(values) != size {
		return nil, fmt.Errorf("%w: got %d values for shape (%d, %d)", ErrDataSize, len(values), rows, cols)
	}

	data := make([]float64, size)
	copy(data, values)
	return &Dense{rows: rows, cols: cols, data: data}, nil
}

// NewDenseFromRows creates a matrix by copying a slice of rows. An empty input
// creates a matrix with shape (0, 0).
func NewDenseFromRows(rows [][]float64) (*Dense, error) {
	if len(rows) == 0 {
		return &Dense{}, nil
	}

	cols := len(rows[0])
	size, err := elementCount(len(rows), cols)
	if err != nil {
		return nil, err
	}
	data := make([]float64, size)
	for i, row := range rows {
		if len(row) != cols {
			return nil, fmt.Errorf("%w: row %d has %d values, want %d", ErrRaggedRows, i, len(row), cols)
		}
		copy(data[i*cols:(i+1)*cols], row)
	}
	return &Dense{rows: len(rows), cols: cols, data: data}, nil
}

// Rows returns the number of rows.
func (m *Dense) Rows() int { return m.rows }

// Cols returns the number of columns.
func (m *Dense) Cols() int { return m.cols }

// Shape returns the number of rows and columns.
func (m *Dense) Shape() (rows, cols int) { return m.rows, m.cols }

// At returns the value at row i and column j. It panics if either index is out
// of range, matching Go's built-in slice indexing behavior.
func (m *Dense) At(i, j int) float64 {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(ErrIndexOutOfRange)
	}
	return m.data[i*m.cols+j]
}

// Set changes the value at row i and column j. It returns an error if either
// index is out of range.
func (m *Dense) Set(i, j int, value float64) error {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		return fmt.Errorf("%w: (%d, %d) for shape (%d, %d)", ErrIndexOutOfRange, i, j, m.rows, m.cols)
	}
	m.data[i*m.cols+j] = value
	return nil
}

// Values returns a copy of the matrix values in row-major order.
func (m *Dense) Values() []float64 {
	values := make([]float64, len(m.data))
	copy(values, m.data)
	return values
}

// Row returns a copy of row i.
func (m *Dense) Row(i int) ([]float64, error) {
	if i < 0 || i >= m.rows {
		return nil, fmt.Errorf("%w: row %d for shape (%d, %d)", ErrIndexOutOfRange, i, m.rows, m.cols)
	}
	start := i * m.cols
	row := make([]float64, m.cols)
	copy(row, m.data[start:start+m.cols])
	return row, nil
}

// Clone returns an independent copy of m.
func (m *Dense) Clone() *Dense {
	data := make([]float64, len(m.data))
	copy(data, m.data)
	return &Dense{rows: m.rows, cols: m.cols, data: data}
}

func elementCount(rows, cols int) (int, error) {
	if rows < 0 || cols < 0 {
		return 0, fmt.Errorf("%w: (%d, %d)", ErrInvalidDimensions, rows, cols)
	}
	if rows != 0 && cols > int(^uint(0)>>1)/rows {
		return 0, fmt.Errorf("%w: (%d, %d) overflows int", ErrInvalidDimensions, rows, cols)
	}
	return rows * cols, nil
}
