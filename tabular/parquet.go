package tabular

import (
	"bytes"
	"fmt"
	"io"

	"github.com/parquet-go/parquet-go"
	"go-learn/matrix"
)

// orderedParquetGroup preserves the table's column order in the schema.
type orderedParquetGroup struct {
	parquet.Group
	fields []parquet.Field
}

func (g orderedParquetGroup) Fields() []parquet.Field { return g.fields }

// WriteParquet writes a numeric named table using required DOUBLE columns.
func WriteParquet(writer io.Writer, table *Table) error {
	if table == nil || table.Data == nil {
		return fmt.Errorf("tabular: table is nil")
	}
	group := make(parquet.Group, len(table.Columns))
	for _, name := range table.Columns {
		group[name] = parquet.Leaf(parquet.DoubleType)
	}
	fields := make([]parquet.Field, len(table.Columns))
	for _, field := range group.Fields() {
		for index, name := range table.Columns {
			if field.Name() == name {
				fields[index] = field
				break
			}
		}
	}
	schema := parquet.NewSchema("go_learn_table", orderedParquetGroup{Group: group, fields: fields})
	pw := parquet.NewWriter(writer, schema)
	for rowIndex := 0; rowIndex < table.Data.Rows(); rowIndex++ {
		row := make(parquet.Row, table.Data.Cols())
		for col := 0; col < table.Data.Cols(); col++ {
			row[col] = parquet.DoubleValue(table.Data.At(rowIndex, col)).Level(0, 0, col)
		}
		if _, err := pw.WriteRows([]parquet.Row{row}); err != nil {
			return fmt.Errorf("tabular: write Parquet row %d: %w", rowIndex+1, err)
		}
	}
	if err := pw.Close(); err != nil {
		return fmt.Errorf("tabular: close Parquet writer: %w", err)
	}
	return nil
}

// ReadParquet reads a complete numeric named Parquet table from an io.Reader.
func ReadParquet(reader io.Reader) (table *Table, resultErr error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			table = nil
			resultErr = fmt.Errorf("tabular: invalid Parquet input: %v", recovered)
		}
	}()
	content, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("tabular: read Parquet input: %w", err)
	}
	if len(content) == 0 {
		return nil, fmt.Errorf("tabular: empty Parquet input")
	}
	pr := parquet.NewReader(bytes.NewReader(content))
	schema := pr.Schema()
	columns := make([]string, len(schema.Columns()))
	for i, path := range schema.Columns() {
		if len(path) != 1 {
			return nil, fmt.Errorf("tabular: nested Parquet columns are unsupported")
		}
		columns[i] = path[0]
		leaf, ok := schema.Lookup(path...)
		if !ok || leaf.Node.Type().Kind() != parquet.Double || !leaf.Node.Required() {
			return nil, fmt.Errorf("tabular: Parquet column %q must be DOUBLE", columns[i])
		}
	}
	rows := make([]parquet.Row, 1024)
	values := make([]float64, 0)
	rowCount := 0
	for {
		n, readErr := pr.ReadRows(rows)
		for _, row := range rows[:n] {
			rowValues := make([]float64, len(columns))
			for _, value := range row {
				col := value.Column()
				if col < 0 || col >= len(columns) || value.IsNull() {
					return nil, fmt.Errorf("tabular: Parquet row %d has null or invalid column", rowCount+1)
				}
				rowValues[col] = value.Double()
			}
			if len(row) != len(columns) {
				return nil, fmt.Errorf("tabular: Parquet row %d has %d values, want %d", rowCount+1, len(row), len(columns))
			}
			values = append(values, rowValues...)
			rowCount++
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, fmt.Errorf("tabular: read Parquet rows: %w", readErr)
		}
	}
	if err := pr.Close(); err != nil {
		return nil, fmt.Errorf("tabular: close Parquet reader: %w", err)
	}
	data, err := matrix.NewDense(rowCount, len(columns), values)
	if err != nil {
		return nil, err
	}
	return NewTable(columns, data)
}
