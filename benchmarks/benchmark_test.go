package benchmarks_test

import (
	"fmt"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/metrics"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func BenchmarkLinearRegressionFit(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{{128, 4}, {10_000, 32}} {
		b.Run(fmt.Sprintf("%dx%d", shape.rows, shape.cols), func(b *testing.B) {
			x, y := benchmarkData(shape.rows, shape.cols)
			checkLinearRegression(b, x, y)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				model := linear_model.NewLinearRegression()
				if err := model.Fit(x, y); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLinearRegressionPredict(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{{128, 4}, {10_000, 32}} {
		b.Run(fmt.Sprintf("%dx%d", shape.rows, shape.cols), func(b *testing.B) {
			x, y := benchmarkData(shape.rows, shape.cols)
			model := linear_model.NewLinearRegression()
			if err := model.Fit(x, y); err != nil {
				b.Fatal(err)
			}
			checkPredictions(b, model, x, y)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := model.Predict(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkStandardScaler(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{{128, 4}, {10_000, 32}} {
		b.Run(fmt.Sprintf("FitTransform/%dx%d", shape.rows, shape.cols), func(b *testing.B) {
			x, _ := benchmarkData(shape.rows, shape.cols)
			if _, err := preprocessing.NewStandardScaler().FitTransform(x); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				scaler := preprocessing.NewStandardScaler()
				if _, err := scaler.FitTransform(x); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("Fit/%dx%d", shape.rows, shape.cols), func(b *testing.B) {
			x, _ := benchmarkData(shape.rows, shape.cols)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := preprocessing.NewStandardScaler().Fit(x); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("Transform/%dx%d", shape.rows, shape.cols), func(b *testing.B) {
			x, _ := benchmarkData(shape.rows, shape.cols)
			scaler := preprocessing.NewStandardScaler()
			if err := scaler.Fit(x); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := scaler.Transform(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkRegressionMetrics(b *testing.B) {
	for _, rows := range []int{128, 10_000} {
		b.Run(fmt.Sprintf("%d", rows), func(b *testing.B) {
			_, y := benchmarkData(rows, 4)
			predictions := make([]float64, len(y))
			for i, value := range y {
				predictions[i] = value + 0.1
			}
			checks := []struct {
				name string
				fn   func([]float64, []float64) (float64, error)
			}{
				{name: "MAE", fn: metrics.MeanAbsoluteError},
				{name: "MSE", fn: metrics.MeanSquaredError},
				{name: "R2", fn: metrics.R2Score},
			}
			for _, check := range checks {
				b.Run(check.name, func(b *testing.B) {
					if _, err := check.fn(y, predictions); err != nil {
						b.Fatal(err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err := check.fn(y, predictions); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}

func BenchmarkRegressionPipelineFitPredict(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{{128, 4}, {10_000, 32}} {
		b.Run(fmt.Sprintf("%dx%d", shape.rows, shape.cols), func(b *testing.B) {
			x, y := benchmarkData(shape.rows, shape.cols)
			preflight, err := pipeline.NewRegressionPipeline(
				[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
				func() pipeline.Regressor { return linear_model.NewLinearRegression() },
			)
			if err != nil {
				b.Fatal(err)
			}
			if err := preflight.Fit(x, y); err != nil {
				b.Fatal(err)
			}
			predictions, err := preflight.Predict(x)
			if err != nil {
				b.Fatal(err)
			}
			for i, prediction := range predictions {
				if difference := prediction - y[i]; difference > 1e-8 || difference < -1e-8 {
					b.Fatalf("pipeline prediction[%d] = %.12g, want %.12g", i, prediction, y[i])
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				model, err := pipeline.NewRegressionPipeline(
					[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
					func() pipeline.Regressor { return linear_model.NewLinearRegression() },
				)
				if err != nil {
					b.Fatal(err)
				}
				if err := model.Fit(x, y); err != nil {
					b.Fatal(err)
				}
				if _, err := model.Predict(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func checkLinearRegression(b *testing.B, x *matrix.Dense, y []float64) {
	b.Helper()
	model := linear_model.NewLinearRegression()
	if err := model.Fit(x, y); err != nil {
		b.Fatal(err)
	}
	checkPredictions(b, model, x, y)
}

func checkPredictions(b *testing.B, model *linear_model.LinearRegression, x *matrix.Dense, y []float64) {
	b.Helper()
	predictions, err := model.Predict(x)
	if err != nil {
		b.Fatal(err)
	}
	for i, prediction := range predictions {
		if difference := prediction - y[i]; difference > 1e-8 || difference < -1e-8 {
			b.Fatalf("prediction[%d] = %.12g, want %.12g", i, prediction, y[i])
		}
	}
}

func benchmarkData(rows, cols int) (*matrix.Dense, []float64) {
	values := make([]float64, rows*cols)
	targets := make([]float64, rows)
	for i := 0; i < rows; i++ {
		targets[i] = 0.5
		for j := 0; j < cols; j++ {
			mixed := (i * 73856093) ^ (j * 19349663) ^ ((i + 1) * (j + 11) * 83492791)
			value := float64(mixed%1009) / 100
			values[i*cols+j] = value
			targets[i] += value * float64(j+1) / float64(cols)
		}
	}
	x, err := matrix.NewDense(rows, cols, values)
	if err != nil {
		panic(err)
	}
	return x, targets
}
