package pipeline_test

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"go-learn/linear_model"
	"go-learn/matrix"
	"go-learn/pipeline"
	"go-learn/preprocessing"
)

func TestClassificationPipelineLifecycle(t *testing.T) {
	x, _ := matrix.NewDenseFromRows([][]float64{{-2}, {-1}, {1}, {2}})
	y := []int{0, 0, 1, 1}
	created := 0
	factory := func() pipeline.Classifier { created++; return linear_model.NewLogisticRegression() }
	p, err := pipeline.NewClassificationPipeline([]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}}, factory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.Predict(x); !errors.Is(err, pipeline.ErrNotFitted) {
		t.Fatal(err)
	}
	if _, err = p.PredictProba(x); !errors.Is(err, pipeline.ErrNotFitted) {
		t.Fatal(err)
	}
	if err = p.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	before, _ := p.Predict(x)
	if err = p.Fit(x, y); err != nil || created != 2 {
		t.Fatal("fresh estimator required", err, created)
	}
	score, err := p.Score(x, y)
	if err != nil || score != 1 {
		t.Fatal(score, err)
	}
	labels := p.Classes()
	labels[0] = 999
	if !reflect.DeepEqual(p.Classes(), []int{0, 1}) {
		t.Fatal("aliased classes")
	}
	bad, _ := matrix.NewDenseFromRows([][]float64{{1, 2}})
	if _, err = p.Predict(bad); err == nil {
		t.Fatal("invalid dimensions")
	}
	if _, err = p.PredictProba(nil); err == nil {
		t.Fatal("nil input")
	}
	if err = p.Fit(x, []int{1}); err == nil {
		t.Fatal("invalid target")
	}
	after, _ := p.Predict(x)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("failed fit replaced state")
	}
}

func TestClassificationPipelineFactories(t *testing.T) {
	if _, err := pipeline.NewClassificationPipeline(nil, nil); err == nil {
		t.Fatal("nil factory")
	}
	factory := func() pipeline.Classifier { return nil }
	step := pipeline.Step{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}
	if _, err := pipeline.NewClassificationPipeline([]pipeline.Step{step, step}, factory); err == nil {
		t.Fatal("duplicate name")
	}
	if _, err := pipeline.NewClassificationPipeline([]pipeline.Step{{}}, factory); err == nil {
		t.Fatal("empty step")
	}
	p, _ := pipeline.NewClassificationPipeline(nil, factory)
	x, _ := matrix.NewDenseFromRows([][]float64{{0}, {1}})
	if err := p.Fit(x, []int{0, 1}); err == nil {
		t.Fatal("nil classifier accepted")
	}
}

func ExampleClassificationPipeline() {
	model, err := pipeline.NewClassificationPipeline(
		[]pipeline.Step{{Name: "scale", New: func() pipeline.Transformer { return preprocessing.NewStandardScaler() }}},
		func() pipeline.Classifier { return linear_model.NewLogisticRegression() },
	)
	if err != nil {
		panic(err)
	}
	train, _ := matrix.NewDenseFromRows([][]float64{{-2}, {-1}, {1}, {2}})
	if err = model.Fit(train, []int{0, 0, 1, 1}); err != nil {
		panic(err)
	}
	test, _ := matrix.NewDenseFromRows([][]float64{{-3}, {3}})
	labels, err := model.Predict(test)
	if err != nil {
		panic(err)
	}
	fmt.Println(labels, model.Classes())
	// Output: [0 1] [0 1]
}
