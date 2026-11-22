// Package model_selection provides utilities for splitting datasets.
package model_selection

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	mathrand "math/rand"

	"go-learn/matrix"
)

var ErrInvalidSplit = errors.New("model_selection: invalid train/test split")

// SplitOption configures TrainTestSplit.
type SplitOption func(*splitConfig)

type splitConfig struct {
	testSize float64
	shuffle  bool
	seed     *uint64
}

// WithTestSize sets the fraction of samples assigned to the test set.
func WithTestSize(fraction float64) SplitOption {
	return func(c *splitConfig) { c.testSize = fraction }
}

// WithShuffle controls whether the sample indices are shuffled. The default is true.
func WithShuffle(shuffle bool) SplitOption {
	return func(c *splitConfig) { c.shuffle = shuffle }
}

// WithRandomState sets a deterministic seed in the range [0, 2^32-1].
func WithRandomState(seed uint64) SplitOption {
	return func(c *splitConfig) { c.seed = &seed }
}

// Split contains the aligned train and test partitions and their source indices.
type Split struct {
	XTrain, XTest *matrix.Dense
	YTrain, YTest []float64
	TrainIndices  []int
	TestIndices   []int
}

// TrainTestSplit partitions x and y, preserving their alignment. The default
// test fraction is 0.25 and shuffling is enabled.
func TrainTestSplit(x *matrix.Dense, y []float64, options ...SplitOption) (*Split, error) {
	config := splitConfig{testSize: 0.25, shuffle: true}
	for _, option := range options {
		if option != nil {
			option(&config)
		}
	}
	if x == nil || x.Rows() != len(y) || x.Rows() < 2 || config.testSize <= 0 || config.testSize >= 1 {
		return nil, fmt.Errorf("%w: rows=%d targets=%d test_size=%g", ErrInvalidSplit, rows(x), len(y), config.testSize)
	}
	if config.seed != nil && *config.seed > uint64(^uint32(0)) {
		return nil, fmt.Errorf("%w: random_state %d is outside [0, 2^32-1]", ErrInvalidSplit, *config.seed)
	}
	testCount := int(float64(x.Rows()) * config.testSize)
	if float64(testCount) < float64(x.Rows())*config.testSize {
		testCount++
	}
	if testCount < 1 || testCount >= x.Rows() {
		return nil, fmt.Errorf("%w: test set would contain %d of %d samples", ErrInvalidSplit, testCount, x.Rows())
	}
	indices := make([]int, x.Rows())
	for i := range indices {
		indices[i] = i
	}
	if config.shuffle {
		seed, err := randomSeed(config.seed)
		if err != nil {
			return nil, fmt.Errorf("model_selection: obtain random seed: %w", err)
		}
		generator := mathrand.New(mathrand.NewSource(int64(seed)))
		generator.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	}
	trainIndices := append([]int(nil), indices[:len(indices)-testCount]...)
	testIndices := append([]int(nil), indices[len(indices)-testCount:]...)
	trainX, trainY, err := selectRows(x, y, trainIndices)
	if err != nil {
		return nil, err
	}
	testX, testY, err := selectRows(x, y, testIndices)
	if err != nil {
		return nil, err
	}
	return &Split{XTrain: trainX, XTest: testX, YTrain: trainY, YTest: testY, TrainIndices: trainIndices, TestIndices: testIndices}, nil
}

func randomSeed(seed *uint64) (uint64, error) {
	if seed != nil {
		return *seed, nil
	}
	var bytes [8]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(bytes[:]), nil
}

func selectRows(x *matrix.Dense, y []float64, indices []int) (*matrix.Dense, []float64, error) {
	values := make([]float64, 0, len(indices)*x.Cols())
	targets := make([]float64, len(indices))
	for i, index := range indices {
		row, err := x.Row(index)
		if err != nil {
			return nil, nil, err
		}
		values = append(values, row...)
		targets[i] = y[index]
	}
	selected, err := matrix.NewDense(len(indices), x.Cols(), values)
	return selected, targets, err
}

func rows(x *matrix.Dense) int {
	if x == nil {
		return 0
	}
	return x.Rows()
}
