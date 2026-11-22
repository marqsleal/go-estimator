package model_selection

import (
	"fmt"
	mathrand "math/rand"
)

// Fold contains the training and validation indices for one partition.
type Fold struct {
	TrainIndices []int
	TestIndices  []int
}

// KFold partitions n samples into k folds. The first remainder folds contain
// one extra sample. Shuffling is disabled unless requested.
func KFold(n, k int, shuffle bool, randomState *uint64) ([]Fold, error) {
	if n < 2 || k < 2 || k > n {
		return nil, fmt.Errorf("%w: KFold requires 2 <= k <= n, got k=%d n=%d", ErrInvalidSplit, k, n)
	}
	if randomState != nil && *randomState > uint64(^uint32(0)) {
		return nil, fmt.Errorf("%w: random_state %d is outside [0, 2^32-1]", ErrInvalidSplit, *randomState)
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	if shuffle {
		seed, err := randomSeed(randomState)
		if err != nil {
			return nil, fmt.Errorf("model_selection: obtain random seed: %w", err)
		}
		mathrand.New(mathrand.NewSource(int64(seed))).Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	}
	folds := make([]Fold, k)
	start := 0
	base, remainder := n/k, n%k
	for fold := range folds {
		size := base
		if fold < remainder {
			size++
		}
		end := start + size
		test := append([]int(nil), indices[start:end]...)
		train := make([]int, 0, n-size)
		train = append(train, indices[:start]...)
		train = append(train, indices[end:]...)
		folds[fold] = Fold{TrainIndices: train, TestIndices: test}
		start = end
	}
	return folds, nil
}
