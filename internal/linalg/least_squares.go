// Package linalg contains the small set of dense numerical routines required
// by the public estimators.
package linalg

import (
	"errors"
	"math"
)

var ErrRankDeficient = errors.New("linalg: matrix is rank deficient")

// LeastSquares solves min ||Ax-b||₂ using Householder QR with column
// pivoting. It rejects numerical rank deficiency using a relative diagonal
// threshold of 1e-12.
func LeastSquares(a []float64, rows, cols int, b []float64) ([]float64, error) {
	if rows < 1 || cols < 1 || len(a) != rows*cols || len(b) != rows || cols > rows {
		return nil, errors.New("linalg: invalid least-squares dimensions")
	}
	qr := append([]float64(nil), a...)
	rhs := append([]float64(nil), b...)
	permutation := make([]int, cols)
	for j := range permutation {
		permutation[j] = j
	}
	var firstDiagonal float64

	for k := 0; k < cols; k++ {
		pivot, pivotNorm := k, -1.0
		for j := k; j < cols; j++ {
			norm := 0.0
			for i := k; i < rows; i++ {
				norm = math.Hypot(norm, qr[i*cols+j])
			}
			if norm > pivotNorm {
				pivot, pivotNorm = j, norm
			}
		}
		if pivot != k {
			for i := 0; i < rows; i++ {
				qr[i*cols+k], qr[i*cols+pivot] = qr[i*cols+pivot], qr[i*cols+k]
			}
			permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
		}

		norm := pivotNorm
		if math.IsNaN(norm) || math.IsInf(norm, 0) {
			return nil, errors.New("linalg: non-finite value during QR factorization")
		}
		if k == 0 {
			firstDiagonal = norm
		}
		if norm == 0 || norm <= firstDiagonal*1e-12 {
			return nil, ErrRankDeficient
		}
		alpha := qr[k*cols+k]
		if alpha >= 0 {
			alpha = -norm
		} else {
			alpha = norm
		}
		v := make([]float64, rows-k)
		for i := range v {
			v[i] = qr[(k+i)*cols+k]
		}
		v[0] -= alpha
		vNorm := 0.0
		for _, value := range v {
			vNorm = math.Hypot(vNorm, value)
		}
		if vNorm == 0 {
			return nil, ErrRankDeficient
		}
		for i := range v {
			v[i] /= vNorm
		}
		const beta = 2.0
		for j := k; j < cols; j++ {
			dot := 0.0
			for i := range v {
				dot += v[i] * qr[(k+i)*cols+j]
			}
			factor := beta * dot
			for i := range v {
				qr[(k+i)*cols+j] -= factor * v[i]
			}
		}
		dot := 0.0
		for i := range v {
			dot += v[i] * rhs[k+i]
		}
		factor := beta * dot
		for i := range v {
			rhs[k+i] -= factor * v[i]
		}
		qr[k*cols+k] = alpha
		for i := k + 1; i < rows; i++ {
			qr[i*cols+k] = 0
		}
	}

	permuted := make([]float64, cols)
	for i := cols - 1; i >= 0; i-- {
		value := rhs[i]
		for j := i + 1; j < cols; j++ {
			value -= qr[i*cols+j] * permuted[j]
		}
		permuted[i] = value / qr[i*cols+i]
	}
	coefficients := make([]float64, cols)
	for i, original := range permutation {
		coefficients[original] = permuted[i]
	}
	return coefficients, nil
}
