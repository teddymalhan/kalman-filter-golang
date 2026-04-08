// Package kalman provides a linear (multivariate) Kalman Filter.
//
// The filter operates in the standard predict → update cycle described at
// https://www.kalmanfilter.net and implements:
//
//	State extrapolation:     x̂[n+1,n] = F·x̂[n,n] + G·u[n]
//	Covariance extrapolation: P[n+1,n] = F·P[n,n]·Fᵀ + Q
//	Kalman gain:              K[n]     = P[n,n-1]·Hᵀ·(H·P[n,n-1]·Hᵀ + R[n])⁻¹
//	State update:             x̂[n,n]  = x̂[n,n-1] + K[n]·(z[n] - H·x̂[n,n-1])
//	Covariance update (Joseph form):
//	                          P[n,n]   = (I - K[n]·H)·P[n,n-1]·(I - K[n]·H)ᵀ + K[n]·R[n]·K[n]ᵀ
package kalman

import (
	"fmt"

	"github.com/teddymalhan/kalman-filter-golang/matrix"
)

// Filter holds the Kalman Filter state.
type Filter struct {
	// State transition matrix (n×n)
	F *matrix.Matrix
	// Input (control) transition matrix (n×l); nil if no control input
	G *matrix.Matrix
	// Observation matrix (m×n)
	H *matrix.Matrix
	// Process noise covariance (n×n)
	Q *matrix.Matrix

	// Current state estimate x̂[n,n]  (n×1 column vector)
	X *matrix.Matrix
	// Current estimate covariance P[n,n] (n×n)
	P *matrix.Matrix

	stateDim int // n
	measDim  int // m
}

// New creates a new Kalman Filter.
//
//   - F: state transition matrix (n×n)
//   - H: observation matrix (m×n)
//   - Q: process noise covariance (n×n)
//   - x0: initial state estimate (n×1)
//   - P0: initial estimate covariance (n×n)
func New(F, H, Q, x0, P0 *matrix.Matrix) *Filter {
	n := F.Rows
	m := H.Rows
	return &Filter{
		F:        F,
		H:        H,
		Q:        Q,
		X:        x0.Clone(),
		P:        P0.Clone(),
		stateDim: n,
		measDim:  m,
	}
}

// Predict propagates the state and covariance one time step.
// Pass a nil u (or zero vector) when there is no control input.
func (f *Filter) Predict(u *matrix.Matrix) {
	// x̂[n+1,n] = F·x̂[n,n] + G·u[n]
	f.X = matrix.Mul(f.F, f.X)
	if f.G != nil && u != nil {
		f.X = matrix.Add(f.X, matrix.Mul(f.G, u))
	}

	// P[n+1,n] = F·P[n,n]·Fᵀ + Q
	f.P = matrix.Add(matrix.Mul(matrix.Mul(f.F, f.P), f.F.T()), f.Q)
}

// Update incorporates measurement z (m×1) with measurement noise covariance R (m×m).
func (f *Filter) Update(z, R *matrix.Matrix) error {
	if z.Rows != f.measDim || z.Cols != 1 {
		return fmt.Errorf("kalman: measurement must be %dx1, got %dx%d", f.measDim, z.Rows, z.Cols)
	}
	if R.Rows != f.measDim || R.Cols != f.measDim {
		return fmt.Errorf("kalman: R must be %dx%d, got %dx%d", f.measDim, f.measDim, R.Rows, R.Cols)
	}

	Ht := f.H.T()

	// S = H·P·Hᵀ + R  (innovation covariance)
	S := matrix.Add(matrix.Mul(matrix.Mul(f.H, f.P), Ht), R)

	// K = P·Hᵀ·S⁻¹
	K := matrix.Mul(matrix.Mul(f.P, Ht), matrix.Inv(S))

	// innovation y = z - H·x̂
	y := matrix.Sub(z, matrix.Mul(f.H, f.X))

	// x̂[n,n] = x̂[n,n-1] + K·y
	f.X = matrix.Add(f.X, matrix.Mul(K, y))

	// Covariance update — Joseph form for numerical stability:
	// P[n,n] = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ
	n := f.stateDim
	I := matrix.Identity(n)
	IKH := matrix.Sub(I, matrix.Mul(K, f.H))
	f.P = matrix.Add(
		matrix.Mul(matrix.Mul(IKH, f.P), IKH.T()),
		matrix.Mul(matrix.Mul(K, R), K.T()),
	)

	return nil
}

// State returns the current state estimate (a copy).
func (f *Filter) State() *matrix.Matrix { return f.X.Clone() }

// Covariance returns the current estimate covariance (a copy).
func (f *Filter) Covariance() *matrix.Matrix { return f.P.Clone() }
