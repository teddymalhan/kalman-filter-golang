// main.go — reproduces the two-iteration radar tracking example from
// https://www.kalmanfilter.net (single-page overview).
//
// System state: x = [range (m), velocity (m/s)]ᵀ
// Dynamic model: constant velocity
// Sampling interval: Δt = 5 s
// Process noise std-dev (acceleration): σ_a = 0.2 m/s²
package main

import (
	"fmt"

	"github.com/teddymalhan/kalman-filter-golang/kalman"
	"github.com/teddymalhan/kalman-filter-golang/matrix"
)

func main() {
	const dt = 5.0  // sampling interval [s]
	const sa = 0.2  // std-dev of random acceleration [m/s²]
	sa2 := sa * sa  // variance

	// ── State transition matrix F ─────────────────────────────────────────
	// x[k+1] = F·x[k]   with  F = [[1, Δt], [0, 1]]
	F := matrix.NewFromSlice(2, 2, []float64{
		1, dt,
		0, 1,
	})

	// ── Observation matrix H ──────────────────────────────────────────────
	// Radar measures both range and velocity → H = I₂
	H := matrix.Identity(2)

	// ── Process noise matrix Q ────────────────────────────────────────────
	// Derived from constant-velocity model with random acceleration noise:
	//   Q = [[Δt⁴/4, Δt³/2], [Δt³/2, Δt²]] · σ_a²
	Q := matrix.NewFromSlice(2, 2, []float64{
		(dt * dt * dt * dt / 4) * sa2, (dt * dt * dt / 2) * sa2,
		(dt * dt * dt / 2) * sa2, (dt * dt) * sa2,
	})

	// ── Iteration 0: initialization ───────────────────────────────────────
	// First measurement: range = 10 000 m, velocity = 200 m/s
	z0 := matrix.NewFromSlice(2, 1, []float64{10_000, 200})

	// Measurement noise covariance: σ_r = 4 m, σ_v = 0.5 m/s
	R0 := matrix.NewFromSlice(2, 2, []float64{
		16, 0,
		0, 0.25,
	})

	// Initialise state and covariance directly from the first measurement.
	x0 := z0.Clone()
	P0 := R0.Clone()

	fmt.Println("═══════════════════════════════════════════════")
	fmt.Println("  Kalman Filter — Radar Tracking Example")
	fmt.Println("  (KalmanFilter.NET single-page walkthrough)")
	fmt.Println("═══════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("── Iteration 0: Initialization ─────────────────")
	x0.Print("Initial state estimate x̂[0,0]  (range m | vel m/s)")
	P0.Print("Initial covariance     P[0,0]")

	// Create the filter.
	kf := kalman.New(F, H, Q, x0, P0)

	// ── Prediction step: t0 → t1 ──────────────────────────────────────────
	kf.Predict(nil)

	fmt.Println()
	fmt.Println("── Prediction: t0 → t1 ─────────────────────────")
	kf.State().Print("Predicted state x̂[1,0]")
	kf.Covariance().Print("Predicted covariance P[1,0]")

	// ── Iteration 1: update with second measurement ───────────────────────
	// Second measurement (noisier): range = 11 020 m, velocity = 202 m/s
	z1 := matrix.NewFromSlice(2, 1, []float64{11_020, 202})

	// Measurement noise covariance: σ_r = 6 m, σ_v = 1.5 m/s
	R1 := matrix.NewFromSlice(2, 2, []float64{
		36, 0,
		0, 2.25,
	})

	fmt.Println()
	fmt.Println("── Iteration 1: Update ──────────────────────────")
	z1.Print("Measurement z[1]  (range m | vel m/s)")
	R1.Print("Measurement covariance R[1]")

	if err := kf.Update(z1, R1); err != nil {
		fmt.Println("Update error:", err)
		return
	}

	kf.State().Print("Updated state estimate x̂[1,1]")
	kf.Covariance().Print("Updated covariance P[1,1]")

	// ── Prediction step: t1 → t2 ──────────────────────────────────────────
	kf.Predict(nil)

	fmt.Println()
	fmt.Println("── Prediction: t1 → t2 ─────────────────────────")
	kf.State().Print("Predicted state x̂[2,1]")
	kf.Covariance().Print("Predicted covariance P[2,1]")

	fmt.Println()
	fmt.Println("Done. Expected values from KalmanFilter.NET:")
	fmt.Println("  x̂[1,1] ≈ [11009.37, 201.43]")
	fmt.Println("  P[1,1]  ≈ [[14.57, 1.43], [1.43, 0.71]]")
	fmt.Println("  x̂[2,1] ≈ [12016.5, 201.43]")
	fmt.Println("  P[2,1]  ≈ [[52.86, 7.47], [7.47, 1.71]]")
}
