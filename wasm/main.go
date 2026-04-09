//go:build js && wasm

// Package main is the WebAssembly entry point for the Kalman Filter demo.
// It exposes a single JS function:
//
//	kalmanSimulate(sigmaA float64, sigmaR float64) string
//
// The function returns a JSON-encoded SimResult.
package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"syscall/js"

	"github.com/teddymalhan/kalman-filter-golang/kalman"
	"github.com/teddymalhan/kalman-filter-golang/matrix"
)

type point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

type simResult struct {
	Truth    []point `json:"truth"`
	Measured []point `json:"measured"`
	Filtered []point `json:"filtered"`
	MeasRMSE float64 `json:"measRMSE"`
	FiltRMSE float64 `json:"filtRMSE"`
}

// simulate generates a 2D constant-velocity trajectory with random
// acceleration noise (sigmaA) and noisy position measurements (sigmaR),
// runs the Kalman filter, and returns the result as a JSON string.
//
// The random seed is fixed so the underlying trajectory shape is stable
// while the user drags the sliders — only the filter parameters change.
func simulate(_ js.Value, args []js.Value) any {
	sigmaA := args[0].Float()
	sigmaR := args[1].Float()

	const (
		steps = 80
		dt    = 1.0
		seed  = 42
	)

	rng := rand.New(rand.NewSource(seed))

	// Ground truth: constant velocity + random acceleration
	truth := make([]point, steps)
	meas := make([]point, steps)

	px, py := 50.0, 200.0
	vx, vy := 7.0, 0.2

	for i := 0; i < steps; i++ {
		vx += rng.NormFloat64() * sigmaA * dt
		vy += rng.NormFloat64() * sigmaA * dt
		px += vx * dt
		py += vy * dt
		truth[i] = point{px, py}
		meas[i] = point{
			px + rng.NormFloat64()*sigmaR,
			py + rng.NormFloat64()*sigmaR,
		}
	}

	// 4D Kalman filter — state: [x, y, vx, vy], measurement: [x, y]
	sa2 := sigmaA * sigmaA
	sr2 := sigmaR * sigmaR

	F := matrix.NewFromSlice(4, 4, []float64{
		1, 0, dt, 0,
		0, 1, 0, dt,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	H := matrix.NewFromSlice(2, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
	})
	Q := matrix.NewFromSlice(4, 4, []float64{
		dt * dt * dt * dt / 4 * sa2, 0, dt * dt * dt / 2 * sa2, 0,
		0, dt * dt * dt * dt / 4 * sa2, 0, dt * dt * dt / 2 * sa2,
		dt * dt * dt / 2 * sa2, 0, dt * dt * sa2, 0,
		0, dt * dt * dt / 2 * sa2, 0, dt * dt * sa2,
	})
	R := matrix.NewFromSlice(2, 2, []float64{
		sr2, 0,
		0, sr2,
	})

	x0 := matrix.NewFromSlice(4, 1, []float64{meas[0].X, meas[0].Y, 0, 0})
	P0 := matrix.NewFromSlice(4, 4, []float64{
		sr2, 0, 0, 0,
		0, sr2, 0, 0,
		0, 0, 100, 0,
		0, 0, 0, 100,
	})

	kf := kalman.New(F, H, Q, x0, P0)

	filtered := make([]point, steps)
	filtered[0] = meas[0]

	for i := 1; i < steps; i++ {
		kf.Predict(nil)
		z := matrix.NewFromSlice(2, 1, []float64{meas[i].X, meas[i].Y})
		if err := kf.Update(z, R); err != nil {
			filtered[i] = filtered[i-1]
			continue
		}
		s := kf.State()
		filtered[i] = point{s.At(0, 0), s.At(1, 0)}
	}

	// RMSE of 2D position error
	var measSS, filtSS float64
	for i := 0; i < steps; i++ {
		dx := meas[i].X - truth[i].X
		dy := meas[i].Y - truth[i].Y
		measSS += dx*dx + dy*dy

		dx = filtered[i].X - truth[i].X
		dy = filtered[i].Y - truth[i].Y
		filtSS += dx*dx + dy*dy
	}
	measRMSE := math.Sqrt(measSS / float64(steps))
	filtRMSE := math.Sqrt(filtSS / float64(steps))

	b, _ := json.Marshal(simResult{truth, meas, filtered, measRMSE, filtRMSE})
	return string(b)
}

func main() {
	js.Global().Set("kalmanSimulate", js.FuncOf(simulate))
	<-make(chan struct{}) // keep the Go runtime alive
}
