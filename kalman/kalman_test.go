package kalman_test

import (
	"math"
	"testing"

	"github.com/teddymalhan/kalman-filter-golang/kalman"
	"github.com/teddymalhan/kalman-filter-golang/matrix"
)

const tol = 1e-2 // tolerance for comparing against site's rounded values

func approx(a, b float64) bool { return math.Abs(a-b) < tol }

// TestRadarExample reproduces the two-iteration worked example from
// https://www.kalmanfilter.net (single-page overview).
func TestRadarExample(t *testing.T) {
	const dt = 5.0
	const sa = 0.2
	sa2 := sa * sa

	F := matrix.NewFromSlice(2, 2, []float64{1, dt, 0, 1})
	H := matrix.Identity(2)
	Q := matrix.NewFromSlice(2, 2, []float64{
		(dt * dt * dt * dt / 4) * sa2, (dt * dt * dt / 2) * sa2,
		(dt * dt * dt / 2) * sa2, (dt * dt) * sa2,
	})

	x0 := matrix.NewFromSlice(2, 1, []float64{10_000, 200})
	P0 := matrix.NewFromSlice(2, 2, []float64{16, 0, 0, 0.25})

	kf := kalman.New(F, H, Q, x0, P0)

	// ── Predict t0 → t1 ───────────────────────────────────────────────────
	kf.Predict(nil)

	// Expected: x̂[1,0] = [11000, 200], P[1,0] = [[28.5, 3.75],[3.75, 1.25]]
	x10 := kf.State()
	if !approx(x10.At(0, 0), 11_000) || !approx(x10.At(1, 0), 200) {
		t.Errorf("x̂[1,0]: got [%.4f, %.4f], want [11000, 200]", x10.At(0, 0), x10.At(1, 0))
	}
	P10 := kf.Covariance()
	checkMatrix(t, "P[1,0]", P10, [][]float64{{28.5, 3.75}, {3.75, 1.25}})

	// ── Update with z1 ────────────────────────────────────────────────────
	z1 := matrix.NewFromSlice(2, 1, []float64{11_020, 202})
	R1 := matrix.NewFromSlice(2, 2, []float64{36, 0, 0, 2.25})

	if err := kf.Update(z1, R1); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// Expected: x̂[1,1] ≈ [11009.37, 201.43]
	x11 := kf.State()
	if !approx(x11.At(0, 0), 11_009.37) || !approx(x11.At(1, 0), 201.43) {
		t.Errorf("x̂[1,1]: got [%.4f, %.4f], want [11009.37, 201.43]", x11.At(0, 0), x11.At(1, 0))
	}

	// ── Predict t1 → t2 ───────────────────────────────────────────────────
	kf.Predict(nil)

	// Expected: x̂[2,1] ≈ [12016.5, 201.43]
	x21 := kf.State()
	if !approx(x21.At(0, 0), 12_016.5) || !approx(x21.At(1, 0), 201.43) {
		t.Errorf("x̂[2,1]: got [%.4f, %.4f], want [12016.5, 201.43]", x21.At(0, 0), x21.At(1, 0))
	}
}

func checkMatrix(t *testing.T, label string, got *matrix.Matrix, want [][]float64) {
	t.Helper()
	for i, row := range want {
		for j, w := range row {
			g := got.At(i, j)
			if !approx(g, w) {
				t.Errorf("%s[%d][%d]: got %.4f, want %.4f", label, i, j, g, w)
			}
		}
	}
}
