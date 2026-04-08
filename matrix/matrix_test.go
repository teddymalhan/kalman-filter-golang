package matrix_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/teddymalhan/kalman-filter-golang/matrix"
)

const eps = 1e-9

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < eps
}

func assertMatrix(t *testing.T, label string, got, want *matrix.Matrix) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: size mismatch got %dx%d want %dx%d", label, got.Rows, got.Cols, want.Rows, want.Cols)
	}
	for i := 0; i < want.Rows; i++ {
		for j := 0; j < want.Cols; j++ {
			if !approxEqual(got.At(i, j), want.At(i, j)) {
				t.Errorf("%s [%d][%d]: got %.6f want %.6f", label, i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}

func TestIdentity(t *testing.T) {
	I := matrix.Identity(3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want := 0.0
			if i == j {
				want = 1.0
			}
			if I.At(i, j) != want {
				t.Errorf("I[%d][%d] = %v, want %v", i, j, I.At(i, j), want)
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	m := matrix.NewFromSlice(2, 3, []float64{1, 2, 3, 4, 5, 6})
	mt := m.T()
	want := matrix.NewFromSlice(3, 2, []float64{1, 4, 2, 5, 3, 6})
	assertMatrix(t, "transpose", mt, want)
}

func TestMul(t *testing.T) {
	a := matrix.NewFromSlice(2, 2, []float64{1, 2, 3, 4})
	b := matrix.NewFromSlice(2, 2, []float64{5, 6, 7, 8})
	got := matrix.Mul(a, b)
	want := matrix.NewFromSlice(2, 2, []float64{19, 22, 43, 50})
	assertMatrix(t, "mul", got, want)
}

func TestInv2x2(t *testing.T) {
	m := matrix.NewFromSlice(2, 2, []float64{4, 7, 2, 6})
	inv := matrix.Inv(m)
	// Verify m × inv ≈ I
	prod := matrix.Mul(m, inv)
	assertMatrix(t, "m·inv", prod, matrix.Identity(2))
}

func TestAddSub(t *testing.T) {
	a := matrix.NewFromSlice(2, 2, []float64{1, 2, 3, 4})
	b := matrix.NewFromSlice(2, 2, []float64{5, 6, 7, 8})
	sum := matrix.Add(a, b)
	wantSum := matrix.NewFromSlice(2, 2, []float64{6, 8, 10, 12})
	assertMatrix(t, "add", sum, wantSum)

	diff := matrix.Sub(b, a)
	wantDiff := matrix.NewFromSlice(2, 2, []float64{4, 4, 4, 4})
	assertMatrix(t, "sub", diff, wantDiff)
}

func TestMulParallel(t *testing.T) {
	a := matrix.NewFromSlice(2, 2, []float64{1, 2, 3, 4})
	b := matrix.NewFromSlice(2, 2, []float64{5, 6, 7, 8})
	got := matrix.MulParallel(a, b)
	want := matrix.NewFromSlice(2, 2, []float64{19, 22, 43, 50})
	assertMatrix(t, "mul_parallel", got, want)
}

// randomMatrix returns a deterministically filled n×n matrix.
func randomMatrix(n int) *matrix.Matrix {
	m := matrix.New(n, n)
	v := 1.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			m.Set(i, j, v)
			v = v*1.1 + 0.3
			if v > 1e6 {
				v = 1.0
			}
		}
	}
	return m
}

var benchSizes = []int{10, 100, 500}

func BenchmarkMul(b *testing.B) {
	for _, n := range benchSizes {
		a := randomMatrix(n)
		bm := randomMatrix(n)
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matrix.Mul(a, bm)
			}
		})
	}
}

func BenchmarkMulParallel(b *testing.B) {
	for _, n := range benchSizes {
		a := randomMatrix(n)
		bm := randomMatrix(n)
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matrix.MulParallel(a, bm)
			}
		})
	}
}
