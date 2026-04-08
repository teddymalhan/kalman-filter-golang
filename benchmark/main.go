// benchmark/main.go — standalone wall-clock comparison of serial vs parallel
// matrix multiplication and Kalman filter throughput.
//
// Run with: go run ./benchmark
package main

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/teddymalhan/kalman-filter-golang/kalman"
	"github.com/teddymalhan/kalman-filter-golang/matrix"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║         Kalman Filter Go — Concurrency Benchmark         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	benchMul()
	fmt.Println()
	benchFilters()
}

// ── Matrix multiplication benchmark ──────────────────────────────────────────

func benchMul() {
	fmt.Println("── Matrix Multiplication: Serial vs Parallel ───────────────")
	fmt.Printf("%-12s  %-16s  %-16s  %-10s\n", "Size", "Serial (ms/op)", "Parallel (ms/op)", "Speedup")
	fmt.Println(strings.Repeat("─", 60))

	for _, n := range []int{10, 100, 500} {
		a := randomMatrix(n)
		b := randomMatrix(n)
		const iters = 10

		// Serial
		start := time.Now()
		for i := 0; i < iters; i++ {
			matrix.Mul(a, b)
		}
		serialDur := time.Since(start)
		serialMs := float64(serialDur.Microseconds()) / float64(iters) / 1000.0

		// Parallel
		start = time.Now()
		for i := 0; i < iters; i++ {
			matrix.MulParallel(a, b)
		}
		parallelDur := time.Since(start)
		parallelMs := float64(parallelDur.Microseconds()) / float64(iters) / 1000.0

		speedup := serialMs / parallelMs
		fmt.Printf("%-12s  %-16.4f  %-16.4f  %.2fx\n",
			fmt.Sprintf("%dx%d", n, n), serialMs, parallelMs, speedup)
	}
}

// ── Kalman filter throughput benchmark ───────────────────────────────────────

func newRadarFilter() *kalman.Filter {
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
	return kalman.New(F, H, Q, x0, P0)
}

func runFilterSteps(steps int) {
	kf := newRadarFilter()
	z := matrix.NewFromSlice(2, 1, []float64{11_020, 202})
	R := matrix.NewFromSlice(2, 2, []float64{36, 0, 0, 2.25})
	for i := 0; i < steps; i++ {
		kf.Predict(nil)
		_ = kf.Update(z, R)
	}
}

func benchFilters() {
	const numTargets = 1000
	const stepsPerTarget = 50

	fmt.Println("── Kalman Filter Throughput: Serial vs Parallel ────────────")
	fmt.Printf("%d targets × %d steps each\n\n", numTargets, stepsPerTarget)
	fmt.Printf("%-16s  %-16s  %-10s\n", "Serial (ms)", "Parallel (ms)", "Speedup")
	fmt.Println(strings.Repeat("─", 48))

	// Serial
	start := time.Now()
	for t := 0; t < numTargets; t++ {
		runFilterSteps(stepsPerTarget)
	}
	serialMs := float64(time.Since(start).Microseconds()) / 1000.0

	// Parallel
	start = time.Now()
	var wg sync.WaitGroup
	wg.Add(numTargets)
	for t := 0; t < numTargets; t++ {
		go func() {
			defer wg.Done()
			runFilterSteps(stepsPerTarget)
		}()
	}
	wg.Wait()
	parallelMs := float64(time.Since(start).Microseconds()) / 1000.0

	speedup := serialMs / parallelMs
	fmt.Printf("%-16.2f  %-16.2f  %.2fx\n", serialMs, parallelMs, speedup)
}

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
