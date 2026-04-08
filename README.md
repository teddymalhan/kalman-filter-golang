# Kalman Filter in Go

A clean, dependency-free implementation of the linear (multivariate) Kalman Filter in Go, based on the worked examples from [KalmanFilter.NET](https://www.kalmanfilter.net).

## What is a Kalman Filter?

The Kalman Filter is an algorithm for estimating the state of a system in the presence of uncertainty — measurement noise, unmodeled dynamics, and unknown external influences. It operates in a continuous **predict → update** cycle:

- **Predict**: use a dynamic model to forecast the next state and its uncertainty.
- **Update**: incorporate a new measurement, weighted by its reliability relative to the prediction.

The result is an optimal estimate that is always more certain than either the prediction or the measurement alone.

## Algorithm

The filter implements the standard linear Kalman Filter equations:

| Step | Equation |
|------|----------|
| State extrapolation | `x̂[n+1,n] = F·x̂[n,n] + G·u[n]` |
| Covariance extrapolation | `P[n+1,n] = F·P[n,n]·Fᵀ + Q` |
| Kalman gain | `K[n] = P[n,n-1]·Hᵀ·(H·P[n,n-1]·Hᵀ + R[n])⁻¹` |
| State update | `x̂[n,n] = x̂[n,n-1] + K[n]·(z[n] - H·x̂[n,n-1])` |
| Covariance update (Joseph form) | `P[n,n] = (I-K·H)·P[n,n-1]·(I-K·H)ᵀ + K·R·Kᵀ` |

The covariance update uses the **Joseph form** for numerical stability.

**Notation:**

| Symbol | Meaning |
|--------|---------|
| `x̂` | State estimate vector |
| `P` | Estimate covariance matrix |
| `F` | State transition matrix |
| `G` | Control input matrix |
| `u` | Control input vector |
| `H` | Observation (measurement) matrix |
| `Q` | Process noise covariance matrix |
| `R` | Measurement noise covariance matrix |
| `K` | Kalman gain |
| `z` | Measurement vector |

## Project Structure

```
kalman-filter-golang/
├── go.mod
├── main.go          # Radar tracking demo (2-iteration worked example)
├── matrix/
│   ├── matrix.go    # Matrix type: New, Add, Sub, Mul, T, Inv, Identity
│   └── matrix_test.go
└── kalman/
    ├── kalman.go    # Filter: New, Predict, Update, State, Covariance
    └── kalman_test.go
```

No external dependencies — only the Go standard library.

## Demo

The `main.go` file reproduces the two-iteration radar tracking example from KalmanFilter.NET:

- A radar tracks an aircraft moving in one dimension.
- State: `[range (m), velocity (m/s)]`
- Dynamic model: constant velocity with random acceleration noise.
- Sampling interval: `Δt = 5 s`

```
go run .
```

Expected output:

```
── Iteration 0: Initialization ─────────────────
Initial state estimate x̂[0,0]:   [10000.0000, 200.0000]
Initial covariance     P[0,0]:    [[16.00, 0.00], [0.00, 0.25]]

── Prediction: t0 → t1 ─────────────────────────
Predicted state x̂[1,0]:          [11000.0000, 200.0000]
Predicted covariance P[1,0]:      [[28.50, 3.75], [3.75, 1.25]]

── Iteration 1: Update ──────────────────────────
Updated state estimate x̂[1,1]:   [11009.3711, 201.4260]
Updated covariance P[1,1]:        [[14.57, 1.43], [1.43, 0.71]]

── Prediction: t1 → t2 ─────────────────────────
Predicted state x̂[2,1]:          [12016.5013, 201.4260]
Predicted covariance P[2,1]:      [[52.86, 7.47], [7.47, 1.71]]
```

## Usage

### 1. Define the filter matrices

```go
import (
    "github.com/teddymalhan/kalman-filter-golang/kalman"
    "github.com/teddymalhan/kalman-filter-golang/matrix"
)

const dt = 5.0  // time step [s]
const sa = 0.2  // std-dev of process acceleration noise [m/s²]
sa2 := sa * sa

// State transition matrix (constant velocity model)
F := matrix.NewFromSlice(2, 2, []float64{
    1, dt,
    0,  1,
})

// Observation matrix (measuring range and velocity directly)
H := matrix.Identity(2)

// Process noise covariance
Q := matrix.NewFromSlice(2, 2, []float64{
    (dt*dt*dt*dt/4)*sa2, (dt*dt*dt/2)*sa2,
    (dt*dt*dt/2)*sa2,    (dt*dt)*sa2,
})
```

### 2. Initialise with the first measurement

```go
x0 := matrix.NewFromSlice(2, 1, []float64{10_000, 200})  // [range, velocity]
P0 := matrix.NewFromSlice(2, 2, []float64{16, 0, 0, 0.25}) // measurement covariance

kf := kalman.New(F, H, Q, x0, P0)
```

### 3. Run the predict-update loop

```go
for _, measurement := range measurements {
    // Predict next state
    kf.Predict(nil) // pass a control vector instead of nil if you have one

    // Update with new measurement and its noise covariance
    R := matrix.NewFromSlice(2, 2, []float64{36, 0, 0, 2.25})
    z := matrix.NewFromSlice(2, 1, measurement)
    kf.Update(z, R)

    // Read current estimate
    state := kf.State()      // x̂[n,n]
    cov   := kf.Covariance() // P[n,n]
}
```

`R` can differ between calls, which lets you model varying measurement quality (e.g. low signal-to-noise ratio on a particular radar return).

## Running Tests

```
go test ./...
```

The `kalman` package test verifies the filter output against the reference values from KalmanFilter.NET to within a tolerance of `0.01`.

## References

- [KalmanFilter.NET](https://www.kalmanfilter.net) — single-page overview and step-by-step tutorial
- Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.
