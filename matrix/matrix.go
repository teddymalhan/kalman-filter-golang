// Package matrix provides basic 2D matrix operations for the Kalman Filter.
package matrix

import (
	"fmt"
	"math"
)

// Matrix is a row-major 2D matrix.
type Matrix struct {
	Rows, Cols int
	Data       [][]float64
}

// New returns a zero-valued matrix of size rows×cols.
func New(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{Rows: rows, Cols: cols, Data: data}
}

// NewFromSlice builds a matrix from a row-major flat slice.
func NewFromSlice(rows, cols int, vals []float64) *Matrix {
	if len(vals) != rows*cols {
		panic(fmt.Sprintf("matrix: expected %d values, got %d", rows*cols, len(vals)))
	}
	m := New(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = vals[i*cols+j]
		}
	}
	return m
}

// Identity returns an n×n identity matrix.
func Identity(n int) *Matrix {
	m := New(n, n)
	for i := 0; i < n; i++ {
		m.Data[i][i] = 1
	}
	return m
}

// Clone returns a deep copy.
func (m *Matrix) Clone() *Matrix {
	c := New(m.Rows, m.Cols)
	for i := range m.Data {
		copy(c.Data[i], m.Data[i])
	}
	return c
}

// At returns the element at row i, column j.
func (m *Matrix) At(i, j int) float64 { return m.Data[i][j] }

// Set sets the element at row i, column j.
func (m *Matrix) Set(i, j int, v float64) { m.Data[i][j] = v }

// T returns the transpose.
func (m *Matrix) T() *Matrix {
	t := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			t.Data[j][i] = m.Data[i][j]
		}
	}
	return t
}

// Add returns m + b (element-wise).
func Add(a, b *Matrix) *Matrix {
	mustSameSize(a, b)
	c := New(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			c.Data[i][j] = a.Data[i][j] + b.Data[i][j]
		}
	}
	return c
}

// Sub returns a - b (element-wise).
func Sub(a, b *Matrix) *Matrix {
	mustSameSize(a, b)
	c := New(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			c.Data[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}
	return c
}

// Mul returns the matrix product a × b.
func Mul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("matrix: incompatible sizes %dx%d × %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
	c := New(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for k := 0; k < a.Cols; k++ {
			for j := 0; j < b.Cols; j++ {
				c.Data[i][j] += a.Data[i][k] * b.Data[k][j]
			}
		}
	}
	return c
}

// Inv computes the matrix inverse using Gauss-Jordan elimination.
// Panics if the matrix is singular or not square.
func Inv(m *Matrix) *Matrix {
	n := m.Rows
	if n != m.Cols {
		panic("matrix: inverse requires a square matrix")
	}

	// Augment [m | I]
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		copy(aug[i], m.Data[i])
		aug[i][n+i] = 1
	}

	for col := 0; col < n; col++ {
		// Find pivot
		pivot := col
		for row := col + 1; row < n; row++ {
			if math.Abs(aug[row][col]) > math.Abs(aug[pivot][col]) {
				pivot = row
			}
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]

		p := aug[col][col]
		if math.Abs(p) < 1e-14 {
			panic("matrix: singular matrix, cannot invert")
		}
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= p
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := 0; j < 2*n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	inv := New(n, n)
	for i := 0; i < n; i++ {
		copy(inv.Data[i], aug[i][n:])
	}
	return inv
}

// Print prints the matrix in a readable format with an optional label.
func (m *Matrix) Print(label string) {
	if label != "" {
		fmt.Printf("%s:\n", label)
	}
	for _, row := range m.Data {
		fmt.Print("  [")
		for j, v := range row {
			if j > 0 {
				fmt.Print("  ")
			}
			fmt.Printf("%10.4f", v)
		}
		fmt.Println(" ]")
	}
}

func mustSameSize(a, b *Matrix) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("matrix: size mismatch %dx%d vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
}
