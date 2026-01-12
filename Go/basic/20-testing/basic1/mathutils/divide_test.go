package mathutils

import "testing"

func TestDivide(t *testing.T) {
	tests := []struct {
		name      string
		a, b      int
		expected  int
		expectErr bool
	}{
		{"正常除法", 6, 3, 2, false},
		{"除以零", 6, 0, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Divide(tt.a, tt.b)

			if (err != nil) != tt.expectErr {
				t.Errorf("Divide(%d, %d) error = %v; expectErr %v", tt.a, tt.b, err, tt.expectErr)
			}

			if result != tt.expected {
				t.Errorf("Divide(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}
