import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad


def resample_along_arc_length(x, y, num_target_samples):

    assert x.shape == y.shape

    cs = CubicSpline(x, y)

    for x_val, y_val in zip(cs.x, y):
        assert abs(y_val - cs(x_val)) < 1e-4

    def integrand(t):
        return np.sqrt(1 + cs.derivative()(t)**2)

    arc_lengths = np.zeros_like(x, dtype=np.float64)
    for i in range(1, len(x)):
        arc_length, _ = quad(integrand, x[i-1], x[i])
        arc_lengths[i] = arc_lengths[i-1] + arc_length
    
    total_arc_length, _ = quad(integrand, x[0], x[-1])
    assert abs(total_arc_length - arc_lengths[-1]) < 1, f"after = {total_arc_length},  before = {arc_lengths[-1]}"

    x_arc = arc_lengths
    y_arc = y

    cs_arc = CubicSpline(x_arc, y_arc)

    y_arc_resampled = []
    a, delta = 0, total_arc_length/num_target_samples
    for _ in range(num_target_samples):
        y_arc_resampled.append(float(cs_arc(a)))
        a+=delta
        
    return y_arc_resampled
    
