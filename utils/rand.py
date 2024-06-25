import numpy as np

nrolls = 100000  # number of experiments
nstars = 1000    # maximum number of stars to distribute



# Function to generate a random sample from the piecewise linear distribution



def get_random_rd_distribution():
    # Define intervals and weights
    intervals = np.array([-1.8, -1.7, -1.6, -1.5, 0.0])
    weights = np.array([5.0, 6.7, 8.37, 10, 10])

    # Calculate the areas under each segment
    segment_lengths = intervals[1:] - intervals[:-1]
    segment_areas = (weights[:-1] + weights[1:]) / 2 * segment_lengths

    # Calculate the total area for normalization
    total_area = segment_areas.sum()

    # Compute the CDF for the piecewise linear distribution
    cumulative_areas = np.cumsum(segment_areas / total_area)

    def piecewise_linear_distribution(size=1):
        samples = []
        for _ in range(size):
            r = np.random.uniform()
            # Find which segment the random number falls into
            segment_index = np.searchsorted(cumulative_areas, r)
            # Linear interpolation within the segment
            b0, b1 = intervals[segment_index], intervals[segment_index + 1]
            w0, w1 = weights[segment_index], weights[segment_index + 1]
            # Calculate the interpolated value
            segment_cdf_start = cumulative_areas[segment_index - 1] if segment_index > 0 else 0
            segment_area = segment_areas[segment_index]
            relative_r = (r - segment_cdf_start) * total_area
            interpolated_value = b0 + (relative_r / segment_area) * (b1 - b0)
            samples.append(interpolated_value)
        return np.array(samples)

    return piecewise_linear_distribution