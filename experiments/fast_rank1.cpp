#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

extern "C" int complete_target_rank1(
    const double* input, int rows, int cols, int target_row,
    const double* initial_rows, const double* initial_cols, double* output) {
  const int size = rows * cols;
  std::vector<unsigned char> observed(size, 0);
  std::vector<double> z(size, 0.0), means(cols), stds(cols);
  std::vector<int> row_counts(rows, 0), col_counts(cols, 0);
  int observed_count = 0;
  double initial_sum = 0.0;

  for (int j = 0; j < cols; ++j) {
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < rows; ++i) {
      const int p = i * cols + j;
      if (std::isfinite(input[p])) {
        const double clipped = std::min(99.5, std::max(0.5, input[p]));
        const double probability = clipped / 100.0;
        z[p] = std::log(probability / (1.0 - probability));
        observed[p] = 1;
        sum += z[p];
        ++count;
        ++row_counts[i];
        ++col_counts[j];
        ++observed_count;
      }
    }
    if (count == 0) return 2;
    means[j] = sum / count;
    double square_sum = 0.0;
    for (int i = 0; i < rows; ++i) {
      const int p = i * cols + j;
      if (observed[p]) {
        const double delta = z[p] - means[j];
        square_sum += delta * delta;
      }
    }
    stds[j] = std::sqrt(square_sum / count);
    if (!std::isfinite(stds[j]) || stds[j] < 1e-12) stds[j] = 1.0;
    for (int i = 0; i < rows; ++i) {
      const int p = i * cols + j;
      if (observed[p]) {
        z[p] = (z[p] - means[j]) / stds[j];
        initial_sum += z[p];
      }
    }
  }
  if (observed_count == 0) return 3;

  std::fill(output, output + cols, 0.0);
  constexpr double reg = 0.1;
  for (int ensemble = 0; ensemble < 10; ++ensemble) {
    std::vector<double> row_bias(rows, 0.0), col_bias(cols, 0.0);
    std::vector<double> row_factor(rows), col_factor(cols);
    std::copy(initial_rows + ensemble * rows,
              initial_rows + (ensemble + 1) * rows, row_factor.begin());
    std::copy(initial_cols + ensemble * cols,
              initial_cols + (ensemble + 1) * cols, col_factor.begin());
    double mean = initial_sum / observed_count;

    for (int iteration = 0; iteration < 40; ++iteration) {
      for (int i = 0; i < rows; ++i) {
        if (row_counts[i] == 0) continue;
        double a01 = 0.0, a11 = reg, b0 = 0.0, b1 = 0.0;
        const double a00 = row_counts[i] + reg;
        for (int j = 0; j < cols; ++j) {
          const int p = i * cols + j;
          if (!observed[p]) continue;
          const double factor = col_factor[j];
          const double target = z[p] - mean - col_bias[j];
          a01 += factor;
          a11 += factor * factor;
          b0 += target;
          b1 += target * factor;
        }
        const double determinant = a00 * a11 - a01 * a01;
        row_bias[i] = (b0 * a11 - b1 * a01) / determinant;
        row_factor[i] = (a00 * b1 - a01 * b0) / determinant;
      }

      for (int j = 0; j < cols; ++j) {
        if (col_counts[j] == 0) continue;
        double a01 = 0.0, a11 = reg, b0 = 0.0, b1 = 0.0;
        const double a00 = col_counts[j] + reg;
        for (int i = 0; i < rows; ++i) {
          const int p = i * cols + j;
          if (!observed[p]) continue;
          const double factor = row_factor[i];
          const double target = z[p] - mean - row_bias[i];
          a01 += factor;
          a11 += factor * factor;
          b0 += target;
          b1 += target * factor;
        }
        const double determinant = a00 * a11 - a01 * a01;
        col_bias[j] = (b0 * a11 - b1 * a01) / determinant;
        col_factor[j] = (a00 * b1 - a01 * b0) / determinant;
      }

      double residual_sum = 0.0;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          const int p = i * cols + j;
          if (!observed[p]) continue;
          residual_sum += z[p] - row_bias[i] - col_bias[j]
                          - row_factor[i] * col_factor[j];
        }
      }
      mean = residual_sum / observed_count;
    }

    for (int j = 0; j < cols; ++j) {
      output[j] += mean + row_bias[target_row] + col_bias[j]
                   + row_factor[target_row] * col_factor[j];
    }
  }

  for (int j = 0; j < cols; ++j) {
    const int p = target_row * cols + j;
    if (observed[p]) {
      output[j] = input[p];
    } else {
      const double restored = (output[j] / 10.0) * stds[j] + means[j];
      const double predicted = 100.0 / (1.0 + std::exp(-restored));
      output[j] = std::min(100.0, std::max(0.0, predicted));
    }
  }
  return 0;
}
