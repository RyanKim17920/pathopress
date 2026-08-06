#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace {
constexpr int kEnsembles = 10;
constexpr int kIterations = 40;
constexpr double kRegularization = 0.1;

double sigmoid_percent(double value) noexcept {
  if (value >= 0.0) {
    return 100.0 / (1.0 + std::exp(-value));
  }
  const double exponential = std::exp(value);
  return 100.0 * exponential / (1.0 + exponential);
}
}  // namespace

extern "C" int complete_target_rank1(
    const double* input, int rows, int cols, int target_row,
    const double* initial_rows, const double* initial_cols,
    double* output) noexcept {
  try {
    if (input == nullptr || initial_rows == nullptr || initial_cols == nullptr ||
        output == nullptr) {
      return 10;
    }
    if (rows <= 0 || cols <= 0 || target_row < 0 || target_row >= rows) {
      return 11;
    }
    const std::size_t row_count = static_cast<std::size_t>(rows);
    const std::size_t col_count = static_cast<std::size_t>(cols);
    if (row_count > std::numeric_limits<std::size_t>::max() / col_count) {
      return 12;
    }
    const std::size_t size = row_count * col_count;
    if (row_count > std::numeric_limits<std::size_t>::max() / kEnsembles ||
        col_count > std::numeric_limits<std::size_t>::max() / kEnsembles) {
      return 12;
    }
    for (std::size_t p = 0; p < row_count * kEnsembles; ++p) {
      if (!std::isfinite(initial_rows[p])) return 13;
    }
    for (std::size_t p = 0; p < col_count * kEnsembles; ++p) {
      if (!std::isfinite(initial_cols[p])) return 13;
    }

    std::vector<unsigned char> observed(size, 0);
    std::vector<double> z(size, 0.0), means(col_count), stds(col_count);
    std::vector<int> row_counts(row_count, 0), col_counts(col_count, 0);
    std::size_t observed_count = 0;
    double initial_sum = 0.0;

    for (int j = 0; j < cols; ++j) {
      double sum = 0.0;
      int count = 0;
      for (int i = 0; i < rows; ++i) {
        const std::size_t p = static_cast<std::size_t>(i) * col_count + j;
        if (std::isfinite(input[p])) {
          if (input[p] < 0.0 || input[p] > 100.0) return 14;
          const double clipped = std::min(99.5, std::max(0.5, input[p]));
          const double probability = clipped / 100.0;
          z[p] = std::log(probability / (1.0 - probability));
          if (!std::isfinite(z[p])) return 15;
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
      if (!std::isfinite(means[j])) return 15;
      double square_sum = 0.0;
      for (int i = 0; i < rows; ++i) {
        const std::size_t p = static_cast<std::size_t>(i) * col_count + j;
        if (observed[p]) {
          const double delta = z[p] - means[j];
          square_sum += delta * delta;
        }
      }
      stds[j] = std::sqrt(square_sum / count);
      if (!std::isfinite(stds[j])) return 15;
      if (stds[j] < 1e-12) stds[j] = 1.0;
      for (int i = 0; i < rows; ++i) {
        const std::size_t p = static_cast<std::size_t>(i) * col_count + j;
        if (observed[p]) {
          z[p] = (z[p] - means[j]) / stds[j];
          if (!std::isfinite(z[p])) return 15;
          initial_sum += z[p];
        }
      }
    }
    if (observed_count == 0 || !std::isfinite(initial_sum)) return 3;

    std::fill(output, output + col_count, 0.0);
    for (int ensemble = 0; ensemble < kEnsembles; ++ensemble) {
      std::vector<double> row_bias(row_count, 0.0), col_bias(col_count, 0.0);
      std::vector<double> row_factor(row_count), col_factor(col_count);
      std::copy(initial_rows + static_cast<std::size_t>(ensemble) * row_count,
                initial_rows + static_cast<std::size_t>(ensemble + 1) * row_count,
                row_factor.begin());
      std::copy(initial_cols + static_cast<std::size_t>(ensemble) * col_count,
                initial_cols + static_cast<std::size_t>(ensemble + 1) * col_count,
                col_factor.begin());
      double mean = initial_sum / static_cast<double>(observed_count);

      for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int i = 0; i < rows; ++i) {
          if (row_counts[i] == 0) continue;
          double a01 = 0.0, a11 = kRegularization, b0 = 0.0, b1 = 0.0;
          const double a00 = row_counts[i] + kRegularization;
          for (int j = 0; j < cols; ++j) {
            const std::size_t p = static_cast<std::size_t>(i) * col_count + j;
            if (!observed[p]) continue;
            const double factor = col_factor[j];
            const double target = z[p] - mean - col_bias[j];
            a01 += factor;
            a11 += factor * factor;
            b0 += target;
            b1 += target * factor;
          }
          const double determinant = a00 * a11 - a01 * a01;
          if (!std::isfinite(determinant) || determinant <= 0.0) return 16;
          row_bias[i] = (b0 * a11 - b1 * a01) / determinant;
          row_factor[i] = (a00 * b1 - a01 * b0) / determinant;
          if (!std::isfinite(row_bias[i]) || !std::isfinite(row_factor[i])) {
            return 16;
          }
        }

        for (int j = 0; j < cols; ++j) {
          if (col_counts[j] == 0) continue;
          double a01 = 0.0, a11 = kRegularization, b0 = 0.0, b1 = 0.0;
          const double a00 = col_counts[j] + kRegularization;
          for (int i = 0; i < rows; ++i) {
            const std::size_t p = static_cast<std::size_t>(i) * col_count + j;
            if (!observed[p]) continue;
            const double factor = row_factor[i];
            const double target = z[p] - mean - row_bias[i];
            a01 += factor;
            a11 += factor * factor;
            b0 += target;
            b1 += target * factor;
          }
          const double determinant = a00 * a11 - a01 * a01;
          if (!std::isfinite(determinant) || determinant <= 0.0) return 16;
          col_bias[j] = (b0 * a11 - b1 * a01) / determinant;
          col_factor[j] = (a00 * b1 - a01 * b0) / determinant;
          if (!std::isfinite(col_bias[j]) || !std::isfinite(col_factor[j])) {
            return 16;
          }
        }

        double residual_sum = 0.0;
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            const std::size_t p = static_cast<std::size_t>(i) * col_count + j;
            if (!observed[p]) continue;
            residual_sum += z[p] - row_bias[i] - col_bias[j] -
                            row_factor[i] * col_factor[j];
          }
        }
        mean = residual_sum / static_cast<double>(observed_count);
        if (!std::isfinite(mean)) return 17;
      }

      for (int j = 0; j < cols; ++j) {
        const double value = mean + row_bias[target_row] + col_bias[j] +
                             row_factor[target_row] * col_factor[j];
        if (!std::isfinite(value)) return 17;
        output[j] += value;
      }
    }

    for (int j = 0; j < cols; ++j) {
      const std::size_t p = static_cast<std::size_t>(target_row) * col_count + j;
      if (observed[p]) {
        output[j] = input[p];
      } else {
        const double restored = (output[j] / kEnsembles) * stds[j] + means[j];
        if (!std::isfinite(restored)) return 18;
        const double predicted = sigmoid_percent(restored);
        if (!std::isfinite(predicted)) return 18;
        output[j] = std::min(100.0, std::max(0.0, predicted));
      }
    }
    return 0;
  } catch (const std::bad_alloc&) {
    return 20;
  } catch (...) {
    return 21;
  }
}
