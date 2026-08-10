#include "periodic_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>


namespace {

void symmetric_eigenvalues(double matrix[3][3], int size, double values[3]) {
    for (int sweep = 0; sweep < 24; ++sweep) {
        bool converged = true;
        for (int first = 0; first < size; ++first) {
            for (int second = first + 1; second < size; ++second) {
                const double off_diagonal = matrix[first][second];
                const double scale = std::max(
                    {std::abs(matrix[first][first]),
                     std::abs(matrix[second][second]),
                     1.0});
                if (std::abs(off_diagonal) <=
                    std::numeric_limits<double>::epsilon() * scale) {
                    continue;
                }
                converged = false;
                const double difference =
                    matrix[second][second] - matrix[first][first];
                const double tangent = difference == 0.0
                    ? 1.0
                    : std::copysign(
                          1.0,
                          difference / off_diagonal) /
                        (std::abs(difference / (2.0 * off_diagonal)) +
                         std::sqrt(
                             1.0 +
                             difference * difference /
                                 (4.0 * off_diagonal * off_diagonal)));
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;
                const double first_diagonal = matrix[first][first];
                const double second_diagonal = matrix[second][second];
                matrix[first][first] = first_diagonal - tangent * off_diagonal;
                matrix[second][second] = second_diagonal + tangent * off_diagonal;
                matrix[first][second] = 0.0;
                matrix[second][first] = 0.0;
                for (int index = 0; index < size; ++index) {
                    if (index == first || index == second) {
                        continue;
                    }
                    const double first_value = matrix[index][first];
                    const double second_value = matrix[index][second];
                    matrix[index][first] =
                        cosine * first_value - sine * second_value;
                    matrix[first][index] = matrix[index][first];
                    matrix[index][second] =
                        sine * first_value + cosine * second_value;
                    matrix[second][index] = matrix[index][second];
                }
            }
        }
        if (converged) {
            break;
        }
    }
    for (int index = 0; index < size; ++index) {
        values[index] = matrix[index][index];
    }
}


void invert_matrix(const double input[3][3], int size, double inverse[3][3]) {
    double augmented[3][6] = {};
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            augmented[row][column] = input[row][column];
            augmented[row][size + column] = row == column ? 1.0 : 0.0;
        }
    }
    for (int column = 0; column < size; ++column) {
        int pivot = column;
        for (int row = column + 1; row < size; ++row) {
            if (std::abs(augmented[row][column]) >
                std::abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (pivot != column) {
            for (int entry = 0; entry < 2 * size; ++entry) {
                std::swap(augmented[column][entry], augmented[pivot][entry]);
            }
        }
        const double pivot_value = augmented[column][column];
        for (int entry = 0; entry < 2 * size; ++entry) {
            augmented[column][entry] /= pivot_value;
        }
        for (int row = 0; row < size; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = augmented[row][column];
            for (int entry = 0; entry < 2 * size; ++entry) {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            inverse[row][column] = augmented[row][size + column];
        }
    }
}

}  // namespace


std::vector<torch::Tensor> build_periodic_metadata_cpu(
    const torch::Tensor& cells,
    const torch::Tensor& pbc,
    const torch::Tensor& atom_counts,
    double cutoff) {
    TORCH_CHECK(!cells.is_cuda() && !pbc.is_cuda() && !atom_counts.is_cuda());
    TORCH_CHECK(cells.scalar_type() == torch::kFloat64);
    TORCH_CHECK(pbc.scalar_type() == torch::kBool);
    TORCH_CHECK(atom_counts.scalar_type() == torch::kInt64);
    TORCH_CHECK(cells.is_contiguous() && pbc.is_contiguous() && atom_counts.is_contiguous());
    const int64_t batch_size = cells.size(0);
    const double* cell_data = cells.data_ptr<double>();
    const bool* pbc_data = pbc.data_ptr<bool>();
    const int64_t* count_data = atom_counts.data_ptr<int64_t>();
    std::vector<double> duals(9 * batch_size, 0.0);
    std::vector<int32_t> shifts;
    std::vector<int64_t> image_ptr = {0};

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const double* cell = cell_data + 9 * batch;
        for (int index = 0; index < 9; ++index) {
            TORCH_CHECK(
                std::isfinite(cell[index]),
                "cells must contain only finite values");
        }
        int active_axes[3];
        int active_count = 0;
        for (int axis = 0; axis < 3; ++axis) {
            if (pbc_data[3 * batch + axis]) {
                active_axes[active_count++] = axis;
            }
        }
        int64_t repeats[3] = {0, 0, 0};
        if (active_count > 0) {
            double gram[3][3] = {};
            for (int row = 0; row < active_count; ++row) {
                for (int column = 0; column < active_count; ++column) {
                    for (int cartesian = 0; cartesian < 3; ++cartesian) {
                        gram[row][column] +=
                            cell[3 * active_axes[row] + cartesian] *
                            cell[3 * active_axes[column] + cartesian];
                    }
                }
            }
            double eigen_matrix[3][3];
            std::memcpy(eigen_matrix, gram, sizeof(gram));
            double eigenvalues[3] = {};
            symmetric_eigenvalues(eigen_matrix, active_count, eigenvalues);
            const auto minimum = *std::min_element(
                eigenvalues, eigenvalues + active_count);
            const auto maximum = *std::max_element(
                eigenvalues, eigenvalues + active_count);
            const double largest_singular =
                std::sqrt(std::max(maximum, 0.0));
            const double smallest_singular =
                std::sqrt(std::max(minimum, 0.0));
            const double tolerance = std::numeric_limits<double>::epsilon() * 3 *
                std::max(largest_singular, 1.0);
            TORCH_CHECK(
                smallest_singular > tolerance,
                "active periodic cell vectors must be linearly independent");

            double inverse_gram[3][3] = {};
            invert_matrix(gram, active_count, inverse_gram);
            for (int column = 0; column < active_count; ++column) {
                double reciprocal_norm_squared = 0.0;
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    double value = 0.0;
                    for (int row = 0; row < active_count; ++row) {
                        value += cell[3 * active_axes[row] + cartesian] *
                            inverse_gram[row][column];
                    }
                    duals[9 * batch + 3 * cartesian + active_axes[column]] = value;
                    reciprocal_norm_squared += value * value;
                }
                const double repeat =
                    std::ceil(cutoff * std::sqrt(reciprocal_norm_squared));
                TORCH_CHECK(
                    repeat <= std::numeric_limits<int32_t>::max(),
                    "periodic image range exceeds int32 cell shifts");
                repeats[active_axes[column]] = static_cast<int64_t>(repeat);
            }
        }

        if (count_data[batch] > 0) {
            for (int64_t x = -repeats[0]; x <= repeats[0]; ++x) {
                for (int64_t y = -repeats[1]; y <= repeats[1]; ++y) {
                    for (int64_t z = -repeats[2]; z <= repeats[2]; ++z) {
                        shifts.push_back(static_cast<int32_t>(x));
                        shifts.push_back(static_cast<int32_t>(y));
                        shifts.push_back(static_cast<int32_t>(z));
                    }
                }
            }
        }
        image_ptr.push_back(static_cast<int64_t>(shifts.size() / 3));
    }

    auto dual_tensor = torch::empty_like(cells);
    auto shift_tensor = torch::empty(
        {static_cast<int64_t>(shifts.size() / 3), 3},
        cells.options().dtype(torch::kInt32));
    auto ptr_tensor = torch::empty(
        {batch_size + 1}, cells.options().dtype(torch::kInt64));
    if (!duals.empty()) {
        std::memcpy(
            dual_tensor.data_ptr<double>(),
            duals.data(),
            duals.size() * sizeof(double));
    }
    if (!shifts.empty()) {
        std::memcpy(
            shift_tensor.data_ptr<int32_t>(),
            shifts.data(),
            shifts.size() * sizeof(int32_t));
    }
    std::memcpy(
        ptr_tensor.data_ptr<int64_t>(),
        image_ptr.data(),
        image_ptr.size() * sizeof(int64_t));
    return {dual_tensor, shift_tensor, ptr_tensor};
}
