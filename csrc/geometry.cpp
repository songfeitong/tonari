#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>


namespace {

constexpr int64_t kMaximumImageShifts = int64_t{1} << 24;


void active_pseudoinverse(
    const double* cell,
    const int (&active_axes)[3],
    int active_count,
    long double (&dual)[3][3]) {
    long double columns[3][3] = {};
    long double right_vectors[3][3] = {};
    for (int column = 0; column < active_count; ++column) {
        right_vectors[column][column] = 1.0L;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            columns[cartesian][column] =
                static_cast<long double>(cell[3 * active_axes[column] + cartesian]);
        }
    }

    // One-sided Jacobi works on the cell rows directly; a Gram matrix would
    // square the condition number and reject valid, nearly parallel rows.
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool converged = true;
        for (int first = 0; first < active_count; ++first) {
            for (int second = first + 1; second < active_count; ++second) {
                long double first_norm_squared = 0.0L;
                long double second_norm_squared = 0.0L;
                long double inner_product = 0.0L;
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    first_norm_squared +=
                        columns[cartesian][first] * columns[cartesian][first];
                    second_norm_squared +=
                        columns[cartesian][second] * columns[cartesian][second];
                    inner_product +=
                        columns[cartesian][first] * columns[cartesian][second];
                }
                const long double convergence_scale = std::sqrt(
                    first_norm_squared * second_norm_squared);
                if (std::abs(inner_product) <=
                    8 * std::numeric_limits<double>::epsilon() *
                        convergence_scale) {
                    continue;
                }
                converged = false;
                const long double tau =
                    (second_norm_squared - first_norm_squared) /
                    (2 * inner_product);
                const long double tangent = tau == 0.0L
                    ? 1.0L
                    : std::copysign(1.0L, tau) /
                        (std::abs(tau) + std::hypot(1.0L, tau));
                const long double cosine =
                    1.0L / std::hypot(1.0L, tangent);
                const long double sine = cosine * tangent;
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    const long double first_value = columns[cartesian][first];
                    const long double second_value = columns[cartesian][second];
                    columns[cartesian][first] =
                        cosine * first_value - sine * second_value;
                    columns[cartesian][second] =
                        sine * first_value + cosine * second_value;
                }
                for (int row = 0; row < active_count; ++row) {
                    const long double first_value = right_vectors[row][first];
                    const long double second_value = right_vectors[row][second];
                    right_vectors[row][first] =
                        cosine * first_value - sine * second_value;
                    right_vectors[row][second] =
                        sine * first_value + cosine * second_value;
                }
            }
        }
        if (converged) {
            break;
        }
    }

    long double singular_values[3] = {};
    long double largest_singular = 0.0L;
    long double smallest_singular = std::numeric_limits<long double>::infinity();
    for (int column = 0; column < active_count; ++column) {
        long double norm_squared = 0.0L;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            norm_squared += columns[cartesian][column] * columns[cartesian][column];
        }
        singular_values[column] = std::sqrt(norm_squared);
        largest_singular = std::max(largest_singular, singular_values[column]);
        smallest_singular = std::min(smallest_singular, singular_values[column]);
    }
    const long double tolerance =
        std::numeric_limits<double>::epsilon() * 3 *
        std::max(largest_singular, 1.0L);
    TORCH_CHECK(
        smallest_singular > tolerance,
        "active periodic cell vectors must be linearly independent");

    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        for (int axis = 0; axis < active_count; ++axis) {
            long double value = 0.0L;
            for (int singular = 0; singular < active_count; ++singular) {
                value += columns[cartesian][singular] *
                    right_vectors[axis][singular] /
                    (singular_values[singular] * singular_values[singular]);
            }
            dual[cartesian][axis] = value;
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
    std::vector<int64_t> image_offsets = {0};

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const double* cell = cell_data + 9 * batch;
        for (int index = 0; index < 9; ++index) {
            TORCH_CHECK(
                std::isfinite(cell[index]),
                "cells must contain only finite values");
        }
        if (count_data[batch] == 0) {
            image_offsets.push_back(static_cast<int64_t>(shifts.size() / 3));
            continue;
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
            long double active_dual[3][3] = {};
            active_pseudoinverse(
                cell,
                active_axes,
                active_count,
                active_dual);
            for (int column = 0; column < active_count; ++column) {
                long double reciprocal_norm_squared = 0.0L;
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    const long double value = active_dual[cartesian][column];
                    TORCH_CHECK(
                        std::isfinite(value) &&
                            std::abs(value) <=
                                std::numeric_limits<double>::max(),
                        "active periodic cell dual is outside the float64 range");
                    duals[9 * batch + 3 * cartesian + active_axes[column]] =
                        static_cast<double>(value);
                    reciprocal_norm_squared += value * value;
                }
                const long double repeat =
                    std::ceil(cutoff * std::sqrt(reciprocal_norm_squared));
                TORCH_CHECK(
                    repeat <= std::numeric_limits<int32_t>::max(),
                    "periodic image range exceeds int32 cell shifts");
                repeats[active_axes[column]] = static_cast<int64_t>(repeat);
            }
        }

        // The limit turns a tiny-cell memory explosion into a deterministic
        // error before the Cartesian product is allocated or enumerated.
        int64_t image_count = 1;
        for (const int64_t repeat : repeats) {
            const int64_t factor = 2 * repeat + 1;
            TORCH_CHECK(
                image_count <= kMaximumImageShifts / factor,
                "periodic image count exceeds the 2^24 resource limit");
            image_count *= factor;
        }
        TORCH_CHECK(
            static_cast<int64_t>(shifts.size() / 3) <=
                kMaximumImageShifts - image_count,
            "batched periodic image count exceeds the 2^24 resource limit");
        const int64_t total_image_count =
            static_cast<int64_t>(shifts.size() / 3) + image_count;
        shifts.reserve(static_cast<size_t>(3 * total_image_count));
        for (int64_t x = -repeats[0]; x <= repeats[0]; ++x) {
            for (int64_t y = -repeats[1]; y <= repeats[1]; ++y) {
                for (int64_t z = -repeats[2]; z <= repeats[2]; ++z) {
                    shifts.push_back(static_cast<int32_t>(x));
                    shifts.push_back(static_cast<int32_t>(y));
                    shifts.push_back(static_cast<int32_t>(z));
                }
            }
        }
        image_offsets.push_back(static_cast<int64_t>(shifts.size() / 3));
    }

    auto dual_tensor = torch::empty_like(cells);
    auto shift_tensor = torch::empty(
        {static_cast<int64_t>(shifts.size() / 3), 3},
        cells.options().dtype(torch::kInt32));
    auto image_offsets_tensor = torch::empty(
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
        image_offsets_tensor.data_ptr<int64_t>(),
        image_offsets.data(),
        image_offsets.size() * sizeof(int64_t));
    return {dual_tensor, shift_tensor, image_offsets_tensor};
}
