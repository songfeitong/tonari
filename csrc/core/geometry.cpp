#include "geometry.h"
#include "errors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>


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
    neighbor_search::require_input(
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


neighbor_search::PeriodicMetadata neighbor_search::build_periodic_metadata(
    std::span<const double> cells,
    std::span<const uint8_t> pbc,
    std::span<const int64_t> atom_counts,
    double cutoff) {
    const int64_t batch_size = static_cast<int64_t>(atom_counts.size());
    require_input(
        cells.size() == static_cast<size_t>(9 * batch_size),
        "cell must have shape (B, 3, 3)");
    require_input(
        pbc.size() == static_cast<size_t>(3 * batch_size),
        "pbc must have shape (B, 3)");
    PeriodicMetadata metadata;
    metadata.duals.resize(9 * batch_size, 0.0);
    metadata.image_offsets = {0};

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const double* cell = cells.data() + 9 * batch;
        for (int index = 0; index < 9; ++index) {
            require_input(
                std::isfinite(cell[index]),
                "cell must contain only finite values");
        }
        if (atom_counts[batch] == 0) {
            metadata.image_offsets.push_back(
                static_cast<int64_t>(metadata.image_shifts.size() / 3));
            continue;
        }
        int active_axes[3];
        int active_count = 0;
        for (int axis = 0; axis < 3; ++axis) {
            if (pbc[3 * batch + axis] != 0) {
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
                    require_input(
                        std::isfinite(value) &&
                            std::abs(value) <=
                                std::numeric_limits<double>::max(),
                        "active periodic cell dual is outside the float64 range");
                    metadata.duals[
                        9 * batch + 3 * cartesian + active_axes[column]] =
                        static_cast<double>(value);
                    reciprocal_norm_squared += value * value;
                }
                const long double repeat =
                    std::ceil(cutoff * std::sqrt(reciprocal_norm_squared));
                require_input(
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
            require_input(
                image_count <= kMaximumImageShifts / factor,
                "periodic image count exceeds the 2^24 resource limit");
            image_count *= factor;
        }
        require_input(
            static_cast<int64_t>(metadata.image_shifts.size() / 3) <=
                kMaximumImageShifts - image_count,
            "batched periodic image count exceeds the 2^24 resource limit");
        const int64_t total_image_count =
            static_cast<int64_t>(metadata.image_shifts.size() / 3) + image_count;
        metadata.image_shifts.reserve(static_cast<size_t>(3 * total_image_count));
        for (int64_t x = -repeats[0]; x <= repeats[0]; ++x) {
            for (int64_t y = -repeats[1]; y <= repeats[1]; ++y) {
                for (int64_t z = -repeats[2]; z <= repeats[2]; ++z) {
                    metadata.image_shifts.push_back(static_cast<int32_t>(x));
                    metadata.image_shifts.push_back(static_cast<int32_t>(y));
                    metadata.image_shifts.push_back(static_cast<int32_t>(z));
                }
            }
        }
        metadata.image_offsets.push_back(
            static_cast<int64_t>(metadata.image_shifts.size() / 3));
    }
    return metadata;
}
