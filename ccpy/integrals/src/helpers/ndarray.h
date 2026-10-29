#pragma once

#include <array>
#include <numeric>
#include <memory>
#include <vector>
#include <complex>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/numeric.hpp"

namespace nb = nanobind;

// Aliases for ndarray types used in ccpy
// No contiguity requirement
using np_vector_int = nb::ndarray<nb::numpy, int, nb::ndim<1>>;
using np_matrix_int = nb::ndarray<nb::numpy, int, nb::ndim<2>>;
using np_vector = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using np_matrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using np_tensor3 = nb::ndarray<nb::numpy, double, nb::ndim<3>>;
using np_tensor4 = nb::ndarray<nb::numpy, double, nb::ndim<4>>;
using np_tensor5 = nb::ndarray<nb::numpy, double, nb::ndim<5>>;
using np_tensor6 = nb::ndarray<nb::numpy, double, nb::ndim<6>>;
using np_tensor7 = nb::ndarray<nb::numpy, double, nb::ndim<7>>;
using np_tensor8 = nb::ndarray<nb::numpy, double, nb::ndim<8>>;
using np_vector_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>;
using np_matrix_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>>;
using np_tensor3_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<3>>;
using np_tensor4_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<4>>;
using np_tensor5_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<5>>;
using np_tensor6_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<6>>;
using np_tensor7_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<7>>;
using np_tensor8_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<8>>;
// C-contiguous versions
using np_vector_c_int = nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::c_contig>;
using np_matrix_c_int = nb::ndarray<nb::numpy, int, nb::ndim<2>, nb::c_contig>;
using np_vector_c = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
using np_matrix_c = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using np_tensor3_c = nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::c_contig>;
using np_tensor4_c = nb::ndarray<nb::numpy, double, nb::ndim<4>, nb::c_contig>;
using np_tensor5_c = nb::ndarray<nb::numpy, double, nb::ndim<5>, nb::c_contig>;
using np_tensor6_c = nb::ndarray<nb::numpy, double, nb::ndim<6>, nb::c_contig>;
using np_tensor7_c = nb::ndarray<nb::numpy, double, nb::ndim<7>, nb::c_contig>;
using np_tensor8_c = nb::ndarray<nb::numpy, double, nb::ndim<8>, nb::c_contig>;
using np_vector_complex_c = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::c_contig>;
using np_matrix_complex_c = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>, nb::c_contig>;
using np_tensor3_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<3>, nb::c_contig>;
using np_tensor4_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<4>, nb::c_contig>;
using np_tensor5_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<5>, nb::c_contig>;
using np_tensor6_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<6>, nb::c_contig>;
using np_tensor7_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<7>, nb::c_contig>;
using np_tensor8_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<8>, nb::c_contig>;
// Fortran-contiguous versions
using np_vector_f_int = nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::f_contig>;
using np_matrix_f_int = nb::ndarray<nb::numpy, int, nb::ndim<2>, nb::f_contig>;
using np_vector_f = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::f_contig>;
using np_matrix_f = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::f_contig>;
using np_tensor3_f = nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::f_contig>;
using np_tensor4_f = nb::ndarray<nb::numpy, double, nb::ndim<4>, nb::f_contig>;
using np_tensor5_f = nb::ndarray<nb::numpy, double, nb::ndim<5>, nb::f_contig>;
using np_tensor6_f = nb::ndarray<nb::numpy, double, nb::ndim<6>, nb::f_contig>;
using np_tensor7_f = nb::ndarray<nb::numpy, double, nb::ndim<7>, nb::f_contig>;
using np_tensor8_f = nb::ndarray<nb::numpy, double, nb::ndim<8>, nb::f_contig>;
using np_vector_complex_f = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::f_contig>;
using np_matrix_complex_f = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>, nb::f_contig>;
using np_tensor3_complex_f =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<3>, nb::f_contig>;
using np_tensor4_complex_f =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<4>, nb::f_contig>;
using np_tensor5_complex_f =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<5>, nb::f_contig>;
using np_tensor6_complex_f =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<6>, nb::f_contig>;
using np_tensor7_complex_f =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<7>, nb::f_contig>;
using np_tensor8_complex_f =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<8>, nb::f_contig>;

template <typename Type, typename T, int N>
nb::ndarray<Type, T, nb::ndim<N>> make_ndarray(std::unique_ptr<std::vector<T>> vec,
                                               const std::array<size_t, N>& shape) {
    // raw pointer to the data
    T* data_ptr = vec->data();

    // release ownership of the vector onto the heap
    // so we can delete it when Python is done
    auto* heap_vec = vec.release(); // a std::vector<T>*

    // capsule will call delete on the vector when the array dies
    nb::capsule deleter(heap_vec, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });

    // construct the ndarray(view) with shape and the deleter capsule
    return nb::ndarray<Type, T, nb::ndim<N>>(data_ptr, static_cast<size_t>(N), shape.data(),
                                             std::move(deleter));
}

/// @brief Allocates the memory for an ndarray of the given shape and type.
/// @details The memory is allocated on the heap and will be freed when the ndarray
///          is deleted via a deleter capsule passed to the ndarray constructor.
/// @tparam Type The type of the ndarray (e.g. nb::numpy, nb::pytorch, etc.)
/// @tparam T The type of the data (e.g. double, float, etc.)
/// @tparam N The number of dimensions of the ndarray
/// @param shape The shape of the ndarray as an array of size N.
/// @details The shape is a list of size N, where N is the number of dimensions.
/// @return An ndarray of the given shape and type.
/// @note The ndarray is not initialized, so the data is not set to any value.
template <typename Type, typename T, int N, typename Order = void>
auto make_ndarray(const std::array<size_t, N>& shape) {
    // compute the number of elements in the array
    const auto size = ccpy::math::product(shape);

    // allocate a vector of the right size
    T* array_ptr = new T[size];

    // define a capsule to delete the data when the ndarray dies
    nb::capsule deleter(array_ptr, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    // construct the ndarray (a view on the array data) with shape and the deleter capsule
    if constexpr (std::is_void_v<Order>) {
        return nb::ndarray<Type, T, nb::ndim<N>>(array_ptr, static_cast<size_t>(N), shape.data(),
                                                 std::move(deleter));
    } else {
        return nb::ndarray<Type, T, nb::ndim<N>, Order>(array_ptr, static_cast<size_t>(N),
                                                        shape.data(), std::move(deleter));
    }
}

/// @brief Creates an ndarray of a given shape and type set to zero.
/// @details The memory is allocated on the heap and will be freed when the ndarray
///          is deleted via a deleter capsule passed to the ndarray constructor.
/// @tparam Type The type of the ndarray (e.g. nb::numpy, nb::pytorch, etc.)
/// @tparam T The type of the data (e.g. double, float, etc.)
/// @tparam N The number of dimensions of the ndarray
/// @tparam Order The memory layout of the ndarray (e.g. nb::c_contig, nb::f_contig, etc.)
/// @param shape The shape of the ndarray as an array of size N.
/// @details The shape is a list of size N, where N is the number of dimensions.
/// @return An ndarray of the given shape and type.
/// @note The ndarray is not initialized, so the data is not set to any value.
template <typename Type, typename T, int N, typename Order = void>
auto make_zeros(const std::array<size_t, N>& shape) {
    // allocate the ndarray
    auto array = make_ndarray<Type, T, N, Order>(shape);

    // get the data pointer and size
    auto data_ptr = array.data();
    auto size = ccpy::math::product(shape);

    // fill the data with zeros
    std::fill(data_ptr, data_ptr + size, static_cast<T>(0));

    return array;
}

template <typename Type, typename T, int N>
auto fortran_to_c(nb::ndarray<Type, T, nb::ndim<N>, nb::f_contig>& arr) {
    std::array<std::size_t, N> C_shape;
    std::array<std::int64_t, N> C_strides;
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        C_shape[i] = arr.shape(N - 1 - i);
        C_strides[i] = arr.stride(N - 1 - i);
    }
    T* data = arr.data();

    nb::object owner = arr.cast();

    return nb::ndarray<Type, T, nb::ndim<N>, nb::c_contig>(data, static_cast<std::size_t>(N), C_shape.data(), owner, C_strides.data());
}

template <typename Type, typename T, int N>
auto c_to_fortran(nb::ndarray<Type, T, nb::ndim<N>, nb::c_contig>& arr) {
    std::array<std::size_t, N> F_shape;
    std::array<std::int64_t, N> F_strides;
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        F_shape[i] = arr.shape(N - 1 - i);
        F_strides[i] = arr.stride(N - 1 - i);
    }
    T* data = arr.data();

    nb::object owner = arr.cast();

    return nb::ndarray<Type, T, nb::ndim<N>, nb::f_contig>(data, static_cast<std::size_t>(N), F_shape.data(), owner, F_strides.data());
}