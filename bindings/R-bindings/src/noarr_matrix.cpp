#include <cassert>
#include <iostream>
#include <string>

#include <R.h>
#include <R_ext/Rdynload.h>

#include "../noarr-structures/noarr/include/noarr/structures_extended.hpp"

using MatrixStructureRows = noarr::vector<'m', noarr::vector<'n', noarr::scalar<int>>>;
using MatrixStructureColumns = noarr::vector<'n', noarr::vector<'m', noarr::scalar<int>>>;

/**
 * @brief Takes 2 noarr matrices and multiplyes them.
 *
 * @tparam matrix1: First noarr matrix
 * @tparam matrix2: Second noarr matrix
 * @tparam structure: Structure defining structure to be used by result noarr matrix
 * @return Matrix noarr matrix created from source noarr matrices
 */
template<typename Matrix1, typename Matrix2, typename MatrixResult>
void matrix_multiply_impl(const Matrix1& matrix1, const Matrix2& matrix2, MatrixResult& result)
{
	std::size_t height1 = matrix1.template get_length<'m'>();
	std::size_t width1 = matrix1.template get_length<'n'>();
	std::size_t height2 = matrix2.template get_length<'m'>();
	std::size_t width2 = matrix2.template get_length<'n'>();

	assert(width1 == height2);

	for (std::size_t i = 0; i < width2; i++)
	{
		for (std::size_t j = 0; j < height1; j++)
		{
			int sum = 0;

			for (std::size_t k = 0; k < width1; k++)
			{
				const int& value1 = matrix1.template at<'n', 'm'>(k, j);
				const int& value2 = matrix2.template at<'n', 'm'>(i, k);

				sum += value1 * value2;
			}

			result.template at<'n', 'm'>(i, j) = sum;
		}
	}
}

template<typename Matrix1, typename Matrix2>
void matrix_multiply_impl(const Matrix1& matrix1, const Matrix2 &matrix2, int *data_results, const char **layout_results)
{
	using std::string_literals::operator""s;

	std::size_t height1 = matrix1.template get_length<'m'>();
	std::size_t width2 = matrix2.template get_length<'n'>();

	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(height1), noarr::set_length<'n'>(width2));

	if (*layout_results == "rows"s) {
		auto matrix_results = noarr::make_bag(MatrixStructureRows() | set_dimensions, (char *)data_results);

		matrix_multiply_impl(matrix1, matrix2, matrix_results);
	} else if (*layout_results == "columns"s) {
		auto matrix_results = noarr::make_bag(MatrixStructureColumns() | set_dimensions, (char *)data_results);

		matrix_multiply_impl(matrix1, matrix2, matrix_results);
	}
}

template<typename Matrix1>
void matrix_multiply_impl(
	const Matrix1& matrix1,
	const int *height2, const int *width2, const int *data2, const char **layout2,
	int *data_results, const char **layout_results)
{
	using std::string_literals::operator""s;

	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(*height2), noarr::set_length<'n'>(*width2));

	if (*layout2 == "rows"s) {
		auto matrix2 = noarr::make_bag(MatrixStructureRows() | set_dimensions, (char *)data2);

		matrix_multiply_impl(matrix1, matrix2, data_results, layout_results);
	} else if (*layout2 == "columns"s) {
		auto matrix2 = noarr::make_bag(MatrixStructureColumns() | set_dimensions, (char *)data2);

		matrix_multiply_impl(matrix1, matrix2, data_results, layout_results);
	}
}

extern "C" {


void matrix_multiply(
	const int *height1, const int *width1, const int *data1, const char **layout1,
	const int *height2, const int *width2, const int *data2, const char **layout2,
	int *data_results, const char **layout_results)
{
	using std::string_literals::operator""s;

	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(*height1), noarr::set_length<'n'>(*width1));

	if (*layout1 == "rows"s) {
		auto matrix1 = noarr::make_bag(MatrixStructureRows() | set_dimensions, (char *)data1);

		matrix_multiply_impl(matrix1, height2, width2, data2, layout2, data_results, layout_results);
	} else if (*layout1 == "columns"s) {
		auto matrix1 = noarr::make_bag(MatrixStructureColumns() | set_dimensions, (char *)data1);

		matrix_multiply_impl(matrix1, height2, width2, data2, layout2, data_results, layout_results);
	}
}

static const R_CMethodDef cMethods[] = {
	{ "matrix_multiply", (DL_FUNC)&matrix_multiply, 10 },
	{ NULL, NULL, 0 }
};

void R_init_Noarr_matrix(DllInfo *info)
{
	R_registerRoutines(info, cMethods, NULL, NULL, NULL);
	R_useDynamicSymbols(info, FALSE);
}

} // extern "C"
