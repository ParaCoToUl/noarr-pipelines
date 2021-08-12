#include <cassert>

#include <R.h>
#include <R_ext/Rdynload.h>

#include "../../../../noarr/noarr/include/noarr/structures_extended.hpp"

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
template<typename Matrix, typename Matrix2, typename MatrixResult>
void matrix_multiply(Matrix& matrix1, Matrix2& matrix2, MatrixResult& result)
{
	int x1_size = matrix1.template get_length<'n'>();
	int y1_size = matrix1.template get_length<'m'>();
	int x2_size = matrix2.template get_length<'n'>();
	int y2_size = matrix2.template get_length<'m'>();

	assert(x1_size == y2_size);

	for (int i = 0; i < x2_size; i++)
		for (int j = 0; j < y1_size; j++)
		{
			int sum = 0;

			for (int k = 0; k < x1_size; k++)
			{
				int& value1 = matrix1.template at<'n', 'm'>(k, j);
				int& value2 = matrix2.template at<'n', 'm'>(i, k);

				sum += value1 * value2;
			}

			result.template at<'n', 'm'>(i, j) = sum;
		}
}

extern "C" {
    void multiply_rows_matrix_by_rows_matrix(int *height, int *width, int *data1, int *data2, int *data_results)
    {
        auto matrix1 = noarr::make_bag(MatrixStructureRows() | noarr::set_length<'m'>(*height) | noarr::set_length<'n'>(*width), (char *)data1);
        auto matrix2 = noarr::make_bag(MatrixStructureRows() | noarr::set_length<'m'>(*width) | noarr::set_length<'n'>(*height), (char *)data2);
        auto matrix_result = noarr::make_bag(MatrixStructureRows() | noarr::set_length<'m'>(*height) | noarr::set_length<'n'>(*height), (char *)data_results);

        matrix_multiply(matrix1, matrix2, matrix_result);
    }

    static const R_CMethodDef cMethods[] = {
        { "multiply_rows_matrix_by_rows_matrix", (DL_FUNC)&multiply_rows_matrix_by_rows_matrix, 5 },
        { NULL, NULL, 0 }
    };

    void R_init_Matrix(DllInfo *info)
    {
        R_registerRoutines(info, cMethods, NULL, NULL, NULL);
        R_useDynamicSymbols(info, FALSE);
    }
}
