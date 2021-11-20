#include <iostream>
#include <fstream>
#include <string>

#ifdef _WIN32
#  define EXPORT extern "C" __declspec(dllexport)
#else
#  define EXPORT extern "C"
#endif

#include <noarr/pipelines.hpp>
#include <noarr/structures_extended.hpp>

// typedef for the matrix structures and the value type stored in them
using value_type = std::uint32_t;
using MatrixStructureRows = noarr::vector<'m', noarr::vector<'n', noarr::scalar<value_type>>>;
using MatrixStructureColumns = noarr::vector<'n', noarr::vector<'m', noarr::scalar<value_type>>>;

using namespace noarr::pipelines;
using namespace std::string_literals;

namespace {

constexpr std::size_t BLOCK_SIZE = 32;

/**
 * @brief bottom case for the matrix multiplication, calls the kernel
 */
template<typename Matrix1, typename Matrix2, typename MatrixResult>
void matrix_multiply_impl(const Matrix1 &matrix1, const Matrix2 &matrix2, MatrixResult result)
{
	std::size_t height1 = matrix1.template get_length<'m'>();
	std::size_t width1 = matrix1.template get_length<'n'>();
	std::size_t height2 = matrix2.template get_length<'m'>();
	std::size_t width2 = matrix2.template get_length<'n'>();

	assert(width1 == height2);
	

	// dummy kernel begin

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

	// dummy kernel end
}

/**
 * @brief this overload transforms arrays for the result matrix into a matrix and then calls the bottom case
 */
template<typename Matrix1, typename Matrix2>
void matrix_multiply_impl(const Matrix1& matrix1, const Matrix2 &matrix2, char *data_results, const char *layout_results)
{
	std::size_t height1 = matrix1.template get_length<'m'>();
	std::size_t width2 = matrix2.template get_length<'n'>();

	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(height1), noarr::set_length<'n'>(width2));

	if (layout_results == "rows"s) {
		auto matrix_results = noarr::make_bag(MatrixStructureRows() | set_dimensions, data_results);

		matrix_multiply_impl(matrix1, matrix2, matrix_results);
	} else if (layout_results == "columns"s) {
		auto matrix_results = noarr::make_bag(MatrixStructureColumns() | set_dimensions, data_results);

		matrix_multiply_impl(matrix1, matrix2, matrix_results);
	}
}

/**
 * @brief this overload transforms arrays for the second matrix into a matrix and then propagates the process further
 */
template<typename Matrix1>
void matrix_multiply_impl(
	const Matrix1& matrix1,
	const int height2, const int width2, const char *data2, const char *layout2,
	char *data_results, const char *layout_results)
{
	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(height2), noarr::set_length<'n'>(width2));

	if (layout2 == "rows"s) {
		auto matrix2 = noarr::make_bag(MatrixStructureRows() | set_dimensions, (char *)data2);

		matrix_multiply_impl(matrix1, matrix2, data_results, layout_results);
	} else if (layout2 == "columns"s) {
		auto matrix2 = noarr::make_bag(MatrixStructureColumns() | set_dimensions, (char *)data2);

		matrix_multiply_impl(matrix1, matrix2, data_results, layout_results);
	}
}

/**
 * @brief this overload transforms arrays for the first matrix into a matrix and then propagates the process further
 */
void matrix_multiply(
	const int height1, const int width1, const char *data1, const char *layout1,
	const int height2, const int width2, const char *data2, const char *layout2,
	char *data_results, const char *layout_results)
{
	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(height1), noarr::set_length<'n'>(width1));

	if (layout1 == "rows"s) {
		auto matrix1 = noarr::make_bag(MatrixStructureRows() | set_dimensions, (char *)data1);

		matrix_multiply_impl(matrix1, height2, width2, data2, layout2, data_results, layout_results);
	} else if (layout1 == "columns"s) {
		auto matrix1 = noarr::make_bag(MatrixStructureColumns() | set_dimensions, (char *)data1);

		matrix_multiply_impl(matrix1, height2, width2, data2, layout2, data_results, layout_results);
	}
}

template<typename Matrix>
void matrix_print_impl(Matrix &matrix) {
	std::size_t height = matrix.template get_length<'m'>();
	std::size_t width = matrix.template get_length<'n'>();

	for (std::size_t i = 0; i < height; i++) {
		for (std::size_t j = 0; j < width; j++) {
			std::cout << matrix.template at<'m','n'>(i, j) << ' ';
		}

		std::cout << std::endl;
	}
	
	std::cout << std::endl;
}

void matrix_print(const int height, const int width, const char *data, const char *layout)
{
	auto set_dimensions = noarr::compose(noarr::set_length<'m'>(height), noarr::set_length<'n'>(width));

	if (layout == "rows"s) {
		auto matrix = noarr::make_bag(MatrixStructureRows() | set_dimensions, (char *)data);

		matrix_print_impl(matrix);
	} else if (layout == "columns"s) {
		auto matrix = noarr::make_bag(MatrixStructureColumns() | set_dimensions, (char *)data);

		matrix_print_impl(matrix);
	}
}

} // namespace

EXPORT
void matrix_multiply_demo(int *n_matrices, char **matrices, int *heights, int *widths) {

    // we want to simulate a GPU and corresponding allocations and memory
    // transfers to demonstrate the full capability of the framework
    HardwareManager::default_manager().register_dummy_gpu();

	/////////////////////////
	// Define the pipeline //
	/////////////////////////

	int max_size = 0;

	for (int i = *n_matrices; i-- > 0;) {
		int size = widths[i] * heights[i];

		if (max_size < size)
			max_size = size;
	}

	// compute nodes perform computation and access data in hubs
	auto reader = LambdaComputeNode("reader");
	auto multiplicator = LambdaComputeNode("multiplicator");
	auto writer = LambdaComputeNode("writer");

	// hubs store data and provide it to compute nodes,
	// they act as queues that can be written to and consumed from
	// (one piece of data in the queue is called a chunk and is represented
	// by a set of envelopes with the same content on different devices)
	auto reader_hub = Hub<std::size_t, char>(max_size * sizeof(value_type));
	auto accumulator_hub = Hub<std::size_t, char>(max_size * sizeof(value_type));
	auto writer_hub = Hub<std::size_t, char>(max_size * sizeof(value_type));

	// give each hub two envelopes (individual data holders) for each device
	// to rotate them and thus overlay writing, reading and transfer operations
	reader_hub.allocate_envelope(Device::HOST_INDEX);
	reader_hub.allocate_envelope(Device::DUMMY_GPU_INDEX);
	writer_hub.allocate_envelope(Device::HOST_INDEX);
	writer_hub.allocate_envelope(Device::DUMMY_GPU_INDEX);
	accumulator_hub.allocate_envelope(Device::DUMMY_GPU_INDEX);

	int finished = 0;

	// the reader wants to access the reader_hub data
	// and it wants to produce new chunks of data,
	auto& reader_link = reader.link(reader_hub.to_produce(
		Device::HOST_INDEX
	));

	// normally a compute node is advanced when all links have envelopes ready,
	// but the reader has to decide to stop producing when the file has been read
	reader.can_advance([&](){
		return finished < *n_matrices;
	});

	reader.advance([&](){
		std::ifstream matrix_file(matrices[finished]);

		std::size_t size = widths[finished] * heights[finished] * sizeof(value_type);

		reader_link.envelope->structure = size;

		matrix_file.read(reader_link.envelope->buffer, size);
		// in production code, we should check more cases

		// the advance method assumes you start an asynchronous operation that
		// will signal its completion by calling back
		reader.callback();
	});

	// Define the multiplicator
	// ----------------------

	// the multiplicator wants to access both hubs,
	// consuming chunks from one and producing chunks into the other
	//
	// it also pretends to call a gpu kernel, so it wants the data to be located
	// on the dummy gpu device
	auto& multiplicator_input_link = multiplicator.link(
		reader_hub.to_consume(Device::DUMMY_GPU_INDEX)
	);

	auto& multiplicator_output_link = multiplicator.link(
		writer_hub.to_produce(Device::DUMMY_GPU_INDEX)
	);

	auto& multiplicator_accumulator_link = multiplicator.link(
		accumulator_hub.to_modify(Device::DUMMY_GPU_INDEX)
	);

	accumulator_hub.push_new_chunk(Device::DUMMY_GPU_INDEX);

	std::size_t output_height;
	std::size_t output_width;

	multiplicator.advance([&](){
		// TL;DR: multiplies accumulator matrix with the input matrix from their respective envelopes
		// and stores the result matrix into the output envelope and accumulator envelope

		// useful shortcuts
		auto* in = multiplicator_input_link.envelope;
		auto* out = multiplicator_output_link.envelope;
		auto* accu = multiplicator_accumulator_link.envelope;

		std::size_t size = in->structure;


		if (finished == 0) {
			// first iteration, we just copy the read matrix into the output matrix

			std::memcpy(out->buffer, in->buffer, size);

			output_height = heights[0];
			output_width = widths[0];
		} else {
			// here we call the multiplication implementation,
			// multiplying accumulator and input into output

			matrix_multiply(
				output_height, output_width, accu->buffer, "rows",
				heights[finished], widths[finished], in->buffer, "rows",
				out->buffer, "rows");

			output_width = widths[finished];
		}

		// here we set output structure (size) and copy it to accumulator
		out->structure = output_width * output_height * sizeof(value_type);
		accu->structure = out->structure;
		std::memcpy(accu->buffer, out->buffer, size);

		// NOTE: we do not need to commit now,
		// as both links have autocommit enabled

		// the advance method assumes you start an asynchronous operation that
		// will signal its completion by calling back
		++finished;
		multiplicator.callback();
	});


	// Define the writer
	// -----------------

	// the writer wants to consume chunks from the writer_hub
	auto& writer_link = writer.link(writer_hub.to_consume(Device::HOST_INDEX));

	writer.advance([&](){
		// TL;DR: prints the contents of the output matrix

		if (finished == *n_matrices)
			matrix_print(output_height, output_width, writer_link.envelope->buffer, "rows");
		else 
			std::cout << finished * 100 / *n_matrices << '%' << std::endl;

		// the advance method assumes you start an asynchronous operation that
		// will signal its completion by calling back
		writer.callback();
	});


	////////////////////////////////////
	// Run the pipeline to completion //
	////////////////////////////////////

	// the scheduler tries to eagerly advance all nodes that are idle and
	// let themselves be advanced (have all links ready and
	// can_advance returns true)
	DebuggingScheduler scheduler;
	scheduler << reader
	          << multiplicator
	          << writer
	          << reader_hub // hubs are also pipeline nodes,
	          << writer_hub // they just serve data management
	          << accumulator_hub; // they just serve data management

	scheduler.run();
}
