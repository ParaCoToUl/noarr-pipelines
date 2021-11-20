#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#include <noarr/pipelines.hpp>
#include <noarr/cuda-pipelines.hpp>

using namespace noarr::pipelines;

// the uppercasing kernel called from the cuda compute node
__global__ void toupper_kernel(char* in, char* out, std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    switch (in[i]) {
        case 'a': out[i] = 'A'; break;
        case 'b': out[i] = 'B'; break;
        case 'c': out[i] = 'C'; break;
        case 'd': out[i] = 'D'; break;
        case 'e': out[i] = 'E'; break;
        case 'f': out[i] = 'F'; break;
        case 'g': out[i] = 'G'; break;
        case 'h': out[i] = 'H'; break;
        case 'i': out[i] = 'I'; break;
        case 'j': out[i] = 'J'; break;
        case 'k': out[i] = 'K'; break;
        case 'l': out[i] = 'L'; break;
        case 'm': out[i] = 'M'; break;
        case 'n': out[i] = 'N'; break;
        case 'o': out[i] = 'O'; break;
        case 'p': out[i] = 'P'; break;
        case 'q': out[i] = 'Q'; break;
        case 'r': out[i] = 'R'; break;
        case 's': out[i] = 'S'; break;
        case 't': out[i] = 'T'; break;
        case 'u': out[i] = 'U'; break;
        case 'v': out[i] = 'V'; break;
        case 'w': out[i] = 'W'; break;
        case 'x': out[i] = 'X'; break;
        case 'y': out[i] = 'Y'; break;
        case 'z': out[i] = 'Z'; break;
        default: out[i] = in[i]; break;
    }
}

int main(int argc, char* argv[]) {

    // we want to use the CUDA extension to noarr pipelines
    CudaPipelines::register_extension();


    ///////////////////////////////////////
    // Parse arguments and open the file //
    ///////////////////////////////////////

    std::string filename;

    if (argc >= 2) {
        filename = std::string(argv[1]);
    } else {
        std::cout << "Usage:" << std::endl;
        std::cout << "    uppercase-cuda [filename]" << std::endl;
        std::cout << std::endl;
        std::cout << "Since you are probably in the build folder, you can try:" << std::endl;
        std::cout << "    uppercase-cuda ../input.txt" << std::endl;
        return 1;
    }

    std::ifstream file(filename);


    /////////////////////////
    // Define the pipeline //
    /////////////////////////

    const std::size_t BUFFER_SIZE = 1024;

    // compute nodes perform computation and access data in hubs
    auto reader = LambdaComputeNode("reader");
    auto capitalizer = LambdaCudaComputeNode("capitalizer");
    auto writer = LambdaComputeNode("writer");

    // hubs store data and provide it to compute nodes,
    // they act as queues that can be written to and consumed from
    // (one piece of data in the queue is called a chunk and is represented
    // by a set of envelopes with the same content on different devices)
    auto reader_hub = Hub<std::size_t, char>(BUFFER_SIZE);
    auto writer_hub = Hub<std::size_t, char>(BUFFER_SIZE);

    // give each hub two envelopes (individual data holders) for each device
    // to rotate them and thus overlay writing, reading and transfer operations
    reader_hub.allocate_envelopes(Device::HOST_INDEX, 2);
    reader_hub.allocate_envelopes(Device::DEVICE_INDEX, 2);
    writer_hub.allocate_envelopes(Device::HOST_INDEX, 2);
    writer_hub.allocate_envelopes(Device::DEVICE_INDEX, 2);

    // NOTE: We allocate all data on the host (the CPU), as this is
    // a simple demonstration. But a compute node may request data on
    // a different device and hubs would handle data transfer for us.


    // Define the reader
    // -----------------

    bool reading_finished = false;

    // the reader wants to access the reder_hub data
    // and it wants to produce new chunks of data,
    // but it does not want to produce always, so the autocommit is disabled
    auto& reader_link = reader.link(reader_hub.to_produce(
        Device::HOST_INDEX, // reader produces data on the CPU
        false // disabled autocommit
    ));

    // normally a compute node is advanced when all links have envelopes ready,
    // but the reader has to decide to stop producing when the file has been read
    reader.can_advance([&](){
        return !reading_finished;
    });

    reader.advance([&](){
        // TL;DR: Transfer a line of text from the open file, to an envelope
        // in the output link. Commit only if we have actually read a line.

        std::string line;
        bool got_a_line = !! std::getline(file, line);

        if (got_a_line) {
            std::size_t bytes_to_copy = line.size();
            if (bytes_to_copy > BUFFER_SIZE) {
                bytes_to_copy = BUFFER_SIZE;
                std::cerr << "Truncated a line that was too long." << std::endl;
            }

            // here we write a new chunk into the hub and then commit our action
            // (we commit that we did produce a new chunk of data)
            reader_link.envelope->structure = bytes_to_copy;
            line.copy(reader_link.envelope->buffer, bytes_to_copy);
            reader_link.commit();
        } else {
            reading_finished = true;
        }

        // the advance method assumes you start an asynchronous operation that
        // will signal its completion by calling back
        reader.callback();
    });


    // Define the capitalizer
    // ----------------------

    // the capitalizer wants to access both hubs,
    // consuming chunks from one and producing chunks into the other
    //
    // it also pretends to be a gpu kernel, so it wants the data to be located
    // on the gpu device
    auto& capitalizer_input_link = capitalizer.link(
        reader_hub.to_consume(Device::DEVICE_INDEX)
    );
    auto& capitalizer_output_link = capitalizer.link(
        writer_hub.to_produce(Device::DEVICE_INDEX)
    );

    capitalizer.advance_cuda([&](cudaStream_t stream){
        // TL;DR: Transfer the data from the envelope in the input link
        // to the envelope in the output link and capitalize the string.

        std::size_t size = capitalizer_input_link.envelope->structure;
        
        char* in = capitalizer_input_link.envelope->buffer;
        char* out = capitalizer_output_link.envelope->buffer;

        if (size > 0) {
            constexpr std::size_t BLOCK_SIZE = 128;
            toupper_kernel<<<
                (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream
            >>>(in, out, size);
            NOARR_CUCH(cudaGetLastError());
        }
        
        capitalizer_output_link.envelope->structure = size;

        // NOTE: we do not need to commit now,
        // as both links have autocommit enabled

        // NOTE: we do not need to call the callback,
        // as the CudaComputeNode does that automatically
        // when the provided CUDA stream becomes empty
    });


    // Define the writer
    // -----------------

    // the writer wants to consume chunks from the writer_hub
    auto& writer_link = writer.link(writer_hub.to_consume(Device::HOST_INDEX));

    writer.advance([&](){
        // TL;DR: Transfer the line of text from the given chunk
        // to the standard output.

        std::size_t size = writer_link.envelope->structure;
        char* in = writer_link.envelope->buffer;

        std::cout << std::string(in, size) << std::endl;

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
    SimpleScheduler scheduler;
    scheduler << reader
              << capitalizer
              << writer
              << reader_hub // hubs are also pipeline nodes,
              << writer_hub; // they just serve data management

    scheduler.run();

    return 0;
}
