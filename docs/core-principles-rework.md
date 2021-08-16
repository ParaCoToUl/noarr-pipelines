# Core principles


## Basics

The noarr pipelines library aims to provide a framework for building computational pipelines for GPGPU computing. These pipelines are designed to process data that would not fit into GPU memory in one batch and needs to be streamed. The user has to define a way to break the data down to a sequence of smaller chunks, process individual chunks and then re-assemble the result from those chunks.

The pipeline is composed of *nodes* - independent units that perform a piece of the computation. Imagine that we have a text file that we want to convert to another file with all the letters capitalized. The entire file would not fit in memory, but we can say that one line of text easily would. We could process the file line by line, thereby streaming the whole process. The pipeline would have one *node* responsible for reading lines of text, another *node* for performing the capitalization and another one for writing capitalized lines to an output file. This kind of separation lets all nodes run concurrently and increases the overall throughput of the system. We could also imagine, that the capitalization process was expensive and we would like to run it on the GPU.

> **Note:** The described task is implemented in the `upcase` example and can be found [here](../examples/upcase).

We described a scenario where we have *compute nodes* on different devices (GPU, CPU) but we need a way to transfer data between them. For this purpose we will create a special type of node called a *hub*. A hub can be imagined as a queue of chunks of data (lines of text in our example). We can produce new chunks by writing into it and consume chunks by reading from it. We can then put one *hub* in between each of our *compute nodes* to serve as the queue in the classic producer-consumer pattern. This gives us the following pipeline:

    [reader] --> {reader_hub} --> [capitalizer] --> {writer_hub} --> [writer]
       |                                                                |
    input.txt                                                      output.txt

*Compute nodes* will be joined to *hubs* by something called a *link*. A link can then hold various metadata used by both sides, like its type (producing / consuming chunks in the hub) or the device on which the *compute node* expects the data to be received. Remember that we imagined the capitalizer to be a GPU kernel. It wants to receive data already located in the GPU memory. The *hub* that provides the data may already handle the memory transfer for us.

TODO...

- how does a compute node "compute"? (the advance method and concurrency)
- what is an envelope
- what **exactly** is a chunk
- how **exactly** is the data accessed via the link (user's perspective - an envelope)
- scheduler in detail (scheduling algorithm)
