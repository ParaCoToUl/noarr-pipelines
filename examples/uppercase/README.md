# Uppercase example

This folder contains a standalone cmake project that demonstrates the absolute basics of noarr pipelines. It features a pipeline, designed to read a file line by line, capitalize each line and print it to the screen. It demonstrates the producer-consumer pattern that is easy to build using noarr pipelines.


## Compilation

Compile the example using cmake inside a folder called `build`:

```bash
# create compilation folder and enter it
mkdir build
cd build

# create build files
cmake ..

# build the project using build files
cmake --build .

# run the compiled example on the input file
./uppercase ../input.txt
```


## License

This folder is a part of the noarr pipelines repository and thus belongs under its MIT license as well.
