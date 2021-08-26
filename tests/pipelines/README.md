# Pipelines tests

To see the individual tests, open the [`tests` folder](tests).


## Running tests

In the terminal (Linux bash, Windows Cygwin, or Gitbash), run the following commands:

```sh
# create and enter the folder that will contain the build files
mkdir build
cd build

# generates build files for your platform
cmake ..

# builds the project using previously generated build files
cmake --build .

# run the built executable
# (this step differs by platform, this example is for linux)
./test-runner
```
