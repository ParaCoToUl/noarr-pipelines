# Noarr pipelines tests

The noarr pipelines repository contains the following two folders with tests:

- `pipelines` Contains CPU-only tests for the noarr pipelines framework. This is where the core functionality is tested.
- `cuda-pipelines` Contains GPU tests for the CUDA extension to the framework. Only GPU-related logic is tested here.

Tests were created as new features were added to validate that these features work as expected. Therefore the tests do not care about coverage from the perspective of lines of code or classes, but the perspective of features. There is no `hub` test, but there are tests that make sure a hub can be used in various pipeline situations (e.g. producer-consumer). There were no features added, that would not be used in a test or an example, so the codebase should be covered well enough to let us easily refactor the code.
