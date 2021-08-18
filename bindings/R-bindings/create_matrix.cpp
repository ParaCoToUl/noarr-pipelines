#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

template<typename Scalar>
void print(std::ostream& out, Scalar number) {
    for (char *c = (char *)&number; c < (char *)(&number + 1); ++c)
        out.put(*c);
}

int main(int argc, char *argv[]) {
    using std::string_literals::operator""s;
    using value_type = std::uint32_t;

    if (argc != 2) {
        std::cerr << "expecting the n of the square n×n matrix as the argument" << std::endl;
        return 1;
    }

    std::istringstream input(argv[1]);
    std::size_t size;

    input >> size;

    std::ofstream matrix("matrix"s + std::to_string(size) + ".data", std::ios::binary);

    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            if (j == i + 1)
                print(matrix, (value_type)1);
            else if (j == 0 && i == size - 1)
                print(matrix, (value_type)2);
            else
                print(matrix, (value_type)0);
        }
    }
}