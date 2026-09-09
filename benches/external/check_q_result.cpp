#include "q_stress_input.hpp"
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: check-q-result ORACLE_FILE RESULT_FILE");
        QInput input(argv[1]);
        std::ifstream stream(argv[2]);
        if (!stream) throw std::runtime_error("cannot read reconstructed expression");
        std::string expression((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        check_q_identity(input, expression);
        std::cout << "exact Q identity verified\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
