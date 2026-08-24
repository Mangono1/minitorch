#include "minitorch/tensor.h"

#include <iostream>

using namespace minitorch;

int main() {
    Tensor a(
        {
            1.0f, 2.0f,
            3.0f, 4.0f
        },
        {2, 2}
    );

    Tensor b(
        {
            5.0f, 6.0f,
            7.0f, 8.0f
        },
        {2, 2}
    );

    Tensor sum = a.add(b);
    Tensor product = a.multiply(b);
    Tensor matrix = a.matmul(b);

    std::cout << "A = " << a.repr() << "\n";
    std::cout << "B = " << b.repr() << "\n";
    std::cout << "A + B = " << sum.repr() << "\n";
    std::cout << "A * B = " << product.repr() << "\n";
    std::cout << "A @ B = " << matrix.repr() << "\n";

    return 0;
}
