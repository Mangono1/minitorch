#include "minitorch/tensor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace minitorch;

static void expect_close(
    float actual,
    float expected,
    const char* name
) {
    if (std::fabs(actual - expected) > 0.0001f) {
        throw std::runtime_error(
            std::string(name) +
            " expected " +
            std::to_string(expected) +
            " but got " +
            std::to_string(actual)
        );
    }
}

static void test_basic_operations() {
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

    expect_close(sum.item(0), 6.0f, "add[0]");
    expect_close(sum.item(3), 12.0f, "add[3]");

    expect_close(product.item(0), 5.0f, "multiply[0]");
    expect_close(product.item(3), 32.0f, "multiply[3]");

    expect_close(matrix.item(0), 19.0f, "matmul[0]");
    expect_close(matrix.item(1), 22.0f, "matmul[1]");
    expect_close(matrix.item(2), 43.0f, "matmul[2]");
    expect_close(matrix.item(3), 50.0f, "matmul[3]");

    std::cout << "Basic operations: PASS\n";
}

static void test_add_autograd() {
    Tensor a(
        {
            1.0f,
            2.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Tensor b(
        {
            3.0f,
            4.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Tensor y = a.add(b);
    Tensor loss = y.sum();

    loss.backward();

    Tensor ga = a.grad();
    Tensor gb = b.grad();

    expect_close(ga.item(0), 1.0f, "add grad a[0]");
    expect_close(ga.item(1), 1.0f, "add grad a[1]");

    expect_close(gb.item(0), 1.0f, "add grad b[0]");
    expect_close(gb.item(1), 1.0f, "add grad b[1]");

    std::cout << "Add autograd: PASS\n";
}

static void test_multiply_autograd() {
    Tensor a(
        {
            2.0f,
            3.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Tensor b(
        {
            4.0f,
            5.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Tensor y = a.multiply(b);
    Tensor loss = y.sum();

    loss.backward();

    Tensor ga = a.grad();
    Tensor gb = b.grad();

    expect_close(ga.item(0), 4.0f, "multiply grad a[0]");
    expect_close(ga.item(1), 5.0f, "multiply grad a[1]");

    expect_close(gb.item(0), 2.0f, "multiply grad b[0]");
    expect_close(gb.item(1), 3.0f, "multiply grad b[1]");

    std::cout << "Multiply autograd: PASS\n";
}

static void test_matmul_autograd() {
    Tensor a(
        {
            1.0f, 2.0f,
            3.0f, 4.0f
        },
        {2, 2},
        Device(DeviceType::CPU),
        true
    );

    Tensor b(
        {
            5.0f, 6.0f,
            7.0f, 8.0f
        },
        {2, 2},
        Device(DeviceType::CPU),
        true
    );

    Tensor y = a.matmul(b);
    Tensor loss = y.sum();

    loss.backward();

    Tensor ga = a.grad();
    Tensor gb = b.grad();

    /*
        y = a @ b
        loss = sum(y)

        dL/dA:
        [11, 15]
        [11, 15]

        dL/dB:
        [4, 4]
        [6, 6]
    */

    expect_close(ga.item(0), 11.0f, "matmul grad a[0]");
    expect_close(ga.item(1), 15.0f, "matmul grad a[1]");
    expect_close(ga.item(2), 11.0f, "matmul grad a[2]");
    expect_close(ga.item(3), 15.0f, "matmul grad a[3]");

    expect_close(gb.item(0), 4.0f, "matmul grad b[0]");
    expect_close(gb.item(1), 4.0f, "matmul grad b[1]");
    expect_close(gb.item(2), 6.0f, "matmul grad b[2]");
    expect_close(gb.item(3), 6.0f, "matmul grad b[3]");

    std::cout << "Matmul autograd: PASS\n";
}

static void test_gradient_accumulation() {
    Tensor x(
        {
            2.0f,
            3.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    /*
        y = x * x
        loss = sum(y)

        dL/dx = 2x
        => [4, 6]
    */

    Tensor y = x.multiply(x);
    Tensor loss = y.sum();

    loss.backward();

    Tensor gx = x.grad();

    expect_close(gx.item(0), 4.0f, "accumulated grad[0]");
    expect_close(gx.item(1), 6.0f, "accumulated grad[1]");

    std::cout << "Gradient accumulation: PASS\n";
}

static void test_zero_grad() {
    Tensor x(
        {
            2.0f,
            3.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Tensor loss =
        x.multiply(x).sum();

    loss.backward();

    x.zero_grad();

    Tensor gx = x.grad();

    expect_close(gx.item(0), 0.0f, "zero_grad[0]");
    expect_close(gx.item(1), 0.0f, "zero_grad[1]");

    std::cout << "zero_grad: PASS\n";
}

int main() {
    try {
        std::cout << "====================================\n";
        std::cout << " MiniTorch V0.2 Autograd Test\n";
        std::cout << "====================================\n\n";

        test_basic_operations();
        test_add_autograd();
        test_multiply_autograd();
        test_matmul_autograd();
        test_gradient_accumulation();
        test_zero_grad();

        std::cout << "\n====================================\n";
        std::cout << " ALL TESTS PASSED\n";
        std::cout << "====================================\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "\nTEST FAILED:\n";
        std::cerr << error.what() << "\n";
        return 1;
    }
}
