#include <assert.h>
#include <iostream>
#include <chrono>
#include "precompute_io.hpp"
#include "vector/multiplication.hpp"
#include "reduction/montgomery.hpp"
// Initialized using -DPYREPLACE_LIMBS1=count from python script during compilation

constexpr int LIMBS1 = PYREPLACE_LIMBS1;
constexpr int LIMBS2 = PYREPLACE_LIMBS2;
constexpr bool q_correct = PYREPLACE_Q_CORRECT;
constexpr int batch_size = 1;

int main(int argc, char* argv[]) {
    assert(argc > 2);
    int pow = std::atoi(argv[1]);
    PrecomputeIO io = PrecomputeIO(argv[2]);
    auto multiplier = PYREPLACE_METHOD_NAME<PYREPLACE_REDUCTION_NAME, LIMBS1, LIMBS2>(io);
    AVXVector<LIMBS1> a1;
    AVXVector<LIMBS2> a2;
    AVXVector<LIMBS1> b1;
    AVXVector<LIMBS2> b2;
    io.read(a1);
    io.read(a2);
    io.read(b1);
    io.read(b2);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < pow; i++) {
        AVXVector<LIMBS1> tmp1;
        AVXVector<LIMBS2> tmp2;
        multiplier.mul_reduce(a1, a2, b1, b2, tmp1, tmp2);
        a1 = tmp1;
        a2 = tmp2;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    a1.print("out1");
    a2.print("out2");
}