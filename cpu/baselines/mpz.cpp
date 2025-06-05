#include <gmp.h>
#include <chrono>
#include <cassert>
#include <stdlib.h>
#include <iostream>

typedef struct Scratch {
    mpz_t s;
    // mpz_t u;
    mpz_t l;
    mpz_t w;
    mpz_t q;
} Scratch;

template <int mont_bits>
void init_scratch(Scratch &s) {
    mpz_init2(s.s, mont_bits * 2);
    // mpz_init2(s.u, mont_bits);
    mpz_init2(s.l, mont_bits);
    mpz_init2(s.w, mont_bits * 2);
    mpz_init2(s.q, mont_bits);
}

// using Montgomery avoids long division, at the expense of three big integer multiplications (versus one mult + division)
template<int mont_bits>
void karatsuba_mont(mpz_t &dst, mpz_t &a, mpz_t &b, mpz_t &mont_factor, mpz_t &modulus, Scratch &s) {
    mpz_mul(s.s, a, b); // Compute a * b
    // mpz_tdiv_q_2exp(s.u, s.s, mont_bits); // hi bits
    mpz_tdiv_r_2exp(s.l, s.s, mont_bits); // low bits
    mpz_mul(s.w, s.l, mont_factor); // product with mont factor
    mpz_tdiv_r_2exp(s.q, s.w, mont_bits); // idk how to take low product only, maybe the compiler will optimize it away
    mpz_addmul(s.s, s.q, modulus); // s + q*p
    mpz_tdiv_q_2exp(dst, s.s, mont_bits); // output / R
    // conditional correct technically required but ommittable if redundancy does not require another limb.
    // if (mont_bits % 64 == 0) {
    //     if (mpz_cmp(dst, modulus) >= 0) {
    //         mpz_sub(dst, dst, modulus);
    //     }
    // }
}

// It seems difficult to extract the mod mul from the mod exp
// so let's take the difference between x^3 (square + multiply) and x^2 (square)?
// template<int power>
// void diff_measurment(mpz_t dst, mpz_t base, mpz_t modulus) {
//     mpz_powm_ui(dst, base, power, modulus);
// }

template <int modbits>
int test(int argc, char**argv) {
    mpz_t a, b, modulus, mont;
    mpz_t c;
    assert(argc > 6);
    int num_multiplies = atoi(argv[1]);
    // argument 2 is unused.
    mpz_init_set_str(a, argv[3], 10);
    mpz_init_set_str(b, argv[4], 10);
    mpz_init_set_str(modulus, argv[5], 10);
    mpz_init_set_str(mont, argv[6], 10);
    mpz_init2(c, 2*modbits);
    Scratch s;
    init_scratch<modbits>(s);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_multiplies; i++) {
        karatsuba_mont<modbits>(c, a, b, mont, modulus, s);
        mpz_swap(a, c);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Karatsuba mont " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    gmp_printf("Mont result %Zd\n", a);

    mpz_t power;
    mpz_init_set_ui(power, 1);
    mpz_mul_2exp(power, power, num_multiplies);

    start = std::chrono::high_resolution_clock::now();
    // Assuming will not reduce exponent via order of unknown group
    mpz_powm(c, b, power, modulus); // only squares
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "Squaring only " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

    gmp_printf("Pow square result %Zd\n", c);

    mpz_sub_ui(power, power, 1);

    start = std::chrono::high_resolution_clock::now();
    // all of the above squares, now with multiplies
    mpz_powm(c, b, power, modulus);

    finish = std::chrono::high_resolution_clock::now();
    std::cout << "Squaring and multiplying (should be at least twice as much work) " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    gmp_printf("Pow square multiply result %Zd\n", c);
    return 0;
}

int main(int argc, char**argv) {
    // hacky just use limbs as modbits so I don't have to change python code
    return test<PYREPLACE_LIMBS1>(argc, argv);
}