// https://flintlib.org/doc/mpn_mod.html 
//up to 1024 bits on 64 bit machines
#include <flint.h>
#include <gr.h>
#include <gr_types.h>
#include <mpn_mod.h>
#include <fmpz_mod.h>
#include <cassert>
#include <fmpz.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

class MPN {

    public:
    gr_ctx_t ctx;
    MPN(const fmpz_t n, int limbs) {
        assert(gr_ctx_init_mpn_mod(ctx, n) == GR_SUCCESS);
        assert(MPN_MOD_CTX_NLIMBS(ctx) == limbs);
    }

    void modred(nn_srcptr a, nn_srcptr b, nn_ptr result) {
        mpn_mod_mul(result, a, b, ctx);
    }
};

// Larger than 1024 bits
class FMPZ {

    public:
    fmpz_mod_ctx_t ctx;
    FMPZ(const fmpz_t n) {
        fmpz_mod_ctx_init(ctx, n);
    }

    void modred(const fmpz_t a, const fmpz_t b, fmpz_t result) {
        fmpz_mod_mul(result, a, b, ctx);
    }
};

template<int limbs>
int read_arguments(int argc, char** argv, fmpz_t (&values)[3]) {
    assert(argc > 4);
    int num_multiplies = atoi(argv[1]);
    for (int i = 0; i < 3; i++) {
        fmpz_init2(values[i], limbs);
        fmpz_set_str(values[i], argv[i+2], 10);
    }
    return num_multiplies;
}

template <int limbs>
void test_mpn(fmpz_t a, fmpz_t b, fmpz_t mod_fmpz, int num_multiplies) {
    MPN m = MPN(mod_fmpz, limbs);
    ulong na[limbs];
    ulong nb[limbs];
    mpn_mod_set_fmpz(na, a, m.ctx);
    mpn_mod_set_fmpz(nb, b, m.ctx);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_multiplies; i++) {
        ulong tmp[limbs];
        m.modred(na, nb, tmp);
        for (int j = 0; j < limbs; j++) {
            na[j] = tmp[j];
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    mpn_mod_get_fmpz(a, na, m.ctx);
}

void test_fmpz(fmpz_t a, fmpz_t b, fmpz_t mod_fmpz, int num_multiplies) {
    FMPZ m = FMPZ(mod_fmpz);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_multiplies; i++) {
        m.modred(a, b, a);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
}

template <int limbs>
void test(int argc, char** argv) {
    fmpz_t arguments[3];
    int num_multiplies = read_arguments<limbs>(argc, argv, arguments);
    fmpz_t output;
    if (limbs < MPN_MOD_MAX_LIMBS) {
        test_mpn<limbs>(arguments[1], arguments[2], arguments[0], num_multiplies);
    } else {
        test_fmpz(arguments[1], arguments[2], arguments[0], num_multiplies);
    }
    flint_printf("Result: %{fmpz}\n", arguments[1]);
}

int main(int argc, char** argv) {
    test<PYREPLACE_LIMBS1>(argc, argv);
}