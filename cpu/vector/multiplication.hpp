#pragma once
#include "changebase.hpp"
#include "avx512ifma.hpp"
#include "../precompute_io.hpp"

template <template <int> class Reduction, int limbs>
inline AVXVector<limbs> ele_mult(const AVXVector<limbs> &a, const AVXVector<limbs> & b, const Reduction<limbs> &r) {
    auto ab_lo = AVXVector<limbs>(0).mullo(a, b);
    auto ab_hi = AVXVector<limbs>(0).mulhi(a, b);
    return r.reduce_small(ab_hi, ab_lo);
}

template<template <int> class Reduction, int limbs1, int limbs2>
class IntRNS2 {

    const RNSMatrix<limbs1, limbs2, true> r1;
    const RNSMatrix<limbs2, limbs1, false> r2;
    
    public:

    const Reduction<limbs1> m1;
    const Reduction<limbs2> m2;

    inline IntRNS2(PrecomputeIO &io) : r1(io), r2(io), m1(io), m2(io) {
    }

    inline void mul_reduce(const AVXVector<limbs1> &a1, const AVXVector<limbs2> &a2, const AVXVector<limbs1> &b1, const AVXVector<limbs2> &b2, AVXVector<limbs1> &out1, AVXVector<limbs2> &out2) {
        auto ab_m1 = ele_mult(a1, b1, m1);
        AVXVector<limbs1> unused1;
        out2 = r1.rns_reduce(ab_m1, a2, b2, m2);
        out1 = r2.rns_reduce(out2, unused1, unused1, m1);
    }
};