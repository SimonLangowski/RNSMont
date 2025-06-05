from rns_helpers import ceil_div
from modcache import modcache
from random import randint

def format_modcache(modulus):
    modbits = modulus.bit_length()
    mod_limbs = ceil_div(modbits, 64)
    a = randint(0, modulus)
    b = randint(0, modulus)
    return f'''#[derive(MontConfig)]
    #[modulus = "{modulus}"]
    #[generator = "2"] // unused
    #[small_subgroup_base = "3"] // unused
    #[small_subgroup_power = "2"] // unused
    pub struct Config{modbits};
    pub type Field{modbits} = Fp<MontBackend<Config{modbits}, {mod_limbs}>, {mod_limbs}>;
    pub const A{modbits} : Field{modbits} = MontFp!("{a}");
    pub const B{modbits} : Field{modbits} = MontFp!("{b}");
    '''

def create_modcache(fn):
    moduli = modcache.values()
    with open(fn, "w") as f:
        header = "use ark_ff::fields::{Fp, MontBackend, MontConfig, MontFp};\n"
        print(header + "\n".join([format_modcache(m) for m in moduli]), file=f)

def format_bench():
    moduli = modcache.values()
    moduli_bits = [m.bit_length() for m in moduli]
    header = '''use criterion::{black_box, criterion_group, criterion_main, Criterion};
    use modmul::modcache::*;'''
    body_start = "pub fn criterion_benchmark(c : &mut Criterion) {"
    bodies = [f'''c.bench_function("mul {modbits}", |b| b.iter(|| black_box(A{modbits}) * black_box(B{modbits})));''' for modbits in moduli_bits]
    body_end = "}"
    body = "\n".join(bodies)
    trailer = '''criterion_group!(benches, criterion_benchmark);
    criterion_main!(benches);
    '''
    return "\n".join([header, body_start, body, body_end, trailer])

def create_bench(fn):
    with open(fn, "w") as f:
        print(format_bench(), file=f)

if __name__ == "__main__":
    create_modcache("../cpu/baselines/modmul/src/modcache.rs")
    create_bench("../cpu/baselines/modmul/benches/bench_arkworks.rs")