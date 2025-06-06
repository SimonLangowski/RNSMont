import sys
import subprocess

from precompute_io import PrecomputeIO
from python_reduction import *
from modcache import modcache
import re

def compile(method_name, reduction_name, limbs1, limbs2, q_correct, out_name="a.out", file="../cpu/loading.cpp", extra_flags=[], compiler="g++", arch="native"):
    compiler_flags = [f"-march={arch}",
                    "-O3",
                    f"-DPYREPLACE_LIMBS1={limbs1}",
                    f"-DPYREPLACE_LIMBS2={limbs2}",
                    f"-DPYREPLACE_Q_CORRECT={str(q_correct).lower()}",
                    f"-DPYREPLACE_METHOD_NAME={method_name}",
                    f"-DPYREPLACE_REDUCTION_NAME={reduction_name}",
                    "-o", out_name]

    command = [compiler, file]
    command.extend(compiler_flags)
    command.extend(extra_flags)
    subprocess.run(command, check=True)

def run(num_mults, io_file, out_name, other_args=[]):
    command = [f"./{out_name}", f"{num_mults}", f"{io_file}"]
    command.extend(other_args)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    out_str = stdout.decode()
    print(out_str)
    print(stderr.decode())
    lines = out_str.splitlines()
    time = 0
    out1 = []
    out2 = []
    for line in lines:
        l = line.strip()
        if l.endswith("ns"):
            time = int(re.search(r'\d+$', l[:l.find("ns")]).group())
            break
    return time

def create_io(method, target, io_name):
    io = PrecomputeIO(io_name)
    params = Params.avx(MontgomeryReduce)
    r = method(target, params, io)
    a = randint(0, target)
    b = randint(0, target)
    a_m1, a_m2 = r.to_mont_avx(a)
    b_m1, b_m2 = r.to_mont_avx(b)
    io.vector(a_m1.data)
    io.vector(a_m2.data)
    io.vector(b_m1.data)
    io.vector(b_m2.data)
    return r, (a, b), (a_m1, a_m2, b_m1, b_m2)

def run_and_test(method_name, target, num_mults=100, test_correct=True):
    assert(target > 0)
    modbits = target.bit_length()
    io_name = f"input{modbits}.txt"
    exe_name = f"avx{method_name.__name__}{modbits}.exe"
    r, py_input, rns_input = create_io(method_name, target, io_name)
    limbs1 = len(r.m1)
    limbs2 = len(r.m2)
    q_correct = False
    reduction_method = "MontgomeryReduce"
    compile(method_name.__name__, reduction_method, limbs1, limbs2, q_correct, exe_name)
    print(f"Running {exe_name} {limbs1} {limbs2} {q_correct} x {num_mults}")
    time = run(num_mults, io_name, exe_name)
    if test_correct:
        print("Checking")
        a, b = py_input
        a_m1, a_m2, b_m1, b_m2 = rns_input
        for _ in range(num_mults):
            a_m1, a_m2 = r.mulreduce(a_m1, a_m2, b_m1, b_m2)
        c = r.from_mont_avx(a_m2.store())
        correct = ((a * pow(b, num_mults, target)) % target)
        assert(c % target == correct)
        # For comparison with c program output (could extract from pipe for full flexibility)
        print("out1", a_m1)
        print("out2", a_m2)
    return time

def run_baseline(target, num_mults=100):
    modbits = target.bit_length()
    limb_size = 64
    limbs = ceil_div(modbits, limb_size)
    exe_name = f"flint{modbits}.exe"
    extra_flags = [
        "-I/usr/local/include/flint",
        "-L/usr/local/lib",
        "-lflint",
        "-lgmp",
    ]
    compile("unused", "unused", limbs, "unused", "unused", f"flint{modbits}.exe", "../cpu/baselines/flint.cpp", extra_flags)
    a = randint(0, target)
    b = randint(0, target)
    io_name = f"{target}"
    other_args = [f"{a}", f"{b}"]
    time = run(num_mults, io_name, exe_name, other_args)
    correct = ((a * pow(b, num_mults, target)) % target)
    print(f"correct {correct}")
    return time

def run_baseline2(target, num_mults = 100):
    modbits = target.bit_length()
    exe_name = f"mpz{modbits}.exe"
    extra_flags = [
        # "-I/usr/local/include/flint",
        # "-L/usr/local/lib",
        # "-lflint",
        "-lgmp",
    ]
    compile("unused", "unused", modbits, "unused", "unused", exe_name, "../cpu/baselines/mpz.cpp", extra_flags)
    a = randint(0, target)
    b = randint(0, target)
    mont_r = 2**modbits
    mont_factor = pow(-mont_r, -1, target)
    b2 = (b * mont_r) % target
    io_name = f"{target}"
    other_args = [f"{a}", f"{b2}", f"{target}", f"{mont_factor}"]
    time = run(num_mults, io_name, exe_name, other_args)
    print(time / num_mults)
    correct = ((a * pow(b, num_mults, target)) % target)
    print(f"correct {correct}")
    return time

class Y:
    def __init__(self, modbits):
        self.modulus = 0
        self.modbits = modbits

def run_all(target, num_mults, results):
    t = run_and_test(IntRNS2, target, num_mults)
    results[(target.bit_length(), "int")] = t
    t = run_baseline(target, num_mults)
    results[(target.bit_length(), "flint")] = t

def run_on_params(num_mults):
    step = 64
    std = list(range(step, 4096+step, step))
    # std2 = [s + step - 1 for s in std]
    # step = 52
    # opt = list(range(step-4, 4096+step, step))
    # opt2 = [s + step - 1 for s in std]
    # std.extend(opt)
    # std.extend(opt2)
    # std.extend(std2)
    # std.sort()
    results = {}
    for modbits in std:
        print(modbits)
        if modbits in modcache:
            target = modcache[modbits]
        else:
            args = Y(modbits)
            target = set_get_modulus(args)
            modcache[modbits] = target
        assert(target.bit_length() == modbits)
        run_all(target, num_mults, results)
        print(results)

if __name__ == "__main__":
    choice = -1
    num_mults = 10000
    modbits = 256
    target = 0
    if len(sys.argv) > 1:
        choice = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_mults = int(sys.argv[2])
    if len(sys.argv) > 3:
        v = int(sys.argv[3])
        if v < 2**32:
            modbits = v
        else:
            target = v
            modbits = target.bit_length()
    if target == 0 and modbits > 0:
        args = Y(modbits)
        if modbits in modcache:
            # Basically takes a while to find
            target = modcache[modbits]
        else:
            target = set_get_modulus(args)
        assert(target.bit_length() == modbits)
    if choice == 1:
        run_on_params(num_mults)
    elif choice == 2:
        run_and_test(IntRNS2, target, num_mults)
    elif choice == 3:
        run_baseline(target, num_mults)
    elif choice == 4:
        run_baseline2(target, num_mults)
    else:
        print("Unknown choice")
