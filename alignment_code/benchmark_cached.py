"""Benchmark functionsgpu_old vs functionsgpu_cached on one frechet iteration."""
import os
import pickle
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "_bench_run_one.py")


def run(mod_name, out_path):
    print(f"\n[run] {mod_name} -> {out_path}", flush=True)
    res = subprocess.run([sys.executable, "-u", RUNNER, mod_name, out_path], cwd=HERE, check=False)
    if res.returncode != 0:
        raise SystemExit(f"{mod_name} failed (exit {res.returncode})")
    with open(out_path, "rb") as f:
        return pickle.load(f)


def main():
    print("=" * 60)
    print("Benchmark: functionsgpu_old vs functionsgpu_cached (1 iter)")
    print("=" * 60)

    old = run("functionsgpu_old", os.path.join(HERE, "_bench_old.pkl"))
    new = run("functionsgpu_cached", os.path.join(HERE, "_bench_cached.pkl"))

    mu_diff = float(np.max(np.abs(old["mu"] - new["mu"])))
    betas_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(old["betas_aligned"], new["betas_aligned"]))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  old    time : {old['time']:.2f} s")
    print(f"  cached time : {new['time']:.2f} s")
    print(f"  speedup     : {old['time'] / new['time']:.2f}x")
    print(f"  max |mu diff|    : {mu_diff:.3e}")
    print(f"  max |betas diff| : {betas_diff:.3e}")
    print(f"  numerically close: {mu_diff < 1e-4 and betas_diff < 1e-4}")


if __name__ == "__main__":
    main()
