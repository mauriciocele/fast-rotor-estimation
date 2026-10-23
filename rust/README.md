# rotor-estimation (Rust port)

Rust port of the C++ benchmark in `../src`: the same estimators for Wahba's
problem (find the rotation R minimizing `sum_i w_i |R p_i - q_i|^2`), the same
data generation and the same report.

| C++ | Rust |
|---|---|
| `Flae`, `FlaeSymbolic`, `FlaeNewton` | `flae`, `flae_symbolic`, `flae_newton` |
| `FA3R_double`, `FA3R_int` | `fa3r_double`, `fa3r_int` |
| `GARotorEstimator`, `LARotorEstimator` | `ga_rotor_estimator`, `la_rotor_estimator` |
| `SVDEigen`, `SVDMcAdams` (+ `svd3.h`) | `svd_eigen`, `svd_mcadams` (+ `svd3`) |
| `Horn`, `Davenport`, `Quest`, `Foam` | `horn`, `davenport`, `quest`, `foam` |
| `GAValkenburg`, `QuaternionDirect` | `ga_valkenburg`, `quaternion_direct` |
| `Multivector`, `BasisBlade` | `multivector` (only what the GA estimator uses) |

Eigen is replaced by [nalgebra](https://nalgebra.org) (matrices, symmetric
eigen-solver, SVD).

## Build and run

```sh
cargo build --release
./target/release/estimator --n=1000 --noise=true
cargo test --release
```

Options, all optional: `--name=out.txt --plane=[x|y|z] --n=1000 --noise=[true|false]
--norm=<double> --axis=<x,y,z> --angle=<double> --seed=<u64>`. As in the C++
version, the report goes both to stdout and to the `--name` file, and each
estimator is timed over 1,000,000 calls.

## Fixes (applied to both the C++ in `../src` and this port)

| Estimator | Defect | Fix |
|---|---|---|
| `FlaeNewton`, `QUEST`, `FOAM` | Newton started at lambda = 1 and could stop at a smaller root: ~30% of noisy runs with random norms returned ~5x the optimal error | start from `sum w \|p\| \|q\|`, an upper bound of the largest eigenvalue |
| `FLAE` (all three) | fixing q_z = -1 fails for any rotation whose axis lies in the xy-plane (q_z = 0) | detect it from the leading minor and p'(lambda); read the eigenvector from the best column of the adjugate |
| `LARotorEstimator`, `GARotorEstimator` | the sum of the four triple wedges is `q (q_w - q_x + q_y - q_z)`, which vanishes for a family of rotations (e.g. 90 degrees about z) | keep the single wedge (adjugate column) with the largest diagonal entry, `AdjugateNullVector.h` |
| `QUEST` | singular at 180 degrees | method of sequential rotations (Shuster & Oh) |
| `SVD` (Eigen/nalgebra) | could return a reflection (planar or noisy data) | flip the smallest singular vector when det < 0 |
| `SVDMcAdams` | singular-value sort read `b23` instead of `b32`; 4 Jacobi sweeps (tuned for float) were not enough in double | fixed the index; 6 sweeps |
| `SVDMcAdams` | returned `U V^T` only through Eigen's column-major `Matrix3d(const double*)` | builds `U V^T` explicitly |
| `FA3R_double`, `FA3R_int` | iteration tuned for \|H\| ~ 1; with large norms it did not converge, and `FA3R_int` overflowed int64 | normalize H first (the rotation does not depend on its scale) |
| `Multivector` | `std::sort` comparator returned `a - b` (not a strict weak ordering: undefined behaviour) | `a < b` |
| `GARotorEstimator` | printed a debug line on every call | removed |
| `main` | lowercased the `--name` file name; timed each call with `clock()` | keeps the case; one wall-clock measurement around the 1,000,000 calls |

After these fixes every estimator reaches the optimum (Horn's exact eigen-solution) on
all test datasets. The exceptions are `FA3R_int`, limited by its 20-bit fixed point (about 1e-8
relative), and `FOAM`, whose matrix is orthogonal only to the precision of its Newton step (about 1e-9).
`cargo test` includes a regression test for each fix.

## Differences from the C++ version

- `--seed` is new: it makes a run reproducible. Without it the seed is the
  current time, as before. The random numbers are not the ones C's `rand()`
  gives, so a given run cannot be reproduced bit for bit across the two versions.

On identical inputs the two versions agree to rounding error: on 21 datasets
(15 estimators each) the rotation matrices differ by at most 6e-13.
