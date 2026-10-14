---
status: accepted
---

# A sorting network is the exact OWA formulation for a long sample

## Context

`OrderedWeightsArray` and `OrderedWeightsArrayRange` had two JuMP formulations (#1677):

- `ExactOrderedWeightsArray`: the dual of the assignment problem, `T × T` rows. One million rows
  at `T = 1000`.
- `ApproxOrderedWeightsArray`, the default: one power-cone block for each entry of `p`. It is an
  upper bound, 0.06 % high on the Gini mean difference and 4.5 % high on a fourth-order L-moment
  vector at `T = 100`.

A third formulation is exact at `O(T log² T)` linear rows. It runs the net portfolio returns
through a sorting network and relaxes each comparator `(u, v) -> (lo, hi)` to
`lo + hi = u + v`, `hi >= u`, `hi >= v`. Goemans (2015) proves that this relaxation of **any**
sorting network is an extended formulation of the permutahedron. For a nondecreasing weight
vector, the minimum of the weighted network outputs is the OWA risk. A vector that decreases makes
the model unbounded.

Measured on 2026-10-14, Clarabel at `1e-9`, eight assets, the Gini weights, minimum risk:

| `T` | Network | Comparators | Solve (s) | Iterations |
| --- | --- | --- | --- | --- |
| 250 | odd-even merge | 3756 | 1.95 | 20 |
| 250 | bitonic merge | 4405 | 1.63 | 21 |
| 500 | odd-even merge | 9511 | 42.6 | 23 |
| 500 | bitonic merge | 11026 | 21.6 | 23 |

At `T = 500` the `T × T` form solved in 40.6 s and `ApproxOrderedWeightsArray` in 0.61 s
(#1677). The order of the network variables had no effect at `T = 500` under Clarabel: 45.8 s in
the order of the comparators, 44.1 s in the order of the positions, 44.4 s in a random order, all
23 iterations. Clarabel permutes the KKT system before it factorises it.

Research #1708 found the cost of one iteration. One interior-point iteration is one LDL
factorisation of the KKT matrix, and that factorisation is 83 to 95 % of the solve. Clarabel's
default QDLDL is a simplicial factorisation: it works on one column at a time, at about 1.5 s per
10⁹ flops. The fill of the factor sets the flops, and the AMD order of QDLDL sets the fill. Bitonic
merge was faster than odd-even merge above because AMD orders its KKT matrix to 59 % of the flops.
Under a METIS order the three networks (odd-even, bitonic, and the pairwise network) have almost
the same flops.

A supernodal factorisation of the same matrix works on dense blocks. Bitonic merge, the same data,
one thread:

| Setup | `T = 500` (s) | `T = 1000` (s) |
| --- | --- | --- |
| Clarabel, QDLDL (the default) | 26.5 to 30.5 | 298.5 |
| Clarabel, `direct_solve_method = :mkl` after `using Pardiso` | 4.50 to 4.72 | 28.4 (odd-even merge: 22.2) |
| Tulip, `KKT_System = Tulip.KKT.K1()`, `KKT_Backend = Tulip.KKT.TlpCholmod.Backend()`, tolerances `1e-9` | 6.28 | 39.9 |
| `ExactOrderedWeightsArray`, Clarabel with `:mkl` | 30.4 | |

The iterations and the risk do not change. MKL Pardiso needs `using Pardiso`, runs on x86-64
only, and orders by METIS. Tulip with `K1` factorises the normal equations with the supernodal
Cholesky of CHOLMOD and needs no other binary. Under MKL at `T = 500` the three networks are
within the noise of the machine, about ±30 %. At `T = 1000` odd-even merge was the fastest, near
the edge of that noise.

## Decision

The maintainer decided on 2026-10-14.

1. **The formulation is `SortingNetworkOrderedWeightsArray`**, the third subtype of
   `OrderedWeightsArrayFormulation`. It serves `OrderedWeightsArray`, and it fuses the two tails
   of `OrderedWeightsArrayRange` into one network over `w1 - w2`, as the exact form does. The
   name states the structure that the formulation builds. It replaced the first name,
   `IterativeOrderedWeightsArray`, before the formulation was released. The keys of the Model
   State take the prefix `sowa`, as the keys of `ApproxOrderedWeightsArray` take `aowa`.
2. **`ApproxOrderedWeightsArray` stays the default** of both measures and of every alias. It is
   the fastest of the three by a wide margin.
3. **`SortingNetworkOrderedWeightsArray` replaces `ExactOrderedWeightsArray` as the recommended
   exact formulation for a long sample.** `ExactOrderedWeightsArray` stays, for a short sample.
4. **The type has two axes, one field each.** `merge::AbstractSortingNetworkMerge` picks the
   network: `OddEvenMerge` or `BitonicMerge`. `ordering::AbstractSortingNetworkOrdering` picks
   the order of the network variables: `ComparatorOrdering` or `WireOrdering`. Both abstract types
   are extension points, with the verbs `sorting_network` and `sorting_network_columns`.
5. **The defaults are `BitonicMerge()` and `ComparatorOrdering()`.** Bitonic merge is two times
   faster at `T = 500` under QDLDL, the default solver, although it has more comparators. The
   cause is the AMD order of QDLDL, not the network. Under a supernodal factorisation no network
   wins past the noise. The maintainer kept bitonic merge on 2026-10-14 after #1708, because the
   default serves a user who sets up no solver. The order had no measured effect, under any solver,
   so the default is the plain one.
6. **A weight vector that is not nondecreasing is refused**, with a `DomainError`: in the
   constructor when the weights are a vector, and when the model is built when they are a callable.
   `ExactOrderedWeightsArray` still takes such a vector.
7. **The network reads the `owa` variable of `set_owa_constraints!`**, so the fees and the costs
   enter as they do for the exact form. The inputs are divided by the largest absolute asset
   return, and the risk is multiplied by it. The returns are not centred, because a shift changes
   the risk of a weight vector whose sum is not zero.
8. **The documentation names a supernodal factorisation for a long sample.** The fill of the
   LDL factorisation of the KKT matrix sets the cost of one iteration, not the row count, and
   QDLDL factorises one column at a time. The docstring of `SortingNetworkOrderedWeightsArray` and
   the OWA example name Clarabel with `direct_solve_method = :mkl` and Tulip with `K1`, with their
   measured times (#1711). The library and its tests take no new dependency for it. A QDLDL with a
   METIS order, registered as a new Clarabel `direct_solve_method`, cut the flops by 45 to 54 %
   for odd-even and pairwise merge, but MKL Pardiso already orders by METIS and is four to six
   times faster again, so it is not built.

## Consequences

- A long sample gets the exact risk at `O(T log² T)` rows. The time of the solve stays well above
  the time of `ApproxOrderedWeightsArray`. A supernodal factorisation cuts it by 83 to 90 %.
- Under a supernodal factorisation the network beats `ExactOrderedWeightsArray` by 6.4 to 6.8
  times at `T = 500`, so decision 3 holds more strongly than under QDLDL.
- A new network or a new order is one subtype and one method.
