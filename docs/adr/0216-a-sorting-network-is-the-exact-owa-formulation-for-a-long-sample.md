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
5. **The defaults are `BitonicMerge()` and `ComparatorOrdering()`.** Bitonic merge was two times
   faster at `T = 500` although it has more comparators. The order had no measured effect, so the
   default is the plain one.
6. **A weight vector that is not nondecreasing is refused**, with a `DomainError`: in the
   constructor when the weights are a vector, and when the model is built when they are a callable.
   `ExactOrderedWeightsArray` still takes such a vector.
7. **The network reads the `owa` variable of `set_owa_constraints!`**, so the fees and the costs
   enter as they do for the exact form. The inputs are divided by the largest absolute asset
   return, and the risk is multiplied by it. The returns are not centred, because a shift changes
   the risk of a weight vector whose sum is not zero.
8. **Why the cost of one iteration is high is a research question**, filed apart from this
   decision as #1708. The fill of the factorisation of the KKT system sets it, not the row count.

## Consequences

- A long sample gets the exact risk at `O(T log² T)` rows. The time of the solve stays well above
  the time of `ApproxOrderedWeightsArray`.
- A new network or a new order is one subtype and one method.
