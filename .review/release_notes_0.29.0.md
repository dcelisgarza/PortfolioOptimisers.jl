# Release notes seed — v0.29.0, the prior carriers

**Status: a seed, not the finished notes.** It holds the one migration section that the review's
`[B2]` box asks for. The full notes are still to be written from `.review/api_ledger.txt`
(6 exports removed, 2 types gone, 69 exports added, 44 exported types change fields).

## The factor block of a prior carrier is nested

`LowOrderPrior` and `HighOrderPrior` no longer carry the factor moments as flat fields. One field,
`fpr`, carries a whole nested prior for the factor block, or `nothing` when there is none.

| Carrier | Fields removed | Fields added |
| --- | --- | --- |
| `LowOrderPrior` | `f_mu`, `f_sigma`, `f_w` | `o_X`, `fpr`, `Z` |
| `HighOrderPrior` | `f_kt`, `f_sk`, `f_V` | `fpr` |

### What breaks: construction, not reading

**The break bites a caller who constructs a carrier with a flat factor keyword. It does not bite a
caller who reads one.** The removed names survive as *virtual reads* through `@forward_properties`
(`src/13_Prior/01_Base_Prior.jl:1089` and `:1378`), so `pr.f_mu` still answers, and answers
`nothing` where there is no factor block — exactly what the flat field did.

```julia
# Still works. No change needed.
pr.f_mu, pr.f_sigma, pr.f_w      # LowOrderPrior
hop.f_kt, hop.f_sk, hop.f_V      # HighOrderPrior

# Breaks: MethodError — got unsupported keyword argument "f_mu".
LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr, f_mu = f_mu, f_sigma = f_sigma)

# The replacement: build the factor block as its own prior and pass it as `fpr`.
LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr,
              fpr = LowOrderPrior(; X = F, mu = f_mu, sigma = f_sigma))
```

The togetherness invariant is unchanged, only respelled. The old constructor needed `rr`, `f_mu`
and `f_sigma` together. The new one needs `rr` and `fpr` together, and says so:
`ArgumentError: rr and fpr are the factor block and must be provided together or not at all`.

Anything that destructures or splats the field set is also affected, because the field set itself
changed: `fieldnames`, `getfield`, a positional constructor call, and `prior_field_values`.

### Reads that come free with the nesting

The nesting exposes seven reads that the flat set never had. None of the seven exists at base.
They need no migration, but they are worth naming, because they answer on any carrier that has a
factor block:

- `LowOrderPrior`: `f_ens`, `f_kld`, `f_ow`.
- `HighOrderPrior`: `f_D2`, `f_L2`, `f_S2`, `f_skmp`.

### Which read is idiomatic

`pr.fpr.mu` is the public read. ADR 0046's 2026-08-03 amendment freezes the flat `f_`-prefixed set:
it stays readable and stays supported, and no name is added to it. The two reads differ only where
the block is absent — `pr.f_mu` answers `nothing`, and `pr.fpr.mu` raises. Guard on `fpr`, then
read through it.

Read ADR 0046 (`docs/adr/0046-wrapping-priors-forward-by-default-and-document-every-drop.md`) for
the reason behind every dropped field.

### Not a migration item

`LowOrderPrior.z_sq` is not readable at `HEAD` — it is neither a field nor a forwarded property.
It needs no note, because it was added and removed inside this PR. No release ever carried it.

## A PMFG weight must be strictly positive

`PMFG_T2s` carries the graph's structure and its weights in one matrix, so an exactly zero weight is
an **absent edge** rather than a weak one. `assert_pmfg_weights` now counts the stored edges against
the `3N - 6` a maximal planar graph has, and raises a `DomainError` naming the shortfall. It runs at
`DBHTs`, `calc_weighted_adjacency_graph` and `calc_distance_weighted_graph`. `logo!` reads only the
cliques and is deliberately unguarded.

### What breaks

A similarity that returns an exact zero on the PMFG path. Two shipped routes reach one:

- `ExponentialSimilarity` or `GeneralExponentialSimilarity` under `LogDistance`, which maps an
  exactly zero correlation to `Inf`, and `exp(-Inf)` is `0` exactly.
- `ComplementSimilarity` or `MaximumDistanceSimilarity` at `D = 1`, which is a correlation of `-1`.

An exactly zero sample correlation is not an everyday event, but `Denoise()` makes one. The measured
case is `NetworkEstimator(; ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn =
Denoise())), de = Distance(; alg = LogDistance()), alg = ExponentialSimilarity())` over noise, which
denoises to the identity correlation. It used to return a network of `0` of its `54` edges. It now
raises.

```text
DomainError with count(!iszero, A) / 2 == 3 * size(A, 1) - 6 must hold. Got
edges => 0
3 * N - 6 => 54
```

The replacement is a similarity that stays strictly positive on the data at hand. Nothing that
worked and returned a full graph changes: the refusal is exactly the set of inputs whose graph was
already missing edges. Before this change the same inputs died later, in `turn_into_Hclust_merges`,
with a `BoundsError` about a matrix index. See ADR 0049, *Non-negative reaches the check, positive
reaches the graph*.
