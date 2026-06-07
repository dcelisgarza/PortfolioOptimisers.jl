# Handoff: Factory Method Generalisation — Ready for Integration

**Date:** 2026-06-05
**Branch:** `factory-spike-a` (worktree at `/mnt/storage/dev/PortfolioOptimisers.jl/factory-spike-a/`)
**ADR (source of truth):** `dev-plots/docs/adr/0001-generalise-factory-methods.md` — status **`accepted`**

---

## What was done this session

All three spikes from the ADR were run and measured:

| Spike | File | Verdict |
|---|---|---|
| A — runtime reflection | `spike/spike_a_reflection.jl` | **Rejected** — type-unstable on deep composites (`FactorPrior`, `HighOrderPriorEstimator` infer `Any`) |
| B — `@generated` unroll | `spike/spike_b_generated.jl` | **Passed** — 15/15 type-stable, 11/15 propagators match |
| C — `@curryable` macro | `spike/spike_c_macro.jl` | **Adopted** — 15/15 type-stable, 3.2× better cold-start TTFX than B |

Cold-start first-call latency (fresh REPL, 15-type corpus): B = 536 ms total, C = 170 ms total.
Full numbers in the ADR Consequences section.

The ADR is fully updated with all findings. Do not re-run the spikes — the decision is final.

---

## What to do next: integrate `@curryable` into the package

The integration follows the **family-by-family migration** plan in the ADR (Verification & migration section). Below is the concrete task list in order.

### Step 0 — Prerequisites (do once)

**a) Add `_factory_child` helper to `src/02_Tools.jl`** (after the existing `factory` definitions, around line 679):

```julia
# Per-field recursion helper for @curryable-generated factory methods.
# Eligible values recurse via factory; everything else passes through.
_factory_child(v, args...; kwargs...) = v
_factory_child(v::Union{<:AbstractEstimator, <:AbstractAlgorithm}, args...; kwargs...) =
    factory(v, args...; kwargs...)
_factory_child(v::AbstractArray{<:Union{<:AbstractEstimator, <:AbstractAlgorithm}}, args...; kwargs...) =
    [_factory_child(vi, args...; kwargs...) for vi in v]
```

**b) Add `AbstractCovarianceEstimator` extension in `src/08_Moments/01_Base_Moments.jl`** (after `AbstractCovarianceEstimator` is defined, around line 52):

```julia
_factory_child(v::AbstractCovarianceEstimator, args...; kwargs...) =
    factory(v, args...; kwargs...)
_factory_child(v::AbstractArray{<:AbstractCovarianceEstimator}, args...; kwargs...) =
    [_factory_child(vi, args...; kwargs...) for vi in v]
```

**c) Define the `@curryable` macro** — best placed in `src/02_Tools.jl` or a new `src/02b_Curryable.jl` included right after it. The production form (from the spike, with the unparameterised constructor fix):

```julia
macro curryable(struct_expr)
    @assert struct_expr isa Expr && struct_expr.head == :struct
    type_head = struct_expr.args[2]
    function bare_name(n)
        n isa Symbol && return n
        n isa Expr && n.head == :curly && return n.args[1]
        n isa Expr && n.head == :<:    && return bare_name(n.args[1])
        n
    end
    struct_name = bare_name(type_head)
    body = struct_expr.args[3]
    fnames = Symbol[]
    for item in body.args
        item isa LineNumberNode && continue
        if item isa Symbol; push!(fnames, item)
        elseif item isa Expr && item.head == :(::); push!(fnames, item.args[1])
        end
    end
    if isempty(fnames)
        factory_def = quote
            function PortfolioOptimisers.factory(x::$struct_name, args...; kwargs...)
                return x
            end
        end
    else
        kw_pairs = [Expr(:kw, f,
                         :(_factory_child($(Expr(:., :x, QuoteNode(f))),
                                          args...; kwargs...)))
                    for f in fnames]
        ctor_call = Expr(:call, struct_name, Expr(:parameters, kw_pairs...))
        factory_def = quote
            function PortfolioOptimisers.factory(x::$struct_name, args...; kwargs...)
                return $ctor_call
            end
        end
    end
    return esc(quote
        $struct_expr
        $factory_def
    end)
end
```

**CRITICAL:** The macro uses the bare struct name (`EquilibriumExpectedReturns(; ...)`) not `T(; ...)`. This is required because `factory` changes field type parameters (e.g. `Nothing` → `ProbabilityWeights{...}`); constructing with the original parameterised type fails.

**d) Split the blanket fallback in `src/02_Tools.jl`** (lines 672–679). Change:

```julia
# BEFORE
function factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                          <:AbstractResult}, args...; kwargs...)
    return a
end
```

to:

```julia
# AFTER — AbstractEstimator/Algorithm now handled by @curryable-generated methods
function factory(a::Union{Nothing, <:AbstractResult}, args...; kwargs...)
    return a
end
```

The vector fallback on line 676 stays (it can be broadened later to include `AbstractCovarianceEstimator`).

### Step 1 — Expected-returns family (first migration, lowest risk)

**Delete these abstract-family identity passthroughs** (they shadow any generic and must go before deleting concrete propagators):

| File | Line (approx) | Method to delete |
|---|---|---|
| `src/08_Moments/01_Base_Moments.jl` | 416 | `factory(me::AbstractExpectedReturnsEstimator, args...; kwargs...) = me` |
| `src/08_Moments/01_Base_Moments.jl` | 479 | `factory(alg::AbstractExpectedReturnsAlgorithm, args...; kwargs...) = alg` |

**Add `@curryable` to these structs** (propagators — their hand-written factory methods become deletable):

| Struct | File | Hand-written factory to DELETE |
|---|---|---|
| `EquilibriumExpectedReturns` | `08_Moments/17_EquilibriumExpectedReturns.jl` | `factory(me::EquilibriumExpectedReturns, w::ObsWeights)` |
| `ExcessExpectedReturns` | `08_Moments/18_ExcessExpectedReturns.jl` | `factory(me::ExcessExpectedReturns, w::ObsWeights)` |
| `ShrunkExpectedReturns` | `08_Moments/16_ShrunkExpectedReturns.jl` | `factory(me::ShrunkExpectedReturns, w::ObsWeights)` |
| `StandardDeviationExpectedReturns` | `08_Moments/27_StandardDeviationExpectedReturns.jl` | `factory(ce::StandardDeviationExpectedReturns, w::ObsWeights)` |
| `VarianceExpectedReturns` | `08_Moments/27_StandardDeviationExpectedReturns.jl` | `factory(ce::VarianceExpectedReturns, w::ObsWeights)` |
| `CustomValueExpectedReturns` | `08_Moments/34_CustomValueExpectedReturns.jl` | *(no explicit factory — relied on abstract passthrough)* |

**Leave untouched** (leaf injectors / logic — harness classified them DIFFER):

- `SimpleExpectedReturns` — stores `w = w` directly
- `MedianExpectedReturns` — stores `w = w` directly
- `WindowedExpectedReturns` — has windowing logic

After adding `@curryable` and deleting the propagators, run the full test suite:

```
julia --project=. -e 'include("test/runtests.jl")'
```

Commit as: `refactor: replace pure-propagation factory methods with @curryable (expected-returns family)`

### Step 2 — Covariance family

Propagators to `@curryable` + delete: `Covariance`, `PortfolioOptimisersCovariance`.
Leaf injectors to keep: `GeneralCovariance` (stores `w = w`), `SimpleVariance` (stores `w = w`).
Abstract passthrough to delete: none (covariance family uses the StatsBase external-boundary passthrough which must stay).

Other covariance propagators to discover: run the differential harness from `spike/spike_b_generated.jl` extended to the full covariance family.

### Step 3 — Remaining families (in order)

Delete these abstract-family identity passthroughs as each family is migrated:

| File | Method |
|---|---|
| `src/11_Phylogeny/01_Base_Phylogeny.jl:81` | `factory(alg::AbstractPhylogenyAlgorithm, ...)` |
| `src/11_Phylogeny/02_Clusters.jl:39` | `factory(alg::AbstractClustersAlgorithm, ...)` |
| `src/19_RiskMeasures/01_Base_RiskMeasures.jl:515` | `factory(rs::AbstractBaseRiskMeasure, ...)` |
| `src/19_RiskMeasures/03_MomentRiskMeasures.jl:52` | `factory(alg::MomentMeasureAlgorithm, ...)` |
| `src/20_Optimisation/08_Base_JuMPOptimisation.jl:75` | `factory(r::JuMPReturnsEstimator, ...)` |

Families: prior, regression, phylogeny, risk measures, optimisers — use the differential harness to classify each concrete type before deleting.

---

## Key invariants (do not break)

- **Keep** `factory(ce::StatsBase.CovarianceEstimator, args...) = ce` — external boundary, not a PO type.
- **Keep** all leaf-injector and logic-bearing factory methods (harness classifies them DIFFER).
- **`@curryable` interacts with `@concrete`**: the structs use `@concrete struct Foo ...` from `ConcreteStructs.jl`. Need to verify `@curryable @concrete struct Foo ...` (or `@concrete @curryable struct Foo ...`) composes correctly. Test on one struct before applying broadly. If composition is tricky, apply `@curryable` separately after the struct definition: annotate with a comment and call the macro on the type name.
- The differential harness (`spike/spike_b_generated.jl`) can be re-run at any point to verify a type is a pure propagator before deleting its hand-written factory.

---

## Environment

- **Active kaimon session:** `99e94f31` on the `factory-spike-a` worktree
- **Julia project path:** `/mnt/storage/dev/PortfolioOptimisers.jl/factory-spike-a/`
- **Load workaround** (required before `using PortfolioOptimisers` in kaimon):

  ```julia
  ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
  ENV["JULIA_PYTHONCALL_EXE"]   = "/usr/bin/python3"
  using PortfolioOptimisers
  ```

- **Git user:** `dcelisgarza`

---

## Suggested skills

- **`/diagnose`** — if a test breaks after deleting a propagator (likely a leaf injector misclassified as a propagator).
- **`/tdd`** — for writing the differential harness as a proper test file (`test/test_factory_generic.jl`) once the first family migration is green.
- **`/code-review`** — after the first family migration commit, before expanding to remaining families.
- **`/grill-with-docs`** — if the `@curryable` + `@concrete` composition question surfaces a deeper design issue worth recording in the ADR.
