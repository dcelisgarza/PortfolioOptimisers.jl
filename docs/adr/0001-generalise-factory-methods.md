---
status: accepted — adopt C (@propagatable macro)
---

# Generalise the pure-propagation `factory` methods

## Context

`factory` is the library's variable propagation mechanism: because all configuration structs are immutable and heavily composed, `factory(x, runtime_data...)` returns a new, fully-configured copy of `x` with runtime-computed values (observation weights, previous portfolio weights, a prior result, a solver, an uncertainty set) threaded down into the fields that need them. See `[[factory]]` in `CONTEXT.md`.

There are ~180 hand-written `factory` methods. They fall into two kinds:

1. **Pure propagation** (the majority). They contain no logic: rebuild the struct via its keyword constructor, recursing `factory(field, args...)` into the data-bearing fields and passing the rest through unchanged. Example:

   ```julia
   factory(ce::Covariance, w::ObsWeights) =
       Covariance(; me = factory(ce.me, w), ce = factory(ce.ce, w), alg = ce.alg)
   ```

   A new method like this must be written every time an eligible composite struct is added — pure boilerplate, and a place to forget a field.

2. **Genuine logic** (the minority). They do real work: extracting `sigma`/`mu` from a prior result (the `factory(r::Variance, pr::AbstractPriorResult, …)` family), the EVaR-style measures with a separate solver dispatch, and the `factory(res::…Result, fb::Option{<:OptE_Opt})` fallback-injection methods.

An identity fallback already exists and terminates recursion at non-eligible values:

```julia
factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, args...; kwargs...) = a
factory(a::AbstractVector{<:Union{…}}, args...; kwargs...) = [factory(ai, args...; kwargs...) for ai in a]
```

## Decision

Introduce a generic propagation `factory` that rebuilds any eligible struct by recursing `factory` into each of its fields, replacing the pure-propagation methods. Genuine-logic methods (and leaf injectors — the methods where runtime data actually lands in a struct) remain explicit overrides and win by dispatch specificity.

**Scope:** only the pure-propagation (intermediate composite) methods are in scope. Out of scope and kept hand-written: leaf injectors (where the runtime value is stored, e.g. `SimpleExpectedReturns` ← `w`) and logic-bearing methods (prior-result extraction, solver injection, result fallback).

**Mechanism — verdict after all three spikes: adopt C (`@propagatable` macro).** A is rejected (type-unstable). B and C are both type-stable and value-identical to the hand-written methods. Warm-call latency is a tie (0.069 ms B vs 0.071 ms C per 1 000 calls on `FactorPrior`). Cold-start first-call latency (fresh REPL, 15-type corpus) is **536 ms for B vs 170 ms for C** (3.2× advantage for C); the outliers are:

- `EquilibriumExpectedReturns` (179 ms B / 11 ms C), and
- `FactorPrior` (88 ms B / 31 ms C).

This is because each concrete parameterisation of a struct triggers a fresh `@generated` code-generation step under B; with C the explicit per-struct method covers all parameterisations and precompiles with the package. The real `@propagatable` advantage is larger still — the spike injected one method per concrete type; the macro emits one method per bare struct name, so novel user-created parameterisations cost nothing extra. The deciding axes are therefore criteria 3 and 4: **C wins on both TTFX (3.2×) and greppability.** B's only advantage — zero ceremony for new structs — is not decisive for a research library.

**Spike A + B outcome:** A was prototyped first; it surfaced the verdict early. On deeply-nested composites (`FactorPrior`, `HighOrderPriorEstimator`, 5-level nests), **A is type-unstable — inference widens to `Any`** — whereas the existing hand-written `factory` is concrete. The cause was isolated to the `ntuple(i -> _fac(getfield(x,i), …), …)` closure, which inference cannot see through at depth. A `@generated` body that **unrolls** the per-field reconstruction (mechanism **B**) is **type-stable, matches the hand-written return type exactly, and is value-identical** across the corpus. Conclusion: A fails criterion 1; B passes.

**Spike C outcome:** C passes the differential harness (11/15 propagators match, 4 keepers) and is 15/15 type-stable — identical classification to B. **Key finding:** the macro must emit the *unparameterised* constructor call (`EquilibriumExpectedReturns(; …)` not `T(; …)`), because `factory` changes field type parameters (e.g. `Nothing` → `ProbabilityWeights{…}`). The unparameterised call is exactly what the hand-written methods already use, so C's generated code is visually identical to the hand-written propagators it replaces. Warm-call overhead is indistinguishable from B. Spike at `spike/spike_c_macro.jl`.

## Considered Options (to be spiked)

- **A. Runtime reflection** — a single generic method, `fieldnames` + `ntuple(_fac∘getfield)` + `Accessors.setproperties`. **REJECTED (spiked).** Correct values, one method, least code — but **type-unstable for deep composites** (`FactorPrior`, `HighOrderPriorEstimator` infer to `Any`); the `ntuple` closure is the inference barrier and `Val`/concrete-arg variants did not fix it. Since the hand-written methods it replaces are type-stable, A is a regression on the top criterion.
- **B. `@generated` function** — one generic method whose body **unrolls** the per-field reconstruction at compile time (`setproperties(x, NamedTuple{fieldnames}((_fac(getfield(x,1),…), _fac(getfield(x,2),…), …)))`). **FRONT-RUNNER (spiked):** type-stable, return type identical to hand-written, value-identical across the corpus. Cons: `@generated` carries precompile/invalidation cost — to be measured against C.
- **C. Struct-definition macro** (`@propagatable struct …`) — auto-generates an explicit per-type `factory` method at definition time using the *unparameterised* constructor call so field type-parameter changes are handled correctly. **ADOPTED (spiked).** 15/15 type-stable, 11/15 propagators match hand-written — identical to B. Warm-call latency ties B (0.071 ms vs 0.069 ms per 1 000 calls). Methods stay explicit, greppable, and precompile-friendly. Cons: `@propagatable` annotation required on every new eligible struct (zero-ceremony advantage goes to B). Spike at `spike/spike_c_macro.jl`.

### Comparison criteria (priority order)

A spike must pass the differential harness to be eligible at all; the rest break ties, most decisive first:

0. **Correctness** *(gate, not a tiebreaker)* — passes the differential harness against the hand-written methods over the full corpus.
1. **Type-stability / inference** — `factory` runs on every `optimise`/`prior` call; an inference-unstable generic poisons every downstream optimiser. Reflection (A) is the risk here; `@generated` (B) and the macro (C) emit concrete constructor calls. Measured with `@code_warntype` / `JET`.
2. **Runtime speed / allocations** — reconstruction overhead per call (`@benchmark` on representative nested estimators).
3. **TTFX / precompile cost** — compile latency, invalidations, precompile-friendliness. `@generated` and reflection are the risks; the macro yields ordinary methods.
4. **Readability / greppability** — C keeps explicit, greppable per-type methods; A/B make propagation invisible at the definition site. Weighted as a research-oriented library whose contributors read the code often, but below the performance axes.
5. **LOC / boilerplate removed** — all three remove comparable boilerplate, so this rarely decides anything.

"Just works" for a newly-added struct (A/B need zero ceremony; C needs a `@propagatable` annotation) is noted per-spike but treated as a property to describe, not a scored axis.

## Shared semantic spec (all three spikes implement this identically)

The three spikes differ **only** in the reconstruction mechanism. The recursion semantics are fixed so the comparison is apples-to-apples:

- **Rebuild-eligible** (generic recurses into fields and reconstructs): `AbstractEstimator ∪ AbstractAlgorithm ∪ AbstractCovarianceEstimator`. **Spike finding:** covariance/variance estimators are **not** `<: AbstractEstimator` — `AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator`, an external root — so PO's covariance root must be named explicitly or the generic never fires for the entire covariance family. (An incomplete union fails safe: it leaves propagators un-deleted rather than producing wrong results, but the boilerplate-reduction win is only realised once it is complete.)
- **External-boundary passthrough (keep):** `factory(ce::StatsBase.CovarianceEstimator, args…) = ce` must be **kept** — StatsBase's own concretes (`SimpleCovariance`, …) appear as leaf fields and are not PO types. This one is the boundary to an external package, unlike the PO family-identities which are deleted. PO covariance types win over it by specificity (`AbstractCovarianceEstimator` is the subtype).
- **Passthrough** (returned unchanged): `AbstractResult ∪ Nothing`, plus every non-eligible field type (numbers, symbols, functions, numeric vectors, `NamedTuple`, `Dict`, …). Results appear inside estimators only as already-computed inputs (the "estimator-or-result" union fields) and must not be rebuilt.
- **Zero-field short-circuit**: structs with `fieldcount == 0` are passed through unchanged (no pointless rebuild of singletons; sidesteps reconstruction edge cases).
- **Dispatch precedence**: leaf injectors (where runtime data is stored) and logic methods are more specific than the generic and win automatically. The generic only fires where no specialised method exists — i.e. the pure propagators.
- **Arg threading**: a single generic `factory(x::Eligible, args...; kwargs...)` forwards every argument shape (`ObsWeights`, portfolio `VecNum`, `AbstractPriorResult`, solver, uncertainty set, `fb` fallback) unchanged into children — no per-arg-type generic needed.

### Required removals (the diff is not purely additive)

The existing abstract-family **identity passthroughs** shadow a blanket generic and must be deleted, or the generic never fires for those families (a deleted concrete propagator would be silently caught by the family identity and returned un-recursed):

- Delete `factory(::AbstractExpectedReturnsEstimator, args…) = me`, `factory(::AbstractExpectedReturnsAlgorithm, …)`, `factory(::MomentMeasureAlgorithm, …)`, `factory(::AbstractClustersAlgorithm, …)`, `factory(::AbstractPhylogenyAlgorithm, …)`, `factory(::JuMPReturnsEstimator, …)`, `factory(::AbstractBaseRiskMeasure, …)`, and any sibling family-identity methods.
- **Split** the blanket fallback in `02_Tools.jl`: keep `factory(a::Union{Nothing, <:AbstractResult}, args…) = a` (passthrough); route `AbstractEstimator ∪ AbstractAlgorithm` to the generic rebuild.
- **Keep** the vector mapping (`VecBaseRM` and friends), generalised to "vector of rebuild-eligible".

### Container fields (settled by code inspection)

Estimators-in-collections are always typed `Vector`s (`VecOptE_Opt`, `VecBaseRM`, …). Tuple / `NamedTuple` / `Dict` fields only ever hold non-eligible values (solver settings, executors, kwargs, string groupings). So the generic recursion handles **eligible scalars** and **`AbstractArray` with eligible element type** (map `factory` over elements); **everything else passes through** — numeric matrices/vectors, `Dict`s, `NamedTuple`s, `Tuple`s, executors. Recursing must *not* touch numeric arrays (would needlessly reallocate and risk changing the array type). Any future struct that hides estimators in a non-Vector container keeps an explicit method.

## Verification & migration

**Gate: differential testing (mechanism-independent harness).** In each spike, introduce the generic as `factory2` *alongside* the existing `factory`, recursing into children via the **real** `factory` so leaf injectors / logic still fire at the leaves. Build a corpus of nested estimator/algorithm instances (harvested from the existing test suite plus a few deep hand-built cases) crossed with every arg shape (`ObsWeights`, portfolio `VecNum`, `AbstractPriorResult`, solver, uncertainty set). Assert `struct_equal(factory2(x, a...), factory(x, a...))` over the whole corpus. Only when green is the family's old methods deleted and `factory2` folded into `factory`. The same harness runs against all three spikes, so "did A, B, C behave identically?" becomes a literal pass/fail — it doubles as the comparison harness.

**Bonus: the harness auto-classifies methods.** For a given struct, `struct_equal(factory2(x,a…), factory(x,a…))` being **true** means the hand-written method is a pure propagator → safe to delete. **False** means it stores/extracts something the reflection can't see (leaf injector or logic) → must stay. So the harness is also the migration inventory: it tells us exactly which of the ~180 methods are deletable rather than relying on manual reading.

**Harness prerequisite — `struct_equal`.** There is no custom `==`/`isequal` in the codebase; Julia's default `==` falls back to `===` (egal), which is too strict here (a propagator that maps over a vector allocates a fresh `Vector`, failing `===` despite structural equality). The harness must define a recursive structural compare: `==`/`isapprox` on numeric leaves and arrays, recurse field-wise on eligible structs, identity/`==` on the rest. This helper is test-only — it must **not** leak a global `==` overload into the package. **Spike finding:** struct fields can hold **`Type` objects** (e.g. `Posdef.alg = NearestCorrelationMatrix.Newton`), so the comparator must treat `Type`/`Function`/`Module` as opaque leaves (`===`), or field-wise recursion into type metadata stack-overflows. The reflective rebuild itself is unaffected (a `Type` is not rebuild-eligible, so it passes through).

**Migration: family-by-family.** Eliminate propagators one family at a time (expected-returns, covariance, prior, phylogeny, risk measures, optimisers, …): delete that family's pure-propagation methods and its shadowing abstract-family identity passthrough, run the differential harness + full suite, commit, move on. Smaller blast radius and a bisectable history versus a big-bang deletion.

## Consequences

- **Type-stability is the deciding axis, and it is mechanism-sensitive.** An unrolled/compile-time reconstruction (B or C) preserves the inference the hand-written methods already have; a runtime-reflection rebuild (A) silently degrades it on deep priors. The differential harness must therefore assert **type-stability** (`isconcretetype(only(return_types(...)))`), not just value-equality, or the regression slips through green value tests.
- **The eligibility union must name `AbstractCovarianceEstimator` explicitly** (it sits under `StatsBase.CovarianceEstimator`, outside `AbstractEstimator`). The `factory(::StatsBase.CovarianceEstimator)=ce` external-boundary passthrough is kept; the PO family-identity passthroughs are deleted.
- **The harness is also the migration inventory**: per struct, value-equal ⇒ deletable propagator, differ ⇒ keep (leaf injector/logic). On the expected-returns + covariance + prior sample, 11/15 classified deletable, 4 keepers — the win concentrates in composite families, not in the leaf-heavy expected-returns family.
- **`struct_equal` must treat `Type`/`Function`/`Module` as opaque leaves** (struct fields can hold types), or it stack-overflows.
- **`@propagatable` must emit the *unparameterised* constructor call**, not `T(; …)`. `factory` changes field type parameters (e.g. a `Nothing` weight field becomes `ProbabilityWeights{…}` after injection); constructing the original parameterised type `T` fails. Using the bare name (e.g. `EquilibriumExpectedReturns(; …)`) lets Julia infer the new parameters. This makes C's generated code visually identical to the hand-written propagators it replaces.
- **B vs C verdict — adopt C (confirmed, status `accepted`).** Cold-start first-call latency over the 15-type corpus: **536 ms total for B, 170 ms for C (3.2× faster)**. Per-type breakdown (B ms / C ms): `EquilibriumExpectedReturns` 179/11, `CustomValueExpectedReturns` 85/6, `FactorPrior` 88/31, `HighOrderPriorEstimator` 30/22, `EmpiricalPrior` 15/10, `PortfolioOptimisersCovariance` 14/9, `Covariance` 12/8, `SimpleVariance` 12/10, `ShrunkExpectedReturns` 14/10, `ExcessExpectedReturns` 43/8, `StdDevExpectedReturns` 12/9, `VarianceExpectedReturns` 11/9, `WindowedExpectedReturns` 11/15, `MedianExpectedReturns` 8/6, `SimpleExpectedReturns` 3/5. Spike C measured one explicit method per concrete type; the real `@propagatable` emits one method per bare struct name, so novel user parameterisations have no additional first-call cost — the actual C advantage is larger. Warm-call latency (1 000 calls, hot): B 0.069 ms / C 0.071 ms — tie.
