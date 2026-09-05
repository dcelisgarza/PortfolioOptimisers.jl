#=
The moment Choice Surfaces, and what membership of one promises.

Three families answer a statistic: expected returns answer `mean`, variance estimators answer
`var` and `std`, and covariance estimators answer `cov` and `cor`. Subtyping the family and
exporting the leaf is the whole contract a caller relies on, so the promise is the family, not
the leaf.

Two censuses read that split, and they must read the SAME one.

  - `test_08l_moment_verb_census.jl` gates the promise: a leaf that joins a family owns the
    family's verbs.
  - `test_08d_dims_guard.jl` drives every leaf-and-verb pairing at `dims = 3` (ADR 0058). A
    pairing the census rejects cannot be driven at all -- the call recurses through the
    surface's own fallback and overflows the stack before it reaches a guard -- so that file
    takes its pairings from this predicate rather than naming a leaf of its own.

The file is not named `test_*.jl`, so `runtests.jl` does not run it as a test file. Both
censuses `include` it.
=#
using PortfolioOptimisers, Statistics, StatsBase, InteractiveUtils

# Every concrete subtype of `T`, at any depth, that `PortfolioOptimisers` itself declares.
#
# The suite gives each file its own module but not its own process, so `subtypes` also
# answers with the probe estimators the files that ran before it declare. The census is
# about the shipped universe, so a leaf from another module is not one of its members, and
# the counts the two censuses take are stable whatever else the worker ran first.
function concrete_leaves(T::Type)
    out = Type[]
    for S in subtypes(T)
        if isabstracttype(S)
            append!(out, concrete_leaves(S))
        elseif parentmodule(S) === PortfolioOptimisers
            push!(out, S)
        end
    end
    return out
end

# The three families, cut so that each leaf appears once. A variance estimator is also a
# covariance estimator in the hierarchy, but it answers `var`/`std` rather than `cov`/`cor`,
# so it belongs to the variance family alone.
function moment_families()
    ve_types = concrete_leaves(PortfolioOptimisers.AbstractVarianceEstimator)
    ce_types = setdiff(concrete_leaves(PortfolioOptimisers.AbstractCovarianceEstimator),
                       ve_types)
    er_types = concrete_leaves(PortfolioOptimisers.AbstractExpectedReturnsEstimator)
    return (er = er_types, ve = ve_types, ce = ce_types)
end

# The three abstract nodes a verb can resolve AT rather than below. A method declared on one
# of them is a fallback the surface offers, not a leaf's own answer.
const MOMENT_SURFACES = (PortfolioOptimisers.AbstractExpectedReturnsEstimator,
                         PortfolioOptimisers.AbstractCovarianceEstimator,
                         PortfolioOptimisers.AbstractVarianceEstimator)

# The method a call on `T` dispatches to, or `nothing` when there is none.
#
# The leaf types come from `subtypes`, so a parametric leaf arrives as a `UnionAll` and
# dispatch on it answers the wrong method: `which(cov, Tuple{Covariance, Matrix{Float64}})`
# reads the surface's fallback, because no single method covers every `Covariance{T1, T2, T3}`.
# The caller holds an INSTANCE, so the census asks about `typeof(S())`.
function dispatched_method(verb::Function, S::Type)
    return try
        which(verb, Tuple{typeof(S()), Matrix{Float64}})
    catch
        nothing
    end
end

# A leaf OWNS a verb when the method it dispatches to is one `PortfolioOptimisers` declares
# BELOW the surface. A method from `StatsBase` is the generic `CovarianceEstimator` one, and a
# method declared ON a surface is a fallback rather than the leaf's answer.
function owns_verb(verb::Function, S::Type)
    m = dispatched_method(verb, S)
    if isnothing(m)
        return false
    end
    if !(m.module === PortfolioOptimisers)
        return false
    end
    slot = m.sig.parameters[2]
    return !any(surface -> slot === surface, MOMENT_SURFACES)
end

# The Correlation Rescale, `cov(ce::AbstractCovarianceEstimator, X)`, is the one fallback the
# covariance surface offers. It reads `cor(ce, X)` and rescales it by `std(ce.ve, X)`, so a
# leaf may take it in place of its own `cov` when it owns `cor` and carries a `ve` field.
# Those two are the fallback's preconditions, and neither is stated by the type.
function takes_correlation_rescale(S::Type)
    m = dispatched_method(Statistics.cov, S)
    if isnothing(m)
        return false
    end
    return m.module === PortfolioOptimisers &&
           m.sig.parameters[2] === PortfolioOptimisers.AbstractCovarianceEstimator &&
           hasfield(typeof(S()), :ve)
end

# What each family's membership promises, and what answers it.
#
# `cor` is never a fallback: nothing declares one on the covariance surface, so a leaf that
# does not own `cor` reaches `StatsBase`'s generic `cor`, which calls `cov` straight back into
# the surface's `cov`, which calls `cor`. That pair is the stack overflow, and requiring `cor`
# of every leaf is what closes it.
function answers_family(family::Symbol, S::Type)
    if family === :ce
        return owns_verb(Statistics.cor, S) &&
               (owns_verb(Statistics.cov, S) || takes_correlation_rescale(S))
    elseif family === :ve
        return owns_verb(Statistics.var, S) && owns_verb(Statistics.std, S)
    else
        return owns_verb(Statistics.mean, S)
    end
end

# The verbs a family's census drives, in the order a report prints them.
function family_verbs(family::Symbol)
    return if family === :ce
        [Statistics.cov, Statistics.cor]
    elseif family === :ve
        [Statistics.var, Statistics.std]
    else
        [Statistics.mean]
    end
end

#=
ONE exemption, and it is the subject of the gate rather than a note beside it.

`RegimeAdjustedExpWeightedCovariance` declares no verb at all, and
`RegimeAdjustedExpWeightedVariance` declares `var` alone while it sits on the covariance
surface. Both entered the library with no implementation behind the mathematics their
docstrings state and with no test: `sweep/manifest.toml` marks both files `swept = false`, and
ADR 0082 records the same two files as the only two at 0.0 % coverage, which is the case the
coverage ratchet was built around. ADR 0058 recorded the gap in its Notes when the `dims`
census met it.

`test/test_08m_variance_series.jl` has since covered the variance file, so the 0.0 % reading
ADR 0082 records for it is history. That changes nothing here: the file still declares no
`std`, and the covariance file still declares nothing at all.

Nothing here can supply the missing mathematics, so issue #637 carries it. This tuple empties
when that issue closes.
=#
const VERB_EXEMPT = (:RegimeAdjustedExpWeightedCovariance,
                     :RegimeAdjustedExpWeightedVariance)
