"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the returns matrix with the missing rows of every investable asset filled with zero, and say so above the fitting estimator's resolved `fill_limit`.

A mask-aware moment estimator answers a young asset from the observations it has, so the asset is investable and its returns column still carries a `NaN` at every row before it listed. Every consumer of a Prior Result reads that column — the JuMP model, the meta-optimisers and the value-level `expected_*` functions among them — so the fill is paid **once**, here, on the estimator's own pass, rather than at each of them.

A non-investable asset keeps its `NaN` column, so [`investable_mask`](@ref) is unchanged. `mu`, `sigma` and every other block are untouched, because the estimator computed them from the rows it saw.

**The share the fill tests is per asset**: the worst investable column's own count of filled entries over the number of observations. A matrix-wide denominator scales with the universe, so the column the notice exists to catch disappears inside it — one asset of a hundred whose column is seven-tenths invented is seven thousandths of the matrix, under any limit a caller would set. The matrix-wide share is reported in the message as context and trips nothing.

How much of the trade passes in silence is the fitting estimator's own answer, carried in its `fill_limit` field and resolved at the fit by [`resolve_fill_limit`](@ref) against the coverage floor of its arms — the share is a property of one fit, not of the session, so two priors in one program may answer differently. Under `strict = false` the trade is silent while the worst column's share stays at or below the resolved limit, and is named through [`strict_diagnostic`](@ref) above it. Under `strict = true` **any** fill refuses, whatever the limit holds.

Two mask-aware families reach this verb. The exponentially weighted family carries no [`CoveragePolicy`](@ref), so a `fill_limit` of `nothing` names every fill there. The plain family is mask-aware exactly where a policy is set, and a policy derives a limit that never fires, so an available-case fit names nothing while it does what it was configured to do. A plain estimator with no policy never reaches this verb at all: under the whole-window rule, an asset it could not cover leaves the Coverage Universe and is not investable.

# Algorithm

 1. Return `X` itself when every entry of `X` is finite, which is every fit over a complete window.
 2. Derive the Investable Mask from `mu` and the diagonal of `sigma`, as [`investable_mask`](@ref) does, and find the pairs to fill with [`scenario_fill_pairs`](@ref). Return `X` itself when there are none, which is a gap that belongs to a non-investable asset alone.
 3. Narrow the pairs to report with [`scenario_fill_report`](@ref), which drops the assets a carry state has already named and is the identity when nothing remembers. Under `strict` the narrowing is skipped, because `strict` refuses any fill and must not depend on how often the estimator has been read out.
 4. Count the reported pairs per asset, take the worst column's share of its own observations, and report through [`strict_diagnostic`](@ref) when `strict` holds, when `fill_limit` is `nothing`, or when that share exceeds `fill_limit`. Record the assets just named with [`scenario_fill_remember!`](@ref).
 5. Return a copy of `X` with zero written at each of the filled pairs — every one of them, not just the reported ones — and the `NaN` of every non-investable asset left where it is.

# The notice fires once per asset, not once per step

`named` is what separates the batch call from the online one. A batch fit passes `nothing`, reports every fill it finds, and remembers nothing. A call of `prior(pe)` with no data on a folded prior passes the set its [`PriorCarryState`](@ref) carries, so a walk-forward names an asset at the step it lists and stays quiet afterwards instead of emitting the same notice at every one of two thousand steps. The **fill itself is unchanged**: every filled pair is written at every call, whatever the set holds. Under `strict = true` the first fill still throws.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `mu`: The expected returns the estimator answered, `NaN` at a non-investable asset.
  - `sigma`: The covariance the estimator answered, `NaN` on the diagonal at a non-investable asset.
  - `strict`: If `true`, any fill raises an `ArgumentError`; if `false`, a fill above the share warns.
  - `fill_limit`: The share of an investable column the fitting estimator accepts in silence, after [`resolve_fill_limit`](@ref), or `nothing` when it accepts none.
  - `named`: The assets already named, written in place when a notice fires, or `nothing` when nothing remembers.

# Validation

  - The worst investable column's filled share is at or below `fill_limit`, else a warning naming the assets is emitted, or an `ArgumentError` naming them is raised under `strict`, which any fill raises. A `fill_limit` of `nothing` is at or below no share, so any fill is named.

# Returns

  - `X::MatNum`: The returns matrix whose investable columns are finite.

# Related

  - [`scenario_fill_pairs`](@ref)
  - [`scenario_fill_msg`](@ref)
  - [`scenario_fill_report`](@ref)
  - [`scenario_fill_remember!`](@ref)
  - [`scenario_window`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`filter_held_gaps`](@ref)
"""
function scenario_fill(X::MatNum, mu::VecNum, sigma::MatNum, strict::Bool,
                       fill_limit::Option{<:Real},
                       named::Option{<:AbstractSet{<:Integer}} = nothing)
    if all(isfinite, X)
        return X
    end
    imsk = isfinite.(mu) .& isfinite.(LinearAlgebra.diag(sigma))
    filled = scenario_fill_pairs(X, imsk)
    if isempty(filled)
        return X
    end
    # `strict` refuses **any** fill, so it never reads the memory: an asset that a previous
    # call of `prior(pe)` with no data named is still a fill, and silencing it here would
    # make `strict` depend on how many times the estimator had been read out.
    report = strict ? filled : scenario_fill_report(filled, named)
    if !isempty(report)
        counts = zeros(Int, size(X, 2))
        for (_, i) in report
            counts[i] += 1
        end
        wi = argmax(counts)
        worst = counts[wi] / size(X, 1)
        frac = length(filled) / length(X)
        if strict || isnothing(fill_limit) || worst > fill_limit
            strict_diagnostic(scenario_fill_msg(report, wi, worst, frac, fill_limit),
                              strict; id = @log_id())
            scenario_fill_remember!(named, report)
        end
    end
    Xf = copy(X)
    for (t, i) in filled
        Xf[t, i] = zero(eltype(Xf))
    end
    return Xf
end
