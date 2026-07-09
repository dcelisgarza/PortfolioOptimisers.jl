"""
    Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)

Deliberately unsupported: combinatorial cross-validation stays returns-level.

Combinatorial folds recombine non-contiguous test groups, which is incompatible with the contiguous-window semantics price-level (pipeline) splitting relies on for stateful preprocessing. Throws an `ArgumentError`; split returns-level data instead, or use [`KFold`](@ref) / a [`WalkForwardEstimator`](@ref) at the prices level.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`Base.split(kf::KFold, rd::Prices_RR)`](@ref)
"""
function Base.split(::CombinatorialCrossValidation, ::AbstractPricesResult)
    return throw(ArgumentError("CombinatorialCrossValidation is returns-level only; price-level (pipeline) splitting requires contiguous windows — use KFold or a walk-forward scheme instead"))
end
"""
    Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)

Deliberately unsupported: multiple-randomised cross-validation stays returns-level.

Its resampled asset subsets and randomised paths are defined over the returns matrix, not over price observations, so it cannot drive the contiguous input-row windows price-level (pipeline) splitting requires. Throws an `ArgumentError`; split returns-level data instead.

# Related

  - [`MultipleRandomised`](@ref)
  - [`Base.split(kf::KFold, rd::Prices_RR)`](@ref)
"""
function Base.split(::MultipleRandomised, ::AbstractPricesResult)
    return throw(ArgumentError("MultipleRandomised is returns-level only; price-level (pipeline) splitting requires contiguous input-row windows — use KFold or a walk-forward scheme instead"))
end
