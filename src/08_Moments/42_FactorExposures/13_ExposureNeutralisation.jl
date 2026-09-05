"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve one Neutralisation name to the raw factor indices it names.

A name is a factor name or a Factor Family label. A factor name takes precedence, so a label that names both resolves to the single factor.

# Arguments

  - `nm::AbstractString`: The name to resolve.
  - `nf::VecStr`: Names of the raw factor axis.
  - `fam::VecStr`: Family label of each raw factor.

# Validation

  - `nm` is a factor name or a family label.

# Returns

  - `idx::Vector{Int}`: The raw factor indices, in increasing order.

# Related

  - [`neutralise_exposures!`](@ref)
"""
function neutralisation_indices(nm::AbstractString, nf::VecStr, fam::VecStr)::Vector{Int}
    k = findfirst(isequal(nm), nf)
    if !isnothing(k)
        return [k]
    end
    idx = findall(isequal(nm), fam)
    @argcheck(!isempty(idx),
              ArgumentError("$nm is neither a factor name nor a Factor Family label. The factors are $(collect(nf)) and the families are $(unique(fam))"))
    return idx
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Expand a list of Neutralisation targets to the raw factor indices they name.

Each target is a factor name or a Factor Family label, resolved by [`neutralisation_indices`](@ref). A factor named twice is kept at its first appearance, so the order the caller wrote is the column order of the design.

# Arguments

  - `targets::VecStr`: The target names.
  - `nf::VecStr`: Names of the raw factor axis.
  - `fam::VecStr`: Family label of each raw factor.

# Validation

  - `targets` is not empty, and the rules of [`neutralisation_indices`](@ref).

# Returns

  - `idx::Vector{Int}`: The raw factor indices, in the order the targets name them.

# Related

  - [`neutralise_exposures!`](@ref)
  - [`neutralisation_indices`](@ref)
"""
function neutralisation_targets(targets::VecStr, nf::VecStr, fam::VecStr)::Vector{Int}
    @argcheck(!isempty(targets), IsEmptyError("a Neutralisation entry needs a target"))
    idx = Int[]
    for t in targets
        for i in neutralisation_indices(t, nf, fam)
            if i ∉ idx
                push!(idx, i)
            end
        end
    end
    return idx
end
"""
    neutralise_exposures!(Ms::AbstractArray{<:Real, 3}, neutralise::AbstractVector{<:Pair},
                          cre::AbstractCrossSectionalRegressionEstimator, bw::MatNum,
                          nf::VecStr, fam::VecStr,
                          ct::AbstractCrossSectionalTransform = CrossSectionalStandardiser())

Neutralise Factor Exposures against other Factor Exposures, in place.

# Algorithm

 1. Take the entries of `neutralise` in the order the caller wrote them, so a later entry sees the exposures an earlier one changed.
 2. Resolve the key and the targets of the entry to raw factor indices, and refuse a key that overlaps its own targets.
 3. For each factor the key names, in increasing index order, build the regression weights: `bw`, with a zero wherever the key exposure or a target exposure of that asset is not finite.
 4. Regress the key's exposure across the assets on the targets' exposures with `cre` under those weights, and take the residual.
 5. Re-standardise the residual with `ct` under the same weights, and write it over the key's exposure.

# Arguments

  - `Ms::AbstractArray{<:Real, 3}`: Exposure history, `observations × assets × factors`. It is changed in place.
  - `neutralise`: Pairs of `key => targets`, where the key and each target is a factor name or a Factor Family label. A family key neutralises each of its members independently against the same targets.
  - `cre`: Cross-sectional regression estimator that fits the residualisation.
  - `bw::MatNum`: Benchmark weight history, `observations × assets`.
  - `nf::VecStr`: Names of the raw factor axis, of length `size(Ms, 3)`.
  - `fam::VecStr`: Family label of each raw factor, of length `size(Ms, 3)`.
  - `ct`: Cross-sectional transform that re-standardises each residual.

# Validation

  - `nf` and `fam` are as long as the factor axis of `Ms`.
  - `bw` matches `Ms` on the observation and asset axes.
  - Every key and every target names a factor or a family.
  - No key overlaps its own targets, because a factor cannot be neutralised against itself.

# Returns

  - `nothing`. `Ms` carries the neutralised exposures.

# Related

  - [`factor_exposure`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
  - [`neutralisation_indices`](@ref)
"""
function neutralise_exposures!(Ms::AbstractArray{<:Real, 3},
                               neutralise::AbstractVector{<:Pair},
                               cre::AbstractCrossSectionalRegressionEstimator, bw::MatNum,
                               nf::VecStr, fam::VecStr,
                               ct::AbstractCrossSectionalTransform = CrossSectionalStandardiser())::Nothing
    @argcheck(!isempty(Ms), IsEmptyError("Ms cannot be empty"))
    T, N, K = size(Ms)
    @argcheck(length(nf) == K,
              DimensionMismatch("nf ($(length(nf))) must match the factor axis of Ms ($K)"))
    @argcheck(length(fam) == K,
              DimensionMismatch("fam ($(length(fam))) must match the factor axis of Ms ($K)"))
    @argcheck(size(bw) == (T, N),
              DimensionMismatch("bw ($(size(bw, 1))×$(size(bw, 2))) must match Ms ($T×$N) on the observation and asset axes"))
    for pr in neutralise
        key = String(first(pr))
        kidx = neutralisation_indices(key, nf, fam)
        tidx = neutralisation_targets(neutralisation_names(last(pr)), nf, fam)
        ovl = intersect(kidx, tidx)
        @argcheck(isempty(ovl),
                  ArgumentError("the Neutralisation key $key names the factors $([nf[i] for i in ovl]), which are also its targets, and a factor cannot be neutralised against itself"))
        X = Ms[:, :, tidx]
        for k in kidx
            W = neutralisation_weights(Ms, X, bw, k)
            csr = cross_sectional_regression(cre, X, Ms[:, :, k], W)
            Ms[:, :, k] = cross_sectional_transform(ct, csr.eps; w = W)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the target names of one Neutralisation entry.

The right of a Pair is one name or a list of names, and both forms answer a vector of names.

# Arguments

  - `targets`: The right of the Pair.

# Returns

  - `targets::VecStr`: The target names.

# Related

  - [`neutralise_exposures!`](@ref)
"""
function neutralisation_names(targets::AbstractString)::Vector{String}
    return [String(targets)]
end
function neutralisation_names(targets::VecStr)::VecStr
    return targets
end
"""
    neutralisation_weights(y::MatNum, X::AbstractArray{<:Real, 3}, bw::MatNum)
    neutralisation_weights(Ms::AbstractArray{<:Real, 3}, X::AbstractArray{<:Real, 3},
                           bw::MatNum, k::Integer)

Return the regression weights of one Neutralisation.

An asset whose response or whose target exposure is not finite carries no weight in that regression, because a cross-sectional fit refuses a non-finite entry at a positive weight. A non-finite base weight and a negative one are read as zero on the same rule.

The response is one Factor Exposure when a Factor Exposure is neutralised, and one Descriptor score when a Descriptor score is: the four-argument method names the exposure by its raw factor index and forwards it to the three-argument one, so the two neutralisations share one rule.

# Arguments

  - `y::MatNum`: The response being neutralised, `observations × assets`.
  - `Ms::AbstractArray{<:Real, 3}`: Exposure history, `observations × assets × factors`.
  - `X::AbstractArray{<:Real, 3}`: Target exposures, `observations × assets × targets`.
  - `bw::MatNum`: Base weight history, `observations × assets`. It is the benchmark weights of a Factor Exposure Neutralisation and the estimation mask of a Descriptor score one.
  - `k::Integer`: Raw index of the factor being neutralised.

# Returns

  - `W::Matrix{<:Real}`: The regression weights, `observations × assets`.

# Related

  - [`neutralise_exposures!`](@ref)
  - [`neutralise_scores!`](@ref)
  - [`cross_sectional_design_mask`](@ref)
"""
function neutralisation_weights(y::MatNum, X::AbstractArray{<:Real, 3}, bw::MatNum)
    Tf = promote_type(float(real(eltype(y))), float(real(eltype(bw))))
    T, N = size(bw)
    W = zeros(Tf, T, N)
    for i in 1:N, t in 1:T
        b = bw[t, i]
        if !isfinite(b) || b <= zero(b) || !isfinite(y[t, i])
            continue
        end
        ok = true
        for p in axes(X, 3)
            if !isfinite(X[t, i, p])
                ok = false
                break
            end
        end
        if ok
            W[t, i] = Tf(b)
        end
    end
    return W
end
function neutralisation_weights(Ms::AbstractArray{<:Real, 3}, X::AbstractArray{<:Real, 3},
                                bw::MatNum, k::Integer)
    return neutralisation_weights(view(Ms, :, :, k), X, bw)
end
