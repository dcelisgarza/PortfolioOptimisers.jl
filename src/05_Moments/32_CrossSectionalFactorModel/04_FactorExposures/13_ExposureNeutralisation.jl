"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve one Neutralisation name to the raw factor indices it names.

A name is a factor name or a Factor Family label. A name that is both resolves to the single factor.

# Arguments

  - `nm::AbstractString`: The name to resolve.
  - `nf::VecStr`: Names of the raw factor axis.
  - `fam::VecStr`: Family label of each raw factor.

# Validation

  - `nm` is a factor name or a family label. Any other name raises an `ArgumentError` that lists the factors and the families.

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

A target is a factor name or a Factor Family label, and [`neutralisation_indices`](@ref) resolves it. A factor that two targets name keeps the place of its first appearance, so the order of the targets is the column order of the design.

# Arguments

  - `targets::VecStr`: The target names.
  - `nf::VecStr`: Names of the raw factor axis.
  - `fam::VecStr`: Family label of each raw factor.

# Validation

  - `!isempty(targets)`. An empty list raises an `IsEmptyError`.
  - The rules of [`neutralisation_indices`](@ref).

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

Each entry of `neutralise` replaces the exposures of its key with the scored residual of a cross-sectional fit on the exposures of its targets. A later entry reads the exposures that an earlier entry wrote, so the order of the entries changes the result.

# Mathematical definition

For one entry, one factor ``k`` that its key names, and one observation ``t``:

```math
\\begin{align}
\\omega_{tik} &= u_{ti} \\, \\mathbb{1}\\left[0 < u_{ti} < \\infty,\\ B_{tik} \\in \\mathbb{R},\\ \\boldsymbol{x}_{ti} \\in \\mathbb{R}^{\\lvert \\mathcal{T} \\rvert}\\right]\\,, \\\\
\\left(\\hat{a}_{tk}, \\hat{\\boldsymbol{\\gamma}}_{tk}\\right) &= \\underset{a,\\, \\boldsymbol{\\gamma}}{\\arg\\min} \\sum_{i = 1}^{N} \\omega_{tik} \\left(B_{tik} - a - \\boldsymbol{x}_{ti}^{\\intercal} \\boldsymbol{\\gamma}\\right)^{2}\\,, \\\\
e_{tik} &= B_{tik} - \\hat{a}_{tk} - \\boldsymbol{x}_{ti}^{\\intercal} \\hat{\\boldsymbol{\\gamma}}_{tk}\\,, \\\\
B^{\\prime}_{tik} &= \\mathcal{Z}_{t}\\left(\\boldsymbol{e}_{t \\cdot k}\\right)_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:B_tik_cs]) It is the exposure before the entry.
  - $(math_dict[:u_ti_cs]) Here it is the benchmark weight in `bw`.
  - ``\\mathcal{T}``: The raw factors that the targets of the entry name. It holds no factor that the key names.
  - ``\\boldsymbol{x}_{ti}``: The target exposures of asset ``i`` at observation ``t``, the exposures ``B_{tij}`` to the factors ``j \\in \\mathcal{T}``.
  - ``\\omega_{tik}``: Regression weight of asset ``i`` in the Neutralisation of factor ``k`` at observation ``t``. It is zero where the benchmark weight, the key exposure or a target exposure is not finite, and where the benchmark weight is not positive.
  - ``\\hat{a}_{tk}``, ``\\hat{\\boldsymbol{\\gamma}}_{tk}``: Intercept and slopes of the weighted fit. The intercept is ``0`` when `cre.intercept` is `false`. The least squares is the fit of [`CrossSectionalLinearRegression`](@ref), and its solve algorithm chooses among the minimisers of a rank deficient fit. [`CrossSectionalTargetRegression`](@ref) fits its own target under the same weights instead.
  - ``e_{tik}``: Residual of asset ``i``. It is defined for every asset whose key exposure and target exposures are finite, including an asset with ``\\omega_{tik} = 0``, and it is `NaN` for every other asset.
  - ``\\mathcal{Z}_{t}``: The transform `ct` of one cross-section of observation ``t``, under the weights ``\\omega_{t \\cdot k}``, as its own docstring states.
  - ``B^{\\prime}_{tik}``: The neutralised exposure that replaces ``B_{tik}``.
  - $(math_dict[:N])

The normal equations of the least squares give ``\\sum_{i} \\omega_{tik} e_{tik} \\boldsymbol{x}_{ti} = \\boldsymbol{0}``, so the residual is orthogonal to the target exposures under the weights. With an intercept, they also give ``\\sum_{i} \\omega_{tik} e_{tik} = 0``, so the residual is uncorrelated with the target exposures under the weights. Targets that span the constant, such as the one-hot exposures of a whole Factor Family, give the same zero sum without an intercept. [`CrossSectionalStandardiser`](@ref) maps each cross-section by one affine map, so it keeps a zero correlation. It keeps the orthogonality only when the weighted sum of the residual is zero.

# Algorithm

 1. Check the axes. When `neutralise` holds an entry, check that the element type of `Ms` is not an `Integer`, because a residual is not an integer.
 2. Take the entries of `neutralise` in the order that the caller wrote them.
 3. Resolve the key and the targets of the entry to the raw factor indices `kidx` and `tidx`. Refuse a key that overlaps its own targets, and an infinite exposure in the columns `kidx` and `tidx`.
 4. Take the target columns of `Ms` as the design `X`.
 5. For each factor `k` in `kidx`, build the regression weights `W` with [`neutralisation_weights`](@ref).
 6. Regress the exposure of `k` across the assets on `X` with `cre` under `W`, and take the residual `csr.eps`.
 7. Transform the residual with `ct` under `W`, and write it over the exposure of `k`.

# Arguments

  - `Ms::AbstractArray{<:Real, 3}`: Exposure history, `observations × assets × factors`. The function changes it in place.
  - `neutralise`: Pairs of `key => targets`. The key and each target is a factor name or a Factor Family label, as a `String` or a `Symbol`, and the targets are one name or a vector of names. A family key neutralises each of its members independently against the same targets.
  - `cre`: Cross-sectional regression estimator of the fit.
  - `bw::MatNum`: Benchmark weight history, `observations × assets`.
  - `nf::VecStr`: Names of the raw factor axis, of length `size(Ms, 3)`.
  - `fam::VecStr`: Family label of each raw factor, of length `size(Ms, 3)`.
  - `ct`: Cross-sectional transform that scores each residual.

# Validation

  - `!isempty(Ms)`.
  - `nf` and `fam` are as long as the factor axis of `Ms`.
  - `bw` matches `Ms` on the observation and asset axes.
  - When `neutralise` holds an entry, the element type of `Ms` is not an `Integer`. An `Integer` element type raises an `ArgumentError`, because the function writes a real residual into `Ms`.
  - Every key and every target names a factor or a family.
  - No key overlaps its own targets, because a factor cannot be neutralised against itself.
  - No exposure of a factor that an entry names is infinite. `NaN` marks a missing exposure, and an infinity raises a `DomainError`, because it survives the fit and the transform.
  - The rules of [`cross_sectional_regression`](@ref) and of `ct` hold on the weights and on the residual.

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
    @argcheck(isempty(neutralise) || !(eltype(Ms) <: Integer),
              ArgumentError("a Neutralisation writes a real residual into Ms, and the element type $(eltype(Ms)) of Ms cannot hold it. Convert Ms to a floating-point type"))
    for pr in neutralise
        key = String(first(pr))
        kidx = neutralisation_indices(key, nf, fam)
        tidx = neutralisation_targets(neutralisation_names(last(pr)), nf, fam)
        ovl = intersect(kidx, tidx)
        @argcheck(isempty(ovl),
                  ArgumentError("the Neutralisation key $key names the factors $([nf[i] for i in ovl]), which are also its targets, and a factor cannot be neutralised against itself"))
        for j in Iterators.flatten((kidx, tidx))
            inf = findfirst(isinf, view(Ms, :, :, j))
            @argcheck(isnothing(inf),
                      DomainError(Ms[inf, j],
                                  "the exposure of asset $(inf[2]) to the factor $(nf[j]) at observation $(inf[1]) is infinite. A Neutralisation reads NaN as a missing exposure, and it refuses an infinity, because an infinity survives the fit and the transform. Replace it with NaN or with a finite value"))
        end
        X = Ms[:, :, tidx]
        for k in kidx
            W = neutralisation_weights(Ms, X, bw, k)
            csr = cross_sectional_regression(cre, X, view(Ms, :, :, k), W)
            Ms[:, :, k] = cross_sectional_transform(ct, csr.eps; w = W)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the target names of one Neutralisation entry.

The right of a Pair is one name or a vector of names. A name is a `String` or a `Symbol`, as the key is, and every form returns a vector of `String` names.

# Arguments

  - `targets`: The right of the Pair.

# Returns

  - `names::Vector{String}`: The target names.

# Related

  - [`neutralise_exposures!`](@ref)
"""
function neutralisation_names(targets::Union{AbstractString, Symbol})::Vector{String}
    return [String(targets)]
end
function neutralisation_names(targets::AbstractVector{<:Union{AbstractString, Symbol}})::Vector{String}
    return String.(targets)
end
"""
    neutralisation_weights(y::MatNum, X::AbstractArray{<:Real, 3}, bw::MatNum)
    neutralisation_weights(Ms::AbstractArray{<:Real, 3}, X::AbstractArray{<:Real, 3},
                           bw::MatNum, k::Integer)

Return the regression weights of one Neutralisation.

The weight of an asset is zero where its response or one of its target exposures is not finite, because a cross-sectional fit refuses a non-finite entry at a positive weight. The weight is also zero where the base weight is not finite or not positive.

The response is one Factor Exposure in [`neutralise_exposures!`](@ref), and one Descriptor score in [`neutralise_scores!`](@ref). The four-argument method takes the exposure by its raw factor index and calls the three-argument method, so the two Neutralisations use one rule.

# Arguments

  - `y::MatNum`: The response that the Neutralisation fits, `observations × assets`.
  - `Ms::AbstractArray{<:Real, 3}`: Exposure history, `observations × assets × factors`.
  - `X::AbstractArray{<:Real, 3}`: Target exposures, `observations × assets × targets`.
  - `bw::MatNum`: Base weight history, `observations × assets`. It holds the benchmark weights in a Neutralisation of Factor Exposures, and the estimation mask in a Neutralisation of Descriptor scores.
  - `k::Integer`: Raw index of the factor that the Neutralisation fits.

# Returns

  - `W::Matrix{<:Real}`: The regression weights, `observations × assets`.

# Related

  - [`neutralise_exposures!`](@ref)
  - [`neutralise_scores!`](@ref)
  - [`cross_sectional_design_mask`](@ref)
"""
function neutralisation_weights(y::MatNum, X::AbstractArray{<:Real, 3}, bw::MatNum)
    Tf = promote_type(real(eltype(y)), real(eltype(bw)))
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
