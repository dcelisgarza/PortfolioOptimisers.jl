"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collect one list-valued argument of a [`CrossSectionalFactorPrior`](@ref) into a vector of Pairs.

Three fields of the prior take the same form: the factors, the Neutralisation and the constrained Factor Families. Each takes Pairs in the order the caller wrote them, or any `AbstractDict`. A dictionary states no order, so the collected order is the one it iterates in, and a caller who needs a stated order writes Pairs.

# Arguments

  - `x`: The Pairs or the dictionary.
  - `sym`: Name of the field, for the messages.

# Validation

  - `x` is not empty. Raises an [`IsEmptyError`](@ref).
  - No key repeats. Raises an `ArgumentError`.

# Returns

  - `pr::Vector{<:Pair}`: The collected Pairs, each with a `String` key.

# Examples

```jldoctest
julia> PortfolioOptimisers.cross_sectional_prior_pairs([\"style\" => \"industry\"], :neutralise)
1-element Vector{Pair{String, String}}:
 "style" => "industry"
```

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`Dict_VecPair`](@ref)
"""
function cross_sectional_prior_pairs(x::Dict_VecPair, sym::Sym_Str)
    pr = [String(first(p)) => last(p) for p in x]
    @argcheck(!isempty(pr), IsEmptyError("$sym cannot be empty"))
    ks = [first(p) for p in pr]
    @argcheck(allunique(ks), ArgumentError("$sym must not repeat a key, got $ks"))
    return pr
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the carrier the Exposure Estimators of a Cross-Sectional Factor Prior are fitted on.

An Exposure Estimator weights its cross-sectional transforms by a benchmark-weight Panel Field it names. The prior computes those weights itself from the market capitalisation, so it writes them onto a copy of the Asset Panel before it builds any Factor Exposure. A field of that name already on the panel is replaced, so the prior's own weights are the ones every member reads.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: Name of the benchmark-weight Panel Field.
  - `W`: The benchmark weights, `observations × assets`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`NumericPanelField`](@ref) and of [`AssetPanel`](@ref).

# Returns

  - `rd::ReturnsResult`: The carrier, with the benchmark weights on its Asset Panel.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`exposure_benchmark_weights`](@ref)
  - [`cross_sectional_cap_weights`](@ref)
"""
function cross_sectional_benchmark_carrier(rd::ReturnsResult, name::AbstractString,
                                           W::MatNum)::ReturnsResult
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Cross-Sectional Factor Prior reads its Factor Exposures off an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl` that asset_panel returns."))
    pf = Any[f for f in pnl.pf if f.name != name]
    push!(pf, NumericPanelField(; name = name, vals = W))
    return ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, nb = rd.nb, B = rd.B,
                         ts = rd.ts, iv = rd.iv, ivpa = rd.ivpa,
                         pnl = AssetPanel(; pf = identity.(pf), amsk = pnl.amsk,
                                          emsk = pnl.emsk))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the order in which a factor list is computed, and the source index of each factor.

A [`DerivedExposure`](@ref) is computed from the Factor Exposure of another factor of the same list, so the list is not always computed in the order it was written. The order this returns puts every source before the factor derived from it.

# Algorithm

 1. Resolve the source name of every [`DerivedExposure`](@ref) to a position in the list.
 2. Sweep the list, appending every factor whose source is already computed, until every factor is placed.
 3. Refuse a sweep that places nothing, because the factors left over depend on each other.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.

# Validation

  - Every [`DerivedExposure`](@ref) names a factor of the list. Raises an `ArgumentError`.
  - No cycle of derived Factor Exposures. Raises an `ArgumentError`.

# Returns

  - `ord::Vector{Int}`: The positions of `factors`, in the order they are computed.
  - `src::Vector{Int}`: The position of each factor's source, and `0` when it has none.

# Related

  - [`DerivedExposure`](@ref)
  - [`cross_sectional_exposure_history`](@ref)
"""
function cross_sectional_exposure_order(factors::AbstractVector{<:Pair})
    n = length(factors)
    nm = [String(first(p)) for p in factors]
    src = zeros(Int, n)
    for i in 1:n
        xe = last(factors[i])
        if isa(xe, DerivedExposure)
            j = something(findfirst(isequal(String(xe.source)), nm), 0)
            @argcheck(j > 0,
                      ArgumentError("the derived Factor Exposure $(nm[i]) is derived from \"$(xe.source)\", which is not one of the factors $nm"))
            @argcheck(j != i,
                      ArgumentError("the derived Factor Exposure $(nm[i]) is derived from itself"))
            src[i] = j
        end
    end
    ord = Int[]
    done = falses(n)
    while length(ord) < n
        moved = false
        for i in 1:n
            if done[i] || (src[i] > 0 && !done[src[i]])
                continue
            end
            push!(ord, i)
            done[i] = true
            moved = true
        end
        @argcheck(moved,
                  ArgumentError("the factors $(nm[.!done]) are derived Factor Exposures that depend on each other, so no order computes a source before the factor derived from it"))
    end
    return ord, src
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the number of columns each factor of a list contributes to the factor axis.

A one-hot member contributes one column per level of its categorical Panel Field, and every other member contributes one.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - $(arg_dict[:rd]) It carries the Asset Panel the one-hot levels are read from.

# Returns

  - `wid::Vector{Int}`: The column count of each factor, in the order of `factors`.

# Related

  - [`exposure_axis_names`](@ref)
  - [`cross_sectional_exposure_history`](@ref)
"""
function cross_sectional_exposure_widths(factors::AbstractVector{<:Pair},
                                         rd::ReturnsResult)::Vector{Int}
    wid = zeros(Int, length(factors))
    for i in eachindex(factors)
        nfi, _ = exposure_axis_names(String(first(factors[i])), last(factors[i]), rd)
        wid[i] = length(nfi)
    end
    return wid
end
"""
    cross_sectional_exposure_write!(Ms::AbstractArray{<:Real, 3}, A::MatNum, c::Integer,
                                    w::Integer, nm::AbstractString) -> nothing
    cross_sectional_exposure_write!(Ms::AbstractArray{<:Real, 3}, A::Arr3Num, c::Integer,
                                    w::Integer, nm::AbstractString) -> nothing

Write one factor's Factor Exposure into the exposure history, in place.

A member answers a matrix when it contributes one factor and a three-dimensional array when it contributes several, so the two shapes are two methods rather than a branch.

# Arguments

  - `Ms`: The exposure history, `observations × assets × factors`, changed in place.
  - `A`: The Factor Exposure of one member.
  - `c`: First column of the member on the factor axis.
  - `w`: Number of columns the member contributes.
  - `nm`: Name of the factor, for the messages.

# Validation

  - `A` matches the observation and asset axes of `Ms`, and its factor axis is `w`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`. `Ms` carries the exposure.

# Related

  - [`factor_exposure`](@ref)
  - [`cross_sectional_exposure_history`](@ref)
"""
function cross_sectional_exposure_write!(Ms::AbstractArray{<:Real, 3}, A::MatNum,
                                         c::Integer, w::Integer,
                                         nm::AbstractString)::Nothing
    @argcheck(isone(w),
              DimensionMismatch("the Factor Exposure $nm answers one factor, but the factor axis expects $w columns for it"))
    @argcheck(size(A) == (size(Ms, 1), size(Ms, 2)),
              DimensionMismatch("the Factor Exposure $nm is observations × assets, got $(size(A)) against $((size(Ms, 1), size(Ms, 2))))"))
    Ms[:, :, c] = A
    return nothing
end
function cross_sectional_exposure_write!(Ms::AbstractArray{<:Real, 3}, A::Arr3Num,
                                         c::Integer, w::Integer,
                                         nm::AbstractString)::Nothing
    @argcheck(size(A) == (size(Ms, 1), size(Ms, 2), w),
              DimensionMismatch("the Factor Exposure $nm is observations × assets × factors, got $(size(A)) against $((size(Ms, 1), size(Ms, 2), w))"))
    Ms[:, :, c:(c + w - 1)] = A
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the exposure history of a factor list, and the factor axis it is written on.

# Algorithm

 1. Read the factor axis with [`cross_sectional_factor_axis`](@ref), and the column count of each factor with [`cross_sectional_exposure_widths`](@ref).
 2. Take the computation order with [`cross_sectional_exposure_order`](@ref).
 3. Compute each member in that order. A [`DerivedExposure`](@ref) is handed the exposure of its source, which the order has already written.
 4. Write each answer into the history with [`cross_sectional_exposure_write!`](@ref).

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - $(arg_dict[:rd]) It carries the Asset Panel every member reads.

# Validation

  - The rules of [`cross_sectional_factor_axis`](@ref) and of [`cross_sectional_exposure_order`](@ref).
  - The source of a [`DerivedExposure`](@ref) contributes one column. Raises an `ArgumentError`.

# Returns

  - `Ms::Array{<:Real, 3}`: The exposure history, `observations × assets × factors`.
  - `nf::Vector{String}`: Name of each factor.
  - `fam::Vector{String}`: Family label of each factor.

# Related

  - [`factor_exposure`](@ref)
  - [`cross_sectional_factor_axis`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_exposure_history(factors::AbstractVector{<:Pair},
                                          rd::ReturnsResult)
    (; nf, fam) = cross_sectional_factor_axis(factors, rd)
    wid = cross_sectional_exposure_widths(factors, rd)
    ord, src = cross_sectional_exposure_order(factors)
    col = cumsum(vcat(1, wid[1:(end - 1)]))
    X = rd.X
    Tf = promote_type(float(real(eltype(X))), Float64)
    Ms = Array{Tf, 3}(undef, size(X, 1), size(X, 2), length(nf))
    for i in ord
        nm = String(first(factors[i]))
        xe = last(factors[i])
        if src[i] > 0
            @argcheck(isone(wid[src[i]]),
                      ArgumentError("the derived Factor Exposure $nm reads one Factor Exposure, and its source \"$(nf[col[src[i]]])\" contributes $(wid[src[i]]) of them"))
            A = factor_exposure(xe, rd, Ms[:, :, col[src[i]]])
            cross_sectional_exposure_write!(Ms, A, col[i], wid[i], nm)
        else
            cross_sectional_exposure_write!(Ms, factor_exposure(xe, rd), col[i], wid[i], nm)
        end
    end
    return (; Ms = Ms, nf = nf, fam = fam)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether one asset carries a finite Factor Exposure to every factor at one observation.

# Arguments

  - `Ms`: The exposure history, `observations × assets × factors`.
  - `t`: The observation.
  - `i`: The asset.

# Returns

  - `ans::Bool`: Whether every exposure of the pair is finite.

# Related

  - [`cross_sectional_warmup`](@ref)
  - [`cross_sectional_eligible`](@ref)
"""
function cross_sectional_exposures_finite(Ms::Arr3Num, t::Integer, i::Integer)::Bool
    for k in axes(Ms, 3)
        if !isfinite(Ms[t, i, k])
            return false
        end
    end
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the number of leading observations a Cross-Sectional Factor Prior discards.

A Descriptor warms up, so the first observations of an exposure history carry no usable asset. An observation is cold when no asset of the estimation universe carries both a finite return and a finite Factor Exposure to every factor. The prior fits from the first observation that is not cold.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `Ms`: The exposure history, `observations × assets × factors`.
  - `emsk`: The estimation mask, `observations × assets`.

# Validation

  - At least one observation is not cold. Raises an `ArgumentError`.

# Returns

  - `n::Int`: The count of leading cold observations.

# Related

  - [`cross_sectional_exposure_history`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_warmup(X::MatNum, Ms::Arr3Num, emsk::AbstractMatrix{Bool})::Int
    for t in axes(Ms, 1)
        for i in axes(Ms, 2)
            if emsk[t, i] && isfinite(X[t, i]) && cross_sectional_exposures_finite(Ms, t, i)
                return t - 1
            end
        end
    end
    return throw(ArgumentError("no observation of this Asset Panel carries an asset of the estimation universe with both a finite return and a finite Factor Exposure to every factor, so the whole history is Descriptor warm-up. Give more observations, or shorten the warm-up of the Descriptors."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the eligibility mask of a cross-sectional fit.

An asset enters the fit of an observation when it is in the estimation universe, its return is finite, and its lagged Factor Exposure is finite for every factor. The mask is what the weight policy writes its weights on, so an ineligible pair carries a weight of zero and the fit never sees it.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `Ms`: The lagged exposure history, `observations × assets × factors`, aligned with `X`.
  - `emsk`: The estimation mask, `observations × assets`, aligned with `X`.

# Validation

  - `Ms` matches `X` on the observation and asset axes, and `emsk` matches `X`. Raises a `DimensionMismatch`.

# Returns

  - `msk::BitMatrix`: The eligibility mask, `observations × assets`.

# Related

  - [`cs_weights_initial`](@ref)
  - [`cross_sectional_design_mask`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_eligible(X::MatNum, Ms::Arr3Num,
                                  emsk::AbstractMatrix{Bool})::BitMatrix
    @argcheck(size(Ms, 1) == size(X, 1) && size(Ms, 2) == size(X, 2),
              DimensionMismatch("Ms ($(size(Ms, 1))×$(size(Ms, 2))) must match X ($(size(X, 1))×$(size(X, 2))) on the observation and asset axes"))
    @argcheck(size(emsk) == size(X),
              DimensionMismatch("emsk ($(size(emsk, 1))×$(size(emsk, 2))) must match X ($(size(X, 1))×$(size(X, 2)))"))
    msk = falses(size(X))
    for t in axes(X, 1), i in axes(X, 2)
        msk[t, i] = emsk[t, i] &&
                    isfinite(X[t, i]) &&
                    cross_sectional_exposures_finite(Ms, t, i)
    end
    return msk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a fit whose observations do not carry enough eligible assets.

A cross-sectional fit of `K` factors needs more assets than factors, and a fit that has barely more is noise. The prior refuses the whole fit rather than dropping the thin observations, because a factor-return series with a gap in it is not a series.

# Arguments

  - `msk`: The eligibility mask, `observations × assets`.
  - `minra`: The smallest eligible asset count an observation may carry.

# Validation

  - Every observation carries at least `minra` eligible assets. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_eligible`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function assert_cross_sectional_coverage(msk::AbstractMatrix{Bool}, minra::Integer)::Nothing
    n = vec(sum(msk; dims = 2))
    bad = findall(x -> x < minra, n)
    @argcheck(isempty(bad),
              ArgumentError("$(length(bad)) observation(s) carry fewer than minra = $minra eligible assets, the fewest being $(minimum(view(n, bad))) at observation $(bad[argmin(view(n, bad))]). Widen the coverage of the Descriptors, lower the min_coverage of the Factor Exposures, or lower minra."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the idiosyncratic covariance of the latest observation.

# Algorithm

 1. A threshold of zero answers the latest idiosyncratic variances, so the block carries a vector and the asset covariance takes a diagonal.
 2. Otherwise, take the standardised idiosyncratic returns, write a zero at every cell that is still not finite, and estimate their covariance with `ce`. A cell is left non-finite only where the asset is inactive, and a zero is the neutral value of a standardised series, so such a pair pulls the correlation toward the threshold it is dropped by.
 3. Convert the covariance to a correlation.
 4. Zero every correlation whose magnitude does not exceed the threshold, and keep the diagonal. A correlation that is not finite is zeroed by the same step.
 5. Rescale the correlation by the latest idiosyncratic volatilities, and make the block of assets with a finite variance positive definite.

# Arguments

  - `th`: The correlation threshold.
  - `ce`: Covariance estimator of the standardised idiosyncratic returns.
  - `pdm`: Positive definite matrix estimator, or `nothing`.
  - `S`: Standardised idiosyncratic returns, `observations × assets`.
  - `ev`: The latest idiosyncratic variances, one per asset.

# Returns

  - `esigma::VecNum`: The latest idiosyncratic variances, when `th` is zero.
  - `esigma::MatNum`: The idiosyncratic covariance, when `th` is positive.

# Related

  - [`cross_sectional_standardised_residuals`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`posdef!`](@ref)
"""
function cross_sectional_idiosyncratic_covariance(th::Real,
                                                  ce::StatsBase.CovarianceEstimator,
                                                  pdm::Option{<:AbstractPosdefEstimator},
                                                  S::MatNum, ev::VecNum)
    if iszero(th)
        return ev
    end
    Z = Matrix{float(real(eltype(S)))}(S)
    for k in CartesianIndices(Z)
        if !isfinite(Z[k])
            Z[k] = zero(eltype(Z))
        end
    end
    C = Statistics.cov(ce, Z; dims = 1)
    s = sqrt.(LinearAlgebra.diag(C))
    R = StatsBase.cov2cor(Matrix(C), s)
    for k in CartesianIndices(R)
        if k[1] != k[2] && !(abs(R[k]) > th)
            R[k] = zero(eltype(R))
        end
    end
    for i in axes(R, 1)
        R[i, i] = one(eltype(R))
    end
    se = sqrt.(ev)
    D = R .* se .* transpose(se)
    idx = findall(isfinite, ev)
    if !isempty(idx)
        B = D[idx, idx]
        posdef!(pdm, B)
        D[idx, idx] = B
    end
    return D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the average of the finite entries of one observation of a matrix.

The cross-section of standardised idiosyncratic returns is the one caller, and it needs the average to stand in for a cell it could not standardise. An observation with no finite entry answers zero, which is the average of a standardised series.

# Arguments

  - `S`: The matrix, `observations × assets`.
  - `t`: The observation.

# Returns

  - `avg::Real`: The average of the finite entries of observation `t`, or zero.

# Related

  - [`cross_sectional_standardised_residuals`](@ref)
"""
function cross_sectional_finite_mean(S::MatNum, t::Integer)
    num = zero(eltype(S))
    cnt = 0
    for i in axes(S, 2)
        if isfinite(S[t, i])
            num += S[t, i]
            cnt += 1
        end
    end
    return iszero(cnt) ? zero(eltype(S)) : num / cnt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the standardised idiosyncratic returns a scenario set is rebuilt from.

Each idiosyncratic return is divided by its own contemporaneous idiosyncratic volatility, so the history is on one scale and the latest volatilities can rescale it. An active pair whose standardised return is not finite takes the average standardised return of its own observation, so a sparse history does not shorten the scenario set. An inactive pair stays `NaN`.

# Arguments

  - `eps`: Idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `amsk`: The active mask, `observations × assets`.

# Returns

  - `S::Matrix{<:Real}`: The standardised idiosyncratic returns, `observations × assets`.

# Related

  - [`cross_sectional_finite_mean`](@ref)
  - [`cross_sectional_scenarios`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_standardised_residuals(eps::MatNum, vs::MatNum,
                                                amsk::AbstractMatrix{Bool})
    Tf = promote_type(float(real(eltype(eps))), float(real(eltype(vs))))
    S = Matrix{Tf}(undef, size(eps))
    for k in CartesianIndices(S)
        S[k] = amsk[k] ? Tf(eps[k]) / sqrt(Tf(vs[k])) : Tf(NaN)
    end
    for t in axes(S, 1)
        avg = cross_sectional_finite_mean(S, t)
        for i in axes(S, 2)
            if amsk[t, i] && !isfinite(S[t, i])
                S[t, i] = avg
            end
        end
    end
    return S
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild the asset return scenarios of a Cross-Sectional Factor Prior.

The factor prior's own scenarios are mapped onto the assets through the latest loadings, and the standardised idiosyncratic returns are rescaled by the latest idiosyncratic volatilities and added to them. A scenario therefore carries the factor risk and the idiosyncratic risk of the latest observation, whatever the risk of the observation it was drawn from was. The two histories may differ in length, so the last rows of the longer one are the ones that pair up.

# Arguments

  - `Fs`: Factor return scenarios on the reduced axis, `scenarios × factors`.
  - `L`: The reduced loadings of the latest observation, `assets × factors`.
  - `S`: The standardised idiosyncratic returns, `observations × assets`.
  - `ev`: The latest idiosyncratic variances, one per asset.

# Validation

  - `L` matches `Fs` on the factor axis, and `S` and `ev` match `L` on the asset axis. Raises a `DimensionMismatch`.

# Returns

  - `Xs::Matrix{<:Real}`: The asset return scenarios, `scenarios × assets`.

# Related

  - [`cross_sectional_standardised_residuals`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_scenarios(Fs::MatNum, L::MatNum, S::MatNum, ev::VecNum)
    @argcheck(size(L, 2) == size(Fs, 2),
              DimensionMismatch("L ($(size(L, 2)) columns) must match Fs ($(size(Fs, 2)) columns)"))
    @argcheck(size(S, 2) == size(L, 1) == length(ev),
              DimensionMismatch("S ($(size(S, 2)) columns), L ($(size(L, 1)) rows) and ev ($(length(ev))) must agree on the asset axis"))
    n = min(size(Fs, 1), size(S, 1))
    Fn = view(Fs, (size(Fs, 1) - n + 1):size(Fs, 1), :)
    Sn = view(S, (size(S, 1) - n + 1):size(S, 1), :)
    return Fn * transpose(L) .+ Sn .* transpose(sqrt.(ev))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the assets a Cross-Sectional Factor Prior can state a finite moment for.

The prior fits on the coverage universe and answers on it, so an asset it cannot state a moment for carries `NaN` rather than leaving the result. Three facts make an asset investable: the Asset Panel activates it at the latest observation, its idiosyncratic variance is finite, and its latest loadings are finite.

# Arguments

  - `amsk`: The active mask of the latest observation, one entry per asset.
  - `L`: The reduced loadings of the latest observation, `assets × factors`.
  - `ev`: The latest idiosyncratic variances, one per asset.

# Returns

  - `idx::Vector{Int}`: The investable assets, in ascending order.

# Related

  - [`investable_mask`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_investable(amsk::AbstractVector{Bool}, L::MatNum,
                                    ev::VecNum)::Vector{Int}
    idx = Int[]
    for i in eachindex(amsk)
        if !(amsk[i] && isfinite(ev[i]))
            continue
        end
        ok = true
        for k in axes(L, 2)
            if !isfinite(L[i, k])
                ok = false
                break
            end
        end
        if ok
            push!(idx, i)
        end
    end
    return idx
end
"""
    cross_sectional_panel_masks(pnl::AssetPanel{<:Any, Nothing, Nothing}) -> Union{}
    cross_sectional_panel_masks(pnl::AssetPanel) -> Tuple

Read the two universe masks a Cross-Sectional Factor Prior fits against.

A static Asset Panel carries no observation axis and no masks, so it states no point-in-time universe. The two shapes are two methods rather than a test, which is what the mask fields being a type parameter is for.

# Arguments

  - `pnl`: The Asset Panel.

# Validation

  - The Asset Panel is time-varying. Raises an `ArgumentError`.

# Returns

  - `amsk::AbstractMatrix{Bool}`: The active mask, `observations × assets`.
  - `emsk::AbstractMatrix{Bool}`: The estimation mask, `observations × assets`.

# Related

  - [`AssetPanel`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_panel_masks(::AssetPanel{<:Any, Nothing, Nothing})
    return throw(ArgumentError("a Cross-Sectional Factor Prior fits a point-in-time panel, so it needs the active mask and the estimation mask of a time-varying Asset Panel, and this Asset Panel is static"))
end
function cross_sectional_panel_masks(pnl::AssetPanel)
    return pnl.amsk, pnl.emsk
end
"""
    cross_sectional_cap_finite!(msk::BitMatrix, mcap::Nothing) -> nothing
    cross_sectional_cap_finite!(msk::BitMatrix, mcap::MatNum) -> nothing

Drop from a mask every pair whose market capitalisation is not finite, in place.

A weight raised from a market capitalisation needs that capitalisation, and the two weight masks of the prior — the benchmark one and the regression one — take the same cut. A power of zero reads no capitalisation, so the `nothing` method leaves the mask as it stands.

# Arguments

  - `msk`: The mask, `observations × assets`, changed in place.
  - `mcap`: The market capitalisation, `observations × assets`, or `nothing`.

# Validation

  - `mcap` matches `msk`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`. `msk` carries the cut.

# Related

  - [`cross_sectional_cap_weights`](@ref)
  - [`cross_sectional_eligible`](@ref)
"""
function cross_sectional_cap_finite!(::BitMatrix, ::Nothing)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a Cross-Sectional Factor Prior reads the market capitalisation.

Two powers raise a weight from it: the benchmark one, which the prior carries, and the regression one, which the weight policy carries. Both at zero make every eligible asset take the same weight, and the prior then needs no market capitalisation and reads no Panel Field for it. Every member of the weight family carries `p`, which is what [`cs_weights_initial`](@ref) already reads.

# Arguments

  - `bp`: The benchmark market-capitalisation power.
  - `alg`: The weight policy of the cross-sectional fit.

# Returns

  - `ans::Bool`: Whether the market capitalisation is read.

# Related

  - [`cross_sectional_cap_weights`](@ref)
  - [`cs_weights_initial`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_needs_market_cap(bp::Real,
                                          alg::AbstractCrossSectionalWeightsAlgorithm)::Bool
    return !iszero(bp) || !iszero(alg.p)
end
"""
    cross_sectional_rows(A::Nothing, r) -> nothing
    cross_sectional_rows(A::MatNum, r) -> MatNum

Take a set of observations out of an optional matrix.

The prior trims every parallel array to the same observations, and the market capitalisation is the one that may be absent. The `nothing` method keeps it absent.

# Arguments

  - `A`: The matrix, `observations × assets`, or `nothing`.
  - `r`: The observations to keep.

# Returns

  - `A::Option{<:MatNum}`: The rows, or `nothing`.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_rows(::Nothing, r)
    return nothing
end
function cross_sectional_rows(A::MatNum, r)
    return A[r, :]
end
"""
    cross_sectional_reduced_loadings(fcb::Nothing, L::MatNum) -> nothing
    cross_sectional_reduced_loadings(fcb::FactorFamilyBasis, L::MatNum) -> MatNum

Return the reduced loadings a [`CrossSectionalFactorModel`](@ref) stores, or `nothing`.

The block holds `L` exactly when it holds a Factor Family Basis, so the two are answered by the same dispatch and cannot disagree.

# Arguments

  - `fcb`: The Factor Family Basis, or `nothing`.
  - `L`: The loadings on the reduced axis.

# Returns

  - `L::Option{<:MatNum}`: The reduced loadings, or `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`FactorFamilyBasis`](@ref)
"""
function cross_sectional_reduced_loadings(::Nothing, ::MatNum)
    return nothing
end
function cross_sectional_reduced_loadings(::FactorFamilyBasis, L::MatNum)
    return L
end
function cross_sectional_cap_finite!(msk::BitMatrix, mcap::MatNum)::Nothing
    @argcheck(size(mcap) == size(msk),
              DimensionMismatch("mcap ($(size(mcap, 1))×$(size(mcap, 2))) must match msk ($(size(msk, 1))×$(size(msk, 2)))"))
    for k in CartesianIndices(msk)
        if msk[k] && !isfinite(mcap[k])
            msk[k] = false
        end
    end
    return nothing
end
"""
    cross_sectional_neutralise!(neutralise::Nothing, args...) -> nothing
    cross_sectional_neutralise!(neutralise::AbstractVector{<:Pair}, Ms, cre, bw, nf, fam)
        -> nothing

Run the Neutralisation of a Cross-Sectional Factor Prior, in place.

A prior that states no Neutralisation runs none, which is the `nothing` method.

# Arguments

  - `neutralise`: Pairs of `key => targets`, or `nothing`.
  - `Ms`: The exposure history, `observations × assets × factors`, changed in place.
  - `cre`: The prior's own Cross-Sectional Regression Estimator.
  - `bw`: The benchmark weights, `observations × assets`.
  - `nf`: Name of each factor.
  - `fam`: Family label of each factor.

# Validation

  - The rules of [`neutralise_exposures!`](@ref).

# Returns

  - `nothing`. `Ms` carries the neutralised exposures.

# Related

  - [`neutralise_exposures!`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_neutralise!(::Nothing, args...)::Nothing
    return nothing
end
function cross_sectional_neutralise!(neutralise::AbstractVector{<:Pair},
                                     Ms::AbstractArray{<:Real, 3},
                                     cre::AbstractCrossSectionalRegressionEstimator,
                                     bw::MatNum, nf::VecStr, fam::VecStr)::Nothing
    return neutralise_exposures!(Ms, neutralise, cre, bw, nf, fam)
end
"""
    cross_sectional_family_basis(families::Nothing, Ms, bw, nf, fam) -> NamedTuple
    cross_sectional_family_basis(families::AbstractVector{<:Pair}, Ms, bw, nf, fam)
        -> NamedTuple

Build the Factor Family Basis of a Cross-Sectional Factor Prior, and reduce the factor axis through it.

A prior that constrains no Factor Family fits on the raw axis, which is the `nothing` method: the basis is absent and the three answers are the inputs.

# Arguments

  - `families`: Pairs of `family label => dropped member`, or `nothing`.
  - `Ms`: The exposure history on the raw axis, `observations × assets × factors`.
  - `bw`: The benchmark weights, `observations × assets`.
  - `nf`: Name of each raw factor.
  - `fam`: Family label of each raw factor.

# Validation

  - The rules of [`factor_family_basis`](@ref).

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The basis, or `nothing`.
  - `Ms::Arr3Num`: The exposure history on the reduced axis.
  - `nf::VecStr`: Name of each reduced factor.
  - `fam::VecStr`: Family label of each reduced factor.

# Related

  - [`factor_family_basis`](@ref)
  - [`reduce_exposures`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_family_basis(::Nothing, Ms::Arr3Num, ::MatNum, nf::VecStr,
                                      fam::VecStr)
    return (; fcb = nothing, Ms = Ms, nf = nf, fam = fam)
end
function cross_sectional_family_basis(families::AbstractVector{<:Pair}, Ms::Arr3Num,
                                      bw::MatNum, nf::VecStr, fam::VecStr)
    fcb = factor_family_basis(families, Ms, bw, nf, fam)
    return (; fcb = fcb, Ms = reduce_exposures(fcb, Ms), nf = reduce_factor_names(fcb, nf),
            fam = reduce_factor_names(fcb, fam))
end
"""
    cross_sectional_basis_now(fcb::Nothing, r) -> nothing
    cross_sectional_basis_now(fcb::FactorFamilyBasis, r) -> FactorFamilyBasis

Slice a Factor Family Basis onto the observations a fit ran on.

The basis is built over the whole post-warm-up history, and the exposure lag then shortens the fit. The block stores the slice the fit ran on, so its observation axis is the one every other history on the block carries.

# Arguments

  - `fcb`: The Factor Family Basis, or `nothing`.
  - `r`: The fitted observations, as an index into the post-warm-up axis.

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The basis of the fitted observations, or `nothing`.

# Related

  - [`factor_basis_slice`](@ref)
  - [`cross_sectional_expand`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_basis_now(::Nothing, r)
    return nothing
end
function cross_sectional_basis_now(fcb::FactorFamilyBasis, r)
    return factor_basis_slice(fcb, r)
end
"""
    cross_sectional_expand(fcb::Nothing, r, lag::Integer, f, mu, sigma) -> NamedTuple
    cross_sectional_expand(fcb::FactorFamilyBasis, r, lag::Integer, f, mu, sigma)
        -> NamedTuple

Expand a factor distribution from the reduced axis onto the raw one.

The nested factor prior of a [`LowOrderPrior`](@ref) sits on the raw axis, so a constraint written in a dropped factor's name still resolves. A prior that constrains no Factor Family fits on the raw axis already, which is the `nothing` method.

The fit of observation `t` regresses the returns of `t` on the exposures of `t - lag`, so its coefficients are coordinates in the basis of `t - lag`. The realised factor returns therefore expand with the lagged ratios, and the moments, which describe the next observation, expand with the current ones. The verb slices the basis itself rather than taking the two slices, because two arguments of that kind admit a mixed pair that no method answers.

# Arguments

  - `fcb`: The Factor Family Basis over the post-warm-up history, or `nothing`.
  - `r`: The fitted observations, as an index into the post-warm-up axis.
  - `lag`: The exposure lag.
  - `f`: Realised factor returns on the reduced axis, `observations × factors`.
  - `mu`: Expected factor returns on the reduced axis.
  - `sigma`: Factor covariance on the reduced axis.

# Validation

  - The rules of [`expand_factor_returns`](@ref), [`expand_factor_mu`](@ref) and [`expand_factor_covariance`](@ref).

# Returns

  - `f::MatNum`: The realised factor returns on the raw axis.
  - `mu::VecNum`: The expected factor returns on the raw axis.
  - `sigma::MatNum`: The factor covariance on the raw axis.

# Related

  - [`cross_sectional_basis_now`](@ref)
  - [`expand_factor_returns`](@ref)
  - [`expand_factor_mu`](@ref)
  - [`expand_factor_covariance`](@ref)
"""
function cross_sectional_expand(::Nothing, r, ::Integer, f::MatNum, mu::VecNum,
                                sigma::MatNum)
    return (; f = f, mu = mu, sigma = sigma)
end
function cross_sectional_expand(fcb::FactorFamilyBasis, r, lag::Integer, f::MatNum,
                                mu::VecNum, sigma::MatNum)
    now = factor_basis_slice(fcb, r)
    return (; f = expand_factor_returns(factor_basis_slice(fcb, r .- lag), f),
            mu = expand_factor_mu(now, mu), sigma = expand_factor_covariance(now, sigma))
end
"""
    cross_sectional_residual_block(esigma::VecNum, idx) -> NamedTuple
    cross_sectional_residual_block(esigma::MatNum, idx) -> NamedTuple

Return the idiosyncratic block of the asset covariance and a square root of it.

A diagonal block and a full block are the two shapes the idiosyncratic covariance takes, so they are two methods. Both answer the block over the investable assets alone, because a non-investable asset carries no finite variance to factorise.

# Arguments

  - `esigma`: The idiosyncratic variances, or the idiosyncratic covariance.
  - `idx`: The investable assets.

# Validation

  - A full block restricted to `idx` factorises. Raises a `PosDefException`.

# Returns

  - `D::MatNum`: The block the asset covariance adds.
  - `R::MatNum`: A square root of the block, `investable assets × investable assets`.

# Related

  - [`cross_sectional_lift`](@ref)
  - [`cross_sectional_idiosyncratic_covariance`](@ref)
"""
function cross_sectional_residual_block(esigma::VecNum, idx::AbstractVector{<:Integer})
    d = esigma[idx]
    return (; D = LinearAlgebra.diagm(d), R = LinearAlgebra.diagm(sqrt.(d)))
end
function cross_sectional_residual_block(esigma::MatNum, idx::AbstractVector{<:Integer})
    D = esigma[idx, idx]
    return (; D = D, R = Matrix(LinearAlgebra.cholesky(D).L))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lift a factor distribution onto the assets of a Cross-Sectional Factor Prior.

# Algorithm

 1. Take the loadings of the investable assets, and project the factor mean and the factor covariance through them.
 2. Process the systematic covariance with `mp`, as [`factor_lift`](@ref) does.
 3. Add the idiosyncratic block, and re-condition the sum.
 4. Build the low-rank square root `[L·chol(F).L | R]`, whose trailing block is a square root of the idiosyncratic block.
 5. Scatter the three answers into the full asset universe, writing `NaN` at every asset the prior states no moment for.

# Arguments

  - `mp`: Matrix processing estimator.
  - `L`: The reduced loadings of the latest observation, `assets × factors`.
  - `f_mu`: Expected factor returns on the reduced axis.
  - `f_sigma`: Factor covariance on the reduced axis.
  - `esigma`: The idiosyncratic variances, or the idiosyncratic covariance.
  - `idx`: The investable assets.
  - `Xs`: The asset return scenarios, `scenarios × assets`, which the processing reads.

# Validation

  - `L`, `f_mu` and `f_sigma` agree on the factor axis. Raises a `DimensionMismatch`.

# Returns

  - `mu::Vector{<:Real}`: Expected asset returns, `NaN` at a non-investable asset.
  - `sigma::Matrix{<:Real}`: Asset covariance, `NaN` in the row and the column of a non-investable asset.
  - `chol::Matrix{<:Real}`: The low-rank square root, `NaN` in the column of a non-investable asset.

# Related

  - [`factor_lift`](@ref)
  - [`cross_sectional_residual_block`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_lift(mp::AbstractMatrixProcessingEstimator, L::MatNum,
                              f_mu::VecNum, f_sigma::MatNum, esigma::VecNum_MatNum,
                              idx::AbstractVector{<:Integer}, Xs::MatNum; kwargs...)
    @argcheck(size(L, 2) == length(f_mu) == size(f_sigma, 1),
              DimensionMismatch("L ($(size(L, 2)) columns), f_mu ($(length(f_mu))) and f_sigma ($(size(f_sigma, 1)) rows) must agree on the factor axis"))
    Li = L[idx, :]
    (; D, R) = cross_sectional_residual_block(esigma, idx)
    mui = Li * f_mu
    si = Li * f_sigma * transpose(Li)
    matrix_processing!(mp, si, Xs[:, idx]; kwargs...)
    si .+= D
    posdef!(mp.pdm, si)
    ci = hcat(Li * Matrix(LinearAlgebra.cholesky(f_sigma).L), R)
    N = size(L, 1)
    Tf = float(real(eltype(si)))
    mu = fill(Tf(NaN), N)
    sigma = fill(Tf(NaN), N, N)
    chol = fill(Tf(NaN), size(ci, 2), N)
    mu[idx] = mui
    sigma[idx, idx] = si
    chol[:, idx] = transpose(ci)
    return (; mu = mu, sigma = sigma, chol = chol)
end
