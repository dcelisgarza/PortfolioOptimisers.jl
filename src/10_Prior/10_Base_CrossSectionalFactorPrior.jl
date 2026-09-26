"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collect one list-valued argument of a [`CrossSectionalFactorPrior`](@ref) into a vector of Pairs.

The factors, the Neutralisation and the constrained Factor Families of the prior take the same form. Each takes Pairs in the order the caller writes them, or any `AbstractDict`. A dictionary states no order, so the collected order is its order of iteration. A caller who needs a stated order writes Pairs.

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

An Exposure Estimator weights its cross-sectional transforms by a benchmark-weight Panel Field that it names. The prior computes those weights from the market capitalisation, and writes them onto a copy of the Asset Panel before it builds any Factor Exposure. The copy replaces a field of that name, so every member reads the weights of the prior.

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

 1. Resolve the source name of every [`DerivedExposure`](@ref) to a position in the list, giving `src`.
 2. Pass over the list, and append to `ord` every factor that has no source or whose source is already in `ord`. Repeat until `ord` holds every factor.
 3. Refuse a pass that appends nothing, because the factors left over depend on each other.

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
    cross_sectional_exposure_write!(Ms::AbstractArray{<:Number, 3}, A::MatNum, c::Integer,
                                    w::Integer, nm::AbstractString) -> nothing
    cross_sectional_exposure_write!(Ms::AbstractArray{<:Number, 3}, A::Arr3Num, c::Integer,
                                    w::Integer, nm::AbstractString) -> nothing

Write one factor's Factor Exposure into the exposure history, in place.

A member gives a matrix when it contributes one factor, and a three-dimensional array when it contributes several. Each shape has its own method.

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
function cross_sectional_exposure_write!(Ms::AbstractArray{<:Number, 3}, A::MatNum,
                                         c::Integer, w::Integer,
                                         nm::AbstractString)::Nothing
    @argcheck(isone(w),
              DimensionMismatch("the Factor Exposure $nm answers one factor, but the factor axis expects $w columns for it"))
    @argcheck(size(A) == (size(Ms, 1), size(Ms, 2)),
              DimensionMismatch("the Factor Exposure $nm is observations × assets, got $(size(A)) against $((size(Ms, 1), size(Ms, 2))))"))
    Ms[:, :, c] = A
    return nothing
end
function cross_sectional_exposure_write!(Ms::AbstractArray{<:Number, 3}, A::Arr3Num,
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
 3. Compute each member in that order. Give a [`DerivedExposure`](@ref) the exposure of its source, which the order has already written.
 4. Write each Factor Exposure into `Ms` with [`cross_sectional_exposure_write!`](@ref). `Ms` takes the element type `float_if_integer(real(eltype(X)))`, so integer returns give a float history that can hold a fractional exposure and the `NaN` of an inactive cell.

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
    col = cumsum(vcat(1, @view(wid[1:(end - 1)])))
    X = rd.X
    Tf = float_if_integer(real(eltype(X)))
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

# Mathematical definition

```math
\\begin{align}
n &= \\min\\left\\{t : \\exists\\, i,\\ e_{ti} = 1,\\ x_{t,\\,i} \\in \\mathbb{R},\\ B_{tik} \\in \\mathbb{R} \\ \\forall k \\in \\{1, \\ldots, K\\}\\right\\} - 1\\,.
\\end{align}
```

Where:

  - ``n``: Count of the leading cold observations.
  - $(math_dict[:e_ti_pnl])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:B_tik_cs])
  - $(math_dict[:K])

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

An asset enters the fit of an observation when it is in the estimation universe, its return is finite, and its lagged Factor Exposure is finite for every factor. The weight policy writes its weights on this mask, so an ineligible pair takes a weight of zero and does not enter the fit.

# Mathematical definition

```math
\\begin{align}
m_{ti} &= e_{ti} \\, \\mathbb{1}\\left[x_{t,\\,i} \\in \\mathbb{R}\\right] \\prod_{k=1}^{K} \\mathbb{1}\\left[\\tilde{B}_{tik} \\in \\mathbb{R}\\right]\\,.
\\end{align}
```

Where:

  - ``m_{ti}``: Eligibility of asset ``i`` at observation ``t``.
  - ``\\tilde{B}_{tik}``: Lagged Factor Exposure of asset ``i`` to factor ``k`` that the fit reads against observation ``t``.
  - $(math_dict[:e_ti_pnl])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:K])

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

A cross-sectional fit of `K` factors needs more assets than factors, and a fit with few more assets than factors gives imprecise factor returns. The prior refuses the whole fit and does not drop the thin observations, because a dropped observation leaves a gap in the factor-return series.

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

Refuse a fit whose factor prior states a non-finite factor moment.

The warm-up of the Descriptors and the warm-up of the factor prior add up. The Descriptors fix the first observation of the factor-return history, and the factor prior then warms up over that history. A window that covers the first warm-up can be too short for the second. A factor prior that gives a `NaN` and does not raise then makes the whole prior non-finite, and a later factorisation fails with no name. So the fit checks the moments that the factor prior gives, and refuses a non-finite one with a message that names the cause.

# Arguments

  - `mu`: The factor means the factor prior stated.
  - `sigma`: The factor covariance the factor prior stated.
  - `n`: The count of fitted observations the factor prior read.

# Validation

  - Every factor mean and every entry of the factor covariance is finite. Raises an [`IsNonFiniteError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_warmup`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function assert_cross_sectional_factor_moments(mu::VecNum, sigma::MatNum,
                                               n::Integer)::Nothing
    nfm = count(!isfinite, mu)
    nfs = count(!isfinite, sigma)
    @argcheck(iszero(nfm) && iszero(nfs),
              IsNonFiniteError("the factor prior read the $n observation(s) left after the Descriptor warm-up and the exposure lag, and stated $nfm non-finite factor mean(s) and $nfs non-finite factor covariance entr(ies). A Descriptor's warm-up and the factor prior's own warm-up are cumulative, so a window must cover both. Give more observations, shorten the warm-up of the Descriptors, or give pe a factor prior that estimates from fewer observations."))
    return nothing
end
"""
    cross_sectional_variance_counts(cnt::Nothing, csr::CrossSectionalRegression)
    cross_sectional_variance_counts(cnt::NamedTuple, csr::CrossSectionalRegression)

Degrees of freedom and divisor of each idiosyncratic variance of a Cross-Sectional Factor Prior.

A cross-sectional fit spends its parameters on the assets of each period, not on the observations of each asset. Observation ``t`` regresses ``n_{t}`` eligible assets on ``K`` factors, and on an intercept when the fit has one. So the ``\\sum_{t} n_{t}`` residuals keep the fraction ``\\phi`` of their degrees of freedom, and the variance of each asset takes that fraction of the effective count that its variance estimator reads.

# Mathematical definition

```math
\\begin{align}
\\phi &= \\dfrac{\\sum_{t} \\left(n_{t} - p\\right)}{\\sum_{t} n_{t}}\\,, \\\\
\\nu_{i} &= \\phi \\, n_{i}^{\\mathrm{eff}}\\,.
\\end{align}
```

Where:

  - ``\\phi``: Fraction of the degrees of freedom that the fit leaves.
  - ``n_{t}``: Number of eligible assets at observation ``t``, `csr.n`.
  - ``p``: Number of parameters of each period, the columns of `csr.f` plus one when `csr.b` is set.
  - ``\\nu_{i}``: Degrees of freedom of the idiosyncratic variance of asset ``i``.
  - ``n_{i}^{\\mathrm{eff}}``: Effective count of the observations of asset ``i``, `cnt.n`.

The fraction spreads the spend evenly over the assets. On a balanced panel of ``T`` observations and ``N`` assets with no intercept, ``\\nu_{i} = T (N - K) / N`` when ``n_{i}^{\\mathrm{eff}} = T``. Under the default exponentially weighted variance, the law of the variance is an approximation, so the level of a bound that reads these counts is also approximate.

# Arguments

  - `cnt`: The count and divisor that [`variance_count`](@ref) states for the variance estimator, or `nothing` when it states none.
  - `csr`: The cross-sectional fit.

# Returns

  - `(; edof, ediv)::NamedTuple`: The degrees of freedom and the divisor of each variance, both `nothing` when the variance estimator states no count.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`variance_count`](@ref)
  - [`ResidualInflation`](@ref)
"""
function cross_sectional_variance_counts(::Nothing, ::CrossSectionalRegression)
    return (; edof = nothing, ediv = nothing)
end
function cross_sectional_variance_counts(cnt::NamedTuple, csr::CrossSectionalRegression)
    p = size(csr.f, 2) + !isnothing(csr.b)
    phi = sum(nt - p for nt in csr.n) / sum(csr.n)
    return (; edof = phi * cnt.n, ediv = cnt.m)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the idiosyncratic covariance of the latest observation.

The variances come from the latest idiosyncratic variances alone, and `ce` gives only the correlations. So the diagonal of the answer does not depend on `ce`, and a positive threshold changes only the entries off the diagonal.

# Mathematical definition

```math
\\begin{align}
\\rho_{ij} &= \\frac{C_{ij}}{\\sqrt{C_{ii} \\, C_{jj}}}\\,, \\\\
\\tilde{\\rho}_{ij} &= \\begin{cases}
    1 & i = j\\,, \\\\
    \\rho_{ij} \\, \\mathbb{1}\\left[\\lvert \\rho_{ij} \\rvert > \\tau\\right] & i \\neq j\\,,
\\end{cases} \\\\
D_{ij} &= \\sqrt{v_{Ti} \\, v_{Tj}} \\, \\tilde{\\rho}_{ij}\\,.
\\end{align}
```

Where:

  - ``C_{ij}``: Entry of the covariance that `ce` estimates from the filled standardised idiosyncratic returns ``\\tilde{z}_{ti}``.
  - ``\\rho_{ij}``, ``\\tilde{\\rho}_{ij}``: Correlation of assets ``i`` and ``j``, before and after the threshold. The indicator is ``0`` when ``\\rho_{ij}`` is not finite.
  - ``\\tau``: The correlation threshold. At ``\\tau = 0`` the answer is the vector of the latest variances ``v_{Ti}``, which is the diagonal of ``\\mathbf{D}``.
  - ``D_{ij}``: Entry of ``\\mathbf{D}``.
  - $(math_dict[:D_orth])
  - $(math_dict[:ztilde_ti_idio])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:T])

# Algorithm

 1. If `th` is zero, return `ev`, the latest idiosyncratic variances. The block then carries a vector, and the asset covariance takes a diagonal.
 2. Otherwise, get `fv` from `ce` with [`gap_fill_value`](@ref). `fv` is the value that `ce` gives a gapped cell of `S`. A cell of `S` is non-finite only where the asset is inactive.
 3. If `fv` is finite, write it over every non-finite cell of a copy of `S`, and estimate the covariance `C` of that copy with `ce`. The fallback `fv` is zero, which is the mean of a standardised series.
 4. If `fv` is not finite, estimate `C` from `S` as it stands, with `amsk` as the `active_mask`. A gap-aware `ce` then freezes the block of an inactive asset and does not decay it.
 5. Convert `C` to the correlation `R`.
 6. Set to zero every entry of `R` off the diagonal whose magnitude does not exceed `th`, and set the diagonal to one. This step also sets a non-finite correlation to zero.
 7. Rescale `R` by the latest idiosyncratic volatilities, giving `D`.
 8. Make the block of `D` over the assets with a finite variance positive definite with [`posdef!`](@ref).

# Arguments

  - `th`: The correlation threshold.
  - `ce`: Covariance estimator of the standardised idiosyncratic returns.
  - `pdm`: Positive definite matrix estimator, or `nothing`.
  - `S`: Standardised idiosyncratic returns, `observations × assets`.
  - `ev`: The latest idiosyncratic variances, one per asset.
  - `amsk`: The active mask, `observations × assets`. Only a `ce` with a non-finite [`gap_fill_value`](@ref) reads it.

# Returns

  - `esigma::VecNum`: The latest idiosyncratic variances, when `th` is zero.
  - `esigma::MatNum`: The idiosyncratic covariance, when `th` is positive.

# Related

  - [`cross_sectional_standardised_residuals`](@ref)
  - [`gap_fill_value`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`posdef!`](@ref)
"""
function cross_sectional_idiosyncratic_covariance(th::Real,
                                                  ce::StatsBase.CovarianceEstimator,
                                                  pdm::Option{<:AbstractPosdefEstimator},
                                                  S::MatNum, ev::VecNum,
                                                  amsk::AbstractMatrix{<:Bool})
    if iszero(th)
        return ev
    end
    fv = gap_fill_value(ce)
    C = if isfinite(fv)
        Z = Matrix{real(eltype(S))}(S)
        for k in CartesianIndices(Z)
            if !isfinite(Z[k])
                Z[k] = fv
            end
        end
        Statistics.cov(ce, Z; dims = 1)
    else
        Statistics.cov(ce, S; dims = 1, active_mask = amsk)
    end
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

The one caller is [`cross_sectional_standardised_residuals`](@ref), which writes the average into each active cell that it cannot standardise. An observation with no finite entry gives zero, which is the mean of a standardised series.

# Mathematical definition

```math
\\begin{align}
\\bar{z}_{t} &= \\begin{cases}
    \\dfrac{1}{n_{t}} \\sum_{i \\in \\mathcal{F}_{t}} z_{ti} & n_{t} > 0\\,, \\\\
    0 & n_{t} = 0\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:zbar_t_idio])
  - $(math_dict[:z_ti_idio])
  - $(math_dict[:F_t_idio])
  - $(math_dict[:n_t_idio])

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

The function divides each idiosyncratic return by the idiosyncratic volatility of the same observation, so the whole history is on one scale and the latest volatilities can rescale it. An active pair whose standardised return is not finite takes the average standardised return of its own observation, so a sparse history does not shorten the scenario set. An inactive pair stays `NaN`.

# Mathematical definition

```math
\\begin{align}
z_{ti} &= \\begin{cases}
    \\varepsilon_{ti} / \\sqrt{v_{ti}} & a_{ti} = 1\\,, \\\\
    \\mathrm{NaN} & a_{ti} = 0\\,,
\\end{cases} \\\\
\\tilde{z}_{ti} &= \\begin{cases}
    z_{ti} & z_{ti} \\in \\mathbb{R}\\,, \\\\
    \\bar{z}_{t} & a_{ti} = 1,\\ z_{ti} \\notin \\mathbb{R}\\,, \\\\
    \\mathrm{NaN} & a_{ti} = 0\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:z_ti_idio])
  - $(math_dict[:ztilde_ti_idio])
  - $(math_dict[:zbar_t_idio])
  - $(math_dict[:eps_ti_idio])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:a_ti_pnl])

A zero variance gives ``0 / 0`` or ``\\pm\\infty``, and a variance still in warm-up is `NaN`, so an active pair takes ``\\bar{z}_{t}`` in each case.

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
    Tf = promote_type(real(eltype(eps)), real(eltype(vs)))
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

The function maps the scenarios of the factor prior onto the assets through the latest loadings, rescales the standardised idiosyncratic returns by the latest idiosyncratic volatilities, and adds the two. So a scenario carries the factor risk and the idiosyncratic risk of the latest observation, whatever the risk of the observation it comes from. The two histories can differ in length, and the last rows of each pair up.

# Mathematical definition

```math
\\begin{align}
n &= \\min(S, T)\\,, \\\\
\\hat{x}_{si} &= \\sum_{k=1}^{K} B_{Tik} \\, f_{S - n + s,\\,k} + \\sqrt{v_{Ti}} \\, \\tilde{z}_{T - n + s,\\,i}\\,, \\quad s = 1, \\ldots, n\\,.
\\end{align}
```

Where:

  - ``n``: Number of asset return scenarios.
  - ``S``: Number of factor return scenarios.
  - ``\\hat{x}_{si}``: Return of asset ``i`` in scenario ``s``.
  - ``f_{sk}``: Return of factor ``k`` in factor return scenario ``s``.
  - ``T``: Number of observations of the standardised idiosyncratic returns.
  - $(math_dict[:B_T_cs])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:ztilde_ti_idio])
  - $(math_dict[:K])

A non-investable asset has a non-finite latest variance or a non-finite row of ``\\mathbf{B}_{T}``, so its column is `NaN` in every scenario. The contract of [`factor_exposure`](@ref) writes `NaN` at every inactive cell, so this also holds for an asset that is inactive at the latest observation.

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

The prior fits on the coverage universe and answers on it, so an asset that it cannot state a moment for stays in the result and carries `NaN`. An asset is investable when the Asset Panel activates it at the latest observation, its latest idiosyncratic variance is finite, and its latest loadings are finite.

# Mathematical definition

```math
\\begin{align}
\\mathcal{I} &= \\left\\{i : a_{Ti} = 1,\\ v_{Ti} \\in \\mathbb{R},\\ B_{Tik} \\in \\mathbb{R} \\ \\forall k \\in \\{1, \\ldots, K\\}\\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:I_inv])
  - $(math_dict[:a_ti_pnl])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:B_T_cs])
  - $(math_dict[:T])
  - $(math_dict[:K])

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

A static Asset Panel carries no observation axis and no masks, so it states no point-in-time universe. The mask fields are type parameters of the Asset Panel, so each shape has its own method.

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

A weight raised from a market capitalisation needs that capitalisation. The prior makes the same cut on its two weight masks, the benchmark mask and the regression mask. A power of zero reads no capitalisation, so the `nothing` method leaves the mask as it is.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a Cross-Sectional Factor Prior reads the market capitalisation.

The benchmark power, which the prior carries, and the regression power, which the weight policy carries, each raise a weight from the market capitalisation. When both are zero, every eligible asset takes the same weight, and the prior reads no Panel Field for the capitalisation. Every member of the weight family carries `p`, which [`cs_weights_initial`](@ref) also reads.

# Arguments

  - `bp`: The benchmark market-capitalisation power.
  - `alg`: The weight policy of the cross-sectional fit.

# Returns

  - `ans::Bool`: Whether the prior reads the market capitalisation.

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

The prior trims every parallel array to the same observations. The market capitalisation is the one array that can be absent, and the `nothing` method keeps it absent.

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

The block holds `L` exactly when it holds a Factor Family Basis. One dispatch on the basis decides both, so the two cannot disagree.

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

A prior that constrains no Factor Family fits on the raw axis. The `nothing` method takes that case. It returns no basis, and returns the exposure history and the two label vectors unchanged.

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

The prior builds the basis over the whole post-warm-up history, and the exposure lag then shortens the fit. The block stores the slice over the fitted observations, so the basis has the observation axis of every other history on the block.

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

The nested factor prior of a [`LowOrderPrior`](@ref) is on the raw axis, so a constraint written in the name of a dropped factor still resolves. A prior that constrains no Factor Family already fits on the raw axis, and the `nothing` method takes that case.

The fit of observation `t` regresses the returns of `t` on the exposures of `t - lag`, so its coefficients are coordinates in the basis of `t - lag`. So the realised factor returns expand with the lagged ratios. The moments describe the next observation, so they expand with the current ratios. The function takes the whole basis and slices it twice. Two slice arguments could hold one basis and one `nothing`, and no method takes that pair.

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

The idiosyncratic covariance is a vector of variances or a full matrix, and each shape has its own method. Both methods return the block over the investable assets alone, because a non-investable asset can carry a non-finite variance, which has no square root.

# Mathematical definition

```math
\\begin{align}
\\mathbf{R} &= \\begin{cases}
    \\operatorname{diag}\\left(\\sqrt{v_{Ti}}\\right)_{i \\in \\mathcal{I}} & \\mathbf{D} \\text{ diagonal}\\,, \\\\
    \\operatorname{chol}\\left(\\mathbf{D}_{\\mathcal{I}\\mathcal{I}}\\right) & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\operatorname{chol}``: Lower Cholesky factor.
  - $(math_dict[:R_idio])
  - $(math_dict[:D_orth])
  - $(math_dict[:I_inv])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:T])

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

The square root `chol` factorises the factor model before `mp` processes it, as in [`factor_lift`](@ref). So `chol' * chol` equals `sigma` only when the processing leaves the matrix unchanged, which the default `mp` does to a positive definite matrix. A detoning `mp` moves `sigma` and leaves `chol` where it was.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{\\mathcal{I}} &= \\mathbf{B}_{T,\\,\\mathcal{I}} \\, \\boldsymbol{\\mu}_{f}\\,, \\\\
\\mathbf{\\Sigma}_{\\mathcal{I}\\mathcal{I}} &= \\mathbf{B}_{T,\\,\\mathcal{I}} \\, \\mathbf{F} \\, \\mathbf{B}_{T,\\,\\mathcal{I}}^{\\intercal} + \\mathbf{D}_{\\mathcal{I}\\mathcal{I}}\\,, \\\\
\\mathbf{C} &= \\begin{bmatrix} \\mathbf{B}_{T,\\,\\mathcal{I}} \\, \\operatorname{chol}(\\mathbf{F}) & \\mathbf{R} \\end{bmatrix}^{\\intercal}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}``: Expected asset returns.
  - ``\\mathbf{\\Sigma}``: Asset covariance.
  - ``\\mathbf{C}``: Low-rank square root of the asset covariance, ``(K + \\lvert \\mathcal{I} \\rvert) \\times N`` over the full asset universe.
  - ``\\mathbf{B}_{T,\\,\\mathcal{I}}``: The rows of ``\\mathbf{B}_{T}`` at the investable assets.
  - ``\\operatorname{chol}``: Lower Cholesky factor.
  - $(math_dict[:B_T_cs])
  - $(math_dict[:mu_f_patt])
  - $(math_dict[:F_patt])
  - $(math_dict[:D_orth])
  - $(math_dict[:R_idio])
  - $(math_dict[:I_inv])

A consequence of the definition: ``\\mathbf{C}_{\\cdot\\mathcal{I}}^{\\intercal} \\mathbf{C}_{\\cdot\\mathcal{I}} = \\mathbf{\\Sigma}_{\\mathcal{I}\\mathcal{I}}``. Every entry of ``\\boldsymbol{\\mu}``, ``\\mathbf{\\Sigma}`` and ``\\mathbf{C}`` outside the investable set is `NaN`.

# Algorithm

 1. Take `Li`, the rows of `L` at the investable assets. Get `D` and `R` from [`cross_sectional_residual_block`](@ref).
 2. Project the factor mean through `Li`, giving `mui`, and the factor covariance, giving `si`.
 3. Process `si` with `mp`, as [`factor_lift`](@ref) does.
 4. Add `D` to `si`, and make the sum positive definite with `mp.pdm`.
 5. Build `ci`, the low-rank square root `[Li * chol(f_sigma).L  R]`.
 6. Scatter `mui`, `si` and `ci` into the full asset universe, giving `mu`, `sigma` and `chol`, with `NaN` at every asset outside `idx`.

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
    Tf = real(eltype(si))
    mu = fill(Tf(NaN), N)
    sigma = fill(Tf(NaN), N, N)
    chol = fill(Tf(NaN), size(ci, 2), N)
    mu[idx] = mui
    sigma[idx, idx] = si
    chol[:, idx] = transpose(ci)
    return (; mu = mu, sigma = sigma, chol = chol)
end
"""
    cross_sectional_alpha_split(cre::AbstractCrossSectionalRegressionEstimator, mu::VecNum,
                                L::MatNum, w::VecNum) -> NamedTuple

Split a Return Forecast into the part the latest Factor Exposures span and the part they do not.

# Mathematical definition

Under [`CrossSectionalLinearRegression`](@ref), the spanned coefficients solve a weighted least squares over the valid assets. Another Cross-Sectional Regression Estimator fits ``\\boldsymbol{g}`` by its own rule over the same assets and weights.

```math
\\begin{align}
\\mathcal{V} &= \\left\\{i : \\alpha_{i} \\in \\mathbb{R},\\ B_{Tik} \\in \\mathbb{R} \\ \\forall k,\\ u_{i} > 0\\right\\}\\,, \\\\
(c, \\boldsymbol{g}) &= \\underset{c,\\,\\boldsymbol{g}}{\\arg\\min} \\sum_{i \\in \\mathcal{V}} u_{i} \\left(\\alpha_{i} - c - \\sum_{k=1}^{K} B_{Tik} \\, g_{k}\\right)^{2}\\,, \\\\
\\boldsymbol{\\alpha}^{\\perp} &= \\boldsymbol{\\alpha} - \\mathbf{B}_{T} \\, \\boldsymbol{g}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{V}``: Valid assets of the split.
  - ``\\alpha_{i}``: Return Forecast of asset ``i``.
  - ``u_{i}``: Regression weight of asset ``i`` in the latest fit.
  - ``c``: Intercept of the regression, held at ``0`` when the estimator fits none.
  - $(math_dict[:alpha_perp])
  - $(math_dict[:g_span])
  - $(math_dict[:B_T_cs])
  - $(math_dict[:K])

The orthogonal part is ``\\boldsymbol{\\alpha} - \\mathbf{B}_{T} \\boldsymbol{g}`` and not the residual of the regression, so the intercept stays in it: the weighted mean of ``\\alpha^{\\perp}_{i}`` over ``\\mathcal{V}`` is ``c``. The split then adds up. The prior's expected return is ``\\mathbf{B}_{T} (\\lambda \\boldsymbol{\\mu}_{f} + (1 - \\lambda) \\boldsymbol{g}) + c_{\\alpha} \\boldsymbol{\\alpha}^{\\perp}``, which is ``\\boldsymbol{\\alpha}`` at ``\\lambda = 0`` and ``c_{\\alpha} = 1``, whatever the intercept. Without an intercept, ``\\mathbf{B}_{T,\\,\\mathcal{V}}^{\\intercal} \\mathbf{U} \\boldsymbol{\\alpha}^{\\perp}_{\\mathcal{V}} = \\boldsymbol{0}``, with ``\\mathbf{U}`` the diagonal matrix of the weights ``u_{i}``. The symbols ``\\lambda``, ``c_{\\alpha}`` and ``\\boldsymbol{\\mu}_{f}`` are the shrinkage, the confidence and the factor mean of [`CrossSectionalFactorPrior`](@ref).

# Algorithm

 1. If `mu` is zero at every asset, return a zero `g` and a zero `ap`. No regression runs.
 2. Copy `w` into `wv`, and set to zero the weight of every asset whose forecast, or whose row of exposures, is not finite. The weights come from the lagged fit, so a positive weight says nothing about the finiteness of the latest forecast. If no weight in `wv` is positive, return the zero split of step 1.
 3. Regress `mu` on `L` across the assets under `wv`, through `cre`, giving `g`.
 4. Subtract `L * g` from `mu`, giving `ap`.

# Arguments

  - `cre`: Cross-Sectional Regression Estimator of the split.
  - `mu`: The Return Forecast, one entry per asset of the coverage universe.
  - `L`: The latest Factor Exposures, `assets × factors`, in the basis the fit ran in.
  - `w`: The regression weights of the latest fit, one entry per asset.

# Validation

  - The rules of [`cross_sectional_regression`](@ref).

# Returns

  - `g::VecNum`: The spanned coefficients, one per factor of `L`.
  - `ap::VecNum`: The orthogonal part of the forecast, one entry per asset.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_return_forecast`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
"""
function cross_sectional_alpha_split(cre::AbstractCrossSectionalRegressionEstimator,
                                     mu::VecNum, L::MatNum, w::VecNum)
    N = size(L, 1)
    K = size(L, 2)
    Tf = promote_type(real(eltype(mu)), real(eltype(L)), real(eltype(w)))
    if all(iszero, mu)
        return (; g = zeros(Tf, K), ap = zeros(Tf, N))
    end
    wv = zeros(Tf, N)
    for i in 1:N
        if isfinite(mu[i]) && all(isfinite, view(L, i, :))
            wv[i] = w[i]
        end
    end
    if !any(x -> x > zero(x), wv)
        return (; g = zeros(Tf, K), ap = zeros(Tf, N))
    end
    csr = cross_sectional_regression(cre, reshape(Tf.(L), 1, N, K), reshape(Tf.(mu), 1, N),
                                     reshape(wv, 1, N))
    g = csr.f[1, :]
    return (; g = g, ap = mu - L * g)
end
"""
    cross_sectional_return_forecast(rfe::Nothing, rd::ReturnsResult,
                                    csfm::CrossSectionalFactorModel,
                                    cre::AbstractCrossSectionalRegressionEstimator,
                                    c::Real) -> NamedTuple
    cross_sectional_return_forecast(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                                    csfm::CrossSectionalFactorModel,
                                    cre::AbstractCrossSectionalRegressionEstimator,
                                    c::Real) -> NamedTuple

Fit the Return Forecast of a [`CrossSectionalFactorPrior`](@ref), and write its split onto the factor-model block.

A prior that states no Return Forecast Estimator takes the method over `Nothing`. That method returns the block with the zero `b` that it carries, and a `g` of `nothing`, so the factor mean does not change.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{b} &= c_{\\alpha} \\, \\boldsymbol{\\alpha}^{\\perp}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{b}``: Intercept of the block, one entry per asset.
  - ``c_{\\alpha}``: Confidence in the orthogonal part of the forecast.
  - $(math_dict[:alpha_perp]) [`cross_sectional_alpha_split`](@ref) states it.

# Algorithm

 1. Fit `rfe` on the coverage universe through [`return_forecast`](@ref), giving `rf`. The carrier is the whole one, so the Descriptors of the forecast warm up over every observation of the panel. [`return_forecast_rows`](@ref) finds the block as a suffix of that carrier by its size.
 2. Split `rf.mu` against the latest exposures with [`cross_sectional_alpha_split`](@ref), giving `g` and `ap`.
 3. Rebuild the block with `b = c * ap` and with `rf` in its field `rf`. Read `L` with `getfield`. The property `L` of [`CrossSectionalFactorModel`](@ref) gives `M` when `L` is unset, and the rebuilt block would then hold `M` as a set `L`.

# Arguments

  - `rfe`: Return Forecast Estimator, or `nothing`.
  - $(arg_dict[:rd]) It is the whole carrier the prior was fitted on, and the block is a suffix of it.
  - `csfm`: The factor-model block, built with a zero `b` and no Return Forecast.
  - `cre`: Cross-Sectional Regression Estimator of the split.
  - `c`: Confidence in the orthogonal part of the forecast.

# Validation

  - The rules of [`return_forecast`](@ref) and of [`cross_sectional_alpha_split`](@ref).

# Returns

  - `rr::CrossSectionalFactorModel`: The block, with `b` and `rf` set.
  - `g::Option{<:VecNum}`: The spanned coefficients, or `nothing` when the prior states no estimator.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_alpha_split`](@ref)
  - [`cross_sectional_forecast_mu`](@ref)
  - [`return_forecast`](@ref)
"""
function cross_sectional_return_forecast(::Nothing, ::ReturnsResult,
                                         csfm::CrossSectionalFactorModel,
                                         ::AbstractCrossSectionalRegressionEstimator,
                                         ::Real)
    return (; rr = csfm, g = nothing)
end
function cross_sectional_return_forecast(rfe::AbstractReturnForecastEstimator,
                                         rd::ReturnsResult, csfm::CrossSectionalFactorModel,
                                         cre::AbstractCrossSectionalRegressionEstimator,
                                         c::Real)
    rf = return_forecast(rfe, rd, csfm)
    rw = csfm.rw
    (; g, ap) = cross_sectional_alpha_split(cre, rf.mu, csfm.L, @view(rw[size(rw, 1), :]))
    return (;
            rr = CrossSectionalFactorModel(; M = csfm.M, L = getfield(csfm, :L), b = c * ap,
                                           csr = csfm.csr, Ms = csfm.Ms, vs = csfm.vs,
                                           esigma = csfm.esigma, edof = csfm.edof,
                                           ediv = csfm.ediv, rw = rw, bw = csfm.bw,
                                           nf = csfm.nf, fam = csfm.fam, fcb = csfm.fcb,
                                           lag = csfm.lag, rf = rf), g = g)
end
"""
    cross_sectional_forecast_mu(lambda::Real, mu::VecNum, g::Nothing) -> VecNum
    cross_sectional_forecast_mu(lambda::Real, mu::VecNum, g::VecNum) -> VecNum

Blend the expected factor returns with the spanned part of a Return Forecast.

A prior that states no Return Forecast Estimator has a spanned part of zero, and the method over `Nothing` takes it. There `lambda` shrinks the factor mean towards zero, and `lambda = 0` gives an expected return of zero.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{\\mu}}_{f} &= \\lambda \\, \\boldsymbol{\\mu}_{f} + (1 - \\lambda) \\, \\boldsymbol{g}\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\boldsymbol{\\mu}}_{f}``: Blended expected factor returns.
  - ``\\lambda``: Shrinkage of the factor mean towards the spanned forecast. At ``\\lambda = 1`` the blend is the factor mean, and at ``\\lambda = 0`` it is the spanned forecast.
  - $(math_dict[:mu_f_patt])
  - $(math_dict[:g_span]) It is ``\\boldsymbol{0}`` when the prior states no Return Forecast Estimator.

# Arguments

  - `lambda`: Shrinkage of the factor mean towards the spanned forecast.
  - `mu`: The expected factor returns of the nested factor prior, on the reduced axis.
  - `g`: The spanned coefficients of the Return Forecast, or `nothing`.

# Returns

  - `mu::VecNum`: The blended expected factor returns, on the reduced axis.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_return_forecast`](@ref)
  - [`cross_sectional_alpha_split`](@ref)
"""
function cross_sectional_forecast_mu(lambda::Real, mu::VecNum, ::Nothing)
    return lambda * mu
end
function cross_sectional_forecast_mu(lambda::Real, mu::VecNum, g::VecNum)
    return lambda * mu + (one(lambda) - lambda) * g
end
