"""
    ew_active_returns(X::AbstractMatrix{<:Number}, pnl::AssetPanel) -> Matrix

Copy the returns with every inactive cell written to `NaN`.

An asset outside the universe has no return, so its cell must not advance a recursion. Every market beta Descriptor takes this copy before its recursion starts. A recursion skips a `NaN` return, so the state of an asset holds its value at every observation where the asset is not listed.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `pnl`: The Asset Panel whose active mask is read.

# Returns

  - `Xm::Matrix{<:Real}`: The returns, `NaN` wherever the active mask is `false`. Integer returns become floating point numbers, so that a cell can hold `NaN`. Every other element type is kept.

# Related

  - [`EWBeta`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function ew_active_returns(X::AbstractMatrix{<:Number}, pnl::AssetPanel)
    Xm = Matrix{float_if_integer(eltype(X))}(X)
    descriptor_active_fill!(Xm, pnl)
    return Xm
end
"""
    ew_agg_series(A::AbstractMatrix{<:Real}, agg_obs::Integer) -> Matrix{<:Real}

Aggregate a matrix into one row per complete window of consecutive observations.

Two series that close at different times of day record one move on different observations, so the exponentially weighted covariance of their raw returns is biased toward zero. A window of several observations holds the move of both series, and the covariance of the aggregated series does not carry that bias. A tail shorter than one window is dropped, so an aggregated series never mixes a complete window with a partial one.

# Mathematical definition

```math
\\begin{align}
\\bar{a}_{k,i} &= \\frac{1}{\\lvert \\mathcal{V}_{k,i} \\rvert} \\sum_{t \\in \\mathcal{V}_{k,i}} a_{t,i}\\,, \\\\
\\mathcal{V}_{k,i} &= \\left\\{ t \\in \\{(k - 1) h + 1, \\ldots, k h\\} : a_{t,i} \\text{ is finite} \\right\\}\\,.
\\end{align}
```

Where:

  - ``a_{t,i}``: Entry of the series at observation ``t`` and column ``i``.
  - ``\\bar{a}_{k,i}``: Aggregated entry of window ``k`` and column ``i``. It is `NaN` when ``\\mathcal{V}_{k,i}`` is empty.
  - ``\\mathcal{V}_{k,i}``: Observations of window ``k`` whose entry in column ``i`` is finite.
  - ``h``: `agg_obs`, the number of observations in one window.

# Arguments

  - `A`: The series, `observations × assets`.
  - $(arg_dict[:agg_obs])

# Returns

  - `B::Matrix{<:Real}`: The aggregated series, `div(observations, agg_obs) × assets`. An integer series gives floating point means, and every other element type is kept.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_agg_series([1.0 2.0; 3.0 NaN; 5.0 6.0], 2)
1×2 Matrix{Float64}:
 2.0  2.0
```

# Related

  - [`ew_agg_vector`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
"""
function ew_agg_series(A::AbstractMatrix{<:Real}, agg_obs::Integer)
    Tf = float_if_integer(eltype(A))
    T, N = size(A)
    K = div(T, agg_obs)
    B = Matrix{Tf}(undef, K, N)
    for k in 1:K, i in 1:N
        s = zero(Tf)
        c = 0
        for t in ((k - 1) * agg_obs + 1):(k * agg_obs)
            a = A[t, i]
            if isfinite(a)
                s += a
                c += 1
            end
        end
        B[k, i] = iszero(c) ? Tf(NaN) : s / c
    end
    return B
end
"""
    ew_agg_vector(v::AbstractVector{<:Real}, agg_obs::Integer) -> Vector{<:Real}

Aggregate a series into one entry per complete window of consecutive observations.

This is [`ew_agg_series`](@ref) over one column. The Descriptors use it to aggregate the market return and the reference return, which are vectors with one entry per observation.

# Arguments

  - `v`: The series, one entry per observation.
  - $(arg_dict[:agg_obs])

# Returns

  - `w::Vector{<:Real}`: The aggregated series, of length `div(length(v), agg_obs)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_agg_vector([1.0, 3.0, 5.0, 9.0], 2)
2-element Vector{Float64}:
 2.0
 7.0
```

# Related

  - [`ew_agg_series`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
"""
function ew_agg_vector(v::AbstractVector{<:Real}, agg_obs::Integer)
    return vec(ew_agg_series(reshape(v, :, 1), agg_obs))
end
"""
    ew_beta_expand(Ba::AbstractMatrix{<:Number}, T::Integer, agg_obs::Integer,
                   b0::Option{<:AbstractVector{<:Number}} = nothing,
                   r0::Integer = 0) -> Matrix

Spread an aggregated beta series back over the observations it was aggregated from.

The recursion advances once per complete window, so each observation takes the beta of the last window that closed at or before it. An observation before the first window closes takes `b0`, or `NaN` when `b0` is `nothing`. The batch call starts at the first observation, where no beta exists yet, so it is `NaN`. A step of the carry fold starts after `r0` observations of a window that is not yet complete, and `b0` is the beta of the last window that the state folded.

# Arguments

  - `Ba`: The aggregated betas, `windows × assets`.
  - `T`: Number of observations of the unaggregated series.
  - $(arg_dict[:agg_obs])
  - `b0`: The beta before the first window of `Ba` closes, one entry per asset, or `nothing` for `NaN`.
  - `r0`: The number of observations of the first window that come before the series.

# Returns

  - `B::Matrix{<:Real}`: The betas, `T × assets`. Integer betas become floating point numbers, and every other element type is kept.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_beta_expand([1.0 2.0; 3.0 4.0], 5, 2)
5×2 Matrix{Float64}:
 NaN    NaN
   1.0    2.0
   1.0    2.0
   3.0    4.0
   3.0    4.0
```

# Related

  - [`ew_agg_series`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
"""
function ew_beta_expand(Ba::AbstractMatrix{<:Number}, T::Integer, agg_obs::Integer,
                        b0::Option{<:AbstractVector{<:Number}} = nothing,
                        r0::Integer = 0)::Matrix
    Tf = float_if_integer(eltype(Ba))
    B = Matrix{Tf}(undef, T, size(Ba, 2))
    for t in 1:T
        k = div(r0 + t, agg_obs)
        if k >= 1
            B[t, :] = view(Ba, k, :)
        elseif isnothing(b0)
            B[t, :] .= Tf(NaN)
        else
            B[t, :] = b0
        end
    end
    return B
end
"""
$(DocStringExtensions.TYPEDEF)

The carried state of an exponentially weighted Descriptor whose recursion reads one aggregated row per window of `agg_obs` observations, an [`EWBeta`](@ref) or an [`EWMacroSensitivity`](@ref).

The recursion folds a window only when it is complete, so a step of the carry fold of a [`CrossSectionalFactorPrior`](@ref) that ends inside a window keeps its observations here, and the next step aggregates them with its own. The observations start at the first observation of a window, so a window aggregates the same values, in the same order, as the batch call. With `agg_obs = 1` every window is complete, and the state keeps no observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWBlockState(; st::AbstractPartialFitState, X::AbstractMatrix{<:Real},
                 m::AbstractVector{<:Real}, f::Option{<:AbstractVector{<:Real}} = nothing,
                 v::Option{<:AbstractVector{<:Real}} = nothing,
                 s::Option{<:AbstractVector{<:Real}} = nothing) -> EWBlockState

Keywords correspond to the struct's fields.

## Validation

  - `X`, `m` and `f` hold the same number of observations, and `v` and `s` one entry per asset of `X`. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`EWMacroSensitivityState`](@ref)
  - [`ew_block_state`](@ref)
  - [`ew_block_rows`](@ref)
  - [`ew_beta_fold`](@ref)
"""
@concrete struct EWBlockState <: AbstractPartialFitState
    """
    The state of the recursion after the last complete window, an [`EWBetaState`](@ref) or an [`EWMacroSensitivityState`](@ref).
    """
    st
    """
    The masked returns of the observations of the window that is not yet complete, `observations × assets`.
    """
    X
    """
    The market return of those observations.
    """
    m
    """
    The reference return of those observations for an [`EWMacroSensitivity`](@ref), or `nothing`.
    """
    f
    """
    The residual variance of each asset after the last complete window, for an [`EWBeta`](@ref) with a `group`, or `nothing`.
    """
    v
    """
    The shrunk beta of each asset after the last complete window, for an [`EWBeta`](@ref) with a `group`, or `nothing`.
    """
    s
    function EWBlockState(st::AbstractPartialFitState, X::AbstractMatrix{<:Real},
                          m::AbstractVector{<:Real}, f::Option{<:AbstractVector{<:Real}},
                          v::Option{<:AbstractVector{<:Real}},
                          s::Option{<:AbstractVector{<:Real}})
        @argcheck(size(X, 1) == length(m) && (isnothing(f) || length(f) == length(m)),
                  DimensionMismatch("the observations of an EWBlockState hold one market return and one reference return each, got $(size(X, 1)) rows, $(length(m)) market returns and $(isnothing(f) ? 0 : length(f)) reference returns"))
        @argcheck(all(x -> isnothing(x) || length(x) == size(X, 2), (v, s)),
                  DimensionMismatch("the residual variance and the shrunk beta of an EWBlockState hold one entry per asset, got $(size(X, 2)) assets"))
        return new{typeof(st), typeof(X), typeof(m), typeof(f), typeof(v), typeof(s)}(st, X,
                                                                                      m, f,
                                                                                      v, s)
    end
end
function EWBlockState(; st::AbstractPartialFitState, X::AbstractMatrix{<:Real},
                      m::AbstractVector{<:Real},
                      f::Option{<:AbstractVector{<:Real}} = nothing,
                      v::Option{<:AbstractVector{<:Real}} = nothing,
                      s::Option{<:AbstractVector{<:Real}} = nothing)::EWBlockState
    return EWBlockState(st, X, m, f, v, s)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWBlockState`](@ref), so that the copy shares no vector with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWBlockState`: A new state, equal to `x`.

# Related

  - [`EWBlockState`](@ref)
"""
function Base.copy(x::EWBlockState)
    return EWBlockState(copy(x.st), copy(x.X), copy(x.m), ew_block_copy(x.f),
                        ew_block_copy(x.v), ew_block_copy(x.s))
end
"""
    ew_block_copy(::Nothing)
    ew_block_copy(x::AbstractVector)

Returns `nothing` for `nothing`, and a copy of a vector otherwise, for the optional vectors of an [`EWBlockState`](@ref).

# Arguments

  - `x`: The vector, or `nothing`.

# Returns

  - `x::Option{<:AbstractVector}`: The copy, or `nothing`.

# Related

  - [`EWBlockState`](@ref)
"""
function ew_block_copy(::Nothing)::Nothing
    return nothing
end
function ew_block_copy(x::AbstractVector)::AbstractVector
    return copy(x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`EWBlockState`](@ref): the recursion of the second block starts from the state after the first one, so two states fitted on disjoint blocks do not give the state of their union to the last bit.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`EWBlockState`](@ref)
  - [`partial_fit!`](@ref)
"""
function merge_states(::EWBlockState, ::EWBlockState)
    return throw(ArgumentError("an EWBlockState cannot merge two states fitted on disjoint blocks: the recursion of the second block starts from the state after the first one. Fold the second block into the state of the first with partial_fit!."))
end
"""
    ew_block_rows(P::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real},
                  agg_obs::Integer) -> NamedTuple

Joins the observations that an [`EWBlockState`](@ref) carries to the observations of a step, and splits the result at the end of the last complete window.

[`ew_agg_series`](@ref) and [`ew_agg_vector`](@ref) aggregate the complete windows of the joined rows and drop the rest, so the rest is what the state carries to the next step.

# Arguments

  - `P`: The carried observations, which start a window.
  - `Y`: The observations of the step.
  - $(arg_dict[:agg_obs])

# Returns

  - `rows::NamedTuple`: `A`, the joined observations, and `P`, a copy of the observations after the last complete window.

# Related

  - [`EWBlockState`](@ref)
  - [`ew_beta_fold`](@ref)
"""
function ew_block_rows(P::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real},
                       agg_obs::Integer)
    A = isempty(P) ? Y : vcat(P, Y)
    n = div(size(A, 1), agg_obs) * agg_obs
    return (; A = A, P = collect(selectdim(A, 1, (n + 1):size(A, 1))))
end
"""
    ew_beta_residual_variance(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                              B::AbstractMatrix{<:Real}, decay::Real, min_obs::Integer,
                              v0::AbstractVector{<:Real} = zeros(…, size(X, 2)),
                              b0::AbstractVector{<:Real} = similar(B, size(X, 2)),
                              k0::Integer = 0) -> Matrix{<:Real}

Run the exponentially weighted variance of the market-model residual of every asset.

The shrinkage of [`EWBeta`](@ref) weighs each raw beta against the noise of its own estimate, and this variance measures that noise. Each residual is measured against the beta of the **previous** observation, so the beta that the residual helps to correct does not enter it. The recursion starts from zero one observation after the warm-up ends, because no residual exists before the first beta. An observation where the return or the previous beta is not finite leaves the variance of that asset unchanged.

The batch call starts at the first observation with the defaults. A step of the carry fold starts after the `k0` observations that its state folded, from the variance `v0` and the beta `b0` of the last of them, so the step runs the arithmetic of the batch call. `v0` is not changed.

# Mathematical definition

```math
\\begin{align}
e_{t,i} &= x_{t,\\,i} - \\beta_{t-1,i}\\, r_{m,t}\\,, \\\\
V^{\\varepsilon}_{t,i} &= \\lambda V^{\\varepsilon}_{t-1,i} + (1 - \\lambda)\\, e_{t,i}^2\\,.
\\end{align}
```

Where:

  - ``e_{t,i}``: Market-model residual of asset ``i`` at observation ``t``.
  - $(math_dict[:V_eps_ti_ewb])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:beta_ti_ewb])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:lambda_ew])

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `B`: The raw betas, `observations × assets`, as [`ew_beta_series`](@ref) returns them.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - `v0`: The residual variance before the first observation, one entry per asset. Its element type is the element type of `Vr`.
  - `b0`: The raw beta of the observation before the first one, one entry per asset. The recursion reads it only when `k0 >= min_obs`, so the default of the batch call holds no value.
  - `k0`: The number of observations before the first one.

# Returns

  - `Vr::Matrix{<:Real}`: The residual variance after each observation, `observations × assets`.

# Examples

```jldoctest
julia> B, Vm = PortfolioOptimisers.ew_beta_series([0.1 0.2; -0.1 0.3; 0.05 -0.05],
                                                  [0.1, -0.05, 0.02], 0.5, 1, 1e-12);

julia> PortfolioOptimisers.ew_beta_residual_variance([0.1 0.2; -0.1 0.3; 0.05 -0.05],
                                                     [0.1, -0.05, 0.02], B, 0.5, 1)
3×2 Matrix{Float64}:
 0.0          0.0
 0.00125      0.08
 0.000897222  0.0406722
```

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_series`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function ew_beta_residual_variance(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                   B::AbstractMatrix{<:Real}, decay::Real, min_obs::Integer,
                                   v0::AbstractVector{<:Real} = zeros(float_if_integer(promote_type(eltype(X),
                                                                                                    eltype(rm),
                                                                                                    eltype(B))),
                                                                      size(X, 2)),
                                   b0::AbstractVector{<:Real} = similar(B, size(X, 2)),
                                   k0::Integer = 0)
    v = copy(v0)
    Tf = eltype(v)
    K, N = size(X)
    Vr = Matrix{Tf}(undef, K, N)
    om = one(Tf) - decay
    p = copy(b0)
    for k in 1:K
        if k0 + k - 1 >= min_obs
            for i in 1:N
                x = X[k, i]
                bp = p[i]
                if isfinite(x) && isfinite(bp)
                    e = x - bp * rm[k]
                    v[i] = decay * v[i] + om * e * e
                end
            end
        end
        Vr[k, :] = v
        p .= view(B, k, :)
    end
    return Vr
end
"""
    ew_masked_mean(v::AbstractVector{<:Number}, msk::AbstractVector{Bool}) -> Number

Mean of the entries a mask selects.

# Arguments

  - `v`: The values.
  - `msk`: The mask, of the same length as `v`.

# Returns

  - `m::Real`: The mean of `v[msk]`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_masked_mean([1.0, 2.0, 6.0], [true, false, true])
3.5
```

# Related

  - [`ew_masked_weighted_mean`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function ew_masked_mean(v::AbstractVector{<:Number}, msk::AbstractVector{Bool})::Number
    Tf = eltype(v)
    s = zero(Tf)
    c = 0
    for i in eachindex(v, msk)
        if msk[i]
            s += v[i]
            c += 1
        end
    end
    return s / c
end
"""
    ew_masked_weighted_mean(v::AbstractVector{<:Number}, w::AbstractVector{<:Number},
                            msk::AbstractVector{Bool}) -> Number

Weighted mean of the entries a mask selects.

The shrinkage of [`EWBeta`](@ref) weights the mean beta of a group by capitalisation, so a large asset moves the mean of its group more than a small asset does.

# Arguments

  - `v`: The values.
  - `w`: The weights, of the same length as `v`.
  - `msk`: The mask, of the same length as `v`.

# Returns

  - `m::Real`: The weighted mean of `v[msk]`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_masked_weighted_mean([1.0, 2.0, 6.0], [1.0, 1.0, 3.0],
                                                   [true, false, true])
4.75
```

# Related

  - [`ew_masked_mean`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function ew_masked_weighted_mean(v::AbstractVector{<:Number}, w::AbstractVector{<:Number},
                                 msk::AbstractVector{Bool})::Number
    Tf = promote_type(eltype(v), eltype(w))
    s = zero(Tf)
    d = zero(Tf)
    for i in eachindex(v, w, msk)
        if msk[i]
            s += w[i] * v[i]
            d += w[i]
        end
    end
    return s / d
end
"""
    ew_beta_group_prior(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                        w::AbstractVector{<:Real},
                        msk::AbstractVector{Bool}) -> Tuple{Real, Real}

Estimate the mean and the prior variance of the betas a mask selects.

The two values are ``\\mu_{t,g}`` and ``\\tau^2_{t,g}`` of the definition that [`ew_beta_shrink`](@ref) states. The prior variance is the observed dispersion of the betas less the mean estimation error variance of the group. It is floored at zero, because a group whose estimation noise exceeds its observed dispersion shows no dispersion of its own.

# Arguments

  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta.
  - `w`: The capitalisation weights.
  - `msk`: The mask that selects the group.

# Returns

  - `m::Real`: The capitalisation-weighted mean beta of the group, ``\\mu_{t,g}``.
  - `pv::Real`: The prior variance of the group, ``\\tau^2_{t,g}``.

# Related

  - [`ew_beta_shrink`](@ref)
  - [`EWBeta`](@ref)
"""
function ew_beta_group_prior(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                             w::AbstractVector{<:Real},
                             msk::AbstractVector{Bool})::Tuple{Real, Real}
    m = ew_masked_weighted_mean(b, w, msk)
    v = ew_masked_weighted_mean((b .- m) .^ 2, w, msk)
    return m, max(v - ew_masked_mean(bev, msk), zero(v))
end
"""
    ew_beta_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                   L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                   min_group_size::Integer, bounds::Tuple{<:Real, <:Real},
                   min_val::Real) -> Vector{<:Real}

Shrink one cross-section of raw betas toward the capitalisation-weighted mean of its group.

A beta estimated from few observations is mostly noise, and the mean beta of the assets in its industry is then a better estimate of it. The empirical Bayes weight is the share of the dispersion of the group that is not noise, so a precise beta keeps most of its own value and a noisy beta moves most of the way to the mean of its group.

# Mathematical definition

```math
\\begin{align}
\\beta^{\\ast}_{t,i} &= q_{t,i}\\, \\beta_{t,i} + (1 - q_{t,i})\\, \\mu_{t,g}\\,, \\\\
q_{t,i} &= \\mathrm{clamp}\\!\\left(\\frac{\\tau^2_{t,g}}{\\tau^2_{t,g} + \\sigma^2_{t,i} + \\texttt{min\\_val}},\\, q_{\\mathrm{lo}},\\, q_{\\mathrm{hi}}\\right)\\,, \\\\
\\sigma^2_{t,i} &= \\frac{V^{\\varepsilon}_{t,i}}{n_{\\mathrm{eff}} \\left(V_{m,t} + \\texttt{min\\_val}\\right)}\\,, \\\\
\\mu_{t,g} &= \\frac{\\sum_{j \\in g} c_{t,j}\\, \\beta_{t,j}}{\\sum_{j \\in g} c_{t,j}}\\,, \\\\
\\tau^2_{t,g} &= \\max\\!\\left(\\frac{\\sum_{j \\in g} c_{t,j} \\left(\\beta_{t,j} - \\mu_{t,g}\\right)^2}{\\sum_{j \\in g} c_{t,j}} - \\frac{1}{\\lvert g \\rvert} \\sum_{j \\in g} \\sigma^2_{t,j},\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\beta^{\\ast}_{t,i}``: Shrunk beta of asset ``i``.
  - ``q_{t,i}``: Weight that the shrunk beta keeps on the raw beta.
  - ``g``: Group of asset ``i``, the assets of the estimation set that carry its label. A group smaller than `min_group_size` takes the whole estimation set in its place.
  - ``\\mu_{t,g}``: Capitalisation-weighted mean beta of group ``g``.
  - ``\\tau^2_{t,g}``: Prior variance of group ``g``, its observed dispersion less its mean estimation error variance.
  - ``\\sigma^2_{t,i}``: Estimation error variance of ``\\beta_{t,i}``.
  - ``c_{t,j}``: Capitalisation of asset ``j`` at observation ``t``.
  - ``q_{\\mathrm{lo}}``, ``q_{\\mathrm{hi}}``: The two entries of `bounds`.
  - ``n_{\\mathrm{eff}}``: Effective sample size of the recursion, twice its half-life.
  - $(math_dict[:beta_ti_ewb])
  - $(math_dict[:V_eps_ti_ewb])
  - $(math_dict[:V_mt_ewb])
  - $(math_dict[:min_val_ewb])

# Algorithm

 1. Build the estimation set `vld`, the assets whose beta is not `NaN`, whose group label is set, and whose weight is finite and strictly positive. An asset outside `vld` keeps its raw beta, and an empty `vld` returns the raw betas unchanged.
 2. Compute the mean `gm` and the prior variance `gpv` of the whole estimation set through [`ew_beta_group_prior`](@ref).
 3. For each group label in `vld`, compute its mean `m` and its prior variance `pv` through [`ew_beta_group_prior`](@ref). A group with fewer than `min_group_size` members takes `gm` and `gpv` instead.
 4. For each member of the group, compute the weight `q` and clamp it to `bounds`.
 5. Write the shrunk beta, `q` times the raw beta plus `1 - q` times `m`.

# Arguments

  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta, ``\\sigma^2_{t,i}``.
  - `L`: The group label of each asset, [`CS_MISSING_GROUP`](@ref) where the asset carries none.
  - `w`: The capitalisation weights.
  - $(arg_dict[:min_group_size_ewb])
  - $(arg_dict[:bounds_ewb])
  - $(arg_dict[:min_val])

# Returns

  - `s::Vector{<:Real}`: The shrunk betas.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_beta_shrink([0.8, 1.4], [0.01, 0.01], [1, 1], [1.0, 1.0], 1,
                                          (0.0, 1.0), 1e-12)
2-element Vector{Float64}:
 0.8333333333362964
 1.3666666666637035
```

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_group_prior`](@ref)
  - [`ew_beta_residual_variance`](@ref)
  - [`cross_sectional_groups`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
"""
function ew_beta_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                        L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                        min_group_size::Integer, bounds::Tuple{<:Real, <:Real},
                        min_val::Real)
    s = collect(b)
    vld = [!isnan(b[i]) &&
               L[i] != CS_MISSING_GROUP &&
               isfinite(w[i]) &&
               w[i] > zero(eltype(w)) for i in eachindex(b, L, w)]
    if !any(vld)
        return s
    end
    gm, gpv = ew_beta_group_prior(b, bev, w, vld)
    lo, hi = bounds
    msk = similar(vld)
    for g in unique(L[i] for i in eachindex(L) if vld[i])
        msk .= vld .& (L .== g)
        m, pv = if count(msk) < min_group_size
            (gm, gpv)
        else
            ew_beta_group_prior(b, bev, w, msk)
        end
        for i in eachindex(msk)
            if msk[i]
                q = clamp(pv / (pv + bev[i] + min_val), lo, hi)
                s[i] = q * b[i] + (one(q) - q) * m
            end
        end
    end
    return s
end
"""
    assert_ew_shrinkage_bounds(bounds::Tuple{<:Real, <:Real}) -> nothing

Check the bounds an exponentially weighted shrinkage clamps its weight to.

# Arguments

  - `bounds`: The `(lo, hi)` bounds.

# Validation

  - `0 <= lo <= hi <= 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function assert_ew_shrinkage_bounds(bounds::Tuple{<:Real, <:Real})::Nothing
    lo, hi = bounds
    @argcheck(zero(lo) <= lo <= hi <= one(hi),
              DomainError(bounds,
                          "bounds clamp the weight a shrunk beta keeps on its raw value, so they must satisfy `0 <= lo <= hi <= 1`, got $bounds"))
    return nothing
end
"""
    assert_ew_agg_obs(agg_obs::Integer) -> nothing

Check the number of observations an exponentially weighted recursion aggregates into one update.

# Arguments

  - $(arg_dict[:agg_obs])

# Validation

  - `agg_obs >= 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
  - [`ew_agg_series`](@ref)
"""
function assert_ew_agg_obs(agg_obs::Integer)::Nothing
    @argcheck(agg_obs >= one(agg_obs),
              DomainError(agg_obs,
                          "agg_obs is the number of consecutive observations aggregated into one update of the recursion, so it must be at least one, got $agg_obs"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted beta of the returns against the market return, at every observation.

This is the archetype of every market beta Descriptor. The beta is the exponentially weighted covariance of an asset with the market, divided by the exponentially weighted variance of the market. [`market_return_series`](@ref) builds the market return from the Asset Panel. Where the estimator names a categorical Panel Field in `group`, [`ew_beta_shrink`](@ref) shrinks each raw beta toward the capitalisation-weighted mean beta of its group.

# Mathematical definition

```math
\\begin{align}
\\beta_{t,i} &= \\frac{C_{t,i}}{V_{m,t} + \\texttt{min\\_val}}\\,, \\\\
C_{t,i} &= \\lambda C_{t-1,i} + (1 - \\lambda)\\left(x_{t,\\,i} - \\mu_{t-1,i}\\right)\\left(r_{m,t} - \\mu_{m,t-1}\\right)\\,, \\\\
V_{m,t} &= \\lambda V_{m,t-1} + (1 - \\lambda)\\left(r_{m,t} - \\mu_{m,t-1}\\right)^2\\,, \\\\
\\mu_{t,i} &= \\lambda \\mu_{t-1,i} + (1 - \\lambda)\\, x_{t,\\,i}\\,, \\\\
\\mu_{m,t} &= \\lambda \\mu_{m,t-1} + (1 - \\lambda)\\, r_{m,t}\\,.
\\end{align}
```

Where:

  - ``C_{t,i}``: Exponentially weighted covariance of asset ``i`` with the market.
  - ``\\mu_{t,i}``, ``\\mu_{m,t}``: Exponentially weighted means of asset ``i`` and of the market.
  - $(math_dict[:beta_ti_ewb])
  - $(math_dict[:V_mt_ewb])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:lambda_ew])
  - $(math_dict[:min_val_ewb])

Every state starts from zero, and each deviation uses the mean of the previous observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWBeta(; mcap::AbstractString = "market_cap", decay::Real, min_obs::Integer,
           agg_obs::Integer = 1, group::Option{<:AbstractString} = nothing,
           min_group_size::Integer = 5,
           bounds::Tuple{<:Real, <:Real} = (0.0, 1.0), min_val::Real = 1e-12,
           cache::Option{<:AbstractPartialFitState} = nothing) -> EWBeta

Keywords correspond to the struct's fields. `decay` and `min_obs` take no default, because their values depend on the data frequency. [`EWMarketBeta`](@ref) takes a half-life instead, and converts it through [`half_life_decay`](@ref) and [`half_life_min_obs`](@ref).

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `agg_obs >= 1`.
  - `min_group_size >= 1`.
  - `0 <= bounds[1] <= bounds[2] <= 1`.
  - `min_val > 0`.

# Examples

```jldoctest
julia> EWBeta(; decay = 0.5, min_obs = 2)
EWBeta
            mcap ┼ String: "market_cap"
           decay ┼ Float64: 0.5
         min_obs ┼ Int64: 2
         agg_obs ┼ Int64: 1
           group ┼ nothing
  min_group_size ┼ Int64: 5
          bounds ┼ Tuple{Float64, Float64}: (0.0, 1.0)
         min_val ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`ew_beta_series`](@ref)
  - [`ew_beta_shrink`](@ref)
  - [`market_return_series`](@ref)

# References

  - $(ref_dict[:sharpe1964])
"""
@concrete struct EWBeta <: AbstractDescriptorEstimator
    """
    $(field_dict[:mcap])
    """
    mcap
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:agg_obs])
    """
    agg_obs
    """
    $(field_dict[:group_ewb])
    """
    group
    """
    $(field_dict[:min_group_size_ewb])
    """
    min_group_size
    """
    $(field_dict[:bounds_ewb])
    """
    bounds
    """
    $(field_dict[:min_val])
    """
    min_val
    """
    $(field_dict[:ew_beta_desc_cache])
    """
    cache
    function EWBeta(mcap::AbstractString, decay::Real, min_obs::Integer, agg_obs::Integer,
                    group::Option{<:AbstractString}, min_group_size::Integer,
                    bounds::Tuple{<:Real, <:Real}, min_val::Real,
                    cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_ew_agg_obs(agg_obs)
        assert_nonempty_gt0_finite_val(min_group_size, :min_group_size)
        assert_ew_shrinkage_bounds(bounds)
        assert_nonempty_gt0_finite_val(min_val, :min_val)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(agg_obs),
                   typeof(group), typeof(min_group_size), typeof(bounds), typeof(min_val),
                   typeof(cache)}(mcap, decay, min_obs, agg_obs, group, min_group_size,
                                  bounds, min_val, cache)
    end
end
function EWBeta(; mcap::AbstractString = "market_cap", decay::Real, min_obs::Integer,
                agg_obs::Integer = 1, group::Option{<:AbstractString} = nothing,
                min_group_size::Integer = 5, bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
                min_val::Real = 1e-12,
                cache::Option{<:AbstractPartialFitState} = nothing)::EWBeta
    return EWBeta(mcap, decay, min_obs, agg_obs, group, min_group_size, bounds, min_val,
                  cache)
end
"""
    ew_beta_output(group::Nothing, de::EWBeta, rd::ReturnsResult,
                   Ba::AbstractMatrix{<:Real}, Vm::AbstractVector{<:Real},
                   Xa::AbstractMatrix{<:Real}, rma::AbstractVector{<:Real},
                   b0::AbstractVector{<:Real}, st::EWBlockState) -> NamedTuple
    ew_beta_output(group::AbstractString, de::EWBeta, rd::ReturnsResult,
                   Ba::AbstractMatrix{<:Real}, Vm::AbstractVector{<:Real},
                   Xa::AbstractMatrix{<:Real}, rma::AbstractVector{<:Real},
                   b0::AbstractVector{<:Real}, st::EWBlockState) -> NamedTuple

Turn the raw betas of the recursion into the Descriptor of an [`EWBeta`](@ref).

Dispatch on the `group` slot selects the method. With no group, the raw betas are spread over the observations that they were aggregated from. With a group, each cross-section is shrunk first. A shrunk cross-section is computed again only at an observation that closes a window, and it holds its value between windows, so the shrinkage runs on the same clock as the recursion.

The windows of `Ba` follow the windows that the state `st` folded, and the observations of `rd` follow the observations of the window that `st` carries. The batch call starts from a new state, and a step of the carry fold from the carried one, so both run the same arithmetic.

# Algorithm

 1. With no group, spread `Ba` over the observations through [`ew_beta_expand`](@ref), and stop.
 2. Read the capitalisation `W` and the group labels `L` at every observation.
 3. Compute the residual variance `Vr` of every window through [`ew_beta_residual_variance`](@ref), from the residual variance of the state.
 4. Compute the effective sample size `en`, twice the half-life that `decay` implies.
 5. At each observation `t` that closes window `k`, with `k >= min_obs`, compute the estimation error variance `bev` of each beta of that window.
 6. Shrink the betas of window `k` through [`ew_beta_shrink`](@ref), with the labels and the weights of observation `t`. The result `s` holds its value until the next window closes. It starts from the shrunk beta of the state.
 7. Spread `Ba` over the observations through [`ew_beta_expand`](@ref), which gives the raw betas of the last closed window during the warm-up and `NaN` before the first window closes, and write `s` over every observation after the warm-up.

# Arguments

  - `group`: The `group` slot of the estimator.
  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `Ba`: The raw betas, `windows × assets`.
  - `Vm`: The market variance after each window.
  - `Xa`: The aggregated returns, `windows × assets`.
  - `rma`: The aggregated market return, one entry per window.
  - `b0`: The raw beta of the last window that the state folded, `NaN` before the first one.
  - `st`: The state before the step.

# Returns

  - `out::NamedTuple`: `D`, the Descriptor before the active mask is applied, `observations × assets`, and `v` and `s`, the residual variance and the shrunk beta after the last window, or `nothing` with no group.

# Related

  - [`EWBeta`](@ref)
  - [`descriptor`](@ref)
  - [`ew_beta_shrink`](@ref)
  - [`ew_beta_expand`](@ref)
  - [`cross_sectional_groups`](@ref)
"""
function ew_beta_output(::Nothing, de::EWBeta, rd::ReturnsResult,
                        Ba::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                        ::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                        b0::AbstractVector{<:Real}, st::EWBlockState)
    return (; D = ew_beta_expand(Ba, size(rd.X, 1), de.agg_obs, b0, size(st.X, 1)),
            v = nothing, s = nothing)
end
function ew_beta_output(group::AbstractString, de::EWBeta, rd::ReturnsResult,
                        Ba::AbstractMatrix{<:Real}, Vm::AbstractVector{<:Real},
                        Xa::AbstractMatrix{<:Real}, rma::AbstractVector{<:Real},
                        b0::AbstractVector{<:Real}, st::EWBlockState)
    W = Matrix(descriptor_field_values(rd, de.mcap))
    L = cross_sectional_groups(descriptor_asset_panel(rd), group)
    Tf = float_if_integer(eltype(Ba))
    T, N = size(W)
    k0 = st.st.t
    r0 = size(st.X, 1)
    v0 = something(st.v,
                   zeros(float_if_integer(promote_type(eltype(Xa), eltype(rma), eltype(Ba))),
                         N))
    Vr = ew_beta_residual_variance(Xa, rma, Ba, de.decay, de.min_obs, v0, b0, k0)
    en = 2 * decay_half_life(de.decay)
    agg_obs = de.agg_obs
    min_obs = de.min_obs
    B = ew_beta_expand(Ba, T, agg_obs, b0, r0)
    s = something(st.s, fill(Tf(NaN), N))
    for t in 1:T
        j = div(r0 + t, agg_obs)
        k = k0 + j
        if iszero((r0 + t) % agg_obs) && k >= min_obs
            bev = view(Vr, j, :) ./ (en * (Vm[j] + de.min_val))
            s = ew_beta_shrink(view(Ba, j, :), bev, view(L, t, :), view(W, t, :),
                               de.min_group_size, de.bounds, de.min_val)
        end
        if k >= min_obs
            B[t, :] = s
        end
    end
    return (; D = B, v = isempty(Vr) ? v0 : Vr[end, :], s = s)
end
"""
    descriptor(de::EWBeta, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted market beta Descriptor from a [`ReturnsResult`](@ref).

# Algorithm

 1. Build the market return `rm` through [`market_return_series`](@ref).
 2. Mask the returns into `X` through [`ew_active_returns`](@ref).
 3. Where `agg_obs` is greater than one, aggregate `X` and `rm` into `Xa` and `rma` through [`ew_agg_series`](@ref) and [`ew_agg_vector`](@ref).
 4. Run the recursion from a new state through [`ew_beta_series!`](@ref), for the raw betas `Ba` and the market variance `Vm` of every window.
 5. Shrink and spread `Ba` into the Descriptor `D` through [`ew_beta_output`](@ref), which dispatches on the `group` slot.
 6. Write `NaN` into the inactive cells of `D` through [`descriptor_active_fill!`](@ref).

[`ew_beta_fold`](@ref) holds these steps, and a step of the carry fold runs them from the carried state.

An asset whose return is missing at an observation keeps its last raw beta, and the market state still advances at that observation. With a group, the shrunk beta of that asset can still change, because the mean and the prior variance of its group change.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`market_return_series`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], pnl = pnl);

julia> descriptor(EWMarketBeta(; half_life = 1), rd)
3×2 Matrix{Float64}:
 0.6       1.2
 0.914432  0.982316
 1.00449   0.927219
```

# Related

  - [`EWBeta`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`ew_beta_series!`](@ref)
  - [`ew_beta_output`](@ref)
  - [`ew_beta_fold`](@ref)
  - [`market_return_series`](@ref)
"""
function descriptor(de::EWBeta, rd::ReturnsResult)::Matrix{<:Real}
    return ew_beta_fold(de, rd, nothing).D
end
"""
    EWMarketBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                 decay::Real = half_life_decay(half_life),
                 min_obs::Integer = half_life_min_obs(half_life), agg_obs::Integer = 1,
                 group::Option{<:AbstractString} = nothing, min_group_size::Integer = 5,
                 bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
                 min_val::Real = 1e-12) -> EWBeta

Exponentially weighted sensitivity of an asset to the market portfolio.

The value is the beta of [`EWBeta`](@ref) against the capitalisation-weighted market return, the systematic risk of the capital asset pricing model. An asset of beta two moves on average twice as far as the market. With daily observations, the default half-life of `60` is about one quarter of a year.

# Arguments

  - $(arg_dict[:mcap])
  - `half_life`: Half-life of the recursion, in observations. It sets the defaults of `decay` and `min_obs`.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:agg_obs])
  - $(arg_dict[:group_ewb])
  - $(arg_dict[:min_group_size_ewb])
  - $(arg_dict[:bounds_ewb])
  - $(arg_dict[:min_val])

# Returns

  - `de::EWBeta`: The estimator, with the half-life fixed.

# Examples

```jldoctest
julia> EWMarketBeta(; half_life = 2)
EWBeta
            mcap ┼ String: "market_cap"
           decay ┼ Float64: 0.7071067811865476
         min_obs ┼ Int64: 2
         agg_obs ┼ Int64: 1
           group ┼ nothing
  min_group_size ┼ Int64: 5
          bounds ┼ Tuple{Float64, Float64}: (0.0, 1.0)
         min_val ┴ Float64: 1.0e-12
```

# Related

  - [`EWBeta`](@ref)
  - [`descriptor`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`half_life_decay`](@ref)
"""
function EWMarketBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                      decay::Real = half_life_decay(half_life),
                      min_obs::Integer = half_life_min_obs(half_life), agg_obs::Integer = 1,
                      group::Option{<:AbstractString} = nothing,
                      min_group_size::Integer = 5,
                      bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
                      min_val::Real = 1e-12)::EWBeta
    return EWBeta(; mcap = mcap, decay = decay, min_obs = min_obs, agg_obs = agg_obs,
                  group = group, min_group_size = min_group_size, bounds = bounds,
                  min_val = min_val)
end

export EWBeta, EWMarketBeta
