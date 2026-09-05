"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Descriptor Estimator types.

A Descriptor Estimator produces one Descriptor: a per-asset value at every observation, computed from one or more Panel Fields of an Asset Panel. A point-in-time ratio of two fundamentals, the logarithm of a market capitalisation and the growth of a field over a lag are each one Descriptor. The estimator is configuration, so it names the Panel Fields it reads and holds no data.

All concrete types producing a Descriptor should be subtypes of `AbstractDescriptorEstimator`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractDescriptorEstimator` and implement the following methods:

## `descriptor`

  - [`descriptor(de::AbstractDescriptorEstimator, rd::ReturnsResult)`](@ref): Computes the Descriptor of a carrier.

### Arguments

  - `de`: The concrete subtype instance.
  - `rd`: The returns result that carries the Asset Panel.

### Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`, `NaN` wherever the active mask is `false`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`PanelFieldRatio`](@ref)
  - [`PanelFieldLog`](@ref)
  - [`Passthrough`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
  - [`EWVolatility`](@ref)
  - [`RollingLogReturn`](@ref)
  - [`RollingMax`](@ref)
  - [`AssetPanel`](@ref)
"""
abstract type AbstractDescriptorEstimator <: AbstractEstimator end
"""
    descriptor(de::AbstractDescriptorEstimator, rd::ReturnsResult) -> Matrix{<:Real}

Compute the Descriptor of a carrier.

This is the verb every Descriptor Estimator answers. It reads the Panel Fields the estimator names from the carrier's feature matrix `rd.Z`, through the field index of `rd.pnl`, and never from the column names of `rd.nz`. Returns are not a Panel Field, so a member that reads them reads `rd.X`. Every member follows two conventions: the value at an observation uses information up to and including that observation, and every cell where the active mask of the Asset Panel is `false` is `NaN`.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`panel_field_values`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor end
"""
    panel_field_values(rd::ReturnsResult, name::AbstractString) -> Matrix{<:Real}
    panel_field_values(rd::ReturnsResult,
                       terms::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Matrix{<:Real}

Read one numeric Panel Field, or a linear combination of numeric Panel Fields, out of a carrier.

This is the one route from a Panel Field's name to its values, and every Descriptor Estimator reads through it. A blank cell never reaches a carrier: [`asset_panel`](@ref) resolves each one to a fill value and records the resolution in the field's observed-mask column. The read undoes that resolution, so a cell the fill touched comes back as `NaN` and a Descriptor cannot mistake a fill value for data.

# Algorithm

 1. Look the Panel Field up by name through [`panel_field`](@ref), and copy its value column of `rd.Z` into a floating point matrix.
 2. When the Panel Field carries an observed-mask column, write `NaN` into every cell whose mask entry is zero.
 3. For a vector of `name => coefficient` pairs, read each named field the same way, and return the sum of the fields, each multiplied by its coefficient. A `NaN` in any term is a `NaN` in the sum.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: The Panel Field's name.
  - `terms`: The Panel Fields to combine, each paired with its coefficient. `[\"a\" => 1, \"b\" => -1]` reads `a - b`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - `name` names a Panel Field, and its kind is [`NumericPanelField`](@ref). Raises a `KeyError` or an `ArgumentError`.
  - `!isempty(terms)`. Raises an [`IsEmptyError`](@ref).

# Returns

  - `V::Matrix{<:Real}`: The values, `observations × assets`, `NaN` where the Panel Field was not observed.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"mcap\", vals = [1.0 2.0; NaN 4.0],
                                            alg = ForwardPanelFill(; val = 0.0)),
                          NumericPanelInput(; name = \"debt\", vals = [0.5 1.0; 1.5 2.0])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), res...);

julia> PortfolioOptimisers.panel_field_values(rd, \"mcap\")
2×2 Matrix{Float64}:
   1.0  2.0
 NaN    4.0

julia> PortfolioOptimisers.panel_field_values(rd, [\"mcap\" => 1, \"debt\" => 1])
2×2 Matrix{Float64}:
   1.5  3.0
 NaN    6.0
```

# Related

  - [`descriptor`](@ref)
  - [`panel_field`](@ref)
  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`NumericPanelField`](@ref)
"""
function panel_field_values(rd::ReturnsResult, name::AbstractString)::Matrix{<:Real}
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor reads its Panel Fields through the field index of an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl`, `nz` and `Z` that asset_panel returns."))
    f = panel_field(pnl, name)
    @argcheck(isa(f.kind, NumericPanelField),
              ArgumentError("a Descriptor reads one number per observation and asset, so the Panel Field \"$name\" must be a NumericPanelField, got a $(nameof(typeof(f.kind))), which occupies $(length(f.cols)) column(s) of the feature axis"))
    Z = rd.Z
    Tf = float(eltype(Z))
    V = Matrix{Tf}(view(Z, :, :, f.cols[1]))
    ocols = f.ocols
    if !isnothing(ocols)
        O = view(Z, :, :, ocols[1])
        for k in eachindex(V, O)
            if iszero(O[k])
                V[k] = Tf(NaN)
            end
        end
    end
    return V
end
function panel_field_values(rd::ReturnsResult,
                            terms::AbstractVector{<:Pair{<:AbstractString, <:Real}})::Matrix{<:Real}
    @argcheck(!isempty(terms),
              IsEmptyError("a Panel Field combination needs at least one `name => coefficient` term"))
    V = panel_field_values(rd, terms[1][1]) .* terms[1][2]
    for k in 2:length(terms)
        V .+= panel_field_values(rd, terms[k][1]) .* terms[k][2]
    end
    return V
end
"""
    descriptor_asset_panel(rd::ReturnsResult) -> AssetPanel

Read the Asset Panel a Descriptor needs out of a carrier.

[`panel_field_values`](@ref) reaches the panel through the name of a Panel Field, and every Descriptor that reads one meets its refusal. A Descriptor over the returns reads no Panel Field, so it takes this route to the same refusal and to the active mask [`descriptor_active_fill!`](@ref) needs.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel the carrier holds.

# Related

  - [`descriptor`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`panel_field_values`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_asset_panel(rd::ReturnsResult)::AssetPanel
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor is `NaN` wherever the active mask of an Asset Panel is `false`, and rd.pnl is nothing. Build the carrier with the `pnl`, `nz` and `Z` that asset_panel returns."))
    return pnl
end
"""
    assert_log_returns(X::AbstractMatrix{<:Real}) -> nothing

Check that every return that is not missing is greater than `-1`.

A Descriptor that compounds returns takes the logarithm of one plus each return. A return of `-1` is a total loss, and the logarithm is undefined below it, so the check refuses the whole matrix rather than write an infinity into one cell of the Descriptor. A missing return is a `NaN`, and it passes the check. Every Descriptor that reads `log1p(rd.X)` runs it, exponentially weighted and rolling alike.

# Arguments

  - `X`: The returns, `observations × assets`.

# Validation

  - Every entry of `X` that is not `NaN` is greater than `-1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`descriptor`](@ref)
  - [`EWMean`](@ref)
  - [`RollingLogReturn`](@ref)
"""
function assert_log_returns(X::AbstractMatrix{<:Real})::Nothing
    k = findfirst(x -> !isnan(x) && x <= -one(x), X)
    @argcheck(isnothing(k),
              DomainError(isnothing(k) ? NaN : X[k],
                          "a Descriptor over log returns takes the logarithm of one plus each return, so every return that is not missing must be greater than -1, and it is $(isnothing(k) ? NaN : X[k]) at observation $(isnothing(k) ? 0 : k[1]) for asset $(isnothing(k) ? 0 : k[2]). A return at or below -1 is a data error, so clean the input rather than pass it through."))
    return nothing
end
"""
    descriptor_active_fill!(D::AbstractMatrix{<:Real}, pnl::AssetPanel) -> nothing

Write `NaN` into every cell of a Descriptor where the active mask of the Asset Panel is `false`, in place.

Every Descriptor Estimator ends with this call, so the convention that an inactive cell is `NaN` is written once. An asset that is not listed at an observation has no Descriptor there, whatever its Panel Fields hold.

# Arguments

  - `D`: The Descriptor, `observations × assets`, changed in place.
  - `pnl`: The Asset Panel whose active mask is read.

# Validation

  - `size(D) == size(pnl.amsk)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`. `D` carries the filled Descriptor.

# Examples

```jldoctest
julia> pnl = AssetPanel(; pf = [PanelField(; name = \"a\", kind = NumericPanelField(), cols = [1])],
                        amsk = [true false; true true], emsk = [true false; true true]);

julia> D = [1.0 2.0; 3.0 4.0];

julia> PortfolioOptimisers.descriptor_active_fill!(D, pnl)

julia> D
2×2 Matrix{Float64}:
 1.0  NaN
 3.0    4.0
```

# Related

  - [`descriptor`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_active_fill!(D::AbstractMatrix{<:Real}, pnl::AssetPanel)::Nothing
    amsk = pnl.amsk
    @argcheck(size(D) == size(amsk),
              DimensionMismatch("a Descriptor is observations × assets, so it must match the active mask of the Asset Panel, got size(D) = $(size(D)) and size(pnl.amsk) = $(size(amsk))"))
    Tf = eltype(D)
    for k in CartesianIndices(D)
        if !amsk[k]
            D[k] = Tf(NaN)
        end
    end
    return nothing
end
"""
    positive_divide(a::Real, b::Real) -> Real

Divide `a` by `b` where `b` is strictly positive, and return `NaN` otherwise.

A ratio Descriptor is undefined where its denominator is zero, and it is meaningless where a quantity that is positive by construction, a market capitalisation or a total of assets, is negative. Both cases answer `NaN` rather than a number or an error, so one bad cell costs one cell of the Descriptor and not the whole fit. A `NaN` denominator compares `false` against zero, so it also answers `NaN`.

# Arguments

  - `a`: The numerator.
  - `b`: The denominator.

# Returns

  - `q::Real`: `a / b` when `b > 0`, `NaN` otherwise, in the floating point type of the quotient.

# Examples

```jldoctest
julia> PortfolioOptimisers.positive_divide(1.0, 4.0)
0.25

julia> PortfolioOptimisers.positive_divide(1.0, 0.0)
NaN

julia> PortfolioOptimisers.positive_divide(1.0, -2.0)
NaN
```

# Related

  - [`descriptor`](@ref)
  - [`PanelFieldRatio`](@ref)
"""
function positive_divide(a::Real, b::Real)::Real
    q = a / b
    return b > zero(b) ? q : oftype(q, NaN)
end

"""
    market_return_series(rd::ReturnsResult, mcap::AbstractString) -> Vector{<:Real}

Build the market return of every observation from a carrier.

This is the one market series every market-relative Descriptor reads, so the rule that defines it is stated once. The market return is the capitalisation-weighted mean of the returns over the estimation universe, and it is rebuilt from the Asset Panel by every member rather than supplied by the caller.

# Algorithm

 1. Read the capitalisation Panel Field through [`panel_field_values`](@ref), so a cell a fill touched comes back as `NaN`.
 2. At each observation, take the pairs where the estimation mask is `true` and where both the return and the weight are finite.
 3. Return the mean of those returns, weighted by the capitalisation.

A weight is not required to be positive: a negative capitalisation is a data error rather than a missing value, so it enters the sum as it stands. The total weight of an observation must be strictly positive, because a total of zero or below divides no mean.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - $(arg_dict[:mcap])

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The total weight of every observation is strictly positive. Raises an `ArgumentError` naming the observation.

# Returns

  - `rm::Vector{<:Real}`: The market return, one entry per observation.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"market_cap\", vals = [1.0 3.0; 2.0 2.0])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.3], res...);

julia> PortfolioOptimisers.market_return_series(rd, \"market_cap\")
2-element Vector{Float64}:
 0.17500000000000002
 0.09999999999999999
```

# Related

  - [`descriptor`](@ref)
  - [`ew_beta_series`](@ref)
  - [`panel_field_values`](@ref)
  - [`AssetPanel`](@ref)
"""
function market_return_series(rd::ReturnsResult, mcap::AbstractString)
    pnl = descriptor_asset_panel(rd)
    W = panel_field_values(rd, mcap)
    X = rd.X
    emsk = pnl.emsk
    Tf = float(promote_type(eltype(X), eltype(W)))
    rm = Vector{Tf}(undef, size(X, 1))
    for t in axes(X, 1)
        s = zero(Tf)
        w = zero(Tf)
        for i in axes(X, 2)
            x = X[t, i]
            m = W[t, i]
            if emsk[t, i] && isfinite(x) && isfinite(m)
                s += m * x
                w += m
            end
        end
        @argcheck(w > zero(w),
                  ArgumentError("the market return is the capitalisation-weighted mean of the returns over the estimation universe, so an observation needs at least one estimable asset whose return and whose \"$mcap\" are both finite, and whose weights total strictly more than zero. The total weight is $w at observation $t"))
        rm[t] = s / w
    end
    return rm
end
"""
    ew_beta_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real}, decay::Real,
                   min_obs::Integer, min_val::Real,
                   amsk::Option{<:AbstractMatrix{Bool}} = nothing) -> Tuple{Matrix{<:Real}, Vector{<:Real}}

Run the exponentially weighted market beta recursion down each column of a return matrix.

This is the recursion every market-relative Descriptor shares, so the two files that hold one call it rather than repeat it. Each asset carries its own state and its own count of valid observations, and the market carries one state for the whole panel.

# Mathematical definition

```math
\\begin{align}
d_{t} &= r_{m,t} - \\mu_{t-1}\\,, \\quad d_{t,i} = r_{t,i} - \\mu_{t-1,i}\\,, \\\\
V_{t} &= \\lambda V_{t-1} + (1 - \\lambda) d_{t}^2\\,, \\\\
C_{t,i} &= \\lambda C_{t-1,i} + (1 - \\lambda) d_{t,i} d_{t}\\,, \\\\
\\beta_{t,i} &= \\frac{C_{t,i}}{V_{t} + \\texttt{min\\_val}}\\,.
\\end{align}
```

Where:

  - ``r_{m,t}``: The market return at observation ``t``.
  - ``r_{t,i}``: The return of asset ``i`` at observation ``t``.
  - ``\\mu_{t}``, ``\\mu_{t,i}``: Exponentially weighted means of the market and of asset ``i``.
  - ``V_{t}``: Exponentially weighted variance of the market return.
  - ``C_{t,i}``: Exponentially weighted covariance of asset ``i`` with the market.
  - ``\\lambda``: The decay factor.

# Algorithm

 1. Take each deviation from the mean of the **previous** step, and update the mean after it. This is the exponentially weighted form of Welford's recursion, and it carries none of the downward bias that a deviation from the updated mean carries.
 2. Advance the market state at every observation, and an asset's state only where its return is valid.
 3. Write a fresh beta where the observation count of the panel and the valid count of the asset have both reached `min_obs`. An asset whose return is not valid keeps the beta it last held, and an asset that has never been ready holds `NaN`.
 4. Where `amsk` is given, [`ew_beta_reset!`](@ref) restarts the mean, the covariance and the count of an asset that turns inactive, and a valid return needs an active cell. Where `amsk` is `nothing`, no state ever restarts and a valid return is any finite return.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:min_val])
  - `amsk`: Optional active mask, `observations × assets`. It resets the state of an asset that turns inactive.

# Returns

  - `B::Matrix{<:Real}`: The betas, `observations × assets`, before any mask is applied.
  - `Vm::Vector{<:Real}`: The market variance after each observation.

# Examples

```jldoctest
julia> B, Vm = PortfolioOptimisers.ew_beta_series([0.1 0.2; -0.1 0.3; 0.05 -0.05],
                                                  [0.1, -0.05, 0.02], 0.5, 1, 1e-12);

julia> B
3×2 Matrix{Float64}:
 1.0       2.0
 1.33333  -0.666667
 1.4557   -1.26582
```

# Related

  - [`market_return_series`](@ref)
  - [`ew_beta_reset!`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`EWResidualVolatility`](@ref)
"""
function ew_beta_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real}, decay::Real,
                        min_obs::Integer, min_val::Real,
                        amsk::Option{<:AbstractMatrix{Bool}} = nothing)
    Tf = float(promote_type(eltype(X), eltype(rm)))
    T, N = size(X)
    B = fill(Tf(NaN), T, N)
    Vm = Vector{Tf}(undef, T)
    b = fill(Tf(NaN), N)
    mu = zeros(Tf, N)
    cv = zeros(Tf, N)
    n = zeros(Int, N)
    act = trues(N)
    mu_m = zero(Tf)
    var_m = zero(Tf)
    om = one(Tf) - decay
    for t in 1:T
        ew_beta_reset!(amsk, mu, cv, n, act, t)
        r = rm[t]
        dm = r - mu_m
        mu_m = decay * mu_m + om * r
        var_m = decay * var_m + om * dm * dm
        Vm[t] = var_m
        for i in 1:N
            x = X[t, i]
            if isfinite(x) && act[i]
                d = x - mu[i]
                mu[i] = decay * mu[i] + om * x
                cv[i] = decay * cv[i] + om * d * dm
                n[i] += 1
                if t >= min_obs && n[i] >= min_obs
                    b[i] = cv[i] / (var_m + min_val)
                end
            end
            B[t, i] = b[i]
        end
    end
    return B, Vm
end
"""
    ew_beta_reset!(amsk::Nothing, mu::AbstractVector{<:Real}, cv::AbstractVector{<:Real},
                   n::AbstractVector{<:Integer}, act::AbstractVector{Bool},
                   t::Integer) -> nothing
    ew_beta_reset!(amsk::AbstractMatrix{Bool}, mu::AbstractVector{<:Real},
                   cv::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                   act::AbstractVector{Bool}, t::Integer) -> nothing

Restart the state of every asset that turns inactive at one observation, in place.

The optional active mask of [`ew_beta_series`](@ref) is read here by dispatch rather than by a branch inside the recursion. With no mask no state ever restarts, and `act` stays `true` everywhere, which is what makes any finite return a valid one. With a mask, `act` carries that observation's activity, so the recursion tests one vector rather than the mask and the mask's absence.

# Arguments

  - `amsk`: The active mask, `observations × assets`, or `nothing`.
  - `mu`: Exponentially weighted mean of each asset, changed in place.
  - `cv`: Exponentially weighted covariance of each asset with the market, changed in place.
  - `n`: Count of the valid observations of each asset, changed in place.
  - `act`: Activity of each asset at the previous observation, changed in place.
  - `t`: The observation.

# Returns

  - `nothing`. The four vectors carry the state of observation `t`.

# Related

  - [`ew_beta_series`](@ref)
  - [`EWResidualVolatility`](@ref)
"""
function ew_beta_reset!(::Nothing, ::AbstractVector{<:Real}, ::AbstractVector{<:Real},
                        ::AbstractVector{<:Integer}, ::AbstractVector{Bool},
                        ::Integer)::Nothing
    return nothing
end
function ew_beta_reset!(amsk::AbstractMatrix{Bool}, mu::AbstractVector{<:Real},
                        cv::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                        act::AbstractVector{Bool}, t::Integer)::Nothing
    for i in eachindex(mu, cv, n, act)
        if act[i] && !amsk[t, i]
            mu[i] = zero(eltype(mu))
            cv[i] = zero(eltype(cv))
            n[i] = 0
        end
        act[i] = amsk[t, i]
    end
    return nothing
end

export descriptor
