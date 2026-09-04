"""
    calc_net_returns(w::VecNum, X::MatNum, args...)
    calc_net_returns(w::VecNum, X::MatNum, fees::Fees)
    calc_net_returns(w::VecVecNum, X::MatNum, args...)

Compute the net portfolio returns. If `fees` is not `nothing`, it deducts the computed fees from the gross returns.

The fee is one scalar and it is charged in **every** period. [`calc_fees`](@ref) contracts the whole weight vector into a single number, and that number is subtracted from every row of ``\\mathbf{X}\\boldsymbol{w}``, so a `T`-row matrix charges it `T` times. On a three-period matrix with ``\\boldsymbol{w} = [0.6,\\, -0.4,\\, 0,\\, 0.25]`` and `l = 0.002`, `s = 0.003`, `fl = 0.01`, `fs = 0.02`, the fee measured `0.0429`, and each of the three net returns sat `0.0429` below its gross value. That is why the [`Fees`](@ref) rates must be stated per period of `X`.

The per asset returns sum to this series. `vec(sum(calc_net_asset_returns(w, X, fees); dims = 2))` reproduces `calc_net_returns(w, X, fees)`, because [`calc_asset_fees`](@ref) splits over the assets what [`calc_fees`](@ref) contracts into a scalar. The two sides add in a different order, so the identity holds to rounding and not to `==`: on the weights above the largest difference measured `6.9e-18`.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{R}(\\mathbf{X},\\, \\boldsymbol{w}) &= \\mathbf{X} \\boldsymbol{w} \\ominus F_{\\text{t}}(\\boldsymbol{w})
\\end{align}
```

Where:

  - ``\\boldsymbol{R}(\\mathbf{X},\\, \\boldsymbol{w})``: `T × 1` vector of portfolio net returns.
  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\boldsymbol{w}``: `N × 1` vector of portfolio weights.
  - ``F_{\\text{t}}(\\boldsymbol{w})``: Total fees computed using [`calc_fees`](@ref).
  - ``\\ominus``: Elementwise (Hadamard) subtraction.

# Algorithm

 1. Contract `X` with `w`, giving `X * w`, the `T × 1` gross portfolio return series.
 2. On the `args...` method, return that series unchanged. The method reads none of its trailing arguments, so a `nothing` `fees` reaches it and charges nothing rather than charging a zero fee.
 3. On the `fees::Fees` method, compute the one scalar `calc_fees(w, fees)` and subtract it from every entry of the series.
 4. On the `w::VecVecNum` method, apply steps 1 to 3 to each weight vector `wi` of `w`, and collect one return series per weight vector.

# Arguments

  - `w`: Portfolio weights, or a vector of portfolio weight vectors.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `val::VecNum`: Portfolio net returns, for a `w::VecNum`.
  - `val::Vector{<:VecNum}`: One net return series per weight vector, for a `w::VecVecNum`.

# Examples

```jldoctest
julia> calc_net_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_asset_returns`](@ref): The per asset split of this series. Its rows sum to this one.
  - [`calc_fees`](@ref): Computes the one scalar that step 3 subtracts.
  - [`Fees`](@ref)
"""
function calc_net_returns(w::VecNum, X::MatNum, args...)
    return X * w
end
function calc_net_returns(w::VecNum, X::MatNum, fees::Fees)
    return X * w .- calc_fees(w, fees)
end
function calc_net_returns(w::VecVecNum, X::MatNum, args...)
    return [calc_net_returns(wi, X, args...) for wi in w]
end
"""
    calc_net_asset_returns(w::VecNum, X::MatNum, args...)
    calc_net_asset_returns(w::VecNum, X::MatNum, fees::Fees)

Compute the per asset net portfolio returns. If `fees` is not `nothing`, it deducts the computed fees from the gross returns.

The rows sum to the portfolio series. `vec(sum(calc_net_asset_returns(w, X, fees); dims = 2))` reproduces [`calc_net_returns(w, X, fees)`](@ref), because [`calc_asset_fees`](@ref) splits over the assets what [`calc_fees`](@ref) contracts into a scalar. The two sides add in a different order, so the identity holds to rounding and not to `==`: on ``\\boldsymbol{w} = [0.6,\\, -0.4,\\, 0,\\, 0.25]`` with all four rate fields set, the largest difference measured `6.9e-18`.

Each per asset fee is charged in **every** period, as it is for [`calc_net_returns`](@ref). The `N × 1` fee vector is subtracted from every row of ``\\mathbf{X} \\odot \\boldsymbol{w}^{\\intercal}``, so a `T`-row matrix charges it `T` times.

# Mathematical definition

```math
\\begin{align}
\\mathbf{R}(\\mathbf{X},\\, \\boldsymbol{w}) &= \\mathbf{X} \\odot \\boldsymbol{w}^{\\intercal} \\ominus \\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w})^{\\intercal}
\\end{align}
```

Where:

  - ``\\mathbf{R}(\\mathbf{X},\\, \\boldsymbol{w})``: `T × N` matrix of per asset portfolio net returns.
  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\boldsymbol{w}``: `N × 1` vector of portfolio weights.
  - ``\\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w})``: `N × 1` per asset vector of total portfolio fees computed using [`calc_fees`](@ref).
  - ``\\odot``: Elementwise (Hadamard) multiplication.
  - ``\\ominus``: Elementwise (Hadamard) subtraction.

# Algorithm

 1. Scale each column of `X` by its weight, giving `X ⊙ transpose(w)`, the `T × N` matrix of gross per asset contributions.
 2. On the `args...` method, return that matrix unchanged. The method reads none of its trailing arguments, so a `nothing` `fees` reaches it and charges nothing rather than charging a zero fee.
 3. On the `fees::Fees` method, compute the `N × 1` vector `calc_asset_fees(w, fees)` and subtract its transpose from every row of the matrix.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `ret::MatNum`: Per asset portfolio net returns.

# Examples

```jldoctest
julia> calc_net_asset_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2×2 Matrix{Float64}:
 0.005  0.01
 0.015  0.02
```

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_returns`](@ref): The portfolio series this matrix sums to along `dims = 2`.
  - [`calc_asset_fees`](@ref): Computes the per asset vector that step 3 subtracts.
  - [`calc_fees`](@ref)
  - [`Fees`](@ref)
"""
function calc_net_asset_returns(w::VecNum, X::MatNum, args...)
    return X ⊙ transpose(w)
end
function calc_net_asset_returns(w::VecNum, X::MatNum, fees::Fees)
    return X ⊙ transpose(w) .- transpose(calc_asset_fees(w, fees))
end
"""
$(DocStringExtensions.TYPEDEF)

Supertype for the algorithms that let a portfolio's weights drift with its own returns over the observations it is scored on.

A cross-validation scheme carries this family in its `wd` field. `nothing` scores every observation against the same weight vector, which is the library's original behaviour and stays its default. [`SelfFinancingDrift`](@ref) is the family's one leaf.

# Related

  - [`SelfFinancingDrift`](@ref)
  - [`weight_path`](@ref)
  - [`held_weights`](@ref)
  - [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractWeightDrift <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Grows each position at its own asset return and holds no trade in between, so the weights drift and the series is the wealth ratio of the drifted holdings.

The portfolio is self-financing over the window: no capital enters it and none leaves it, and the implicit cash position ``1 - \\sum_j w_j`` earns zero. The recursion therefore holds for a long-short book and for a partly invested one.

# Examples

```jldoctest
julia> SelfFinancingDrift()
SelfFinancingDrift()
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`weight_path`](@ref)
  - [`held_weights`](@ref)
  - [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref)
"""
struct SelfFinancingDrift <: AbstractWeightDrift end
"""
    drift_position_values(wd::AbstractWeightDrift, w::VecNum, X::MatNum)

Compute the value of each position at each observation of a drifted window.

Each position starts at its weight and compounds at its own asset return, so row `t` holds what one unit of initial capital has become in each asset by the end of observation `t`. The matrix is the quantity every other drift verb reads, and it is a cumulative product, not a per step renormalisation.

# Mathematical definition

```math
\\begin{align}
P_{t j} &= w_{j} \\prod\\limits_{s=1}^{t} \\left(1 + X_{s j}\\right)
\\end{align}
```

Where:

  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``, per unit of initial capital.
  - ``X_{s j}``: Return of asset ``j`` at observation ``s``.
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Compound the asset returns with [`relative_cumulative_returns`](@ref), giving the `T × N` matrix of growth factors of each asset up to each observation.
 2. Scale column `j` of that matrix by `w[j]`, giving the position values.

# Arguments

  - `wd`: Weight drift algorithm.
  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).

# Returns

  - `P::MatNum`: `T × N` matrix of position values.

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`drift_wealth`](@ref): Contracts this matrix into the wealth of each observation.
  - [`relative_cumulative_returns`](@ref): The helper that step 1 reaches.
  - [`weight_path`](@ref)
"""
function drift_position_values(::AbstractWeightDrift, w::VecNum, X::MatNum)
    return relative_cumulative_returns(X) ⊙ transpose(w)
end
"""
    drift_wealth(P::MatNum, w::VecNum)

Compute the wealth of each observation of a drifted window from its position values.

The cash position is what the weights leave uninvested, `1 - sum(w)`, and it earns zero, so it enters every observation's wealth unchanged. A fully invested book holds none of it, and a levered or short book holds a negative one.

# Mathematical definition

```math
\\begin{align}
V_{t} &= \\sum\\limits_{j=1}^{N} P_{t j} + \\left(1 - \\sum\\limits_{j=1}^{N} w_{j}\\right)
\\end{align}
```

Where:

  - ``V_{t}``: Wealth at the end of observation ``t``, per unit of initial capital.
  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``.
  - $(math_dict[:w_port])
  - $(math_dict[:N])

# Algorithm

 1. Sum `P` along its asset axis, giving the invested wealth of each observation.
 2. Add the cash position `1 - sum(w)` to every entry.

# Arguments

  - `P`: `T × N` matrix of position values, from [`drift_position_values`](@ref).
  - `w`: Portfolio weights.

# Returns

  - `V::VecNum`: `T × 1` vector of wealth, one entry per observation.

# Related

  - [`drift_position_values`](@ref): The matrix this verb contracts.
  - [`drift_returns`](@ref): Reads this vector as a series of returns.
  - [`assert_positive_wealth`](@ref): The domain check this vector must pass.
  - [`weight_path`](@ref)
"""
function drift_wealth(P::MatNum, w::VecNum)
    return vec(sum(P; dims = 2)) .+ (one(eltype(P)) - sum(w))
end
"""
    drift_returns(V::VecNum)

Read a wealth vector as the return series of the drifted window.

The return of an observation is the ratio of its wealth to the wealth before it, and the wealth before the first observation is the initial capital of one. The series is therefore the wealth ratio of the drifted holdings, and it is **not** the dot product of the held weights with the asset returns: the two differ by up to 809 ulp, and the wealth ratio is what the reference arithmetic computes.

# Mathematical definition

```math
\\begin{align}
R_{t} &= \\frac{V_{t}}{V_{t-1}} - 1\\,, \\qquad V_{0} = 1
\\end{align}
```

Where:

  - ``R_{t}``: Return of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``, per unit of initial capital.
  - $(math_dict[:T])

# Algorithm

 1. Prepend the initial capital of one to `V`, and drop its last entry, giving the wealth before each observation.
 2. Divide `V` by that vector elementwise, and subtract one.

# Arguments

  - `V`: `T × 1` vector of wealth, from [`drift_wealth`](@ref).

# Returns

  - `R::VecNum`: `T × 1` vector of drifted returns.

# Related

  - [`drift_wealth`](@ref): The vector this verb reads.
  - [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref): Charges the fee against this series.
  - [`cumulative_returns`](@ref)
"""
function drift_returns(V::VecNum)
    return V ./ vcat(one(eltype(V)), @view(V[1:(end - 1)])) .- one(eltype(V))
end
"""
    non_positive_wealth_index(V::VecNum)

Find the first observation whose wealth is not positive.

A `NaN` wealth fails `> 0` and is found, so an exactly zero wealth and the non-finite values it makes downstream are both caught here.

# Algorithm

 1. Return the index of the first entry of `V` that does not satisfy `x > 0`, and `nothing` when every entry does.

# Arguments

  - `V`: `T × 1` vector of wealth, from [`drift_wealth`](@ref).

# Returns

  - `i::Option{<:Integer}`: Index of the first non-positive wealth, or `nothing`.

# Related

  - [`drift_wealth`](@ref)
  - [`assert_positive_wealth`](@ref): The check that raises on this index.
  - [`NonPositiveWealthError`](@ref)
  - [`Option`](@ref)
"""
function non_positive_wealth_index(V::VecNum)
    return findfirst(x -> !(x > zero(x)), V)
end
"""
    assert_positive_wealth(V::VecNum, obs = nothing, member = nothing)

Check that every observation of a drifted window has a positive wealth.

The check runs before any return is formed, so a ruined window gives no partial series. `obs` names the failing observation in the message: a vector of labels names it by its label, a vector of integers names it by its panel row, and `nothing` names it by its row inside the window.

# Algorithm

 1. Find the first non-positive wealth with [`non_positive_wealth_index`](@ref). Return `nothing` when there is none.
 2. Build the phrase that names the failing observation, reading `obs`.
 3. Raise a [`NonPositiveWealthError`](@ref) that states the condition, the failing wealth, that phrase, and the member when one is named.

# Arguments

  - `V`: `T × 1` vector of wealth, from [`drift_wealth`](@ref).
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`.
  - `member`: Index of the population member the wealth belongs to, or `nothing` for a single weight vector.

# Validation

  - `V`: `all(>(0), V)`, else a [`NonPositiveWealthError`](@ref) is raised.

# Returns

  - `nothing`.

# Related

  - [`non_positive_wealth_index`](@ref): The finder step 1 reaches.
  - [`NonPositiveWealthError`](@ref)
  - [`drift_wealth`](@ref)
  - [`weight_path`](@ref)
"""
function assert_positive_wealth(V::VecNum, obs = nothing, member = nothing)::Nothing
    i = non_positive_wealth_index(V)
    if isnothing(i)
        return nothing
    end
    where_at = if isnothing(obs)
        "row $(i) of the window"
    elseif isa(obs[i], Integer)
        "panel row $(obs[i])"
    else
        "observation $(obs[i])"
    end
    whose = isnothing(member) ? "the wealth" : "the wealth of member $(member)"
    return throw(NonPositiveWealthError("the drifted wealth must satisfy `all(>(0), wealth)`, but $(whose) is $(V[i]) at $(where_at)"))
end
"""
    weight_path(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)

Compute the weights held at each observation of a drifted window.

Row `t` holds the weights the book carries **through** observation `t`, so the first row is the target weights and every later row is the previous observation's position values deflated by its wealth. The rows and the deflated cash position sum to one at every observation.

# Mathematical definition

```math
\\begin{align}
U_{1 j} &= w_{j}\\\\
U_{t j} &= \\frac{P_{t-1,\\, j}}{V_{t-1}}\\,, \\qquad t > 1
\\end{align}
```

The equivalent closed form is the self-financing recursion, which a reader may recognise from the literature:

```math
\\begin{align}
\\boldsymbol{u}_{t+1} &= \\frac{\\boldsymbol{u}_{t} \\odot \\left(1 + \\boldsymbol{x}_{t}\\right)}{1 + \\boldsymbol{u}_{t} \\cdot \\boldsymbol{x}_{t}}
\\end{align}
```

The two forms agree in exact arithmetic and differ in floating point: the cumulative product is what the code computes, and the recursion was measured up to 824 ulp away from it.

Where:

  - ``U_{t j}``: Weight held in asset ``j`` through observation ``t``.
  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``.
  - ``\\boldsymbol{u}_{t}``: `N × 1` vector of weights held through observation ``t``.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - ``\\odot``: Elementwise (Hadamard) multiplication.

# Algorithm

 1. Compute the position values with [`drift_position_values`](@ref) and the wealth with [`drift_wealth`](@ref).
 2. Check the wealth with [`assert_positive_wealth`](@ref).
 3. Write the target weights into the first row.
 4. Write each earlier observation's position values, deflated by that observation's wealth, into the row after it.

# Arguments

  - `wd`: Weight drift algorithm.
  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`. Read only by the message of step 2.

# Validation

  - The wealth of every observation is positive, else a [`NonPositiveWealthError`](@ref) is raised.

# Returns

  - `U::MatNum`: `T × N` matrix of held weights, one row per observation.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_path(SelfFinancingDrift(), [0.5, 0.5], [0.1 -0.1; 0.2 0.0])
2×2 Matrix{Float64}:
 0.5   0.5
 0.55  0.45
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`held_weights`](@ref): The weights held after the last observation.
  - [`drift_position_values`](@ref)
  - [`drift_wealth`](@ref)
  - [`assert_positive_wealth`](@ref)
"""
function weight_path(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    U = similar(P, float(eltype(P)))
    U[1, :] .= w
    if size(P, 1) > 1
        U[2:end, :] .= @view(P[1:(end - 1), :]) ./ @view(V[1:(end - 1)])
    end
    return U
end
"""
    held_weights(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)

Compute the weights held after the last observation of a drifted window.

These are the weights a fund carries into the next rebalance, so they are the base a trade is measured against. They are one step beyond the last row of [`weight_path`](@ref), which holds the weights carried **through** the last observation.

# Mathematical definition

```math
\\begin{align}
h_{j} &= \\frac{P_{T j}}{V_{T}}
\\end{align}
```

Where:

  - ``h_{j}``: Weight held in asset ``j`` after the last observation.
  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Compute the position values with [`drift_position_values`](@ref) and the wealth with [`drift_wealth`](@ref).
 2. Check the wealth with [`assert_positive_wealth`](@ref).
 3. Divide the last row of the position values by the last wealth.

# Arguments

  - `wd`: Weight drift algorithm.
  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`. Read only by the message of step 2.

# Validation

  - The wealth of every observation is positive, else a [`NonPositiveWealthError`](@ref) is raised.

# Returns

  - `h::VecNum`: `N × 1` vector of held weights.

# Examples

```jldoctest
julia> PortfolioOptimisers.held_weights(SelfFinancingDrift(), [0.5, 0.5], [0.1 -0.1; 0.2 0.0])
2-element Vector{Float64}:
 0.5945945945945945
 0.4054054054054054
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`weight_path`](@ref): The weights held through each observation.
  - [`drift_position_values`](@ref)
  - [`drift_wealth`](@ref)
  - [`assert_positive_wealth`](@ref)
"""
function held_weights(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    return @view(P[end, :]) ./ V[end]
end
"""
    calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs = nothing)
    calc_net_returns(w::VecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs = nothing)
    calc_net_returns(w::VecNum, X::MatNum, fees, wd::Nothing, args...)
    calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::Nothing, args...)

Compute the net portfolio returns of a window, reading the weight drift `wd`.

A `nothing` `wd` scores every observation against the same weight vector and reproduces [`calc_net_returns(w::VecNum, X::MatNum, args...)`](@ref) exactly. An [`AbstractWeightDrift`](@ref) lets each position grow at its own return, and the series becomes the wealth ratio of the drifted holdings. The fee is charged as it is on the undrifted series: [`calc_fees`](@ref) contracts the target weights into one scalar, and that scalar is subtracted from every observation.

A vector of weight vectors is a population, and a ruined member does not stop the run. The member's series is filled with `NaN`, one warning names every member that fell, and the run raises only when no member survives. A single weight vector is a population of one, so it raises.

# Mathematical definition

```math
\\begin{align}
R_{t} &= \\frac{V_{t}}{V_{t-1}} - 1 - F_{\\text{t}}(\\boldsymbol{w})\\,, \\qquad V_{0} = 1\\\\
V_{t} &= \\sum\\limits_{j=1}^{N} w_{j} \\prod\\limits_{s=1}^{t} \\left(1 + X_{s j}\\right) + \\left(1 - \\sum\\limits_{j=1}^{N} w_{j}\\right)
\\end{align}
```

Where:

  - ``R_{t}``: Net return of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``, per unit of initial capital.
  - ``X_{s j}``: Return of asset ``j`` at observation ``s``.
  - ``F_{\\text{t}}(\\boldsymbol{w})``: Total fees computed using [`calc_fees`](@ref).
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. On a `nothing` `wd`, delegate to [`calc_net_returns(w::VecNum, X::MatNum, args...)`](@ref), which charges the fee and reads no drift.
 2. On an [`AbstractWeightDrift`](@ref), compute the position values with [`drift_position_values`](@ref) and the wealth with [`drift_wealth`](@ref).
 3. Check the wealth with [`assert_positive_wealth`](@ref).
 4. Read the wealth as a return series with [`drift_returns`](@ref), and subtract the one fee scalar from every observation.
 5. On a population, run steps 2 to 4 for each member. Fill a ruined member's series with `NaN`, warn once naming every member that fell, and raise when every member fell.

# Arguments

  - `w`: Portfolio weights, or a vector of portfolio weight vectors.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure, or `nothing`.
  - `wd`: Weight drift algorithm, or `nothing`.
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`. Read only by the message of step 3.
  - `args...`: Additional arguments (ignored).

# Validation

  - The wealth of every observation is positive, else a [`NonPositiveWealthError`](@ref) is raised. On a population, the raise happens only when no member survives.

# Returns

  - `val::VecNum`: Portfolio net returns, for a `w::VecNum`.
  - `val::Vector{<:VecNum}`: One net return series per weight vector, for a `w::VecVecNum`.

# Examples

```jldoctest
julia> calc_net_returns([0.5, 0.5], [0.1 -0.1; 0.2 0.0], nothing, SelfFinancingDrift())
2-element Vector{Float64}:
 0.0
 0.1100000000000001
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`calc_net_returns(w::VecNum, X::MatNum, args...)`](@ref): The undrifted series a `nothing` `wd` reaches.
  - [`drift_returns`](@ref)
  - [`assert_positive_wealth`](@ref)
  - [`NonPositiveWealthError`](@ref)
  - [`VecVecNum`](@ref)
  - [`Fees`](@ref)
"""
function calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift,
                          obs = nothing)
    Tr = float(promote_type(eltype(X), eltype(first(w))))
    ret = Vector{Vector{Tr}}(undef, length(w))
    ruined = Int[]
    for (i, wi) in pairs(w)
        P = drift_position_values(wd, wi, X)
        V = drift_wealth(P, wi)
        if isnothing(non_positive_wealth_index(V))
            ret[i] = drift_returns(V) .- calc_fees(wi, fees)
        else
            push!(ruined, i)
            ret[i] = fill(convert(Tr, NaN), size(X, 1))
        end
    end
    if length(ruined) == length(w)
        assert_positive_wealth(drift_wealth(drift_position_values(wd, first(w), X),
                                            first(w)), obs, 1)
    elseif !isempty(ruined)
        @warn "the drifted wealth of $(length(ruined)) of $(length(w)) population member(s) is not positive, so their series are `NaN` and their members are dropped: $(ruined)"
    end
    return ret
end
function calc_net_returns(w::VecNum, X::MatNum, fees, wd::AbstractWeightDrift,
                          obs = nothing)
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    return drift_returns(V) .- calc_fees(w, fees)
end
function calc_net_returns(w::VecNum, X::MatNum, fees, ::Nothing, args...)
    return calc_net_returns(w, X, fees)
end
function calc_net_returns(w::VecVecNum, X::MatNum, fees, ::Nothing, args...)
    return calc_net_returns(w, X, fees)
end
"""
    relative_cumulative_returns(X; dims = 1)

Compute the relative cumulative returns from a return matrix.

Internal helper that computes cumulative returns as `cumprod(1 .+ X; dims=dims)`, returning the cumulative portfolio value relative to the starting value.

The series is quoted against an initial capital of one, so an entry below one is a loss and an entry above one is a gain. [`relative_drawdown_arr`](@ref) reads that convention through its `init = one(eltype(X))`.

# Algorithm

 1. Add one to every entry of `X`, giving the per period growth factors.
 2. Take the running product of those factors along `dims`, giving the cumulative portfolio value relative to the starting value.

# Arguments

  - `X`: Return matrix.
  - $(arg_dict[:dims])

# Returns

  - `ret::ArrNum`: Relative cumulative return matrix, same shape as `X`.

# Related

  - [`ArrNum`](@ref)
  - [`absolute_cumulative_returns`](@ref): The additive sibling, whose series starts at zero.
  - [`cumulative_returns`](@ref): The public entry point that reaches this helper when `compound` is `true`.
  - [`relative_drawdown_arr`](@ref)
"""
function relative_cumulative_returns(X::ArrNum; dims::Int = 1)
    return cumprod(one(eltype(X)) .+ X; dims = dims)
end
"""
    absolute_cumulative_returns(X; dims = 1)

Compute the absolute cumulative returns from a return matrix.

Internal helper that computes `cumsum(X; dims=dims)`, returning the cumulative sum of portfolio returns.

The series is quoted against an initial capital of zero, so an entry below zero is a loss and an entry above zero is a gain. [`absolute_drawdown_arr`](@ref) reads that convention through its `init = zero(eltype(X))`.

# Algorithm

 1. Take the running sum of `X` along `dims`, giving the cumulative portfolio return.

# Arguments

  - `X`: Return matrix.
  - $(arg_dict[:dims])

# Returns

  - `ret::ArrNum`: Cumulative return matrix, same shape as `X`.

# Related

  - [`ArrNum`](@ref)
  - [`relative_cumulative_returns`](@ref): The multiplicative sibling, whose series starts at one.
  - [`cumulative_returns`](@ref): The public entry point that reaches this helper when `compound` is `false`.
  - [`absolute_drawdown_arr`](@ref)
"""
function absolute_cumulative_returns(X::ArrNum; dims::Int = 1)
    return cumsum(X; dims = dims)
end
"""
    cumulative_returns(X::ArrNum, compound::Bool = false; dims::Int = 1)

Compute simple or compounded cumulative returns along a specified dimension.

`cumulative_returns` computes the cumulative returns for an array of asset or portfolio returns. By default, it computes simple cumulative returns using `cumsum`. If `compound` is `true`, it computes compounded cumulative returns using `cumprod(one(eltype(X)) .+ X)`.

# Mathematical definition

## Portfolio cumulative returns

```math
\\begin{align}
\\boldsymbol{CR}_{a}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\boldsymbol{CR}_{a,\\, j}\\right\\} \\\\
CR_{a,\\, j}(\\boldsymbol{X}) &= \\sum\\limits_{i=1}^{j} X_{i} \\\\
\\boldsymbol{CR}_{r}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\boldsymbol{CR}_{r,\\, j}\\right\\} \\\\
CR_{r,\\, j}(\\boldsymbol{X}) &= \\prod\\limits_{i=1}^{j} (1 + X_{i})
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``CR_{a,\\, j}(\\boldsymbol{X})``: Simple cumulative portfolio returns at period `j`.
  - ``CR_{r,\\, j}(\\boldsymbol{X})``: Compound cumulative portfolio returns at period `j`.
  - ``\\boldsymbol{CR}_{a}``: `T × 1` vector of simple cumulative portfolio returns.
  - ``\\boldsymbol{CR}_{r}``: `T × 1` vector of compound cumulative portfolio returns.

## Per asset cumulative returns

The same definitions apply as above, but for each individual asset in the returns matrix ``\\mathbf{X}`` instead of the portfolio return ``\\mathbf{X} \\boldsymbol{w}``.

# Algorithm

 1. Read `compound`, which selects one of two helpers.
 2. When `compound` is `false`, reach [`absolute_cumulative_returns`](@ref), giving the running sum of `X` along `dims`.
 3. When `compound` is `true`, reach [`relative_cumulative_returns`](@ref), giving the running product of `one(eltype(X)) .+ X` along `dims`.

# Arguments

  - `X`: Array of asset or portfolio returns (vector or matrix).
  - `compound`: If `true`, computes compounded cumulative returns; otherwise, computes simple cumulative returns.
  - $(arg_dict[:dims])

# Returns

  - `ret::ArrNum`: Array of cumulative returns, same shape as `X`.

# Examples

```jldoctest
julia> cumulative_returns([0.01, 0.02, -0.01])
3-element Vector{Float64}:
 0.01
 0.03
 0.02

julia> cumulative_returns([0.01, 0.02, -0.01], true)
3-element Vector{Float64}:
 1.01
 1.0302
 1.019898
```

# Related

  - [`ArrNum`](@ref)
  - [`absolute_cumulative_returns`](@ref): The helper that step 2 reaches.
  - [`relative_cumulative_returns`](@ref): The helper that step 3 reaches.
  - [`drawdowns`](@ref)
"""
function cumulative_returns(X::ArrNum, compound::Bool = false; dims::Int = 1)
    return if !compound
        absolute_cumulative_returns(X; dims = dims)
    else
        relative_cumulative_returns(X; dims = dims)
    end
end
"""
    absolute_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1) -> ArrNum

Compute the absolute drawdown array for a matrix of cumulative (or raw) returns.

Each element represents the drawdown from the running peak along the specified dimension.

**The running peak starts at the initial capital, not at the first observation.** `accumulate(max, ...)` is seeded with `init = zero(eltype(X))`, which is the value an additive cumulative series starts from, so a series that is under water from the first period reports a negative drawdown there. `absolute_drawdown_arr([-0.1, 0.05])` returns `[-0.1, -0.05]` and not `[0.0, 0.0]`. That `init` is the whole convention, and this function holds one of its two definitions.

# Algorithm

 1. Read `cX`, which says whether `X` is already cumulative.
 2. When `cX` is `false`, reach [`absolute_cumulative_returns`](@ref) along `dims`, giving the cumulative series `cX`. When `cX` is `true`, take `X` as that series unchanged.
 3. Take the running maximum of `cX` along `dims` with `init = zero(eltype(X))`, giving the running peak. The `init` starts the peak at the initial capital of zero.
 4. Subtract the running peak from `cX`, giving the drawdown array.

# Arguments

  - `X::ArrNum`: Returns array (or cumulative-returns array if `cX = true`).
  - `cX::Bool = false`: If `true`, treat `X` as already cumulative returns.
  - $(arg_dict[:dims])

# Returns

  - `dd::ArrNum`: Drawdown array of the same shape as `X`. Every entry is zero or negative.

# Related

  - [`ArrNum`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_arr`](@ref): The compound sibling, seeded with `init = one(eltype(X))`.
  - [`drawdowns`](@ref): The public entry point that reaches this function when `compound` is `false`.
  - [`absolute_cumulative_returns`](@ref): The helper that step 2 reaches.
  - [`cumulative_returns`](@ref)
"""
function absolute_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1)
    cX = !cX ? absolute_cumulative_returns(X; dims = dims) : X
    return cX - accumulate(max, cX; dims = dims, init = zero(eltype(X)))
end
"""
    relative_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1) -> ArrNum

Compute the relative drawdown array for a matrix of cumulative (or raw) compounded returns.

Each element represents the relative drawdown from the running peak along the specified dimension.

**The running peak starts at the initial capital, not at the first observation.** `accumulate(max, ...)` is seeded with `init = one(eltype(X))`, which is the value a compound cumulative series starts from, so a series that is under water from the first period reports a negative drawdown there. `relative_drawdown_arr([-0.1, 0.05])` returns `[-0.09999999999999998, -0.05499999999999994]` and not `[0.0, 0.0]`. That `init` is the whole convention, and this function holds one of its two definitions.

The two conventions agree to first order on a small return. On `[-1e-6, 5e-7]` the additive and the compound answers differ by `5.0e-13`.

# Algorithm

 1. Read `cX`, which says whether `X` is already cumulative.
 2. When `cX` is `false`, reach [`relative_cumulative_returns`](@ref) along `dims`, giving the cumulative series `cX`. When `cX` is `true`, take `X` as that series unchanged.
 3. Take the running maximum of `cX` along `dims` with `init = one(eltype(X))`, giving the running peak. The `init` starts the peak at the initial capital of one.
 4. Divide `cX` by the running peak and subtract one, giving the relative drawdown array.

# Arguments

  - `X::ArrNum`: Returns array (or cumulative-returns array if `cX = true`).
  - `cX::Bool = false`: If `true`, treat `X` as already cumulative compounded returns.
  - $(arg_dict[:dims])

# Returns

  - `dd::ArrNum`: Relative drawdown array of the same shape as `X`. Every entry is zero or negative.

# Related

  - [`ArrNum`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`absolute_drawdown_arr`](@ref): The additive sibling, seeded with `init = zero(eltype(X))`.
  - [`drawdowns`](@ref): The public entry point that reaches this function when `compound` is `true`.
  - [`relative_cumulative_returns`](@ref): The helper that step 2 reaches.
  - [`cumulative_returns`](@ref)
"""
function relative_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1)
    cX = !cX ? relative_cumulative_returns(X; dims = dims) : X
    return cX ./ accumulate(max, cX; dims = dims, init = one(eltype(X))) .- one(eltype(X))
end
"""
    drawdowns(X::ArrNum, compound::Bool = false; cX::Bool = false, dims::Int = 1)

Compute simple or compounded drawdowns along a specified dimension.

**The running peak starts at the initial capital, not at the first observation**, so a series that is under water from the first period reports a negative drawdown there. `drawdowns([-0.1, 0.05])` returns `[-0.1, -0.05]` and not `[0.0, 0.0]`. `drawdowns` dispatches to [`absolute_drawdown_arr`](@ref) or to [`relative_drawdown_arr`](@ref), which hold the single definition of the peak: the `init` of their `accumulate(max, ...)` is the initial capital, `zero(eltype(X))` for the additive path and `one(eltype(X))` for the compound one.

The two paths agree to first order on a small return. On `[-1e-6, 5e-7]` the additive and the compound answers differ by `5.0e-13`.

# Mathematical definition

## Portfolio drawdowns

```math
\\begin{align}
\\boldsymbol{DD}_{a}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T] \\, |\\, \\mathrm{DD_{a}}(\\boldsymbol{X},\\, j)\\right\\}\\\\
DD_{a}(\\boldsymbol{X},\\, j) &= \\sum\\limits_{i=1}^{j} X_{i} - \\underset{t \\in [1,\\, j]}{\\max}\\left( \\sum\\limits_{i=1}^{t} X_{i} \\right)\\\\
\\boldsymbol{DD}_{r}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T] \\, |\\, \\mathrm{DD_{r}}(\\boldsymbol{X},\\, j)\\right\\}\\\\
DD_{r}(\\boldsymbol{X},\\, j) &= \\dfrac{\\prod\\limits_{i=1}^{j}\\left(1 + X_{i}\\right)}{\\underset{t \\in [1,\\, j]}{\\max}\\left( \\prod\\limits_{i=1}^{t}\\left(1 + X_{i}\\right) \\right)} - 1
\\end{align}
```

Where:

  - ``\\boldsymbol{DD}_{a}(\\boldsymbol{X})``: `T × 1` vector of simple drawdowns.
  - ``DD_{a}(\\boldsymbol{X},\\, j)``: Simple drawdown at period `j`.
  - ``\\boldsymbol{DD}_{r}(\\boldsymbol{X})``: `T × 1` vector of compound drawdowns.
  - ``DD_{r}(\\boldsymbol{X},\\, j)``: Compound drawdown at period `j`.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.

## Per asset portfolio drawdowns

The same definitions apply as above, but for each individual asset in the returns matrix ``\\mathbf{X}`` instead of the portfolio return ``\\mathbf{X} \\boldsymbol{w}``.

# Algorithm

 1. Read `compound`, which selects one of two builders.
 2. When `compound` is `false`, reach [`absolute_drawdown_arr`](@ref), forwarding `cX` and `dims`.
 3. When `compound` is `true`, reach [`relative_drawdown_arr`](@ref), forwarding `cX` and `dims`.
 4. The builder that step 2 or step 3 reaches cumulates `X` along `dims` first, unless `cX` is `true`, in which case it takes `X` as the cumulative series it already is.
 5. That builder then subtracts the running peak, seeded at the initial capital.

# Arguments

  - `X`: Array of asset or portfolio returns (vector or matrix).
  - `compound`: If `true`, computes compounded drawdowns; otherwise, computes simple drawdowns.
  - `cX`: If `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns first.
  - $(arg_dict[:dims])

# Returns

  - `dd::ArrNum`: Array of drawdowns, same shape as `X`. Every entry is zero or negative.

# Examples

```jldoctest
julia> drawdowns([0.01, 0.02, -0.01])
3-element Vector{Float64}:
  0.0
  0.0
 -0.009999999999999998

julia> drawdowns([0.01, 0.02, -0.01], true)
3-element Vector{Float64}:
  0.0
  0.0
 -0.010000000000000009

julia> drawdowns([-0.1, 0.05])
2-element Vector{Float64}:
 -0.1
 -0.05
```

# Related

  - [`ArrNum`](@ref)
  - [`cumulative_returns`](@ref)
  - [`absolute_drawdown_arr`](@ref): The builder that step 2 reaches, and one of the two homes of the running-peak convention.
  - [`relative_drawdown_arr`](@ref): The builder that step 3 reaches, and the other home of the running-peak convention.
"""
function drawdowns(X::ArrNum, compound::Bool = false; cX::Bool = false, dims::Int = 1)
    return if !compound
        absolute_drawdown_arr(X; cX = cX, dims = dims)
    else
        relative_drawdown_arr(X; cX = cX, dims = dims)
    end
end

export calc_net_returns, calc_net_asset_returns, cumulative_returns, drawdowns,
       SelfFinancingDrift
