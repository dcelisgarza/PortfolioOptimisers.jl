"""
    calc_net_returns(w::VecNum, X::MatNum, args...)
    calc_net_returns(w::VecNum, X::MatNum, fees::Fees)

Compute the net portfolio returns. If `fees` is provided, it deducts the computed fees from the gross returns.

Returns the portfolio returns as the product of the asset return matrix `X` and portfolio weights `w`.

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

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `val::VecNum`: Portfolio net returns.

# Examples

```jldoctest
julia> calc_net_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_asset_returns`](@ref)
  - [`calc_fees`](@ref)
"""
function calc_net_returns(w::VecNum, X::MatNum, args...)
    return X * w
end
function calc_net_returns(w::VecNum, X::MatNum, fees::Fees)
    return X * w .- calc_fees(w, fees)
end
"""
    calc_net_asset_returns(w::VecNum, X::MatNum, args...)
    calc_net_asset_returns(w::VecNum, X::MatNum, fees::Fees)

Compute the per asset net portfolio returns. If `fees` is provided, it deducts the computed fees from the gross returns.

Returns the per asset portfolio returns as the product of the asset return matrix `X` and portfolio weights `w`.

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

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (assets × periods).
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
  - [`calc_net_returns`](@ref)
  - [`calc_fees`](@ref)
"""
function calc_net_asset_returns(w::VecNum, X::MatNum, args...)
    return X ⊙ transpose(w)
end
function calc_net_asset_returns(w::VecNum, X::MatNum, fees::Fees)
    return X ⊙ transpose(w) .- transpose(calc_asset_fees(w, fees))
end
"""
    cumulative_returns(X::ArrNum; compound::Bool = false, dims::Int = 1)

Compute simple or compounded cumulative returns along a specified dimension.

`cumulative_returns` computes the cumulative returns for an array of asset or portfolio returns. By default, it computes simple cumulative returns using `cumsum`. If `compound` is `true`, it computes compounded cumulative returns using `cumprod(one(eltype(X)) .+ X)`.

## Portfolio cumulative returns

```math
\\begin{align}
\\boldsymbol{CR}_{a}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\boldsymbol{CR}_{a,\\, j}\\right\\} \\\\
\\boldsymbol{CR}_{a,\\, j}(\\boldsymbol{X}) &= \\sum\\limits_{i=1}^{j} X_{i} \\\\
\\boldsymbol{CR}_{r}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\boldsymbol{CR}_{r,\\, j}\\right\\} \\\\
\\boldsymbol{CR}_{r,\\, j}(\\boldsymbol{X}) &= \\prod\\limits_{i=1}^{j} (1 + X_{i})
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\boldsymbol{CR}_{a,\\, j}(\\boldsymbol{X})``: Simple cumulative portfolio returns at period `j`.
  - ``\\boldsymbol{CR}_{r,\\, j}(\\boldsymbol{X})``: Compound cumulative portfolio returns at period `j`.
  - ``\\boldsymbol{CR}_{a}``: `T × 1` vector of simple cumulative portfolio returns.
  - ``\\boldsymbol{CR}_{r}``: `T × 1` vector of compound cumulative portfolio returns.

## Per asset portfolio cumulative returns

```math
\\begin{align}
\\mathbf{CR}_{a}(\\mathbf{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\mathbf{CR}_{a,\\, j}\\right\\} \\\\
\\mathbf{CR}_{a,\\, j}(\\mathbf{X}) &= \\sum\\limits_{i=1}^{j} \\boldsymbol{X}_{i} \\\\
\\mathbf{CR}_{r}(\\mathbf{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\mathbf{CR}_{r,\\, j}\\right\\} \\\\
\\mathbf{CR}_{r,\\, j}(\\mathbf{X}) &= \\prod\\limits_{i=1}^{j} (1 \\oplus \\boldsymbol{X}_{i}) \\odot \\boldsymbol{w}^{\\intercal}
\\end{align}
```

Where:

  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\boldsymbol{X}_{i}``: `1 × N` vector of asset returns at period `i`.
  - ``\\mathbf{CR}_{a,\\, j}(\\mathbf{X})``: `1 × N` vector of simple cumulative per asset portfolio returns at period `j`.
  - ``\\mathbf{CR}_{r,\\, j}(\\mathbf{X})``: `1 × N` vector of compound cumulative per asset portfolio returns at period `j`.
  - ``\\mathbf{CR}_{a}``: `T × N` (observations × assets) matrix of simple cumulative per asset portfolio returns.
  - ``\\mathbf{CR}_{r}``: `T × N` (observations × assets) matrix of compound cumulative per asset portfolio returns.
  - ``\\oplus``: Elementwise (Hadamard) addition.
  - ``\\odot``: Elementwise (Hadamard) multiplication.

# Arguments

  - `X`: Array of asset or portfolio returns (vector or matrix).
  - `compound`: If `true`, computes compounded cumulative returns; otherwise, computes simple cumulative returns.
  - `dims`: Dimension along which to compute cumulative returns.

# Returns

  - `ret::ArrNum`: Array of cumulative returns, same shape as `X`.

# Examples

```jldoctest
julia> cumulative_returns([0.01, 0.02, -0.01])
3-element Vector{Float64}:
 0.01
 0.03
 0.02

julia> cumulative_returns([0.01, 0.02, -0.01]; compound = true)
3-element Vector{Float64}:
 1.01
 1.0302
 1.019898
```

# Related

  - [`ArrNum`](@ref)
  - [`drawdowns`](@ref)
"""
function cumulative_returns(X::ArrNum; compound::Bool = false, dims::Int = 1)
    return if compound
        cumprod(one(eltype(X)) .+ X; dims = dims)
    else
        cumsum(X; dims = dims)
    end
end
"""
    drawdowns(X::ArrNum; cX::Bool = false, compound::Bool = false, dims::Int = 1)

Compute simple or compounded drawdowns along a specified dimension.

```math
\\begin{align}
\\boldsymbol{DD_{a}}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T] \\, |\\, \\mathrm{DD_{a}}(\\boldsymbol{X},\\, j)\\right\\}\\\\
\\DD_{a}(\\boldsymbol{X},\\, j) &= \\underset{t \\in [1,\\, j]}{\\max}\\left( \\sum\\limits_{i=1}^{t} X_{i} \\right) - \\sum\\limits_{i=1}^{j} X_{i}\\\\
\\boldsymbol{DD_{r}}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T] \\, |\\, \\mathrm{DD_{r}}(\\boldsymbol{X},\\, j)\\right\\}\\\\
\\DD_{r}(\\boldsymbol{X},\\, j) &= \\underset{t \\in [1,\\, j]}{\\max}\\left( \\prod\\limits_{i=1}^{t}\\left(1 + X_{i}\\right) \\right) - \\prod\\limits_{i=1}^{j}\\left(1 + X_{i}\\right)
\\end{align}
```

Where:

  - ``\\boldsymbol{DD_{a}}(\\boldsymbol{X})``: `T × 1` vector of simple drawdowns.
  - ``\\DD_{a}(\\boldsymbol{X},\\, j)``: Simple drawdown at period `j`.
  - ``\\boldsymbol{DD_{r}}(\\boldsymbol{X})``: `T × 1` vector of compound drawdowns.
  - ``\\DD_{r}(\\boldsymbol{X},\\, j)``: Compound drawdown at period `j`.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.

# Returns

  - `dd::ArrNum`: Array of drawdowns, same shape as `X`.

# Details

`drawdowns` computes the drawdowns for an array of asset or portfolio returns. By default, it computes drawdowns from cumulative returns using `cumulative_returns`. If `compound` is `true`, it computes compounded drawdowns. If `cX` is `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns first.

# Examples

```jldoctest
julia> drawdowns([0.01, 0.02, -0.01])
3-element Vector{Float64}:
  0.0
  0.0
 -0.009999999999999998

julia> drawdowns([0.01, 0.02, -0.01]; compound = true)
3-element Vector{Float64}:
  0.0
  0.0
 -0.010000000000000009
```

# Related

  - [`ArrNum`](@ref)
  - [`cumulative_returns`](@ref)
"""
function drawdowns(X::ArrNum; cX::Bool = false, compound::Bool = false, dims::Int = 1)
    cX = !cX ? cumulative_returns(X; compound = compound, dims = dims) : X
    if compound
        return cX ./ accumulate(max, cX; dims = dims) .- one(eltype(X))
    else
        return cX - accumulate(max, cX; dims = dims)
    end
    return nothing
end

export calc_net_returns, calc_net_asset_returns, cumulative_returns, drawdowns
