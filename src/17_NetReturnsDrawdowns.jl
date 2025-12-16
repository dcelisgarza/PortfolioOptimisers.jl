"""
    calc_net_returns(w::VecNum, X::MatNum, args...)
    calc_net_returns(w::VecNum, X::MatNum, fees::Fees)

Compute the net portfolio returns. If `fees` is provided, it deducts the calculated fees from the gross returns.

Returns the portfolio returns as the product of the asset return matrix `X` and portfolio weights `w`.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (assets × periods).
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

Compute the per asset net portfolio returns. If `fees` is provided, it deducts the calculated fees from the gross returns.

Returns the portfolio returns as the product of the asset return matrix `X` and portfolio weights `w`.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (assets × periods).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `ret::Matrix{<:Number}`: Per asset portfolio net returns.

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

`cumulative_returns` calculates the cumulative returns for an array of asset or portfolio returns. By default, it computes simple cumulative returns using `cumsum`. If `compound` is `true`, it computes compounded cumulative returns using `cumprod(one(eltype(X)) .+ X)`.

# Arguments

  - `X`: Array of asset or portfolio returns (vector, matrix, or higher-dimensional).
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

`drawdowns` calculates the drawdowns for an array of asset or portfolio returns. By default, it computes drawdowns from cumulative returns using `cumulative_returns`. If `compound` is `true`, it computes compounded drawdowns. If `cX` is `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns first.

# Arguments

  - `X`: Array of asset or portfolio returns (vector, matrix, or higher-dimensional).
  - `cX`: If `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns from `X`.
  - `compound`: If `true`, computes compounded drawdowns; otherwise, computes simple drawdowns.
  - `dims`: Dimension along which to compute drawdowns.

# Returns

  - `dd::ArrNum`: Array of drawdowns, same shape as `X`.

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
