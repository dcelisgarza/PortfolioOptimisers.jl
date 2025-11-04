"""
    struct FeesEstimator{T1, T2, T3, T4, T5, T6} <: AbstractEstimator
        tn::T1
        l::T2
        s::T3
        fl::T4
        fs::T5
        kwargs::T6
    end

Estimator for portfolio transaction fees constraints.

`FeesEstimator` specifies transaction fee constraints for each asset in a portfolio, including turnover fees, long/short proportional fees, and long/short fixed fees. Asset-specific fees can be provided via dictionaries, pairs, or vectors of pairs. Input validation ensures all fee inputs are non-empty and non-negative where applicable.

# Fields

  - `tn`: Turnover estimator or result.
  - `l`: Long proportional fees.
  - `s`: Short proportional fees.
  - `fl`: Long fixed fees.
  - `fs`: Short fixed fees.
  - `kwargs`: Named tuple of keyword arguments for rounding fixed fees calculation.

# Constructor

    FeesEstimator(; tn::Union{Nothing, <:TurnoverEstimator, <:Turnover} = nothing,
                  l::Union{Nothing, <:EstValType} = nothing,
                  s::Union{Nothing, <:EstValType} = nothing,
                  fl::Union{Nothing, <:EstValType} = nothing,
                  fs::Union{Nothing, <:EstValType} = nothing,
                  kwargs::NamedTuple = (; atol = 1e-8))

## Validation

  - `l`, `s`, `fl`, `fs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> FeesEstimator(; tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1), 0.0),
                     l = Dict("A" => 0.001, "B" => 0.002), s = ["A" => 0.001, "B" => 0.002],
                     fl = Dict("A" => 5.0), fs = ["B" => 10.0])
FeesEstimator
      tn ┼ TurnoverEstimator
         │         w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │       val ┼ Dict{String, Float64}: Dict("A" => 0.1)
         │   default ┴ Float64: 0.0
       l ┼ Dict{String, Float64}: Dict("B" => 0.002, "A" => 0.001)
       s ┼ Vector{Pair{String, Float64}}: ["A" => 0.001, "B" => 0.002]
      fl ┼ Dict{String, Float64}: Dict("A" => 5.0)
      fs ┼ Vector{Pair{String, Float64}}: ["B" => 10.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`Fees`](@ref)
  - [`fees_constraints`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`AbstractEstimator`](@ref)
"""
struct FeesEstimator{T1, T2, T3, T4, T5, T6} <: AbstractEstimator
    tn::T1
    l::T2
    s::T3
    fl::T4
    fs::T5
    kwargs::T6
    function FeesEstimator(tn::Union{Nothing, <:TurnoverEstimator, <:Turnover},
                           l::Union{Nothing, <:EstValType}, s::Union{Nothing, <:EstValType},
                           fl::Union{Nothing, <:EstValType},
                           fs::Union{Nothing, <:EstValType},
                           kwargs::NamedTuple = (; atol = 1e-8))
        assert_nonempty_nonneg_finite_val(l, :l)
        assert_nonempty_nonneg_finite_val(s, :s)
        assert_nonempty_nonneg_finite_val(fl, :fl)
        assert_nonempty_nonneg_finite_val(fs, :fs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs),
                   typeof(kwargs)}(tn, l, s, fl, fs, kwargs)
    end
end
"""
    fees_constraints(fees::FeesEstimator, sets::AssetSets; datatype::DataType = Float64,
                     strict::Bool = false)

Generate portfolio transaction fee constraints from a `FeesEstimator` and asset set.

`fees_constraints` constructs a [`Fees`](@ref) object representing transaction fee constraints for the assets in `sets`, using the specifications in `fees`. Supports asset-specific turnover, long/short proportional fees, and long/short fixed fees via dictionaries, pairs, or vectors of pairs, with flexible assignment and validation.

# Arguments

  - `fees`: [`FeesEstimator`](@ref) specifying turnover, proportional, and fixed fee values.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for fee values.
  - `strict`: If `true`, enforces strict matching between assets and fee values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `Fees`: Object containing turnover, proportional, and fixed fee values aligned with `sets`.

# Details

  - Fee values are extracted and mapped to assets using `estimator_to_val`.
  - If a fee value is missing for an asset, assigns zero unless `strict` is `true`.
  - Turnover constraints are generated using `turnover_constraints`.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> fees = FeesEstimator(; tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1), 0.0),
                            l = Dict("A" => 0.001, "B" => 0.002), s = ["A" => 0.001, "B" => 0.002],
                            fl = Dict("A" => 5.0), fs = ["B" => 10.0]);

julia> fees_constraints(fees, sets)
Fees
      tn ┼ Turnover
         │     w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │   val ┴ Vector{Float64}: [0.1, 0.0, 0.0]
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ Vector{Float64}: [0.001, 0.002, 0.0]
      fl ┼ Vector{Float64}: [5.0, 0.0, 0.0]
      fs ┼ Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
  - [`turnover_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function FeesEstimator(; tn::Union{Nothing, <:TurnoverEstimator, <:Turnover} = nothing,
                       l::Union{Nothing, <:EstValType} = nothing,
                       s::Union{Nothing, <:EstValType} = nothing,
                       fl::Union{Nothing, <:EstValType} = nothing,
                       fs::Union{Nothing, <:EstValType} = nothing,
                       kwargs::NamedTuple = (; atol = 1e-8))
    return FeesEstimator(tn, l, s, fl, fs, kwargs)
end
function fees_view(fees::FeesEstimator, i::NumVec)
    tn = turnover_view(fees.tn, i)
    return FeesEstimator(; tn = tn, l = fees.l, s = fees.s, fl = fees.fl, fs = fees.fs,
                         kwargs = fees.kwargs)
end
"""
    struct Fees{T1, T2, T3, T4, T5, T6} <: AbstractResult
        tn::T1
        l::T2
        s::T3
        fl::T4
        fs::T5
        kwargs::T6
    end

Container for portfolio transaction fee constraints.

`Fees` stores transaction fee constraints for each asset in a portfolio, including turnover fees, long/short proportional fees, and long/short fixed fees. Fee values can be specified as scalars (applied to all assets) or as vectors of per-asset values. Input validation ensures all fee values are non-negative and, if vectors, non-empty.

# Fields

  - `tn`: Turnover constraint result.
  - `l`: Long proportional fees.
  - `s`: Short proportional fees.
  - `fl`: Long fixed fees.
  - `fs`: Short fixed fees.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered negligible.

# Constructor

    Fees(; tn::Union{Nothing, <:Turnover} = nothing,
         l::Union{Nothing, <:Number, <:NumVec} = nothing,
         s::Union{Nothing, <:Number, <:NumVec} = nothing,
         fl::Union{Nothing, <:Number, <:NumVec} = nothing,
         fs::Union{Nothing, <:Number, <:NumVec} = nothing,
         kwargs::NamedTuple = (; atol = 1e-8))

## Validation

  - `l`, `s`, `fl`, `fs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> Fees(; tn = Turnover([0.2, 0.3, 0.5], [0.1, 0.0, 0.0]), l = [0.001, 0.002, 0.0],
            s = [0.001, 0.002, 0.0], fl = [5.0, 0.0, 0.0], fs = [0.0, 10.0, 0.0])
Fees
      tn ┼ Turnover
         │     w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │   val ┴ Vector{Float64}: [0.1, 0.0, 0.0]
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ Vector{Float64}: [0.001, 0.002, 0.0]
      fl ┼ Vector{Float64}: [5.0, 0.0, 0.0]
      fs ┼ Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`FeesEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`AbstractResult`](@ref)
  - [`fees_constraints`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
struct Fees{T1, T2, T3, T4, T5, T6} <: AbstractResult
    tn::T1
    l::T2
    s::T3
    fl::T4
    fs::T5
    kwargs::T6
    function Fees(tn::Union{Nothing, <:Turnover}, l::Union{Nothing, <:Number, <:NumVec},
                  s::Union{Nothing, <:Number, <:NumVec},
                  fl::Union{Nothing, <:Number, <:NumVec},
                  fs::Union{Nothing, <:Number, <:NumVec},
                  kwargs::NamedTuple = (; atol = 1e-8))
        assert_nonempty_nonneg_finite_val(l, :l)
        assert_nonempty_nonneg_finite_val(s, :s)
        assert_nonempty_nonneg_finite_val(fl, :fl)
        assert_nonempty_nonneg_finite_val(fs, :fs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs),
                   typeof(kwargs)}(tn, l, s, fl, fs, kwargs)
    end
end
function Fees(; tn::Union{Nothing, <:Turnover} = nothing,
              l::Union{Nothing, <:Number, <:NumVec} = nothing,
              s::Union{Nothing, <:Number, <:NumVec} = nothing,
              fl::Union{Nothing, <:Number, <:NumVec} = nothing,
              fs::Union{Nothing, <:Number, <:NumVec} = nothing,
              kwargs::NamedTuple = (; atol = 1e-8))
    return Fees(tn, l, s, fl, fs, kwargs)
end
"""
    fees_constraints(fees::FeesEstimator, sets::AssetSets; datatype::DataType = Float64,
                     strict::Bool = false)

Generate portfolio transaction fee constraints from a `FeesEstimator` and asset set.

`fees_constraints` constructs a [`Fees`](@ref) object representing transaction fee constraints for the assets in `sets`, using the specifications in `fees`. Supports asset-specific turnover, long/short proportional fees, and long/short fixed fees via dictionaries, pairs, or vectors of pairs, with flexible assignment and validation.

# Arguments

  - `fees`: [`FeesEstimator`](@ref) specifying turnover, proportional, and fixed fee values.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for fee values.
  - `strict`: If `true`, enforces strict matching between assets and fee values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `Fees`: Object containing turnover, proportional, and fixed fee values aligned with `sets`.

# Details

  - Fee values are extracted and mapped to assets using `estimator_to_val`.
  - If a fee value is missing for an asset, assigns zero unless `strict` is `true`.
  - Turnover constraints are generated using `turnover_constraints`.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> fees = FeesEstimator(; tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1), 0.0),
                            l = Dict("A" => 0.001, "B" => 0.002), s = ["A" => 0.001, "B" => 0.002],
                            fl = Dict("A" => 5.0), fs = ["B" => 10.0]);

julia> fees_constraints(fees, sets)
Fees
      tn ┼ Turnover
         │     w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │   val ┴ Vector{Float64}: [0.1, 0.0, 0.0]
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ Vector{Float64}: [0.001, 0.002, 0.0]
      fl ┼ Vector{Float64}: [5.0, 0.0, 0.0]
      fs ┼ Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
  - [`turnover_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function fees_constraints(fees::FeesEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)
    return Fees(; tn = turnover_constraints(fees.tn, sets; strict = strict),
                l = estimator_to_val(fees.l, sets, zero(datatype); strict = strict),
                s = estimator_to_val(fees.s, sets, zero(datatype); strict = strict),
                fl = estimator_to_val(fees.fl, sets, zero(datatype); strict = strict),
                fs = estimator_to_val(fees.fs, sets, zero(datatype); strict = strict))
end
"""
    fees_constraints(fees::Union{Nothing, <:Fees}, args...; kwargs...)

Propagate or pass through portfolio transaction fee constraints.

`fees_constraints` returns the input [`Fees`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed fee constraints or missing constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `fees`: An existing [`Fees`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `Fees` or `nothing`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> fees = Fees(; tn = Turnover([0.2, 0.3, 0.5], [0.1, 0.0, 0.0]), l = [0.001, 0.002, 0.0]);

julia> fees_constraints(fees)
Fees
      tn ┼ Turnover
         │     w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │   val ┴ Vector{Float64}: [0.1, 0.0, 0.0]
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ nothing
      fl ┼ nothing
      fs ┼ nothing
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)

julia> fees_constraints(nothing)

```

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
"""
function fees_constraints(fees::Union{Nothing, <:Fees}, args...; kwargs...)
    return fees
end
function fees_view(::Nothing, ::Any)
    return nothing
end
function fees_view(fees::Fees, i::NumVec)
    tn = turnover_view(fees.tn, i)
    l = nothing_scalar_array_view(fees.l, i)
    s = nothing_scalar_array_view(fees.s, i)
    fl = nothing_scalar_array_view(fees.fl, i)
    fs = nothing_scalar_array_view(fees.fs, i)
    return Fees(; tn = tn, l = l, s = s, fl = fl, fs = fs, kwargs = fees.kwargs)
end
function factory(fees::Fees, w::NumVec)
    return Fees(; tn = factory(fees.tn, w), l = fees.l, s = fees.s, fl = fees.fl,
                fs = fees.fs, kwargs = fees.kwargs)
end
"""
    calc_fees(w::NumVec, p::NumVec, ::Nothing, ::Function)
    calc_fees(w::NumVec, p::NumVec, fees::Number, op::Function)
    calc_fees(w::NumVec, p::NumVec, fees::NumVec, op::Function)

Compute the actual proportional fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `NumVec`: Vector of fee values per asset.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::Number`: Total actual proportional fee.

# Examples

```jldoctest
julia> calc_fees([0.1, 0.2], [100, 200], 0.01, .>=)
0.5
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::NumVec, p::NumVec, ::Nothing, ::Function)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::NumVec, p::NumVec, fees::Number, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot_scalar(fees * w[idx], p[idx])
end
function calc_fees(w::NumVec, p::NumVec, fees::NumVec, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot(fees[idx], w[idx] .* p[idx])
end
"""
    calc_fees(w::NumVec, p::NumVec, ::Nothing)
    calc_fees(w::NumVec, p::NumVec, tn::Turnover)

Compute the actual turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: single turnover fee applied to all assets.
      + `tn.val::NumVec`: vector of turnover fees per asset.

# Returns

  - `val::Number`: Actual turnover fee.

# Examples

```jldoctest
julia> calc_fees([0.1, 0.2], [100, 200], Turnover([0.0, 0.0], 0.01))
0.5
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::NumVec, p::NumVec, ::Nothing)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::NumVec, p::NumVec, tn::Turnover{<:Any, <:Number})
    return dot_scalar(tn.val * abs.(w - tn.w), p)
end
function calc_fees(w::NumVec, p::NumVec, tn::Turnover{<:Any, <:NumVec})
    return dot(tn.val, abs.(w - tn.w) .* p)
end
"""
    calc_fees(w::NumVec, p::NumVec, fees::Fees)

Compute total actual fees for portfolio weights and prices.

Sums actual proportional, fixed, and turnover fees for all assets.

# Arguments

  - `w`: Portfolio weights.
  - `p`: Asset prices.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::Number`: Total actual fees.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.01, 0.02], s = [0.01, 0.02], fl = [5.0, 0.0], fs = [0.0, 10.0]);

julia> calc_fees([0.1, -0.2], [100, 200], fees)
15.9
```

# Related

  - [`Fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::NumVec, p::NumVec, fees::Fees)
    fees_long = calc_fees(w, p, fees.l, .>=)
    fees_short = -calc_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_fees(w::NumVec, ::Nothing, ::Function)
    calc_fees(w::NumVec, fees::Number, op::Function)
    calc_fees(w::NumVec, fees::NumVec, op::Function)

Compute the proportional fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `NumVec`: Vector of fee values per asset.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::Number`: Total proportional fee.

# Examples

```jldoctest
julia> calc_fees([0.1, 0.2], 0.01, .>=)
0.003
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::NumVec, ::Nothing, ::Function)
    return zero(eltype(w))
end
function calc_fees(w::NumVec, fees::Number, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return sum(fees * w[idx])
end
function calc_fees(w::NumVec, fees::NumVec, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return dot(fees[idx], w[idx])
end
"""
    calc_fees(w::NumVec, ::Nothing)
    calc_fees(w::NumVec, tn::Turnover)

Compute the turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: single turnover fee applied to all assets.
      + `tn.val::NumVec`: vector of turnover fees per asset.

# Returns

  - `val::Number`: Turnover fee.

# Examples

```jldoctest
julia> calc_fees([0.1, 0.2], Turnover([0.0, 0.0], 0.01))
0.003
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::NumVec, ::Nothing)
    return zero(eltype(w))
end
function calc_fees(w::NumVec, tn::Turnover{<:Any, <:Number})
    return sum(tn.val * abs.(w - tn.w))
end
function calc_fees(w::NumVec, tn::Turnover{<:Any, <:NumVec})
    return dot(tn.val, abs.(w - tn.w))
end
"""
    calc_fixed_fees(w::NumVec, ::Nothing, kwargs::NamedTuple, ::Function)
    calc_fixed_fees(w::NumVec, fees::Number, kwargs::NamedTuple, op::Function)
    calc_fixed_fees(w::NumVec, fees::NumVec, kwargs::NamedTuple,
                    op::Function)

Compute the fixed portfolio fees for assets that have been allocated.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `NumVec`: Vector of fee values per asset.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered negligible.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::Number`: Total fixed fee.

# Examples

```jldoctest
julia> calc_fixed_fees([0.1, 0.2], 0.01, (; atol = 1e-6), .>=)
0.02
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fixed_fees(w::NumVec, ::Nothing, kwargs::NamedTuple, op::Function)
    return zero(eltype(w))
end
function calc_fixed_fees(w::NumVec, fees::Number, kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return fees * sum(idx2)
end
function calc_fixed_fees(w::NumVec, fees::NumVec, kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return sum(fees[idx1][idx2])
end
"""
    calc_fees(w::NumVec, fees::Fees)

Compute total fees for portfolio weights and prices.

Sums proportional, fixed, and turnover fees for all assets.

# Arguments

  - `w`: Portfolio weights.
  - `p`: Asset prices.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::Number`: Total fees.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.01, 0.02], s = [0.01, 0.02], fl = [5.0, 0.0], fs = [0.0, 10.0]);

julia> calc_fees([0.1, -0.2], fees)
15.004999999999999
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::NumVec, fees::Fees)
    fees_long = calc_fees(w, fees.l, .>=)
    fees_short = -calc_fees(w, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_asset_fees(w::NumVec, p::NumVec, ::Nothing, ::Function)
    calc_asset_fees(w::NumVec, p::NumVec, fees::Number, op::Function)
    calc_asset_fees(w::NumVec, p::NumVec, fees::NumVec,
                    op::Function)

Compute the actual proportional per asset fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `NumVec`: Vector of fee values per asset.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::NumVec`: Total actual proportional per asset fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], [100, 200], 0.01, .>=)
2-element Vector{Float64}:
 0.1
 0.4
```

# Related

  - [`Fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::NumVec, p::NumVec, ::Nothing, ::Function)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::NumVec, p::NumVec, fees::Number, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] = fees * w[idx] ⊙ p[idx]
    return fees_w
end
function calc_asset_fees(w::NumVec, p::NumVec, fees::NumVec, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] = fees[idx] ⊙ w[idx] ⊙ p[idx]
    return fees_w
end
"""
    calc_asset_fees(w::NumVec, p::NumVec, ::Nothing)
    calc_asset_fees(w::NumVec, p::NumVec, tn::Turnover)

Compute the actual per asset turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: single turnover fee applied to all assets.
      + `tn.val::NumVec`: vector of turnover fees per asset.

# Returns

  - `val::Vector{<:Number}`: Actual per asset turnover fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], [100, 200], Turnover([0.0, 0.0], 0.01))
2-element Vector{Float64}:
 0.1
 0.4
```

# Related

  - [`Fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::NumVec, p::NumVec, ::Nothing)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::NumVec, p::NumVec, tn::Turnover{<:Any, <:Number})
    return tn.val * abs.(w - tn.w) ⊙ p
end
function calc_asset_fees(w::NumVec, p::NumVec, tn::Turnover{<:Any, <:NumVec})
    return tn.val ⊙ abs.(w - tn.w) ⊙ p
end
"""
    calc_asset_fees(w::NumVec, p::NumVec, fees::Fees)

Compute total actual per asset fees for portfolio weights and prices.

Sums actual proportional, fixed, and turnover fees for all assets.

# Arguments

  - `w`: Portfolio weights.
  - `p`: Asset prices.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::Vector{<:Number}`: Total actual per asset fees.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.01, 0.02], s = [0.01, 0.02], fl = [5.0, 0.0], fs = [0.0, 10.0]);

julia> calc_asset_fees([0.1, -0.2], [100, 200], fees)
2-element Vector{Float64}:
  5.1
 10.8
```

# Related

  - [`Fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::NumVec, p::NumVec, fees::Fees)
    fees_long = calc_asset_fees(w, p, fees.l, .>=)
    fees_short = -calc_asset_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_asset_fees(w::NumVec, ::Nothing, ::Function)
    calc_asset_fees(w::NumVec, fees::Number, op::Function)
    calc_asset_fees(w::NumVec, fees::NumVec, op::Function)

Compute the proportional per asset fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `NumVec`: Vector of fee values per asset.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::Vector{<:Number}`: Total proportional per asset fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], 0.01, .>=)
2-element Vector{Float64}:
 0.001
 0.002
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::NumVec, ::Nothing, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::NumVec, fees::Number, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] = fees * w[idx]
    return fees_w
end
function calc_asset_fees(w::NumVec, fees::NumVec, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] = fees[idx] ⊙ w[idx]
    return fees_w
end
"""
    calc_asset_fees(w::NumVec, ::Nothing)
    calc_asset_fees(w::NumVec, tn::Turnover)

Compute the per asset turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: single turnover fee applied to all assets.
      + `tn.val::NumVec`: vector of turnover fees per asset.

# Returns

  - `val::Vector{<:Number}`: Per asset turnover fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], Turnover([0.0, 0.0], 0.01))
2-element Vector{Float64}:
 0.001
 0.002
```

# Related

  - [`Fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::NumVec, ::Nothing)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::NumVec, tn::Turnover{<:Any, <:Number})
    return tn.val * abs.(w - tn.w)
end
function calc_asset_fees(w::NumVec, tn::Turnover{<:Any, <:NumVec})
    return tn.val ⊙ abs.(w - tn.w)
end
"""
    calc_asset_fixed_fees(w::NumVec, ::Nothing, kwargs::NamedTuple, ::Function)
    calc_asset_fixed_fees(w::NumVec, fees::Number, kwargs::NamedTuple, op::Function)
    calc_asset_fixed_fees(w::NumVec, fees::NumVec, kwargs::NamedTuple,
                          op::Function)

Compute the per asset fixed portfolio fees for assets that have been allocated.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `NumVec`: Vector of fee values per asset.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered negligible.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::Vector{<:Number}`: Total per asset fixed fee.

# Examples

```jldoctest
julia> calc_asset_fixed_fees([0.1, 0.2], 0.01, (; atol = 1e-6), .>=)
2-element Vector{Float64}:
 0.01
 0.01
```

# Related

  - [`Fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fixed_fees(w::NumVec, ::Nothing, ::NamedTuple, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fixed_fees(w::NumVec, fees::Number, kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] = fees * idx2
    return fees_w
end
function calc_asset_fixed_fees(w::NumVec, fees::NumVec, kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] = fees[idx1][idx2]
    return fees_w
end
"""
    calc_asset_fees(w::NumVec, fees::Fees)

Compute total per asset fees for portfolio weights and prices.

Sums proportional, fixed, and turnover fees for all assets.

# Arguments

  - `w`: Portfolio weights.
  - `p`: Asset prices.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::Vector{<:Number}`: Total per asset fees.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.01, 0.02], s = [0.01, 0.02], fl = [5.0, 0.0], fs = [0.0, 10.0]);

julia> calc_asset_fees([0.1, -0.2], fees)
2-element Vector{Float64}:
  5.001
 10.004
```

# Related

  - [`Fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::NumVec, fees::Fees)
    fees_long = calc_asset_fees(w, fees.l, .>=)
    fees_short = -calc_asset_fees(w, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_net_returns(w::NumVec, X::NumMat, args...)
    calc_net_returns(w::NumVec, X::NumMat, fees::Fees)

Compute the net portfolio returns. If `fees` is provided, it deducts the calculated fees from the gross returns.

Returns the portfolio returns as the product of the asset return matrix `X` and portfolio weights `w`.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (assets × periods).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `Vector{<:Number}`: Portfolio net returns.

# Examples

```jldoctest
julia> calc_net_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`calc_net_asset_returns`](@ref)
  - [`calc_fees`](@ref)
"""
function calc_net_returns(w::NumVec, X::NumMat, args...)
    return X * w
end
function calc_net_returns(w::NumVec, X::NumMat, fees::Fees)
    return X * w .- calc_fees(w, fees)
end
"""
    calc_net_asset_returns(w::NumVec, X::NumMat, args...)
    calc_net_asset_returns(w::NumVec, X::NumMat, fees::Fees)

Compute the per asset net portfolio returns. If `fees` is provided, it deducts the calculated fees from the gross returns.

Returns the portfolio returns as the product of the asset return matrix `X` and portfolio weights `w`.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (assets × periods).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `Matrix{<:Number}`: Per asset portfolio net returns.

# Examples

```jldoctest
julia> calc_net_asset_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2×2 Matrix{Float64}:
 0.005  0.01
 0.015  0.02
```

# Related

  - [`calc_net_returns`](@ref)
  - [`calc_fees`](@ref)
"""
function calc_net_asset_returns(w::NumVec, X::NumMat, args...)
    return X ⊙ transpose(w)
end
function calc_net_asset_returns(w::NumVec, X::NumMat, fees::Fees)
    return X ⊙ transpose(w) .- transpose(calc_asset_fees(w, fees))
end
"""
    cumulative_returns(X::NumArr; compound::Bool = false, dims::Int = 1)

Compute simple or compounded cumulative returns along a specified dimension.

`cumulative_returns` calculates the cumulative returns for an array of asset or portfolio returns. By default, it computes simple cumulative returns using `cumsum`. If `compound` is `true`, it computes compounded cumulative returns using `cumprod(one(eltype(X)) .+ X)`.

# Arguments

  - `X`: Array of asset or portfolio returns (vector, matrix, or higher-dimensional).
  - `compound`: If `true`, computes compounded cumulative returns; otherwise, computes simple cumulative returns.
  - `dims`: Dimension along which to compute cumulative returns.

# Returns

  - `ret::NumArr`: Array of cumulative returns, same shape as `X`.

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

  - [`drawdowns`](@ref)
"""
function cumulative_returns(X::NumArr; compound::Bool = false, dims::Int = 1)
    return if compound
        cumprod(one(eltype(X)) .+ X; dims = dims)
    else
        cumsum(X; dims = dims)
    end
end
"""
    drawdowns(X::NumArr; cX::Bool = false, compound::Bool = false, dims::Int = 1)

Compute simple or compounded drawdowns along a specified dimension.

`drawdowns` calculates the drawdowns for an array of asset or portfolio returns. By default, it computes drawdowns from cumulative returns using `cumulative_returns`. If `compound` is `true`, it computes compounded drawdowns. If `cX` is `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns first.

# Arguments

  - `X`: Array of asset or portfolio returns (vector, matrix, or higher-dimensional).
  - `cX`: If `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns from `X`.
  - `compound`: If `true`, computes compounded drawdowns; otherwise, computes simple drawdowns.
  - `dims`: Dimension along which to compute drawdowns.

# Returns

  - `dd::NumArr`: Array of drawdowns, same shape as `X`.

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

  - [`cumulative_returns`](@ref)
"""
function drawdowns(X::NumArr; cX::Bool = false, compound::Bool = false, dims::Int = 1)
    cX = !cX ? cumulative_returns(X; compound = compound, dims = dims) : X
    if compound
        return cX ./ accumulate(max, cX; dims = dims) .- one(eltype(X))
    else
        return cX - accumulate(max, cX; dims = dims)
    end
    return nothing
end

export FeesEstimator, Fees, fees_constraints, calc_fees, calc_fixed_fees, calc_asset_fees,
       calc_asset_fixed_fees, calc_net_returns, calc_net_asset_returns, cumulative_returns,
       drawdowns
