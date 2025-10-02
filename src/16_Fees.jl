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

```julia
FeesEstimator(; tn::Union{Nothing, <:TurnoverEstimator, <:Turnover} = nothing,
              l::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                       <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
              s::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                       <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
              fl::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                        <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
              fs::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                        <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
              kwargs::NamedTuple = (; atol = 1e-8))
```

## Validation

  - If any fee input (`l`, `s`, `fl`, `fs`) is a dictionary or vector, it must be non-empty, finite, and non-negative.

# Examples

```jldoctest
julia> FeesEstimator(; tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1), 0.0),
                     l = Dict("A" => 0.001, "B" => 0.002), s = ["A" => 0.001, "B" => 0.002],
                     fl = Dict("A" => 5.0), fs = ["B" => 10.0])
FeesEstimator
      tn | TurnoverEstimator
         |         w | Vector{Float64}: [0.2, 0.3, 0.5]
         |       val | Dict{String, Float64}: Dict("A" => 0.1)
         |   default | Float64: 0.0
       l | Dict{String, Float64}: Dict("B" => 0.002, "A" => 0.001)
       s | Vector{Pair{String, Float64}}: ["A" => 0.001, "B" => 0.002]
      fl | Dict{String, Float64}: Dict("A" => 5.0)
      fs | Vector{Pair{String, Float64}}: ["B" => 10.0]
  kwargs | @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
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
                           l::Union{Nothing, <:AbstractDict,
                                    <:Pair{<:AbstractString, <:Real},
                                    <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           s::Union{Nothing, <:AbstractDict,
                                    <:Pair{<:AbstractString, <:Real},
                                    <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           fl::Union{Nothing, <:AbstractDict,
                                     <:Pair{<:AbstractString, <:Real},
                                     <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           fs::Union{Nothing, <:AbstractDict,
                                     <:Pair{<:AbstractString, <:Real},
                                     <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           kwargs::NamedTuple = (; atol = 1e-8))
        assert_nonneg_finite_val(l)
        assert_nonneg_finite_val(s)
        assert_nonneg_finite_val(fl)
        assert_nonneg_finite_val(fs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs),
                   typeof(kwargs)}(tn, l, s, fl, fs, kwargs)
    end
end
"""
    fees_constraints(fees::FeesEstimator, sets::AssetSets;
                     datatype::DataType = Float64, strict::Bool = false)

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
      tn | Turnover
         |     w | Vector{Float64}: [0.2, 0.3, 0.5]
         |   val | Vector{Float64}: [0.1, 0.0, 0.0]
       l | Vector{Float64}: [0.001, 0.002, 0.0]
       s | Vector{Float64}: [0.001, 0.002, 0.0]
      fl | Vector{Float64}: [5.0, 0.0, 0.0]
      fs | Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs | @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
  - [`turnover_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function FeesEstimator(; tn::Union{Nothing, <:TurnoverEstimator, <:Turnover} = nothing,
                       l::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       s::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       fl::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                 <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       fs::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                 <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       kwargs::NamedTuple = (; atol = 1e-8))
    return FeesEstimator(tn, l, s, fl, fs, kwargs)
end
function fees_view(fees::FeesEstimator, i::AbstractVector)
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
  - `kwargs`: Named tuple of keyword arguments for rounding fixed fees calculation.

# Constructor

```julia
Fees(; tn::Union{Nothing, <:Turnover} = nothing,
     l::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
     s::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
     fl::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
     fs::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
     kwargs::NamedTuple = (; atol = 1e-8))
```

## Validation

  - If any fee (`l`, `s`, `fl`, `fs`) is a scalar: must be finite and non-negative.
  - If any fee is a vector: must be non-empty, all elements finite, and non-negative.

# Examples

```jldoctest
julia> Fees(; tn = Turnover([0.2, 0.3, 0.5], [0.1, 0.0, 0.0]), l = [0.001, 0.002, 0.0],
            s = [0.001, 0.002, 0.0], fl = [5.0, 0.0, 0.0], fs = [0.0, 10.0, 0.0])
Fees
      tn | Turnover
         |     w | Vector{Float64}: [0.2, 0.3, 0.5]
         |   val | Vector{Float64}: [0.1, 0.0, 0.0]
       l | Vector{Float64}: [0.001, 0.002, 0.0]
       s | Vector{Float64}: [0.001, 0.002, 0.0]
      fl | Vector{Float64}: [5.0, 0.0, 0.0]
      fs | Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs | @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
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
    function Fees(tn::Union{Nothing, <:Turnover},
                  l::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                  s::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                  fl::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                  fs::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                  kwargs::NamedTuple = (; atol = 1e-8))
        assert_nonneg_finite_val(l)
        assert_nonneg_finite_val(s)
        assert_nonneg_finite_val(fl)
        assert_nonneg_finite_val(fs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs),
                   typeof(kwargs)}(tn, l, s, fl, fs, kwargs)
    end
end
function Fees(; tn::Union{Nothing, <:Turnover} = nothing,
              l::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              s::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fl::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fs::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              kwargs::NamedTuple = (; atol = 1e-8))
    return Fees(tn, l, s, fl, fs, kwargs)
end
"""
    fees_constraints(fees::FeesEstimator, sets::AssetSets;
                     datatype::DataType = Float64, strict::Bool = false)

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
      tn | Turnover
         |     w | Vector{Float64}: [0.2, 0.3, 0.5]
         |   val | Vector{Float64}: [0.1, 0.0, 0.0]
       l | Vector{Float64}: [0.001, 0.002, 0.0]
       s | Vector{Float64}: [0.001, 0.002, 0.0]
      fl | Vector{Float64}: [5.0, 0.0, 0.0]
      fs | Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs | @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
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
      tn | Turnover
         |     w | Vector{Float64}: [0.2, 0.3, 0.5]
         |   val | Vector{Float64}: [0.1, 0.0, 0.0]
       l | Vector{Float64}: [0.001, 0.002, 0.0]
       s | nothing
      fl | nothing
      fs | nothing
  kwargs | @NamedTuple{atol::Float64}: (atol = 1.0e-8,)

julia> fees_constraints(nothing)

```

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
  - [`fees_constraints`](@ref)
"""
function fees_constraints(fees::Union{Nothing, <:Fees}, args...; kwargs...)
    return fees
end
function fees_view(::Nothing, ::Any)
    return nothing
end
function fees_view(fees::Fees, i::AbstractVector)
    tn = turnover_view(fees.tn, i)
    l = nothing_scalar_array_view(fees.l, i)
    s = nothing_scalar_array_view(fees.s, i)
    fl = nothing_scalar_array_view(fees.fl, i)
    fs = nothing_scalar_array_view(fees.fs, i)
    return Fees(; tn = tn, l = l, s = s, fl = fl, fs = fs, kwargs = fees.kwargs)
end
function factory(fees::Fees, w::AbstractVector)
    return Fees(; tn = factory(fees.tn, w), l = fees.l, s = fees.s, fl = fees.fl,
                fs = fees.fs, kwargs = fees.kwargs)
end
function calc_fees(w::AbstractVector, p::AbstractVector, ::Nothing, ::Function)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::Real, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot_scalar(fees * w[idx], p[idx])
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::AbstractVector{<:Real},
                   op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot(fees[idx], w[idx] .* p[idx])
end
function calc_fees(w::AbstractVector, p::AbstractVector, ::Nothing)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::AbstractVector, p::AbstractVector, tn::Turnover{<:Any, <:Real})
    return dot_scalar(tn.val * abs.(w - tn.w), p)
end
function calc_fees(w::AbstractVector, p::AbstractVector,
                   tn::Turnover{<:Any, <:AbstractVector})
    return dot(tn.val, abs.(w - tn.w) .* p)
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, p, fees.l, .>=)
    fees_short = -calc_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_fees(w::AbstractVector, ::Nothing, ::Function)
    return zero(eltype(w))
end
function calc_fees(w::AbstractVector, fees::Real, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return sum(fees * w[idx])
end
function calc_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return dot(fees[idx], w[idx])
end
function calc_fees(w::AbstractVector, tn::Turnover{<:Any, <:Real})
    return sum(tn.val * abs.(w - tn.w))
end
function calc_fees(w::AbstractVector, tn::Turnover{<:Any, <:AbstractVector})
    return dot(tn.val, abs.(w - tn.w))
end
function calc_fees(w::AbstractVector, ::Nothing)
    return zero(eltype(w))
end
function calc_fixed_fees(w::AbstractVector, ::Nothing, kwargs::NamedTuple, op::Function)
    return zero(eltype(w))
end
function calc_fixed_fees(w::AbstractVector, fees::Real, kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return fees * sum(idx2)
end
function calc_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                         kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return sum(fees[idx1][idx2])
end
"""
"""
function calc_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, fees.l, .>=)
    fees_short = -calc_fees(w, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, ::Nothing, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] .= fees * w[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] .= fees[idx] ⊙ w[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, ::Nothing)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::AbstractVector, tn::Turnover{<:Any, <:Real})
    return tn.val * abs.(w - tn.w)
end
function calc_asset_fees(w::AbstractVector, tn::Turnover{<:Any, <:AbstractVector})
    return tn.val ⊙ abs.(w - tn.w)
end
function calc_asset_fixed_fees(w::AbstractVector, ::Nothing, ::NamedTuple, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fixed_fees(w::AbstractVector, fees::Real, kwargs::NamedTuple,
                               op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] .= fees * idx2
    return fees_w
end
function calc_asset_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                               kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] .= fees[idx1][idx2]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, fees.l, .>=)
    fees_short = -calc_asset_fees(w, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, ::Nothing, ::Function)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] .= fees * w[idx] ⊙ p[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::AbstractVector{<:Real},
                         op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] .= fees[idx] ⊙ w[idx] ⊙ p[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, tn::Turnover{<:Any, <:Real})
    return tn.val * abs.(w - tn.w) ⊙ p
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, ::Nothing)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector,
                         tn::Turnover{<:Any, <:AbstractVector})
    return tn.val ⊙ abs.(w - tn.w) ⊙ p
end
"""
"""
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, p, fees.l, .>=)
    fees_short = -calc_asset_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
"""
function calc_net_returns(w::AbstractVector, X::AbstractMatrix, args...)
    return X * w
end
function calc_net_returns(w::AbstractVector, X::AbstractMatrix, fees::Fees)
    return X * w .- calc_fees(w, fees)
end
function calc_net_asset_returns(w::AbstractVector, X::AbstractMatrix, args...)
    return X ⊙ transpose(w)
end
function calc_net_asset_returns(w::AbstractVector, X::AbstractMatrix, fees::Fees)
    return X ⊙ transpose(w) .- transpose(calc_asset_fees(w, fees))
end
function cumulative_returns(X::AbstractArray; compound::Bool = false, dims::Int = 1)
    return if compound
        cumprod(one(eltype(X)) .+ X; dims = dims)
    else
        cumsum(X; dims = dims)
    end
end
function drawdowns(X::AbstractArray; cX::Bool = false, compound::Bool = false,
                   dims::Int = 1)
    cX = !cX ? cumulative_returns(X; compound = compound, dims = dims) : X
    if compound
        return cX ./ accumulate(max, cX; dims = dims) .- one(eltype(X))
    else
        return cX - accumulate(max, cX; dims = dims)
    end
    return nothing
end

export FeesEstimator, Fees, fees_constraints, calc_fees, calc_asset_fees, calc_net_returns,
       calc_net_asset_returns, cumulative_returns, drawdowns
