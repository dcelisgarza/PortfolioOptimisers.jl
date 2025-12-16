"""
    struct FeesEstimator{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <: AbstractEstimator
        tn::T1
        l::T2
        s::T3
        fl::T4
        fs::T5
        dl::T6
        ds::T7
        dfl::T8
        dfs::T9
        kwargs::T10
    end

Estimator for portfolio transaction fees constraints.

`FeesEstimator` specifies transaction fee constraints for each asset in a portfolio, including turnover fees, long/short proportional fees, and long/short fixed fees. Supports asset-specific fees via dictionaries, pairs, or vectors of pairs.

This estimator can be converted into a concrete [`Fees`](@ref) constraint using the [`fees_constraints`](@ref) function, which maps the estimator's specifications to the assets in a given [`AssetSets`](@ref) object.

!!! warning

    The value of the fees must match the periodicity and units of the returns used in the portfolio optimisation. For example, if using daily returns, the fees should be specified as daily fees.

# Fields

  - `tn`: Turnover estimator or result.
  - `l`: Long proportional fees.
  - `s`: Short proportional fees.
  - `fl`: Long fixed fees.
  - `fs`: Short fixed fees.
  - `dl`: Default long proportional fees.
  - `ds`: Default short proportional fees.
  - `dfl`: Default long fixed fees.
  - `dfs`: Default short fixed fees.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered zero.

# Constructor

    FeesEstimator(; tn::Option{<:TnE_Tn} = nothing, l::Option{<:EstValType} = nothing,
                  s::Option{<:EstValType} = nothing, fl::Option{<:EstValType} = nothing,
                  fs::Option{<:EstValType} = nothing, dl::Option{<:Number} = nothing,
                  ds::Option{<:Number} = nothing, dfl::Option{<:Number} = nothing,
                  dfs::Option{<:Number} = nothing, kwargs::NamedTuple = (; atol = 1e-8))

Keyword arguments correspond to the fields above.

## Validation

  - `l`, `s`, `fl`, `fs`, `dl`, `ds`, `dfl`, `dfs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> FeesEstimator(; tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1)),
                     l = Dict("A" => 0.001, "B" => 0.002), s = ["A" => 0.001, "B" => 0.002],
                     fl = Dict("A" => 5.0), fs = ["B" => 10.0])
FeesEstimator
      tn ┼ TurnoverEstimator
         │      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │    val ┼ Dict{String, Float64}: Dict("A" => 0.1)
         │   dval ┴ nothing
       l ┼ Dict{String, Float64}: Dict("B" => 0.002, "A" => 0.001)
       s ┼ Vector{Pair{String, Float64}}: ["A" => 0.001, "B" => 0.002]
      fl ┼ Dict{String, Float64}: Dict("A" => 5.0)
      fs ┼ Vector{Pair{String, Float64}}: ["B" => 10.0]
      dl ┼ nothing
      ds ┼ nothing
     dfl ┼ nothing
     dfs ┼ nothing
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`Fees`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`Option`](@ref)
  - [`TnE_Tn`](@ref)
  - [`EstValType`](@ref)
  - [`fees_constraints`](@ref)
"""
struct FeesEstimator{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <: AbstractEstimator
    tn::T1
    l::T2
    s::T3
    fl::T4
    fs::T5
    dl::T6
    ds::T7
    dfl::T8
    dfs::T9
    kwargs::T10
    function FeesEstimator(tn::Option{<:TnE_Tn}, l::Option{<:EstValType},
                           s::Option{<:EstValType}, fl::Option{<:EstValType},
                           fs::Option{<:EstValType}, dl::Option{<:Number} = nothing,
                           ds::Option{<:Number} = nothing, dfl::Option{<:Number} = nothing,
                           dfs::Option{<:Number} = nothing,
                           kwargs::NamedTuple = (; atol = 1e-8))
        assert_nonempty_nonneg_finite_val(l, :l)
        assert_nonempty_nonneg_finite_val(s, :s)
        assert_nonempty_nonneg_finite_val(fl, :fl)
        assert_nonempty_nonneg_finite_val(fs, :fs)
        assert_nonempty_nonneg_finite_val(dl, :dl)
        assert_nonempty_nonneg_finite_val(ds, :ds)
        assert_nonempty_nonneg_finite_val(dfl, :dfl)
        assert_nonempty_nonneg_finite_val(dfs, :dfs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs), typeof(dl),
                   typeof(ds), typeof(dfl), typeof(dfs), typeof(kwargs)}(tn, l, s, fl, fs,
                                                                         dl, ds, dfl, dfs,
                                                                         kwargs)
    end
end
function FeesEstimator(; tn::Option{<:TnE_Tn} = nothing, l::Option{<:EstValType} = nothing,
                       s::Option{<:EstValType} = nothing,
                       fl::Option{<:EstValType} = nothing,
                       fs::Option{<:EstValType} = nothing, dl::Option{<:Number} = nothing,
                       ds::Option{<:Number} = nothing, dfl::Option{<:Number} = nothing,
                       dfs::Option{<:Number} = nothing,
                       kwargs::NamedTuple = (; atol = 1e-8))
    return FeesEstimator(tn, l, s, fl, fs, dl, ds, dfl, dfs, kwargs)
end
"""
    fees_view(fees::FeesEstimator, i)

Create a view of a `FeesEstimator` for a subset of assets.

Returns a new `FeesEstimator` with all fee fields restricted to the indices or assets specified by `i`. The default fee values and keyword arguments are propagated unchanged.

# Arguments

  - `fees`: Instance of `FeesEstimator`.
  - `i`: Index or indices specifying the subset of assets.

# Returns

  - `fe::FeesEstimator`: New estimator with fields restricted to the specified subset.

# Details

  - Uses `turnover_view` to subset the turnover estimator/result.
  - Uses `nothing_scalar_array_view` to subset proportional and fixed fee fields.
  - Propagates default fee values and keyword arguments unchanged.
  - Enables composable processing of asset subsets for fee constraints.

# Examples

```jldoctest
julia> fees = FeesEstimator(; tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1)),
                            l = Dict("A" => 0.001, "B" => 0.002), s = ["A" => 0.001, "B" => 0.002],
                            fl = Dict("A" => 5.0), fs = ["B" => 10.0]);

julia> PortfolioOptimisers.fees_view(fees, 1:2)
FeesEstimator
      tn ┼ TurnoverEstimator
         │      w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
         │    val ┼ Dict{String, Float64}: Dict("A" => 0.1)
         │   dval ┴ nothing
       l ┼ Dict{String, Float64}: Dict("B" => 0.002, "A" => 0.001)
       s ┼ Vector{Pair{String, Float64}}: ["A" => 0.001, "B" => 0.002]
      fl ┼ Dict{String, Float64}: Dict("A" => 5.0)
      fs ┼ Vector{Pair{String, Float64}}: ["B" => 10.0]
      dl ┼ nothing
      ds ┼ nothing
     dfl ┼ nothing
     dfs ┼ nothing
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
  - [`fees_constraints`](@ref)
  - [`turnover_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function fees_view(fees::FeesEstimator, i)
    tn = turnover_view(fees.tn, i)
    l = nothing_scalar_array_view(fees.l, i)
    s = nothing_scalar_array_view(fees.s, i)
    fl = nothing_scalar_array_view(fees.fl, i)
    fs = nothing_scalar_array_view(fees.fs, i)
    return FeesEstimator(; tn = tn, l = l, s = s, fl = fl, fs = fs, dl = fees.dl,
                         ds = fees.ds, dfl = fees.dfl, dfs = fees.dfs, kwargs = fees.kwargs)
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

`Fees` stores transaction fee constraints for each asset in a portfolio, including turnover fees, long/short proportional fees, and long/short fixed fees. Fixed fees do not depend on the value of the asset weights, only whether it is positive or negative---up to a tolerance defined by how close the weight is to zero defined by `isapprox` and the `kwargs` field.

Fee values can be specified as scalars (applied to all assets) or as vectors of per-asset values. The portfolio fees are computed by [`calc_fees`](@ref) and asset fees by [`calc_asset_fees`](@ref).

!!! warning

    The value of the fees must match the periodicity and units of the returns used in the portfolio optimisation. For example, if using daily returns, the fees should be specified as daily fees.

## Portfolio fees

For non-finite optimisations, the total portfolio transaction fees are computed as:

```math
\\begin{align}
F_{\\text{t}} &\\coloneqq F_{\\text{Tn}} + F_{\\text{p}} + F_{\\text{f}} \\\\
F_{\\text{Tn}} &= \\boldsymbol{Tn} \\cdot \\boldsymbol{f}_{\\text{Tn}}\\\\
F_{\\text{p}} &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{+} + \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
F_{\\text{f}} &= 1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\cdot \\boldsymbol{f}_{\\text{f}}^{+} + 1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\cdot\\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

Where:

  - ``F``: Portfolio fee.
  - ``\\boldsymbol{f}``: Per asset fee vector. If it is a scalar, it is broadcasted to all assets.
  - ``\\boldsymbol{Tn}``: Turnover vector as defined in [`Turnover`](@ref).
  - ``+,\\, -``: Superscripts denote long and short fees respectively. This is because brokers sometimes charge different fees for long and short positions.
  - ``\\text{t},\\, \\text{Tn},\\, \\text{p},\\, \\text{f}``: Subscripts for total, turnover, proportional, and fixed fees respectively. The turnover fee is an instance of [`Turnover`](@ref), where `val` is the per asset fee.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise (Hadamard) indicator function returning `1` when the condition is true, `0` otherwise. This activates long or short fees based on whether the asset weight is non-negative or otherwise.

The finite optimisations use fees somewhat differently because they use a finite amount of capital as well as asset prices to compute the actual fees incurred when buying or selling assets. As such, these fees require a vector of asset prices to compute the actual fees incurred. However, this method lets us automatically adjust the available cash amount during the optimisation so that fees are discounted from the available cash.

```math
\\begin{align}
F_{\\text{t}} &\\coloneqq F_{\\text{Tn}} + F_{\\text{p}} + F_{\\text{f}} \\\\
F_{\\text{Tn}} &= \\left(\\boldsymbol{Tn} \\odot \\boldsymbol{X} \\right) \\cdot \\boldsymbol{f}_{\\text{Tn}}\\\\
F_{\\text{p}} &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{+} + \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
F_{\\text{f}} &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{X}\\right) \\cdot \\boldsymbol{f}_{\\text{f}}^{+} + \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{X}\\right) \\cdot \\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

Where:

  - ``F``: Portfolio fee.
  - ``\\boldsymbol{X}``: Asset price vector.
  - ``\\boldsymbol{f}``: Per asset fee vector. If it is a scalar, it is broadcasted to all assets.
  - ``\\boldsymbol{Tn}``: Turnover vector as defined in [`Turnover`](@ref).
  - ``+,\\, -``: Superscripts denote long and short fees respectively. This is because brokers sometimes charge different fees for long and short positions.
  - ``\\text{t},\\, \\text{Tn},\\, \\text{p},\\, \\text{f}``: Subscripts for total, turnover, proportional, and fixed fees respectively. The turnover fee is an instance of [`Turnover`](@ref), where `val` is the per asset fee.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise (Hadamard) indicator function returning `1` when the condition is true, `0` otherwise. This activates long or short fees based on whether the asset weight is non-negative or otherwise.
  - ``\\odot``: Elementwise (Hadamard) product.

## Per-asset fees

It is also possible to compute per-asset fees incurred using the same definitions as above, but replacing the dot products with elementwise (Hadamard) products.

```math
\\begin{align}
F_{\\text{t}} &\\coloneqq F_{\\text{Tn}} + F_{\\text{p}} + F_{\\text{f}} \\\\
F_{\\text{Tn}} &= \\boldsymbol{Tn} \\odot \\boldsymbol{f}_{\\text{Tn}}\\\\
F_{\\text{p}} &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{+} + \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
F_{\\text{f}} &= 1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{f}_{\\text{f}}^{+} + 1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot\\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

Where:

  - ``F``: Portfolio fee.
  - ``\\boldsymbol{f}``: Per asset fee vector. If it is a scalar, it is broadcasted to all assets.
  - ``\\boldsymbol{Tn}``: Turnover vector as defined in [`Turnover`](@ref).
  - ``+,\\, -``: Superscripts denote long and short fees respectively. This is because brokers sometimes charge different fees for long and short positions.
  - ``\\text{t},\\, \\text{Tn},\\, \\text{p},\\, \\text{f}``: Subscripts for total, turnover, proportional, and fixed fees respectively. The turnover fee is an instance of [`Turnover`](@ref), where `val` is the per asset fee.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise (Hadamard) indicator function returning `1` when the condition is true, `0` otherwise. This activates long or short fees based on whether the asset weight is non-negative or otherwise.
  - ``\\odot``: Elementwise (Hadamard) product.

The finite optimisation uses fees somewhat differently because it uses a finite amount of capital and utilises the asset prices to compute the actual fees incurred when buying or selling assets. As such, these fees require a vector of asset prices to compute the actual fees incurred.

```math
\\begin{align}
F_{\\text{t}} &\\coloneqq F_{\\text{Tn}} + F_{\\text{p}} + F_{\\text{f}} \\\\
F_{\\text{Tn}} &= \\left(\\boldsymbol{Tn} \\odot \\boldsymbol{X} \\right) \\odot \\boldsymbol{f}_{\\text{Tn}}\\\\
F_{\\text{p}} &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{+} + \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
F_{\\text{f}} &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{X}\\right) \\odot \\boldsymbol{f}_{\\text{f}}^{+} + \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{X}\\right) \\odot \\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

Where:

  - ``F``: Portfolio fee.
  - ``\\boldsymbol{X}``: Asset price vector.
  - ``\\boldsymbol{f}``: Per asset fee vector. If it is a scalar, it is broadcasted to all assets.
  - ``\\boldsymbol{Tn}``: Turnover vector as defined in [`Turnover`](@ref).
  - ``+,\\, -``: Superscripts denote long and short fees respectively. This is because brokers sometimes charge different fees for long and short positions.
  - ``\\text{t},\\, \\text{Tn},\\, \\text{p},\\, \\text{f}``: Subscripts for total, turnover, proportional, and fixed fees respectively. The turnover fee is an instance of [`Turnover`](@ref), where `val` is the per asset fee.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise (Hadamard) indicator function returning `1` when the condition is true, `0` otherwise. This activates long or short fees based on whether the asset weight is non-negative or otherwise.
  - ``\\odot``: Elementwise (Hadamard) product.

# Fields

  - `tn`: Turnover constraint result.
  - `l`: Long proportional fees.
  - `s`: Short proportional fees.
  - `fl`: Long fixed fees.
  - `fs`: Short fixed fees.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered zero.

# Constructor

    Fees(; tn::Option{<:Turnover} = nothing, l::Option{<:Num_VecNum} = nothing,
         s::Option{<:Num_VecNum} = nothing, fl::Option{<:Num_VecNum} = nothing,
         fs::Option{<:Num_VecNum} = nothing, kwargs::NamedTuple = (; atol = 1e-8))

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
  - [`Option`](@ref)
  - [`Turnover`](@ref)
  - [`Num_VecNum`](@ref)
  - [`AbstractResult`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
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
    function Fees(tn::Option{<:Turnover}, l::Option{<:Num_VecNum}, s::Option{<:Num_VecNum},
                  fl::Option{<:Num_VecNum}, fs::Option{<:Num_VecNum},
                  kwargs::NamedTuple = (; atol = 1e-8))
        assert_nonempty_nonneg_finite_val(l, :l)
        assert_nonempty_nonneg_finite_val(s, :s)
        assert_nonempty_nonneg_finite_val(fl, :fl)
        assert_nonempty_nonneg_finite_val(fs, :fs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs),
                   typeof(kwargs)}(tn, l, s, fl, fs, kwargs)
    end
end
function Fees(; tn::Option{<:Turnover} = nothing, l::Option{<:Num_VecNum} = nothing,
              s::Option{<:Num_VecNum} = nothing, fl::Option{<:Num_VecNum} = nothing,
              fs::Option{<:Num_VecNum} = nothing, kwargs::NamedTuple = (; atol = 1e-8))
    return Fees(tn, l, s, fl, fs, kwargs)
end
"""
    const FeesE_Fees = Union{<:Fees, <:FeesEstimator}

Union type for fee constraint objects and estimators.

# Related

  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
"""
const FeesE_Fees = Union{<:Fees, <:FeesEstimator}
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

  - `fe::Fees`: Object containing turnover, proportional, and fixed fee values aligned with `sets`.

# Details

  - Fee values are extracted and mapped to assets using [`estimator_to_val`](@ref).
  - If a fee value is missing for an asset, assigns zero unless `strict` is `true`.
  - Turnover constraints are generated using [`turnover_constraints`](@ref).

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
  - [`estimator_to_val`](@ref)
  - [`AssetSets`](@ref)
"""
function fees_constraints(fees::FeesEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)
    return Fees(;
                tn = turnover_constraints(fees.tn, sets; datatype = datatype,
                                          strict = strict),
                l = estimator_to_val(fees.l, sets, fees.dl; datatype = datatype,
                                     strict = strict),
                s = estimator_to_val(fees.s, sets, fees.ds; datatype = datatype,
                                     strict = strict),
                fl = estimator_to_val(fees.fl, sets, fees.dfl; datatype = datatype,
                                      strict = strict),
                fs = estimator_to_val(fees.fs, sets, fees.dfs; datatype = datatype,
                                      strict = strict))
end
"""
    fees_constraints(fees::Option{<:Fees}, args...; kwargs...)

Propagate or pass through portfolio transaction fee constraints.

`fees_constraints` returns the input [`Fees`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed fee constraints or missing constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `fees`: An existing [`Fees`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `fe::Option{<:Fees}`: The input constraint object, unchanged.

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
function fees_constraints(fees::Option{<:Fees}, args...; kwargs...)
    return fees
end
"""
    fees_view(::Nothing, ::Any)

Return `nothing` when no fee estimator or constraint is provided.

This method is used as a fallback for missing fee estimators or constraints, ensuring composability and uniform interface handling in fee constraint processing workflows.

# Arguments

  - `::Nothing`: Indicates absence of a fee estimator or constraint.
  - `::Any`: Index or argument (ignored).

# Returns

  - `nothing`.

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
  - [`fees_view`](@ref)
"""
function fees_view(::Nothing, ::Any)
    return nothing
end
"""
    fees_view(fees::Fees, i)

Create a view of a `Fees` constraint for a subset of assets.

Returns a new `Fees` object with all fee fields restricted to the indices or assets specified by `i`. The keyword arguments are propagated unchanged.

# Arguments

  - `fees`: A `Fees` constraint object containing turnover, proportional, and fixed fee values.
  - `i`: Index or indices specifying the subset of assets.

# Returns

  - `fe::Fees`: New constraint object with fields restricted to the specified subset.

# Details

  - Uses `turnover_view` to subset the turnover constraint.
  - Uses `nothing_scalar_array_view` to subset proportional and fixed fee fields.
  - Propagates keyword arguments unchanged.
  - Enables composable processing of asset subsets for fee constraints.

# Examples

```jldoctest
julia> fees = Fees(; tn = Turnover([0.2, 0.3, 0.5], [0.1, 0.0, 0.0]), l = [0.001, 0.002, 0.0],
                   s = [0.001, 0.002, 0.0], fl = [5.0, 0.0, 0.0], fs = [0.0, 10.0, 0.0]);

julia> PortfolioOptimisers.fees_view(fees, 1:2)
Fees
      tn ┼ Turnover
         │     w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
         │   val ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.1, 0.0]
       l ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.001, 0.002]
       s ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.001, 0.002]
      fl ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [5.0, 0.0]
      fs ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.0, 10.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
  - [`fees_constraints`](@ref)
  - [`turnover_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function fees_view(fees::Fees, i)
    tn = turnover_view(fees.tn, i)
    l = nothing_scalar_array_view(fees.l, i)
    s = nothing_scalar_array_view(fees.s, i)
    fl = nothing_scalar_array_view(fees.fl, i)
    fs = nothing_scalar_array_view(fees.fs, i)
    return Fees(; tn = tn, l = l, s = s, fl = fl, fs = fs, kwargs = fees.kwargs)
end
"""
    factory(fees::Fees, w::VecNum)

Create a new `Fees` constraint with updated portfolio weights.

`factory` constructs a new [`Fees`](@ref) object using the provided portfolio weights `w` and the fee values from an existing `Fees` constraint `fees`. The turnover constraint is updated using `factory(fees.tn, w)`, while all other fee fields and keyword arguments are preserved.

# Arguments

  - `fees`: Existing `Fees` constraint object.

      + Contains turnover, proportional, and fixed fee values.

  - `w`: Portfolio weights vector.

      + New weights to assign to the constraint.

# Returns

  - `fe::Fees`: New constraint object with updated weights in the turnover field and original fee values.

# Details

  - Updates only the weights field in the turnover constraint via `factory(fees.tn, w)`.
  - Propagates all other fields unchanged

# Examples

```jldoctest
julia> fees = Fees(; tn = Turnover([0.2, 0.3, 0.5], [0.1, 0.0, 0.0]), l = [0.001, 0.002, 0.0],
                   s = [0.001, 0.002, 0.0], fl = [5.0, 0.0, 0.0], fs = [0.0, 10.0, 0.0]);

julia> factory(fees, [0.4, 0.4, 0.2])
Fees
      tn ┼ Turnover
         │     w ┼ Vector{Float64}: [0.4, 0.4, 0.2]
         │   val ┴ Vector{Float64}: [0.1, 0.0, 0.0]
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ Vector{Float64}: [0.001, 0.002, 0.0]
      fl ┼ Vector{Float64}: [5.0, 0.0, 0.0]
      fs ┼ Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`Fees`](@ref)
  - [`Turnover`](@ref)
  - [`VecNum`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`fees_constraints`](@ref)
"""
function factory(fees::Fees, w::VecNum)
    return Fees(; tn = factory(fees.tn, w), l = fees.l, s = fees.s, fl = fees.fl,
                fs = fees.fs, kwargs = fees.kwargs)
end
"""
    calc_fees(w::VecNum, p::VecNum, ::Nothing, ::Function)
    calc_fees(w::VecNum, p::VecNum, fees::Number, op::Function)
    calc_fees(w::VecNum, p::VecNum, fees::VecNum, op::Function)

Compute the actual proportional fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `VecNum`: Vector of fee values per asset.
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
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, p::VecNum, ::Nothing, ::Function)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::VecNum, p::VecNum, fees::Number, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return fees * dot(w[idx], p[idx])
end
function calc_fees(w::VecNum, p::VecNum, fees::VecNum, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot(fees[idx], w[idx] .* p[idx])
end
"""
    calc_fees(w::VecNum, p::VecNum, ::Nothing)
    calc_fees(w::VecNum, p::VecNum, tn::Turnover)

Compute the actual turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: Single turnover fee applied to all assets.
      + `tn.val::VecNum`: Vector of turnover fees per asset.

# Returns

  - `val::Number`: Actual turnover fee.

# Examples

```jldoctest
julia> calc_fees([0.1, 0.2], [100, 200], Turnover([0.0, 0.0], 0.01))
0.5
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`Turnover`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, p::VecNum, ::Nothing)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::VecNum, p::VecNum, tn::Turnover{<:Any, <:Number})
    return tn.val * dot(abs.(w - tn.w), p)
end
function calc_fees(w::VecNum, p::VecNum, tn::Turnover{<:Any, <:VecNum})
    return dot(tn.val, abs.(w - tn.w) .* p)
end
"""
    calc_fees(w::VecNum, p::VecNum, fees::Fees)

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
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, p::VecNum, fees::Fees)
    fees_long = calc_fees(w, p, fees.l, .>=)
    fees_short = -calc_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_fees(w::VecNum, ::Nothing, ::Function)
    calc_fees(w::VecNum, fees::Number, op::Function)
    calc_fees(w::VecNum, fees::VecNum, op::Function)

Compute the proportional fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `VecNum`: Vector of fee values per asset.
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
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, ::Nothing, ::Function)
    return zero(eltype(w))
end
function calc_fees(w::VecNum, fees::Number, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return sum(fees * w[idx])
end
function calc_fees(w::VecNum, fees::VecNum, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return dot(fees[idx], w[idx])
end
"""
    calc_fees(w::VecNum, ::Nothing)
    calc_fees(w::VecNum, tn::Turnover)

Compute the turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: Single turnover fee applied to all assets.
      + `tn.val::VecNum`: Vector of turnover fees per asset.

# Returns

  - `val::Number`: Turnover fee.

# Examples

```jldoctest
julia> calc_fees([0.8, 0.2], Turnover([0.0, 0.0], 0.02))
0.02
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`Turnover`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, ::Nothing)
    return zero(eltype(w))
end
function calc_fees(w::VecNum, tn::Turnover{<:Any, <:Number})
    return tn.val * sum(abs.(w - tn.w))
end
function calc_fees(w::VecNum, tn::Turnover{<:Any, <:VecNum})
    return dot(tn.val, abs.(w - tn.w))
end
"""
    calc_fixed_fees(w::VecNum, ::Nothing, kwargs::NamedTuple, ::Function)
    calc_fixed_fees(w::VecNum, fees::Number, kwargs::NamedTuple, op::Function)
    calc_fixed_fees(w::VecNum, fees::VecNum, kwargs::NamedTuple, op::Function)

Compute the fixed portfolio fees for assets that have been allocated.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `VecNum`: Vector of fee values per asset.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered zero.
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
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fixed_fees(w::VecNum, ::Nothing, kwargs::NamedTuple, op::Function)
    return zero(eltype(w))
end
function calc_fixed_fees(w::VecNum, fees::Number, kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return fees * sum(idx2)
end
function calc_fixed_fees(w::VecNum, fees::VecNum, kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return sum(fees[idx1][idx2])
end
"""
    calc_fees(w::VecNum, fees::Fees)

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
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, fees::Fees)
    fees_long = calc_fees(w, fees.l, .>=)
    fees_short = -calc_fees(w, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_asset_fees(w::VecNum, p::VecNum, ::Nothing, ::Function)
    calc_asset_fees(w::VecNum, p::VecNum, fees::Number, op::Function)
    calc_asset_fees(w::VecNum, p::VecNum, fees::VecNum, op::Function)

Compute the actual proportional per asset fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `VecNum`: Vector of fee values per asset.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::VecNum`: Total actual proportional per asset fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], [100, 200], 0.01, .>=)
2-element Vector{Float64}:
 0.1
 0.4
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, p::VecNum, ::Nothing, ::Function)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::VecNum, p::VecNum, fees::Number, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] = fees * w[idx] ⊙ p[idx]
    return fees_w
end
function calc_asset_fees(w::VecNum, p::VecNum, fees::VecNum, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] = fees[idx] ⊙ w[idx] ⊙ p[idx]
    return fees_w
end
"""
    calc_asset_fees(w::VecNum, p::VecNum, ::Nothing)
    calc_asset_fees(w::VecNum, p::VecNum, tn::Turnover)

Compute the actual per asset turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `p`: Asset prices.
  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: Single turnover fee applied to all assets.
      + `tn.val::VecNum`: Vector of turnover fees per asset.

# Returns

  - `val::VecNum`: Actual per asset turnover fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], [100, 200], Turnover([0.0, 0.0], 0.01))
2-element Vector{Float64}:
 0.1
 0.4
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, p::VecNum, ::Nothing)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::VecNum, p::VecNum, tn::Turnover{<:Any, <:Number})
    return tn.val * abs.(w - tn.w) ⊙ p
end
function calc_asset_fees(w::VecNum, p::VecNum, tn::Turnover{<:Any, <:VecNum})
    return tn.val ⊙ abs.(w - tn.w) ⊙ p
end
"""
    calc_asset_fees(w::VecNum, p::VecNum, fees::Fees)

Compute total actual per asset fees for portfolio weights and prices.

Sums actual proportional, fixed, and turnover fees for all assets.

# Arguments

  - `w`: Portfolio weights.
  - `p`: Asset prices.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::VecNum`: Total actual per asset fees.

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
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, p::VecNum, fees::Fees)
    fees_long = calc_asset_fees(w, p, fees.l, .>=)
    fees_short = -calc_asset_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
"""
    calc_asset_fees(w::VecNum, ::Nothing, ::Function)
    calc_asset_fees(w::VecNum, fees::Number, op::Function)
    calc_asset_fees(w::VecNum, fees::VecNum, op::Function)

Compute the proportional per asset fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `VecNum`: Vector of fee values per asset.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::VecNum`: Total proportional per asset fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], 0.01, .>=)
2-element Vector{Float64}:
 0.001
 0.002
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, ::Nothing, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::VecNum, fees::Number, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] = fees * w[idx]
    return fees_w
end
function calc_asset_fees(w::VecNum, fees::VecNum, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] = fees[idx] ⊙ w[idx]
    return fees_w
end
"""
    calc_asset_fees(w::VecNum, ::Nothing)
    calc_asset_fees(w::VecNum, tn::Turnover)

Compute the per asset turnover fees for portfolio weights and prices.

# Arguments

  - `w`: Portfolio weights.

  - `tn`: Turnover structure.

      + `nothing`: No turnover fee, returns zero.
      + `tn.val::Number`: Single turnover fee applied to all assets.
      + `tn.val::VecNum`: Vector of turnover fees per asset.

# Returns

  - `val::VecNum`: Per asset turnover fee.

# Examples

```jldoctest
julia> calc_asset_fees([0.1, 0.2], Turnover([0.0, 0.0], 0.01))
2-element Vector{Float64}:
 0.001
 0.002
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, ::Nothing)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::VecNum, tn::Turnover{<:Any, <:Number})
    return tn.val * abs.(w - tn.w)
end
function calc_asset_fees(w::VecNum, tn::Turnover{<:Any, <:VecNum})
    return tn.val ⊙ abs.(w - tn.w)
end
"""
    calc_asset_fixed_fees(w::VecNum, ::Nothing, kwargs::NamedTuple, ::Function)
    calc_asset_fixed_fees(w::VecNum, fees::Number, kwargs::NamedTuple, op::Function)
    calc_asset_fixed_fees(w::VecNum, fees::VecNum, kwargs::NamedTuple, op::Function)

Compute the per asset fixed portfolio fees for assets that have been allocated.

# Arguments

  - `w`: Portfolio weights.

  - `fees`: Scalar fee value.

      + `nothing`: No proportional fee, returns zero.
      + `Number`: Single fee applied to all relevant assets.
      + `VecNum`: Vector of fee values per asset.
  - `kwargs`: Named tuple of keyword arguments for deciding how small an asset weight has to be before being considered zero.
  - `op`: Function to select assets, `.>=` for long, `<` for short (ignored if `fees` is `nothing`).

# Returns

  - `val::VecNum`: Total per asset fixed fee.

# Examples

```jldoctest
julia> calc_asset_fixed_fees([0.1, 0.2], 0.01, (; atol = 1e-6), .>=)
2-element Vector{Float64}:
 0.01
 0.01
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fixed_fees(w::VecNum, ::Nothing, ::NamedTuple, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fixed_fees(w::VecNum, fees::Number, kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] = fees * idx2
    return fees_w
end
function calc_asset_fixed_fees(w::VecNum, fees::VecNum, kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] = fees[idx1][idx2]
    return fees_w
end
"""
    calc_asset_fees(w::VecNum, fees::Fees)

Compute total per asset fees for portfolio weights and prices.

Sums proportional, fixed, and turnover fees for all assets.

# Arguments

  - `w`: Portfolio weights.
  - `p`: Asset prices.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::VecNum`: Total per asset fees.

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
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, fees::Fees)
    fees_long = calc_asset_fees(w, fees.l, .>=)
    fees_short = -calc_asset_fees(w, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end

export FeesEstimator, Fees, fees_constraints, calc_fees, calc_fixed_fees, calc_asset_fees,
       calc_asset_fixed_fees
