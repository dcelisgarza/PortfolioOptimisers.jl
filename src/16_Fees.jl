"""
$(DocStringExtensions.TYPEDEF)

Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe.

Every fee field accepts a dictionary, a pair, or a vector of pairs keyed by asset or group name, and the matching `d*` field fills every asset the keys miss. Each default fills only its own field: `l` draws on `dl`, `s` on `ds`, `fl` on `dfl` and `fs` on `dfs`, and never on a neighbour's. [`fees_constraints`](@ref) resolves the names against a [`UniverseSets`](@ref) and returns a [`Fees`](@ref), whose fee fields are plain per-asset vectors and whose `kwargs` is the `kwargs` of this estimator.

!!! warning

    The turnover and proportional fees must match the periodicity of the returns series, and the fixed fees must be divided by the portfolio's holding period. The units of the fees and returns must also be consistent.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeesEstimator(;
        tn::Option{<:TnE_Tn} = nothing,
        l::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        s::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        fl::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        fs::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        dl::Option{<:Number} = nothing,
        ds::Option{<:Number} = nothing,
        dfl::Option{<:Number} = nothing,
        dfs::Option{<:Number} = nothing,
        kwargs::NamedTuple = (; atol = 1e-8)
    ) -> FeesEstimator

Keywords correspond to the struct's fields.

## Validation

  - `l`, `s`, `fl`, `fs`, `dl`, `ds`, `dfl`, `dfs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `tn`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `tn`: Recursively viewed via [`port_opt_view`](@ref).
  - `l`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `s`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `fl`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `fs`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> FeesEstimator(; tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1)),
                     l = Dict(\"A\" => 0.001, \"B\" => 0.002), s = [\"A\" => 0.001, \"B\" => 0.002],
                     fl = Dict(\"A\" => 5.0), fs = [\"B\" => 10.0])
FeesEstimator
      tn ┼ TurnoverEstimator
         │       w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │     val ┼ Dict{String, Float64}: Dict("A" => 0.1)
         │    dval ┼ nothing
         │   fixed ┴ Bool: false
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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct FeesEstimator <: AbstractEstimator
    """
    $(field_dict[:tn_fees])
    """
    @fprop @vprop tn
    """
    $(field_dict[:l_fees])
    """
    @vprop l
    """
    $(field_dict[:s_fees])
    """
    @vprop s
    """
    $(field_dict[:fl])
    """
    @vprop fl
    """
    $(field_dict[:fs])
    """
    @vprop fs
    """
    $(field_dict[:dl])
    """
    dl
    """
    $(field_dict[:ds])
    """
    ds
    """
    $(field_dict[:dfl])
    """
    dfl
    """
    $(field_dict[:dfs])
    """
    dfs
    """
    $(field_dict[:kwargs_fee])
    """
    kwargs
    function FeesEstimator(tn::Option{<:TnE_Tn},
                           l::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           s::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           fl::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           fs::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           dl::Option{<:Number} = nothing, ds::Option{<:Number} = nothing,
                           dfl::Option{<:Number} = nothing, dfs::Option{<:Number} = nothing,
                           kwargs::NamedTuple = (; atol = 1e-8))::FeesEstimator
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
function FeesEstimator(; tn::Option{<:TnE_Tn} = nothing,
                       l::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       s::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       fl::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       fs::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       dl::Option{<:Number} = nothing, ds::Option{<:Number} = nothing,
                       dfl::Option{<:Number} = nothing, dfs::Option{<:Number} = nothing,
                       kwargs::NamedTuple = (; atol = 1e-8))::FeesEstimator
    return FeesEstimator(tn, l, s, fl, fs, dl, ds, dfl, dfs, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Charges a portfolio a turnover fee, a long or short proportional fee, and a long or short fixed fee.

Each of the three components keys on the **sign of the position**, not on the sign of the trade: an asset held long pays `l` and `fl`, an asset held short pays `s` and `fs`, whatever the direction of the rebalance that reached it. The turnover component is the one that keys on the trade, and it carries a [`Turnover`](@ref) whose `val` is read as a per-asset **fee rate** rather than as a bound.

A fixed fee is charged per position held, whatever its size. The position must be non-zero to attract one, and `kwargs` is forwarded to `isapprox` to decide how near zero counts as zero.

Fee values can be specified as scalars (applied to all assets) or as vectors of per-asset values. The portfolio fees are computed by [`calc_fees`](@ref) and asset fees by [`calc_asset_fees`](@ref).

!!! warning

    The turnover and proportional fees must match the periodicity of the returns series, and the fixed fees must be divided by the portfolio's holding period. The units of the fees and returns must also be consistent.

# Mathematical definition

## Portfolio fees

For non-finite optimisations, the total portfolio transaction fees are computed as:

```math
\\begin{align}
F_{\\text{t}}(\\boldsymbol{w}) &\\coloneqq F_{\\text{Tn}} + F_{\\text{p}} + F_{\\text{f}} \\\\
F_{\\text{Tn}}(\\boldsymbol{w}) &= \\boldsymbol{Tn} \\cdot \\boldsymbol{f}_{\\text{Tn}}\\\\
F_{\\text{p}}(\\boldsymbol{w}) &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{+} - \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
F_{\\text{f}}(\\boldsymbol{w}) &= 1\\left\\{\\boldsymbol{w} \\geq 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\cdot \\boldsymbol{f}_{\\text{f}}^{+} + 1\\left\\{\\boldsymbol{w} \\lt 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\cdot\\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

The finite optimisations use fees somewhat differently because they use a finite amount of capital as well as asset prices to compute the actual fees incurred when buying or selling assets. As such, these fees require a vector of asset prices to compute the actual fees incurred.

This method lets us automatically adjust the available cash amount during the optimisation so that fees are discounted from the available cash. It also lets us account for the budget constraints properly when fees are involved.

```math
\\begin{align}
F_{\\text{t}}(\\boldsymbol{w}) &\\coloneqq F_{\\text{Tn}} + F_{\\text{p}} + F_{\\text{f}} \\\\
F_{\\text{Tn}}(\\boldsymbol{w}) &= \\left(\\boldsymbol{Tn} \\odot \\boldsymbol{X} \\right) \\cdot \\boldsymbol{f}_{\\text{Tn}}\\\\
F_{\\text{p}}(\\boldsymbol{w}) &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{+} - \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\cdot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
F_{\\text{f}}(\\boldsymbol{w}) &= 1\\left\\{\\boldsymbol{w} \\geq 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\cdot \\boldsymbol{f}_{\\text{f}}^{+} + 1\\left\\{\\boldsymbol{w} \\lt 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\cdot \\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

The fixed term carries no ``\\boldsymbol{X}``. A fixed fee is a currency amount already, so the price vector reaches only the two terms that are stated as a fraction of the position.

## Per asset fees

It is also possible to compute per-asset fees incurred using the same definitions as above, but replacing the dot products with elementwise (Hadamard) products.

```math
\\begin{align}
\\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w}) &\\coloneqq \\boldsymbol{F}_{\\text{Tn}} + \\boldsymbol{F}_{\\text{p}} + \\boldsymbol{F}_{\\text{f}} \\\\
\\boldsymbol{F}_{\\text{Tn}}(\\boldsymbol{w}) &= \\boldsymbol{Tn} \\odot \\boldsymbol{f}_{\\text{Tn}}\\\\
\\boldsymbol{F}_{\\text{p}}(\\boldsymbol{w}) &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{+} - \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
\\boldsymbol{F}_{\\text{f}}(\\boldsymbol{w}) &= 1\\left\\{\\boldsymbol{w} \\geq 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\odot \\boldsymbol{f}_{\\text{f}}^{+} + 1\\left\\{\\boldsymbol{w} \\lt 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\odot\\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

The finite optimisation uses fees somewhat differently because it uses a finite amount of capital and utilises the asset prices to compute the actual fees incurred when buying or selling assets. As such, these fees require a vector of asset prices to compute the actual fees incurred.

```math
\\begin{align}
\\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w}) &\\coloneqq \\boldsymbol{F}_{\\text{Tn}} + \\boldsymbol{F}_{\\text{p}} + \\boldsymbol{F}_{\\text{f}} \\\\
\\boldsymbol{F}_{\\text{Tn}}(\\boldsymbol{w}) &= \\left(\\boldsymbol{Tn} \\odot \\boldsymbol{X} \\right) \\odot \\boldsymbol{f}_{\\text{Tn}} \\\\
\\boldsymbol{F}_{\\text{p}}(\\boldsymbol{w}) &= \\left(1\\left\\{\\boldsymbol{w} \\geq 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{+} - \\left(1\\left\\{\\boldsymbol{w} \\lt 0\\right\\} \\odot \\boldsymbol{w} \\odot \\boldsymbol{X}\\right) \\odot \\boldsymbol{f}_{\\text{p}}^{-} \\\\
\\boldsymbol{F}_{\\text{f}}(\\boldsymbol{w}) &= 1\\left\\{\\boldsymbol{w} \\geq 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\odot \\boldsymbol{f}_{\\text{f}}^{+} + 1\\left\\{\\boldsymbol{w} \\lt 0 \\land \\boldsymbol{w} \\neq 0\\right\\} \\odot \\boldsymbol{f}_{\\text{f}}^{-}
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``F``: Portfolio fee.
  - ``\\boldsymbol{F}``: `N × 1` per asset vector of portfolio fees.
  - ``\\boldsymbol{X}``: `N × 1` asset price vector.
  - ``\\boldsymbol{f}``: `N × 1` per asset fee vector. If it is a scalar, it is broadcasted to all assets.
  - ``\\boldsymbol{Tn}``: `N × 1` turnover vector as defined in [`Turnover`](@ref). The benchmark weight vector is encoded in the `w` field of the turnover object and the new weight vector is the portfolio weight vector.
  - ``+,\\, -``: Superscripts denote long and short fees respectively. This is because brokers sometimes charge different fees for long and short positions.
  - ``\\text{t},\\, \\text{Tn},\\, \\text{p},\\, \\text{f}``: Subscripts for total, turnover, proportional, and fixed fees respectively. The turnover fee is encoded an instance of [`Turnover`](@ref), where `val` is the per asset fee.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise (Hadamard) indicator function returning `1` when the condition is true, `0` otherwise. This activates long or short fees based on whether the asset weight is non-negative or otherwise.
  - ``\\boldsymbol{w} \\neq 0``: Read as `!isapprox(w, 0; kwargs...)`, so `kwargs` decides how near zero counts as zero. Only the fixed terms carry it: a proportional fee on a zero weight is zero anyway.
  - ``\\odot``: Elementwise (Hadamard) product.

The short proportional term is **subtracted**. ``\\boldsymbol{w}`` is negative wherever its indicator fires, so the minus sign is what makes the fee a positive charge. On ``\\boldsymbol{w} = [0.6,\\, -0.4]`` with a short rate of `0.01` and no other term, [`calc_fees`](@ref) returns `0.004`.

## The per asset fees sum to the portfolio fee

The two families compute one definition. [`calc_asset_fees`](@ref) splits over the assets what [`calc_fees`](@ref) contracts into a scalar, so the entries of the vector sum to the scalar. The sums differ in the order in which they add, so the identity holds to rounding and not to `==`. On ``\\boldsymbol{w} = [0.6,\\, -0.4,\\, 0,\\, 0.25]`` with all four rate fields set, a [`Turnover`](@ref) whose `w` differs from the candidate, and prices ``[100,\\, 50,\\, 20,\\, 10]``, both sides gave `12.530000000000001`. Without the price vector they gave `11.036000000000001` and `11.036`, a difference of `1.8e-15`.

## The JuMP model charges the same fee only when the decomposition is pinned

[`set_non_fixed_fees!`](@ref) writes the proportional terms against the model's `lw` and `sw` variables rather than against ``\\boldsymbol{w}``, and it writes no fixed term at all — a fixed fee needs a binary and is emitted by the MIP builder instead.

Under a [`PartsBoundWeights`](@ref) head those variables only *bound* the parts of ``\\boldsymbol{w}``, so the model's fee is an upper bound on this definition.

The sample that measures the gap is the last 201 rows of the first five columns of `test/assets/SP500.csv.gz`, turned into 200 returns. It is solved with [`MeanRisk`](@ref) over a [`Variance`](@ref), under `lb = -1`, `ub = 1`, `bgt = 1`, `sbgt = 1`, `l = 0.002` and `s = 0.003`. The model reported `0.007` and the functor `0.003282488224724545`, a gap of `0.0037`. The budget pins `sum(lw)` to `2` and `sum(sw)` to `sbgt`, whether or not a short position is held, so the model charges both sides in full.

Setting `xbgt = true` on the [`JuMPOptimiser`](@ref) pins the decomposition. It writes binaries, so the same problem then needs a mixed-integer conic solver rather than a conic one. On that sample the model reported `0.0069999999999999975` and the functor `0.006999999999999652`, a difference of `3.5e-16`.

A long-only model needs no pinning. With `lb = 0`, `bgt = 1` and `l = 0.002` as the only fee, the model and the functor both reported `0.002`, and the difference was exactly zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Fees(;
        tn::Option{<:Turnover} = nothing,
        l::Option{<:Num_VecNum} = nothing,
        s::Option{<:Num_VecNum} = nothing,
        fl::Option{<:Num_VecNum} = nothing,
        fs::Option{<:Num_VecNum} = nothing,
        kwargs::NamedTuple = (; atol = 1e-8)
    ) -> Fees

Keywords correspond to the struct's fields.

## Validation

  - `l`, `s`, `fl`, `fs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `tn`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `tn`: Recursively viewed via [`port_opt_view`](@ref).
  - `l`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `s`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `fl`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `fs`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> Fees(; tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.0, 0.0]), l = [0.001, 0.002, 0.0],
            s = [0.001, 0.002, 0.0], fl = [5.0, 0.0, 0.0], fs = [0.0, 10.0, 0.0])
Fees
      tn ┼ Turnover
         │       w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │     val ┼ Vector{Float64}: [0.1, 0.0, 0.0]
         │   fixed ┴ Bool: false
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ Vector{Float64}: [0.001, 0.002, 0.0]
      fl ┼ Vector{Float64}: [5.0, 0.0, 0.0]
      fs ┼ Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
```

# Related

  - [`add_to_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
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
  - [`set_non_fixed_fees!`](@ref)
  - [`PartsBoundWeights`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`MeanRisk`](@ref)
  - [`Variance`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.14. The book states the linear variable transaction cost, which is the turnover term with equal rates on the two sides of the trade. The long/short proportional and fixed fees here key on the sign of the **position** rather than of the trade, and are a generalisation of it.
"""
@propagatable @concrete struct Fees <: AbstractResult
    """
    $(field_dict[:tnr])
    """
    @fprop @vprop tn
    """
    $(field_dict[:l_fees])
    """
    @vprop l
    """
    $(field_dict[:s_fees])
    """
    @vprop s
    """
    $(field_dict[:fl])
    """
    @vprop fl
    """
    $(field_dict[:fs])
    """
    @vprop fs
    """
    $(field_dict[:kwargs_fee])
    """
    kwargs
    function Fees(tn::Option{<:Turnover}, l::Option{<:Num_VecNum}, s::Option{<:Num_VecNum},
                  fl::Option{<:Num_VecNum}, fs::Option{<:Num_VecNum},
                  kwargs::NamedTuple = (; atol = 1e-8))::Fees
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
              fs::Option{<:Num_VecNum} = nothing,
              kwargs::NamedTuple = (; atol = 1e-8))::Fees
    return Fees(tn, l, s, fl, fs, kwargs)
end
"""
    const FeesE_Fees = Union{<:Fees, <:FeesEstimator}

Union type for fee constraint objects and estimators.

There is no vector counterpart, and [`fees_constraints`](@ref) has no vector method. [`Fees`](@ref) already sums every fee component into one object, so an optimiser holds exactly one. See [`RkbE_Rkb`](@ref) for why some constraint families are singular and others are not.

# Related

  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
"""
const FeesE_Fees = Union{<:Fees, <:FeesEstimator}
"""
    needs_previous_weights(fe::FeesE_Fees) -> Bool

Check if a fee constraint or estimator requires previous portfolio weights by calling [`needs_previous_weights`](@ref) on `fe.tn`.

Only the turnover term reads a previous weight vector. The proportional and fixed terms key on the sign of the position that `w` already carries, so a [`Fees`](@ref) whose `tn` is `nothing` needs none.

# Algorithm

 1. Read `fe.tn` and forward it to [`needs_previous_weights`](@ref), which answers `!tn.fixed` on a turnover object and `false` on `nothing`.

# Arguments

  - `fe`: Fee constraint or estimator.

# Returns

  - `Bool`: `true` if previous weights are needed, `false` otherwise.

# Related

  - [`FeesEstimator`](@ref)
  - [`Fees`](@ref)
"""
function needs_previous_weights(fe::FeesE_Fees)::Bool
    return needs_previous_weights(fe.tn)
end
"""
    fees_constraints(fees::FeesEstimator, sets::UniverseSets; datatype::DataType = Float64,
                     strict::Bool = false)

Resolve the name-keyed fee fields of a [`FeesEstimator`](@ref) against a universe, giving a [`Fees`](@ref) of plain per-asset vectors.

Nine fields carry the specification and each of the four proportional and fixed fields draws its gaps from its **own** default: `l` from `dl`, `s` from `ds`, `fl` from `dfl` and `fs` from `dfs`. A default never fills a neighbour's field. The nested `tn` resolves through [`turnover_constraints`](@ref), so a [`FeesEstimator`](@ref) holding a [`TurnoverEstimator`](@ref) returns a [`Fees`](@ref) holding a [`Turnover`](@ref).

# Algorithm

 1. Resolve `fees.tn` against the universe of `sets` with [`turnover_constraints`](@ref), giving a [`Turnover`](@ref) whose `val` is one turnover fee rate per asset. A `nothing` `tn` stays `nothing`.
 2. Resolve `fees.l` with [`estimator_to_val`](@ref), giving `l`, one long proportional rate per asset in the order of the universe. Every asset the keys miss takes `fees.dl`, or `zero(datatype)` when `fees.dl` is `nothing`. A `nothing` `fees.l` stays `nothing`.
 3. Resolve `fees.s` the same way against `fees.ds`, giving `s`.
 4. Resolve `fees.fl` the same way against `fees.dfl`, giving `fl`.
 5. Resolve `fees.fs` the same way against `fees.dfs`, giving `fs`.
 6. Build a [`Fees`](@ref) from the five resolved fields and `fees.kwargs`, which reaches the result unchanged and sets the boundary the fixed terms read.

# Arguments

  - `fees`: [`FeesEstimator`](@ref) specifying turnover, proportional, and fixed fee values.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for fee values.
  - `strict`: If `true`, enforces strict matching between assets and fee values (throws error on mismatch); if `false`, issues a warning.

# Validation

  - A key that names neither an asset nor a group of `sets` raises an `ArgumentError` when `strict` is `true`, and warns otherwise. Steps 1 to 5 each check their own field, so one bad key in `l` raises whatever `s`, `fl` and `fs` hold.

# Returns

  - `fe::Fees`: Object containing turnover, proportional, and fixed fee values aligned with `sets`.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> fees = FeesEstimator(;
                            tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1),
                                                   dval = 0.0),
                            l = Dict(\"A\" => 0.001, \"B\" => 0.002), s = [\"A\" => 0.001, \"B\" => 0.002],
                            fl = Dict(\"A\" => 5.0), fs = [\"B\" => 10.0]);

julia> fees_constraints(fees, sets)
Fees
      tn ┼ Turnover
         │       w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │     val ┼ Vector{Float64}: [0.1, 0.0, 0.0]
         │   fixed ┴ Bool: false
       l ┼ Vector{Float64}: [0.001, 0.002, 0.0]
       s ┼ Vector{Float64}: [0.001, 0.002, 0.0]
      fl ┼ Vector{Float64}: [5.0, 0.0, 0.0]
      fs ┼ Vector{Float64}: [0.0, 10.0, 0.0]
  kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)

julia> fees = FeesEstimator(;
                            tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1),
                                                   dval = 0.0, fixed = true),
                            l = Dict(\"A\" => 0.001, \"B\" => 0.002), s = [\"A\" => 0.001, \"B\" => 0.002],
                            fl = Dict(\"A\" => 5.0), fs = [\"B\" => 10.0]);

julia> fees_constraints(fees, sets)
Fees
      tn ┼ Turnover
         │       w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │     val ┼ Vector{Float64}: [0.1, 0.0, 0.0]
         │   fixed ┴ Bool: true
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
  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
"""
function fees_constraints(fees::FeesEstimator, sets::UniverseSets;
                          datatype::DataType = Float64, strict::Bool = false)::Fees
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
                                      strict = strict), kwargs = fees.kwargs)
end
"""
    fees_constraints(fees::Option{<:Fees}, args...; kwargs...)

Propagate or pass through portfolio transaction fee constraints.

`fees_constraints` returns the input [`Fees`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed fee constraints or missing constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Algorithm

 1. Return `fees`. A [`Fees`](@ref) already carries one rate per asset, so no universe is resolved. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `fees`: An existing [`Fees`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `fe::Option{<:Fees}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> fees = Fees(; tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.0, 0.0]),
                   l = [0.001, 0.002, 0.0]);

julia> fees_constraints(fees)
Fees
      tn ┼ Turnover
         │       w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
         │     val ┼ Vector{Float64}: [0.1, 0.0, 0.0]
         │   fixed ┴ Bool: false
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
  - [`Option`](@ref)
"""
function fees_constraints(fees::Option{<:Fees}, args...; kwargs...)::Option{<:Fees}
    return fees
end
"""
    calc_fees(w::VecNum, p::VecNum, ::Nothing, ::Function)
    calc_fees(w::VecNum, p::VecNum, fees::Number, op::Function)
    calc_fees(w::VecNum, p::VecNum, fees::VecNum, op::Function)

Compute the actual proportional fees for portfolio weights and prices.

This is one term of the total fee, not the whole fee. [`calc_fees(w::VecNum, p::VecNum, fees::Fees)`](@ref) calls it twice, under `.>=` for the long side and under `.<` for the short side, and negates the short call. [`Fees`](@ref) states the closed form as ``F_{\\text{p}}``.

# Algorithm

 1. On a `nothing` `fees`, return `zero(promote_type(eltype(w), eltype(p)))`. The method reads neither `w` nor `op`.
 2. Otherwise build `idx`, the mask of the assets that `op` selects against a zero of the promoted element type.
 3. On a `Number` `fees`, contract the selected weights with the selected prices, and scale that sum by the one rate.
 4. On a `VecNum` `fees`, contract the selected rates with the selected weights multiplied elementwise by the selected prices.

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
    return fees * LinearAlgebra.dot(w[idx], p[idx])
end
function calc_fees(w::VecNum, p::VecNum, fees::VecNum, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return LinearAlgebra.dot(fees[idx], w[idx] .* p[idx])
end
"""
    calc_fees(w::VecNum, p::VecNum, ::Nothing)
    calc_fees(w::VecNum, p::VecNum, tn::Turnover)

Compute the actual turnover fees for portfolio weights and prices.

This is one term of the total fee, not the whole fee. [`Fees`](@ref) states the closed form as ``F_{\\text{Tn}}``, and reads `tn.val` as a per-asset fee rate rather than as a bound. The `fixed` flag of [`Turnover`](@ref) reaches no method here: it decides which reference weights `tn.w` holds, through [`factory`](@ref), and by the time this method runs `tn.w` is already the vector the fee must be charged against.

# Algorithm

 1. On a `nothing` `tn`, return `zero(promote_type(eltype(w), eltype(p)))`. The method reads neither `w` nor `p`.
 2. Otherwise form the traded amount per asset, the absolute difference between `w` and the reference weights `tn.w`.
 3. On a `Number` `tn.val`, contract the traded amount with the prices, and scale that sum by the one rate.
 4. On a `VecNum` `tn.val`, contract the rates with the traded amount multiplied elementwise by the prices.

Steps 3 and 4 are not the same expression. They agree to rounding when `tn.val` is a constant vector, and they differed by `-2.22e-16` on `w = [0.6, -0.4, 0.0, 0.25]`, `p = [100.0, 50.0, 20.0, 10.0]`, `tn.w = [0.1, 0.2, 0.3, 0.4]` and a rate of `0.02`.

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
julia> calc_fees([0.1, 0.2], [100, 200], Turnover(; w = [0.0, 0.0], val = 0.01))
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
    return tn.val * LinearAlgebra.dot(abs.(w - tn.w), p)
end
function calc_fees(w::VecNum, p::VecNum, tn::Turnover{<:Any, <:VecNum})
    return LinearAlgebra.dot(tn.val, abs.(w - tn.w) .* p)
end
"""
    calc_fees(w::VecNum, p::VecNum, fees::Fees)

Compute total actual fees for portfolio weights and prices.

Sums actual proportional, fixed, and turnover fees for all assets. [`calc_asset_fees(w::VecNum, p::VecNum, fees::Fees)`](@ref) splits the same total over the assets, and its sum is this number up to the order of summation.

# Algorithm

 1. Charge the long proportional term `fees_long`, the call of [`calc_fees(w::VecNum, p::VecNum, fees::Number, op::Function)`](@ref) on `fees.l` under `.>=`.
 2. Charge the short proportional term `fees_short`, the negated call of the same name on `fees.s` under `.<`. `w` is negative on that side, so the negation is what makes the term a positive charge.
 3. Charge the long fixed term `fees_fixed_long`, the call of [`calc_fixed_fees`](@ref) on `fees.fl` under `.>=`. It carries no price, because a fixed fee is a currency amount already.
 4. Charge the short fixed term `fees_fixed_short`, the call of the same name on `fees.fs` under `.<`.
 5. Charge the turnover term `fees_turnover`, the call of [`calc_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref) on `fees.tn`.
 6. Return the sum of the five terms.

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

Compute the proportional fees for portfolio weights.

This is one term of the total fee, not the whole fee. [`calc_fees(w::VecNum, fees::Fees)`](@ref) calls it twice, under `.>=` for the long side and under `.<` for the short side, and negates the short call. [`Fees`](@ref) states the closed form as ``F_{\\text{p}}``, in the pair of equations that carries no price vector.

# Algorithm

 1. On a `nothing` `fees`, return `zero(eltype(w))`. The method reads neither `w` nor `op`.
 2. Otherwise build `idx`, the mask of the assets that `op` selects against a zero of the promoted element type.
 3. On a `Number` `fees`, scale the selected weights by the one rate, and sum them.
 4. On a `VecNum` `fees`, contract the selected rates with the selected weights.

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
    return LinearAlgebra.dot(fees[idx], w[idx])
end
"""
    calc_fees(w::VecNum, ::Nothing)
    calc_fees(w::VecNum, tn::Turnover)

Compute the turnover fees for portfolio weights.

This is one term of the total fee, not the whole fee. [`Fees`](@ref) states the closed form as ``F_{\\text{Tn}}``, in the pair of equations that carries no price vector. The `fixed` flag of [`Turnover`](@ref) reaches no method here, for the reason [`calc_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref) gives.

# Algorithm

 1. On a `nothing` `tn`, return `zero(eltype(w))`. The method reads `w` only for its element type.
 2. Otherwise form the traded amount per asset, the absolute difference between `w` and the reference weights `tn.w`.
 3. On a `Number` `tn.val`, sum the traded amount and scale it by the one rate.
 4. On a `VecNum` `tn.val`, contract the rates with the traded amount.

Steps 3 and 4 are not the same expression. They differed by `3.47e-18` on the sample that [`calc_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref) names.

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
julia> calc_fees([0.8, 0.2], Turnover(; w = [0.0, 0.0], val = 0.02))
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
    return LinearAlgebra.dot(tn.val, abs.(w - tn.w))
end
"""
    calc_fixed_fees(w::VecNum, ::Nothing, kwargs::NamedTuple, ::Function)
    calc_fixed_fees(w::VecNum, fees::Number, kwargs::NamedTuple, op::Function)
    calc_fixed_fees(w::VecNum, fees::VecNum, kwargs::NamedTuple, op::Function)

Compute the fixed portfolio fees for assets that have been allocated.

A fixed fee is charged per position held, whatever its size, so no price vector reaches this name: the fee is a currency amount already. [`Fees`](@ref) states the closed form as ``F_{\\text{f}}``, which is the one term that carries no ``\\boldsymbol{X}`` in either pair of equations.

# Algorithm

 1. On a `nothing` `fees`, return `zero(eltype(w))`. The method reads neither `kwargs` nor `op`.
 2. Otherwise build `idx1`, the mask of the assets that `op` selects against a zero of the promoted element type.
 3. Build `idx2`, marking the selected positions that `isapprox` does not call zero. `kwargs` is forwarded to `isapprox`, so its `atol` sets the boundary. Under the default `atol = 1e-8` a weight of `1e-9` attracts no fee and a weight of `1e-7` attracts one.
 4. On a `Number` `fees`, scale the count of the positions that `idx2` marks by the one rate.
 5. On a `VecNum` `fees`, sum the rates of the positions that `idx2` marks.

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

Compute total fees for portfolio weights.

Sums proportional, fixed, and turnover fees for all assets. [`calc_asset_fees(w::VecNum, fees::Fees)`](@ref) splits the same total over the assets, and its sum is this number up to the order of summation.

# Algorithm

 1. Charge the long proportional term `fees_long`, the call of [`calc_fees(w::VecNum, fees::Number, op::Function)`](@ref) on `fees.l` under `.>=`.
 2. Charge the short proportional term `fees_short`, the negated call of the same name on `fees.s` under `.<`. `w` is negative on that side, so the negation is what makes the term a positive charge.
 3. Charge the long fixed term `fees_fixed_long`, the call of [`calc_fixed_fees`](@ref) on `fees.fl` under `.>=`.
 4. Charge the short fixed term `fees_fixed_short`, the call of the same name on `fees.fs` under `.<`.
 5. Charge the turnover term `fees_turnover`, the call of [`calc_fees(w::VecNum, tn::Turnover)`](@ref) on `fees.tn`.
 6. Return the sum of the five terms.

# Arguments

  - `w`: Portfolio weights.
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

This is one term of the total fee, not the whole fee. It is the elementwise form of [`calc_fees(w::VecNum, p::VecNum, fees::Number, op::Function)`](@ref), and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{p}}``.

# Algorithm

 1. Allocate `fees_w`, a vector of zeros one entry long per asset, in the promoted element type. An asset the mask of step 2 leaves out keeps its zero.
 2. On a `nothing` `fees`, return `fees_w`. The method reads neither `w` nor `op` beyond their element types.
 3. Otherwise build `idx`, the mask of the assets that `op` selects against a zero of the promoted element type.
 4. On a `Number` `fees`, write the selected weights, multiplied elementwise by the selected prices and scaled by the one rate, into the selected entries of `fees_w`.
 5. On a `VecNum` `fees`, write the same product, weighted by the selected per-asset rates, into the selected entries of `fees_w`.

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

This is one term of the total fee, not the whole fee. It is the elementwise form of [`calc_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref), and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{Tn}}``. The `fixed` flag of [`Turnover`](@ref) reaches no method here, for the reason [`calc_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref) gives.

# Algorithm

 1. On a `nothing` `tn`, return a vector of zeros one entry long per asset, in the promoted element type.
 2. Otherwise form the traded amount per asset, the absolute difference between `w` and the reference weights `tn.w`.
 3. On a `Number` `tn.val`, multiply the traded amount elementwise by the prices, and scale it by the one rate.
 4. On a `VecNum` `tn.val`, multiply the traded amount elementwise by the prices and by the per-asset rates.

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
julia> calc_asset_fees([0.1, 0.2], [100, 200], Turnover(; w = [0.0, 0.0], val = 0.01))
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

Sums actual proportional, fixed, and turnover fees for all assets. The entries sum to the number [`calc_fees(w::VecNum, p::VecNum, fees::Fees)`](@ref) returns, up to the order of summation.

# Algorithm

 1. Charge the long proportional term `fees_long`, the call of [`calc_asset_fees(w::VecNum, p::VecNum, fees::Number, op::Function)`](@ref) on `fees.l` under `.>=`.
 2. Charge the short proportional term `fees_short`, the negated call of the same name on `fees.s` under `.<`. `w` is negative on that side, so the negation is what makes the term a positive charge.
 3. Charge the long fixed term `fees_fixed_long`, the call of [`calc_asset_fixed_fees`](@ref) on `fees.fl` under `.>=`. It carries no price, because a fixed fee is a currency amount already.
 4. Charge the short fixed term `fees_fixed_short`, the call of the same name on `fees.fs` under `.<`.
 5. Charge the turnover term `fees_turnover`, the call of [`calc_asset_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref) on `fees.tn`.
 6. Return the elementwise sum of the five vectors.

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

Compute the proportional per asset fees for portfolio weights.

This is one term of the total fee, not the whole fee. It is the elementwise form of [`calc_fees(w::VecNum, fees::Number, op::Function)`](@ref), and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{p}}``, in the pair of equations that carries no price vector.

# Algorithm

 1. Allocate `fees_w`, a vector of zeros one entry long per asset, in the promoted element type. An asset the mask of step 3 leaves out keeps its zero.
 2. On a `nothing` `fees`, return `fees_w`. The method reads neither `w` nor `op` beyond the element type of `w`.
 3. Otherwise build `idx`, the mask of the assets that `op` selects against a zero of the promoted element type.
 4. On a `Number` `fees`, write the selected weights, scaled by the one rate, into the selected entries of `fees_w`.
 5. On a `VecNum` `fees`, write the selected weights, multiplied elementwise by the selected rates, into the selected entries of `fees_w`.

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

Compute the per asset turnover fees for portfolio weights.

This is one term of the total fee, not the whole fee. It is the elementwise form of [`calc_fees(w::VecNum, tn::Turnover)`](@ref), and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{Tn}}``, in the pair of equations that carries no price vector. The `fixed` flag of [`Turnover`](@ref) reaches no method here, for the reason [`calc_fees(w::VecNum, p::VecNum, tn::Turnover)`](@ref) gives.

# Algorithm

 1. On a `nothing` `tn`, return a vector of zeros one entry long per asset, in the element type of `w`.
 2. Otherwise form the traded amount per asset, the absolute difference between `w` and the reference weights `tn.w`.
 3. On a `Number` `tn.val`, scale the traded amount by the one rate.
 4. On a `VecNum` `tn.val`, multiply the traded amount elementwise by the per-asset rates.

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
julia> calc_asset_fees([0.1, 0.2], Turnover(; w = [0.0, 0.0], val = 0.01))
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

This is the elementwise form of [`calc_fixed_fees`](@ref), and its entries sum to the number that name returns. No price vector reaches it, and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{f}}``.

# Algorithm

 1. Allocate `fees_w`, a vector of zeros one entry long per asset, in the promoted element type. An asset the masks of steps 3 and 4 leave out keeps its zero.
 2. On a `nothing` `fees`, return `fees_w`. The method reads neither `kwargs` nor `op`.
 3. Otherwise build `idx1`, the mask of the assets that `op` selects against a zero of the promoted element type.
 4. Build `idx2`, marking the selected positions that `isapprox` does not call zero. `kwargs` is forwarded to `isapprox`, so its `atol` sets the boundary.
 5. On a `Number` `fees`, write the one rate, gated by `idx2`, into the selected entries of `fees_w`.
 6. On a `VecNum` `fees`, write the selected per-asset rates, gated by `idx2`, into the selected entries of `fees_w`.

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
    fees_w[idx1] = fees[idx1] ⊙ idx2
    return fees_w
end
"""
    calc_asset_fees(w::VecNum, fees::Fees)

Compute total per asset fees for portfolio weights.

Sums proportional, fixed, and turnover fees for all assets. The entries sum to the number [`calc_fees(w::VecNum, fees::Fees)`](@ref) returns, up to the order of summation.

# Algorithm

 1. Charge the long proportional term `fees_long`, the call of [`calc_asset_fees(w::VecNum, fees::Number, op::Function)`](@ref) on `fees.l` under `.>=`.
 2. Charge the short proportional term `fees_short`, the negated call of the same name on `fees.s` under `.<`. `w` is negative on that side, so the negation is what makes the term a positive charge.
 3. Charge the long fixed term `fees_fixed_long`, the call of [`calc_asset_fixed_fees`](@ref) on `fees.fl` under `.>=`.
 4. Charge the short fixed term `fees_fixed_short`, the call of the same name on `fees.fs` under `.<`.
 5. Charge the turnover term `fees_turnover`, the call of [`calc_asset_fees(w::VecNum, tn::Turnover)`](@ref) on `fees.tn`.
 6. Return the elementwise sum of the five vectors.

# Arguments

  - `w`: Portfolio weights.
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
