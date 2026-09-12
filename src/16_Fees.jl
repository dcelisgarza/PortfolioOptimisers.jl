"""
$(DocStringExtensions.TYPEDEF)

Supertype for the algorithms that name the clock a one-off fee charge falls on.

[`Fees`](@ref) and [`FeesEstimator`](@ref) each carry this family in their `fa` field, bound to `Option{<:AbstractFeeAmortisation}`. The field decides where the two fixed charges `fl` and `fs` land on a return series, and it reaches no other term, because `l`, `s` and `tn` are rates per period and charge on every observation. The family has two leaves. [`FirstObservationFees`](@ref) charges the two fixed amounts one time, on the first observation, and [`AmortisedFees`](@ref) spreads them evenly over a horizon instead. A `nothing` `fa` is the default, and it names the first-observation clock.

Every site that reads the field dispatches on the leaf it holds rather than on this supertype, so a third clock added to the family gets a `MethodError` until its own methods are written. The supertype names the question, and it decides no answer.

The cross-validation schemes carry the same family in a field of the same name, where it overrides the fee's own clock for a fold's realised series. [`fold_evaluation`](@ref) reads it, and there a `nothing` inherits the fee's clock rather than naming one.

# Related

  - [`AmortisedFees`](@ref)
  - [`FirstObservationFees`](@ref)
  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
  - [`fold_evaluation`](@ref)
  - [`Option`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractFeeAmortisation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Spreads the one-off terms of a fee, the two fixed charges `fl` and `fs`, evenly over a holding period.

The algorithm carries no number. Every site that charges a fee knows the observation count it charges over, and hands it in: [`charge_fees`](@ref) hands the length of the series, [`calc_total_fees`](@ref) takes the horizon as an argument, and the model hands the observation count of the fit. So the count of the holding period is never stored, and never stale.

[`FirstObservationFees`](@ref) names the other clock, which charges the two fixed terms one time, on the first observation. A `nothing` `fa` names that clock too.

The turnover charge `tn` is a rate per period, so it charges on every observation beside `l` and `s`. This algorithm never divides it.

# Constructors

    AmortisedFees() -> AmortisedFees

# Examples

```jldoctest
julia> AmortisedFees()
AmortisedFees()
```

# Related

  - [`AbstractFeeAmortisation`](@ref)
  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
  - [`calc_fees`](@ref)
  - [`charge_fees`](@ref)
"""
struct AmortisedFees <: AbstractFeeAmortisation end
"""
$(DocStringExtensions.TYPEDEF)

Charges the one-off terms of a fee, the two fixed charges `fl` and `fs`, on the first observation of a return series.

The algorithm carries no number, and it names the clock a `nothing` `fa` names. On a [`Fees`](@ref) it is therefore a synonym for `nothing`, and it exists so that a caller who must *state* this clock has a word for it. A cross-validation scheme's `fa` field is that caller: `nothing` there means inherit the fee's own clock, so the two answers a `Fees` spells one way each need two words.

The turnover charge `tn` is a rate per period, so it charges on every observation beside `l` and `s`. This algorithm never moves it.

# Constructors

    FirstObservationFees() -> FirstObservationFees

# Examples

```jldoctest
julia> FirstObservationFees()
FirstObservationFees()
```

# Related

  - [`AbstractFeeAmortisation`](@ref)
  - [`AmortisedFees`](@ref)
  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
  - [`calc_fees`](@ref)
  - [`charge_fees`](@ref)
  - [`fold_evaluation`](@ref)
"""
struct FirstObservationFees <: AbstractFeeAmortisation end
"""
$(DocStringExtensions.TYPEDEF)

Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe.

Every fee field accepts a dictionary, a pair, or a vector of pairs keyed by asset or group name, and the matching `d*` field fills every asset the keys miss. Each default fills only its own field: `l` draws on `dl`, `s` on `ds`, `fl` on `dfl` and `fs` on `dfs`, and never on a neighbour's. [`fees_constraints`](@ref) resolves the names against a [`UniverseSets`](@ref) and returns a [`Fees`](@ref), whose fee fields are plain per-asset vectors and whose `kwargs` is the `kwargs` of this estimator.

!!! warning

    `l`, `s` and `tn` are rates per period, and each of them charges on every observation of a return series. `fa` reaches none of the three. `fl` and `fs` charge each non-zero position one time for the whole holding period. A return series reads them as a fraction of capital, and the finite allocation reads them as a currency amount. `fa` names the clock they fall on, and a `nothing` `fa` charges them on the first observation. The units of the fees and returns must also be consistent.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeesEstimator(;
        tn::Option{<:TnE_Tn} = nothing,
        l::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        s::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        fl::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        fs::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
        lq::Option{<:TnE_Tn} = nothing,
        flq::Option{<:TnE_Tn} = nothing,
        dl::Option{<:Number} = nothing,
        ds::Option{<:Number} = nothing,
        dfl::Option{<:Number} = nothing,
        dfs::Option{<:Number} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        kwargs::NamedTuple = (; atol = 1e-8)
    ) -> FeesEstimator

Keywords correspond to the struct's fields.

## Validation

  - `l`, `s`, `fl`, `fs`, `dl`, `ds`, `dfl`, `dfs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `tn`: Recursively updated via [`factory`](@ref).
  - `lq`: Recursively updated via [`factory`](@ref).
  - `flq`: Recursively updated via [`factory`](@ref).

## View parameters

This type spans **two axes**, so [`port_opt_view`](@ref) is written by hand for it rather than generated from `@vprop` tags, and no field carries one. `tn`, `l`, `s`, `fl` and `fs` price the positions the portfolio holds, so they are sliced to the selected indices. `lq` and `flq` price the positions it is forced to sell, so they are sliced to the **complement** of those indices, which the verb derives from the width of the unreduced returns matrix it is handed:

  - `tn`: Recursively viewed at the selected indices via [`port_opt_view`](@ref).
  - `l`: Sliced to the selected indices.
  - `s`: Sliced to the selected indices.
  - `fl`: Sliced to the selected indices.
  - `fs`: Sliced to the selected indices.
  - `lq`: Recursively viewed at the **complement** of the selected indices.
  - `flq`: Recursively viewed at the **complement** of the selected indices.

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
      lq ┼ nothing
     flq ┼ nothing
      dl ┼ nothing
      ds ┼ nothing
     dfl ┼ nothing
     dfs ┼ nothing
      fa ┼ nothing
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
  - [`AbstractFeeAmortisation`](@ref)
  - [`AmortisedFees`](@ref)
  - [`fees_constraints`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct FeesEstimator <: AbstractEstimator
    """
    $(field_dict[:tn_fees])
    """
    @fprop tn
    """
    $(field_dict[:l_fees])
    """
    l
    """
    $(field_dict[:s_fees])
    """
    s
    """
    $(field_dict[:fl])
    """
    fl
    """
    $(field_dict[:fs])
    """
    fs
    """
    $(field_dict[:lq_fees])
    """
    @fprop lq
    """
    $(field_dict[:flq_fees])
    """
    @fprop flq
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
    $(field_dict[:fa_fees])
    """
    fa
    """
    $(field_dict[:kwargs_fee])
    """
    kwargs
    function FeesEstimator(tn::Option{<:TnE_Tn},
                           l::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           s::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           fl::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           fs::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                           lq::Option{<:TnE_Tn} = nothing, flq::Option{<:TnE_Tn} = nothing,
                           dl::Option{<:Number} = nothing, ds::Option{<:Number} = nothing,
                           dfl::Option{<:Number} = nothing, dfs::Option{<:Number} = nothing,
                           fa::Option{<:AbstractFeeAmortisation} = nothing,
                           kwargs::NamedTuple = (; atol = 1e-8))::FeesEstimator
        assert_nonempty_nonneg_finite_val(l, :l)
        assert_nonempty_nonneg_finite_val(s, :s)
        assert_nonempty_nonneg_finite_val(fl, :fl)
        assert_nonempty_nonneg_finite_val(fs, :fs)
        assert_nonempty_nonneg_finite_val(dl, :dl)
        assert_nonempty_nonneg_finite_val(ds, :ds)
        assert_nonempty_nonneg_finite_val(dfl, :dfl)
        assert_nonempty_nonneg_finite_val(dfs, :dfs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs), typeof(lq),
                   typeof(flq), typeof(dl), typeof(ds), typeof(dfl), typeof(dfs),
                   typeof(fa), typeof(kwargs)}(tn, l, s, fl, fs, lq, flq, dl, ds, dfl, dfs,
                                               fa, kwargs)
    end
end
function FeesEstimator(; tn::Option{<:TnE_Tn} = nothing,
                       l::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       s::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       fl::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       fs::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = nothing,
                       lq::Option{<:TnE_Tn} = nothing, flq::Option{<:TnE_Tn} = nothing,
                       dl::Option{<:Number} = nothing, ds::Option{<:Number} = nothing,
                       dfl::Option{<:Number} = nothing, dfs::Option{<:Number} = nothing,
                       fa::Option{<:AbstractFeeAmortisation} = nothing,
                       kwargs::NamedTuple = (; atol = 1e-8))::FeesEstimator
    return FeesEstimator(tn, l, s, fl, fs, lq, flq, dl, ds, dfl, dfs, fa, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Charges a portfolio a turnover fee, a long or short proportional fee, and a long or short fixed fee.

Each of the three components keys on the **sign of the position**, not on the sign of the trade: an asset held long pays `l` and `fl`, an asset held short pays `s` and `fs`, whatever the direction of the rebalance that reached it. The turnover component is the one that keys on the trade, and it carries a [`Turnover`](@ref) whose `val` is read as a per-asset **fee rate** rather than as a bound.

A fixed fee is charged per position held, whatever its size. The position must be non-zero to attract one, and `kwargs` is forwarded to `isapprox` to decide how near zero counts as zero.

Fee values can be specified as scalars (applied to all assets) or as vectors of per-asset values. The portfolio fees are computed by [`calc_fees`](@ref) and asset fees by [`calc_asset_fees`](@ref).

!!! warning

    `l`, `s` and `tn` are rates per period, and each of them charges on every observation of a return series. `fa` reaches none of the three. `fl` and `fs` charge each non-zero position one time for the whole holding period. A return series reads them as a fraction of capital, and the finite allocation reads them as a currency amount. `fa` names the clock they fall on, and a `nothing` `fa` charges them on the first observation. The units of the fees and returns must also be consistent.

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

The finite allocation charges the same three terms on the money it actually buys. It holds the share counts and the prices, so ``\\boldsymbol{x} \\odot \\boldsymbol{p}`` is the money in each position exactly. [`set_allocation_fees!`](@ref) writes the terms in the allocation's own variables, and the charge enters the budget constraint rather than the cash. A return series reads `fl` and `fs` as a fraction of capital, and the allocation reads them as a currency amount.

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

  - $(math_dict[:w_port])
  - ``F``: Portfolio fee.
  - ``\\boldsymbol{F}``: `N × 1` per asset vector of portfolio fees.
  - ``\\boldsymbol{f}``: `N × 1` per asset fee vector. If it is a scalar, it is broadcasted to all assets.
  - ``\\boldsymbol{Tn}``: `N × 1` turnover vector as defined in [`Turnover`](@ref). The benchmark weight vector is encoded in the `w` field of the turnover object and the new weight vector is the portfolio weight vector.
  - ``+,\\, -``: Superscripts denote long and short fees respectively. This is because brokers sometimes charge different fees for long and short positions.
  - ``\\text{t},\\, \\text{Tn},\\, \\text{p},\\, \\text{f}``: Subscripts for total, turnover, proportional, and fixed fees respectively. The turnover fee is encoded an instance of [`Turnover`](@ref), where `val` is the per asset fee.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise (Hadamard) indicator function returning `1` when the condition is true, `0` otherwise. This activates long or short fees based on whether the asset weight is non-negative or otherwise.
  - ``\\boldsymbol{w} \\neq 0``: Read as `!isapprox(w, 0; kwargs...)`, so `kwargs` decides how near zero counts as zero. Only the fixed terms carry it: a proportional fee on a zero weight is zero anyway.
  - ``\\odot``: Elementwise (Hadamard) product.

The short proportional term is **subtracted**. ``\\boldsymbol{w}`` is negative wherever its indicator fires, so the minus sign is what makes the fee a positive charge. On ``\\boldsymbol{w} = [0.6,\\, -0.4]`` with a short rate of `0.01` and no other term, [`calc_fees`](@ref) returns `0.004`.

## The per asset fees sum to the portfolio fee

The two families compute one definition. [`calc_asset_fees`](@ref) splits over the assets what [`calc_fees`](@ref) contracts into a scalar, so the entries of the vector sum to the scalar. The sums differ in the order in which they add, so the identity holds to rounding and not to `==`. On ``\\boldsymbol{w} = [0.6,\\, -0.4,\\, 0,\\, 0.25]`` with all four rate fields set and a [`Turnover`](@ref) whose `w` differs from the candidate, they gave `11.036000000000001` and `11.036`, a difference of `1.8e-15`.

## The JuMP model charges the same fee only when the decomposition is pinned

[`set_non_fixed_fees!`](@ref) writes the proportional terms against the model's `lw` and `sw` variables rather than against ``\\boldsymbol{w}``, and it writes no fixed term at all — a fixed fee needs a binary and is emitted by the MIP builder instead.

Under a [`PartsBoundWeights`](@ref) head those variables only *bound* the parts of ``\\boldsymbol{w}``, so the model's fee is an upper bound on this definition.

The sample that measures the gap is the last 201 rows of the first five columns of `test/assets/SP500.csv.gz`, turned into 200 returns. It is solved with [`MeanRisk`](@ref) over a [`Variance`](@ref), under `lb = -1`, `ub = 1`, `bgt = 1`, `sbgt = 1`, `l = 0.002` and `s = 0.003`. The model reported `0.007` and the functor `0.003282488224724545`, a gap of `0.0037`. The budget pins `sum(lw)` to `2` and `sum(sw)` to `sbgt`, whether or not a short position is held, so the model charges both sides in full.

Setting `xbgt = true` on the [`JuMPOptimiser`](@ref) pins the decomposition. It writes binaries, so the same problem then needs a mixed-integer conic solver rather than a conic one. On that sample the model reported `0.0069999999999999975` and the functor `0.006999999999999652`, a difference of `3.5e-16`.

A long-only model needs no pinning. With `lb = 0`, `bgt = 1` and `l = 0.002` as the only fee, the model and the functor both reported `0.002`, and the difference was exactly zero.

## Fee amortisation

`l`, `s` and `tn` are rates per period. Each of them charges one time per observation of a return series, and `fa` reaches none of them.

`fl` and `fs` are currency amounts charged one time for the whole holding period, and the `fa` field names the clock they fall on. Both [`calc_fees`](@ref) and [`calc_asset_fees`](@ref) return a pair, `(amortised, one_time)`.

A `nothing` `fa` puts the two fixed charges in `one_time`, and [`charge_fees`](@ref) subtracts that from the first observation alone. An [`AmortisedFees`](@ref) divides them by the observation count and adds them to `amortised`, so every observation carries an equal share and `one_time` is zero.

The field carries no count. Every site that charges a fee knows the count it charges over and hands it in, so the count is never stored and never stale. The JuMP model states the same rule, through `:one_time_fees` and [`charge_one_time_fees`](@ref).

The two clocks charge the same total over a horizon of `T` observations, which is the number [`calc_total_fees`](@ref) reports. They give a different drawdown, because one charges the whole cost on one observation and the other charges a fraction of it on each.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Fees(;
        tn::Option{<:Turnover} = nothing,
        l::Option{<:Num_VecNum} = nothing,
        s::Option{<:Num_VecNum} = nothing,
        fl::Option{<:Num_VecNum} = nothing,
        fs::Option{<:Num_VecNum} = nothing,
        lq::Option{<:Turnover} = nothing,
        flq::Option{<:Turnover} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        kwargs::NamedTuple = (; atol = 1e-8)
    ) -> Fees

Keywords correspond to the struct's fields.

## Validation

  - `l`, `s`, `fl`, `fs` are validated with [`assert_nonempty_nonneg_finite_val`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `tn`: Recursively updated via [`factory`](@ref).
  - `lq`: Recursively updated via [`factory`](@ref).
  - `flq`: Recursively updated via [`factory`](@ref).

## View parameters

This type spans **two axes**, so [`port_opt_view`](@ref) is written by hand for it rather than generated from `@vprop` tags, and no field carries one. `tn`, `l`, `s`, `fl` and `fs` price the positions the portfolio holds, so they are sliced to the selected indices. `lq` and `flq` price the positions it is forced to sell, so they are sliced to the **complement** of those indices, which the verb derives from the width of the unreduced returns matrix it is handed:

  - `tn`: Recursively viewed at the selected indices via [`port_opt_view`](@ref).
  - `l`: Sliced to the selected indices.
  - `s`: Sliced to the selected indices.
  - `fl`: Sliced to the selected indices.
  - `fs`: Sliced to the selected indices.
  - `lq`: Recursively viewed at the **complement** of the selected indices.
  - `flq`: Recursively viewed at the **complement** of the selected indices.

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
      lq ┼ nothing
     flq ┼ nothing
      fa ┼ nothing
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
  - [`AbstractFeeAmortisation`](@ref)
  - [`AmortisedFees`](@ref)
  - [`charge_fees`](@ref)
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
    @fprop tn
    """
    $(field_dict[:l_fees])
    """
    l
    """
    $(field_dict[:s_fees])
    """
    s
    """
    $(field_dict[:fl])
    """
    fl
    """
    $(field_dict[:fs])
    """
    fs
    """
    $(field_dict[:lq_fees])
    """
    @fprop lq
    """
    $(field_dict[:flq_fees])
    """
    @fprop flq
    """
    $(field_dict[:fa_fees])
    """
    fa
    """
    $(field_dict[:kwargs_fee])
    """
    kwargs
    function Fees(tn::Option{<:Turnover}, l::Option{<:Num_VecNum}, s::Option{<:Num_VecNum},
                  fl::Option{<:Num_VecNum}, fs::Option{<:Num_VecNum},
                  lq::Option{<:Turnover} = nothing, flq::Option{<:Turnover} = nothing,
                  fa::Option{<:AbstractFeeAmortisation} = nothing,
                  kwargs::NamedTuple = (; atol = 1e-8))::Fees
        assert_nonempty_nonneg_finite_val(l, :l)
        assert_nonempty_nonneg_finite_val(s, :s)
        assert_nonempty_nonneg_finite_val(fl, :fl)
        assert_nonempty_nonneg_finite_val(fs, :fs)
        return new{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs), typeof(lq),
                   typeof(flq), typeof(fa), typeof(kwargs)}(tn, l, s, fl, fs, lq, flq, fa,
                                                            kwargs)
    end
end
function Fees(; tn::Option{<:Turnover} = nothing, l::Option{<:Num_VecNum} = nothing,
              s::Option{<:Num_VecNum} = nothing, fl::Option{<:Num_VecNum} = nothing,
              fs::Option{<:Num_VecNum} = nothing, lq::Option{<:Turnover} = nothing,
              flq::Option{<:Turnover} = nothing,
              fa::Option{<:AbstractFeeAmortisation} = nothing,
              kwargs::NamedTuple = (; atol = 1e-8))::Fees
    return Fees(tn, l, s, fl, fs, lq, flq, fa, kwargs)
end
"""
    override_fee_amortisation(fees::Option{<:Fees}, fa::Nothing)
    override_fee_amortisation(fees::Nothing, fa::AbstractFeeAmortisation)
    override_fee_amortisation(fees::Fees, fa::AbstractFeeAmortisation)

Return the fee a report charges, from the fee a fit saw and the clock the report states.

A [`Fees`](@ref) answers one question with its `fa` field: the clock the two fixed charges `fl` and `fs` fall on. A cross-validation scheme asks a second question with a field of the same name, and the two answers need not agree. The optimiser prices a fixed fee the way the objective must, and the report charges it the way a fund saw it. This verb resolves the pair, and it reaches the fold's realised series alone.

A `nothing` `fa` on the scheme inherits, so the fee comes back unchanged and no object is built. A stated `fa` rebuilds the fee with that clock and leaves every other field of it alone. The rate fields `l`, `s` and `tn` charge on every observation whatever the clock is, so the override moves no number of theirs.

# Algorithm

 1. On a `nothing` `fa`, return `fees` unchanged, whether it is a [`Fees`](@ref) or `nothing`.
 2. On a stated `fa` and a `nothing` `fees`, return `nothing`. There is no fee to charge, so there is no clock to state.
 3. On a stated `fa` and a [`Fees`](@ref), rebuild the fee with that `fa` and its own five fee fields and `kwargs`.

# Arguments

  - `fees`: The fee the fit saw, or `nothing`.
  - `fa`: The clock the scheme states, or `nothing` to inherit the fee's own.

# Returns

  - `Option{<:Fees}`: The fee the report charges.

# Related

  - [`Fees`](@ref)
  - [`AbstractFeeAmortisation`](@ref)
  - [`AmortisedFees`](@ref)
  - [`FirstObservationFees`](@ref)
  - [`fold_evaluation`](@ref)
  - [`charge_fees`](@ref)
"""
function override_fee_amortisation(fees::Option{<:Fees}, ::Nothing)
    return fees
end
function override_fee_amortisation(::Nothing, ::AbstractFeeAmortisation)
    return nothing
end
function override_fee_amortisation(fees::Fees, fa::AbstractFeeAmortisation)
    return Fees(; tn = fees.tn, l = fees.l, s = fees.s, fl = fees.fl, fs = fees.fs,
                lq = fees.lq, flq = fees.flq, fa = fa, kwargs = fees.kwargs)
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
    return needs_previous_weights(fe.tn) ||
           needs_previous_weights(fe.lq) ||
           needs_previous_weights(fe.flq)
end
"""
    strip_liquidation_carriers(fees, imsk)

Drop the two liquidation carriers when no asset left the universe.

A caller states `lq` and `flq` over the **full** universe, because they cannot know in advance which asset will delist. [`port_opt_view`](@ref) narrows them to the complement of the Investable Mask at the door — but a window in which every asset is investable derives **no mask at all**, and the `nothing` sentinel short-circuits the door precisely so that the all-investable path allocates nothing. The carriers would then survive at full width and be charged in full, for assets that never left.

This verb closes that gap explicitly rather than by taking the view on the common path, so the `nothing` mask stays allocation-free. A `BitVector` mask means the door has already run and the carriers are on the right axis, so the fee is returned untouched. A `nothing` mask means nothing exited, so both carriers go. A fee that carries neither is returned untouched under either mask, so a caller who states no liquidation pays for nothing.

# Algorithm

 1. On a `nothing` `fees`, or a `BitVector` `imsk`, return `fees` unchanged. A stated `nothing` fee under a derived mask satisfies both, so a third method names that pair and breaks the ambiguity.
 2. On a `nothing` `imsk` with both carriers already `nothing`, return `fees` unchanged, so the common case allocates nothing.
 3. Otherwise rebuild the fee with `lq` and `flq` set to `nothing`, carrying every other field through.

# Arguments

  - `fees`: The fee to strip, or `nothing`.
  - `imsk`: The Investable Mask, or `nothing` when every asset is investable.

# Returns

  - `fees::typeof(fees)`: The fee, with no liquidation carrier when none can apply.

# Related

  - [`Fees`](@ref)
  - [`port_opt_view`](@ref)
  - [`investable_mask`](@ref)
  - [`fees_constraints`](@ref)
"""
function strip_liquidation_carriers(fees, ::BitVector)
    return fees
end
function strip_liquidation_carriers(::Nothing, ::Any)
    return nothing
end
function strip_liquidation_carriers(::Nothing, ::BitVector)
    return nothing
end
function strip_liquidation_carriers(fees::Fees, ::Nothing)
    return if isnothing(fees.lq) && isnothing(fees.flq)
        fees
    else
        Fees(; tn = fees.tn, l = fees.l, s = fees.s, fl = fees.fl, fs = fees.fs,
             lq = nothing, flq = nothing, fa = fees.fa, kwargs = fees.kwargs)
    end
end
"""
    investable_fees_view(fees, imsk, X)

Place a fee resolved over the caller's universe onto the axes an Investable Mask leaves.

A fee is resolved **before** the door, against the `sets` the caller stated, and this verb takes it through. That order is what lets a name-keyed fee name an asset the data later delists: after the door `sets` holds the investable names alone, so the name is gone and `strict` refuses it, and a carrier keyed by name cannot resolve at all, because its `w` already sits on the complement while `sets` sits on the mask. Resolving first and viewing after removes both, and needs no new arithmetic: [`port_opt_view`](@ref) already splits a resolved [`Fees`](@ref) across its two axes.

The two methods are the two things a mask can be. A `BitVector` means assets left, so the view runs at `findall(imsk)` and derives the complement from the width of the **unreduced** `X`. A `nothing` mask means every asset is investable, so there is no complement to slice to and [`strip_liquidation_carriers`](@ref) drops both carriers instead, which is what keeps that path allocation-free.

# Algorithm

 1. On a `nothing` mask, hand the fee to [`strip_liquidation_carriers`](@ref), which drops both carriers and returns a fee carrying neither untouched.
 2. On a `BitVector` mask, take [`port_opt_view`](@ref) at `findall(imsk)` with the unreduced `X`, which slices the five per-asset fields to the mask and the two carriers to its complement.

# Arguments

  - `fees`: The fee resolved over the caller's full universe, or `nothing`.
  - `imsk`: The Investable Mask, or `nothing` when every asset is investable.
  - `X`: The **unreduced** returns matrix. Only its width is read, to derive the complement.

# Returns

  - `fees::typeof(fees)`: The fee on the axes the mask leaves.

# Related

  - [`Fees`](@ref)
  - [`port_opt_view`](@ref)
  - [`strip_liquidation_carriers`](@ref)
  - [`investable_mask`](@ref)
  - [`fees_constraints`](@ref)
"""
function investable_fees_view(fees, ::Nothing, ::Any)
    return strip_liquidation_carriers(fees, nothing)
end
function investable_fees_view(fees, imsk::BitVector, X::MatNum)
    return port_opt_view(fees, findall(imsk), X)
end
"""
    port_opt_view(fees::Fees, i, X::MatNum, args...)
    port_opt_view(fees::FeesEstimator, i, X::MatNum, args...)
    port_opt_view(fees::FeesE_Fees, i, args...)

Sub-select a fee to the assets an optimisation keeps, on **both** of its axes.

A [`Fees`](@ref) spans two axes once an optimisation has reduced to its Investable Mask. `tn`, `l`, `s`, `fl` and `fs` price the positions the portfolio **holds**, so they live on the investable axis and are sliced at `i`. `lq` and `flq` price the positions it is **forced to sell**, so they live on the **complement** of that axis and are sliced at the assets `i` leaves out.

That is why neither carrier is tagged `@vprop` and why this verb is written by hand rather than generated: the generic machinery threads one index through every tagged field, and here the two groups need different ones. The complement is derived and never stored, from `i` and the width of `X`. `X` is the **unreduced** returns matrix, which is what [`investable_reduction`](@ref) and [`investable_view`](@ref) already pass, so `size(X, 2)` is the full universe.

The three-argument method serves a caller that hands no matrix. It cannot derive a complement, so it slices the five per-asset fields and passes the two carriers through untouched, which is correct because they already sit on their own axis.

# Algorithm

 1. Build `j`, the complement of `i` in `1:size(X, 2)`.
 2. Slice `tn` at `i` through [`port_opt_view`](@ref), whose `@vprop` tags take `w` and a vector `val` and leave a scalar or dictionary `val` alone.
 3. Slice `l`, `s`, `fl` and `fs` at `i` through [`nothing_scalar_array_view`](@ref).
 4. On an empty `j` no asset left the universe, so set `lq` and `flq` to `nothing`: there is nothing to liquidate, and a [`Turnover`](@ref) refuses an empty `w` in any case. Otherwise slice both at `j`, by the same verb as step 2.
 5. Rebuild through the keyword constructor, carrying `fa` and `kwargs` unchanged.

# Arguments

  - `fees`: The fee to reduce.
  - `i`: Indices of the assets the optimisation keeps.
  - `X`: The **unreduced** returns matrix, read for its width alone.

# Returns

  - `fees::typeof(fees)`: The fee on the two reduced axes.

# Related

  - [`Fees`](@ref)
  - [`FeesEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
  - [`investable_reduction`](@ref)
"""
function port_opt_view(fees::Fees, i, X::MatNum, args...)::Fees
    j = setdiff(1:size(X, 2), i)
    # An empty complement means no asset left the universe, so there is nothing to
    # liquidate and the carriers go. A `Turnover` refuses an empty `w`, so this is the
    # answer the type asks for as well as the one the rule asks for.
    ex = !isempty(j)
    return Fees(; tn = port_opt_view(fees.tn, i, X, args...),
                l = nothing_scalar_array_view(fees.l, i),
                s = nothing_scalar_array_view(fees.s, i),
                fl = nothing_scalar_array_view(fees.fl, i),
                fs = nothing_scalar_array_view(fees.fs, i),
                lq = ex ? port_opt_view(fees.lq, j, X, args...) : nothing,
                flq = ex ? port_opt_view(fees.flq, j, X, args...) : nothing, fa = fees.fa,
                kwargs = fees.kwargs)
end
function port_opt_view(fees::FeesEstimator, i, X::MatNum, args...)::FeesEstimator
    j = setdiff(1:size(X, 2), i)
    ex = !isempty(j)
    return FeesEstimator(; tn = port_opt_view(fees.tn, i, X, args...),
                         l = nothing_scalar_array_view(fees.l, i),
                         s = nothing_scalar_array_view(fees.s, i),
                         fl = nothing_scalar_array_view(fees.fl, i),
                         fs = nothing_scalar_array_view(fees.fs, i),
                         lq = ex ? port_opt_view(fees.lq, j, X, args...) : nothing,
                         flq = ex ? port_opt_view(fees.flq, j, X, args...) : nothing,
                         dl = fees.dl, ds = fees.ds, dfl = fees.dfl, dfs = fees.dfs,
                         fa = fees.fa, kwargs = fees.kwargs)
end
function port_opt_view(fees::Fees, i, args...)::Fees
    return Fees(; tn = port_opt_view(fees.tn, i, args...),
                l = nothing_scalar_array_view(fees.l, i),
                s = nothing_scalar_array_view(fees.s, i),
                fl = nothing_scalar_array_view(fees.fl, i),
                fs = nothing_scalar_array_view(fees.fs, i), lq = fees.lq, flq = fees.flq,
                fa = fees.fa, kwargs = fees.kwargs)
end
function port_opt_view(fees::FeesEstimator, i, args...)::FeesEstimator
    return FeesEstimator(; tn = port_opt_view(fees.tn, i, args...),
                         l = nothing_scalar_array_view(fees.l, i),
                         s = nothing_scalar_array_view(fees.s, i),
                         fl = nothing_scalar_array_view(fees.fl, i),
                         fs = nothing_scalar_array_view(fees.fs, i), lq = fees.lq,
                         flq = fees.flq, dl = fees.dl, ds = fees.ds, dfl = fees.dfl,
                         dfs = fees.dfs, fa = fees.fa, kwargs = fees.kwargs)
end
"""
    lift_fees(fees::Nothing, imsk)
    lift_fees(fees::Fees, imsk::Nothing)
    lift_fees(fees::Fees, imsk::BitVector)

Put a fee an optimisation reduced back onto the full universe, on **both** of its axes.

The inverse of [`port_opt_view`](@ref). ADR 0115 reduces an optimisation to its Investable Mask and expands the solved weights back to the caller's universe, so a result pairs a **full-length** `w` with a fee that spans two **reduced** axes: `tn`, `l`, `s`, `fl` and `fs` on the investable assets, `lq` and `flq` on the complement. A consumer that indexes the fee by the full-length weights meets a four-element field and a five-element selector, which is the defect of #914.

This verb closes the gap by moving the fee onto the axis the weights already live on. Every per-asset field comes back at `length(imsk)`, zero-filled where it says nothing: the five holding fields carry a zero at each asset that left, and the two carriers carry a zero at each asset that stayed. A zero rate charges nothing and a zero reference weight trades nothing, so the lift moves no number the reduced fee already charged. A scalar rate applies to every asset whatever the axis is, so it is carried through untouched.

# Algorithm

 1. On a `nothing` `fees`, or a `nothing` `imsk`, return `fees` unchanged. A `nothing` mask means the optimisation reduced on nothing, so the fee is on the full universe already.
 2. Lift `tn` at `imsk` with [`lift_turnover`](@ref), and `l`, `s`, `fl` and `fs` with [`lift_fee_rate`](@ref).
 3. Lift `lq` and `flq` at `.!imsk`, the complement the two carriers live on, by the same verb as step 2.
 4. Rebuild through the keyword constructor, carrying `fa` and `kwargs` unchanged.

# Arguments

  - `fees`: The fee to lift, or `nothing`.
  - `imsk`: The Investable Mask the fee was reduced on, or `nothing` when it was reduced on nothing.

# Returns

  - `fees::typeof(fees)`: The fee on the full universe.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.001, 0.002], lq = Turnover(; w = [0.25], val = [0.01]));

julia> PortfolioOptimisers.lift_fees(fees, BitVector([true, false, true])).l
3-element Vector{Float64}:
 0.001
 0.0
 0.002

julia> PortfolioOptimisers.lift_fees(fees, BitVector([true, false, true])).lq.w
3-element Vector{Float64}:
 0.0
 0.25
 0.0
```

# Related

  - [`Fees`](@ref)
  - [`lift_fee_rate`](@ref)
  - [`lift_turnover`](@ref)
  - [`port_opt_view`](@ref)
  - [`result_investable_mask`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function lift_fees(::Nothing, ::Any)
    return nothing
end
function lift_fees(fees::Fees, ::Nothing)
    return fees
end
function lift_fees(fees::Fees, imsk::BitVector)::Fees
    # The complement is where the two carriers live, so it is the mask they lift at.
    cmsk = .!imsk
    return Fees(; tn = lift_turnover(fees.tn, imsk), l = lift_fee_rate(fees.l, imsk),
                s = lift_fee_rate(fees.s, imsk), fl = lift_fee_rate(fees.fl, imsk),
                fs = lift_fee_rate(fees.fs, imsk), lq = lift_turnover(fees.lq, cmsk),
                flq = lift_turnover(fees.flq, cmsk), fa = fees.fa, kwargs = fees.kwargs)
end
"""
    lift_fee_rate(x::Nothing, ::BitVector)
    lift_fee_rate(x::Number, ::BitVector)
    lift_fee_rate(x::VecNum, imsk::BitVector)

Put one per-asset fee rate back onto the full universe, zero where the mask is `false`.

The per-field step of [`lift_fees`](@ref). A `nothing` field states no fee and a scalar field states one rate for every asset, so neither carries an axis and both are returned untouched. A vector field carries one entry per asset of the reduced axis, so it is expanded with [`expand_investable_weights`](@ref).

# Arguments

  - `x`: The rate to lift: `nothing`, a scalar, or one entry per asset of the reduced axis.
  - `imsk`: The mask the rate was reduced on.

# Returns

  - `x`: The rate on the full universe.

# Related

  - [`lift_fees`](@ref)
  - [`lift_turnover`](@ref)
  - [`expand_investable_weights`](@ref)
"""
function lift_fee_rate(::Nothing, ::BitVector)
    return nothing
end
function lift_fee_rate(x::Number, ::BitVector)
    return x
end
function lift_fee_rate(x::VecNum, imsk::BitVector)
    return expand_investable_weights(imsk, x)
end
"""
    lift_turnover(tn::Nothing, ::BitVector)
    lift_turnover(tn::Turnover, imsk::BitVector)

Put a turnover carrier back onto the full universe, zero where the mask is `false`.

The nested step of [`lift_fees`](@ref). A [`Turnover`](@ref) holds a reference weight per asset and a rate that is either a scalar or one entry per asset, so `w` always expands and `val` expands through [`lift_fee_rate`](@ref). A zero reference weight trades nothing, so an asset the mask leaves out is charged nothing.

# Arguments

  - `tn`: The carrier to lift, or `nothing`.
  - `imsk`: The mask the carrier was reduced on.

# Returns

  - `tn`: The carrier on the full universe, or `nothing`.

# Related

  - [`lift_fees`](@ref)
  - [`lift_fee_rate`](@ref)
  - [`Turnover`](@ref)
"""
function lift_turnover(::Nothing, ::BitVector)
    return nothing
end
function lift_turnover(tn::Turnover, imsk::BitVector)::Turnover
    return Turnover(; w = expand_investable_weights(imsk, tn.w),
                    val = lift_fee_rate(tn.val, imsk), fixed = tn.fixed)
end
"""
    fees_constraints(fees::FeesEstimator, sets::UniverseSets; datatype::DataType = Float64,
                     strict::Bool = false)

Resolve the name-keyed fee fields of a [`FeesEstimator`](@ref) against a universe, giving a [`Fees`](@ref) of plain per-asset vectors.

Ten fields carry the specification and each of the four proportional and fixed fields draws its gaps from its **own** default: `l` from `dl`, `s` from `ds`, `fl` from `dfl` and `fs` from `dfs`. A default never fills a neighbour's field. The nested `tn` resolves through [`turnover_constraints`](@ref), so a [`FeesEstimator`](@ref) holding a [`TurnoverEstimator`](@ref) returns a [`Fees`](@ref) holding a [`Turnover`](@ref). `fees.fa` carries no universe-keyed specification, so it reaches the result unchanged.

# Algorithm

 1. Resolve `fees.tn` against the universe of `sets` with [`turnover_constraints`](@ref), giving a [`Turnover`](@ref) whose `val` is one turnover fee rate per asset. A `nothing` `tn` stays `nothing`.
 2. Resolve `fees.l` with [`estimator_to_val`](@ref), giving `l`, one long proportional rate per asset in the order of the universe. Every asset the keys miss takes `fees.dl`, or `zero(datatype)` when `fees.dl` is `nothing`. A `nothing` `fees.l` stays `nothing`.
 3. Resolve `fees.s` the same way against `fees.ds`, giving `s`.
 4. Resolve `fees.fl` the same way against `fees.dfl`, giving `fl`.
 5. Resolve `fees.fs` the same way against `fees.dfs`, giving `fs`.
 6. Build a [`Fees`](@ref) from the five resolved fields, `fees.fa` unchanged, and `fees.kwargs`, which reaches the result unchanged and sets the boundary the fixed terms read.

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
      lq ┼ nothing
     flq ┼ nothing
      fa ┼ nothing
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
      lq ┼ nothing
     flq ┼ nothing
      fa ┼ nothing
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
                                      strict = strict),
                lq = turnover_constraints(fees.lq, sets; datatype = datatype,
                                          strict = strict),
                flq = turnover_constraints(fees.flq, sets; datatype = datatype,
                                           strict = strict), fa = fees.fa,
                kwargs = fees.kwargs)
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
      lq ┼ nothing
     flq ┼ nothing
      fa ┼ nothing
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
    calc_fees(w::VecNum, ::Nothing, ::Function)
    calc_fees(w::VecNum, fees::Number, op::Function)
    calc_fees(w::VecNum, fees::VecNum, op::Function)

Compute the proportional fees for portfolio weights.

This is one term of the total fee, not the whole fee. [`calc_fees(w::VecNum, fees::Fees)`](@ref) calls it twice, under `.>=` for the long side and under `.<` for the short side, and negates the short call. [`Fees`](@ref) states the closed form as ``F_{\\text{p}}``.

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

This is one term of the total fee, not the whole fee. [`Fees`](@ref) states the closed form as ``F_{\\text{Tn}}``, and reads `tn.val` as a per-asset fee rate rather than as a bound. The `fixed` flag of [`Turnover`](@ref) reaches no method here: it decides which reference weights `tn.w` holds, through [`factory`](@ref), and by the time this method runs `tn.w` is already the vector the fee must be charged against.

# Algorithm

 1. On a `nothing` `tn`, return `zero(eltype(w))`. The method reads `w` only for its element type.
 2. Otherwise form the traded amount per asset, the absolute difference between `w` and the reference weights `tn.w`.
 3. On a `Number` `tn.val`, sum the traded amount and scale it by the one rate.
 4. On a `VecNum` `tn.val`, contract the rates with the traded amount.

Steps 3 and 4 are not the same expression. They agree to rounding when `tn.val` is a constant vector, and they differed by `3.47e-18` on `w = [0.6, -0.4, 0.0, 0.25]`, `tn.w = [0.1, 0.2, 0.3, 0.4]` and a rate of `0.02`.

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

A fixed fee is charged per position held, whatever its size. [`Fees`](@ref) states the closed form as ``F_{\\text{f}}``.

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
    calc_fees(w::VecNum, T::Number, fees::Fees)

Compute total fees for portfolio weights.

Sums proportional, fixed, and turnover fees for all assets. [`calc_asset_fees(w::VecNum, fees::Fees)`](@ref) splits the same total over the assets, and its sum is this number up to the order of summation.

The verb returns a pair, `(amortised, one_time)`. `l`, `s` and `tn` are rates per period, so they charge on every observation and land in `amortised`. `fl` and `fs` are currency amounts charged one time for the whole holding period, so `fees.fa` decides where they land: a `nothing` `fa` puts them in `one_time`, and an [`AmortisedFees`](@ref) divides them by `T`, adds them to `amortised` and leaves `one_time` zero.

`T` is the observation count the calling site charges over, and the site always knows it, so no fee stores one. [`charge_fees`](@ref) hands the length of the series it lays the pair onto, and [`calc_total_fees`](@ref) contracts the pair to the cost of a whole holding period.

# Algorithm

 1. Charge the per period terms, the call of [`calc_periodic_fees`](@ref).
 2. Charge the one-off terms, the call of [`calc_one_off_fees`](@ref).
 3. On a `nothing` `fees.fa`, return the two charges unchanged.
 4. On an [`AmortisedFees`](@ref) `fees.fa`, divide the one-off charge by `T`, add it to the per period charge, and return that number beside a zero of the same type.

# Arguments

  - `w`: Portfolio weights.
  - `T`: Observation count the fee is charged over.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `amortised::Number`: The charge every observation carries.
  - `one_time::Number`: The charge the first observation carries alone.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.01, 0.02], s = [0.01, 0.02], fl = [5.0, 0.0], fs = [0.0, 10.0]);

julia> calc_fees([0.1, -0.2], 21, fees)
(0.005, 15.0)
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_fees(w::VecNum, T::Number, fees::Fees)
    return calc_fees(w, T, fees, fees.fa)
end
function calc_fees(w::VecNum, ::Number, fees::Fees,
                   ::Union{Nothing, <:FirstObservationFees})
    return (calc_periodic_fees(w, fees), calc_one_off_fees(w, fees))
end
function calc_fees(w::VecNum, T::Number, fees::Fees, ::AmortisedFees)
    val = calc_periodic_fees(w, fees) + calc_one_off_fees(w, fees) / T
    return (val, zero(val))
end
"""
    calc_asset_fees(w::VecNum, ::Nothing, ::Function)
    calc_asset_fees(w::VecNum, fees::Number, op::Function)
    calc_asset_fees(w::VecNum, fees::VecNum, op::Function)

Compute the proportional per asset fees for portfolio weights.

This is one term of the total fee, not the whole fee. It is the elementwise form of [`calc_fees(w::VecNum, fees::Number, op::Function)`](@ref), and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{p}}``.

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

This is one term of the total fee, not the whole fee. It is the elementwise form of [`calc_fees(w::VecNum, tn::Turnover)`](@ref), and [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{Tn}}``. The `fixed` flag of [`Turnover`](@ref) reaches no method here, for the reason that name gives.

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

This is the elementwise form of [`calc_fixed_fees`](@ref), and its entries sum to the number that name returns. [`Fees`](@ref) states the closed form as ``\\boldsymbol{F}_{\\text{f}}``.

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
    calc_asset_fees(w::VecNum, T::Number, fees::Fees)

Compute total per asset fees for portfolio weights.

Sums proportional, fixed, and turnover fees for all assets. The entries sum to the number [`calc_fees(w::VecNum, fees::Fees)`](@ref) returns, up to the order of summation.

The verb returns a pair, `(amortised, one_time)`, and **each half is itself a pair**, one entry per axis of a reduced [`Fees`](@ref): the charge of the assets that stayed, and the charge of the assets that left. `l`, `s` and `tn` are rates per period and land in `amortised`, as `lq` does on the other axis. `fl` and `fs` are currency amounts charged one time for the whole holding period, so `fees.fa` decides where they land, as it does for `flq`: a `nothing` `fa` puts them in `one_time`, and an [`AmortisedFees`](@ref) divides them by `T`, adds them to `amortised` and leaves `one_time` zero. The doctest below shows all four vectors.

`T` is the observation count the calling site charges over, and the site always knows it, so no fee stores one. [`charge_asset_fees`](@ref) hands the row count of the matrix it lays the pair onto, and [`calc_total_asset_fees`](@ref) contracts the pair to the cost of a whole holding period.

# Algorithm

 1. Charge the per period terms, the call of [`calc_asset_periodic_fees`](@ref), which answers both axes.
 2. Charge the one-off terms, the call of [`calc_asset_one_off_fees`](@ref), which answers both axes.
 3. On a `nothing` `fees.fa`, return the two charges unchanged.
 4. On an [`AmortisedFees`](@ref) `fees.fa`, divide each one-off charge by `T`, add it to the per period charge of its own axis, and return those beside zeros of the same type. The liquidation axis adds through [`add_liquidation_terms`](@ref), because `lq` and `flq` are set independently and an unset one is an empty vector.

# Arguments

  - `w`: Portfolio weights.
  - `T`: Observation count the fee is charged over.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `amortised::Tuple{<:VecNum, <:VecNum}`: The charge every observation carries, per asset that stayed and per asset that left.
  - `one_time::Tuple{<:VecNum, <:VecNum}`: The charge the first observation carries alone, on the same two axes. Zero under an [`AmortisedFees`](@ref), which spread it into `amortised`.

# Examples

```jldoctest
julia> fees = Fees(; l = [0.01, 0.02], s = [0.01, 0.02], fl = [5.0, 0.0], fs = [0.0, 10.0]);

julia> calc_asset_fees([0.1, -0.2], 21, fees)
(([0.001, 0.004], Float64[]), ([5.0, 10.0], Float64[]))
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_asset_fees(w::VecNum, T::Number, fees::Fees)
    return calc_asset_fees(w, T, fees, fees.fa)
end
function calc_asset_fees(w::VecNum, ::Number, fees::Fees,
                         ::Union{Nothing, <:FirstObservationFees})
    return (calc_asset_periodic_fees(w, fees), calc_asset_one_off_fees(w, fees))
end
function calc_asset_fees(w::VecNum, T::Number, fees::Fees, ::AmortisedFees)
    pi, pl = calc_asset_periodic_fees(w, fees)
    oi, ol = calc_asset_one_off_fees(w, fees)
    vi = pi + oi / T
    vl = add_liquidation_terms(pl, ol / T)
    return ((vi, vl), (zero(vi), zero(vl)))
end
"""
    add_liquidation_terms(a::VecNum, b::VecNum)

Add the two terms of the liquidation axis, either of which may be unset.

`lq` and `flq` are set independently, and the verb that prices an unset carrier returns an empty vector rather than a vector of zeros, because it holds no length to build one from: the axis is the complement of the Investable Mask, and a `Fees` does not carry the mask. So a `Fees` that sets `flq` and no `lq` gives one term spanning the complement and one spanning nothing, and adding them elementwise would raise a `DimensionMismatch` on a fee the caller set correctly.

An empty term is a term that charges nothing, so the sum is the other term. Two set terms span the same complement and add elementwise. The investable axis needs no such verb: its terms are built from `w`, so they always span it.

# Arguments

  - `a`: One term of the axis.
  - `b`: The other term of the axis.

# Returns

  - `val::VecNum`: The two terms summed, or whichever of them is set.

# Related

  - [`calc_asset_fees`](@ref)
  - [`calc_asset_liquidation_fees`](@ref)
  - [`calc_asset_fixed_liquidation_fees`](@ref)
  - [`AmortisedFees`](@ref)
"""
function add_liquidation_terms(a::VecNum, b::VecNum)
    return if isempty(a)
        b
    elseif isempty(b)
        a
    else
        a + b
    end
end

"""
    calc_liquidation_fees(::Nothing)
    calc_liquidation_fees(lq::Turnover{<:Any, <:Number})
    calc_liquidation_fees(lq::Turnover{<:Any, <:VecNum})

Charge the proportional cost of the positions a forced exit sells.

`lq` lives on the **complement** of the Investable Mask, so its entries are the assets that left, never the assets the programme holds. A forced exit is a trade to zero, so the target weight of every entry is zero, the turnover `|target - lq.w|` is `abs.(lq.w)`, and the charge is the rate times the absolute previous weight. The carrier is the whole of the input: the verb needs no weight vector and takes none.

The charge is a rate, so it falls on every period beside `l`, `s` and `tn`, and [`calc_periodic_fees`](@ref) adds it there.

# Algorithm

 1. On a `nothing` `lq`, return `false`. Nothing left the universe, so nothing is owed, and a `Bool` zero adds to a charge of any element type without widening it.
 2. On a scalar rate, return `lq.val * sum(abs, lq.w)`.
 3. On a per asset rate, return `dot(lq.val, abs.(lq.w))`.

# Arguments

  - `lq`: The proportional liquidation carrier, or `nothing`.

# Returns

  - `val::Number`: The proportional charge of the forced exit.

# Examples

```jldoctest
julia> PortfolioOptimisers.calc_liquidation_fees(nothing)
false

julia> PortfolioOptimisers.calc_liquidation_fees(Turnover(; w = [0.25], val = [0.01]))
0.0025
```

# Related

  - [`Fees`](@ref)
  - [`Turnover`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_periodic_fees`](@ref)
  - [`calc_fixed_liquidation_fees`](@ref)
"""
function calc_liquidation_fees(::Nothing)
    return false
end
function calc_liquidation_fees(lq::Turnover{<:Any, <:Number})
    return lq.val * sum(abs, lq.w)
end
function calc_liquidation_fees(lq::Turnover{<:Any, <:VecNum})
    return LinearAlgebra.dot(lq.val, abs.(lq.w))
end
"""
    calc_fixed_liquidation_fees(::Nothing, ::NamedTuple)
    calc_fixed_liquidation_fees(flq::Turnover, kwargs::NamedTuple)

Charge the fixed cost of the positions a forced exit sells.

The fixed twin of [`calc_liquidation_fees`](@ref), and it takes no weight vector for the same reason. `flq` lives on the complement of the Investable Mask, and its amount is charged once for each entry whose absolute previous weight is not `isapprox` to zero under `kwargs`, the threshold `fl` and `fs` already use.

A liquidated short is a trade as much as a liquidated long, so **both sides are charged**. The verb therefore calls [`calc_fixed_fees`](@ref) twice against `flq.w`, once under `.>=` and once under `.<`, which is the pattern [`calc_one_off_fees`](@ref) spells for `fl` and `fs` with one rate serving both sides. The two selections are disjoint, so no entry is charged twice.

The charge is a currency amount, so it falls one time for the whole holding period beside `fl` and `fs`, on the clock `fees.fa` names.

# Algorithm

 1. On a `nothing` `flq`, return `false`, the same `Bool` zero [`calc_liquidation_fees`](@ref) returns.
 2. Otherwise charge [`calc_fixed_fees`](@ref) on `flq.w` and `flq.val` under `.>=`, the liquidated long positions.
 3. Charge the same under `.<`, the liquidated short positions.
 4. Return the sum of the two.

# Arguments

  - `flq`: The fixed liquidation carrier, or `nothing`.
  - `kwargs`: Forwarded to `isapprox` to decide how near zero counts as zero.

# Returns

  - `val::Number`: The fixed charge of the forced exit.

# Examples

```jldoctest
julia> PortfolioOptimisers.calc_fixed_liquidation_fees(nothing, (; atol = 1e-8))
false

julia> PortfolioOptimisers.calc_fixed_liquidation_fees(Turnover(; w = [0.25, -0.4],
                                                                val = [5.0, 7.0]), (; atol = 1e-8))
12.0
```

# Related

  - [`Fees`](@ref)
  - [`Turnover`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_one_off_fees`](@ref)
  - [`calc_liquidation_fees`](@ref)
"""
function calc_fixed_liquidation_fees(::Nothing, ::NamedTuple)
    return false
end
function calc_fixed_liquidation_fees(flq::Turnover, kwargs::NamedTuple)
    return calc_fixed_fees(flq.w, flq.val, kwargs, .>=) +
           calc_fixed_fees(flq.w, flq.val, kwargs, .<)
end
"""
    calc_asset_liquidation_fees(w::VecNum, ::Nothing)
    calc_asset_liquidation_fees(w::VecNum, lq::Turnover)

Split the proportional cost of a forced exit over the assets that left.

The per asset twin of [`calc_liquidation_fees`](@ref). Its entries sum to that number, and they sit on the **complement** of the Investable Mask: one entry per asset that left, in the order the carrier holds them, so each charge names the asset that caused it.

`nothing` gives an **empty** vector rather than a vector of zeros, because the complement is empty when nothing exited. That keeps the two axes honest: a reader can tell "no asset left" from "an asset left and owed nothing".

# Algorithm

 1. On a `nothing` `lq`, return an empty vector in the element type of `w`.
 2. Otherwise call [`calc_asset_fees`](@ref) on `lq` against a zero vector the length of `lq.w`, giving `lq.val .* abs.(lq.w)` elementwise.

# Arguments

  - `w`: Portfolio weights, read for their element type.
  - `lq`: The proportional liquidation carrier, or `nothing`.

# Returns

  - `val::VecNum`: The proportional charge per liquidated asset.

# Examples

```jldoctest
julia> PortfolioOptimisers.calc_asset_liquidation_fees([0.5, 0.5],
                                                       Turnover(; w = [-0.4, 0.25],
                                                                val = [0.002, 0.010]))
2-element Vector{Float64}:
 0.0008
 0.0025
```

# Related

  - [`Fees`](@ref)
  - [`calc_liquidation_fees`](@ref)
  - [`calc_asset_periodic_fees`](@ref)
  - [`calc_asset_fixed_liquidation_fees`](@ref)
"""
function calc_asset_liquidation_fees(w::VecNum, ::Nothing)
    return zeros(eltype(w), 0)
end
function calc_asset_liquidation_fees(w::VecNum, lq::Turnover)
    return calc_asset_fees(zeros(eltype(w), length(lq.w)), lq)
end
"""
    calc_asset_fixed_liquidation_fees(w::VecNum, ::Nothing, ::NamedTuple)
    calc_asset_fixed_liquidation_fees(w::VecNum, flq::Turnover, kwargs::NamedTuple)

Split the fixed cost of a forced exit over the assets that left.

The per asset twin of [`calc_fixed_liquidation_fees`](@ref), on the complement axis, and empty when nothing exited. Both liquidated sides are charged, so the two selections of [`calc_asset_fixed_fees`](@ref) are summed; they are disjoint, so no entry is charged twice.

# Algorithm

 1. On a `nothing` `flq`, return an empty vector in the element type of `w`.
 2. Otherwise charge [`calc_asset_fixed_fees`](@ref) on `flq.w` and `flq.val` under `.>=` and again under `.<`, and return the elementwise sum.

# Arguments

  - `w`: Portfolio weights, read for their element type.
  - `flq`: The fixed liquidation carrier, or `nothing`.
  - `kwargs`: Forwarded to `isapprox` to decide how near zero counts as zero.

# Returns

  - `val::VecNum`: The fixed charge per liquidated asset.

# Examples

```jldoctest
julia> PortfolioOptimisers.calc_asset_fixed_liquidation_fees([0.5, 0.5],
                                                             Turnover(; w = [-0.4, 0.25],
                                                                      val = [7.0, 5.0]),
                                                             (; atol = 1e-8))
2-element Vector{Float64}:
 7.0
 5.0
```

# Related

  - [`Fees`](@ref)
  - [`calc_fixed_liquidation_fees`](@ref)
  - [`calc_asset_one_off_fees`](@ref)
  - [`calc_asset_liquidation_fees`](@ref)
"""
function calc_asset_fixed_liquidation_fees(w::VecNum, ::Nothing, ::NamedTuple)
    return zeros(eltype(w), 0)
end
function calc_asset_fixed_liquidation_fees(::VecNum, flq::Turnover, kwargs::NamedTuple)
    return calc_asset_fixed_fees(flq.w, flq.val, kwargs, .>=) +
           calc_asset_fixed_fees(flq.w, flq.val, kwargs, .<)
end
"""
    calc_periodic_fees(w::VecNum, fees::Fees)

Charge the terms of a fee that fall on every observation.

`l`, `s` and `tn` are rates per period, so each of them charges one time per observation of a return series. `fees.fa` reaches none of the three. [`calc_fees`](@ref) adds this number to the one-off terms of [`calc_one_off_fees`](@ref), and [`calc_total_fees`](@ref) multiplies it by the horizon.

# Algorithm

 1. Charge the long proportional term, the call of [`calc_fees`](@ref) on `fees.l` under `.>=`.
 2. Charge the short proportional term, the negated call of the same name on `fees.s` under `.<`. `w` is negative on that side, so the negation is what makes the term a positive charge.
 3. Charge the turnover term, the call of [`calc_fees`](@ref) on `fees.tn`.
 4. Return the sum of the three terms.

# Arguments

  - `w`: Portfolio weights.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::Number`: The per period charge, before the one-off terms.

# Examples

```jldoctest
julia> fees = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002));

julia> PortfolioOptimisers.calc_periodic_fees([0.5, 0.5], fees)
0.012
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_one_off_fees`](@ref)
  - [`calc_total_fees`](@ref)
  - [`calc_asset_periodic_fees`](@ref)
"""
function calc_periodic_fees(w::VecNum, fees::Fees)
    return calc_fees(w, fees.l, .>=) - calc_fees(w, fees.s, .<) +
           calc_fees(w, fees.tn) +
           calc_liquidation_fees(fees.lq)
end
"""
    calc_asset_periodic_fees(w::VecNum, fees::Fees)

Split over the assets the terms of a fee that fall on every observation.

The per asset twin of [`calc_periodic_fees`](@ref). Its entries sum to that number, up to the order of summation.

# Algorithm

 1. Charge the long proportional term, the call of [`calc_asset_fees`](@ref) on `fees.l` under `.>=`.
 2. Charge the short proportional term, the negated call of the same name on `fees.s` under `.<`.
 3. Charge the turnover term, the call of [`calc_asset_fees`](@ref) on `fees.tn`.
 4. Charge the proportional forced exit, the call of [`calc_asset_liquidation_fees`](@ref) on `fees.lq`. `lq` is a rate, so it falls on every period beside `l`, `s` and `tn`, and no clock reaches it.
 5. Return the pair: the elementwise sum of the three vectors of steps 1 to 3, and the vector of step 4.

# Arguments

  - `w`: Portfolio weights.
  - `fees`: [`Fees`](@ref) structure.

# Returns

The verb returns a pair, one entry per axis of a reduced [`Fees`](@ref).

  - `investable::VecNum`: The per period charge of each asset that stayed, from `l`, `s` and `tn`, which were sliced to the Investable Mask.
  - `liquidation::VecNum`: The per period charge of each asset that left, from `lq`, which was sliced to the mask's complement. Empty when `lq` is unset, because the axis has no length to build a vector of zeros from.

# Examples

```jldoctest
julia> fees = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002));

julia> PortfolioOptimisers.calc_asset_periodic_fees([0.5, 0.5], fees)
([0.006, 0.006], Float64[])
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_asset_one_off_fees`](@ref)
  - [`calc_total_asset_fees`](@ref)
  - [`calc_periodic_fees`](@ref)
"""
function calc_asset_periodic_fees(w::VecNum, fees::Fees)
    return (calc_asset_fees(w, fees.l, .>=) - calc_asset_fees(w, fees.s, .<) +
            calc_asset_fees(w, fees.tn), calc_asset_liquidation_fees(w, fees.lq))
end
"""
    calc_one_off_fees(w::VecNum, fees::Fees)

Charge the terms of a fee that fall one time over a holding period.

`fl` and `fs` are currency amounts charged one time for the whole holding period, and they are the only terms `fees.fa` reaches. The method carries no price, because a fixed fee is a currency amount already.

# Algorithm

 1. Charge the long fixed term, the call of [`calc_fixed_fees`](@ref) on `fees.fl` under `.>=`.
 2. Charge the short fixed term, the call of the same name on `fees.fs` under `.<`.
 3. Return the sum of the two terms.

# Arguments

  - `w`: Portfolio weights.
  - `fees`: [`Fees`](@ref) structure.

# Returns

  - `val::Number`: The charge of one holding period.

# Examples

```jldoctest
julia> fees = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002));

julia> PortfolioOptimisers.calc_one_off_fees([0.5, 0.5], fees)
10.0
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fixed_fees`](@ref)
  - [`calc_periodic_fees`](@ref)
  - [`calc_total_fees`](@ref)
  - [`calc_asset_one_off_fees`](@ref)
"""
function calc_one_off_fees(w::VecNum, fees::Fees)
    return calc_fixed_fees(w, fees.fl, fees.kwargs, .>=) +
           calc_fixed_fees(w, fees.fs, fees.kwargs, .<) +
           calc_fixed_liquidation_fees(fees.flq, fees.kwargs)
end
"""
    calc_asset_one_off_fees(w::VecNum, fees::Fees)

Split over the assets the terms of a fee that fall one time over a holding period.

The per asset twin of [`calc_one_off_fees`](@ref). Its entries sum to that number, up to the order of summation.

# Algorithm

 1. Charge the long fixed term, the call of [`calc_asset_fixed_fees`](@ref) on `fees.fl` under `.>=`.
 2. Charge the short fixed term, the call of the same name on `fees.fs` under `.<`.
 3. Charge the fixed forced exit, the call of [`calc_asset_fixed_liquidation_fees`](@ref) on `fees.flq`. `flq` is a currency amount charged one time, so it falls on the clock `fees.fa` names, beside `fl` and `fs`.
 4. Return the pair: the elementwise sum of the two vectors of steps 1 and 2, and the vector of step 3.

# Arguments

  - `w`: Portfolio weights.
  - `fees`: [`Fees`](@ref) structure.

# Returns

The verb returns a pair, one entry per axis of a reduced [`Fees`](@ref).

  - `investable::VecNum`: The one-off charge of each asset that stayed, from `fl` and `fs`, which were sliced to the Investable Mask.
  - `liquidation::VecNum`: The one-off charge of each asset that left, from `flq`, which was sliced to the mask's complement. Empty when `flq` is unset, because the axis has no length to build a vector of zeros from.

# Examples

```jldoctest
julia> fees = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002));

julia> PortfolioOptimisers.calc_asset_one_off_fees([0.5, 0.5], fees)
([5.0, 5.0], Float64[])
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_asset_fixed_fees`](@ref)
  - [`calc_asset_periodic_fees`](@ref)
  - [`calc_total_asset_fees`](@ref)
  - [`calc_one_off_fees`](@ref)
"""
function calc_asset_one_off_fees(w::VecNum, fees::Fees)
    return (calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=) +
            calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<),
            calc_asset_fixed_liquidation_fees(w, fees.flq, fees.kwargs))
end
"""
    calc_total_fees(w::VecNum, T::Number, fees::Option{<:Fees})

Charge the whole cost of holding a portfolio for `T` periods.

[`calc_fees`](@ref) answers one observation of a return series. This verb answers the whole holding period, so it charges the per period terms `T` times and the one-off terms one time. It needs the rates, the fixed amounts and the horizon, and nothing else. `fees.fa` reaches no term here, because that field names where a one-off cost lands on a return series, and this verb reports no series. The finite allocation reads this verb to take the fees out of the cash before it allocates.

# Algorithm

 1. Charge `T` times the per period terms, the call of [`calc_periodic_fees`](@ref).
 2. Charge the one-off terms one time, the call of [`calc_one_off_fees`](@ref).
 3. Return the sum of the two terms.

# Arguments

  - `w`: Portfolio weights.
  - `T`: Horizon, in periods.
  - `fees`: [`Fees`](@ref) structure, or `nothing`.

# Returns

  - `val::Number`: The whole cost of the holding period.

# Examples

```jldoctest
julia> fees = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002));

julia> calc_total_fees([0.5, 0.5], 252, fees)
13.024000000000001
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_fees`](@ref)
  - [`calc_periodic_fees`](@ref)
  - [`calc_one_off_fees`](@ref)
  - [`calc_total_asset_fees`](@ref)
"""
function calc_total_fees(w::VecNum, ::Number, ::Nothing)
    return zero(eltype(w))
end
function calc_total_fees(w::VecNum, T::Number, fees::Fees)
    return T * calc_periodic_fees(w, fees) + calc_one_off_fees(w, fees)
end
"""
    calc_total_asset_fees(w::VecNum, T::Number, fees::Option{<:Fees})

Split over the assets the whole cost of holding a portfolio for `T` periods.

The per asset twin of [`calc_total_fees`](@ref). Its entries sum to that number, up to the order of summation.

# Algorithm

 1. Charge `T` times the per period terms, the call of [`calc_asset_periodic_fees`](@ref). Both halves of its pair are scaled, because both are rates.
 2. Charge the one-off terms one time, the call of [`calc_asset_one_off_fees`](@ref). Neither half is scaled, because both are currency amounts.
 3. Return the pair, each axis summed with its own half: the investable total, and the liquidation total through [`add_liquidation_terms`](@ref), which answers the axis whose two terms are set independently.

# Arguments

  - `w`: Portfolio weights.
  - `T`: Horizon, in periods.
  - `fees`: [`Fees`](@ref) structure, or `nothing`.

# Returns

The verb returns a pair, one entry per axis of a reduced [`Fees`](@ref).

  - `investable::VecNum`: The whole cost of the holding period for each asset that stayed, from `l`, `s`, `tn`, `fl` and `fs`, which were sliced to the Investable Mask.
  - `liquidation::VecNum`: The whole cost of the forced exit of each asset that left, from `lq` and `flq`, which were sliced to the mask's complement. Empty when neither carrier is set.

# Examples

```jldoctest
julia> fees = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002));

julia> calc_total_asset_fees([0.5, 0.5], 252, fees)
([6.5120000000000005, 6.5120000000000005], Float64[])
```

# Related

  - [`Fees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`calc_asset_periodic_fees`](@ref)
  - [`calc_asset_one_off_fees`](@ref)
  - [`calc_total_fees`](@ref)
"""
function calc_total_asset_fees(w::VecNum, ::Number, ::Nothing)
    return (zeros(eltype(w), length(w)), zeros(eltype(w), 0))
end
function calc_total_asset_fees(w::VecNum, T::Number, fees::Fees)
    pi, pl = calc_asset_periodic_fees(w, fees)
    oi, ol = calc_asset_one_off_fees(w, fees)
    return (T * pi + oi, add_liquidation_terms(T * pl, ol))
end

export FeesEstimator, Fees, AmortisedFees, FirstObservationFees, fees_constraints,
       calc_fees, calc_fixed_fees, calc_asset_fees, calc_asset_fixed_fees, calc_total_fees,
       calc_total_asset_fees
