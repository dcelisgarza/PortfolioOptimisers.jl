"""
$(DocStringExtensions.TYPEDEF)

Weights a risk measure inside an aggregate, and bounds its risk expression from **below**.

This is the settings type of a quantity that the optimisation wants **more** of, so its bound is a floor and not a ceiling. It is the skewness floor ``\\bar{\\mu}_3`` of Problem 13 of [sdpmom](@cite). [`Skewness`](@ref) is its one holder, and a stated `lb` reaches the model only through [`VarianceSkewKurtosis`](@ref), which builds the skewness term. `Skewness` is a [`NonOptimisationRiskMeasure`](@ref), so no optimiser takes it alone.

Unlike the `ub` of [`RiskMeasureSettings`](@ref), `lb` is not checked for non-negativity. A third moment can be negative, so a negative floor is valid.

!!! warning

    The floor holds the relaxed third moment of the model, not the third moment of the returned weights. The method of [`set_risk_constraints!`](@ref) for [`VarianceSkewKurtosis`](@ref) states the gap under `## Relaxation`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaxRiskMeasureSettings(;
        scale::Number = 1.0,
        lb::Option{<:RkRtBounds} = nothing,
        rke::Bool = true,
    ) -> MaxRiskMeasureSettings

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(scale)`.
  - `lb` is validated with [`assert_nonempty_finite_val`](@ref).

# Examples

```jldoctest
julia> MaxRiskMeasureSettings()
MaxRiskMeasureSettings
  scale ┼ Float64: 1.0
     lb ┼ nothing
    rke ┴ Bool: true
```

# Related

  - [`JuMPRiskMeasureSettings`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Skewness`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
  - [`Frontier`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)

# References

  - $(ref_dict[:sdpmom]) Section 3, Problem 13.
"""
@concrete struct MaxRiskMeasureSettings <: JuMPRiskMeasureSettings
    """
    $(field_dict[:scale_rm])
    """
    scale
    """
    $(field_dict[:lb_rms])
    """
    lb
    """
    $(field_dict[:rke])
    """
    rke
    function MaxRiskMeasureSettings(scale::Number, lb::Option{<:RkRtBounds},
                                    rke::Bool)::MaxRiskMeasureSettings
        assert_nonempty_finite_val(lb, :lb)
        @argcheck(isfinite(scale), IsNonFiniteError("scale must be finite, got $scale"))
        return new{typeof(scale), typeof(lb), typeof(rke)}(scale, lb, rke)
    end
end
function MaxRiskMeasureSettings(; scale::Number = 1.0, lb::Option{<:RkRtBounds} = nothing,
                                rke::Bool = true)::MaxRiskMeasureSettings
    return MaxRiskMeasureSettings(scale, lb, rke)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the third central moment of the portfolio returns from the coskewness matrix `sk`.

The moment is ``\\boldsymbol{w}^\\intercal \\mathbf{M}_3 (\\boldsymbol{w} \\otimes \\boldsymbol{w})``. It is not divided by the cube of the standard deviation, so it is not the skewness coefficient. A larger moment is better, so [`bigger_is_better`](@ref) returns `true`. The measure is a [`NonOptimisationRiskMeasure`](@ref) and carries a [`MaxRiskMeasureSettings`](@ref), whose bound is the floor `lb`. So it answers [`bigger_is_better`](@ref), [`bounds_risk_measure`](@ref), [`no_bounds_risk_measure`](@ref), [`no_risk_expr_risk_measure`](@ref), [`no_bounds_no_risk_expr_risk_measure`](@ref) and [`unit_scale_risk_measure`](@ref) with methods of its own: the methods for a [`RiskMeasure`](@ref) build a [`RiskMeasureSettings`](@ref), which has a ceiling `ub` and no floor.

No optimiser takes the measure alone. [`VarianceSkewKurtosis`](@ref) reads its `sk` and its `settings`, and builds the term ``m_3(\\boldsymbol{w})`` from `sk`. The functor reads the same matrix, so `settings.lb` and `settings.scale` act on the value of the functor, and [`expected_risk`](@ref) gives the value that the floor holds.

!!! info

    The matrix states the moment, so the centre and the observation weights belong to the estimator that makes the matrix, such as `sk = Coskewness(; w = w)`. [`ThirdCentralMoment`](@ref) computes the third moment from the returns, with observation weights and a Moment Target of its own. With a variance estimator in its `ve`, it divides the moment by the cube of the standard deviation and gives the standardised skewness. That measure reads one return series, so it is the score for an asset selection or a parameter search.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Skew}(\\boldsymbol{w}) &= m_3(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:m3_port])
  - $(math_dict[:M3_cosk])
  - $(math_dict[:w_port])

``m_3`` is the observation mean of the cube of the centred portfolio returns when ``\\mathbf{M}_3`` is the sample coskewness about the per-asset centres. That is Equation 1 of [sdpmom](@cite).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Skewness(;
        settings::MaxRiskMeasureSettings = MaxRiskMeasureSettings(),
        sk::Option{<:SkSlot} = nothing
    ) -> Skewness

Keywords correspond to the struct's fields.

## Validation

  - If `sk` is a matrix: `!isempty(sk)` and `size(sk, 1)^2 == size(sk, 2)`.

!!! info

    `sk` also admits a [`CoskewnessEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), which [`resolve_deferred_quantities`](@ref) resolves against the prior of the optimisation. The measure carries one deferrable slot, so it takes no `pe`.

# Functor

    (r::Skewness)(w::VecNum)

Computes ``m_3(\\boldsymbol{w})`` from the matrix `r.sk`, with [`portfolio_third_moment`](@ref). The functor reads `sk` as it stands, so a measure whose `sk` is `nothing` or a **Deferred Quantity** needs [`factory`](@ref) with a [`HighOrderPrior`](@ref) result first.

## Arguments

  - $(arg_dict[:pw])

# Examples

```jldoctest
julia> Skewness()
Skewness
  settings ┼ MaxRiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      lb ┼ nothing
           │     rke ┴ Bool: true
        sk ┴ nothing

julia> X = [0.02 -0.01; -0.01 0.03; 0.03 -0.02; -0.04 0.01; 0.01 0.02];

julia> w = [0.6, 0.4];

julia> Skewness(; sk = first(coskewness(Coskewness(), X)))(w)
-2.331648000000001e-6

julia> ThirdCentralMoment()(w, X)
-2.3316480000000003e-6
```

The sample coskewness about the mean of each asset gives the third moment of the portfolio returns about their mean, so the two values agree.

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`ThirdCentralMoment`](@ref): the third moment of the returns, standardised or not.
  - [`bigger_is_better`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`portfolio_third_moment`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
  - [`NegativeSkewness`](@ref): a convex part of the third moment that an optimiser can minimise alone.

# References

  - $(ref_dict[:sdpmom]) Section 2, Equation 1.
"""
@concrete struct Skewness <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:sk_slot])
    """
    sk
    function Skewness(settings::MaxRiskMeasureSettings, sk::Option{<:SkSlot})
        if isa(sk, MatNum)
            @argcheck(!isempty(sk), IsEmptyError("sk cannot be empty"))
            @argcheck(size(sk, 1)^2 == size(sk, 2),
                      DimensionMismatch("size(sk, 1)^2 ($(size(sk, 1)^2)) must equal size(sk, 2) ($(size(sk, 2)))"))
        end
        return new{typeof(settings), typeof(sk)}(settings, sk)
    end
end
function Skewness(; settings::MaxRiskMeasureSettings = MaxRiskMeasureSettings(),
                  sk::Option{<:SkSlot} = nothing)::Skewness
    return Skewness(settings, sk)
end
function (r::Skewness)(w::VecNum)
    return portfolio_third_moment(r.sk, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `sk` slot of [`Skewness`](@ref) `r` against prior result `pr`.

`sk` is the one slot of the measure that a prior can supply, so the slot holds the estimator itself. The measure reads no `V`, so it keeps only the `sk` half of the coskewness pair. [`NegativeSkewness`](@ref) reads both halves. A coskewness estimator centres the matrix on its own `me`. A measure whose `sk` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`Skewness`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::Skewness, pr::AbstractPriorResult,
                                     ::Any = nothing)::Skewness
    return r
end
function resolve_deferred_quantities(r::Skewness{<:Any, <:DeferredQuantity},
                                     pr::AbstractPriorResult, ::Any = nothing)::Skewness
    return rebuild_with_slots(r, (; sk = resolve_slot(r.sk, :sk, pr)))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::Skewness) = (; sk = r.sk)
# The functor reads `sk` as it stands, so an empty `sk` is refused on a value-level route —
# see `functor_slots`.
functor_slots(r::Skewness) = (; sk = r.sk)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`, because a larger third moment is better.

The generic method for an [`AbstractBaseRiskMeasure`](@ref) returns `false`. The asset selection, the parameter search and the ranking of cross-validation results read this trait to choose the direction of the score.

# Related

  - [`Skewness`](@ref)
  - [`bigger_is_better`](@ref)
"""
function bigger_is_better(::Skewness)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Skewness`](@ref) that holds its coskewness matrix, from the measure or from the [`HighOrderPrior`](@ref) result `pr`.

# Algorithm

 1. Resolve a **Deferred Quantity** in `sk` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `sk` with [`nothing_scalar_array_selector`](@ref). A measure that states a matrix keeps it, and a measure whose `sk` is `nothing` takes `pr.sk`.
 3. Build a new `Skewness` from `sk` and the `settings` of `r`.

# Related

  - [`Skewness`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Skewness, pr::HighOrderPrior, args...; kwargs...)::Skewness
    r = resolve_deferred_quantities(r, pr)
    return Skewness(; settings = r.settings,
                    sk = nothing_scalar_array_selector(r.sk, pr.sk))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `sk` slot of [`Skewness`](@ref) `r` against a [`LowOrderPrior`](@ref) result, and otherwise return `r` unchanged.

A [`LowOrderPrior`](@ref) holds no coskewness matrix, so the method has no matrix to take from `pr`. A coskewness estimator in `sk` needs only the returns of the result, so it resolves here too. The functor and the model need a matrix. A value-level entry point refuses a measure whose `sk` is `nothing`, and so does [`set_risk_constraints!`](@ref).

# Related

  - [`Skewness`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
function factory(r::Skewness, pr::LowOrderPrior, args...; kwargs...)::Skewness
    return resolve_deferred_quantities(r, pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return [`Skewness`](@ref) `r` unchanged, because its `sk` holds no matrix to slice.

`sk` is `nothing`, or it holds a **Deferred Quantity**, which crosses the view unresolved and then fits on the subset.

# Related

  - [`Skewness`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(r::Skewness, i, args...)::Skewness
    return r
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Skewness`](@ref) `r` sliced to asset indices `i`, with a view of its coskewness matrix `sk`.

The view agrees with the coskewness that a refit on the assets `i` gives.

# Algorithm

 1. Compute `idx`, the columns of `sk` that belong to the pairs of assets in `i`, with [`fourth_moment_index_generator`](@ref).
 2. Take the view of `sk` at the rows `i` and the columns `idx` with [`nothing_scalar_array_view_odd_order`](@ref).
 3. Build a new `Skewness` from the view and the `settings` of `r`.

# Related

  - [`Skewness`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view_odd_order`](@ref)
  - [`fourth_moment_index_generator`](@ref)
"""
function port_opt_view(r::Skewness{<:Any, <:MatNum}, i, args...)::Skewness
    idx = fourth_moment_index_generator(size(r.sk, 1), i)
    sk = nothing_scalar_array_view_odd_order(r.sk, i, idx)
    return Skewness(; settings = r.settings, sk = sk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with `rke = false`, so that the measure adds nothing to the risk of the objective.

The copy keeps `lb` and `scale`. The constructor of [`VarianceSkewKurtosis`](@ref) calls this method on its `sk`, so the composite registers the skewness term once, inside its own risk expression. The method exists because the method for a [`RiskMeasure`](@ref) builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
"""
function no_risk_expr_risk_measure(r::Skewness)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = false, lb = r.settings.lb,
                                                      scale = r.settings.scale), sk = r.sk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with `rke = false`, `lb = nothing` and a unit `scale`.

This is the copy that a measure takes when it only measures a distance, as the method for a [`RiskMeasure`](@ref) does. The method exists because that method builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::Skewness, ::Any = nothing)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = false, lb = nothing,
                                                      scale = one(r.settings.scale)),
                    sk = r.sk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with the floor `lb` set to `ub`.

[`bounds_risk_measure`](@ref) passes the bound of every measure as `ub`. The bound of a [`MaxRiskMeasureSettings`](@ref) is a floor, so this method writes it to `lb`, and keeps `rke` and `scale`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`bounds_risk_measure`](@ref)
"""
function bounds_risk_measure(r::Skewness, ub::Number)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = r.settings.rke, lb = ub,
                                                      scale = r.settings.scale), sk = r.sk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with `lb = nothing`, and keep `rke` and `scale`.

[`no_bounds_risk_measure`](@ref) of a [`VarianceSkewKurtosis`](@ref) calls this method on its `sk`, so that the floor of the skewness term leaves the model with the other bounds. The method exists because the method for a [`RiskMeasure`](@ref) builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`no_bounds_risk_measure`](@ref)
"""
function no_bounds_risk_measure(r::Skewness, ::Any = nothing)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = r.settings.rke, lb = nothing,
                                                      scale = r.settings.scale), sk = r.sk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with a unit `scale`, and keep `lb` and `rke`.

A measure that already has a unit `scale` returns unchanged. The method exists because the method for a [`RiskMeasure`](@ref) builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`unit_scale_risk_measure`](@ref)
"""
function unit_scale_risk_measure(r::Skewness)
    settings = r.settings
    scale = settings.scale
    if isone(scale)
        return r
    end
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = settings.rke,
                                                      lb = settings.lb, scale = one(scale)),
                    sk = r.sk)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures a weighted sum of the variance, the third central moment and the fourth central moment of the portfolio returns.

The third moment enters with a negative sign, because a larger skewness is better. Each term takes the `scale` of its child as its weight, and the bound of its child as its bound. The skewness bound is the floor `sk.settings.lb`, and the other two are ceilings. The terms are the raw moments: the third moment is not standardised, and the fourth moment is not square-rooted under any `alg` of `kt`. The model and the functor read the matrices of the children. To weight a term, name a **Deferred Quantity** that carries the weights, such as `kt = Kurtosis(; kt = Cokurtosis(; w = w))`.

[`set_risk_constraints!`](@ref) states the model. It is the semidefinite relaxation of Problems 12 to 14 of [sdpmom](@cite), and that work states that the relaxation is tight over a range of skewness values only. The `## Relaxation` of the method states the gap of each term.

!!! warning

    A skewness scale ``s_{\\mathrm{sk}}`` that is large against ``s_{\\sigma^2}`` and ``s_{\\kappa}`` loosens the relaxation. The optimum moves to relaxation blocks far from any product of the weights, and the value of the model stops tracking the value of the functor, without bound. A first-order solver can then stop at a point that breaks the weight bounds and the budget, and the solution check accepts that point as solved when `check_sol = (; allow_almost = true)`. Raise ``s_{\\mathrm{sk}}`` in small steps, and compare the value of the model with the value of the functor at the returned weights. The default `check_sol` rejects an approximate solution, so a solver that stops short fails and returns no such point.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}(\\boldsymbol{w}) &= s_{\\sigma^2}\\, \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}
    - s_{\\mathrm{sk}}\\, m_3(\\boldsymbol{w})
    + s_{\\kappa}\\, m_4(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:m3_port])
  - $(math_dict[:M3_cosk])
  - $(math_dict[:m4_port])
  - $(math_dict[:K_cokurt])
  - ``s_{\\sigma^2}``, ``s_{\\mathrm{sk}}``, ``s_{\\kappa}``: The `settings.scale` of `vr`, `sk` and `kt`.

The three matrices are the moments of the returns, so ``\\mathcal{R}`` is Equation 2 of [sdpmom](@cite) for orders two and four and Equation 1 for order three. With the default prior, the covariance has the divisor ``T - 1`` and the two higher moments have the divisor ``T``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceSkewKurtosis(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        vr::Variance = Variance(),
        sk::Skewness = Skewness(),
        kt::Kurtosis = Kurtosis(),
        pe::Option{<:AbstractPriorEstimator} = nothing
    ) -> VarianceSkewKurtosis

Keywords correspond to the struct's fields.

!!! warning

    The three children each state their quantities independently, so nothing makes `sigma`, `sk` and `kt` agree with each other. A caller who wants one consistent set names `pe` on the container alone and lets it fill all three from a single fit. A child that names its own quantity keeps it, and the `pe` fills only what is left.

!!! info

    The constructor passes `vr`, `sk` and `kt` through [`no_risk_expr_risk_measure`](@ref), so a child never registers a risk expression of its own. This is why the children show `rke ┴ Bool: false` below.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `vr`: Recursively updated through [`factory`](@ref).
  - `sk`: Recursively updated through [`factory`](@ref).
  - `kt`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `vr`: Recursively viewed through [`port_opt_view`](@ref).
  - `sk`: Recursively viewed through [`port_opt_view`](@ref).
  - `kt`: Recursively viewed through [`port_opt_view`](@ref).

# Functor

    (r::VarianceSkewKurtosis)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)

Computes ``\\mathcal{R}(\\boldsymbol{w})`` from `vr.sigma`, `sk.sk` and `kt.kt`, so a measure needs [`factory`](@ref) with a [`HighOrderPrior`](@ref) result first, unless its children state the three matrices. `X` and `fees` take part in no term. The matrices fix the moments, and a central moment does not move when `fees` shifts every observation by the same constant.

## Arguments

  - $(arg_dict[:pw])
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - $(arg_dict[:fees])

# Examples

```jldoctest
julia> r = VarianceSkewKurtosis()
VarianceSkewKurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        vr ┼ Variance
           │    settings ┼ RiskMeasureSettings
           │             │   scale ┼ Float64: 1.0
           │             │      ub ┼ nothing
           │             │     rke ┴ Bool: false
           │       sigma ┼ nothing
           │        chol ┼ nothing
           │          rc ┼ nothing
           │         alg ┼ SquaredSOCRiskExpr()
           │    mtx_sqrt ┼ EigenFallbackSquareRoot()
           │   ratio_alg ┼ SquareRootRatio()
           │      corner ┴ nothing
        sk ┼ Skewness
           │   settings ┼ MaxRiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      lb ┼ nothing
           │            │     rke ┴ Bool: false
           │         sk ┴ nothing
        kt ┼ Kurtosis
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │         kt ┼ nothing
           │          N ┼ nothing
           │        alg ┼ SOCRiskExpr()
           │   mtx_sqrt ┴ EigenFallbackSquareRoot()
        pe ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Variance`](@ref)
  - [`Skewness`](@ref)
  - [`Kurtosis`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`set_risk_constraints!`](@ref): states the model and its relaxation.
  - [`no_bounds_risk_measure`](@ref): removes the bounds of the container and of the three children.

# References

  - $(ref_dict[:sdpmom]) Section 2, Equations 1 and 2, and Section 3, Problems 12 to 14.
"""
@propagatable @concrete struct VarianceSkewKurtosis <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:vr_rm])
    """
    @fprop @vprop vr
    """
    $(field_dict[:sk_rm])
    """
    @fprop @vprop sk
    """
    $(field_dict[:kt_rm])
    """
    @fprop @vprop kt
    """
    $(field_dict[:pe_rm])
    """
    pe
    function VarianceSkewKurtosis(settings::RiskMeasureSettings, vr::Variance, sk::Skewness,
                                  kt::Kurtosis, pe::Option{<:AbstractPriorEstimator})
        assert_returns_source(returns_source(pe), (nothing,), :VarianceSkewKurtosis,
                              "pe.x_src")
        vr = no_risk_expr_risk_measure(vr)
        sk = no_risk_expr_risk_measure(sk)
        kt = no_risk_expr_risk_measure(kt)
        return new{typeof(settings), typeof(vr), typeof(sk), typeof(kt), typeof(pe)}(settings,
                                                                                     vr, sk,
                                                                                     kt, pe)
    end
end
function VarianceSkewKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              vr::Variance = Variance(), sk::Skewness = Skewness(),
                              kt::Kurtosis = Kurtosis(),
                              pe::Option{<:AbstractPriorEstimator} = nothing)
    return VarianceSkewKurtosis(settings, vr, sk, kt, pe)
end
# `X` and `fees` are unused, and both facts follow from the model. The three terms are the
# raw moments read off the pinned comoment matrices, exactly as `set_risk_constraints!` reads
# them off the relaxation blocks, so no return series enters. Fees shift every observation by
# the same per-period constant given `w`, and a central moment is invariant to that shift, so
# dropping them changes no value. The fourth-moment term is the raw moment under every `alg` of
# `r.kt`, while the functor of `r.kt` reports its square root under `SOCRiskExpr`, so the terms
# read the matrices and do not delegate to the children.
function (r::VarianceSkewKurtosis)(w::VecNum, ::MatNum, ::Option{<:Fees} = nothing)
    v, s, k = variance_skew_kurtosis_terms(r, w)
    return v - s + k
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the three scaled terms of [`VarianceSkewKurtosis`](@ref) `r` at the weights `w`, as the tuple `(v, s, k)`.

The functor of `r` returns `v - s + k`. The terms have the degrees of homogeneity 2, 3 and 4, so the sum has no single degree. [`adjusted_risk`](@ref) divides each term by its own degree, and so [`risk_contribution`](@ref) decomposes the sum exactly.

# Mathematical definition

```math
\\begin{align}
v &= s_{\\sigma^2}\\, \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,, \\\\
s &= s_{\\mathrm{sk}}\\, m_3(\\boldsymbol{w})\\,, \\\\
k &= s_{\\kappa}\\, m_4(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:m3_port])
  - $(math_dict[:m4_port])
  - ``s_{\\sigma^2}``, ``s_{\\mathrm{sk}}``, ``s_{\\kappa}``: The `settings.scale` of `vr`, `sk` and `kt`.

# Arguments

  - `r::VarianceSkewKurtosis`: Measure whose matrices are stated.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.

# Returns

  - `(v, s, k)::Tuple`: The scaled variance, third moment and fourth moment terms.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`adjusted_risk`](@ref)
  - [`homogeneity_degree`](@ref)
"""
function variance_skew_kurtosis_terms(r::VarianceSkewKurtosis, w::VecNum)
    return (r.vr(w) * r.vr.settings.scale,
            portfolio_third_moment(r.sk.sk, w) * r.sk.settings.scale,
            portfolio_fourth_moment(r.kt.kt, w) * r.kt.settings.scale)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the third moment of the portfolio at the weights `w`, read off the coskewness matrix `sk`.

This is the term that the model of [`VarianceSkewKurtosis`](@ref) relaxes, that the `lb` of a [`Skewness`](@ref) bounds, and that the functor of `Skewness` gives.

Where:

  - $(math_dict[:m3_port])
  - $(math_dict[:M3_cosk])

# Arguments

  - `sk::MatNum`: Coskewness matrix `assets × assets^2`.
  - $(arg_dict[:pw])

# Returns

  - `m3::Number`: The third moment of the portfolio.

# Related

  - [`portfolio_fourth_moment`](@ref)
  - [`variance_skew_kurtosis_terms`](@ref)
  - [`Skewness`](@ref)
"""
function portfolio_third_moment(sk::MatNum, w::VecNum)
    return LinearAlgebra.dot(w, sk, LinearAlgebra.kron(w, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the fourth moment of the portfolio at the weights `w`, read off the cokurtosis matrix `kt`.

This is the term that the models of [`Kurtosis`](@ref) and of [`VarianceSkewKurtosis`](@ref) read. The functor of `Kurtosis` gives it under the [`QuadSecondMomentFormulations`](@ref), and its square root under [`SOCRiskExpr`](@ref).

Where:

  - $(math_dict[:m4_port])
  - $(math_dict[:K_cokurt])

# Arguments

  - `kt::MatNum`: Cokurtosis matrix `assets^2 × assets^2`.
  - $(arg_dict[:pw])

# Returns

  - `m4::Number`: The fourth moment of the portfolio.

# Related

  - [`portfolio_third_moment`](@ref)
  - [`variance_skew_kurtosis_terms`](@ref)
  - [`Kurtosis`](@ref)
"""
function portfolio_fourth_moment(kt::MatNum, w::VecNum)
    w2 = LinearAlgebra.kron(w, w)
    return LinearAlgebra.dot(w2, kt, w2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`VarianceSkewKurtosis`](@ref) `r` against prior result `pr`.

A child that names its own quantity keeps it, whether it states the quantity or defers it. The composite adds a variance, a skewness and a kurtosis term, so a caller who wants the three to describe one distribution names `pe` on the container and states nothing on the children.

# Algorithm

 1. Resolve each child with its own method of [`resolve_deferred_quantities`](@ref), giving `vr`, `sk` and `kt`.
 2. When `pe` is not `nothing`, fit it once against `pr` with [`fan_out_pe`](@ref), and fill each child slot that is still `nothing` from that one fit: `sigma` and its `chol` on `vr`, `sk` on `sk`, and `kt` on `kt`.
 3. Return the rebuilt measure with `pe = nothing`.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::VarianceSkewKurtosis, pr::AbstractPriorResult,
                                     slv = nothing)
    vr = resolve_deferred_quantities(r.vr, pr, slv)
    sk = resolve_deferred_quantities(r.sk, pr, slv)
    kt = resolve_deferred_quantities(r.kt, pr, slv)
    return fan_out_pe(r, r.pe, pr, (; vr = vr, sk = sk, kt = kt))
end
# The children carry their own slots, and `chol` is derived from `sigma`, so the fan-out of
# this measure is its own method of `fan_out_pe`.
function fan_out_pe(r::VarianceSkewKurtosis, pe::AbstractPriorEstimator,
                    pr::AbstractPriorResult, slots::NamedTuple)
    (; vr, sk, kt) = slots
    fitted = fit_deferred_quantity(pe, pr)
    # `chol` is derived from `sigma`, so it comes from the fan-out only when the fan-out
    # also supplies the `sigma` it factorises.
    chol = isnothing(vr.sigma) ? deferred_derived_quantity(fitted, :chol) : vr.chol
    vr = rebuild_with_slots(vr,
                            (; sigma = fan_out_slot(fitted, vr.sigma, :sigma), chol = chol))
    sk = rebuild_with_slots(sk, (; sk = fan_out_slot(fitted, sk.sk, :sk)))
    kt = rebuild_with_slots(kt, (; kt = fan_out_slot(fitted, kt.kt, :kt)))
    return rebuild_with_slots(r, (; vr = vr, sk = sk, kt = kt, pe = nothing))
end
# Deferrable slots — see `deferred_slots`. The three children carry their own, so the check
# recurses into them.
deferred_slots(r::VarianceSkewKurtosis) = (; vr = r.vr, sk = r.sk, kt = r.kt, pe = r.pe)
# The functor reads `sigma`, `sk` and `kt` off the children, and each child names its own in
# `functor_slots`, so the check reaches them through `deferred_slots`.
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`VarianceSkewKurtosis`](@ref) by fanning `pe` out over its three children, then threading `pr` into each of them.

The container holds no quantity of its own, so [`@fprop`](@ref) alone would reach the children and leave `pe` in place, and the measure would then hold a `pe` that it never applied. This method resolves first, which is the same order that [`set_risk_constraints!`](@ref) uses.

# Algorithm

 1. Resolve the container with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Thread `pr` into each child with its own method of [`factory`](@ref), giving `vr`, `sk` and `kt`.
 3. Build a new `VarianceSkewKurtosis` from the three children and the `settings` of `r`, with `pe = nothing`.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`factory`](@ref)
"""
function factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    return VarianceSkewKurtosis(; settings = r.settings,
                                vr = factory(r.vr, pr, args...; kwargs...),
                                sk = factory(r.sk, pr, args...; kwargs...),
                                kt = factory(r.kt, pr, args...; kwargs...), pe = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`VarianceSkewKurtosis`](@ref) `r` without its bounds, the bound of the container and the bounds of its three children.

A child holds the bound of its own term, so the method for a [`RiskMeasure`](@ref), which rebuilds only `settings`, would leave `vr.settings.ub`, `sk.settings.lb` and `kt.settings.ub` in the model. [`NearOptimalCentering`](@ref) solves its corner portfolios with this copy, so a child bound would constrain a corner that must be free. Every child and the container keep `rke` and `scale`, because the scale of a child weights its term.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`no_bounds_risk_measure`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_risk_measure(r::VarianceSkewKurtosis, flag::Any = nothing)
    return VarianceSkewKurtosis(;
                                settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                               scale = r.settings.scale),
                                vr = no_bounds_risk_measure(r.vr, flag),
                                sk = no_bounds_risk_measure(r.sk, flag),
                                kt = no_bounds_risk_measure(r.kt, flag), pe = r.pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`VarianceSkewKurtosis`](@ref) `r` without its bounds and outside the risk of the objective.

The container takes `rke = false` and a unit `scale`, as the method for a [`RiskMeasure`](@ref) does. The children lose their bounds through [`no_bounds_risk_measure`](@ref) and keep their scales, because the scale of a child weights its term in the composite.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`no_bounds_risk_measure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::VarianceSkewKurtosis, flag::Any = nothing)
    return VarianceSkewKurtosis(;
                                settings = RiskMeasureSettings(; rke = false,
                                                               scale = one(r.settings.scale)),
                                vr = no_bounds_risk_measure(r.vr, flag),
                                sk = no_bounds_risk_measure(r.sk, flag),
                                kt = no_bounds_risk_measure(r.kt, flag), pe = r.pe)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Skewness) = WeightsInput()
risk_input_kind(::VarianceSkewKurtosis) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: the functor of [`VarianceSkewKurtosis`](@ref) reads the weights in all three terms, through the comoment matrices, and has no form that reads a series alone.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
"""
supports_precomputed_returns(::VarianceSkewKurtosis)::Bool = false

export MaxRiskMeasureSettings, Skewness, VarianceSkewKurtosis
