"""
    const NSkeFormulations = Union{<:NSkeQuadFormulations, <:SOCRiskExpr}

Union of valid optimisation formulations for the [`NegativeSkewness`](@ref) risk measure.

# Related

  - [`NSkeQuadFormulations`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`NegativeSkewness`](@ref)
"""
const NSkeFormulations = Union{<:NSkeQuadFormulations, <:SOCRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Represents the Negative Skewness risk measure.

`NegativeSkewness` quantifies the portfolio's exposure to negative asymmetry in returns by computing a quadratic or SOC (second-order cone) form of the coskewness matrix. It penalises portfolio constructions that exhibit heavy left-tail behaviour.

# Mathematical definition

Let ``\\boldsymbol{w}`` be the portfolio weight vector and ``\\mathbf{V}`` the negative semi-definite coskewness matrix (spectral decomposition of the negative part of the sample coskewness tensor). The Negative Skewness risk measure is:

```math
\\begin{align}
\\mathrm{NSke}(\\boldsymbol{w}) &= \\begin{cases}
  \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w}} & \\text{(SOC formulation)} \\\\
  \\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w} & \\text{(Quadratic formulation)}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{NSke}(\\boldsymbol{w})``: Negative Skewness risk measure.
  - $(math_dict[:w_port])
  - ``\\mathbf{V}``: Negative semi-definite coskewness matrix (spectral decomposition of the negative part of the sample coskewness tensor).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NegativeSkewness(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        sk::Option{<:SkSlot} = nothing,
        V::Option{<:MatNum} = nothing,
        alg::NSkeFormulations = SOCRiskExpr(),
        window::Option{<:Int_VecInt} = nothing
    ) -> NegativeSkewness

Keywords correspond to the struct's fields.

## Validation

  - If `sk` is a matrix, `V` must be given as well, and the reverse. Both must be non-empty, with `size(sk, 1)^2 == size(sk, 2)` and `V` square.
  - If `sk` holds a **Deferred Quantity**, `V` must be `nothing`. The fit supplies the pair.
  - `window` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

!!! warning

    `sk` and `V` are a pair, and a stated `V` factors the `sk` beside it. A caller who wants one consistent pair names `sk` alone — a **Deferred Quantity** there supplies both from one fit, together with the `mp` that built them. A caller who states both by hand must make sure that they agree. A stated matrix is also pinned: it crosses a Cross-Validation fold or a subset view as the whole universe's answer, while a **Deferred Quantity** crosses unresolved and refits on the subset.

!!! info

    `sk` also admits a [`CoskewnessEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), resolved against the optimisation's own prior — see [`resolve_deferred_quantities`](@ref). `V` never defers: it is derived from `sk`, so it travels out of that same fit. The processor that built it travels with it and **replaces** `mp`, so a later rebuild uses the same one. The measure carries one deferrable slot, so it takes no `pe`.

# Functor

    (r::NegativeSkewness)(w::VecNum)

Computes the Negative Skewness risk of a portfolio weight vector `w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.

# Examples

```jldoctest
julia> NegativeSkewness()
NegativeSkewness
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        mp ┼ MatrixProcessing
           │     pdm ┼ Posdef
           │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │      dn ┼ nothing
           │      dt ┼ nothing
           │     alg ┼ nothing
           │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        sk ┼ nothing
         V ┼ nothing
       alg ┼ SOCRiskExpr()
    window ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Kurtosis`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`NSkeQuadFormulations`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`expected_risk`](@ref)

# References

  - $(ref_dict[:nskew])
"""
@concrete struct NegativeSkewness <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:sk_slot])
    """
    sk
    """
    $(field_dict[:V_slot])
    """
    V
    """
    $(field_dict[:alg])
    """
    alg
    """
    $(field_dict[:window])
    """
    window
    function NegativeSkewness(settings::RiskMeasureSettings,
                              mp::AbstractMatrixProcessingEstimator, sk::Option{<:SkSlot},
                              V::Option{<:MatNum}, alg::NSkeFormulations,
                              window::Option{<:Int_VecInt})
        if isa(sk, DeferredQuantity)
            assert_derived_slot_has_source(V, sk, :V, :sk)
        else
            sk_flag = isnothing(sk)
            V_flag = isnothing(V)
            if sk_flag || V_flag
                @argcheck(sk_flag,
                          IsNothingError("V cannot be nothing when sk is provided"))
                @argcheck(V_flag, IsNothingError("sk cannot be nothing when V is provided"))
            else
                @argcheck(!isempty(sk), IsEmptyError("sk cannot be empty"))
                @argcheck(!isempty(V), IsEmptyError("V cannot be empty"))
                @argcheck(size(sk, 1)^2 == size(sk, 2),
                          DimensionMismatch("size(sk, 1)^2 = $(size(sk, 1)^2) must equal size(sk, 2) = $(size(sk, 2))"))
                assert_matrix_issquare(V, :V)
            end
        end
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(settings), typeof(mp), typeof(sk), typeof(V), typeof(alg),
                   typeof(window)}(settings, mp, sk, V, alg, window)
    end
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                          sk::Option{<:SkSlot} = nothing, V::Option{<:MatNum} = nothing,
                          alg::NSkeFormulations = SOCRiskExpr(),
                          window::Option{<:Int_VecInt} = nothing)::NegativeSkewness
    return NegativeSkewness(settings, mp, sk, V, alg, window)
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:SOCRiskExpr})(w::VecNum)
    return sqrt(LinearAlgebra.dot(w, r.V, w))
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:NSkeQuadFormulations})(w::VecNum)
    return LinearAlgebra.dot(w, r.V, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in [`NegativeSkewness`](@ref)'s `sk` slot against prior result `pr`.

`sk` and `V` travel together, so both come from the same fit. `V = negative_spectral_coskewness(sk, X, mp)` is never a function of `sk` alone, so **the fit's own processor** builds it and is recorded in `mp` in place of the one the measure held. A [`CoskewnessEstimator`](@ref) supplies it through [`coskewness_processor`](@ref); an [`AbstractPriorEstimator`](@ref) supplies it as the prior result's `skmp`, which is the field [`HighOrderPrior`](@ref) already carries for exactly this reason.

Recording it keeps the windowed rebuild in [`port_opt_view`](@ref) on the same processor that built the `V` it replaces. `V` is never rebuilt from a resolved `sk`: under a factor prior the negative spectral part is special, and a rebuild would throw that structure away. This is the `sigma`/`chol` rule on the `sk`/`V` pair.

The measure carries one deferrable slot, so there is no fan-out to make and it takes no `pe`. A coskewness estimator needs only a returns matrix, so `sk` resolves against a [`LowOrderPrior`](@ref) as readily as against a [`HighOrderPrior`](@ref).

# Related

  - [`NegativeSkewness`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`coskewness_processor`](@ref)
  - [`fit_deferred_quantity`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function resolve_deferred_quantities(r::NegativeSkewness, pr::AbstractPriorResult,
                                     ::Any = nothing)::NegativeSkewness
    if !isa(r.sk, DeferredQuantity)
        return r
    end
    fitted = fit_deferred_quantity(r.sk, pr)
    skmp = deferred_derived_quantity(fitted, :skmp)
    return rebuild_with_slots(r,
                              (; mp = isnothing(skmp) ? r.mp : skmp,
                               sk = deferred_quantity(fitted, :sk),
                               V = deferred_derived_quantity(fitted, :V)))
end
# Deferrable slots — see `deferred_slots`. `V` is derived and never defers on its own, and
# `mp` holds a processor by design.
deferred_slots(r::NegativeSkewness) = (; sk = r.sk)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`NegativeSkewness`](@ref) by resolving a **Deferred Quantity** in `sk`, then falling back to a [`HighOrderPrior`](@ref) result for the coskewness matrix and its spectral decomposition.

The two are selected field by field rather than as a pair, because the constructor already refuses every mixed state: a stated `sk` always carries its own `V`, and a deferred `sk` always resolves to both at once. So the fallback is reached only when the measure names neither.

# Related

  - [`NegativeSkewness`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::NegativeSkewness, pr::HighOrderPrior, args...;
                 kwargs...)::NegativeSkewness
    r = resolve_deferred_quantities(r, pr)
    sk = nothing_scalar_array_selector(r.sk, pr.sk)
    V = nothing_scalar_array_selector(r.V, pr.V)
    return NegativeSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V, alg = r.alg,
                            window = r.window)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in [`NegativeSkewness`](@ref)'s `sk` slot against a [`LowOrderPrior`](@ref) result, and otherwise return `r` unchanged.

Coskewness is not available on a [`LowOrderPrior`](@ref), so there is no fallback to make. A coskewness estimator in `sk` needs only the returns matrix the result carries, so it resolves here all the same.

# Related

  - [`NegativeSkewness`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
function factory(r::NegativeSkewness, pr::LowOrderPrior, args...;
                 kwargs...)::NegativeSkewness
    return resolve_deferred_quantities(r, pr)
end
function port_opt_view(r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any}, ::Any,
                       args...)::NegativeSkewness
    return r
end
function port_opt_view(r::NegativeSkewness{<:Any, <:Any, <:MatNum, <:MatNum}, i, X::MatNum,
                       args...)::NegativeSkewness
    sk = r.sk
    idx = fourth_moment_index_generator(size(sk, 1), i)
    sk = nothing_scalar_array_view_odd_order(r.sk, i, idx)
    window = get_window(r.window, X)
    V = negative_spectral_coskewness(sk, view(X, window, i), r.mp)
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V,
                            window = r.window)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::NegativeSkewness) = WeightsInput()

export NegativeSkewness
