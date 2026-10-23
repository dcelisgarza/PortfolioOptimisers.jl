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

Measures the negative quadratic skewness of a portfolio, a convex part of its third moment.

The measure is the quadratic form of the weights in ``\\mathbf{V}``, or its square root. ``\\mathbf{V}`` sums the negative spectral parts of the slices of the coskewness matrix, so the measure is never negative, and it is larger for a portfolio whose returns have a heavier left tail. Smaller is better, so [`bigger_is_better`](@ref) returns `false`. To minimise the measure is the way of [nskew](@cite) to raise the skewness of a portfolio with a convex programme. [`set_risk_constraints!`](@ref) states the model, which is the second-order cone of Equation 2 of [nskew](@cite).

# Mathematical definition

```math
\\begin{align}
\\mathbf{M}_3 &= \\begin{bmatrix} \\mathbf{S}_1 & \\mathbf{S}_2 & \\cdots & \\mathbf{S}_N \\end{bmatrix}\\,, \\\\
\\mathbf{V} &= -\\sum_{i=1}^{N} \\mathbf{S}_i^{-}\\,, \\\\
\\mathrm{NSke}(\\boldsymbol{w}) &= \\begin{cases}
  \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w}} & \\text{under } \\texttt{SOCRiskExpr}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w} & \\text{under } \\texttt{QuadRiskExpr}\\,,\\ \\texttt{SquaredSOCRiskExpr}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:M3_cosk])
  - ``\\mathbf{S}_i``: Slice ``i`` of ``\\mathbf{M}_3``, a symmetric ``N \\times N`` matrix.
  - ``\\mathbf{S}_i^{-}``: Negative spectral part of ``\\mathbf{S}_i``, the sum of the terms of its eigendecomposition with a negative eigenvalue. It is negative semidefinite, so ``\\mathbf{V}`` is positive semidefinite.
  - ``\\mathbf{V}``: Negative spectral coskewness, ``N \\times N``. It is the measure's own `V` when the measure states one, and the prior's otherwise.
  - $(math_dict[:w_port])

[`negative_spectral_coskewness`](@ref) builds ``\\mathbf{V}``, and it then repairs the matrix with the matrix processing `mp`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NegativeSkewness(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        sk::Option{<:SkSlot} = nothing,
        V::Option{<:MatNum} = nothing,
        alg::NSkeFormulations = SOCRiskExpr(),
        window::Option{<:Int_VecInt} = nothing,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot()
    ) -> NegativeSkewness

Keywords correspond to the struct's fields.

## Validation

  - If `sk` is a matrix, `V` must be given as well, and the reverse. Both must be non-empty, with `size(sk, 1)^2 == size(sk, 2)` and `V` square.
  - If `sk` holds a **Deferred Quantity**, `V` must be `nothing`. The fit supplies the pair.
  - `window` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

!!! warning

    `sk` and `V` are a pair, and a stated `V` factors the `sk` beside it. A caller who wants one consistent pair names `sk` alone. A **Deferred Quantity** there supplies both from one fit, together with the `mp` that built them. A caller who states both by hand must make sure that they agree. A stated matrix is also fixed. It crosses a Cross-Validation fold as the answer for the whole universe, while a **Deferred Quantity** crosses unresolved and refits on the fold. A subset view slices `sk` and rebuilds `V` from the slice, with `mp` and `window`.

!!! info

    `sk` also admits a [`CoskewnessEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), which [`resolve_deferred_quantities`](@ref) resolves against the prior of the optimisation. `V` never defers. It comes from `sk`, so it comes out of that same fit. The processor that built it travels with it and **replaces** `mp`, so a later rebuild uses the same one. The measure carries one deferrable slot, so it takes no `pe`.

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
    window ┼ nothing
  mtx_sqrt ┴ EigenFallbackSquareRoot()
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Kurtosis`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`NSkeQuadFormulations`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`expected_risk`](@ref)
  - [`negative_spectral_coskewness`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:nskew]) Section 2, and Section 3, Equations 1 and 2.
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
    """
    Square-root algorithm of ``\\mathbf{V}`` that the SOC formulation reads, or `nothing` for the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. ``\\mathbf{V}`` keeps only the negative spectral parts of the coskewness slices, so it is often singular, and the default takes the square root of its eigendecomposition then. [`matrix_square_root`](@ref) states each algorithm. The quadratic formulation reads no square root.
    """
    mtx_sqrt
    function NegativeSkewness(settings::RiskMeasureSettings,
                              mp::AbstractMatrixProcessingEstimator, sk::Option{<:SkSlot},
                              V::Option{<:MatNum}, alg::NSkeFormulations,
                              window::Option{<:Int_VecInt},
                              mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm})
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
                   typeof(window), typeof(mtx_sqrt)}(settings, mp, sk, V, alg, window,
                                                     mtx_sqrt)
    end
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                          sk::Option{<:SkSlot} = nothing, V::Option{<:MatNum} = nothing,
                          alg::NSkeFormulations = SOCRiskExpr(),
                          window::Option{<:Int_VecInt} = nothing,
                          mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())::NegativeSkewness
    return NegativeSkewness(settings, mp, sk, V, alg, window, mtx_sqrt)
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

`sk` and `V` come from the same fit. `V` depends on the matrix processing as well as on `sk`, so the method also takes the processor of the fit and writes it to `mp`, in place of the one that the measure held. A [`CoskewnessEstimator`](@ref) gives the processor through [`coskewness_processor`](@ref). An [`AbstractPriorEstimator`](@ref) gives it as the field `skmp` of its [`HighOrderPrior`](@ref).

The method takes `V` from the fit, and does not compute it again from the resolved `sk`. A factor prior computes `V` with the returns that its factor model implies, so a new computation with the observed returns would give a different matrix. A later view in [`port_opt_view`](@ref) rebuilds `V` from a slice of `sk`, and it then uses the same processor, because the method wrote it to `mp`.

The measure carries one deferrable slot, so it has no `pe` to fan out. A coskewness estimator needs only a returns matrix, so `sk` resolves against a [`LowOrderPrior`](@ref) as well as against a [`HighOrderPrior`](@ref). A measure whose `sk` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`NegativeSkewness`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`coskewness_processor`](@ref)
  - [`fit_deferred_quantity`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function resolve_deferred_quantities(r::NegativeSkewness, pr::AbstractPriorResult,
                                     ::Any = nothing)::NegativeSkewness
    return r
end
function resolve_deferred_quantities(r::NegativeSkewness{<:Any, <:Any, <:DeferredQuantity},
                                     pr::AbstractPriorResult,
                                     ::Any = nothing)::NegativeSkewness
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
# The functor reads `V`, which the constructor ties to `sk`: the two are stated together or
# not at all, so the refusal names the slot the caller states — see `functor_slots`.
functor_slots(r::NegativeSkewness) = (; sk = r.sk)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`NegativeSkewness`](@ref) that holds its coskewness matrix and its negative spectral coskewness, from the measure or from the [`HighOrderPrior`](@ref) result `pr`.

The method selects the two field by field and not as a pair, because the constructor refuses every mixed state. A stated `sk` always carries its own `V`, and a deferred `sk` always resolves to both. So the method takes the pair of the prior only when the measure names neither.

# Algorithm

 1. Resolve a **Deferred Quantity** in `sk` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `sk` and `V` with [`nothing_scalar_array_selector`](@ref), from `r` or else from `pr.sk` and `pr.V`.
 3. Build a new `NegativeSkewness` from `sk`, `V` and the `settings`, `mp`, `alg`, `window` and `mtx_sqrt` of `r`.

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
                            window = r.window, mtx_sqrt = r.mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in [`NegativeSkewness`](@ref)'s `sk` slot against a [`LowOrderPrior`](@ref) result, and otherwise return `r` unchanged.

A [`LowOrderPrior`](@ref) holds no coskewness, so the method has no pair to take from `pr`. A coskewness estimator in `sk` needs only the returns matrix of the result, so it resolves here too. The model needs `V`, and [`set_risk_constraints!`](@ref) refuses a measure that has none.

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
                            window = r.window, mtx_sqrt = r.mtx_sqrt)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::NegativeSkewness) = WeightsInput()

export NegativeSkewness
