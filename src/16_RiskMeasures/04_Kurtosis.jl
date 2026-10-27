"""
$(DocStringExtensions.TYPEDEF)

Measures the fourth central moment of the portfolio returns, or its square root, from the cokurtosis matrix `kt`.

The moment is the quadratic form of ``\\boldsymbol{w} \\otimes \\boldsymbol{w}`` in ``\\mathbf{K}``. It is not divided by the square of the variance, so it is not the kurtosis coefficient, and it is not an excess kurtosis. `alg` selects the quantity that the measure reports: [`SOCRiskExpr`](@ref) reports ``\\sqrt{m_4}``, and each of the [`QuadSecondMomentFormulations`](@ref) reports ``m_4``. The functor and the model read the same matrix, so [`expected_risk`](@ref) gives the value that a bound of the measure holds.

[`set_risk_constraints!`](@ref) states the model, and neither model is exact. When `N` is `nothing`, the model is problem 13 of [pkurt](@cite), which replaces ``\\boldsymbol{w} \\boldsymbol{w}^\\intercal`` with a lifted matrix and drops the condition that the lift has rank one. When `N` is an integer, the model is the sum of squared quadratic forms of [pkurtapprox](@cite), and a larger `N` is more accurate and more expensive. The library calls the first model the exact formulation and the second the approximate formulation, because only the second truncates the cokurtosis. The `## Relaxation` of each method states the gap.

!!! warning

    With an integer `N`, the model bounds the approximate moment of [`cokurtosis_spectrum`](@ref), which lies at or below the moment of the matrix. The functor reads the whole matrix, so it reports the full moment, and the value at the solved weights can lie above the bound of the model.

!!! info

    The matrix states the moment, so the centre, the observation weights and the tail of the moment belong to the estimator that makes the matrix. A semi-moment needs a lower semi-cokurtosis, such as `kt = Cokurtosis(; alg = SemiMoment())`, and a weighted moment needs a weighted cokurtosis, such as `kt = Cokurtosis(; w = w)`. A lower semi-cokurtosis gives the heuristic of Section 3.1 of [pkurt](@cite). It clips the deviations of each asset, not the deviations of the portfolio, so its moment is not the semi-moment of the portfolio returns. The measures that compute a fourth moment from the returns are [`HighOrderMoment`](@ref) with [`FourthMoment`](@ref), which gives ``m_4``, and [`LowOrderMoment`](@ref) with [`EvenMoment`](@ref) at `p = 2`, which gives ``\\sqrt{m_4}``. Each takes observation weights, a Moment Target and a [`SemiMoment`](@ref) of its own.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Kurt}(\\boldsymbol{w}) &= \\begin{cases}
  \\sqrt{m_4(\\boldsymbol{w})} & \\text{under } \\texttt{SOCRiskExpr}\\,, \\\\
  m_4(\\boldsymbol{w}) & \\text{under } \\texttt{QuadRiskExpr}\\,,\\ \\texttt{SquaredSOCRiskExpr}\\,,\\ \\texttt{RSOCRiskExpr}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:m4_port])
  - $(math_dict[:K_cokurt])
  - $(math_dict[:w_port])

``m_4`` is the observation mean of the fourth power of the centred portfolio returns when ``\\mathbf{K}`` is the sample cokurtosis about the per-asset centres. That is Equation 4 of [pkurt](@cite).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Kurtosis(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        kt::Option{<:KtSlot} = nothing,
        N::Option{<:Integer} = nothing,
        alg::SecondMomentFormulation = SOCRiskExpr(),
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
    ) -> Kurtosis

Keywords correspond to the struct's fields.

## Validation

  - If `kt` is a matrix: `!isempty(kt)` and `size(kt, 1) == size(kt, 2)`.
  - If `N` is not `nothing`: must be positive.

!!! info

    `kt` also admits a [`CokurtosisEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), which [`resolve_deferred_quantities`](@ref) resolves against the prior of the optimisation. The measure carries one deferrable slot, so it takes no `pe`.

# Functor

    (r::Kurtosis)(w::VecNum)

Computes ``\\sqrt{m_4(\\boldsymbol{w})}`` or ``m_4(\\boldsymbol{w})`` from the matrix `r.kt`, with [`portfolio_fourth_moment`](@ref). The functor reads `kt` as it stands, so a measure whose `kt` is `nothing` or a **Deferred Quantity** needs [`factory`](@ref) with a [`HighOrderPrior`](@ref) result first.

## Arguments

  - $(arg_dict[:pw])

# Examples

```jldoctest
julia> Kurtosis()
Kurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        kt ┼ nothing
         N ┼ nothing
       alg ┼ SOCRiskExpr()
  mtx_sqrt ┴ EigenFallbackSquareRoot()

julia> X = [0.02 -0.01; -0.01 0.03; 0.03 -0.02; -0.04 0.01; 0.01 0.02];

julia> w = [0.6, 0.4];

julia> Kurtosis(; kt = cokurtosis(Cokurtosis(), X))(w)
0.00025455401627160726

julia> LowOrderMoment(; alg = EvenMoment())(w, X)
0.00025455401627159604
```

The sample cokurtosis about the mean of each asset gives the root of the fourth moment of the portfolio returns about their mean, so the two values agree.

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`QuadSecondMomentFormulations`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`HighOrderMoment`](@ref): the fourth moment of the returns, with [`FourthMoment`](@ref).
  - [`EvenMoment`](@ref): the root of the fourth moment of the returns, at `p = 2`.
  - [`resolve_deferred_quantities`](@ref)
  - [`expected_risk`](@ref)
  - [`portfolio_fourth_moment`](@ref)
  - [`set_risk_constraints!`](@ref): states both models and their relaxations.

# References

  - $(ref_dict[:pkurt]) Section 2, Equation 4, and Section 3.1, Problem 13.
  - $(ref_dict[:pkurtapprox])
"""
@concrete struct Kurtosis <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:kt_slot])
    """
    kt
    """
    $(field_dict[:N_kt])
    """
    N
    """
    $(field_dict[:alg])
    """
    alg
    """
    Square-root algorithm of the projected co-kurtosis matrix ``\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal`` that the exact formulation reads, or `nothing` for the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. A sample co-kurtosis of ``T`` observations has rank at most ``T``, so the projected matrix is singular when ``N(N+1)/2 > T`` and no matrix processing repairs it. The default takes the square root of the eigendecomposition of such a matrix. [`matrix_square_root`](@ref) states each algorithm. The approximate formulation reads no square root.
    """
    mtx_sqrt
    function Kurtosis(settings::RiskMeasureSettings, kt::Option{<:KtSlot},
                      N::Option{<:Integer}, alg::SecondMomentFormulation,
                      mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm})
        if isa(kt, MatNum)
            @argcheck(!isempty(kt), IsEmptyError("kt cannot be empty"))
            assert_matrix_issquare(kt, :kt)
        end
        if !isnothing(N)
            @argcheck(N > zero(N), DomainError(N, "N must be positive"))
        end
        return new{typeof(settings), typeof(kt), typeof(N), typeof(alg), typeof(mtx_sqrt)}(settings,
                                                                                           kt,
                                                                                           N,
                                                                                           alg,
                                                                                           mtx_sqrt)
    end
end
function Kurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  kt::Option{<:KtSlot} = nothing, N::Option{<:Integer} = nothing,
                  alg::SecondMomentFormulation = SOCRiskExpr(),
                  mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())::Kurtosis
    return Kurtosis(settings, kt, N, alg, mtx_sqrt)
end
function (r::Kurtosis{<:Any, <:Any, <:Any, <:SOCRiskExpr})(w::VecNum)
    return sqrt(portfolio_fourth_moment(r.kt, w))
end
function (r::Kurtosis{<:Any, <:Any, <:Any, <:QuadSecondMomentFormulations})(w::VecNum)
    return portfolio_fourth_moment(r.kt, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `kt` slot of [`Kurtosis`](@ref) `r` against prior result `pr`.

`kt` is the one slot of the measure that a prior can supply, so the slot holds the estimator itself, and no other slot reads its fit. A cokurtosis estimator centres the matrix on its own `me`. A prior estimator gives the cokurtosis of its [`HighOrderPrior`](@ref). A measure whose `kt` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`Kurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::Kurtosis, pr::AbstractPriorResult,
                                     ::Any = nothing)::Kurtosis
    return r
end
function resolve_deferred_quantities(r::Kurtosis{<:Any, <:DeferredQuantity},
                                     pr::AbstractPriorResult, ::Any = nothing)::Kurtosis
    return rebuild_with_slots(r, (; kt = resolve_slot(r.kt, :kt, pr)))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::Kurtosis) = (; kt = r.kt)
# The functor reads `kt` as it stands, so an empty `kt` is refused on a value-level route —
# see `functor_slots`.
functor_slots(r::Kurtosis) = (; kt = r.kt)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Kurtosis`](@ref) that holds its cokurtosis matrix, from the measure or from the [`HighOrderPrior`](@ref) result `pr`.

# Algorithm

 1. Resolve a **Deferred Quantity** in `kt` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `kt` with [`nothing_scalar_array_selector`](@ref). A measure that states a matrix keeps it, and a measure whose `kt` is `nothing` takes `pr.kt`.
 3. Build a new `Kurtosis` from `kt` and from the `settings`, `N`, `alg` and `mtx_sqrt` of `r`.

# Related

  - [`Kurtosis`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Kurtosis, pr::HighOrderPrior, args...; kwargs...)::Kurtosis
    r = resolve_deferred_quantities(r, pr)
    kt = nothing_scalar_array_selector(r.kt, pr.kt)
    return Kurtosis(; settings = r.settings, kt = kt, N = r.N, alg = r.alg,
                    mtx_sqrt = r.mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `kt` slot of [`Kurtosis`](@ref) `r` against a [`LowOrderPrior`](@ref) result, and otherwise return `r` unchanged.

A [`LowOrderPrior`](@ref) holds no cokurtosis matrix, so the method has no matrix to take from `pr`. A cokurtosis estimator in `kt` needs only the returns of the result, so it resolves here too. The functor and the model need a matrix. A value-level entry point refuses a measure whose `kt` is `nothing`, and so does [`set_risk_constraints!`](@ref).

# Related

  - [`Kurtosis`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
function factory(r::Kurtosis, pr::LowOrderPrior, args...; kwargs...)::Kurtosis
    return resolve_deferred_quantities(r, pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Kurtosis`](@ref) `r` sliced to asset indices `i`.

The view of a stated matrix agrees with the cokurtosis that a refit on the assets `i` gives. A **Deferred Quantity** and `nothing` pass through unsliced, and a **Deferred Quantity** resolves on the subset.

# Algorithm

 1. Return `r` unchanged when `kt` is not a matrix.
 2. Compute `idx`, the rows and columns of `kt` that belong to the pairs of assets in `i`, with [`fourth_moment_index_generator`](@ref). The number of assets is the square root of the size of `kt`.
 3. Build a new `Kurtosis` from the view of `kt` at `idx` and from the other fields of `r`.

# Related

  - [`Kurtosis`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
  - [`fourth_moment_index_generator`](@ref)
"""
function port_opt_view(r::Kurtosis, i, args...)::Kurtosis
    kt = r.kt
    if !isa(kt, MatNum)
        return r
    end
    idx = fourth_moment_index_generator(isqrt(size(kt, 1)), i)
    return Kurtosis(; settings = r.settings, kt = nothing_scalar_array_view(kt, idx),
                    N = r.N, alg = r.alg, mtx_sqrt = r.mtx_sqrt)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Kurtosis) = WeightsInput()

export Kurtosis
