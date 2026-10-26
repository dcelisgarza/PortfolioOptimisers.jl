"""
$(DocStringExtensions.TYPEDEF)

Measures the fourth central moment of the portfolio returns, or its square root.

The moment has the divisor ``T``. It is not divided by the square of the variance, so it is not the kurtosis coefficient, and it is not an excess kurtosis. `alg2` selects the quantity that the measure reports: [`SOCRiskExpr`](@ref) reports ``\\sqrt{m_4}``, and each of the [`QuadSecondMomentFormulations`](@ref) reports ``m_4``. `alg1` selects the full moment or the lower semi-moment of the functor.

[`set_risk_constraints!`](@ref) states the model, and neither model is exact. When `N` is `nothing`, the model is problem 13 of [pkurt](@cite), which replaces ``\\boldsymbol{w} \\boldsymbol{w}^\\intercal`` with a lifted matrix and drops the condition that the lift has rank one. When `N` is an integer, the model is the sum of squared quadratic forms of [pkurtapprox](@cite), and a larger `N` is more accurate and more expensive. The library calls the first model the exact formulation and the second the approximate formulation, because only the second truncates the cokurtosis. The `## Relaxation` of each method states the gap.

!!! warning

    `alg1` and `w` act on the functor alone. The functor gets only the weights and the returns, so it has no cokurtosis to read, and `alg1` and `w` tell it which moment to take. The model reads `kt`, or the cokurtosis of the prior, and reads neither field. So under [`SemiMoment`](@ref) with a full cokurtosis, the functor reports the lower semi-moment and the model minimises the full moment. With observation weights in `w`, the functor reports the weighted moment and the model minimises the moment of the tensor. To give the model a semi-moment or a weighted moment, name a **Deferred Quantity** in `kt`, such as `kt = Cokurtosis(; alg = SemiMoment())` or `kt = Cokurtosis(; w = w)`. [`resolve_deferred_quantities`](@ref) fits it against the prior of the optimisation, as it fits a full `Cokurtosis()` or any other cokurtosis estimator. A lower semi-cokurtosis gives the heuristic of Section 3.1 of [pkurt](@cite). That model clips the deviations of each asset, not the deviations of the portfolio, so its value is not the semi-moment that the functor reports.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= x_t - \\mu\\,, \\\\
m_4(\\boldsymbol{w}) &= \\frac{1}{T} \\sum_{t=1}^{T} \\delta_t^4\\,, \\\\
m_4^{-}(\\boldsymbol{w}) &= \\frac{1}{T} \\sum_{t=1}^{T} \\min(\\delta_t, 0)^4\\,, \\\\
\\mathrm{Kurt}(\\boldsymbol{w}) &= \\begin{cases}
  \\sqrt{m_4(\\boldsymbol{w})} & \\text{under } \\texttt{SOCRiskExpr}\\,, \\\\
  m_4(\\boldsymbol{w}) & \\text{under } \\texttt{QuadRiskExpr}\\,,\\ \\texttt{SquaredSOCRiskExpr}\\,,\\ \\texttt{RSOCRiskExpr}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Centre of the moment. [`calc_moment_target`](@ref) states it for each type of `mu`.
  - ``\\delta_t``: Centred portfolio return at observation ``t``.
  - $(math_dict[:m4_port])
  - ``m_4^{-}(\\boldsymbol{w})``: Lower semi-moment of order four, under [`SemiMoment`](@ref). It replaces ``m_4`` in the last line.
  - $(math_dict[:K_cokurt])
  - $(math_dict[:w_port])

The two forms of ``m_4`` agree when ``\\mathbf{K}`` is the cokurtosis about the per-asset centres whose image under ``\\boldsymbol{w}`` is ``\\mu``. That is Equation 4 of [pkurt](@cite). When `w` holds observation weights, a weighted mean replaces each ``\\frac{1}{T} \\sum_{t}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Kurtosis(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:MuSlot} = nothing,
        kt::Option{<:KtSlot} = nothing,
        N::Option{<:Integer} = nothing,
        alg1::AbstractMomentAlgorithm = FullMoment(),
        alg2::SecondMomentFormulation = SOCRiskExpr(),
        pe::Option{<:AbstractPriorEstimator} = nothing,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
    ) -> Kurtosis

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is not `nothing`:

      + `::Number`: `isfinite(mu)`.
      + `::VecNum`: `!isempty(mu)` and `all(isfinite, mu)`.

  - $(val_dict[:oow_rm])

  - If `kt` is not `nothing`:

      + `!isempty(kt)` and `size(kt, 1) == size(kt, 2)`.

      + If `mu` is not `nothing`:

          * `::VecNum`: `length(mu)^2 == size(kt, 1)`.
          * `::VecScalar`: `length(mu.v)^2 == size(kt, 1)`.

  - If `N` is not `nothing`: must be positive.

!!! warning

    `mu` and `kt` are stated independently, so nothing makes them agree with each other. A caller who wants one consistent set names `pe` alone and lets it fill both from a single fit. A caller who states them by hand must make sure that they agree.

!!! info

    `kt` also admits a [`CokurtosisEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), and `mu` an [`AbstractExpectedReturnsEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref). [`resolve_deferred_quantities`](@ref) resolves either against the prior of the optimisation. A cokurtosis estimator in `kt` also supplies `mu` from its own `me`, so that the tensor and its centre come out of **one** object. A deferred slot wins over `pe`.

!!! warning

    A `Number` or a `VecScalar` `mu` beside a [`CokurtosisEstimator`](@ref) in `kt` gives the target of the measure only when the weights sum to one. The estimator takes the tensor about the per-asset centre that [`centring_target`](@ref) makes, and a scalar part ``s`` is spread to every asset. So the tensor centres the portfolio return on ``s \\sum_i w_i``, and the functor centres it on ``s``. Under any other budget, the `JuMP` model and the functor disagree. A budget of 0.5, a leveraged budget and a market-neutral budget are examples. A vector `mu`, or `mu = nothing`, agrees under every budget.

# Examples

```jldoctest
julia> Kurtosis()
Kurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
        kt ┼ nothing
         N ┼ nothing
      alg1 ┼ FullMoment()
      alg2 ┼ SOCRiskExpr()
        pe ┼ nothing
  mtx_sqrt ┴ EigenFallbackSquareRoot()
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`QuadSecondMomentFormulations`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`expected_risk`](@ref)
  - [`calc_moment_target`](@ref)
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
    $(field_dict[:oow])
    """
    w
    """
    $(field_dict[:mu_slot])
    """
    mu
    """
    $(field_dict[:kt_slot])
    """
    kt
    """
    $(field_dict[:N_kt])
    """
    N
    """
    $(field_dict[:alg1])
    """
    alg1
    """
    $(field_dict[:alg2])
    """
    alg2
    """
    $(field_dict[:pe_rm])
    """
    pe
    """
    Square-root algorithm of the projected co-kurtosis matrix ``\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal`` that the exact formulation reads, or `nothing` for the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. A sample co-kurtosis of ``T`` observations has rank at most ``T``, so the projected matrix is singular when ``N(N+1)/2 > T`` and no matrix processing repairs it. The default takes the square root of the eigendecomposition of such a matrix. [`matrix_square_root`](@ref) states each algorithm. The approximate formulation reads no square root.
    """
    mtx_sqrt
    function Kurtosis(settings::RiskMeasureSettings, w::Option{<:ObsWeights},
                      mu::Option{<:MuSlot}, kt::Option{<:KtSlot}, N::Option{<:Integer},
                      alg1::AbstractMomentAlgorithm, alg2::SecondMomentFormulation,
                      pe::Option{<:AbstractPriorEstimator},
                      mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm})
        assert_returns_source(returns_source(pe), (nothing,), :Kurtosis, "pe.x_src")
        mu_flag = isa(mu, VecNum)
        kt_flag = isa(kt, MatNum)
        if mu_flag
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu), IsNonFiniteError("mu must be finite, got $mu"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        assert_observation_weights(w, :w)
        if kt_flag
            @argcheck(!isempty(kt), IsEmptyError("kt cannot be empty"))
            assert_matrix_issquare(kt, :kt)
        end
        if mu_flag && kt_flag
            @argcheck(length(mu)^2 == size(kt, 1),
                      DimensionMismatch("length(mu)^2 ($(length(mu)^2)) must match size(kt, 1) ($(size(kt, 1)))"))
        elseif isa(mu, VecScalar) && kt_flag
            @argcheck(length(mu.v)^2 == size(kt, 1),
                      DimensionMismatch("length(mu.v)^2 ($(length(mu.v)^2)) must match size(kt, 1) ($(size(kt, 1)))"))
        end
        if !isnothing(N)
            @argcheck(N > zero(N), DomainError(N, "N must be positive"))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(kt), typeof(N),
                   typeof(alg1), typeof(alg2), typeof(pe), typeof(mtx_sqrt)}(settings, w,
                                                                             mu, kt, N,
                                                                             alg1, alg2, pe,
                                                                             mtx_sqrt)
    end
end
function Kurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  w::Option{<:ObsWeights} = nothing, mu::Option{<:MuSlot} = nothing,
                  kt::Option{<:KtSlot} = nothing, N::Option{<:Integer} = nothing,
                  alg1::AbstractMomentAlgorithm = FullMoment(),
                  alg2::SecondMomentFormulation = SOCRiskExpr(),
                  pe::Option{<:AbstractPriorEstimator} = nothing,
                  mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())::Kurtosis
    return Kurtosis(settings, w, mu, kt, N, alg1, alg2, pe, mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`Kurtosis`](@ref) `r` against prior result `pr`.

A deferred slot **wins over `pe`**, because `pe` fills only the slots that the first two steps leave `nothing`. `kt` lives on a [`HighOrderPrior`](@ref), so a prior estimator named in `kt` or `pe` must compute one. A measure that holds no **Deferred Quantity** and no `pe` returns unchanged.

# Algorithm

 1. Resolve a deferred `mu` on its own, giving `mu`.
 2. Resolve a deferred `kt` with its centre, giving `kt` and `mu`. `kt` is a moment **about** a centre, so the two come out of one object. When `mu` is still `nothing`, [`deferred_centre`](@ref) reads the centre off the `me` of the cokurtosis estimator, the fit takes it as `mean =`, and it becomes the resolved `mu`. A stated `mu` takes its place in the fit. An [`AbstractPriorEstimator`](@ref) centres itself, so the centre is read back off the prior result that it gives.
 3. Fan `pe` out into each slot that steps 1 and 2 left `nothing`, with [`fan_out_pe`](@ref), and return the rebuilt measure.

# Related

  - [`Kurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_centre`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::Kurtosis, pr::AbstractPriorResult,
                                     ::Any = nothing)::Kurtosis
    if isnothing(r.pe) && !isa(r.mu, DeferredQuantity) && !isa(r.kt, DeferredQuantity)
        return r
    end
    mu = resolve_slot(r.mu, :mu, pr)
    kt = r.kt
    if isa(kt, DeferredQuantity)
        centre = isnothing(mu) ? deferred_centre(kt, pr) : mu
        fitted = fit_deferred_moment(kt, pr, centre)
        kt = deferred_quantity(fitted, :kt)
        mu = nothing_scalar_array_selector(centre, deferred_derived_quantity(fitted, :mu))
    end
    return fan_out_pe(r, r.pe, pr, (; mu = mu, kt = kt))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::Kurtosis) = (; mu = r.mu, kt = r.kt, pe = r.pe)
"""
    calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, ...}, ::Any, x::VecNum)
    calc_moment_target(r::Kurtosis{<:Any, <:ObsWeights, Nothing, ...}, ::Any, x::VecNum)
    calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecNum, ...}, w::VecNum, ::Any)
    calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecScalar, ...}, w::VecNum, ::Any)
    calc_moment_target(r::Kurtosis{<:Any, <:Any, <:Number, ...}, ::Any, ::Any)

Compute the centre ``\\mu`` about which [`Kurtosis`](@ref) takes its moment.

The method dispatches on the types of `r.w` and `r.mu`. After [`factory`](@ref) with a prior result, a `mu` that was `nothing` holds the mean of the prior, so the centre is the image of that mean under ``\\boldsymbol{w}``, and observation weights in `r.w` change the average but not the centre.

# Mathematical definition

```math
\\begin{align}
\\mu &= \\begin{cases}
  \\frac{1}{T} \\sum_{t=1}^{T} x_t & \\texttt{mu} \\text{ and } \\texttt{w} \\text{ are } \\texttt{nothing}\\,, \\\\
  \\sum_{t=1}^{T} \\omega_t x_t \\Big/ \\sum_{t=1}^{T} \\omega_t & \\texttt{mu} \\text{ is } \\texttt{nothing} \\text{ and } \\texttt{w} \\text{ holds weights}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\boldsymbol{\\mu} & \\texttt{mu} \\text{ is a vector}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\boldsymbol{v} + s & \\texttt{mu} \\text{ is a } \\texttt{VecScalar}\\,, \\\\
  \\mu_0 & \\texttt{mu} \\text{ is a number}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\omega_t``: Observation weight of observation ``t``, from `r.w`.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\mu}``: Vector `mu`, one centre for each asset.
  - ``\\boldsymbol{v}``, ``s``: The fields `v` and `s` of a [`VecScalar`](@ref) `mu`.
  - ``\\mu_0``: Number `mu`.

# Related

  - [`Kurtosis`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_deviations_vec`](@ref)
"""
function calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, <:Any, <:Any, <:Any, <:Any},
                            ::Any, x::VecNum)
    return Statistics.mean(x)
end
function calc_moment_target(r::Kurtosis{<:Any, <:ObsWeights, Nothing, <:Any, <:Any, <:Any,
                                        <:Any}, ::Any, x::VecNum)
    return Statistics.mean(x, r.w)
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecNum, <:Any, <:Any, <:Any, <:Any},
                            w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecScalar, <:Any, <:Any, <:Any,
                                        <:Any}, w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:Number, <:Any, <:Any, <:Any, <:Any},
                            ::Any, ::Any)
    return r.mu
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deviations ``\\delta_t`` of the net portfolio returns from the centre of [`Kurtosis`](@ref) `r`.

# Algorithm

 1. Compute `x`, the portfolio returns net of `fees`, with [`calc_net_returns`](@ref).
 2. Compute `tgt`, the centre of [`calc_moment_target`](@ref) less the mean fee per observation of [`moment_target_fees`](@ref). The fee is not zero only for a per-asset `mu`, which states a gross expected return.
 3. Return `x .- tgt`.

# Related

  - [`Kurtosis`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::Kurtosis, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x) - moment_target_fees(r.mu, w, fees, length(x))
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deviations ``\\delta_t`` of a precomputed series of portfolio returns `x` from the centre of [`Kurtosis`](@ref) `r`.

The precomputed-returns functor `r(x::VecNum)` calls this method. The series carries no weights, so the centre is one that [`supports_precomputed_returns`](@ref) accepts: the mean of `x`, its weighted mean, or a number.

# Related

  - [`Kurtosis`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::Kurtosis, x::VecNum)
    return x .- calc_moment_target(r, nothing, x)
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:FullMoment, <:SOCRiskExpr}, val::VecNum)
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w))
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:SemiMoment, <:SOCRiskExpr}, val::VecNum)
    val = min.(val, zero(eltype(val)))
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w))
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:FullMoment, <:QuadSecondMomentFormulations},
                     val::VecNum)
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:SemiMoment, <:QuadSecondMomentFormulations},
                     val::VecNum)
    val = min.(val, zero(eltype(val)))
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
# Evaluation entry points — see `MomentRiskMeasures` in `25_RiskMeasureTools.jl`.
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Kurtosis`](@ref) that holds its weights, its centre and its cokurtosis matrix, from the measure or from the [`HighOrderPrior`](@ref) result `pr`.

# Algorithm

 1. Resolve each **Deferred Quantity** with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `w`, `mu` and `kt` field by field with [`nothing_scalar_array_selector`](@ref). A field that the measure states keeps its value, and a field that is `nothing` takes `pr.w`, `pr.mu` or `pr.kt`.
 3. Build a new `Kurtosis` from the three fields and from the `settings`, `N`, `alg1`, `alg2` and `mtx_sqrt` of `r`, with `pe = nothing`.

# Related

  - [`Kurtosis`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Kurtosis, pr::HighOrderPrior, args...; kwargs...)::Kurtosis
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    kt = nothing_scalar_array_selector(r.kt, pr.kt)
    return Kurtosis(; settings = r.settings, w = w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2, pe = nothing, mtx_sqrt = r.mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Kurtosis`](@ref) that holds its weights and its centre, from the measure or from the [`LowOrderPrior`](@ref) result `pr`.

A [`LowOrderPrior`](@ref) holds no cokurtosis matrix, so `kt` keeps the matrix that the measure states, or the one that a **Deferred Quantity** resolves to. The functor reads only the returns, so a measure with `kt = nothing` evaluates. The model needs a matrix, and [`set_risk_constraints!`](@ref) refuses a measure that has none.

# Related

  - [`Kurtosis`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Kurtosis, pr::LowOrderPrior, args...; kwargs...)::Kurtosis
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    kt = nothing_scalar_array_selector(r.kt, nothing)
    return Kurtosis(; settings = r.settings, w = w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2, pe = nothing, mtx_sqrt = r.mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Kurtosis`](@ref) `r` sliced to asset indices `i`.

The view of a stated matrix agrees with the cokurtosis that a refit on the assets `i` gives. A **Deferred Quantity** passes through unsliced, and resolves on the subset.

# Algorithm

 1. Read the number of assets `j`, from the length of a vector `mu`, else from the square root of the size of a matrix `kt`. It is `nothing` when neither is stated.
 2. When `j` and `kt` are both stated, compute `idx`, the rows and columns of `kt` that belong to the pairs of assets in `i`, with [`fourth_moment_index_generator`](@ref), and take the view of `kt` at `idx`.
 3. Take the view of `mu` at `i` with [`nothing_scalar_array_view`](@ref).
 4. Build a new `Kurtosis` from `mu`, `kt` and the other fields of `r`.

# Related

  - [`Kurtosis`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
  - [`fourth_moment_index_generator`](@ref)
"""
function port_opt_view(r::Kurtosis, i, args...)::Kurtosis
    mu = r.mu
    kt = r.kt
    j = nothing
    j = if isa(mu, VecNum)
        length(mu)
    elseif isa(kt, MatNum)
        isqrt(size(kt, 1))
    end
    if !isnothing(j) && !isnothing(kt)
        idx = fourth_moment_index_generator(j, i)
        kt = nothing_scalar_array_view(kt, idx)
    end
    mu = nothing_scalar_array_view(mu, i)
    return Kurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2, pe = r.pe, mtx_sqrt = r.mtx_sqrt)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Kurtosis) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`Kurtosis`](@ref) `r` supports precomputed-return evaluation.

The answer is [`weight_independent_target`](@ref) of `r.mu`. It is `true` when `mu` is `nothing` or a number, because the centre is then a function of the series alone. It is `false` for a vector, a [`VecScalar`](@ref) and a **Deferred Quantity**, whose centre needs the weights.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`Kurtosis`](@ref)
"""
supports_precomputed_returns(r::Kurtosis) = weight_independent_target(r.mu)

export Kurtosis
