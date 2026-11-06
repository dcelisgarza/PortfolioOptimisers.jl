"""
$(DocStringExtensions.TYPEDEF)

Fits a box uncertainty set whose half-width is a fixed fraction of the absolute value of each prior moment.

The method is the delta method of Equation 11.15 of [cajas2025](@cite). `dmu` sets the fraction on the mean axis, and `dsigma` sets it on the covariance axis. Three estimators fit a box, and this one is the only one that draws no sample. The other two are [`NormalUncertaintySet`](@ref) and [`ARCHUncertaintySet`](@ref).

The two axes write different kinds of bounds. [`mu_delta_box_set`](@ref) writes a width, and [`sigma_delta_box_set`](@ref) writes absolute bounds. Read both before you read a fitted set entry by entry. The constructor accepts a fraction of zero on each axis. A zero gives that axis a box with no width, and the model then uses the nominal expression of that axis with no worst case.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DeltaUncertaintySet(;
        pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
        dmu::Number = 0.1,
        dsigma::Number = 0.1
    ) -> DeltaUncertaintySet

The keywords are the fields of the struct.

## Validation

  - `dmu >= 0`.
  - `dsigma >= 0`.

# Examples

```jldoctest
julia> DeltaUncertaintySet()
DeltaUncertaintySet
      pe ┼ EmpiricalPrior
         │           ce ┼ PortfolioOptimisersCovariance
         │              │   ce ┼ Covariance
         │              │      │    me ┼ SimpleExpectedReturns
         │              │      │       │   w ┴ nothing
         │              │      │    ce ┼ GeneralCovariance
         │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │              │      │       │    w ┴ nothing
         │              │      │   alg ┼ FullMoment()
         │              │      │     w ┴ nothing
         │              │   mp ┼ MatrixProcessing
         │              │      │     pdm ┼ Posdef
         │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │              │      │      dn ┼ nothing
         │              │      │      dt ┼ nothing
         │              │      │     alg ┼ nothing
         │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │           me ┼ SimpleExpectedReturns
         │              │   w ┴ nothing
         │      horizon ┼ nothing
         │   fill_limit ┼ nothing
         │        x_src ┴ nothing
     dmu ┼ Float64: 0.1
  dsigma ┴ Float64: 0.1
```

# Related

  - [`BoxUncertaintySet`](@ref): the result of both axes, which states the rule that the two axes read their bounds differently.
  - [`mu_delta_box_set`](@ref): the builder of the mean axis, which writes a width.
  - [`sigma_delta_box_set`](@ref): the builder of the covariance axis, which writes absolute bounds.
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractPriorEstimator`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.15. $(ref_dict[:no_original_source])
"""
@concrete struct DeltaUncertaintySet <: AbstractUncertaintySetEstimator
    """
    $(field_dict[:pe_ucs])
    """
    pe
    """
    $(field_dict[:dmu])
    """
    dmu
    """
    $(field_dict[:dsigma])
    """
    dsigma
    function DeltaUncertaintySet(pe::Option{<:AbstractLowOrderPriorEstimator}, dmu::Number,
                                 dsigma::Number)
        @argcheck(dmu >= 0.0, DomainError(dmu, "dmu must be >= 0"))
        @argcheck(dsigma >= 0.0, DomainError(dsigma, "dsigma must be >= 0"))
        return new{typeof(pe), typeof(dmu), typeof(dsigma)}(pe, dmu, dsigma)
    end
end
function DeltaUncertaintySet(;
                             pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
                             dmu::Number = 0.1, dsigma::Number = 0.1)::DeltaUncertaintySet
    return DeltaUncertaintySet(pe, dmu, dsigma)
end
"""
    mu_delta_box_set(pr, dmu::Number) -> BoxUncertaintySet

Builds the delta box of the mean axis, which writes a width in the upper bound and zero in the lower bound.

Neither bound is a bound on the mean. [`BoxUncertaintySet`](@ref) states the rule. On this axis, [`set_ucs_return_constraints!`](@ref) reads only the half-width ``(\\boldsymbol{u} - \\boldsymbol{\\ell}) / 2`` of the pair, and it centres that width on `val`. This lets a builder put the full width in ``\\boldsymbol{u}``, and this builder does that. The model sees the set ``\\hat{\\boldsymbol{\\mu}} \\pm \\delta_{\\mu} \\lvert \\hat{\\boldsymbol{\\mu}} \\rvert``, with the ``\\delta_{\\mu}`` of Equation 11.15.

The zero lower bound does not state that the mean is non-negative. `abs` sets the width, and it never moves the centre. For ``\\hat{\\mu}_i = -0.6`` and ``\\delta_{\\mu} = 0.1``, the builder writes ``\\ell_i = 0`` and ``u_i = 0.12``. The model then sees ``[-0.66,\\, -0.54]``, whose centre is the negative value. The builder of the covariance axis, [`sigma_delta_box_set`](@ref), is on the other side of the same rule.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\ell} &= \\boldsymbol{0}\\,, \\\\
\\boldsymbol{u} &= 2 \\delta_{\\mu} \\lvert \\hat{\\boldsymbol{\\mu}} \\rvert\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bounds that the builder writes.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - ``\\delta_{\\mu}``: Fraction of the absolute mean that sets the half-width of the mean box.
  - ``\\lvert \\cdot \\rvert``: Element-wise absolute value.

# Algorithm

 1. Build `lb`, a range of `length(pr.mu)` zeros of the element type of `pr.mu`. It is a range and not a vector, because every entry holds the same value.
 2. Build `ub` from `dmu` and the element-wise absolute value of `pr.mu`. It holds twice the half-width that the model sees, so the half-width is ``\\delta_{\\mu} \\lvert \\hat{\\boldsymbol{\\mu}} \\rvert``.
 3. Build a [`BoxUncertaintySet`](@ref) from `lb`, `ub` and `val = pr.mu`. `val` is the centre of the width.

# Arguments

  - `pr`: Prior Result. The builder reads only `pr.mu`.
  - `dmu`: Fraction of the absolute mean that sets the half-width. A `dmu` of zero writes `lb == ub == 0`. The half-width is then zero, and the worst case of the model is the nominal ``\\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w}``.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Box of the mean axis. Its `lb` is a range of zeros, its `ub` holds twice the half-width, and its `val` is `pr.mu`.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref): the type that states the rule for the two axes. This builder is on one side of the rule.
  - [`sigma_delta_box_set`](@ref): the other side of the rule, which writes absolute bounds.
  - [`set_ucs_return_constraints!`](@ref): the consumer that halves the difference of the bounds.

# References

  - $(ref_dict[:cajas2025]) Equation 11.15. $(ref_dict[:no_original_source])
"""
function mu_delta_box_set(pr, dmu::Number)
    return BoxUncertaintySet(;
                             lb = range(zero(eltype(pr.mu)), zero(eltype(pr.mu));
                                        length = length(pr.mu)), ub = dmu * abs.(pr.mu) * 2,
                             val = pr.mu)
end
"""
    sigma_delta_box_set(pr, dsigma::Number) -> BoxUncertaintySet

Builds the delta box of the covariance axis, which writes absolute bounds. Both bounds bind in the model.

The builder is on the other side of the rule that [`BoxUncertaintySet`](@ref) states. On this axis, [`set_ucs_variance_risk!`](@ref) reads ``\\operatorname{tr}(\\mathbf{A}_{u} \\mathbf{\\Sigma}_{u}) - \\operatorname{tr}(\\mathbf{A}_{l} \\mathbf{\\Sigma}_{l})`` under ``\\mathbf{A}_{u},\\, \\mathbf{A}_{l} \\geq 0`` and ``\\mathbf{A}_{u} - \\mathbf{A}_{l} = \\mathbf{W}``. Each bound enters the expression alone, and the consumer does not halve one bound against the other. This route has no centre, so it never reads `val`. The builder of the mean axis is [`mu_delta_box_set`](@ref).

The map is element-wise, and it does not scale the matrix. A positive entry of ``\\hat{\\mathbf{\\Sigma}}`` becomes ``1 - \\delta_{\\sigma}`` times its value in ``\\mathbf{\\Sigma}_{l}``. A negative entry becomes ``1 + \\delta_{\\sigma}`` times its value. So the order ``\\mathbf{\\Sigma}_{l} \\leq \\hat{\\mathbf{\\Sigma}} \\leq \\mathbf{\\Sigma}_{u}`` holds entry by entry at every ``\\delta_{\\sigma}``, for negative and positive entries. For example, ``-0.2`` with ``\\delta_{\\sigma} = 0.2`` gives ``-0.24`` and ``-0.16``.

The order of the positive semidefinite cone does not follow. This builder applies no `posdef!`, but `sigma_normal_box_set`, the builder of the normal box, applies `posdef!` to both bounds. A large ``\\delta_{\\sigma}`` can make ``\\mathbf{\\Sigma}_{l}`` indefinite. The consumer does not need a definite bound. It reads the two bounds entry by entry through the two traces, and it factorises neither bound. An indefinite lower bound builds and solves, and it makes the worst case larger.

The library documents this, and it adds no guard. A caller who sets ``\\delta_{\\sigma}`` far outside ``(0, 1)`` must justify the value.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{l} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} \\lvert \\hat{\\mathbf{\\Sigma}} \\rvert\\,, \\\\
\\mathbf{\\Sigma}_{u} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} \\lvert \\hat{\\mathbf{\\Sigma}} \\rvert\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{l}``, ``\\mathbf{\\Sigma}_{u}``: Lower and upper bounds of the covariance matrix.
  - $(math_dict[:Sigma_hat])
  - ``\\delta_{\\sigma}``: Fraction of the absolute covariance that sets the half-width of the covariance box.
  - ``\\lvert \\cdot \\rvert``: Element-wise absolute value.

# Algorithm

 1. Build `d_sigma`, the element-wise half-width, from `dsigma` and the element-wise absolute value of `pr.sigma`. It is non-negative at every entry, and this puts the two bounds in order.
 2. Subtract `d_sigma` from `pr.sigma` to get the lower bound. The lower bound is symmetric, because both operands are symmetric.
 3. Add `d_sigma` to `pr.sigma` to get the upper bound.
 4. Build a [`BoxUncertaintySet`](@ref) from the two bounds and `val = pr.sigma`, the covariance at the centre of the bounds. The mean route reads `val`, but the covariance route does not. On this axis, the field holds the centre for a reader of the result.

# Arguments

  - `pr`: Prior Result. The builder reads only `pr.sigma`.
  - `dsigma`: Fraction of the absolute covariance that sets the half-width. A `dsigma` of zero writes `lb == ub == pr.sigma`. The two traces then become ``\\operatorname{tr}(\\mathbf{W} \\hat{\\mathbf{\\Sigma}})``, and the model sees the nominal variance.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Box of the covariance axis. Its `lb` and `ub` are absolute bounds, and its `val` is `pr.sigma`.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref): the type that states the rule for the two axes. This builder is on one side of the rule.
  - [`mu_delta_box_set`](@ref): the other side of the rule, which writes a width.
  - [`set_ucs_variance_risk!`](@ref): the consumer that reads each bound as an absolute bound.

# References

  - $(ref_dict[:cajas2025]) Equation 11.15. $(ref_dict[:no_original_source])
"""
function sigma_delta_box_set(pr, dsigma::Number)
    d_sigma = dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma,
                             val = pr.sigma)
end
# A DeltaUncertaintySet with no prior of its own reads the prior result it is handed
# (see [`reads_prior_result`](@ref)).
function reads_prior_result(::DeltaUncertaintySet{Nothing})::Bool
    return true
end
"""
    ucs(ue::DeltaUncertaintySet, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    ucs(ue::DeltaUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the delta boxes of the mean axis and the covariance axis, from returns data or from a Prior Result.

The two methods are the two routes of a set that has a `pe` field, and the first route ends in the second. The returns-data method fits `ue.pe` once through [`ucs_prior`](@ref). It then gives the result to the prior-result method of the same set with `pe = nothing`. So both routes build the two boxes from one `pr` with the same two calls.

Only a set whose `pe` is `nothing` has the prior-result method. A set with a prior of its own fits its boxes on that prior alone. A set with no prior of its own fits its boxes on the `pr` that the caller gives. Inside an optimiser, that `pr` is the prior that the optimiser solves on.

The single-axis verbs [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) each fit the prior once on the returns-data route. Two single-axis calls cost two prior fits, and one `ucs` call costs one fit for the same two boxes. The three verbs give identical boxes on their common axes, because they call the same two builders on identical priors.

# Mathematical definition

The prior gives the mean ``\\hat{\\boldsymbol{\\mu}}`` and the covariance ``\\hat{\\mathbf{\\Sigma}}``. The bounds of the two boxes are:

```math
\\begin{align}
\\boldsymbol{\\mu}_{lb} &= \\boldsymbol{0}\\,, \\\\
\\boldsymbol{\\mu}_{ub} &= 2 \\delta_{\\mu} |\\hat{\\boldsymbol{\\mu}}|\\,.
\\end{align}
```

```math
\\begin{align}
\\mathbf{\\Sigma}_{lb} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,, \\\\
\\mathbf{\\Sigma}_{ub} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{lb}``, ``\\boldsymbol{\\mu}_{ub}``: Lower and upper bounds that the builder writes for the mean axis.
  - ``\\mathbf{\\Sigma}_{lb}``, ``\\mathbf{\\Sigma}_{ub}``: Lower and upper bounds of the covariance matrix.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - $(math_dict[:Sigma_hat])
  - ``\\delta_{\\mu}``: Fraction of the absolute mean that sets the half-width of the mean box.
  - ``\\delta_{\\sigma}``: Fraction of the absolute covariance that sets the half-width of the covariance box.
  - ``|\\cdot|``: Element-wise absolute value.

# Algorithm

The returns-data method does step 1 and forwards the call. The prior-result method does steps 2 to 6 on the `pr` that the caller gives.

 1. Fit the prior with [`ucs_prior`](@ref) on `ue.pe`, `X` and `F` to get `pr`. Forward `pr` to the prior-result method of the set with `pe = nothing`. If `ue.pe` is `nothing`, `ucs_prior` raises an `ArgumentError` that names the prior-result calls.
 2. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 3. Call [`mu_delta_box_set`](@ref) on the reduced Prior Result and `ue.dmu` to get the box of the mean axis. The box holds a width, so the model halves the difference of its bounds.
 4. Call [`sigma_delta_box_set`](@ref) on the reduced Prior Result and `ue.dsigma` to get the box of the covariance axis. The box holds absolute bounds, and both bounds bind.
 5. Write both boxes back onto the full universe with [`expand_investable_ucs`](@ref). A box fitted outside an optimiser then covers the same assets as the prior. A view of the box at the mask gives the reduced fit again.
 6. Return the two boxes as a tuple, with the mean axis first.

# Arguments

  - `ue`: Delta uncertainty set estimator. It gives the two fractions, and on the returns-data route it also gives the prior estimator.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `pr`: Prior Result that the set fits on. The method reads `pr.mu` and `pr.sigma`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the returns-data method gives to the prior estimator. The prior-result method ignores them.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Box of the mean axis, whose bounds hold a width.
  - `sigma_ucs::BoxUncertaintySet`: Box of the covariance axis, whose bounds are absolute.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_delta_box_set`](@ref)
  - [`sigma_delta_box_set`](@ref)
  - [`ucs_prior`](@ref)
  - [`mu_ucs`](@ref): the mean axis alone, with a prior fit of its own.
  - [`sigma_ucs`](@ref): the covariance axis alone, with a prior fit of its own.
"""
function ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
function ucs(ue::DeltaUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing,
             kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    mu_set, sigma_set = mu_delta_box_set(prr, ue.dmu), sigma_delta_box_set(prr, ue.dsigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::DeltaUncertaintySet, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    mu_ucs(ue::DeltaUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the delta box of the mean axis, from returns data or from a Prior Result.

The two methods are the two routes of a set that has a `pe` field, and the first route ends in the second, as [`ucs`](@ref) states. On the returns-data route, the method fits its own prior. It gives the same box as the first element of [`ucs`](@ref), but it costs one more prior fit. This path does not read `ue.dsigma`.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{lb} &= \\boldsymbol{0}\\,, \\\\
\\boldsymbol{\\mu}_{ub} &= 2 \\delta_{\\mu} |\\hat{\\boldsymbol{\\mu}}|\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{lb}``, ``\\boldsymbol{\\mu}_{ub}``: Lower and upper bounds that the builder writes for the mean axis.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - ``\\delta_{\\mu}``: Fraction of the absolute mean that sets the half-width of the mean box.
  - ``|\\cdot|``: Element-wise absolute value.

# Algorithm

 1. On the returns-data route, fit the prior with [`ucs_prior`](@ref) on `ue.pe`, `X` and `F` to get `pr`. Forward `pr` to the prior-result method of the set with `pe = nothing`.
 2. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 3. Call [`mu_delta_box_set`](@ref) on the reduced Prior Result and `ue.dmu` to get the box of the mean axis. This path reads only `pr.mu`.
 4. Write the box back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A box fitted outside an optimiser then covers the same assets as the prior. A view of the box at the mask gives the reduced fit again.

# Arguments

  - `ue`: Delta uncertainty set estimator. It gives the fraction of the mean axis, and on the returns-data route it also gives the prior estimator.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `pr`: Prior Result that the set fits on. The method reads only `pr.mu`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the returns-data method gives to the prior estimator. The prior-result method ignores them.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Box of the mean axis. Its bounds hold a width, not a pair of bounds on the mean.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_delta_box_set`](@ref): the builder that this method calls, which states the convention of the width.
  - [`ucs_prior`](@ref)
  - [`ucs`](@ref): both axes from one prior fit.
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return mu_ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
function mu_ucs(ue::DeltaUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing,
                kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    set = mu_delta_box_set(prr, ue.dmu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::DeltaUncertaintySet, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    sigma_ucs(ue::DeltaUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the delta box of the covariance axis, from returns data or from a Prior Result.

The two methods are the two routes of a set that has a `pe` field, and the first route ends in the second, as [`ucs`](@ref) states. On the returns-data route, the method fits its own prior. It gives the same box as the second element of [`ucs`](@ref), but it costs one more prior fit. This path does not read `ue.dmu`.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{lb} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,, \\\\
\\mathbf{\\Sigma}_{ub} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{lb}``, ``\\mathbf{\\Sigma}_{ub}``: Lower and upper bounds of the covariance matrix.
  - $(math_dict[:Sigma_hat])
  - ``\\delta_{\\sigma}``: Fraction of the absolute covariance that sets the half-width of the covariance box.
  - ``|\\cdot|``: Element-wise absolute value.

# Algorithm

 1. On the returns-data route, fit the prior with [`ucs_prior`](@ref) on `ue.pe`, `X` and `F` to get `pr`. Forward `pr` to the prior-result method of the set with `pe = nothing`.
 2. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 3. Call [`sigma_delta_box_set`](@ref) on the reduced Prior Result and `ue.dsigma` to get the box of the covariance axis. This path reads only `pr.sigma`.
 4. Write the box back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A box fitted outside an optimiser then covers the same assets as the prior. A view of the box at the mask gives the reduced fit again.

# Arguments

  - `ue`: Delta uncertainty set estimator. It gives the fraction of the covariance axis, and on the returns-data route it also gives the prior estimator.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `pr`: Prior Result that the set fits on. The method reads only `pr.sigma`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the returns-data method gives to the prior estimator. The prior-result method ignores them.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Box of the covariance axis. Its two bounds are absolute, and each bound binds alone.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`sigma_delta_box_set`](@ref): the builder that this method calls, which states the convention of absolute bounds and the note on positive semidefiniteness.
  - [`ucs_prior`](@ref)
  - [`ucs`](@ref): both axes from one prior fit.
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                   dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return sigma_ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
function sigma_ucs(ue::DeltaUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing,
                   kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    set = sigma_delta_box_set(prr, ue.dsigma)
    return expand_investable_ucs(set, imsk, pr)
end

export DeltaUncertaintySet
