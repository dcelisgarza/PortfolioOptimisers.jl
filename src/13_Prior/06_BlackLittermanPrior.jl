"""
$(DocStringExtensions.TYPEDEF)

Black-Litterman prior estimator for asset returns.

`BlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using the Black-Litterman model. It combines a prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and a blending parameter `tau`. The estimator supports both direct and constraint-based views, and allows for flexible confidence specification and matrix processing.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BlackLittermanPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(;
            me = EquilibriumExpectedReturns()
        ),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        rf::Number = 0.0,
        tau::Option{<:Number} = nothing
    ) -> BlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

The views are applied to the **assets**. Under ADR 0046 the wrapped prior is forwarded whole and only the deviations are spelled out: `mu` and `sigma` become the posterior, and `chol` is **dropped** because the posterior covariance supersedes the one it factorises. Everything else forwards — Black-Litterman leaves the observation axis untouched, so `w`, `ens`, `kld`, `ow` and `Z` all still describe the axis they were computed over, and `rr` and the factor block `fpr` are structural, over data the views do not modify.

!!! warning

    The returned `mu` and `sigma` are the Black-Litterman posterior, but `w` is the **wrapped prior's** observation weighting, forwarded unchanged. Black-Litterman produces no observation-level posterior, so there is no Black-Litterman-consistent alternative to forward — and dropping `w` would substitute the unweighted empirical distribution, which is further from the caller's intent than the weights they computed. A caller reading `pr.w`, `pr.ens`, `pr.kld` or `pr.ow` is therefore reading a property of the prior, not of the posterior.

!!! warning

    When the wrapped prior carries a factor block, `pr.fpr` describes the **prior** factor distribution while `pr.mu` is a **posterior** asset mean, so `pr.mu != pr.rr.M * pr.fpr.mu + pr.rr.b`. The block stays *structurally* true — the regression is over data Black-Litterman does not modify — while becoming *distributionally* inconsistent with the asset block. There is nothing better to report: the views land on the assets, so this estimator never computes a posterior factor distribution at all.

    Its siblings differ, and the difference is worth knowing. [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) apply their views to the factors and report the resulting posterior block, so both satisfy `mu == rr.M * fpr.mu + rr.b` exactly. [`AugmentedBlackLittermanPrior`](@ref) reports a posterior factor block too, but stays inconsistent for a different reason — see its own warning.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> BlackLittermanPrior(;
                           sets = UniverseSets(; xkey = \"nx\",
                                               dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                           views = LinearConstraintEstimator(;
                                                             val = [\"A == 0.03\", \"B + C == 0.04\"]))
BlackLittermanPrior
          pe ┼ EmpiricalPrior
             │        ce ┼ PortfolioOptimisersCovariance
             │           │   ce ┼ Covariance
             │           │      │    me ┼ SimpleExpectedReturns
             │           │      │       │   w ┴ nothing
             │           │      │    ce ┼ GeneralCovariance
             │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │       │    w ┴ nothing
             │           │      │   alg ┼ FullMoment()
             │           │      │     w ┴ nothing
             │           │   mp ┼ MatrixProcessing
             │           │      │     pdm ┼ Posdef
             │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │           │      │      dn ┼ nothing
             │           │      │      dt ┼ nothing
             │           │      │     alg ┼ nothing
             │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │        me ┼ EquilibriumExpectedReturns
             │           │   ce ┼ PortfolioOptimisersCovariance
             │           │      │   ce ┼ Covariance
             │           │      │      │    me ┼ SimpleExpectedReturns
             │           │      │      │       │   w ┴ nothing
             │           │      │      │    ce ┼ GeneralCovariance
             │           │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │      │       │    w ┴ nothing
             │           │      │      │   alg ┼ FullMoment()
             │           │      │      │     w ┴ nothing
             │           │      │   mp ┼ MatrixProcessing
             │           │      │      │     pdm ┼ Posdef
             │           │      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │           │      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │           │      │      │      dn ┼ nothing
             │           │      │      │      dt ┼ nothing
             │           │      │      │     alg ┼ nothing
             │           │      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │           │    w ┼ nothing
             │           │    l ┴ Int64: 1
             │   horizon ┴ nothing
          mp ┼ MatrixProcessing
             │     pdm ┼ Posdef
             │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      dn ┼ nothing
             │      dt ┼ nothing
             │     alg ┼ nothing
             │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
       views ┼ LinearConstraintEstimator
             │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
             │   key ┴ nothing
        sets ┼ UniverseSets
             │    xkey ┼ String: "nx"
             │   uxkey ┼ String: "ux"
             │    fkey ┼ String: "nf"
             │   ufkey ┼ String: "uf"
             │    zkey ┼ String: "nz"
             │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
  views_conf ┼ nothing
          rf ┼ Float64: 0.0
         tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`UniverseSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:black1992])
  - $(ref_dict[:cajas2025]) Section 5.1, Equations 5.13 to 5.15.
  - $(ref_dict[:walters2011])
  - $(ref_dict[:idzorek2007]) For the `views_conf` branch of [`calc_omega`](@ref).
"""
@propagatable @concrete struct BlackLittermanPrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:views])
    """
    views
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:views_conf])
    """
    views_conf
    """
    $(field_dict[:bl_rf])
    """
    rf
    """
    $(field_dict[:tau])
    """
    tau
    function BlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mp::AbstractMatrixProcessingEstimator, views::Lc_BLV,
                                 sets::Option{<:UniverseSets},
                                 views_conf::Option{<:Num_VecNum}, rf::Number,
                                 tau::Option{<:Number})
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(mp), typeof(views), typeof(sets), typeof(views_conf),
                   typeof(rf), typeof(tau)}(pe, mp, views, sets, views_conf, rf, tau)
    end
end
function BlackLittermanPrior(;
                             pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(;
                                                                                        me = EquilibriumExpectedReturns()),
                             mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                             views::Lc_BLV, sets::Option{<:UniverseSets} = nothing,
                             views_conf::Option{<:Num_VecNum} = nothing, rf::Number = 0.0,
                             tau::Option{<:Number} = nothing)::BlackLittermanPrior
    return BlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties BlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that the Black-Litterman prior's views, sets, view confidences, and blending parameter are valid.

This is the one guard every Black-Litterman constructor calls, so the four families refuse the same input. It does not look at the returns matrix, which the constructor never sees; [`prior`](@ref) checks the universe against `X` and [`bl_preroll`](@ref) checks the width of `P` against the covariance.

# Validation

  - When `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`, because the names of such a view resolve against a universe.
  - `views_conf` is checked by [`assert_bl_views_conf`](@ref), against the shape of `views`.
  - When `tau` is given, `tau > 0`. A `nothing` is admitted, and [`bl_preroll`](@ref) resolves it to `1/T`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`assert_bl_views_conf`](@ref)
  - [`bl_preroll`](@ref)
"""
function assert_bl(views::Lc_BLV, sets::Option{<:UniverseSets},
                   views_conf::Option{<:Num_VecNum}, tau::Option{<:Number})
    if isa(views, LinearConstraintEstimator)
        @argcheck(!isnothing(sets),
                  IsNothingError("sets cannot be nothing when views is a LinearConstraintEstimator"))
    end
    assert_bl_views_conf(views_conf, views)
    if !isnothing(tau)
        @argcheck(tau > zero(tau), DomainError(tau, "tau must be > 0"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pre-compute shared Black-Litterman inputs from views, prior covariance, and blending parameters.

Extracts the view matrix `P`, view returns vector `Q`, and excluded indices from `views` and `sets` via [`black_litterman_views`](@ref), resolves `tau`, filters excluded rows from `views_conf` via [`remove_excl_views`](@ref), and computes the scaled uncertainty matrix `omega = tau * Ω` via [`calc_omega`](@ref).

`axis` names the declared axis of the distribution the views land on, and every caller knows it from its own type rather than from the views: [`BlackLittermanPrior`](@ref) takes the default `:xkey` (the asset axis), while a member whose views update the **factor** distribution passes `:fkey`. It is the last argument because it is the only one an asset-space caller never supplies.

The selector is a *field of* [`UniverseSets`](@ref) rather than a key resolved from one, so a caller states its axis and nothing else. Resolving the key is this function's work, and it happens only when there is a `sets` to read it from — reading `sets.fkey` to describe a universe that does not exist is the same error as reading the universe itself. Views supplied as a [`BlackLittermanViews`](@ref) result are the one shape that arrives with no `sets` at all: they resolve no names and ignore both the sets and the axis.

This is also where `P` meets the distribution it updates, so it is where their widths are reconciled. A `P` assembled from names is the right width by construction; a **precomputed** [`BlackLittermanViews`](@ref) resolves no names and is checked nowhere else.

The returned `omega` already carries ``\\tau``, so a caller passes it to [`vanilla_posteriors`](@ref) as it stands. That scaling has a consequence worth knowing: [`calc_omega`](@ref) is homogeneous of degree one in the covariance it reads, so ``\\tau`` multiplies both ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal`` and ``\\mathbf{\\Omega}``, and cancels out of the posterior **mean** on every confidence branch. It does not cancel out of the posterior covariance. [`vanilla_posteriors`](@ref) states the measurement.

# Algorithm

 1. Check that `axis` names a declared axis a view can land on.
 2. Resolve `axis` to a universe key. When `sets` is `nothing` the key is `nothing` too, because a precomputed views object resolves no name and needs none.
 3. Assemble the views with [`black_litterman_views`](@ref) under that key, giving `blv`.
 4. Check that at least one view survived, so that `blv` is not `nothing`.
 5. Read `P`, `Q` and `excl` off `blv`.
 6. Check that `P` is as wide as `prior_sigma` is tall.
 7. Resolve `tau`, which is `pe_tau` when the estimator carries one and `1/T` otherwise.
 8. Drop the confidences of the views that step 3 excluded, with [`remove_excl_views`](@ref).
 9. Build the view uncertainty matrix from the surviving confidences with [`calc_omega`](@ref), scale it by `tau`, and return it as `omega` alongside `P`, `Q` and `tau`.

# Validation

  - `axis in (:xkey, :fkey)`.
  - At least one view resolves, so [`black_litterman_views`](@ref) does not answer `nothing`.
  - `size(P, 2) == size(prior_sigma, 1)`.

# Arguments

  - $(arg_dict[:views])
  - $(arg_dict[:sets])
  - $(arg_dict[:views_conf])
  - `prior_sigma::MatNum`: Prior covariance matrix of the distribution the views update, `n × n` over that axis.
  - `pe_tau::Option{<:Number}`: Optional user-specified blending parameter. If `nothing`, defaults to `1/T`.
  - `T::Integer`: Number of observations used to compute the default `tau = 1/T`.
  - $(arg_dict[:datatype])
  - $(arg_dict[:strict])
  - $(arg_dict[:bl_axis])

# Returns

  - `(; P, Q, tau, omega)`: Named tuple where:

      + `P::MatNum`: View matrix `views × assets`.
      + `Q::VecNum`: View returns vector `views × 1`.
      + `tau::Number`: Resolved blending parameter.
      + `omega::LinearAlgebra.Diagonal`: Scaled view uncertainty matrix `tau * Ω`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`black_litterman_views`](@ref)
  - [`calc_omega`](@ref)
  - [`remove_excl_views`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function bl_preroll(views, sets, views_conf, prior_sigma, pe_tau, T, datatype, strict,
                    axis::Symbol = :xkey)
    @argcheck(axis in (:xkey, :fkey),
              DomainError(axis,
                          "axis must name a declared axis a view can land on, :xkey or :fkey"))
    # The caller states the axis; resolving it to a key is this function's work. That is what
    # lets a caller which admits `sets === nothing` — precomputed views resolve no names —
    # say which distribution its views update without guarding the sets it may not have.
    key = isnothing(sets) ? nothing : getproperty(sets, axis)
    blv = black_litterman_views(views, sets, key; datatype = datatype, strict = strict)
    # Under `strict = false` an unresolvable name only warns, so a set of views none of which
    # resolves leaves `get_black_litterman_views` with no row to return and it answers
    # `nothing`. Destructuring that gave `FieldError: type Nothing has no field P`, which names
    # neither the views nor the universe they failed against.
    @argcheck(!isnothing(blv),
              IsNothingError("no view resolved against the universe under $(repr(key)), so there is no view matrix to update the distribution with. Pass strict = true to raise on the first name that does not resolve"))
    (; P, Q, excl) = blv
    # A `P` assembled from names is the right width by construction — the universe it resolved
    # against is the one the caller already checked. A **precomputed** `P` resolves no names, so
    # this is the only thing that sees its width at all, and without it a wrong one surfaces as a
    # bare `DimensionMismatch` from the multiplication below.
    @argcheck(size(P, 2) == size(prior_sigma, 1),
              DimensionMismatch("the view matrix and the distribution the views update disagree on how many variables there are. Got\nsize(P, 2) => $(size(P, 2))\nsize(prior_sigma, 1) => $(size(prior_sigma, 1))"))
    tau = isnothing(pe_tau) ? inv(T) : pe_tau
    views_conf = remove_excl_views(views_conf, excl)
    omega = tau * calc_omega(views_conf, P, prior_sigma)
    return (; P, Q, tau, omega)
end
"""
    calc_omega(::Nothing, P::MatNum, sigma::MatNum) -> LinearAlgebra.Diagonal
    calc_omega(views_conf::Number, P::MatNum, sigma::MatNum) -> LinearAlgebra.Diagonal
    calc_omega(views_conf::VecNum, P::MatNum, sigma::MatNum) -> LinearAlgebra.Diagonal

Compute the Black-Litterman view uncertainty matrix `Ω`.

Each method selects one shape of `views_conf` and computes the branch of the closed form below that the shape names: `::Nothing` the unscaled diagonal, `::Number` the same diagonal under one shared scale, and `::VecNum` the same diagonal under one scale per view.

# Mathematical definition

Let ``\\mathbf{P}`` be the ``K \\times N`` view matrix and ``\\mathbf{\\Sigma}`` the ``N \\times N`` prior covariance matrix. The view uncertainty matrix ``\\mathbf{\\Omega}`` for each `views_conf` variant is:

```math
\\begin{align}
\\mathbf{\\Omega} &= \\mathrm{Diag}(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^\\intercal) \\quad (\\text{no confidence})\\,.
\\end{align}
```

```math
\\begin{align}
\\mathbf{\\Omega} &= \\left(\\frac{1}{v} - 1\\right) \\mathrm{Diag}(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^\\intercal) \\quad (\\text{scalar confidence } v)\\,.
\\end{align}
```

```math
\\begin{align}
\\mathbf{\\Omega} &= \\mathrm{Diag}\\!\\left(\\left(\\frac{1}{\\boldsymbol{v}} - \\boldsymbol{1}\\right) \\odot \\mathrm{diag}(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^\\intercal)\\right) \\quad (\\text{vector confidence } \\boldsymbol{v})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Omega}``: ``K \\times K`` diagonal view uncertainty matrix.
  - ``\\mathbf{P}``: ``K \\times N`` views matrix.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior covariance matrix.
  - ``v``: Scalar view confidence level.
  - ``\\boldsymbol{v}``: ``K \\times 1`` vector of view confidence levels.
  - ``\\odot``: Element-wise multiplication.

The no-confidence branch is the diagonal uncertainty of the view creation model. [`bl_preroll`](@ref) scales the result by ``\\tau``, so the pair returns ``\\mathrm{Diag}(\\mathbf{P}(\\tau\\mathbf{\\Sigma})\\mathbf{P}^\\intercal)``.

A confidence ``v`` rescales that diagonal by ``1/v - 1``, which is Idzorek's method in Walters' closed form. A high confidence therefore shrinks the view uncertainty and a low one widens it. The scale is negative for every ``v`` outside ``(0, 1)``, which is why [`assert_bl_views_conf`](@ref) refuses such a value.

The scalar branch and the vector branch agree where they overlap: a scalar ``v`` gives the same ``\\mathbf{\\Omega}`` as the constant vector of ``v``, to the last bit. So the two shapes are two ways of writing one input, and [`assert_bl_views_conf`](@ref) counts only the vector against the views.

Both endpoints are refused too, and the bound is strict on purpose. ``v = 1`` gives ``\\mathbf{\\Omega} = \\mathbf{0}``, a view held with no uncertainty at all, which makes ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}`` singular whenever ``\\mathbf{P}`` is rank-deficient. Two identical views over a three-asset sample give a rank-one ``\\mathbf{P}``; at ``v = 1`` the sum is the constant matrix `1.0021e-6`, whose determinant is `0.0` and whose rank is 1, and the solve raises `SingularException`. Just inside the bound it is merely ill-conditioned: at ``v = 1 - 10^{-8}`` the condition number is `2.0e8`, and at ``v = 0.99`` it is `199`. ``v = 0`` gives an infinite uncertainty, which is the same thing as omitting the view.

# Arguments

  - `views_conf`:

      + `::Nothing`: No confidence specified; `Ω = Diag(P * sigma * P')`.
      + `::Number`: Scalar confidence `v`; `Ω = (1/v - 1) * Diag(P * sigma * P')`.
      + `::VecNum`: Per-view confidences `v`; `Ω = Diag((1 ./ v .- 1) .* diag(P * sigma * P'))`.

  - $(arg_dict[:P])

  - $(arg_dict[:sigma])

# Returns

  - `omega::LinearAlgebra.Diagonal`: Diagonal view uncertainty matrix `views × views`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_preroll`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function calc_omega(::Nothing, P::MatNum, sigma::MatNum)
    return LinearAlgebra.Diagonal(P * sigma * transpose(P))
end
function calc_omega(views_conf::Number, P::MatNum, sigma::MatNum)
    alphas = inv(views_conf) - one(eltype(views_conf))
    return LinearAlgebra.Diagonal(alphas .* P * sigma * transpose(P))
end
function calc_omega(views_conf::VecNum, P::MatNum, sigma::MatNum)
    alphas = inv.(views_conf) .- one(eltype(views_conf))
    return LinearAlgebra.Diagonal(alphas .* P * sigma * transpose(P))
end
"""
    vanilla_posteriors(tau::Number, prior_mu::VecNum, prior_sigma::MatNum,
                       omega::MatNum, P::MatNum, Q::VecNum)

Compute the Black-Litterman posterior mean and covariance for asset returns.

`vanilla_posteriors` implements the standard Black-Litterman update equations, combining the prior mean and covariance with user or algorithmic views. The function returns the posterior mean and covariance matrix, incorporating the blending parameter `tau`, view uncertainty matrix `omega`, view matrix `P`, and view returns vector `Q`.

The kernel carries no risk-free rate. Each Black-Litterman prior estimator adds its own `rf` once, to the posterior asset expected returns, through [`apply_rf`](@ref).

The two equations below are the **inverse-free** form of the master equations. They are algebraically the same object as the form stated on [`prior`](@ref): the covariance term is the Woodbury expansion of ``\\left[(\\tau\\mathbf{\\Sigma})^{-1} + \\mathbf{P}^\\intercal\\mathbf{\\Omega}^{-1}\\mathbf{P}\\right]^{-1}``, and the two agree to `2.2e-19` on the mean and `1.4e-20` on the covariance for a ``200 \\times 6`` sample with three views. This form is used because it inverts one ``K \\times K`` matrix rather than three ``N \\times N`` ones.

# Mathematical definition

Let ``\\boldsymbol{\\Pi}`` be the prior mean, ``\\mathbf{\\Sigma}`` the prior covariance, ``\\tau`` the scaling parameter, ``\\mathbf{P}`` the view matrix, ``\\boldsymbol{q}`` the view vector, and ``\\mathbf{\\Omega}`` the view uncertainty matrix:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{BL} &= \\boldsymbol{\\Pi} + \\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal \\left(\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}\\right)^{-1} (\\boldsymbol{q} - \\mathbf{P}\\boldsymbol{\\Pi})\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BL} &= \\mathbf{\\Sigma} + \\tau\\mathbf{\\Sigma} - \\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal \\left(\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}\\right)^{-1} \\mathbf{P}\\tau\\mathbf{\\Sigma}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{BL}``: Black-Litterman posterior mean vector.
  - ``\\hat{\\mathbf{\\Sigma}}_{BL}``: Black-Litterman posterior covariance matrix.
  - ``\\boldsymbol{\\Pi}``: ``N \\times 1`` prior (equilibrium) expected returns.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior covariance matrix.
  - ``\\tau``: Scaling parameter for the uncertainty in the prior.
  - ``\\mathbf{P}``: ``K \\times N`` views matrix.
  - ``\\boldsymbol{q}``: ``K \\times 1`` views vector.
  - ``\\mathbf{\\Omega}``: ``K \\times K`` view uncertainty matrix.

``\\tau`` stands in both equations, but it does **not** move the posterior mean when ``\\mathbf{\\Omega}`` comes from the [`bl_preroll`](@ref) pair. [`calc_omega`](@ref) is homogeneous of degree one in the covariance it reads, and `bl_preroll` scales its answer by ``\\tau``, so the gain ``\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal(\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega})^{-1}`` has one ``\\tau`` above and one below, and they cancel. Over ``\\tau \\in \\{1/200, 0.05, 0.5\\}`` on a ``200 \\times 3`` sample with two views the posterior mean moves by at most `3.3e-19` on every confidence branch — no confidence, the scalar `0.4`, and the vector `[0.25, 0.75]`. The posterior **covariance** does move, because ``\\mathbf{\\Sigma} + \\tau\\mathbf{\\Sigma} - \\ldots`` carries a bare ``\\tau``: its excess over ``\\mathbf{\\Sigma}`` has trace `2.013e-7`, `2.013e-6` and `2.013e-5` at ``\\tau = 0.001``, `0.01` and `0.1`, which is linear in ``\\tau`` to three figures.

A view that repeats the prior is a null update. With ``\\boldsymbol{q} = \\mathbf{P}\\boldsymbol{\\Pi}`` the residual ``\\boldsymbol{q} - \\mathbf{P}\\boldsymbol{\\Pi}`` is zero, so the posterior mean equals the prior mean exactly — measured at `0.0` on the same sample.

# Algorithm

 1. Scale the prior covariance by `tau` and carry it through the views, giving `v1`, the ``N \\times K`` matrix ``\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal``.
 2. Close `v1` under the views and add the view uncertainty, giving `v2`, the ``K \\times K`` matrix ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}``. This is the only matrix the body inverts.
 3. Take the view residual `v3`, which is ``\\boldsymbol{q}`` less the prior's own answer to the views.
 4. Solve `v2` against `v3`, carry the solution through `v1`, and add it to `prior_mu`, giving `posterior_mu`.
 5. Solve `v2` against the transpose of `v1`, carry that through `v1`, and subtract it from `prior_sigma + tau * prior_sigma`, giving `posterior_sigma`.

# Arguments

  - `tau`: Scalar blending parameter for prior and views.
  - `prior_mu`: Prior mean vector of asset returns.
  - `prior_sigma`: Prior covariance matrix of asset returns.
  - `omega`: View uncertainty matrix.
  - `P`: View matrix (views × assets).
  - `Q`: Vector of view returns (views).

# Returns

  - `posterior_mu::VecNum`: Posterior mean vector of asset returns.
  - `posterior_sigma::Matrix{<:Number}`: Posterior covariance matrix of asset returns.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`calc_omega`](@ref)
  - [`apply_rf`](@ref)
"""
function vanilla_posteriors(tau::Number, prior_mu::VecNum, prior_sigma::MatNum,
                            omega::MatNum, P::MatNum, Q::VecNum)
    v1 = tau * prior_sigma * transpose(P)
    v2 = P * v1 + omega
    v3 = Q - P * prior_mu
    posterior_mu = prior_mu + v1 * (v2 \ v3)
    posterior_sigma = prior_sigma + tau * prior_sigma - v1 * (v2 \ transpose(v1))
    return posterior_mu, posterior_sigma
end
"""
    apply_rf(rf::Number, mu::VecNum)

Shift a Black-Litterman posterior asset mean by the risk-free rate.

`apply_rf` is the **single site that adds** the `rf` field of a Black-Litterman prior estimator. The four families -- [`BlackLittermanPrior`](@ref), [`BayesianBlackLittermanPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) -- each call it once, on the asset expected returns they return, and nowhere else. [`remove_rf`](@ref) is its inverse and the only site that takes the rate off.

Three properties follow, and all three are contracts of the family:

  - **The rate is added once.** No body adds it twice.
  - **The update runs on excess returns.** A prior mean that arrives as a total return loses the rate first, through [`remove_rf`](@ref), so the rate `apply_rf` puts back is the one that came off. A mean that is an excess return already -- the equilibrium mean of [`equilibrium_mu`](@ref) -- is used as it stands.
  - **A prior is isolated.** The rate applies to the posterior this estimator computes. A wrapped prior estimator is never re-fitted, and the scale conversion around the update is a round trip, so a risk-free rate one of them applied internally stays where it is.

The rate is a property of the asset axis, so the factor block of a result never carries it. Where [`remove_rf`](@ref) took the rate off a factor mean, nothing puts it back: the factor posterior is reported on the excess scale the update ran on, which is `pe.rf` below the scale the factor prior was supplied on. `FactorBlackLittermanPrior` is therefore not a plain round trip -- it takes the rate off the factor axis and adds it on the asset axis, and the two moves reach the assets differently. Writing `s` for the row sums of the loadings, its answer moves by `rf * (1 - s)`, and cancels only where an asset's loadings sum to one.

"Once" is measured, not asserted. Two [`BlackLittermanPrior`](@ref) fits over one ``200 \\times 3`` sample, differing only in `rf`, give posterior means whose difference is `rf` in every entry to the last bit: at `rf = 0.03` the difference is `[0.03, 0.03, 0.03]` and `max|diff - rf|` is `0.0`. The pair with [`remove_rf`](@ref) round-trips a mean to `1.6e-18` in both orders.

# Algorithm

 1. Add `rf` to every entry of `mu`, and return the result. The input is not modified.

# Arguments

  - `rf`: Risk-free rate.
  - `mu`: Posterior asset expected returns vector.

# Returns

  - `mu::VecNum`: `mu` shifted by `rf`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`BayesianBlackLittermanPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function apply_rf(rf::Number, mu::VecNum)
    return mu .+ rf
end
"""
    remove_rf(rf::Number, mu::VecNum)

Convert a total-return mean to an excess-return mean.

`remove_rf` is the **single site that subtracts** the `rf` field of a Black-Litterman prior estimator. It is the inverse of [`apply_rf`](@ref), and the two act as a pair: a mean goes onto the excess scale before the Black-Litterman update, and comes off it after.

The update is written in excess returns. The view returns in `Q` are excess returns, and the equilibrium mean [`equilibrium_mu`](@ref) computes is an excess return by construction, because `l * sigma * w` is the risk premium that reverse optimisation implies. A mean taken from a wrapped prior estimator is a total return, so it must lose the rate before it can blend against either of them.

Only [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) call it, and only where `l` is `nothing`. Those two are the members that can build the equilibrium mean themselves, so they are the two whose prior mean must reach [`vanilla_posteriors`](@ref) on one known scale either way. [`BlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) have no equilibrium branch, and take the wrapped mean as it stands.

# Algorithm

 1. Subtract `rf` from every entry of `mu`, and return the result. The input is not modified.

# Arguments

  - `rf`: Risk-free rate.
  - `mu`: Prior asset or factor expected returns vector, as a total return.

# Returns

  - `mu::VecNum`: `mu` less `rf`.

# Related

  - [`apply_rf`](@ref)
  - [`equilibrium_mu`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function remove_rf(rf::Number, mu::VecNum)
    return mu .- rf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Remove excluded views from `views_conf`.

This is the method for a confidence that is `nothing` or a scalar. Neither is indexed by view — a scalar is one confidence for every view — so dropping a view changes nothing, and every argument after the first is ignored.

# Algorithm

 1. Return `views_conf` unchanged.

# Arguments

  - `views_conf`: `nothing`, or one confidence shared by every view.
  - `args...`: The excluded indices, ignored.

# Returns

  - `views_conf::Option{<:Number}`: The input, unchanged.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_preroll`](@ref)
  - [`get_black_litterman_views`](@ref)
"""
function remove_excl_views(views_conf::Option{<:Number}, args...)
    return views_conf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Remove excluded views from `views_conf`.

This is the method for a per-view confidence vector when no view was excluded. [`get_black_litterman_views`](@ref) passes `nothing` rather than an empty vector when every view resolved, so this method carries the common case.

# Algorithm

 1. Return `views_conf` unchanged.

# Arguments

  - `views_conf`: One confidence per view.
  - `::Nothing`: No view was excluded.

# Returns

  - `views_conf::VecNum`: The input, unchanged.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_preroll`](@ref)
  - [`get_black_litterman_views`](@ref)
"""
function remove_excl_views(views_conf::VecNum, ::Nothing)
    return views_conf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Remove excluded views from `views_conf`.

This is the method that does the work: a per-view confidence vector, and the indices of the views that [`get_black_litterman_views`](@ref) dropped. The surviving entries keep the order the caller wrote them in, so entry `k` of the answer still belongs to row `k` of the `P` the same call assembled. Excluding every view leaves an empty vector.

# Algorithm

 1. Take the indices of `views_conf` that are not members of `excl`, in ascending order.
 2. Return the corresponding entries as a lazy view, with [`nothing_scalar_array_view`](@ref).

# Arguments

  - `views_conf`: One confidence per view, over the views the caller wrote.
  - `excl`: The indices of the views that resolved no name.

# Returns

  - `views_conf::VecNum`: A view of the input, holding one confidence per surviving view.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_preroll`](@ref)
  - [`get_black_litterman_views`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function remove_excl_views(views_conf::VecNum, excl::VecInt)
    return nothing_scalar_array_view(views_conf, setdiff(1:length(views_conf), excl))
end
"""
    prior(pe::BlackLittermanPrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute the Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the Black-Litterman model, combining a prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and blending parameter `tau`. The method supports both direct and constraint-based views, flexible confidence specification, and matrix processing.

When `pe.tau` is `nothing` the blending parameter is `1/T`, where `T` is the number of observations of the oriented `X`. `pe.rf` reaches the answer once, on the posterior asset expected returns; [`apply_rf`](@ref) owns that contract.

# Mathematical definition

The Black-Litterman posterior distribution combines the prior ``(\\boldsymbol{\\Pi}, \\tau \\mathbf{\\Sigma})`` with investor views ``(\\mathbf{P}, \\boldsymbol{q}, \\mathbf{\\Omega})``. [`vanilla_posteriors`](@ref) computes the algebraically equivalent inverse-free form:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{BL} &= \\left[(\\tau\\mathbf{\\Sigma})^{-1} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\mathbf{P}\\right]^{-1} \\left[(\\tau\\mathbf{\\Sigma})^{-1} \\boldsymbol{\\Pi} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\boldsymbol{q}\\right]\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BL} &= \\mathbf{\\Sigma} + \\left[(\\tau\\mathbf{\\Sigma})^{-1} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\mathbf{P}\\right]^{-1}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\Pi}``: `N × 1` prior (equilibrium) expected returns.
  - ``\\mathbf{\\Sigma}``: `N × N` prior covariance matrix.
  - ``\\tau``: Scaling parameter for the uncertainty in the prior.
  - ``\\mathbf{P}``: `K × N` views matrix (each row is one view).
  - ``\\boldsymbol{q}``: `K × 1` views vector.
  - ``\\mathbf{\\Omega}``: `K × K` views uncertainty matrix.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. When `pe.views` resolves names, check that the asset universe is as long as `X` is wide. A precomputed [`BlackLittermanViews`](@ref) resolves no name, so it is not checked here; step 4 checks its width instead.
 3. Fit the wrapped prior `pe.pe` on `(X, F)`, giving `prior_model`, and read `posterior_X`, `prior_mu` and `prior_sigma` off it.
 4. Assemble the views and their uncertainty with [`bl_preroll`](@ref), over `prior_sigma` and `size(X, 1)` observations, giving `P`, `Q`, `tau` and `omega`. The axis is left at its default, `:xkey`, because these views land on the assets.
 5. Run the master equations with [`vanilla_posteriors`](@ref), giving `posterior_mu` and `posterior_sigma`.
 6. Add `pe.rf` to `posterior_mu` with [`apply_rf`](@ref). This is the one site that adds it.
 7. Process `posterior_sigma` in place with [`matrix_processing!`](@ref), under `pe.mp` and `posterior_X`.
 8. Forward the whole of `prior_model` with [`forward_prior`](@ref), replacing `mu` and `sigma` by the posterior pair and dropping `chol`.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)`.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, and posterior covariance matrix.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`bl_preroll`](@ref) Assembles `P`, `Q`, `tau` and `omega`, and resolves `pe.tau` to `1/T` when the estimator carries none.
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`apply_rf`](@ref)
  - [`forward_prior`](@ref)
"""
function prior(pe::BlackLittermanPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    # The axis is checked only by the views that resolve names against it. A `BlackLittermanViews`
    # result carries its own `P` and never touches `sets`, so demanding a universe for it would
    # reject the legitimate precomputed-views configuration, which `assert_bl` deliberately permits
    # to supply no `sets` at all.
    if isa(pe.views, LinearConstraintEstimator)
        @argcheck(length(pe.sets.dict[pe.sets.xkey]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.xkey]) ($(length(pe.sets.dict[pe.sets.xkey]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    prior_model = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_mu, prior_sigma = prior_model.X, prior_model.mu, prior_model.sigma
    (; P, Q, tau, omega) = bl_preroll(pe.views, pe.sets, pe.views_conf, prior_sigma, pe.tau,
                                      size(X, 1), eltype(posterior_X), strict)
    posterior_mu, posterior_sigma = vanilla_posteriors(tau, prior_mu, prior_sigma, omega, P,
                                                       Q)
    # `pe.rf` is applied here and only here (see [`apply_rf`](@ref)): once, on the asset
    # expected returns this estimator returns. `prior_model.mu` is the wrapped prior's own
    # answer and is used as it stands, so a rate that prior applied internally is left alone.
    posterior_mu = apply_rf(pe.rf, posterior_mu)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    # Everything the wrapped prior carried is forwarded (see [`forward_prior`](@ref)); `chol`
    # is the only drop, because `posterior_sigma` supersedes the covariance it factorises.
    # Black-Litterman leaves the observation axis untouched (`posterior_X === prior_model.X`),
    # so the wrapped `w` still describes exactly the rows of the returned `X`, its `ens`/`kld`/
    # `ow` still describe that `w`, and `Z` is still indexed by the axis it was derived from —
    # which is also why nesting order does not matter: `BlackLittermanPrior(; pe =
    # FeaturePrior(…))` reaches `distance` with the same feature matrix as the other order.
    # `rr` is structural — the regression of `X` on `F`, over data Black-Litterman does not
    # modify — and the factor block `fpr` travels with it.
    return forward_prior(prior_model; mu = posterior_mu, sigma = posterior_sigma,
                         chol = nothing)
end

function factor_residual_config(pe::BlackLittermanPrior)
    return factor_residual_config(pe.pe)
end

export BlackLittermanPrior
