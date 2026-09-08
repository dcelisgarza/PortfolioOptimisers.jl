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
             │     xkey ┼ String: "nx"
             │    uxkey ┼ String: "ux"
             │    tfkey ┼ String: "nf"
             │   utfkey ┼ String: "uf"
             │    cfkey ┼ String: "ncf"
             │   ucfkey ┼ String: "ucf"
             │    nikey ┼ String: "ni"
             │     dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
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

`axis` names the declared axis of the distribution the views land on, and every caller knows it from its own type rather than from the views: [`BlackLittermanPrior`](@ref) takes the default `:xkey` (the asset axis), while a member whose views update the **factor** distribution passes `:tfkey` or `:cfkey`. It is the last argument because it is the only one an asset-space caller never supplies.

[`UniverseSets`](@ref) declares two factor axes, so the accepted set is three symbols wide. Every member in the library today updates a factor distribution whose columns are the columns of `F`, so every one of them passes `:tfkey`; `:cfkey` is accepted because the axis exists and a view can land on it, not because a member reaches it yet.

The selector is a *field of* [`UniverseSets`](@ref) rather than a key resolved from one, so a caller states its axis and nothing else. Resolving the key is this function's work, and it happens only when there is a `sets` to read it from — reading `sets.tfkey` to describe a universe that does not exist is the same error as reading the universe itself. Views supplied as a [`BlackLittermanViews`](@ref) result are the one shape that arrives with no `sets` at all: they resolve no names and ignore both the sets and the axis.

This is also where `P` meets the distribution it updates, so it is where their widths are reconciled. A `P` assembled from names is the right width by construction; a **precomputed** [`BlackLittermanViews`](@ref) resolves no names and is checked nowhere else.

**No view left answers `nothing`, whatever emptied the set.** A row a builder cannot assemble is dropped whole rather than fitted without its term, so a view set can end with no row for several reasons: a departure took the last one, the universe this fit was handed is a *sub-universe* — a cluster of a nested optimisation, a subset of a resampling — in which the names sit outside, or the caller mistyped. None of them changes what is left to condition on, so none of them changes the answer: the fit carries on as though no view had been stated, and each estimator spells what that means for it through [`bl_posteriors`](@ref) or [`bl_view_block`](@ref).

The unresolved name is still reported where it is met, by [`strict_diagnostic`](@ref) — an `ArgumentError` under `strict = true`, a warning otherwise, naming the name and the universe it failed against. That report names the cause; a refusal here could only name the symptom, and could not tell a typo from a name that is legitimately outside a sub-universe. The ledger still separates the two *for the announcement*: only a departed name writes into it, and [`announce_bl_departures`](@ref) reads it to say which of the two happened.

The returned `omega` already carries ``\\tau``, so a caller passes it to [`vanilla_posteriors`](@ref) as it stands. That scaling has a consequence worth knowing: [`calc_omega`](@ref) is homogeneous of degree one in the covariance it reads, so ``\\tau`` multiplies both ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal`` and ``\\mathbf{\\Omega}``, and cancels out of the posterior **mean** on every confidence branch. It does not cancel out of the posterior covariance. [`vanilla_posteriors`](@ref) states the measurement.

# Algorithm

 1. Check that `axis` names a declared axis a view can land on.
 2. Resolve `axis` to a universe key. When `sets` is `nothing` the key is `nothing` too, because a precomputed views object resolves no name and needs none.
 3. Assemble the views with [`black_litterman_views`](@ref) under that key, giving `blv`.
 4. When no view survived, answer `nothing`. The caller decides what a view-free fit is.
 5. Read `P`, `Q` and `excl` off `blv`.
 6. Check that `P` is as wide as `prior_sigma` is tall.
 7. Resolve `tau`, which is `pe_tau` when the estimator carries one and `1/T` otherwise.
 8. Drop the confidences of the views that step 3 excluded, with [`remove_excl_views`](@ref).
 9. Build the view uncertainty matrix from the surviving confidences with [`calc_omega`](@ref), scale it by `tau`, and return it as `omega` alongside `P`, `Q` and `tau`.

# Validation

  - `axis in (:xkey, :tfkey, :cfkey)`.
  - `size(P, 2) == size(prior_sigma, 1)`, when a view survived.

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
  - `ledger`: The door's ledger of departure casualties, or `nothing` when nobody is collecting. It is threaded into [`black_litterman_views`](@ref), and it is what tells a view set emptied by a departure from one emptied any other way — a distinction the *announcement* makes, not the answer.

# Returns

  - `(; P, Q, tau, omega)`: Named tuple where:

      + `P::MatNum`: View matrix `views × assets`.
      + `Q::VecNum`: View returns vector `views × 1`.
      + `tau::Number`: Resolved blending parameter.
      + `omega::LinearAlgebra.Diagonal`: Scaled view uncertainty matrix `tau * Ω`.

  - `nothing`: Every view was dropped, so the fit proceeds view-free.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`black_litterman_views`](@ref)
  - [`calc_omega`](@ref)
  - [`remove_excl_views`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`announce_bl_departures`](@ref)
  - [`investable_views`](@ref)
"""
function bl_preroll(views, sets, views_conf, prior_sigma, pe_tau, T, datatype, strict,
                    axis::Symbol = :xkey; ledger::Option{<:AbstractVector} = nothing)
    @argcheck(axis in (:xkey, :tfkey, :cfkey),
              DomainError(axis,
                          "axis must name a declared axis a view can land on, :xkey, :tfkey or :cfkey"))
    # The caller states the axis; resolving it to a key is this function's work. That is what
    # lets a caller which admits `sets === nothing` — precomputed views resolve no names —
    # say which distribution its views update without guarding the sets it may not have.
    key = isnothing(sets) ? nothing : getproperty(sets, axis)
    blv = black_litterman_views(views, sets, key; datatype = datatype, strict = strict,
                                ledger = ledger)
    if isnothing(blv)
        # A row that cannot be assembled is dropped, and a view set with every row dropped
        # is a view set with nothing in it. What emptied it does not change what is left to
        # condition on, so it does not change the answer either: the fit carries on as
        # though the caller had stated no view. ADR 0125 states the rule, and the callers
        # spell what "no view" means for each of them — the wrapped prior wherever there is
        # one to hand back, and the empty-block algebra where there is not.
        #
        # A name that resolved against nothing is still reported where it is met, through
        # `strict_diagnostic`: an `ArgumentError` under `strict = true` and a warning
        # otherwise, naming the name and the universe it failed against. That is the report
        # a caller can act on, and it is strictly more than a refusal here could say. The
        # refusal this replaces could not tell a typo from a name that is simply outside the
        # universe *this* fit was handed — a cluster of a nested optimisation, a subset of a
        # resampling — and it killed the second along with the first.
        return nothing
    end
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
    announce_bl_departures(ni::VecStr, ledger::VecStr, viewless::Bool) -> Nothing

Report, once per Black-Litterman fit, who left the investable universe and what their leaving cost the view set.

This is the family's call of [`announce_non_investable`](@ref), written once so the four `prior` methods each spend one line on it and none of them can word it differently. It names the process a Black-Litterman fit, because the message is otherwise the optimisation door's and would tell a standalone `prior(pe, X)` call that it is inside an optimisation it is not.

The sentence is one sentence for all four members, and it is exactly true of each. Two of them write their views on the **factor** axis, where no asset name ever appears, so the view clause simply does not bite for them; what does bite for every member is the first clause, because all four estimate their posterior over the assets that remain.

`viewless` says the fit ended with no view at all, and it is reported two ways because it happens two ways. When the **ledger** is non-empty a departure took the last surviving view, which is the case ADR 0125 singles out: the message is raised to a warning and says so in the departure's own words. When the ledger is empty nobody departed, so [`announce_non_investable`](@ref) is silent — it has no name to report — and the view-free fit gets a warning of its own instead. That is the sub-universe case: a cluster of a nested optimisation or a subset of a resampling holds none of the names the caller wrote, every row is dropped whole, and the caller is handed an unconditioned answer with nothing else to tell them. Every drop that only *trims* the view set stays `@info`.

# Arguments

  - `ni`: The names the Investable Mask left out, from [`investable_views`](@ref).
  - `ledger`: What the departures cost, as the view builder recorded it. Non-empty is what makes a view-free fit a *departure's* doing.
  - `viewless`: Whether the fit ended up with no view at all.

# Returns

  - `nothing`.

# Related

  - [`announce_non_investable`](@ref)
  - [`investable_views`](@ref)
  - [`record_non_investable_drop!`](@ref)
  - [`bl_preroll`](@ref)
  - [`BlackLittermanPrior`](@ref)
"""
function announce_bl_departures(ni::VecStr, ledger::VecStr, viewless::Bool)::Nothing
    # A departure emptied the set only if a departure wrote into the ledger. Every other
    # view-free fit — a sub-universe the names sit outside, a typo, a row that cancelled —
    # left the ledger empty, and has news of its own that the departure sentence would
    # misreport.
    departure_emptied = viewless && !isempty(ledger)
    consequence = if departure_emptied
        "Every view stated named one of them, so the fit proceeds with no view at all and its posterior is its wrapped prior."
    else
        "The posterior is estimated over the assets that remain, an asset-side view row naming one of them is dropped whole, and a group sheds them before its coefficient is spread."
    end
    announce_non_investable(ni, ledger, "Black-Litterman fit", consequence;
                            warn = departure_emptied)
    if viewless && !departure_emptied
        # Nothing above is guaranteed to have said anything: `announce_non_investable` is
        # silent when nobody departed, which is exactly the sub-universe case. Handing back
        # an unconditioned answer where the caller asked for views changes the answer rather
        # than trimming it, and this is the caller's only way to learn it.
        @warn("No Black-Litterman view resolved against the universe this fit was handed, so it proceeds with no view at all and its posterior is the unconditioned one. A view row is dropped whole when a name in it does not resolve, so a universe narrowed to a sub-problem — a cluster, a resampled subset — can empty a view set that is correct over the universe the caller stated it against.")
    end
    return nothing
end
"""
    bl_posteriors(::Nothing, prior_mu::VecNum, prior_sigma::MatNum) -> Tuple
    bl_posteriors(blp::NamedTuple, prior_mu::VecNum, prior_sigma::MatNum) -> Tuple

Run the master equations, or hand back the prior pair when nothing is left to run them with.

[`bl_preroll`](@ref) answers `nothing` whenever no view row survived — a departure took the last one, the universe this fit was handed holds none of the names, or the caller mistyped. ADR 0125 says the fit proceeds rather than refusing, and a Black-Litterman posterior with no view **is** the distribution it was going to update — so that is what this returns.

It is the prior pair itself and not the empty-view algebra, and the difference is not rounding. [`vanilla_posteriors`](@ref) adds the estimation-error term ``[(\\tau\\mathbf{\\Sigma})^{-1}]^{-1} = \\tau\\mathbf{\\Sigma}`` to the covariance, so an empty ``\\mathbf{P}`` would answer ``(1 + \\tau)\\mathbf{\\Sigma}`` — a *wider* covariance than the prior, produced by views that are not there. Forwarding the prior pair is what makes the missing view cost the caller the view and nothing else.

[`BayesianBlackLittermanPrior`](@ref) is the member this verb does **not** serve, and the reason is worth stating: its update is a precision sum, ``\\hat{\\mathbf{\\Sigma}}^{-1} = \\mathbf{\\Sigma}_f^{-1} + \\mathbf{P}^\\intercal\\mathbf{\\Omega}^{-1}\\mathbf{P}``, so an empty ``\\mathbf{P}`` contributes exactly zero and carries no ``\\tau\\mathbf{\\Sigma}`` term to inflate anything. Its no-view answer is therefore the empty-block algebra of [`bl_view_block`](@ref) — which is *not* its wrapped prior: it collapses by Woodbury to ``\\mathbf{\\Sigma}_a + \\mathbf{M}\\mathbf{\\Sigma}_f\\mathbf{M}^\\intercal`` and ``\\mathbf{M}\\boldsymbol{\\mu}_f + \\mathbf{b}``, the moments the factor model implies. That member has no unadjusted prior to hand back, because it transforms its wrapped prior whether or not a view is stated.

The covariance is copied, because every caller of this passes what it gets to [`matrix_processing!`](@ref), which writes in place, and the prior result must not be mutated under a caller still holding it.

The split is dispatch: [`bl_preroll`](@ref) answers a concrete `NamedTuple` or a literal `nothing` at each call site, so the method pair is resolved statically.

# Arguments

  - `blp`: What [`bl_preroll`](@ref) answered, or `nothing`.
  - `prior_mu`: The prior mean of the distribution the views update.
  - `prior_sigma`: The prior covariance of that distribution.

# Returns

  - `(posterior_mu, posterior_sigma)::Tuple{VecNum, MatNum}`: The posterior pair, or the prior pair when there is no view.

# Related

  - [`bl_preroll`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`announce_bl_departures`](@ref)
"""
function bl_posteriors(::Nothing, prior_mu::VecNum, prior_sigma::MatNum)
    return prior_mu, copy(prior_sigma)
end
function bl_posteriors(blp::NamedTuple, prior_mu::VecNum, prior_sigma::MatNum)
    return vanilla_posteriors(blp.tau, prior_mu, prior_sigma, blp.omega, blp.P, blp.Q)
end
"""
    bl_view_block(::Nothing, n::Integer, datatype::DataType) -> Tuple
    bl_view_block(blp::NamedTuple, ::Integer, ::DataType) -> Tuple

Read the `P`, `Q` and `omega` of a view system, as a block with no row when it has no view left.

Two members need this rather than [`bl_posteriors`](@ref), and for two different reasons.

[`AugmentedBlackLittermanPrior`](@ref) stacks an asset-side view block above a factor-side one, and **either half can empty on its own**. An empty half contributes no row rather than collapsing the stack to the prior, because the *joint* posterior is still conditioned by whatever the other half kept: handing back the prior stack there would throw away views the caller stated and that still resolve. Only when both halves empty is there nothing left, and that member reads the pair itself to take the prior stack. The block is `0 × n` rather than a zero row, so the stack carries no phantom view and `aug_Q` is one entry shorter rather than one entry of zero.

[`BayesianBlackLittermanPrior`](@ref) has one view system and no stack, and takes an empty block because for it the empty-block algebra **is** the no-view answer. Its update is a precision sum, ``\\hat{\\mathbf{\\Sigma}}^{-1} = \\mathbf{\\Sigma}_f^{-1} + \\mathbf{P}^\\intercal\\mathbf{\\Omega}^{-1}\\mathbf{P}``, so a `0 × n` ``\\mathbf{P}`` adds exactly zero and there is no ``\\tau\\mathbf{\\Sigma}`` estimation-error term for it to inflate — which is precisely what stops [`bl_posteriors`](@ref) from serving the members that run [`vanilla_posteriors`](@ref). What comes out is not that member's wrapped prior but ``\\mathbf{\\Sigma}_a + \\mathbf{M}\\mathbf{\\Sigma}_f\\mathbf{M}^\\intercal``, the moments its factor model implies, and it has no unadjusted prior to offer instead.

# Arguments

  - `blp`: What [`bl_preroll`](@ref) answered for this system, or `nothing`.
  - `n`: The width of this system's axis, for the empty `P`.
  - `datatype`: The numeric type of the block.

# Returns

  - `(P, Q, omega)::Tuple`: The view matrix, view returns and uncertainty, with no row when the system was emptied.

# Related

  - [`bl_preroll`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`BayesianBlackLittermanPrior`](@ref)
  - [`bl_posteriors`](@ref)
"""
function bl_view_block(::Nothing, n::Integer, datatype::DataType)
    return Matrix{datatype}(undef, 0, n), Vector{datatype}(undef, 0),
           LinearAlgebra.Diagonal(Vector{datatype}(undef, 0))
end
function bl_view_block(blp::NamedTuple, ::Integer, ::DataType)
    return blp.P, blp.Q, blp.omega
end
"""
    assert_bl_precomputed_universe(sets::UniverseSets, pr::AbstractPriorResult) -> Nothing
    assert_bl_precomputed_universe(sets::Nothing, pr::AbstractPriorResult) -> Nothing

Refuse the one configuration in which an asset-side Black-Litterman view cannot be reduced: a **precomputed** view matrix over a universe an asset has left.

The reduction works by resolving the caller's view *names* against the universe that survives, which is what lets a row naming a departed asset be dropped whole and every other row be rebuilt over the investable columns. A [`BlackLittermanViews`](@ref) passed in ready-made resolves no name — [`black_litterman_views`](@ref) hands it straight back, and its docstring says why: it was assembled against whatever universe the caller held, and nothing downstream can re-check it. So its `P` cannot be reduced, and there is nothing to reduce it *by*: an estimator whose views are precomputed is permitted to carry no `sets` at all.

Left alone, this is the defect the whole ticket is about, unfixed for one configuration and **silent**: `calc_omega` forms `P * sigma * transpose(P)` over the full universe, `0 * NaN` is `NaN`, and the posterior comes back all `NaN`. Under the default `mp` that surfaces one layer later as `ArgumentError: matrix contains Infs or NaNs` from `posdef!`, which names the wrong cause; under a matrix processing estimator that does nothing, the fit *succeeds* and hands the caller an empty universe. This is the map's rule applied where it has to be: handle it, or refuse by name.

The split is dispatch on the type of `sets`, which is a field of a `@concrete` estimator and so a type fact. The `UniverseSets` method is the whole of the ordinary path and costs a dispatch; only the sets-less path derives the mask, and only to refuse.

# Arguments

  - $(arg_dict[:sets])
  - $(arg_dict[:pr])

# Validation

  - When `sets` is `nothing`, every asset of `pr` must be investable.

# Returns

  - `nothing`.

# Related

  - [`investable_views`](@ref)
  - [`investable_mask`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`black_litterman_views`](@ref)
  - [`BlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
"""
function assert_bl_precomputed_universe(::UniverseSets, ::AbstractPriorResult)::Nothing
    return nothing
end
function assert_bl_precomputed_universe(::Nothing, pr::AbstractPriorResult)::Nothing
    @argcheck(isnothing(investable_mask(pr)),
              ArgumentError("the wrapped prior left at least one asset out of the investable universe, and this estimator carries no `sets`, so its views are a precomputed `$(nameof(BlackLittermanViews))`. Such a `P` was assembled against the universe the caller held and names nothing, so there is no way to tell which of its rows the departed asset belonged to, and no way to reduce it. Building the update over the full universe instead would return an all-NaN posterior, because `0 * NaN` is `NaN` and a zero coefficient does not protect a row.\nState the views as a `$(nameof(LinearConstraintEstimator))` over a `$(nameof(UniverseSets))`, which resolves names and drops only the rows a departure actually took, or fit over a universe in which every asset is investable."))
    return nothing
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

The kernel carries no risk-free rate. Each Black-Litterman prior estimator adds its own `rf` once, through [`apply_rf`](@ref), which owns the site each member adds it at.

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

Shift a Black-Litterman mean by the risk-free rate.

`apply_rf` is the **single site that reads** the `rf` field of a Black-Litterman prior estimator. The four families -- [`BlackLittermanPrior`](@ref), [`BayesianBlackLittermanPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) -- each call it once, and nowhere else. Nothing subtracts the rate.

Three properties follow, and all three are contracts of the family:

  - **The rate is added once.** No body adds it twice.
  - **The update runs on the scale the views are written on.** A Black-Litterman update blends the prior mean against the view returns in `Q` by forming the residual `Q - P * mu`, so the prior mean must be on the scale of `Q`, which is a total return. A mean taken from a wrapped prior estimator is one already and reaches the update untouched. The equilibrium mean of [`equilibrium_mu`](@ref) is a bare risk premium, so the two members that can build one add the rate to it **before** the update, on the axis that mean lives on. A level that is missing from the prior mean is a level the views are blended against wrongly, which is why the rate goes on first rather than last.
  - **A prior is isolated.** A wrapped prior estimator is never re-fitted and its mean is never rescaled, so a risk-free rate one of them applied internally stays where it is.

Where each member calls it, and what the field therefore does:

  - [`BlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) have no equilibrium branch and so have nothing to convert. They add the rate to the posterior asset mean, last, and the field is a plain shift of the answer.
  - [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) call it on the equilibrium mean, and only where `l` is set. Where `l` is `nothing` neither reads the field at all, so it does not reach the answer.

"Once" is measured, not asserted. Two [`BlackLittermanPrior`](@ref) fits over one ``200 \\times 3`` sample, differing only in `rf`, give posterior means whose difference is `rf` in every entry to the last bit: at `rf = 0.03` the difference is `[0.03, 0.03, 0.03]` and `max|diff - rf|` is `0.0`.

# Algorithm

 1. Add `rf` to every entry of `mu`, and return the result. The input is not modified.

# Arguments

  - `rf`: Risk-free rate.
  - `mu`: Expected returns vector. It is a posterior asset mean for the two members with no equilibrium branch, and a prior equilibrium mean for the two with one.

# Returns

  - `mu::VecNum`: `mu` shifted by `rf`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`BayesianBlackLittermanPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function apply_rf(rf::Number, mu::VecNum)
    return mu .+ rf
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
    prior(pe::BlackLittermanPrior, X::MatNum, F::Option{<:MatNum} = nothing,
          pnl::Option{<:AssetPanel} = nothing;
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
 3. Fit the wrapped prior `pe.pe` on `(X, F)`, giving `prior_model`.
 4. Derive the Investable Mask and the reduced view universe with [`investable_views`](@ref), and refuse a precomputed view matrix over a gapped universe with [`assert_bl_precomputed_universe`](@ref).
 5. View the fitted prior at the mask with [`investable_prior`](@ref), and read `posterior_X`, `prior_mu` and `prior_sigma` off *that*.
 6. Assemble the views and their uncertainty with [`bl_preroll`](@ref), over the reduced `prior_sigma` and `size(X, 1)` observations, giving `P`, `Q`, `tau` and `omega`, or `nothing` when a departure took the last view. The axis is left at its default, `:xkey`, because these views land on the assets.
 7. Run the master equations with [`bl_posteriors`](@ref), giving `posterior_mu` and `posterior_sigma` over the investable assets.
 8. Add `pe.rf` to `posterior_mu` with [`apply_rf`](@ref). This is the one site that adds it.
 9. Process `posterior_sigma` in place with [`matrix_processing!`](@ref), under `pe.mp` and `posterior_X`, while it is still the reduced block a factorisation exists for.
10. Announce the departures once with [`announce_bl_departures`](@ref).
11. Write both posteriors back onto the full asset universe with [`expand_moment`](@ref), so a non-investable asset carries `NaN` in `mu` and on the diagonal of `sigma`.
12. Forward the whole of `prior_model` with [`forward_prior`](@ref), replacing `mu` and `sigma` by the expanded pair and dropping `chol`.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - $(arg_dict[:pnl_prior])
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
function prior(pe::BlackLittermanPrior, X::MatNum, F::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    X, F = dims_oriented(dims, X, F)
    # The axis is checked only by the views that resolve names against it. A `BlackLittermanViews`
    # result carries its own `P` and never touches `sets`, so demanding a universe for it would
    # reject the legitimate precomputed-views configuration, which `assert_bl` deliberately permits
    # to supply no `sets` at all.
    if isa(pe.views, LinearConstraintEstimator)
        @argcheck(length(pe.sets.dict[pe.sets.xkey]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.xkey]) ($(length(pe.sets.dict[pe.sets.xkey]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    prior_model = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    # The reduction, once, at this estimator's entry. A view is a dense linear form over the
    # asset axis and `0 * NaN` is `NaN`, so a single departed asset poisons `omega` and with
    # it every entry of both posteriors — including under a view naming only live assets.
    # See [`investable_views`](@ref); the whole of the fix is building on the investable
    # columns, and the expansion below is what puts the answer back on the caller's universe.
    imsk, vsets, ni = investable_views(prior_model, pe.sets)
    assert_bl_precomputed_universe(pe.sets, prior_model)
    vpr = investable_prior(imsk, prior_model)
    posterior_X, prior_mu, prior_sigma = vpr.X, vpr.mu, vpr.sigma
    ledger = String[]
    blp = bl_preroll(pe.views, vsets, pe.views_conf, prior_sigma, pe.tau, size(X, 1),
                     eltype(posterior_X), strict; ledger = ledger)
    # `nothing` is the view set a departure emptied, and the pair is then the prior's own.
    posterior_mu, posterior_sigma = bl_posteriors(blp, prior_mu, prior_sigma)
    # `pe.rf` is applied here and only here (see [`apply_rf`](@ref)): once, on the asset
    # expected returns this estimator returns. `prior_model.mu` is the wrapped prior's own
    # answer and is used as it stands, so a rate that prior applied internally is left alone.
    posterior_mu = apply_rf(pe.rf, posterior_mu)
    # Processed on the reduced covariance, before the expansion: `posdef!` and the denoise
    # and detone steps all read every entry, and a `NaN` frame has no factorisation.
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    announce_bl_departures(ni, ledger, isnothing(blp))
    # The expansion. The contract is that a prior result lives on the FULL asset universe
    # with a `NaN` in `mu` and on the diagonal of `sigma` for an asset that is not
    # investable, so that the next layer derives the same mask this one did. `expand_moment`
    # is the [`coverage_reduction`](@ref) family's own idiom for it, and its `nothing`
    # methods are the all-investable path — no branch here, and no allocation there.
    posterior_mu = expand_moment(posterior_mu, imsk, 1)
    posterior_sigma = expand_moment(posterior_sigma, imsk)
    # Everything the wrapped prior carried is forwarded (see [`forward_prior`](@ref)); `chol`
    # is the only drop, because `posterior_sigma` supersedes the covariance it factorises.
    # `prior_model` is forwarded, not `vpr`: the reduction is this estimator's own working
    # universe and nothing outside it may see a narrowed carrier. Black-Litterman leaves the
    # observation axis untouched — the reduction takes columns, never rows — so the wrapped
    # `w` still describes exactly the rows of the returned `X`, and its `ens`/`kld`/`ow`
    # still describe that `w`.
    # `rr` is structural — the regression of `X` on `F`, over data Black-Litterman does not
    # modify — and the factor block `fpr` travels with it.
    return forward_prior(prior_model; mu = posterior_mu, sigma = posterior_sigma,
                         chol = nothing)
end

function factor_residual_config(pe::BlackLittermanPrior)
    return factor_residual_config(pe.pe)
end

export BlackLittermanPrior
