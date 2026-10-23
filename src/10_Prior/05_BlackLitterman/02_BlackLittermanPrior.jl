"""
$(DocStringExtensions.TYPEDEF)

Black-Litterman prior estimator for asset returns.

`BlackLittermanPrior` is a low order prior estimator. It fits the wrapped prior `pe`, and it updates the asset mean and covariance of that prior with views on the assets. The views are view equations in a [`LinearConstraintEstimator`](@ref), or a precomputed [`BlackLittermanViews`](@ref). `views_conf` sets the confidence of the views, `tau` scales the uncertainty of the prior mean, and `rf` shifts the posterior mean. `mp` processes the posterior covariance.

[black1991](@citet) state the posterior mean as the precision-weighted blend of the prior and the views. The library computes the equivalent update form of [walters2011](@citet). [black1991](@citet) give no posterior covariance. The library returns the prior covariance plus the posterior covariance of the mean, which is the convention of [walters2011](@citet). [helitterman1999](@citet) set the view uncertainty to ``\\tau\\,\\mathrm{Diag}(\\mathbf{P}\\mathbf{\\Sigma}\\mathbf{P}^\\intercal)``. The library does the same when `views_conf` is `nothing`. [cajas2025](@citet) sets `tau` to `1/T`, which is the library default. A confidence `v` in `views_conf` scales that uncertainty by `1/v - 1`. This is the closed form of [walters2011](@citet) for the method of [idzorek2007](@citet), which finds the uncertainty of each view by a numerical search for the weight tilt that the confidence asks for. [idzorek2007](@citet) also gives the meaning of the confidence: the fraction of the full tilt towards the view.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BlackLittermanPrior(;
        pe::Onl{<:AbstractLowOrderPriorEstimator_A_F_AF} = EmpiricalPrior(;
            me = EquilibriumExpectedReturns()
        ),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        rf::Number = 0.0,
        tau::Option{<:Number} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> BlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

The views update the **assets**. The estimator forwards the full result of the wrapped prior, with three changes. `mu` and `sigma` become the posterior. The estimator **drops** `chol`, because the posterior covariance replaces the covariance that `chol` factorises. Black-Litterman does not change the observation axis, so `w`, `ens`, `kld` and `ow` still describe the axis that the wrapped prior fitted them over. `rr` and the factor block `fpr` are structural: they come from data that the views do not change.

!!! warning

    `mu` and `sigma` of the result are the Black-Litterman posterior, but `w` holds the observation weights of the **wrapped prior**, with no change. Black-Litterman makes no posterior over the observations, so no other weights agree with the update. Without `w`, a reader uses the unweighted empirical distribution, which is further from the intent of the caller than the weights that the caller computed. Thus `pr.w`, `pr.ens`, `pr.kld` and `pr.ow` describe the prior, not the posterior.

!!! warning

    When the wrapped prior has a factor block, `pr.fpr` describes the **prior** factor distribution, but `pr.mu` is a **posterior** asset mean, so `pr.mu != pr.rr.M * pr.fpr.mu + pr.rr.b`. The block is still true as a *structure*, because the regression uses data that Black-Litterman does not change. But as a *distribution* it does not agree with the asset block. The estimator has no better block to report: the views update the assets, so the estimator does not compute a posterior factor distribution.

    The other members are different. [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) apply their views to the factors and report the posterior factor block, so `mu == rr.M * fpr.mu + rr.b` holds exactly for both. [`AugmentedBlackLittermanPrior`](@ref) also reports a posterior factor block, but its blocks do not agree, for a different reason that its own warning states.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, [`assert_bl_views_conf`](@ref) checks it.
  - If `tau` is not `nothing`, `tau > 0`.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pe`: Recursively viewed through [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> BlackLittermanPrior(;
                           sets = UniverseSets(; xkey = \"nx\",
                                               dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                           views = LinearConstraintEstimator(;
                                                             val = [\"A == 0.03\", \"B + C == 0.04\"]))
BlackLittermanPrior
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
             │           me ┼ EquilibriumExpectedReturns
             │              │   ce ┼ PortfolioOptimisersCovariance
             │              │      │   ce ┼ Covariance
             │              │      │      │    me ┼ SimpleExpectedReturns
             │              │      │      │       │   w ┴ nothing
             │              │      │      │    ce ┼ GeneralCovariance
             │              │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │              │      │      │       │    w ┴ nothing
             │              │      │      │   alg ┼ FullMoment()
             │              │      │      │     w ┴ nothing
             │              │      │   mp ┼ MatrixProcessing
             │              │      │      │     pdm ┼ Posdef
             │              │      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │              │      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │              │      │      │      dn ┼ nothing
             │              │      │      │      dt ┼ nothing
             │              │      │      │     alg ┼ nothing
             │              │      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │              │    w ┼ nothing
             │              │    l ┴ Int64: 1
             │      horizon ┼ nothing
             │   fill_limit ┼ nothing
             │        x_src ┴ nothing
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
         tau ┼ nothing
       x_src ┴ nothing
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

  - $(ref_dict[:black1991]) Appendix, items 7 and 8. $(ref_dict[:no_original_source])
  - $(ref_dict[:helitterman1999]) Footnote 6.
  - $(ref_dict[:walters2011]) Equations 25 to 27, 41 and 44. $(ref_dict[:no_original_source])
  - $(ref_dict[:idzorek2007]) Section 3.2.
  - $(ref_dict[:cajas2025]) Section 5.1.3, Equation 5.6, and Section 5.1.2, footnote 1.
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
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function BlackLittermanPrior(pe::Onl{<:AbstractLowOrderPriorEstimator_A_F_AF},
                                 mp::AbstractMatrixProcessingEstimator, views::Lc_BLV,
                                 sets::Option{<:UniverseSets},
                                 views_conf::Option{<:Num_VecNum}, rf::Number,
                                 tau::Option{<:Number},
                                 x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :BlackLittermanPrior)
        assert_inner_reader(pe, :pe, :BlackLittermanPrior)
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(mp), typeof(views), typeof(sets), typeof(views_conf),
                   typeof(rf), typeof(tau), typeof(x_src)}(pe, mp, views, sets, views_conf,
                                                           rf, tau, x_src)
    end
end
function BlackLittermanPrior(;
                             pe::Onl{<:AbstractLowOrderPriorEstimator_A_F_AF} = EmpiricalPrior(;
                                                                                               me = EquilibriumExpectedReturns()),
                             mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                             views::Lc_BLV, sets::Option{<:UniverseSets} = nothing,
                             views_conf::Option{<:Num_VecNum} = nothing, rf::Number = 0.0,
                             tau::Option{<:Number} = nothing,
                             x_src::Option{<:AbstractReturnsSource} = nothing)::BlackLittermanPrior
    return BlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau, x_src)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties BlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check the views, the sets, the view confidences and `tau` of a Black-Litterman prior estimator.

Each Black-Litterman constructor calls this guard, so the four members refuse the same input. The guard does not read the returns matrix, because the constructor does not have it. [`assert_bl_views_axis`](@ref) checks the universe against `X` in [`prior`](@ref), and [`bl_preroll`](@ref) checks the width of `P` against the covariance.

# Validation

  - When `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`, because the names of such a view resolve against a universe. Raises an [`IsNothingError`](@ref).
  - [`assert_bl_views_conf`](@ref) checks `views_conf` against the shape of `views`.
  - When `tau` is not `nothing`, `tau > 0`. Raises a `DomainError`. The guard accepts `nothing`, and [`bl_preroll`](@ref) changes it to `1/T`.

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

Assemble the views, `tau` and the view uncertainty that the Black-Litterman members share.

`axis` names the declared axis of the distribution that the views update. Each caller knows it from its own type, not from the views. [`BlackLittermanPrior`](@ref) uses the default `:xkey`, the asset axis. A member whose views update the **factor** distribution passes `:tfkey` or `:cfkey`. `axis` is the last positional argument, because a caller on the asset axis never gives it.

[`UniverseSets`](@ref) declares two factor axes, so the function accepts three symbols. Each member of the library that updates a factor distribution reads the columns of `F`, and passes `:tfkey`. The function accepts `:cfkey` because the axis exists and a view can update it. No member passes `:cfkey` now.

`axis` is the name of a *field of* [`UniverseSets`](@ref), not a key, so a caller states its axis and nothing more. This function reads the key, and only when `sets` is not `nothing`. To read `sets.tfkey` when there is no universe is the same error as to read the universe. A precomputed [`BlackLittermanViews`](@ref) is the one shape of views that can come with no `sets`. It resolves no names, and it ignores the sets and the axis.

This function is also where `P` meets the distribution that it updates, so it checks that their widths agree. A `P` assembled from names has the correct width. A **precomputed** [`BlackLittermanViews`](@ref) resolves no names, and no other site checks its width.

**When no view is left, the function returns `nothing`, for each cause.** [`get_black_litterman_views`](@ref) drops a row that it cannot assemble, and it does not fit the row without the term. Thus a set of views can lose all its rows for different causes:

  - A departure removed the last row.
  - The fit uses a *sub-universe* that does not hold the names, such as a cluster of a nested optimisation or a subset of a resampling.
  - The caller wrote a typo.

No cause changes the data that is left for the update, so no cause changes the result. The fit continues as if the caller stated no view. Each member states what that means for it through [`bl_posteriors`](@ref) or [`bl_view_block`](@ref).

[`strict_diagnostic`](@ref) still reports each name that resolves against neither axis, at the site where the assembly reads it. It raises an `ArgumentError` under `strict = true` and warns otherwise, and the message names the name and the universe that it failed against. That report names the cause. A refusal here can only name the result, and it cannot tell a typo from a correct name that is outside a sub-universe. The ledger keeps the two cases apart *for the report*: only a departed name writes into it, and [`announce_bl_departures`](@ref) reads it to say which case happened.

The returned `omega` includes ``\\tau``, so a caller passes it to [`vanilla_posteriors`](@ref) with no change. [`calc_omega`](@ref) is homogeneous of degree one in the covariance that it reads. Thus ``\\tau`` multiplies ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal`` and ``\\mathbf{\\Omega}``, and it cancels from the posterior **mean** on each confidence branch. It does not cancel from the posterior covariance. [`vanilla_posteriors`](@ref) gives the measurement.

# Algorithm

 1. Check that `axis` names a declared axis that a view can update.
 2. Read the universe key from the field `axis` of `sets`. When `sets` is `nothing`, the key is `nothing`, because a precomputed views object resolves no name.
 3. Assemble the views with [`black_litterman_views`](@ref) under that key. The result is `blv`.
 4. When no view is left, return `nothing`. The caller decides what a fit with no view gives.
 5. Read `P`, `Q` and `excl` from `blv`.
 6. Check that the number of columns of `P` is equal to the number of rows of `prior_sigma`.
 7. Set `tau` to `pe_tau` when it is not `nothing`, and to `1/T` otherwise.
 8. Remove the confidences of the views that step 3 dropped, with [`remove_excl_views`](@ref).
 9. Make the view uncertainty matrix from the remaining confidences with [`calc_omega`](@ref), and multiply it by `tau`. Return it as `omega`, with `P`, `Q` and `tau`.

# Validation

  - `axis in (:xkey, :tfkey, :cfkey)`. Raises a `DomainError`.
  - `size(P, 2) == size(prior_sigma, 1)`, when a view is left. Raises a `DimensionMismatch`.

# Arguments

  - $(arg_dict[:views])
  - $(arg_dict[:sets])
  - $(arg_dict[:views_conf])
  - `prior_sigma::MatNum`: The prior covariance matrix of the distribution that the views update, `n × n` over that axis.
  - `pe_tau::Option{<:Number}`: The `tau` of the estimator, or `nothing`. If `nothing`, the function uses `1/T`.
  - `T::Integer`: The number of observations, for the default `tau = 1/T`.
  - $(arg_dict[:datatype])
  - $(arg_dict[:strict])
  - $(arg_dict[:bl_axis])
  - `ledger`: The Departure Ledger of the caller that derives the Investable Mask, or `nothing` when no caller collects the drops. The function passes it to [`black_litterman_views`](@ref). The ledger tells a set of views that a departure emptied from a set that another cause emptied. Only the *report* uses this difference, and the result does not.

# Returns

  - `(; P, Q, tau, omega)`: A named tuple of:

      + `P::MatNum`: The view matrix, `views × n`.
      + `Q::VecNum`: The view returns, `views × 1`.
      + `tau::Number`: The value of `tau` that the function used.
      + `omega::LinearAlgebra.Diagonal`: The view uncertainty matrix, multiplied by `tau`.

  - `nothing`: The function dropped every view, so the fit continues with no view.

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

Report, once for each Black-Litterman fit, the assets that left the investable universe and the views that they removed.

This is the call of [`announce_non_investable`](@ref) for the Black-Litterman members. It is written once, so each of the four `prior` methods calls it in one line, and the four use the same words. It names the process "Black-Litterman fit". The default message names an optimisation, which is wrong for a call of `prior(pe, X)` outside an optimisation.

The message is one sentence for the four members, and it is true for each. Two members write their views on the **factor** axis, where no asset name occurs, so the clause about view rows does not apply to them. The first clause applies to each member, because the four estimate their posterior over the assets that remain.

`viewless` is `true` when the fit has no view left. The function reports this in two ways, because it has two causes.

  - When the **ledger** is not empty, a departure removed the last view. The function raises the message to a warning, and the message states the departure.
  - When the ledger is empty, no departure removed a view, and the departure message cannot explain the empty set. [`announce_non_investable`](@ref) says nothing about the views, so the function gives its own warning for the fit with no view. This is the case of a sub-universe: a cluster of a nested optimisation or a subset of a resampling holds none of the names that the caller wrote. The fit drops each row, and the caller receives a result with no update and no other report.

A drop that removes only some of the views stays at the `@info` level.

# Arguments

  - `ni`: The names that the Investable Mask leaves out, from [`investable_views`](@ref).
  - `ledger`: The views that the departures removed, as the view assembly recorded them. When it is not empty and `viewless` is `true`, a *departure* removed the last view.
  - `viewless`: `true` when the fit has no view left.

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
        @warn("No Black-Litterman view resolved against the universe this fit was handed, so it proceeds with no view at all and its posterior is the unconditioned one. A view row is dropped whole when a name in it does not resolve, so a universe narrowed to a sub-problem — a cluster, a resampled subset — can empty a view set that is correct over the universe the caller stated it against.",
              maxlog = log_limit(:warn), _id = @log_id())
    end
    return nothing
end
"""
    bl_posteriors(::Nothing, prior_mu::VecNum, prior_sigma::MatNum) -> Tuple
    bl_posteriors(blp::NamedTuple, prior_mu::VecNum, prior_sigma::MatNum) -> Tuple

Run the master equations, or return the prior pair when no view is left.

[`bl_preroll`](@ref) returns `nothing` when no view row is left. The fit continues and does not refuse. A Black-Litterman posterior with no view **is** the distribution that the views were to update, so this function returns that distribution.

The function returns the prior pair, not the result of the master equations with an empty ``\\mathbf{P}``. The difference is not a rounding error. [`vanilla_posteriors`](@ref) adds the term of estimation error ``[(\\tau\\mathbf{\\Sigma})^{-1}]^{-1} = \\tau\\mathbf{\\Sigma}`` to the covariance. With an empty ``\\mathbf{P}`` it gives ``(1 + \\tau)\\mathbf{\\Sigma}``, a covariance *wider* than the prior, from views that do not exist. With the prior pair, the caller loses the views and nothing more.

[`BayesianBlackLittermanPrior`](@ref) does **not** use this function. Its update is a sum of precisions, ``\\hat{\\mathbf{\\Sigma}}^{-1} = \\mathbf{\\Sigma}_f^{-1} + \\mathbf{P}^\\intercal\\mathbf{\\Omega}^{-1}\\mathbf{P}``, so an empty ``\\mathbf{P}`` adds zero, and there is no ``\\tau\\mathbf{\\Sigma}`` term to widen the covariance. Its result with no view is thus the algebra of an empty block from [`bl_view_block`](@ref). That result is *not* its wrapped prior. By Woodbury, it is ``\\mathbf{\\Sigma}_a + \\mathbf{M}\\mathbf{\\Sigma}_f\\mathbf{M}^\\intercal`` and ``\\mathbf{M}\\boldsymbol{\\mu}_f + \\mathbf{b}``, the moments that the factor model gives. That member has no unchanged prior to return, because it changes its wrapped prior with or without views.

The function copies the covariance. Each caller passes the result to [`matrix_processing!`](@ref), which writes in place, and the function must not change a prior result that a caller still holds.

The two methods split by dispatch. The result of [`bl_preroll`](@ref) is `nothing` or one concrete `NamedTuple`, so the compiler splits that small union at the call.

# Arguments

  - `blp`: The result of [`bl_preroll`](@ref), or `nothing`.
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

Return the `P`, `Q` and `omega` of a view system, or a block with no row when the system has no view left.

Two members use this function and not [`bl_posteriors`](@ref), for two different reasons.

[`AugmentedBlackLittermanPrior`](@ref) puts a block of asset views above a block of factor views, and **each half can become empty alone**. An empty half adds no row, and the stack does not change to the prior. The other half still holds views that the caller stated and that resolve, so the *joint* posterior still uses them. The prior stack discards those views. Only when both halves are empty is no view left, and that member then reads the two results itself and takes the prior stack. The empty block is `0 × n` and not a row of zeros, so the stack holds no false view, and `aug_Q` has one entry less, not an entry of zero.

[`BayesianBlackLittermanPrior`](@ref) has one view system and no stack. It uses an empty block because, for it, the algebra of an empty block **is** the result with no view. Its update is a sum of precisions, ``\\hat{\\mathbf{\\Sigma}}^{-1} = \\mathbf{\\Sigma}_f^{-1} + \\mathbf{P}^\\intercal\\mathbf{\\Omega}^{-1}\\mathbf{P}``, so a `0 × n` ``\\mathbf{P}`` adds zero, and there is no term of estimation error ``\\tau\\mathbf{\\Sigma}`` to widen the covariance. The members that run [`vanilla_posteriors`](@ref) have that term, and that is why they use [`bl_posteriors`](@ref). The result is not the wrapped prior of that member but ``\\mathbf{\\Sigma}_a + \\mathbf{M}\\mathbf{\\Sigma}_f\\mathbf{M}^\\intercal``, the moments that its factor model gives, and the member has no unchanged prior to return.

# Arguments

  - `blp`: The result of [`bl_preroll`](@ref) for this system, or `nothing`.
  - `n`: The width of the axis of this system, for the empty `P`.
  - `datatype`: The numeric type of the block.

# Returns

  - `(P, Q, omega)::Tuple`: The view matrix, the view returns and the view uncertainty, with no row when the system has no view left.

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

Refuse a **precomputed** view matrix over a universe that an asset left. This is the one configuration where the estimator cannot reduce the asset views.

The reduction resolves the view *names* of the caller against the universe that remains. With the names, the estimator can drop a row that names a departed asset, and build each other row again over the investable columns. A precomputed [`BlackLittermanViews`](@ref) resolves no name. [`black_litterman_views`](@ref) returns it unchanged, because the caller assembled it over its own universe and no later step can check it. Thus the estimator cannot reduce its `P`, and it has no names to reduce it *with*: an estimator with precomputed views can have no `sets`.

Without this check, the configuration fails with no clear report. [`calc_omega`](@ref) forms `P * sigma * transpose(P)` over the full universe, `0 * NaN` is `NaN`, and each entry of the posterior is `NaN`. Under the default `mp`, `posdef!` then raises `ArgumentError: matrix contains Infs or NaNs`, which names the wrong cause. Under a matrix processing estimator that does nothing, the fit *gives no error*, and the next layer finds no investable asset. This function refuses the configuration with an error that names it.

The two methods split by dispatch on the type of `sets`. `sets` is a field of a `@concrete` estimator, so its type is known when the method compiles. The `UniverseSets` method is the usual path, and it does nothing. Only the method for `nothing` derives the mask, and only to refuse.

# Arguments

  - $(arg_dict[:sets])
  - $(arg_dict[:pr])

# Validation

  - When `sets` is `nothing`, each asset of `pr` is investable. Raises an `ArgumentError`.

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

Each method selects one shape of `views_conf` and computes one branch of the closed forms below. `::Nothing` gives the diagonal with no scale, `::Number` gives the same diagonal with one scale for all the views, and `::VecNum` gives the same diagonal with one scale for each view.

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

The branch with no confidence is the diagonal uncertainty of the view creation model. [`bl_preroll`](@ref) multiplies the result by ``\\tau``, so the two functions together return ``\\mathrm{Diag}(\\mathbf{P}(\\tau\\mathbf{\\Sigma})\\mathbf{P}^\\intercal)``.

A confidence ``v`` scales that diagonal by ``1/v - 1``. This is the method of Idzorek in the closed form of Walters. A high confidence makes the view uncertainty smaller, and a low confidence makes it larger. The scale is negative for each ``v`` outside ``(0, 1)``, and [`assert_bl_views_conf`](@ref) refuses such a value.

A scalar ``v`` gives the same ``\\mathbf{\\Omega}`` as the constant vector of ``v``, to the last bit. The two shapes are two forms of one input, and [`assert_bl_views_conf`](@ref) counts only the vector against the views.

The bound is strict, and it also refuses the two endpoints. ``v = 1`` gives ``\\mathbf{\\Omega} = \\mathbf{0}``, a view with no uncertainty. Then ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}`` is singular when ``\\mathbf{P}`` is rank-deficient. Two identical views over a sample of three assets give a ``\\mathbf{P}`` of rank one. At ``v = 1`` the sum is the constant matrix `1.0021e-6`, its determinant is `0.0` and its rank is 1, and the solve raises `SingularException`. Near the bound the sum is ill-conditioned but not singular: at ``v = 1 - 10^{-8}`` the condition number is `2.0e8`, and at ``v = 0.99`` it is `199`. ``v = 0`` gives an infinite uncertainty, which has the same effect as no view.

# Arguments

  - `views_conf`:

      + `::Nothing`: No confidence. `Ω = Diag(P * sigma * P')`.
      + `::Number`: One confidence `v` for all the views. `Ω = (1/v - 1) * Diag(P * sigma * P')`.
      + `::VecNum`: One confidence for each view, the vector `v`. `Ω = Diag((1 ./ v .- 1) .* diag(P * sigma * P'))`.

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

Compute the Black-Litterman posterior mean and covariance.

The function does not add a risk-free rate. Each Black-Litterman prior estimator adds its own `rf` once, through [`apply_rf`](@ref), which states where each member adds it.

The two equations below are the **inverse-free** form of the master equations. They are algebraically equal to the form on [`prior`](@ref): the covariance term is the Woodbury expansion of ``\\left[(\\tau\\mathbf{\\Sigma})^{-1} + \\mathbf{P}^\\intercal\\mathbf{\\Omega}^{-1}\\mathbf{P}\\right]^{-1}``. The function uses this form because it solves against one ``K \\times K`` matrix, and the other form inverts three ``N \\times N`` matrices.

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

``\\tau`` occurs in both equations, but it does **not** move the posterior mean when ``\\mathbf{\\Omega}`` comes from [`bl_preroll`](@ref). [`calc_omega`](@ref) is homogeneous of degree one in the covariance that it reads, and `bl_preroll` multiplies its result by ``\\tau``. Thus the gain ``\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal(\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega})^{-1}`` has one ``\\tau`` outside the inverse and one inside it, and the two cancel. On a ``200 \\times 3`` sample with two views, over ``\\tau \\in \\{1/200, 0.05, 0.5\\}``, the posterior mean moves by at most `3.3e-19` on each confidence branch: no confidence, the scalar `0.4`, and the vector `[0.25, 0.75]`. The posterior **covariance** does move, because ``\\mathbf{\\Sigma} + \\tau\\mathbf{\\Sigma} - \\ldots`` holds a ``\\tau`` that does not cancel. Its excess over ``\\mathbf{\\Sigma}`` has trace `2.013e-7`, `2.013e-6` and `2.013e-5` at ``\\tau = 0.001``, `0.01` and `0.1`, which is linear in ``\\tau`` to three figures.

A view that repeats the prior changes nothing. With ``\\boldsymbol{q} = \\mathbf{P}\\boldsymbol{\\Pi}`` the residual ``\\boldsymbol{q} - \\mathbf{P}\\boldsymbol{\\Pi}`` is zero, so the posterior mean is equal to the prior mean. On the same sample the difference is `0.0`.

# Algorithm

 1. Multiply the prior covariance by `tau` and by the transpose of `P`. The result is `v1`, the ``N \\times K`` matrix ``\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal``.
 2. Multiply `P` by `v1`, and add the view uncertainty. The result is `v2`, the ``K \\times K`` matrix ``\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}``. This is the only matrix that the function solves against.
 3. Subtract `P * prior_mu` from `Q`. The result is the view residual `v3`.
 4. Solve `v2` against `v3`, multiply the solution by `v1`, and add it to `prior_mu`. The result is `posterior_mu`.
 5. Solve `v2` against the transpose of `v1`, multiply the solution by `v1`, and subtract it from `prior_sigma + tau * prior_sigma`. The result is `posterior_sigma`.

# Arguments

  - `tau`: The scale of the uncertainty of the prior mean.
  - `prior_mu`: The prior mean of the distribution that the views update.
  - `prior_sigma`: The prior covariance of that distribution.
  - `omega`: The view uncertainty matrix, multiplied by `tau`.
  - `P`: The view matrix, `views × n`.
  - `Q`: The view returns, one for each view.

# Returns

  - `posterior_mu::VecNum`: The posterior mean.
  - `posterior_sigma::Matrix{<:Number}`: The posterior covariance matrix.

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

Add the risk-free rate to a Black-Litterman mean.

`apply_rf` is the **one site that reads** the `rf` field of a Black-Litterman prior estimator. Each of the four members, [`BlackLittermanPrior`](@ref), [`BayesianBlackLittermanPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref), calls it once. No function subtracts the rate.

The four members keep three rules:

  - **The estimator adds the rate once.** No method adds it twice.
  - **The update uses the scale of the views.** The update forms the residual `Q - P * mu`, so the prior mean must have the scale of `Q`, which is a total return. A mean from a wrapped prior estimator has that scale, and it goes to the update with no change. The equilibrium mean of [`equilibrium_mu`](@ref) is only a risk premium. Thus the two members that can make one add the rate to it **before** the update, on the axis of that mean. If the prior mean has no rate, the update compares the views with the wrong level, so these members add the rate first and not last.
  - **The wrapped prior does not change.** The estimator does not fit a wrapped prior again and does not scale its mean, so a risk-free rate that a wrapped prior applied stays in its mean.

Each member calls it at one site:

  - [`BlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) have no equilibrium branch, so they have no mean to convert. They add the rate to the posterior asset mean as the last step, and `rf` shifts the result.
  - [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) call it on the equilibrium mean, and only when `l` is not `nothing`. When `l` is `nothing`, they do not read `rf`, and it has no effect on the result.

A measurement shows that the estimator adds the rate once. Two [`BlackLittermanPrior`](@ref) fits over one ``200 \\times 3`` sample, with different `rf` and no other difference, give posterior means whose difference is `rf` in each entry, to the last bit. At `rf = 0.03` the difference is `[0.03, 0.03, 0.03]`, and `max|diff - rf|` is `0.0`.

# Algorithm

 1. Add `rf` to every entry of `mu`, and return the result. The input is not modified.

# Arguments

  - `rf`: Risk-free rate.
  - `mu`: Expected returns vector. It is a posterior asset mean for the two members with no equilibrium branch, and a prior equilibrium mean for the two members that have one.

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

This method is for a confidence that is `nothing` or a scalar. Neither has one entry for each view, because a scalar is one confidence for all the views. A dropped view changes nothing for them, and the method ignores each argument after the first.

# Algorithm

 1. Return `views_conf` unchanged.

# Arguments

  - `views_conf`: `nothing`, or one confidence for all the views.
  - `args...`: The indices of the dropped views. The method ignores them.

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

This method is for a vector of confidences when no view was dropped. When [`get_black_litterman_views`](@ref) drops no view, it writes `excl = nothing` and not an empty vector, so this method is the usual case.

# Algorithm

 1. Return `views_conf` unchanged.

# Arguments

  - `views_conf`: One confidence for each view.
  - `::Nothing`: No view was dropped.

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

This method removes entries. It takes a vector with one confidence for each view that the caller wrote, and the indices of the views that [`get_black_litterman_views`](@ref) dropped. The remaining entries keep the order of the caller, so entry `k` of the result belongs to row `k` of the `P` of the same assembly. When `excl` holds every view, the result is an empty vector.

# Algorithm

 1. Take the indices of `views_conf` that are not in `excl`, in ascending order.
 2. Return the entries at those indices as a lazy view, with [`nothing_scalar_array_view`](@ref).

# Arguments

  - `views_conf`: One confidence for each view that the caller wrote.
  - `excl`: The indices of the dropped views, over the views that the caller wrote.

# Returns

  - `views_conf::VecNum`: A view of the input, with one confidence for each kept view.

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

Fit the Black-Litterman posterior of the asset returns.

When `pe.tau` is `nothing`, `tau` is `1/T`, where `T` is the number of rows of `X` in the result of the wrapped prior. `pe.rf` changes the result once, on the posterior asset expected returns. [`apply_rf`](@ref) states that rule.

# Mathematical definition

The Black-Litterman posterior combines the prior ``(\\boldsymbol{\\Pi}, \\tau \\mathbf{\\Sigma})`` with the views ``(\\mathbf{P}, \\boldsymbol{q}, \\mathbf{\\Omega})``. The equations below are the precision form. [`vanilla_posteriors`](@ref) computes the inverse-free form, which is algebraically equal:

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
 2. When `pe.views` resolves names, check that the length of the asset universe is equal to the number of columns of `X`, with [`assert_bl_views_axis`](@ref). A precomputed [`BlackLittermanViews`](@ref) resolves no name, so this step does not check it. [`bl_preroll`](@ref) checks its width in step 6.
 3. Fit the wrapped prior `pe.pe` on `(X, F)`. The result is `prior_model`.
 4. Derive the Investable Mask and the reduced view universe with [`investable_views`](@ref). Refuse a precomputed view matrix over a universe with a departed asset, with [`assert_bl_precomputed_universe`](@ref).
 5. View the fitted prior at the mask with [`investable_prior`](@ref), and read `posterior_X`, `prior_mu` and `prior_sigma` from *that* result.
 6. Assemble the views and their uncertainty with [`bl_preroll`](@ref), over the reduced `prior_sigma`, with `T` equal to the number of rows of `prior_model.X`. The result is `P`, `Q`, `tau` and `omega`, or `nothing` when no view is left. The axis keeps its default, `:xkey`, because these views update the assets.
 7. Run the master equations with [`bl_posteriors`](@ref). The result is `posterior_mu` and `posterior_sigma` over the investable assets.
 8. Add `pe.rf` to `posterior_mu` with [`apply_rf`](@ref). This is the one site that adds it.
 9. Process `posterior_sigma` in place with [`matrix_processing!`](@ref), under `pe.mp` and `posterior_X`. Do this before the expansion, because a factorisation exists only for the reduced block.
10. Report the departures once with [`announce_bl_departures`](@ref).
11. Write both posteriors back onto the full asset universe with [`expand_moment`](@ref), so a non-investable asset carries `NaN` in `mu` and on the diagonal of `sigma`.
12. Forward the whole of `prior_model` with [`forward_prior`](@ref), replacing `mu` and `sigma` by the expanded pair and dropping `chol`.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `strict`: If `true`, a view name that does not resolve raises an error. If `false`, it warns. The method also passes `strict` to the wrapped prior.
  - `kwargs...`: More keyword arguments, for the wrapped prior and for the matrix processing.

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)`. Raises a `DimensionMismatch`.
  - If `pe.sets` is `nothing`, each asset of the result of the wrapped prior is investable, see [`assert_bl_precomputed_universe`](@ref).
  - The number of columns of a precomputed `P` is equal to the number of investable assets, see [`bl_preroll`](@ref).

# Returns

  - `pr::LowOrderPrior`: The result of the wrapped prior, with the posterior mean in `mu`, the posterior covariance in `sigma`, and `chol = nothing`.

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
    assert_bl_views_axis(pe, X)
    prior_model = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    return bl_posterior(pe, prior_model, strict; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a [`BlackLittermanPrior`](@ref) whose asset universe does not match the returns matrix.

The function checks the axis only for views that resolve names against it. A precomputed [`BlackLittermanViews`](@ref) holds its own `P` and does not read `sets`. A check of the universe would refuse that valid configuration, which [`assert_bl`](@ref) lets have no `sets`.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Asset returns, `observations × assets`.

# Validation

  - `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)` when the views resolve names. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`assert_bl`](@ref)
"""
function assert_bl_views_axis(pe::BlackLittermanPrior, X::MatNum)
    if isa(pe.views, LinearConstraintEstimator)
        @argcheck(length(pe.sets.dict[pe.sets.xkey]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.xkey]) ($(length(pe.sets.dict[pe.sets.xkey]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Update the result of the wrapped prior of a [`BlackLittermanPrior`](@ref) with the views of the estimator.

This function holds each step of the estimator after the wrapped prior gives its result. The batch method calls it with the result of a fit over the matrix of the caller. `prior(pe)` with no data calls it with the result of the fold of the wrapped prior. The function reads no returns matrix that the result does not hold: `T`, the axis and the matrix for the processing all come from `prior_model`. Both routes use this one body, so they cannot become different.

# Algorithm

 1. Derive the Investable Mask and the reduced view universe with [`investable_views`](@ref). A view is a dense linear form over the asset axis, so one departed asset makes each entry of `omega` `NaN`.
 2. Refuse a precomputed view matrix over a universe with a departed asset, with [`assert_bl_precomputed_universe`](@ref).
 3. View the prior result at the mask with [`investable_prior`](@ref), and read `posterior_X`, `prior_mu` and `prior_sigma` from it.
 4. Assemble `P`, `Q`, `tau` and `omega` with [`bl_preroll`](@ref). Its `T` is the number of rows of `prior_model.X`.
 5. Run the update with [`bl_posteriors`](@ref), and add `pe.rf` once with [`apply_rf`](@ref).
 6. Process the reduced covariance with [`matrix_processing!`](@ref), before the expansion, because a frame with `NaN` entries has no factorisation.
 7. Report the departures once with [`announce_bl_departures`](@ref).
 8. Expand both moments to the universe of the caller with [`expand_moment`](@ref).
 9. Forward each field of the wrapped result with [`forward_prior`](@ref), and drop only `chol`.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `prior_model`: The result of the wrapped prior.
  - `strict`: If `true`, a view name that does not resolve raises an error. If `false`, it warns.
  - `kwargs...`: More keyword arguments, for the matrix processing.

# Returns

  - `pr::AbstractPriorResult`: The wrapped result, with the posteriors in place of its two moments.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_preroll`](@ref)
  - [`bl_posteriors`](@ref)
  - [`investable_views`](@ref)
  - [`forward_prior`](@ref)
"""
function bl_posterior(pe::BlackLittermanPrior, prior_model::AbstractPriorResult,
                      strict::Bool; kwargs...)
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
    blp = bl_preroll(pe.views, vsets, pe.views_conf, prior_sigma, pe.tau,
                     size(prior_model.X, 1), eltype(posterior_X), strict; ledger = ledger)
    # `nothing` is a view set with no row left, whatever emptied it, and the pair is then the
    # prior's own.
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
    # universe and nothing outside it may see a narrowed prior result. Black-Litterman
    # leaves the observation axis untouched — the reduction takes columns, never rows — so
    # the wrapped `w` still describes exactly the rows of the returned `X`, and its
    # `ens`/`kld`/`ow` still describe that `w`.
    # `rr` is structural — the regression of `X` on `F`, over data Black-Litterman does not
    # modify — and the factor block `fpr` travels with it.
    return forward_prior(prior_model; mu = posterior_mu, sigma = posterior_sigma,
                         chol = nothing)
end

function factor_residual_config(pe::BlackLittermanPrior)
    return factor_residual_config(pe.pe)
end

"""
    fold_prior_step(pe::BlackLittermanPrior, rd::ReturnsResult,
                    gaps::Option{<:AbstractVector{Bool}})

Fold the observations of a [`ReturnsResult`](@ref) into a [`BlackLittermanPrior`](@ref). The function forwards `rd` and its marks to the wrapped prior.

This is the forwarding form of [`fold_prior_step`](@ref), the fold that [`partial_fit!`](@ref) uses for each prior over a `ReturnsResult`. The outer prior keeps no rows. The wrapped prior keeps them, so the route of the wrapped prior decides what the step uses. The estimation mask and the Panel Fields reach it with no change, and it refuses what its route cannot use.

# Arguments

  - `pe`: The Black-Litterman prior estimator.
  - $(arg_dict[:rd])
  - `gaps`: One mark for each observation of `rd`, `true` at an observation with a Shared Series gap, or `nothing`.

# Validation

  - Each condition that the fold of `pe.pe` over `rd` checks.

# Returns

  - `pe`: A new estimator, with the folded wrapped prior.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`fold_prior_step`](@ref)
"""
function fold_prior_step(pe::BlackLittermanPrior, rd::ReturnsResult,
                         gaps::Option{<:AbstractVector{Bool}})
    return rebuild_estimator(pe, (; pe = fold_prior_step(pe.pe, rd, gaps)))
end

export BlackLittermanPrior
