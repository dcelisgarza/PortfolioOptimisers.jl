"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments and its tails meet a set of views.

`EntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports views on the mean, the variance, the covariance, the correlation, the skewness and the kurtosis, views on the value at risk, the conditional and entropic value at risk views of [EPTail](@cite), and the relativistic value at risk views of [EPRLVaR](@cite).

The tail views are the difference with [`MeucciEntropyPoolingPrior`](@ref). There, a CVaR view is a target the recursive algorithm of Meucci et al. hunts by re-solving the whole entropy pooling problem for each candidate value at risk level, which supports equalities alone. Here each tail view is written as constraints of the single entropy pooling problem, so one solve answers every view, and the operators `==`, `>=` and `<=` are all available, along with views on the entropic and the relativistic value at risk, and views over several assets of every measure: a positive combination is convex and exact, and a relative view with coefficients of both signs takes the integer formulation or a sequential convex one.

!!! warning

    An infeasible view set is not raised on by the [`OptimEntropyPooling`](@ref) route. The dual of an infeasible set is unbounded, so the minimiser runs away, the posterior collapses onto one observation, and `Optim` reports the solve as converged. A grossly infeasible view overflows instead, and the non-finite weights reach the moment estimators as an `ArgumentError` naming Infs or NaNs. Read the result rather than the flag: `ens` falls to a handful out of the number of observations, one weight sits near one, `kld` is large, and the posterior statistic the view named is far from its target. [`entropy_pooling`](@ref) states the mechanism. The [`JuMPEntropyPooling`](@ref) route does not share it: the solver reports an infeasible model itself.

# Algorithm

The constructor derives the prior probabilities, and validates everything else.

 1. When `w` is `nothing`, derive nothing. [`prior`](@ref) builds the uniform weights `1/T` at solve time, one per observation.
 2. When `w` is not `nothing`, normalise it to sum to one, giving the prior probabilities the pooling starts from. A mutable `w.values` is normalised in place with `LinearAlgebra.normalize!`, and an immutable one is replaced by a new `StatsBase.pweights` over the normalised values.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyPoolingPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        mu_views::Option{<:LinearConstraintEstimator} = nothing,
        var_views::Option{<:VV_VecVV} = nothing,
        cvar_views::Option{<:CVV_VecCVV} = nothing,
        evar_views::Option{<:EVV_VecEVV} = nothing,
        rlvar_views::Option{<:RVV_VecRVV} = nothing,
        sigma_views::Option{<:LinearConstraintEstimator} = nothing,
        sk_views::Option{<:LinearConstraintEstimator} = nothing,
        kt_views::Option{<:LinearConstraintEstimator} = nothing,
        cov_views::Option{<:LinearConstraintEstimator} = nothing,
        rho_views::Option{<:LinearConstraintEstimator} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        opt::NonCVaREP = OptimEntropyPooling(),
        w::Option{<:StatsBase.ProbabilityWeights} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> EntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If any view constraint is not `nothing`, `sets` must not be `nothing`.
  - If `cvar_views` is not `nothing`, `opt` must be a [`JuMPEntropyPooling`](@ref).
  - If a view field is a vector, it must not be empty.
  - If `w` is not `nothing`, it must be non-empty and match the number of observations.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `pe`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# View comparison operators

The comparison operators accepted in each view's constraint strings depend on the moment being constrained. An unsupported operator raises a `ParseError` listing the operators allowed for that view.

  - `mu_views`, `sigma_views`, `sk_views`, `kt_views`, `cov_views`, `rho_views` accept `==`, `>=` and `<=`.
  - `var_views` (Value at Risk) accepts only `==` and `>=`.
  - `cvar_views`, `evar_views` and `rlvar_views` accept `==`, `>=` and `<=`.

# Tail views

A tail view needs auxiliary variables, so it is expressed in the JuMP model rather than reduced to rows that multiply the posterior probabilities. Two consequences follow.

  - `opt` must be a [`JuMPEntropyPooling`](@ref) whenever `cvar_views` is set, and whenever an `evar_views` or `rlvar_views` entry is anything other than a lower bound under [`GridEntropicValueatRiskView`](@ref) or [`GridRelativisticValueatRiskView`](@ref), which are the tail formulations that are linear in the posterior probabilities alone.
  - A view that needs a binary variable — every [`IntegerConditionalValueatRiskView`](@ref), and an upper bound or equality under [`GridEntropicValueatRiskView`](@ref) or [`GridRelativisticValueatRiskView`](@ref) — needs a solver that handles mixed-integer exponential cone programs.
  - [`ConicRelativisticValueatRiskView`](@ref) writes power cones, so its solver must handle the power cone alongside the exponential cone the objective needs.

The `alg` field of a view group picks the formulation. A single formulation applies to every view in that group, a vector supplies one per view, and `nothing` lets each view take the cheapest formulation that expresses it exactly: [`LinearConditionalValueatRiskView`](@ref), [`ConicEntropicValueatRiskView`](@ref) and [`ConicRelativisticValueatRiskView`](@ref) where they apply, [`IntegerConditionalValueatRiskView`](@ref), [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref) otherwise.

# Tail views at several significance levels

A significance level is part of the statistic, not a detail of the solve: the conditional value at risk at 1% and at 10% are different numbers on the same series. So the level lives on the view rather than on the estimator. `var_views`, `cvar_views`, `evar_views` and `rlvar_views` each take one [`ValueatRiskView`](@ref), [`ConditionalValueatRiskView`](@ref), [`EntropicValueatRiskView`](@ref) or [`RelativisticValueatRiskView`](@ref), or a vector of them, and each group carries the `alpha` its equations are read under. A [`RelativisticValueatRiskView`](@ref) carries a `kappa` as well, on the same reasoning: the deformation parameter is part of the statistic. A `prior(...)` reference inside a group is replaced by the prior value at *that* group's level.

A tail view group also carries `alg`, the formulation. For [`EntropicValueatRiskView`](@ref) that is where the grid of dual variables and the big-M multiplier live, so a [`GridEntropicValueatRiskView`](@ref) there gives one group its own `pct`, `K` and `M`. [`ValueatRiskView`](@ref) has no such field: a value at risk view is linear in the posterior probabilities, so there is no formulation to choose.

# Examples

```jldoctest
julia> EntropyPoolingPrior(;
                           sets = UniverseSets(; xkey = \"nx\",
                                               dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                           mu_views = LinearConstraintEstimator(;
                                                                val = [\"A == 0.03\",
                                                                       \"B + C == 0.04\"]))
EntropyPoolingPrior
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
              │   fill_limit ┴ nothing
     mu_views ┼ LinearConstraintEstimator
              │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
              │   key ┴ nothing
    var_views ┼ nothing
   cvar_views ┼ nothing
   evar_views ┼ nothing
  rlvar_views ┼ nothing
  sigma_views ┼ nothing
     sk_views ┼ nothing
     kt_views ┼ nothing
    cov_views ┼ nothing
    rho_views ┼ nothing
         sets ┼ UniverseSets
              │     xkey ┼ String: "nx"
              │    uxkey ┼ String: "ux"
              │    tfkey ┼ String: "nf"
              │   utfkey ┼ String: "uf"
              │    cfkey ┼ String: "ncf"
              │   ucfkey ┼ String: "ucf"
              │    nikey ┼ String: "ni"
              │     dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
          opt ┼ OptimEntropyPooling
              │     args ┼ Tuple{}: ()
              │   kwargs ┼ @NamedTuple{}: NamedTuple()
              │      sc1 ┼ Int64: 1
              │      sc2 ┼ Float64: 1000.0
              │      alg ┼ ExpEntropyPooling()
              │      err ┴ nothing
            w ┼ nothing
          alg ┴ H1_EntropyPooling()
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:meucci2008])
  - $(ref_dict[:vorobets2021])
  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
@propagatable @concrete struct EntropyPoolingPrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:mu_views])
    """
    mu_views
    """
    $(field_dict[:var_views])
    """
    var_views
    """
    $(field_dict[:cvar_views])
    """
    cvar_views
    """
    $(field_dict[:evar_views])
    """
    evar_views
    """
    $(field_dict[:rlvar_views])
    """
    rlvar_views
    """
    $(field_dict[:sigma_views])
    """
    sigma_views
    """
    $(field_dict[:sk_views])
    """
    sk_views
    """
    $(field_dict[:kt_views])
    """
    kt_views
    """
    $(field_dict[:cov_views])
    """
    cov_views
    """
    $(field_dict[:rho_views])
    """
    rho_views
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:opt_ep])
    """
    opt
    """
    $(field_dict[:ep_w])
    """
    @wprop w
    """
    $(field_dict[:epalg])
    """
    alg
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function EntropyPoolingPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mu_views::Option{<:LinearConstraintEstimator},
                                 var_views::Option{<:VV_VecVV},
                                 cvar_views::Option{<:CVV_VecCVV},
                                 evar_views::Option{<:EVV_VecEVV},
                                 rlvar_views::Option{<:RVV_VecRVV},
                                 sigma_views::Option{<:LinearConstraintEstimator},
                                 sk_views::Option{<:LinearConstraintEstimator},
                                 kt_views::Option{<:LinearConstraintEstimator},
                                 cov_views::Option{<:LinearConstraintEstimator},
                                 rho_views::Option{<:LinearConstraintEstimator},
                                 sets::Option{<:UniverseSets}, opt::NonCVaREP,
                                 w::Option{<:StatsBase.ProbabilityWeights},
                                 alg::AbstractEntropyPoolingAlgorithm,
                                 cache::Option{<:AbstractPartialFitState})
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
            if ismutable(w.values)
                LinearAlgebra.normalize!(w, 1)
            else
                w = StatsBase.pweights(LinearAlgebra.normalize(w, 1))
            end
        end
        if !isnothing(mu_views) ||
           !isnothing(var_views) ||
           !isnothing(cvar_views) ||
           !isnothing(evar_views) ||
           !isnothing(rlvar_views) ||
           !isnothing(sigma_views) ||
           !isnothing(sk_views) ||
           !isnothing(kt_views) ||
           !isnothing(cov_views) ||
           !isnothing(rho_views)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if !isnothing(cvar_views)
            @argcheck(isa(opt, JuMPEntropyPooling),
                      ArgumentError("A CVaR view needs auxiliary variables, which the dual formulation `OptimEntropyPooling` solves has no room for. Use `JuMPEntropyPooling` in `opt`."))
        end
        if isa(var_views, AbstractVector)
            @argcheck(!isempty(var_views), IsEmptyError("var_views cannot be empty"))
        end
        if isa(cvar_views, AbstractVector)
            @argcheck(!isempty(cvar_views), IsEmptyError("cvar_views cannot be empty"))
        end
        if isa(evar_views, AbstractVector)
            @argcheck(!isempty(evar_views), IsEmptyError("evar_views cannot be empty"))
        end
        if isa(rlvar_views, AbstractVector)
            @argcheck(!isempty(rlvar_views), IsEmptyError("rlvar_views cannot be empty"))
        end
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(evar_views), typeof(rlvar_views), typeof(sigma_views),
                   typeof(sk_views), typeof(kt_views), typeof(cov_views), typeof(rho_views),
                   typeof(sets), typeof(opt), typeof(w), typeof(alg), typeof(cache)}(pe,
                                                                                     mu_views,
                                                                                     var_views,
                                                                                     cvar_views,
                                                                                     evar_views,
                                                                                     rlvar_views,
                                                                                     sigma_views,
                                                                                     sk_views,
                                                                                     kt_views,
                                                                                     cov_views,
                                                                                     rho_views,
                                                                                     sets,
                                                                                     opt, w,
                                                                                     alg,
                                                                                     cache)
    end
end
function EntropyPoolingPrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             mu_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_views::Option{<:VV_VecVV} = nothing,
                             cvar_views::Option{<:CVV_VecCVV} = nothing,
                             evar_views::Option{<:EVV_VecEVV} = nothing,
                             rlvar_views::Option{<:RVV_VecRVV} = nothing,
                             sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                             sk_views::Option{<:LinearConstraintEstimator} = nothing,
                             kt_views::Option{<:LinearConstraintEstimator} = nothing,
                             cov_views::Option{<:LinearConstraintEstimator} = nothing,
                             rho_views::Option{<:LinearConstraintEstimator} = nothing,
                             sets::Option{<:UniverseSets} = nothing,
                             opt::NonCVaREP = OptimEntropyPooling(),
                             w::Option{<:StatsBase.ProbabilityWeights} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling(),
                             cache::Option{<:AbstractPartialFitState} = nothing)::EntropyPoolingPrior
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, evar_views, rlvar_views,
                               sigma_views, sk_views, kt_views, cov_views, rho_views, sets,
                               opt, w, alg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`EntropyPoolingPrior`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:EntropyPoolingPrior, true)` to render it.

# Arguments

  - `::EntropyPoolingPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :mu_views, :var_views, :cvar_views, :evar_views, :rlvar_views, :sigma_views, :sk_views, :kt_views, :cov_views, :rho_views, :sets, :opt, :w, :alg)`.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::EntropyPoolingPrior)
    return (:pe, :mu_views, :var_views, :cvar_views, :evar_views, :rlvar_views,
            :sigma_views, :sk_views, :kt_views, :cov_views, :rho_views, :sets, :opt, :w,
            :alg)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties EntropyPoolingPrior begin
    forward(pe, me, ce)
end
"""
    const VecEP = AbstractVector{<:Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}}

Alias for an abstract vector of entropy pooling prior estimators of either family.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
const VecEP = AbstractVector{<:Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}}
"""
    prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute the entropy pooling prior of asset returns with tail views.

`prior` orients the data and forwards the estimator's algorithm as a value to [`ep_prior`](@ref), which enforces the views in stages or in one optimisation.

# Algorithm

 1. Orient `X` and `F` along `dims` with [`dims_oriented`](@ref), so the observations lie in the rows.
 2. Forward `pe.alg` as a value to [`ep_prior`](@ref), and return the [`LowOrderPrior`](@ref) it builds.

# Arguments

  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix.
  - `F`: Optional factor returns matrix.
  - $(arg_dict[:pnl_prior])
  - `dims`: Dimension along which the observations lie.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`ep_prior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    X, F = dims_oriented(dims, X, F)
    return ep_prior(pe.alg, pe, X, F, pnl; strict = strict, kwargs...)
end
"""
    ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum},
             pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing the views in stages.

`ep_prior` accumulates the views of each stage into one constraint set and solves once per stage, so a stage's views hold alongside every view of the stages before it. The mean of an asset a later stage constrains is pinned to the value the earlier stage produced, so a higher moment view does not silently move it.

# Algorithm

 1. Fit the wrapped prior estimator, giving `pr`. The fit states the observation axis: `T` is `size(pr.X, 1)`, which a nested prior that drops rows makes smaller than `size(X, 1)`.
 2. Read the prior probabilities `w0` on that axis with [`ep_prior_probabilities`](@ref). They are `pe.w` where the caller set one, `pr.w` where the fit answered one, and the uniform `1/T` otherwise. A caller's `pe.w` reaches the wrapped estimator through [`factory`](@ref), and `pr` is refitted under it.
 3. Stage one holds the mean, value at risk, conditional, entropic and relativistic value at risk views. Accumulate them into the constraint dictionary `epc` and the tail view vector `tvs`. Where either is non-empty, solve from `w0` with [`entropy_pooling`](@ref), giving `w1`, and refit `pr` at `w1`.
 4. Stage two holds the variance and covariance views, with the mean of every asset they name pinned by [`fix_mu!`](@ref). Where `epc` or `tvs` is non-empty, solve from `w0` under [`H1_EntropyPooling`](@ref), or from the previous `w1` under [`H2_EntropyPooling`](@ref), and refit `pr` at the new `w1`.
 5. Stage three holds the skewness, kurtosis and correlation views, with the mean and the variance of every asset they name pinned by [`fix_mu!`](@ref) and [`fix_sigma!`](@ref). Under the same emptiness test, solve from the same start step 4 takes, and refit `pr` at the new `w1`.
 6. Compute `ens`, the effective number of scenarios of `w1`, and `kld`, the divergence of `w1` from `w0`.
 7. Return a [`LowOrderPrior`](@ref) carrying the refit moments, `w1`, `ens` and `kld`. The feature matrix and the factor block come from `pr` unchanged.

Every row of every family can drop under `strict = false`, and a stage then states no view. The emptiness test is on the rows the stages hold, not on the fields the caller set, so such a fit answers the prior: `w1` is `w0`, `kld` is zero, and no refit runs.

# Arguments

  - `alg`: Staged entropy pooling algorithm.

      + `::H1_EntropyPooling`: Each stage re-solves from the original prior weights.
      + `::H2_EntropyPooling`: Each stage re-solves from the previous stage's weights.

  - `pe`: Entropy pooling prior estimator.

  - `X`: Asset returns matrix, already oriented.

  - `F`: Optional factor returns matrix, already oriented.

  - $(arg_dict[:pnl_prior])

  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum},
                  pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)
    # A prior that reweights observations works on the observation axis its nested prior
    # ANSWERED, not on the axis it was handed: a nested prior may drop rows. So the nested
    # prior is fitted first, and `ep_prior_probabilities` reads the prior probabilities on
    # the rows of `pr.X`. See ADR 0116.
    pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    w1 = w0 = ep_prior_probabilities(pe.w, pr, size(X, 1))
    if !isnothing(pe.w)
        # A caller's prior probabilities weight the moments the nested estimator measures,
        # so the estimator is refitted under them. A uniform vector states no tilt, and the
        # nested result's own `w` is already carried by the fit that answered it, so
        # neither is pushed.
        pe = factory(pe, w0)
        pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    end
    # A view is a dense linear form over the asset axis, and `0 * NaN` is `NaN`, so every
    # row is built on the investable columns alone. The mask does not move between stages,
    # so it is derived once and each refit is viewed at it. Nothing is expanded: an `epc`
    # row runs over observations, and the moments come from the refit prior, which already
    # carries the full-universe frame. See ADR 0115 and ADR 0125.
    imsk, vsets, ni = investable_views(pr, pe.sets)
    led = String[]
    vpr = investable_prior(imsk, pr)
    fixed = falses(size(vpr.X, 2), 2)
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    tvs = Vector{AbstractEntropyPoolingTailView}(undef, 0)
    # mu, VaR, CVaR, EVaR and RLVaR
    ep_mu_views!(pe.mu_views, epc, vpr, vsets; strict = strict, ledger = led)
    ep_var_views!(pe.var_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    ep_tail_views!(pe.cvar_views, epc, tvs, vpr, vsets, w0; strict = strict, ledger = led)
    ep_tail_views!(pe.evar_views, epc, tvs, vpr, vsets, w0; strict = strict, ledger = led)
    ep_tail_views!(pe.rlvar_views, epc, tvs, vpr, vsets, w0; strict = strict, ledger = led)
    if !isempty(epc) || !isempty(tvs)
        w1 = entropy_pooling(w0, epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
        vpr = investable_prior(imsk, pr)
    end
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            to_fix = ep_sigma_views!(pe.sigma_views, epc, vpr, vsets; strict = strict,
                                     ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
        end
        # cov
        if !isnothing(pe.cov_views)
            to_fix = ep_cov_views!(pe.cov_views, epc, vpr, vsets; strict = strict,
                                   ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
        end
        # Every row of every family can drop under `strict = false`, and the stage then
        # states no view. The prior is the answer, so neither the solve nor the refit runs.
        # See issue #852.
        if !isempty(epc) || !isempty(tvs)
            w1 = entropy_pooling(ifelse(isa(alg, H1_EntropyPooling), w0, w1), epc, tvs,
                                 pe.opt)
            pe = factory(pe, w1)
            pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
            vpr = investable_prior(imsk, pr)
        end
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            to_fix = ep_sk_views!(pe.sk_views, epc, vpr, vsets; strict = strict,
                                  ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, vpr)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            to_fix = ep_kt_views!(pe.kt_views, epc, vpr, vsets; strict = strict,
                                  ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, vpr)
        end
        # rho
        if !isnothing(pe.rho_views)
            to_fix = ep_rho_views!(pe.rho_views, epc, vpr, vsets; strict = strict,
                                   ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, vpr)
        end
        # See the twin note one stage up: a stage that states no view does not solve.
        if !isempty(epc) || !isempty(tvs)
            w1 = entropy_pooling(ifelse(isa(alg, H1_EntropyPooling), w0, w1), epc, tvs,
                                 pe.opt)
            pe = factory(pe, w1)
            pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
        end
    end
    announce_ep_departures(ni, led, isempty(epc) && isempty(tvs) && !isempty(led))
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same seam in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr)
end
"""
    ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing;
             strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing every view in one optimisation.

`ep_prior` builds every view constraint against the same prior and solves once. It is faster than the staged algorithms and pins nothing, so a higher moment view is free to move a lower moment.

# Algorithm

 1. Fit the wrapped prior estimator, giving `pr`. The fit states the observation axis: `T` is `size(pr.X, 1)`, which a nested prior that drops rows makes smaller than `size(X, 1)`.
 2. Read the prior probabilities `w0` on that axis with [`ep_prior_probabilities`](@ref). They are `pe.w` where the caller set one, `pr.w` where the fit answered one, and the uniform `1/T` otherwise. A caller's `pe.w` reaches the wrapped estimator through [`factory`](@ref), and `pr` is refitted under it.
 3. Build every view against that one `pr`: the mean, value at risk, conditional, entropic and relativistic value at risk, variance, covariance, skewness, kurtosis and correlation views. Each row that is linear in the posterior probabilities reaches the constraint dictionary `epc`, and each tail view that needs auxiliary variables reaches the tail view vector `tvs`. No asset's mean or variance is pinned.
 4. Where `epc` or `tvs` is non-empty, solve once from `w0` with [`entropy_pooling`](@ref), giving `w1`, and refit `pr` at `w1`. Every row of every family can drop under `strict = false`, and the view set then states nothing: `w1` is `w0`, `kld` is zero, and no refit runs.
 5. Compute `ens`, the effective number of scenarios of `w1`, and `kld`, the divergence of `w1` from `w0`.
 6. Return a [`LowOrderPrior`](@ref) carrying the refit moments, `w1`, `ens` and `kld`. The feature matrix and the factor block come from `pr` unchanged.

# Arguments

  - `alg`: Single-shot entropy pooling algorithm.
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, already oriented.
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
                  F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing;
                  strict::Bool = false, kwargs...)
    # See the note at the same seam in the staged method: the nested prior is fitted
    # first, and the prior probabilities are read on the rows it answered. ADR 0116.
    pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    w0 = ep_prior_probabilities(pe.w, pr, size(X, 1))
    if !isnothing(pe.w)
        # A caller's prior probabilities weight the moments the nested estimator measures,
        # so the estimator is refitted under them. A uniform vector states no tilt, and the
        # nested result's own `w` is already carried by the fit that answered it, so
        # neither is pushed.
        pe = factory(pe, w0)
        pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    end
    # See the note at the same seam in the staged method: every row is built on the
    # investable columns, because `0 * NaN` is `NaN`. ADR 0115 and ADR 0125.
    imsk, vsets, ni = investable_views(pr, pe.sets)
    led = String[]
    vpr = investable_prior(imsk, pr)
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    tvs = Vector{AbstractEntropyPoolingTailView}(undef, 0)
    # mu, VaR, CVaR, EVaR and RLVaR
    ep_mu_views!(pe.mu_views, epc, vpr, vsets; strict = strict, ledger = led)
    ep_var_views!(pe.var_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    ep_tail_views!(pe.cvar_views, epc, tvs, vpr, vsets, w0; strict = strict, ledger = led)
    ep_tail_views!(pe.evar_views, epc, tvs, vpr, vsets, w0; strict = strict, ledger = led)
    ep_tail_views!(pe.rlvar_views, epc, tvs, vpr, vsets, w0; strict = strict, ledger = led)
    # sigma
    if !isnothing(pe.sigma_views)
        ep_sigma_views!(pe.sigma_views, epc, vpr, vsets; strict = strict, ledger = led)
    end
    # cov
    if !isnothing(pe.cov_views)
        ep_cov_views!(pe.cov_views, epc, vpr, vsets; strict = strict, ledger = led)
    end
    # skew
    if !isnothing(pe.sk_views)
        ep_sk_views!(pe.sk_views, epc, vpr, vsets; strict = strict, ledger = led)
    end
    # kurtosis
    if !isnothing(pe.kt_views)
        ep_kt_views!(pe.kt_views, epc, vpr, vsets; strict = strict, ledger = led)
    end
    # rho
    if !isnothing(pe.rho_views)
        ep_rho_views!(pe.rho_views, epc, vpr, vsets; strict = strict, ledger = led)
    end
    announce_ep_departures(ni, led, isempty(epc) && isempty(tvs) && !isempty(led))
    w1 = w0
    # Every row of every family can drop under `strict = false`, and the view set then
    # states nothing. The prior is the answer, so neither the solve nor the refit runs. See
    # issue #852.
    if !isempty(epc) || !isempty(tvs)
        w1 = entropy_pooling(w0, epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    end
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same seam in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr)
end
function factor_residual_config(pe::EntropyPoolingPrior)
    return factor_residual_config(pe.pe)
end

export EntropyPoolingPrior
