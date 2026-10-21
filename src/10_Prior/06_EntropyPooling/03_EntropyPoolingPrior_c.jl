"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments and its tails meet a set of views.

`EntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports views on the mean, the variance, the covariance, the correlation, the skewness and the kurtosis, views on the value at risk, the conditional and entropic value at risk views of [EPTail](@cite), and the relativistic value at risk views of [EPRLVaR](@cite).

The tail views are the difference with [`MeucciEntropyPoolingPrior`](@ref). There, a CVaR view is a target the recursive algorithm of [meucciardiakeel2011](@citet) hunts by re-solving the whole entropy pooling problem for each candidate value at risk level, which supports equalities alone. Here each tail view is written as constraints of the single entropy pooling problem, so one solve answers every view, and the operators `==`, `>=` and `<=` are all available, along with views on the entropic and the relativistic value at risk, and views over several assets of every measure: a positive combination is convex and exact, and a relative view with coefficients of both signs takes the integer formulation or a sequential convex one. [EPTail](@cite) states the views on one asset, and the relative CVaR view on two. The views on several assets of the other measures, the upper-bound and equality views of the relativistic value at risk, and the sequential convex formulation are the library's own.

The posterior is the entropy pooling of [meucci2008](@cite). [`H1_EntropyPooling`](@ref) and [`H2_EntropyPooling`](@ref) are the two sequential heuristics of [vorobets2021](@cite). [`H0_EntropyPooling`](@ref), a single solve, is the library's name for the original formulation. [vorobets2021](@citet) defines no H0.

!!! warning

    An infeasible view set is not raised on by the [`OptimEntropyPooling`](@ref) route. The dual of an infeasible set is unbounded, so the minimiser runs away, the posterior collapses onto one observation, and `Optim` reports the solve as converged. A grossly infeasible view overflows instead, and the non-finite weights reach the moment estimators as an `ArgumentError` naming Infs or NaNs. Read the result rather than the flag: `ens` falls to a handful out of the number of observations, one weight sits near one, `kld` is large, and the posterior statistic the view named is far from its target. [`entropy_pooling`](@ref) states the mechanism. The [`JuMPEntropyPooling`](@ref) route does not share it: the solver reports an infeasible model itself.

# Algorithm

The constructor derives the prior probabilities, and validates everything else.

 1. When `w` is `nothing`, derive nothing. [`prior`](@ref) builds the uniform weights `1/T` at solve time, one per observation.
 2. When `w` is not `nothing`, normalise it to sum to one, giving the prior probabilities the pooling starts from. The caller's `w` is left as it is: new weights of the same kind hold the normalised values, through [`scenario_probabilities`](@ref). The kind fixes the bias correction of the posterior covariance, see [`ScenarioProbabilities`](@ref).

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
        w::Option{<:ScenarioProbabilities} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling(),
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> EntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If any view constraint is not `nothing`, `sets` must not be `nothing`.
  - If `cvar_views` is not `nothing`, `opt` must be a [`JuMPEntropyPooling`](@ref).
  - If a view field is a vector, it must not be empty.
  - If `w` is not `nothing`, it must be non-empty and match the number of observations.
  - The refit under the posterior weights can state the model of `pe`, through [`assert_scenario_refit`](@ref). An [`OpinionPoolingPrior`](@ref) with `pe1` inside a factor prior or a Black-Litterman prior of `pe` is refused.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).

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

# The variance a moment view reads

A skewness, kurtosis or correlation view divides by the variance of an asset, and a staged algorithm holds that variance at its value from the stage before. Both read the variance from the prior that `pe` fits: the diagonal of its `sigma`, less the residual variance of a factor model. The covariance estimator of `pe` therefore decides which variance the rows hold.

The rows state the uncorrected moment of the scenarios, ``\\sum_{t} p_{t} (x_{t,\\,i} - \\mu_{i})^{2}``. The default covariance estimator is corrected. So, under [`H1_EntropyPooling`](@ref) or [`H2_EntropyPooling`](@ref), a skewness or kurtosis view moves the variance of the scenarios up by the correction factor of the stage before. The view also standardises by the corrected variance. To hold and standardise by the uncorrected variance of the scenarios, give `pe` an uncorrected covariance estimator:

    ce = GeneralCovariance(; ce = StatsBase.SimpleCovariance(; corrected = false))
    pe = EmpiricalPrior(; ce = PortfolioOptimisersCovariance(; ce = Covariance(; ce = ce)))

The posterior `sigma` is then uncorrected too. [`H0_EntropyPooling`](@ref) has no stage before, and it reads the variance of the prior the same way.

A `prior(...)` reference in a skewness or kurtosis view reads the sample statistic of the scenarios under the observation weights of the prior that the stage reads, as a value at risk reference does. The statistic standardises by its own variance estimator, not by the variance that the row reads.

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
              │   fill_limit ┼ nothing
              │        x_src ┴ nothing
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
          alg ┼ H1_EntropyPooling()
        x_src ┴ nothing
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

  - $(ref_dict[:meucci2008]) Section 4, Equations 25 to 28, and Appendix A.2 and A.3.
  - $(ref_dict[:vorobets2021]) Section 3, Algorithms 1 and 2.
  - $(ref_dict[:EPTail]) Section 2.2, Equations 3 to 6, and Section 2.3, Equations 7 to 11.
  - $(ref_dict[:EPRLVaR]) Section 2, Equations 1 to 5, and Section 3.2, Equation 7.
  - $(ref_dict[:meucciardiakeel2011])
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
    """
    $(field_dict[:x_src_pe])
    """
    x_src
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
                                 w::Option{<:ScenarioProbabilities},
                                 alg::AbstractEntropyPoolingAlgorithm,
                                 cache::Option{<:AbstractPartialFitState},
                                 x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :EntropyPoolingPrior)
        assert_inner_reader(pe, :pe, :EntropyPoolingPrior)
        assert_scenario_refit(pe)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
            w = scenario_probabilities(w, LinearAlgebra.normalize(w.values, 1))
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
                   typeof(sets), typeof(opt), typeof(w), typeof(alg), typeof(cache),
                   typeof(x_src)}(pe, mu_views, var_views, cvar_views, evar_views,
                                  rlvar_views, sigma_views, sk_views, kt_views, cov_views,
                                  rho_views, sets, opt, w, alg, cache, x_src)
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
                             w::Option{<:ScenarioProbabilities} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling(),
                             cache::Option{<:AbstractPartialFitState} = nothing,
                             x_src::Option{<:AbstractReturnsSource} = nothing)::EntropyPoolingPrior
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, evar_views, rlvar_views,
                               sigma_views, sk_views, kt_views, cov_views, rho_views, sets,
                               opt, w, alg, cache, x_src)
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
            :alg, :x_src)
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

 1. Fit the wrapped prior estimator with [`ep_wrapped_prior`](@ref), giving `pr` and the prior probabilities `w0` on its rows. The fit states the observation axis: `T` is `size(pr.X, 1)`, which a nested prior that drops rows makes smaller than `size(X, 1)`. The probabilities are `pe.w` where the caller set one, `pr.w` where the fit answered one, and the uniform `1/T` otherwise. A caller's `pe.w` reaches the wrapped estimator through [`prior_probability_factory`](@ref), which puts it on the rows that each weighted field reads, and `pr` is fitted under it. Before that fit, [`scenario_count`](@ref) reads `T` with only the steps that drop rows, so most wrapped priors need no fit to state it. Where the fit under `pe.w` answers its own `pr.w`, a nested prior that reweights its scenarios solved its views from `pe.w`, so `w0` is `pr.w`, the posterior of the nested prior.
 2. Pin each choice of `pe.pe` under a [`PinnedChoice`](@ref) at this fit with [`pin_prior_choice`](@ref), so every refit reads the choice that the views read. Every refit goes through [`scenario_prior`](@ref), which holds the model of this fit, its regression and its residual block included, so the refit reads the scenarios that the views read.
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
    # The nested prior is fitted first, and the prior probabilities are read on the rows
    # it answered. A uniform vector states no tilt, and the nested result's own `w` is
    # already carried by the fit that answered it, so neither is pushed. ADR 0116.
    pe, pr, w0 = ep_wrapped_prior(pe, X, F, pnl; strict = strict, kwargs...)
    # Every refit reads the cache of the fit that the views read (#1751).
    pr0 = pr
    w1 = w0
    # The views read the regressions of this fit. A choice under `PinnedChoice` is pinned
    # here, so every refit under the posterior weights reads the same choice (#1641).
    pe = pin_prior_choice(pe, X, F, nothing, pnl; strict = strict, kwargs...)
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
        # The posterior weights are probabilities, and the solver meets their sum to its
        # tolerance alone, so each solve normalises them. No constructor does it for the
        # refit: `scenario_prior` gives them to no entropy pooling prior (#1649).
        w1 = LinearAlgebra.normalize!(entropy_pooling(w0, epc, tvs, pe.opt), 1)
        pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                            cache_reg = true)
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
            ws = ifelse(isa(alg, H1_EntropyPooling), w0, w1)
            w1 = LinearAlgebra.normalize!(entropy_pooling(ws, epc, tvs, pe.opt), 1)
            pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                                cache_reg = true)
            vpr = investable_prior(imsk, pr)
        end
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            to_fix = ep_sk_views!(pe.sk_views, epc, vpr, vsets, scenario_probabilities(w1);
                                  strict = strict, ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, vpr)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            to_fix = ep_kt_views!(pe.kt_views, epc, vpr, vsets, scenario_probabilities(w1);
                                  strict = strict, ledger = led)
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
            ws = ifelse(isa(alg, H1_EntropyPooling), w0, w1)
            w1 = LinearAlgebra.normalize!(entropy_pooling(ws, epc, tvs, pe.opt), 1)
            pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                                cache_reg = true)
        end
    end
    announce_ep_departures(ni, led, isempty(epc) && isempty(tvs) && !isempty(led))
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same step in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr, ss_gaps) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, ss_gaps = ss_gaps,
                         pc = opinion_pooling_cache(pr))
end
"""
    ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing;
             strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing every view in one optimisation.

`ep_prior` builds every view constraint against the same prior and solves once. It is faster than the staged algorithms and pins nothing, so a higher moment view is free to move a lower moment.

# Algorithm

 1. Fit the wrapped prior estimator with [`ep_wrapped_prior`](@ref), giving `pr` and the prior probabilities `w0` on its rows. The fit states the observation axis: `T` is `size(pr.X, 1)`, which a nested prior that drops rows makes smaller than `size(X, 1)`. The probabilities are `pe.w` where the caller set one, `pr.w` where the fit answered one, and the uniform `1/T` otherwise. A caller's `pe.w` reaches the wrapped estimator through [`prior_probability_factory`](@ref), which puts it on the rows that each weighted field reads, and `pr` is fitted under it. Before that fit, [`scenario_count`](@ref) reads `T` with only the steps that drop rows, so most wrapped priors need no fit to state it. Where the fit under `pe.w` answers its own `pr.w`, a nested prior that reweights its scenarios solved its views from `pe.w`, so `w0` is `pr.w`, the posterior of the nested prior.
 2. Pin each choice of `pe.pe` under a [`PinnedChoice`](@ref) at this fit with [`pin_prior_choice`](@ref), so every refit reads the choice that the views read. Every refit goes through [`scenario_prior`](@ref), which holds the model of this fit, its regression and its residual block included, so the refit reads the scenarios that the views read.
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
    # See the note at the same step in the staged method: the nested prior is fitted
    # first, and the prior probabilities are read on the rows it answered. ADR 0116.
    pe, pr, w0 = ep_wrapped_prior(pe, X, F, pnl; strict = strict, kwargs...)
    # Every refit reads the cache of the fit that the views read (#1751).
    pr0 = pr
    # The views read the regressions of this fit. A choice under `PinnedChoice` is pinned
    # here, so every refit under the posterior weights reads the same choice (#1641).
    pe = pin_prior_choice(pe, X, F, nothing, pnl; strict = strict, kwargs...)
    # See the note at the same step in the staged method: every row is built on the
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
        ep_sk_views!(pe.sk_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    end
    # kurtosis
    if !isnothing(pe.kt_views)
        ep_kt_views!(pe.kt_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
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
        w1 = LinearAlgebra.normalize!(entropy_pooling(w0, epc, tvs, pe.opt), 1)
        pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                            cache_reg = true)
    end
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same step in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr, ss_gaps) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, ss_gaps = ss_gaps,
                         pc = opinion_pooling_cache(pr))
end
function factor_residual_config(pe::EntropyPoolingPrior)
    return factor_residual_config(pe.pe)
end
"""
    ep_wrapped_prior(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior},
                     X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                     strict::Bool = false, kwargs...)

Fit the prior that an entropy pooling prior wraps, and read the prior probabilities that its solve starts from.

A caller's `pe.w` states the probabilities of the scenarios that the wrapped prior answers, and the wrapped prior is fitted under them. A wrapped prior can answer fewer rows than it receives, so this function reads the count of its scenarios with [`scenario_count`](@ref) before the fit under `pe.w`. That count runs only the steps of the fit that drop rows. It needs no fit for a prior that [`keeps_observations`](@ref), a Scenario Cap, or a prior that wraps one of these, and it needs only the exposure stage of a [`CrossSectionalFactorPrior`](@ref). A prior that states no count is fitted once to read it.

# Algorithm

 1. When `pe.w` is `nothing`, fit `pe.pe`, giving `pr`. Read the prior probabilities `w0` from `pr` with [`ep_prior_probabilities`](@ref), and return `pe`, `pr` and `w0`.
 2. Otherwise find the count `T` of the scenarios of `pe.pe` with [`scenario_count`](@ref).
 3. Check `pe.w` against `T` with [`ep_prior_probabilities`](@ref), giving `w0`.
 4. Put `w0` on the rows that each weighted field of `pe.pe` reads with [`prior_probability_factory`](@ref), giving a new `pe`, and fit its `pe.pe`, giving `pr`.
 5. Return the new `pe`, `pr`, and `pr.w` through [`scenario_probabilities`](@ref) when it is not `nothing`, or `w0` otherwise. A wrapped prior that reweights its scenarios solved its own views from `w0`, and its posterior is the prior of this estimator.

Each fit of `pe.pe` passes `cache_reg = true`, so a prior that stores a cache gives it to the refits of [`scenario_prior`](@ref), which read it from `pr`.

# Arguments

  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, already oriented.
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Validation

  - The rules of [`ep_prior_probabilities`](@ref).
  - Under a caller's `pe.w`, the fit of `pe.pe` answers the `T` rows that [`scenario_count`](@ref) counted. A `DimensionMismatch` that names both counts is thrown otherwise.

# Returns

  - `pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}`: The estimator whose `pe.pe` answered `pr`.
  - `pr::AbstractPriorResult`: Prior result of the wrapped prior.
  - `w0::ScenarioProbabilities`: Prior probabilities, on the rows of `pr.X`.

# Related

  - [`scenario_count`](@ref)
  - [`keeps_observations`](@ref)
  - [`ep_prior_probabilities`](@ref)
  - [`prior_probability_factory`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_wrapped_prior(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior},
                          X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                          strict::Bool = false, kwargs...)
    # The refits under the posterior weights read the cache of this fit (#1751).
    if isnothing(pe.w)
        pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs..., cache_reg = true)
        return pe, pr, ep_prior_probabilities(nothing, pr, size(X, 1))
    end
    # A prior that reweights observations works on the observation axis its nested prior
    # ANSWERED, not on the axis it was handed: a nested prior may drop rows. The count runs
    # only the steps that drop rows, and most priors state it with no fit. See ADR 0116.
    T = scenario_count(pe.pe, X, F, pnl; strict = strict, kwargs...)
    w0 = ep_prior_probabilities(pe.w, T, size(X, 1))
    # A caller's prior probabilities weight the moments the nested estimator measures, so
    # the estimator is refitted under them. A field that reads more rows than the scenarios,
    # under a Scenario Cap, takes them with a zero weight on the rows the cap removed.
    pe = prior_probability_factory(pe, w0, X, F, pnl; kwargs...)
    pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs..., cache_reg = true)
    # The count and the fit are two authorities on one axis, so the fit must answer the
    # counted rows: `pe.w` names no row of a result on another axis (ADR 0116).
    @argcheck(size(pr.X, 1) == T,
              DimensionMismatch("the wrapped prior, a $(nameof(typeof(pe.pe))), was counted at $T scenarios and its fit under `w` answered $(size(pr.X, 1)). A method of `kept_scenarios`, `scenario_count` or `keeps_observations` for it states a count that its fit does not keep."))
    # A nested prior that reweights its scenarios solves its own views from the caller's
    # probabilities, and its posterior is the prior of this estimator, as it is when the
    # caller states none. A nested prior that answers no `w` keeps the caller's (ADR 0116).
    return pe, pr, scenario_probabilities(something(pr.w, w0))
end
"""
    embedded_pin_weights(pe::AbstractPriorEstimator, w)
    embedded_pin_weights(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}, w)

Return the observation weights of the fit of `pe.pe`, the prior that `pe` embeds, so that [`pin_prior_choice`](@ref) pins the choice of that fit.

A prior that embeds a prior fits it under the weights `w` that the prior around it gives it. An entropy pooling prior fits it under the caller's prior probabilities `pe.w`, builds its views on that fit, and refits under the posterior weights. Each refit reads the choice of that first fit. [`factory`](@ref) writes the weights of the outer prior into `pe.w`, so the weights of the first fit are `w` when `w` is set, and `pe.w` otherwise. They are not the `w0` of [`ep_prior`](@ref), which is the posterior of a nested prior that reweights its scenarios. When both are `nothing`, the fit is unweighted.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return `w`.
 2. An entropy pooling prior: return `w` when it is not `nothing`, and `pe.w` otherwise.

# Arguments

  - `pe`: The prior that embeds a prior in `pe.pe`.
  - $(arg_dict[:w_pin])

# Returns

  - `w::Option{<:AbstractWeights}`: The weights of the fit of `pe.pe`, or `nothing`.

# Related

  - [`pin_prior_choice`](@ref)
  - [`ep_prior_probabilities`](@ref)
  - [`PinnedChoice`](@ref)
"""
function embedded_pin_weights(::AbstractPriorEstimator, w)
    return w
end
function embedded_pin_weights(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior},
                              w)
    return isnothing(w) ? pe.w : w
end
"""
    scenario_factory(pe::AbstractPriorEstimator, w, pr::Option{<:AbstractPriorResult},
                     X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    scenario_factory(pe::EmpiricalPrior, w, pr::Option{<:AbstractPriorResult}, X::MatNum,
                     F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    scenario_factory(pe::Union{<:FactorPrior, <:FactorBlackLittermanPrior}, w,
                     pr::Option{<:AbstractPriorResult}, X::MatNum, F::MatNum,
                     pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    scenario_factory(pe::Union{<:BlackLittermanPrior, <:BayesianBlackLittermanPrior}, w,
                     pr::Option{<:AbstractPriorResult}, X::MatNum, F::Option{<:MatNum},
                     pnl::Option{<:AssetPanel}; kwargs...)
    scenario_factory(pe::AugmentedBlackLittermanPrior, w, pr::Option{<:AbstractPriorResult},
                     X::MatNum, F::MatNum, pnl::Option{<:AssetPanel}; strict::Bool = false,
                     kwargs...)
    scenario_factory(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}, w,
                     pr::Option{<:AbstractPriorResult}, X::MatNum, F::Option{<:MatNum},
                     pnl::Option{<:AssetPanel}; kwargs...)
    scenario_factory(pe::OpinionPoolingPrior, w, pr::Option{<:AbstractPriorResult},
                     X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    scenario_factory(pe::CrossSectionalFactorPrior, w, pr::Option{<:AbstractPriorResult},
                     X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                     strict::Bool = false, cache_reg::Bool = false, kwargs...)

Return the estimator that an entropy pooling prior refits under its posterior weights `w`.

Entropy pooling holds the scenarios and changes only their probabilities, so the posterior moments are the moments of the same scenarios under `w`. [`EntropyPoolingPrior`](@ref) and [`MeucciEntropyPoolingPrior`](@ref) build their views on the rows `pr.X` of the fit of `pe` under the prior probabilities. A prior that fits a time-series regression makes those rows from the regression, `F * transpose(M) .+ transpose(b)`. [`factory`](@ref) gives `w` to the regression too, so the refit fitted a new regression, and the rows that the views read moved under the posterior.

This function gives `w` to the estimators that read the scenarios, and holds the model that makes them. The regression `re` keeps the weights of the fit that the views read, so the refit reproduces the loadings, the intercepts and the rows of that fit. The variance estimator `ve` of the residual block keeps them too, so the residual block of the posterior is the block of that fit, which the views take off their targets with [`scenario_residual_variances`](@ref). Then the posterior moments are the moments of the model of the prior under `w`, and every mean view holds on the posterior mean. The same model is what [`FactorPrior`](@ref) answers over an entropy pooling prior of the factors, whose regression and residual block read no posterior weights.

# Mathematical definition

The rows of the fit under the prior probabilities are ``\\mathbf{X} = \\mathbf{F} \\mathbf{B}^\\intercal + \\mathbf{1} \\boldsymbol{\\alpha}^\\intercal``. Under the posterior probabilities ``\\boldsymbol{p}`` the refit answers

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}}_{\\boldsymbol{p}} + \\boldsymbol{\\alpha}\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_{f,\\,\\boldsymbol{p}} \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,,
\\end{align}
```

where ``\\mathbf{B}``, ``\\boldsymbol{\\alpha}`` and ``\\mathbf{\\Sigma}_\\varepsilon`` are those of the fit under the prior probabilities, and ``\\hat{\\boldsymbol{f}}_{\\boldsymbol{p}}`` and ``\\mathbf{\\Sigma}_{f,\\,\\boldsymbol{p}}`` are the factor moments under ``\\boldsymbol{p}``. When the factor prior reads the weighted sample mean, ``\\hat{\\boldsymbol{\\mu}} = \\mathbf{X}^\\intercal \\boldsymbol{p}``, which is the mean that the views state.

The result `pr` of the fit that the views read travels down with `pe`, and each method gives a nested prior the part of `pr` that its fit answered. A prior that reads a cache from it skips the steps that the cache holds. A `pr` with no cache, or `nothing`, changes nothing, and the method then reads the data.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return [`factory`](@ref) of `pe` under `w`. Its rows are the returns, which no weight moves.
 2. A [`FactorPrior`](@ref) or a [`FactorBlackLittermanPrior`](@ref): replace `pe.pe` with this function of `pe.pe` under `w` on the factor returns `F` alone and on the factor prior `pr.fpr` of the fit, as the fit of `pe` fits `pe.pe`, and keep every other field. The two factor priors keep `re` and `ve`.
 3. A [`BlackLittermanPrior`](@ref) or a [`BayesianBlackLittermanPrior`](@ref): replace `pe.pe` with this function of `pe.pe` under `w` on `pr`, `X`, `F` and `pnl`. Their `pe` is their only weighted field, and their result forwards every field of the result of `pe.pe`.
 4. An [`AugmentedBlackLittermanPrior`](@ref): replace the asset prior `a_pe` with this function of it on `pr`, `X` and `pnl`, and the factor prior `f_pe` with this function of it on `pr.fpr` and `F` alone, each under `w`. Keep `re`. Its result keeps the cache of the pool in `a_pe`, and its `fpr` the cache of the pool in `f_pe`.
 5. A nested entropy pooling prior: return this function of the estimator that fits its model, `pe.pe`, on `pr`, under [`prior_probability_factory`](@ref) of `pe.w` when the caller set one, which puts `pe.w` on the rows that each weighted field reads. Its result forwards the fields of the result of `pe.pe` that a cache reads. Its posterior is already the prior probabilities of the entropy pooling prior that embeds it, so its views are already in `w`, and [`factory`](@ref) would give it `w` as its own prior probabilities and solve its views again. This holds for an entropy pooling prior of the factors in step 2 too.
 6. An [`OpinionPoolingPrior`](@ref): read the pooled weights `wp` from the cache of `pr`, or pool the opinions on `X` with [`opinion_pooling_weights`](@ref) when `pr` holds no cache. Its own fit fits `pe.pe2` under `wp`. Return this function of [`factory`](@ref) of `pe.pe2` under `wp`, under `w`, on `pr`. The posterior of the pool is already in the prior probabilities of the entropy pooling prior, so [`factory`](@ref) of the pool under `w` pooled its opinions again from `w`, and a [`FactorPrior`](@ref) around it read the factor moments of other weights.
 7. An [`EmpiricalPrior`](@ref): return [`factory`](@ref) of `pe` under [`scenario_weights`](@ref) of `w` on the rows of `X`. A Scenario Cap keeps the last rows as scenarios, and `w` weighs these alone. Its moment estimators read every row, so the rows that the cap removed take a zero weight, and the posterior moments are those of the scenarios under `w`. Without a cap, `w` weighs every row and passes unchanged.
 8. A [`CrossSectionalFactorPrior`](@ref): take the factor returns of `pe.pe` from the block of `pr` with [`cross_sectional_cached_scenarios`](@ref) when [`cross_sectional_cached_block`](@ref) finds a cache, and with [`cross_sectional_factor_scenarios`](@ref) on `X`, `F` and `pnl` otherwise. Replace `pe.pe` with this function of `pe.pe` under `w` on these factor returns alone, as the fit of `pe` fits `pe.pe`, and keep every other field. The regression, the weights of the cross-sectional fit, the residual block and the Return Forecast keep the weights of the fit that the views read, so the scenarios of the refit are the scenarios of that fit. A Scenario Cap in `pe.pe` keeps the last factor returns as scenarios, and step 7 gives a zero weight to the rows that it removed.

Every step recurses through this function and never through [`factory`](@ref) of a prior that reweights scenarios, so no entropy pooling constructor reads `w` and normalises it a second time. Each step gives a nested prior the rows that the fit of `pe` gives it, so a pool computes `wp` on the rows of the fit that the views read. A pool with `pe1` refits on the rows of `pe1`, and no estimator that is fitted on `X` states that model. [`assert_scenario_model`](@ref) refuses such a pool when the entropy pooling prior is built, and [`scenario_prior`](@ref) refits a pool that the entropy pooling prior holds directly.

# Arguments

  - `pe`: The prior that an entropy pooling prior embeds.
  - `w`: The posterior weights.
  - `pr`: The result of the fit of `pe` that the views read, or `nothing`. A prior that stores a cache reads it from here.
  - $(arg_dict[:X])
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings. Default is `false`.
  - $(arg_dict[:cache_reg])
  - `kwargs...`: Additional keyword arguments, forwarded where the fit of `pe` forwards them.

# Returns

  - `pe::AbstractPriorEstimator`: The estimator to refit under `w`.

# Related

  - [`factory`](@ref)
  - [`scenario_prior`](@ref)
  - [`scenario_residual_variances`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`FactorPrior`](@ref)
"""
function scenario_factory(pe::AbstractPriorEstimator, w, ::Option{<:AbstractPriorResult},
                          ::MatNum, ::Option{<:MatNum}, ::Option{<:AssetPanel}; kwargs...)
    return factory(pe, w)
end
function scenario_factory(pe::EmpiricalPrior, w, ::Option{<:AbstractPriorResult}, X::MatNum,
                          ::Option{<:MatNum}, ::Option{<:AssetPanel}; kwargs...)
    return factory(pe, scenario_weights(w, size(X, 1)))
end
function scenario_factory(pe::Union{<:FactorPrior, <:FactorBlackLittermanPrior}, w,
                          pr::Option{<:AbstractPriorResult}, ::MatNum, F::MatNum,
                          ::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    inner = scenario_factory(pe.pe, w, scenario_factor_result(pr), F, nothing, nothing;
                             strict = strict)
    return rebuild_estimator(pe, (; pe = inner))
end
function scenario_factory(pe::Union{<:BlackLittermanPrior, <:BayesianBlackLittermanPrior},
                          w, pr::Option{<:AbstractPriorResult}, X::MatNum,
                          F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    return rebuild_estimator(pe,
                             (; pe = scenario_factory(pe.pe, w, pr, X, F, pnl; kwargs...)))
end
function scenario_factory(pe::AugmentedBlackLittermanPrior, w,
                          pr::Option{<:AbstractPriorResult}, X::MatNum, F::MatNum,
                          pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    a_pe = scenario_factory(pe.a_pe, w, pr, X, nothing, pnl; strict = strict, kwargs...)
    f_pe = scenario_factory(pe.f_pe, w, scenario_factor_result(pr), F, nothing, nothing;
                            strict = strict)
    return rebuild_estimator(pe, (; a_pe = a_pe, f_pe = f_pe))
end
function scenario_factory(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}, w,
                          pr::Option{<:AbstractPriorResult}, X::MatNum, F::Option{<:MatNum},
                          pnl::Option{<:AssetPanel}; kwargs...)
    # A caller's `pe.w` weighs the scenarios. Under a Scenario Cap the regression and the
    # residual block read more rows, so `factory` gave them too few weights.
    inner = if isnothing(pe.w)
        pe.pe
    else
        prior_probability_factory(pe.pe, pe.w, X, F, pnl; kwargs...)
    end
    return scenario_factory(inner, w, pr, X, F, pnl; kwargs...)
end
"""
    scenario_factor_result(pr::Option{<:AbstractPriorResult})
    scenario_factor_result(pr::LowOrderPrior)

Return the result of the factor prior of a fit, which a prior that holds a factor prior gives to [`scenario_factory`](@ref) of that factor prior.

A [`FactorPrior`](@ref) and a [`FactorBlackLittermanPrior`](@ref) fit their factor prior `pe.pe` on the factor returns, and their result carries the result of that fit in `fpr`. So a cache that the factor prior stores reaches the refit of the factor prior through `fpr`.

# Arguments

  - `pr`: The result of the fit, or `nothing`.

# Returns

  - `fpr::Option{<:AbstractPriorResult}`: `pr.fpr` of a [`LowOrderPrior`](@ref), and `nothing` for no result or a result of any other type.

# Related

  - [`scenario_factory`](@ref)
  - [`FactorPrior`](@ref)
"""
function scenario_factor_result(::Option{<:AbstractPriorResult})::Nothing
    return nothing
end
function scenario_factor_result(pr::LowOrderPrior)
    return pr.fpr
end
"""
    scenario_prior(pe::AbstractPriorEstimator, w, pr::Option{<:AbstractPriorResult},
                   X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    scenario_prior(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}, w,
                   pr::Option{<:AbstractPriorResult}, X::MatNum, F::Option{<:MatNum},
                   pnl::Option{<:AssetPanel}; kwargs...)
    scenario_prior(pe::BlackLittermanPrior, w, pr::Option{<:AbstractPriorResult}, X::MatNum,
                   F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; strict::Bool = false,
                   kwargs...)
    scenario_prior(pe::BayesianBlackLittermanPrior, w, pr::Option{<:AbstractPriorResult},
                   X::MatNum, F::MatNum, pnl::Option{<:AssetPanel}; strict::Bool = false,
                   kwargs...)
    scenario_prior(pe::OpinionPoolingPrior, w, pr::Option{<:AbstractPriorResult},
                   X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    scenario_prior(pe::CrossSectionalFactorPrior, w, pr::Option{<:AbstractPriorResult},
                   X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                   strict::Bool = false, cache_reg::Bool = false, kwargs...)

Refit the prior that an entropy pooling prior embeds, with its scenarios under the posterior weights `w`.

[`EntropyPoolingPrior`](@ref) and [`MeucciEntropyPoolingPrior`](@ref) call this after each solve. Most priors are an estimator whose model [`scenario_factory`](@ref) holds, and the refit is that estimator under `w`. A prior that reweights scenarios itself has no such estimator. Its posterior is already the prior probabilities of the entropy pooling prior that embeds it, so its own views or opinions are already in `w`. A refit through [`factory`](@ref) gave `w` to it as its own prior probabilities, and it solved its views or pooled its opinions again from `w`. The weights then moved away from `w`, and the prior result reported the moments of other weights than the `w` that the result carries. This function refits the model of such a prior under `w` and runs no view and no opinion.

The result `pr` of the fit that the views read travels down with `pe`, as it does in [`scenario_factory`](@ref). A prior that the entropy pooling prior fitted with `cache_reg = true` reads its cache from `pr`, and runs only the steps that read the new weights. A `pr` with no cache, or `nothing`, changes nothing.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: fit [`scenario_factory`](@ref) of `pe` under `w` on `pr`, `X`, `F` and `pnl`.
 2. A nested entropy pooling prior: apply this function to the estimator that fits its model, on `pr`, as step 5 of [`scenario_factory`](@ref) does. This method exists because that model can be an [`OpinionPoolingPrior`](@ref).
 3. A [`BlackLittermanPrior`](@ref) or a [`BayesianBlackLittermanPrior`](@ref): refit `pe.pe` with this function under `w` on `pr`, and blend the views into its result, with [`bl_posterior`](@ref) or [`bbl_posterior`](@ref). The result of the fit forwards every field of the result of `pe.pe`, so `pr` holds the cache of `pe.pe`. It is step 1 for this prior, with the refit of `pe.pe` through this function.
 4. An [`OpinionPoolingPrior`](@ref): take the rows of `pe.pe1` when it is set, and the returns otherwise, as its own fit does, and the pooled weights `wp` of the opinions with [`opinion_pooling_weights`](@ref), under which its own fit fits `pe.pe2`. A `pr` whose cache holds the rows and the pooled weights gives both, so neither `pe.pe1` nor an opinion runs again. Apply this function to [`factory`](@ref) of `pe.pe2` under `wp`, under `w`, on those rows and on `pr`. The opinions run for `wp` alone, and their views do not reach the result.
 5. A [`CrossSectionalFactorPrior`](@ref): when [`cross_sectional_cached_block`](@ref) finds a cache on `pr`, take the factor returns of the block with [`cross_sectional_cached_scenarios`](@ref), fit [`scenario_factory`](@ref) of `pe.pe` under `w` on them with [`cross_sectional_factor_moments`](@ref), split the Return Forecast of the block with [`cross_sectional_forecast_split`](@ref), and build the result with [`cross_sectional_moment_prior`](@ref) on the block, the scenarios `pr.X` and the base-currency returns `pr.o_X` of the fit. It runs no regression. With no cache, fit the regression once with [`cross_sectional_factor_scenarios`](@ref), fit the factor prior on its factor returns in the same way, and assemble the result from the stage with [`cross_sectional_assemble`](@ref) under `cache_reg`.

# Arguments

  - `pe`: The prior that an entropy pooling prior embeds.
  - `w`: The posterior weights.
  - `pr`: The result of the fit of `pe` that the views read, or `nothing`. A prior that stores a cache reads it from here.
  - $(arg_dict[:X])
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings. Default is `false`.
  - $(arg_dict[:cache_reg])
  - `kwargs...`: Additional keyword arguments forwarded to [`prior`](@ref).

# Returns

  - `pr::AbstractPriorResult`: The refitted prior result.

# Related

  - [`scenario_factory`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
function scenario_prior(pe::AbstractPriorEstimator, w, pr::Option{<:AbstractPriorResult},
                        X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                        kwargs...)
    return prior(scenario_factory(pe, w, pr, X, F, pnl; kwargs...), X, F, pnl; kwargs...)
end
function scenario_prior(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}, w,
                        pr::Option{<:AbstractPriorResult}, X::MatNum, F::Option{<:MatNum},
                        pnl::Option{<:AssetPanel}; kwargs...)
    # See the note at the same step in `scenario_factory`.
    inner = if isnothing(pe.w)
        pe.pe
    else
        prior_probability_factory(pe.pe, pe.w, X, F, pnl; kwargs...)
    end
    return scenario_prior(inner, w, pr, X, F, pnl; kwargs...)
end
# The views of a Black-Litterman prior blend into the result of its prior, so the refit
# refits that prior through `scenario_prior`, which reads its cache from `pr` (#1751).
function scenario_prior(pe::BlackLittermanPrior, w, pr::Option{<:AbstractPriorResult},
                        X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                        strict::Bool = false, kwargs...)
    assert_bl_views_axis(pe, X)
    inner = scenario_prior(pe.pe, w, pr, X, F, pnl; strict = strict, kwargs...)
    return bl_posterior(pe, inner, strict; kwargs...)
end
function scenario_prior(pe::BayesianBlackLittermanPrior, w,
                        pr::Option{<:AbstractPriorResult}, X::MatNum, F::MatNum,
                        pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    assert_bbl_views_axis(pe, F)
    inner = scenario_prior(pe.pe, w, pr, X, F, pnl; strict = strict, kwargs...)
    return bbl_posterior(pe, inner, F, strict; kwargs...)
end
"""
    prior_probability_factory(pe::AbstractPriorEstimator, w, X::MatNum,
                              F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::EmpiricalPrior, w, X::MatNum, F::Option{<:MatNum},
                              pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::Union{<:FactorPrior, <:FactorBlackLittermanPrior}, w,
                              X::MatNum, F::MatNum, pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::Union{<:BlackLittermanPrior,
                                        <:BayesianBlackLittermanPrior,
                                        <:EntropyPoolingPrior,
                                        <:MeucciEntropyPoolingPrior}, w, X::MatNum,
                              F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::AugmentedBlackLittermanPrior, w, X::MatNum, F::MatNum,
                              pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::CrossSectionalFactorPrior, w, X::MatNum,
                              F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::OpinionPoolingPrior, w, X::MatNum,
                              F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_probability_factory(pe::AbstractPriorEstimator, w, n::Integer)
    prior_probability_factory(pe::EmpiricalPrior, w, n::Integer)
    prior_probability_factory(pe::Union{<:BlackLittermanPrior,
                                        <:BayesianBlackLittermanPrior,
                                        <:EntropyPoolingPrior,
                                        <:MeucciEntropyPoolingPrior}, w, n::Integer)
    prior_probability_factory(pe::OpinionPoolingPrior, w, n::Integer)

Return [`factory`](@ref) of `pe` under the prior probabilities `w` of the scenarios, with each weighted field on the rows that it reads.

[`EntropyPoolingPrior`](@ref) and [`MeucciEntropyPoolingPrior`](@ref) state a caller's `w` on the scenarios of the fit of the prior that they embed, and refit that prior under it. A Scenario Cap keeps the last rows of the observations as scenarios, and the moment estimators and the regression read every observation. [`factory`](@ref) gives the same `w` to every field, so these fields received fewer weights than rows. This function gives each such field [`scenario_weights`](@ref) of `w` on its rows, and every other field `w` as [`factory`](@ref) does. A nested entropy pooling prior keeps `w` as its own prior probabilities, on its scenarios.

A prior that reads no Asset Panel reads its data here only for the number of its rows. The methods that take the number of rows `n` in place of the data hold its rules, and every method on the data of such a prior answers through them. A factor prior is fitted on the factor returns alone, so a prior that holds one gives it the number of rows of its factor returns. A [`CrossSectionalFactorPrior`](@ref) reads that number off its exposure stage, so it runs no regression to count the rows.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return [`factory`](@ref) of `pe` under `w`. On the data, return this function of `pe` on the number of rows of `X`.
 2. An [`EmpiricalPrior`](@ref): return [`factory`](@ref) of `pe` under [`scenario_weights`](@ref) of `w` on the `n` rows.
 3. A [`FactorPrior`](@ref) or a [`FactorBlackLittermanPrior`](@ref): take [`factory`](@ref) of `pe` under [`scenario_weights`](@ref) of `w` on the rows of `F`, which the regression and the residual block read. Replace its `pe` with this function of `pe.pe` under `w` on the number of rows of `F`.
 4. A [`BlackLittermanPrior`](@ref), a [`BayesianBlackLittermanPrior`](@ref) or a nested entropy pooling prior: take [`factory`](@ref) of `pe` under `w`, and replace its `pe` with this function of `pe.pe` under `w` on `X`, `F` and `pnl`, or on the `n` rows.
 5. An [`AugmentedBlackLittermanPrior`](@ref): take [`factory`](@ref) of `pe` under [`scenario_weights`](@ref) of `w` on the rows of `X`. Replace `a_pe` with this function of it on `X` and `pnl`, and `f_pe` with this function of it on the number of rows of `F`.
 6. A [`CrossSectionalFactorPrior`](@ref): take the rows `rw` after the Descriptor warm-up with [`cross_sectional_exposure_stage`](@ref), and check them against the exposure lag with [`assert_cross_sectional_lag_rows`](@ref). The regression answers `n = length(rw) - pe.lag` factor returns. Take [`factory`](@ref) of `pe` under [`scenario_weights`](@ref) of `w` on the `n` rows, which the regression, the residual block and every other weighted field read. Replace its `pe` with this function of `pe.pe` under `w` on the `n` rows.
 7. An [`OpinionPoolingPrior`](@ref): take [`factory`](@ref) of `pe` under `w`, which gives `w` to the opinions and to `pe2` on the scenarios of the pool. When `pe1` is set, replace it with this function of `pe1` under `w` on `X`, `F` and `pnl`, or on the `n` rows, because a Scenario Cap in `pe1` keeps fewer rows than it reads.

# Arguments

  - `pe`: The prior that an entropy pooling prior embeds.
  - `w`: The prior probabilities of the scenarios.
  - $(arg_dict[:X])
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `n`: The number of rows that the fit of `pe` reads, in place of the data.
  - `kwargs...`: Additional keyword arguments of the fit of `pe`, forwarded where a nested prior reads `X` and `pnl`.

# Validation

  - Everything [`scenario_weights`](@ref) refuses.

# Returns

  - `pe::AbstractPriorEstimator`: The estimator to refit under `w`.

# Related

  - [`factory`](@ref)
  - [`scenario_factory`](@ref)
  - [`scenario_weights`](@ref)
  - [`ep_prior_probabilities`](@ref)
"""
function prior_probability_factory(pe::AbstractPriorEstimator, w, X::MatNum,
                                   ::Option{<:MatNum}, ::Option{<:AssetPanel}; kwargs...)
    return prior_probability_factory(pe, w, size(X, 1))
end
function prior_probability_factory(pe::AbstractPriorEstimator, w, ::Integer)
    return factory(pe, w)
end
function prior_probability_factory(pe::EmpiricalPrior, w, n::Integer)
    return factory(pe, scenario_weights(w, n))
end
function prior_probability_factory(pe::Union{<:FactorPrior, <:FactorBlackLittermanPrior}, w,
                                   ::MatNum, F::MatNum, ::Option{<:AssetPanel}; kwargs...)
    inner = prior_probability_factory(pe.pe, w, size(F, 1))
    return rebuild_estimator(factory(pe, scenario_weights(w, size(F, 1))), (; pe = inner))
end
function prior_probability_factory(pe::Union{<:BlackLittermanPrior,
                                             <:BayesianBlackLittermanPrior,
                                             <:EntropyPoolingPrior,
                                             <:MeucciEntropyPoolingPrior}, w, X::MatNum,
                                   F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                   kwargs...)
    inner = prior_probability_factory(pe.pe, w, X, F, pnl; kwargs...)
    return rebuild_estimator(factory(pe, w), (; pe = inner))
end
function prior_probability_factory(pe::Union{<:BlackLittermanPrior,
                                             <:BayesianBlackLittermanPrior,
                                             <:EntropyPoolingPrior,
                                             <:MeucciEntropyPoolingPrior}, w, n::Integer)
    return rebuild_estimator(factory(pe, w),
                             (; pe = prior_probability_factory(pe.pe, w, n)))
end
function prior_probability_factory(pe::AugmentedBlackLittermanPrior, w, X::MatNum,
                                   F::MatNum, pnl::Option{<:AssetPanel}; kwargs...)
    a_pe = prior_probability_factory(pe.a_pe, w, X, nothing, pnl; kwargs...)
    f_pe = prior_probability_factory(pe.f_pe, w, size(F, 1))
    return rebuild_estimator(factory(pe, scenario_weights(w, size(X, 1))),
                             (; a_pe = a_pe, f_pe = f_pe))
end
"""
    assert_scenario_model(pe::AbstractPriorEstimator)
    assert_scenario_model(pe::Union{<:FactorPrior, <:FactorBlackLittermanPrior,
                                    <:BlackLittermanPrior, <:BayesianBlackLittermanPrior,
                                    <:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior})
    assert_scenario_model(pe::AugmentedBlackLittermanPrior)
    assert_scenario_model(pe::OpinionPoolingPrior)
    assert_scenario_model(pe::CrossSectionalFactorPrior)
    assert_scenario_model(pe::LowDimensionEnsemblePrior)

Check that [`scenario_factory`](@ref) can state the model of `pe` as one estimator.

[`scenario_factory`](@ref) returns the estimator that the refit of an entropy pooling prior fits on the rows of `pe`. An [`OpinionPoolingPrior`](@ref) with `pe1` refits `pe2` on the rows of `pe1`, which differ from the rows of `pe`, so no such estimator states its model. A [`LowDimensionEnsemblePrior`](@ref) reads no observation weights, so its refit under the posterior weights is the fit under the prior probabilities, and the views do not reach its moments. The check walks the priors that [`scenario_factory`](@ref) walks, and refuses each of the two, so the entropy pooling prior fails when it is built and not at the refit.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return `nothing`.
 2. A prior that [`scenario_factory`](@ref) rebuilds around `pe.pe`: apply this function to `pe.pe`.
 3. An [`AugmentedBlackLittermanPrior`](@ref): apply this function to `pe.a_pe` and to `pe.f_pe`.
 4. An [`OpinionPoolingPrior`](@ref): check that `pe.pe1` is `nothing`, and apply this function to `pe.pe2`.
 5. A [`LowDimensionEnsemblePrior`](@ref): throw.

# Arguments

  - `pe`: The prior that [`scenario_factory`](@ref) rebuilds.

# Validation

  - Every [`OpinionPoolingPrior`](@ref) that the walk meets has no `pe1`. An `ArgumentError` is thrown otherwise.
  - The walk meets no [`LowDimensionEnsemblePrior`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`assert_scenario_refit`](@ref)
  - [`scenario_factory`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
function assert_scenario_model(::AbstractPriorEstimator)
    return nothing
end
function assert_scenario_model(pe::Union{<:FactorPrior, <:FactorBlackLittermanPrior,
                                         <:BlackLittermanPrior,
                                         <:BayesianBlackLittermanPrior,
                                         <:EntropyPoolingPrior,
                                         <:MeucciEntropyPoolingPrior})
    return assert_scenario_model(pe.pe)
end
function assert_scenario_model(pe::AugmentedBlackLittermanPrior)
    assert_scenario_model(pe.a_pe)
    return assert_scenario_model(pe.f_pe)
end
"""
    assert_scenario_refit(pe::AbstractPriorEstimator)
    assert_scenario_refit(pe::Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior})
    assert_scenario_refit(pe::OpinionPoolingPrior)

Check that [`scenario_prior`](@ref) can refit `pe`, the prior that an entropy pooling prior embeds, under its posterior weights.

The constructors of [`EntropyPoolingPrior`](@ref) and [`MeucciEntropyPoolingPrior`](@ref) call this on their `pe`. The check walks the priors that [`scenario_prior`](@ref) walks, and applies [`assert_scenario_model`](@ref) where [`scenario_prior`](@ref) calls [`scenario_factory`](@ref). [`scenario_prior`](@ref) refits a pool that the entropy pooling prior holds directly on the rows of its `pe1`, so this function lets such a pool keep its `pe1`.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: apply [`assert_scenario_model`](@ref) to `pe`.
 2. A nested entropy pooling prior: apply this function to `pe.pe`.
 3. An [`OpinionPoolingPrior`](@ref): apply this function to `pe.pe2`.

# Arguments

  - `pe`: The prior that an entropy pooling prior embeds.

# Validation

  - The rules of [`assert_scenario_model`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_scenario_model`](@ref)
  - [`scenario_prior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function assert_scenario_refit(pe::AbstractPriorEstimator)
    return assert_scenario_model(pe)
end
function assert_scenario_refit(pe::Union{<:EntropyPoolingPrior,
                                         <:MeucciEntropyPoolingPrior})
    return assert_scenario_refit(pe.pe)
end

export EntropyPoolingPrior
