"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments meet a set of views, and root-finds a CVaR view.

`MeucciEntropyPoolingPrior` is a low order prior estimator. It calculates the mean and the covariance of the asset returns with entropy pooling. It takes views on the mean, the value at risk, the conditional value at risk, the variance, the covariance, the correlation, the skewness and the kurtosis. It also takes prior weights and the configuration of the solvers.

This estimator is the earlier of the two entropy pooling estimators of the library. The library keeps it because its CVaR route is a different algorithm, not a different formulation of the same problem. A CVaR view goes through the recursive algorithm of [meucciardiakeel2011](@citet). The search of this library finds the value at risk, and it solves the whole entropy pooling problem again at each candidate. For one view, [`ConditionalValueatRiskEntropyPooling`](@ref) configures that search. This route takes only equality CVaR views, with one asset in each view.

[EPTail](@cite) states that the recursive algorithm takes only equality views. The staged algorithms come from [vorobets2021](@cite). That work puts the skewness and the kurtosis in the class of parameters that need a fixed mean and a fixed variance.

Use [`EntropyPoolingPrior`](@ref) when a tail view must be an inequality, must name two assets, or must state the entropic or the relativistic value at risk. That estimator writes each tail view as a constraint of one entropy pooling problem, so one solve meets every view.

The comparison operators that a view accepts depend on the moment that it constrains. `var_views` takes `==` and `>=`. `cvar_views` takes only `==`. Every other view family takes `==`, `>=` and `<=`. An unsupported operator raises a `Meta.ParseError` that names the operators that the view accepts.

A skewness, kurtosis or correlation view divides by the variance of an asset. A staged algorithm holds that variance at its value from the stage before. Both read the variance from the diagonal of the `sigma` that `pe` fits, but the rows state the uncorrected moment of the scenarios. The default covariance estimator is corrected. So under [`H1_EntropyPooling`](@ref) or [`H2_EntropyPooling`](@ref), a skewness or kurtosis view moves the variance of the scenarios up by the correction factor of the stage before. To hold the uncorrected variance and divide by it, give `pe` an uncorrected covariance estimator, as [`EntropyPoolingPrior`](@ref) shows.

A `prior(...)` reference in a skewness or kurtosis view reads the sample statistic under the observation weights of the prior of the stage.

!!! warning

    The estimator does not detect an infeasible view set. The entropy pooling solve and the CVaR search both miss it. The dual of an infeasible set is unbounded, so the minimiser does not stop at a finite point. The posterior puts almost all of its mass on one observation, and `Optim` reports the solve as converged. The residual of the CVaR search is small on such a posterior, so the search also reports success. [`entropy_pooling`](@ref) states the mechanism.

    Read the result, not the flag. On such a result, `ens` is a small number against the number of observations, and one weight is near one. `kld` is large, and the posterior statistic of the view is far from its target. A frequent cause is two views that move one asset in opposite directions. An example is a `sigma_views` row that decreases the variance of an asset, beside a `cvar_views` row that increases the tail loss of the same asset. The same pair on two different assets is feasible and solves correctly, so the cause is the direction and not the pair.

    Sometimes the dual overflows before the search stops, and the view set raises an error instead of a result. On the staged route, the moment estimators get weights that are not finite, and they raise an `ArgumentError` that names Infs or NaNs. On the single-shot route, the CVaR search raises its own `ErrorException`. Neither error names the infeasibility, because both come from the same unbounded dual at a later point. The outcome of a view set is not stable, and a small change in the sequence of candidate value at risk levels can change it. Treat the three outcomes as one result, and read the views, not the message.

    To refuse an infeasible set of moment views before the solve, give `diagnose` a solver of linear programmes. [`ep_diagnose_views`](@ref) then decides with one linear programme for each stage whether the `mu_views`, `var_views`, `sigma_views`, `cov_views`, `sk_views`, `kt_views` and `rho_views` rows can hold together, and its error names the view fields. It does not read `cvar_views`, because the CVaR search makes its rows only during the solve.

!!! warning "The search can miss a feasible view with no error"

    The CVaR search checks its own variable, not the view. The view is a constraint of an inner solve, and the search does not read how closely that solve meets it. With one view, `Roots.find_zero` finds the root of the posterior tail mass minus `alpha` over the candidate value at risk. It stops when that residual is small. With more than one view, the guard is `Optim.converged`, which accepts a solve that stopped on the step in `x` and not on stationarity. So a feasible view set can return a posterior that misses its target, with no error.

    This result is different from the result of the warning above. `ens` is normal, `kld` is small, and only the statistic of the view misses its target. The size of the miss is a property of the run, not of the estimator, and the miss does not repeat. A case that meets its target closely in a process of its own can miss it by orders of magnitude after other cases in the same process. In one measurement, the residual also changed after an edit to a comment in this file. So the search is not the only cause, and the cause is not known.

    Read the result, not the flag. Measure the posterior with [`ConditionalValueatRisk`](@ref) at the `alpha` of the view, and compare it with the target of the view. For a result that repeats, give a tighter `Optim.Options` in the `args` of the optimiser. [`OptimEntropyPooling`](@ref) shows the block. The library reads no tolerance on the residual, because a threshold that decides if a solve succeeded is a policy that this library does not set.

# Algorithm

The constructor derives the prior probabilities and validates every other field.

 1. When `w` is `nothing`, derive nothing. At solve time, [`prior`](@ref) uses the `w` of the wrapped result, or the uniform weights `1/T`, one for each observation, when that result has none.
 2. When `w` is not `nothing`, normalise it to sum to one. The normalised weights are the prior probabilities that the pooling starts from. The caller's `w` does not change, because new weights of the same kind hold the normalised values, through [`scenario_probabilities`](@ref). The kind fixes the bias correction of the posterior covariance. [`ScenarioProbabilities`](@ref) states the rule.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeucciEntropyPoolingPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        mu_views::Option{<:LinearConstraintEstimator} = nothing,
        var_views::Option{<:VV_VecVV} = nothing,
        cvar_views::Option{<:CVV_VecCVV} = nothing,
        sigma_views::Option{<:LinearConstraintEstimator} = nothing,
        sk_views::Option{<:LinearConstraintEstimator} = nothing,
        kt_views::Option{<:LinearConstraintEstimator} = nothing,
        cov_views::Option{<:LinearConstraintEstimator} = nothing,
        rho_views::Option{<:LinearConstraintEstimator} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        ds_opt::Option{<:ConditionalValueatRiskEntropyPooling} = nothing,
        dm_opt::Option{<:OptimEntropyPooling} = nothing,
        opt::NonCVaREP = OptimEntropyPooling(),
        diagnose::Option{<:Slv_VecSlv} = nothing,
        w::Option{<:ScenarioProbabilities} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling(),
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> MeucciEntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If a view field is not `nothing`, `sets` must not be `nothing`.
  - If `var_views` is a vector, it must not be empty.
  - If `diagnose` is a vector, it must not be empty.
  - If `w` is not `nothing`, it must not be empty. At solve time, its length must equal the number of observations of the wrapped result.
  - [`assert_scenario_refit`](@ref) checks that the refit under the posterior weights can state the model of `pe`. It refuses an [`OpinionPoolingPrior`](@ref) whose `pe1` is inside a factor prior or a Black-Litterman prior of `pe`.
  - `x_src` must be `nothing`, `DataSource()` or `ExcessSource()`. [`assert_returns_source`](@ref) checks it.
  - The `x_src` of `pe` must be `nothing`, because this prior selects the returns that it reads. [`assert_inner_reader`](@ref) checks it.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated through [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pe`: Recursively viewed through [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `pe`: Recursively indexed through [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> MeucciEntropyPoolingPrior(;
                                 sets = UniverseSets(; xkey = \"nx\",
                                                     dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                                 mu_views = LinearConstraintEstimator(;
                                                                      val = [\"A == 0.03\",
                                                                             \"B + C == 0.04\"]))
MeucciEntropyPoolingPrior
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
       ds_opt ┼ nothing
       dm_opt ┼ nothing
          opt ┼ OptimEntropyPooling
              │     args ┼ Tuple{}: ()
              │   kwargs ┼ @NamedTuple{}: NamedTuple()
              │      sc1 ┼ Int64: 1
              │      sc2 ┼ Float64: 1000.0
              │      alg ┼ ExpEntropyPooling()
              │      err ┴ nothing
     diagnose ┼ nothing
            w ┼ nothing
          alg ┼ H1_EntropyPooling()
        x_src ┴ nothing
```

## The incremental fit

This prior has no exact incremental recursion, and it takes the online step by a **refit from a sample buffer**. [`Online`](@ref) seeds `cache`, and [`partial_fit!`](@ref) adds each observation to it with no change. The [`prior`](@ref) method with one argument then runs the batch verb of this estimator over the rows that the buffer kept. The result is the batch fit over those rows, and a `max_history` on the wrapper sets the window of the whole fit.

`cache` goes through the three propagation channels as every partial-fit state does. [`factory`](@ref) carries it with no change, and [`port_opt_view`](@ref) cuts it to the selected assets. [`obs_weights_view`](@ref) drops it, because a state has no cut on the observation axis. The show method does not print `cache`, because a buffer that changes at each step is not part of the configuration.

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 25 to 27, and Appendix A.2, Equations 65 to 73.
  - $(ref_dict[:meucciardiakeel2011]) Pages 39–49. $(ref_dict[:no_original_source])
  - $(ref_dict[:EPTail]) Section 2.2.
  - $(ref_dict[:vorobets2021]) Section 3.
"""
@propagatable @concrete struct MeucciEntropyPoolingPrior <:
                               AbstractLowOrderPriorEstimator_AF
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
    $(field_dict[:ds_opt])
    """
    ds_opt
    """
    $(field_dict[:dm_opt])
    """
    dm_opt
    """
    $(field_dict[:opt_ep])
    """
    opt
    """
    $(field_dict[:ep_diagnose])
    """
    diagnose
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
    function MeucciEntropyPoolingPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                       mu_views::Option{<:LinearConstraintEstimator},
                                       var_views::Option{<:VV_VecVV},
                                       cvar_views::Option{<:CVV_VecCVV},
                                       sigma_views::Option{<:LinearConstraintEstimator},
                                       sk_views::Option{<:LinearConstraintEstimator},
                                       kt_views::Option{<:LinearConstraintEstimator},
                                       cov_views::Option{<:LinearConstraintEstimator},
                                       rho_views::Option{<:LinearConstraintEstimator},
                                       sets::Option{<:UniverseSets},
                                       ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                                       dm_opt::Option{<:OptimEntropyPooling},
                                       opt::NonCVaREP, diagnose::Option{<:Slv_VecSlv},
                                       w::Option{<:ScenarioProbabilities},
                                       alg::AbstractEntropyPoolingAlgorithm,
                                       cache::Option{<:AbstractPartialFitState},
                                       x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :MeucciEntropyPoolingPrior)
        assert_inner_reader(pe, :pe, :MeucciEntropyPoolingPrior)
        assert_scenario_refit(pe)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
            w = scenario_probabilities(w, LinearAlgebra.normalize(w.values, 1))
        end
        if !isnothing(mu_views) ||
           !isnothing(var_views) ||
           !isnothing(cvar_views) ||
           !isnothing(sigma_views) ||
           !isnothing(sk_views) ||
           !isnothing(kt_views) ||
           !isnothing(cov_views) ||
           !isnothing(rho_views)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(var_views, AbstractVector)
            @argcheck(!isempty(var_views), IsEmptyError("var_views cannot be empty"))
        end
        if isa(diagnose, VecSlv)
            @argcheck(!isempty(diagnose), IsEmptyError("diagnose cannot be empty"))
        end
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(sigma_views), typeof(sk_views), typeof(kt_views),
                   typeof(cov_views), typeof(rho_views), typeof(sets), typeof(ds_opt),
                   typeof(dm_opt), typeof(opt), typeof(diagnose), typeof(w), typeof(alg),
                   typeof(cache), typeof(x_src)}(pe, mu_views, var_views, cvar_views,
                                                 sigma_views, sk_views, kt_views, cov_views,
                                                 rho_views, sets, ds_opt, dm_opt, opt,
                                                 diagnose, w, alg, cache, x_src)
    end
end
function MeucciEntropyPoolingPrior(;
                                   pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                                   mu_views::Option{<:LinearConstraintEstimator} = nothing,
                                   var_views::Option{<:VV_VecVV} = nothing,
                                   cvar_views::Option{<:CVV_VecCVV} = nothing,
                                   sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                                   sk_views::Option{<:LinearConstraintEstimator} = nothing,
                                   kt_views::Option{<:LinearConstraintEstimator} = nothing,
                                   cov_views::Option{<:LinearConstraintEstimator} = nothing,
                                   rho_views::Option{<:LinearConstraintEstimator} = nothing,
                                   sets::Option{<:UniverseSets} = nothing,
                                   ds_opt::Option{<:ConditionalValueatRiskEntropyPooling} = nothing,
                                   dm_opt::Option{<:OptimEntropyPooling} = nothing,
                                   opt::NonCVaREP = OptimEntropyPooling(),
                                   diagnose::Option{<:Slv_VecSlv} = nothing,
                                   w::Option{<:ScenarioProbabilities} = nothing,
                                   alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling(),
                                   cache::Option{<:AbstractPartialFitState} = nothing,
                                   x_src::Option{<:AbstractReturnsSource} = nothing)::MeucciEntropyPoolingPrior
    return MeucciEntropyPoolingPrior(pe, mu_views, var_views, cvar_views, sigma_views,
                                     sk_views, kt_views, cov_views, rho_views, sets, ds_opt,
                                     dm_opt, opt, diagnose, w, alg, cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give every field of a [`MeucciEntropyPoolingPrior`](@ref) to print, except `cache`.

The state in `cache` changes at each step of an incremental fit, and it is not part of the configuration. Each site that prints the estimator also prints this state, unless this method leaves it out. Set `set_show_nothing_fields!(:MeucciEntropyPoolingPrior, true)` to print it.

# Arguments

  - `::MeucciEntropyPoolingPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :mu_views, :var_views, :cvar_views, :sigma_views, :sk_views, :kt_views, :cov_views, :rho_views, :sets, :ds_opt, :dm_opt, :opt, :diagnose, :w, :alg, :x_src)`.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::MeucciEntropyPoolingPrior)
    return (:pe, :mu_views, :var_views, :cvar_views, :sigma_views, :sk_views, :kt_views,
            :cov_views, :rho_views, :sets, :ds_opt, :dm_opt, :opt, :diagnose, :w, :alg,
            :x_src)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties MeucciEntropyPoolingPrior begin
    forward(pe, me, ce)
end
"""
    const VecMeucciEP = AbstractVector{<:MeucciEntropyPoolingPrior}

Groups the vector forms of [`MeucciEntropyPoolingPrior`](@ref) that a method can dispatch on.

A caller that pools opinions holds many estimators, and a method that takes such a collection needs one type in its signature. The opinion pooling of the library uses the wider [`VecEP`](@ref), which also admits [`EntropyPoolingPrior`](@ref). `VecMeucciEP` is the narrower name for a method that takes only the recursive CVaR route.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref): the element type of the vector.
  - [`VecEP`](@ref): the wider group in the signature of [`OpinionPoolingPrior`](@ref), which admits both entropy pooling estimators.
"""
const VecMeucciEP = AbstractVector{<:MeucciEntropyPoolingPrior}
"""
    ep_cvar_views_setup(cvar_views::Nothing, args...; kwargs...)

Return `nothing` when the estimator has no conditional value at risk view.

# Arguments

  - `cvar_views::Nothing`: Shows that the estimator has no CVaR view.
  - `args...`: More positional arguments, which the method does not read.
  - `kwargs...`: More keyword arguments, which the method does not read.

# Returns

  - `nothing`: There is no search to prepare. [`ep_cvar_views_solve!`](@ref) dispatches on this value.

# Related

  - [`ep_cvar_views_setup`](@ref)
  - [`ep_cvar_views_solve!`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_cvar_views_setup(cvar_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_cvar_views_setup(cvar_views::CVV_VecCVV, pr::AbstractPriorResult,
                        sets::UniverseSets, w::ScenarioProbabilities,
                        ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                        dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false,
                        ledger::Option{<:AbstractVector} = nothing)

Resolve the conditional value at risk views once, into the data of one search.

`ep_cvar_views_setup` parses and checks the CVaR view groups, and it resolves every `prior(...)` reference against `pr` under `w`. It puts the groups into one loss matrix, one target vector and one level vector. It also selects the search that the targets need. [`ep_cvar_views_solve!`](@ref) reads this result and adds nothing to it.

The staged route of [`ep_prior`](@ref) searches up to three times, once for each stage, and each stage must read the same reference value. So [`ep_prior`](@ref) calls this function once, before the first solve. The call reads the first fit of the wrapped estimator and the prior probabilities of that fit. A later stage holds a refit `pr` whose weights are the posterior of the stage before. A reference resolved there gives a different target at each stage, and the view that the caller wrote changes. One call also keeps the parse out of the search, so the search does not parse again at each stage.

# Algorithm

 1. Loop over the view groups of `cvar_views`. One view group iterates as a collection of one element, so one loop reads both forms.
 2. For each view group, check that its `alg` is `nothing`, and read its significance level into `alpha`. Parse its equations, and accept only `==`.
 3. Replace every group name by the assets of the group. Replace every `prior(...)` reference by the prior conditional value at risk at `alpha` under `w`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the equality block `lcs`. Under `strict = false`, every row of the group can drop, and `lcs` is then `nothing`. The group then states no view, and the loop goes to the next group.
 5. Check that each row of `lcs` names one asset, and that each target is not negative.
 6. For each row of the block, add the asset that it names to `cols`, its target to `B`, its level to `alphas` and its text to `eqns`. All the groups go into one search, because each level is only the divisor of the positive part of its own view.
 7. Read the worst realisation of every named asset into `min_X`. Raise an error when a target is equal to or larger than it.
 8. Select the search `d_opt`. One view takes `ds_opt`, or a default [`ConditionalValueatRiskEntropyPooling`](@ref). More than one view takes `dm_opt`, or a default [`OptimEntropyPooling`](@ref) over `Optim.Fminbox`.
 9. Return `nothing` when no group states a view, so the stage uses the plain solve. Otherwise, return the loss columns `X`, the targets `B`, the levels `alphas` and the search `d_opt`.

# Arguments

  - `cvar_views`: One CVaR view group, or a vector of groups.
  - `pr`: Prior result that the `prior(...)` references read. It is the first fit of the wrapped estimator, not a refit of a later stage.
  - `sets`: Universe sets that map the asset names to indices.
  - `w`: Observation weights of the fit of `pr`. The `prior(...)` references resolve under these weights.
  - `ds_opt`: Search for one CVaR view, or `nothing` for the default.
  - `dm_opt`: Optimiser for more than one CVaR view, or `nothing` for the default.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - $(arg_dict[:ledger]) [`replace_group_by_assets`](@ref) and [`get_linear_constraints`](@ref) fill it.

# Validation

  - No view group declares a view formulation. A group whose `alg` is not `nothing` raises an `ArgumentError`, because this route writes no constraint formulation.
  - Every view names one asset. A view over more than one asset raises an `ArgumentError`.
  - Every target is not negative. A negative target raises a `DomainError`.
  - Every target is at most the worst realisation of its asset over the scenarios at a positive weight of `w`. The CVaR equals that loss when the posterior puts at least the level of the view on the scenarios of that loss, and a posterior that is positive on every scenario does this. A larger target raises a `DomainError`. The message names every view that is too large, and the largest target that its asset accepts. [`ep_assert_reachable_view`](@ref) raises the same error type for one view, so one `catch` reads both.

# Returns

  - `nothing`: No group states a view, because `strict` is `false` and every row of every group dropped.
  - `cvv::NamedTuple`: The search data, which holds the loss columns `X`, the targets `B`, the levels `alphas` and the search `d_opt`.

# Related

  - [`ep_cvar_views_solve!`](@ref): runs the search that this function prepares.
  - [`replace_prior_views`](@ref): resolves the `prior(...)` reference of a target.
  - [`get_pr_value`](@ref): reads the prior conditional value at risk that the reference resolves to.
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucciardiakeel2011]) Pages 39–49. $(ref_dict[:no_original_source])
"""
function ep_cvar_views_setup(cvar_views::CVV_VecCVV, pr::AbstractPriorResult,
                             sets::UniverseSets, w::ScenarioProbabilities,
                             ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                             dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false,
                             ledger::Option{<:AbstractVector} = nothing)
    X0 = pr.X
    # Each group is parsed under its own significance level, because a `prior(...)`
    # reference resolves to the prior CVaR at that level. The groups are then flattened
    # into one root-find: the recursive algorithm carries one `eta` per view already, and
    # the level enters only as the divisor of that view's positive part.
    cols = Vector{Int}(undef, 0)
    B = Vector{eltype(X0)}(undef, 0)
    alphas = Vector{eltype(X0)}(undef, 0)
    eqns = Vector{String}(undef, 0)
    for cvar_view in cvar_views
        @argcheck(isnothing(cvar_view.alg),
                  ArgumentError("The recursive CVaR route of `MeucciEntropyPoolingPrior` writes no constraint formulation, so a view group carrying `alg` has nothing to apply it to. Leave `alg` as `nothing`, or use `EntropyPoolingPrior`."))
        alpha = cvar_view.alpha
        views = parse_equation(cvar_view.views.val; ops1 = ("==",), ops2 = (:call, :(==)),
                               datatype = eltype(X0))
        views = replace_group_by_assets(views, sets, false, true, false; ledger = ledger)
        views = replace_prior_views(views, pr, sets, :cvar, alpha, w; strict = strict)
        lcs = get_linear_constraints(views, sets; datatype = eltype(X0), strict = strict,
                                     ledger = ledger)
        #! Under `strict = false` a view that names no asset is warned about and dropped,
        #! and a group whose every row drops parses to `nothing`. The warning is the whole
        #! diagnosis, so the group states no view and the search skips it. See issue #852.
        if isnothing(lcs)
            continue
        end
        @argcheck(!any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2)),
                  ArgumentError("Cannot mix multiple assets in a single cvar_view.\n$(views)"))
        @argcheck(!any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq),
                  DomainError(views,
                              "A `cvar_view` states a loss magnitude, so its target is non-negative, and one of these is negative:\n$(views)"))
        if !isa(views, AbstractVector)
            views = [views]
        end
        for i in axes(lcs.A_eq, 1)
            j = findfirst(!iszero, view(lcs.A_eq, i, :))
            push!(cols, j)
            push!(B, lcs.B_eq[i] / lcs.A_eq[i, j])
            push!(alphas, alpha)
            push!(eqns, views[i].eqn)
        end
    end
    #! Every group dropped every row, so there is no value at risk to search over. `nothing`
    #! is what `ep_cvar_views_setup` answers when no group was written at all, and it sends
    #! the stage down the plain solve. See issue #852.
    if isempty(cols)
        return nothing
    end
    X = view(X0, :, cols)
    # The worst loss on the support of `w` bounds a target: a posterior puts no mass where
    # the prior puts none (#1260). The CVaR reaches that loss when the posterior puts `alpha`
    # on its scenarios, which a posterior that is positive on every scenario does (#1925).
    min_X = dropdims(-minimum(view(X, w .> zero(eltype(w)), :); dims = 1); dims = 1)
    invalid = B .> min_X
    if any(invalid)
        msg = "The following views are too extreme, the maximum viable view for a given asset is its worst realisation:"
        for (v, m) in zip(eqns[invalid], min_X[invalid])
            msg *= "\n$v\t(> $m)."
        end
        msg *= "\nPlease lower the views or use a different prior with fatter tails."
        throw(DomainError(B[invalid], msg))
    end
    d_opt = if isone(length(B))
        ifelse(!isnothing(ds_opt), ds_opt, ConditionalValueatRiskEntropyPooling())
    else
        ifelse(!isnothing(dm_opt), dm_opt,
               OptimEntropyPooling(;
                                   args = (Optim.Fminbox(),
                                           Optim.Options(; outer_x_abstol = 1e-4,
                                                         x_abstol = 1e-4))))
    end
    return (; X = X, B = B, alphas = alphas, d_opt = d_opt)
end
"""
    ep_cvar_views_solve!(cvv::Nothing, epc::AbstractDict,
                         w::ScenarioProbabilities,
                         opt::AbstractEntropyPoolingOptimiser)

Solve the entropy pooling problem when the estimator has no CVaR view.

[`ep_cvar_views_setup`](@ref) prepares no search when the estimator has no conditional value at risk view. Then this method solves the entropy pooling problem once, with the prior weights, the constraint dictionary and the optimiser that it gets. Both routes of [`ep_prior`](@ref) call it, so their code is the same with and without a CVaR view.

# Algorithm

 1. Call [`entropy_pooling`](@ref) with the prior weights `w`, the constraint dictionary `epc` and the optimiser `opt`. The result is the posterior probability weights `pw`.
 2. Return `pw`. With no conditional value at risk view, there is no value at risk to search for. So the solve of the stage is the one call of step 1.

# Arguments

  - `cvv`: Shows that [`ep_cvar_views_setup`](@ref) prepared no search.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `w`: Prior probability weights.
  - `opt`: Entropy pooling optimiser.

# Returns

  - `pw::ScenarioProbabilities`: Posterior probability weights that meet the constraints, in the kind of `w`.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_cvar_views_setup`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_cvar_views_solve!(cvv::Nothing, epc::AbstractDict, w::ScenarioProbabilities,
                              opt::AbstractEntropyPoolingOptimiser)
    return entropy_pooling(w, epc, opt)
end
"""
    ep_cvar_views_solve!(cvv::NamedTuple, epc::AbstractDict,
                         w::ScenarioProbabilities,
                         opt::AbstractEntropyPoolingOptimiser)

Solve the entropy pooling problem with conditional value at risk (CVaR) views.

`ep_cvar_views_solve!` reads the targets, the levels and the search that [`ep_cvar_views_setup`](@ref) resolved. It finds the posterior probability weights with a root search for one CVaR view, and with an optimisation for more than one. It parses nothing and resolves nothing, because the staged route calls it once for each stage, and every stage must read the same targets.

The search is over the value at risk levels `etas`, one for each view, each in `[0, B]`. This search is a continuous relaxation of the recursive algorithm, which searches over discrete tail sizes. It gets the same target, and it also takes more than one view.

The bracket for one view stops a small distance inside `B`, at `B * (1 - sqrt(eps))`. At `B`, the constraint needs a posterior tail contribution of exactly zero. No posterior with all its weights above zero has such a contribution, so the problem at `B` is degenerate and its dual is unbounded. `Roots` evaluates the two ends of a bracket before it searches, so without the shrink each search for one view solves that problem once. The root is between about a third and two thirds of `B`, so the shrunk bracket holds it.

# Mathematical definition

The conditional value at risk of asset ``i`` is the value of the Rockafellar-Uryasev programme, and the minimiser of that programme is the value at risk. A view that sets it to ``\\bar{c}`` is a pair of conditions on the posterior. The first condition is linear in ``\\boldsymbol{p}`` at a fixed ``\\eta``, and the second condition fixes ``\\eta``:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(X) &= \\underset{\\eta}{\\min}\\; \\left\\{ \\eta + \\frac{1}{\\alpha} \\sum_{t=1}^{T} p_{t} \\left(-x_{t,\\,i} - \\eta\\right)^{+} \\right\\}\\,, \\\\
\\frac{1}{\\alpha} \\sum_{t=1}^{T} p_{t} \\left(-x_{t,\\,i} - \\eta\\right)^{+} &= \\bar{c} - \\eta\\,, \\\\
\\sum_{t=1}^{T} p_{t} \\mathbb{1}\\left\\{-x_{t,\\,i} > \\eta\\right\\} &= \\alpha\\,.
\\end{align}
```

The second line is linear in ``\\boldsymbol{p}``, so one entropy pooling solve meets it at each candidate ``\\eta``. The third line holds only when that ``\\eta`` is the value at risk of the posterior of the solve. So the two lines have one common fixed point, and the posterior meets the view only at that point.

Where:

  - $(math_dict[:cvar_stat])
  - $(math_dict[:cvar_target])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``\\eta``: Value at risk of asset ``i`` at level ``\\alpha``. It is the minimiser of the first line, and the variable of the outer search.
  - ``(\\cdot)^{+}``: Positive part, ``\\max(\\cdot,\\, 0)``.

# Algorithm

 1. Read the loss columns `X`, the targets `B`, the levels `alphas` and the search `d_opt` from `cvv`.
 2. Define `func(etas)`. At the candidate `etas`, it writes the second line above into `epc` under the key `:cvar_eq`. It solves the whole constraint set with [`entropy_pooling`](@ref) into `wi`, and returns `wi` and the residual of the third line. For one view, the residual is the posterior tail mass minus `alpha`. For more than one view, it is the [`norm_error`](@ref) over the views of the posterior conditional value at risk minus the target.
 3. Search for the value at risk `res`. For one view, `Roots.find_zero` finds the root of the residual of `func` over `[0, B[1] * (1 - sqrt(eps))]`. For more than one view, `Optim.optimize` minimises the residual over the box `[0, B]` from the start `0.5 * B`.
 4. Call `func(res)` one more time, and return the posterior probability weights that it calculates.

# Arguments

  - `cvv`: Search data from [`ep_cvar_views_setup`](@ref).
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `w`: Prior probability weights that the stage projects.
  - `opt`: Optimiser of each inner entropy pooling solve.

# Validation

  - Every candidate value at risk is in `[0, B]`. A candidate outside the box raises a `DomainError`.
  - The search must succeed. When `Roots.find_zero` raises an error, this method raises an `ErrorException` that holds it. When `Optim.converged` reports that `Optim.optimize` failed, this method raises an `ErrorException`.
  - The method does not detect an infeasible view set. The residual of this search is the posterior tail mass minus `alpha`. A posterior with almost all of its mass on one observation has a small residual, but it can miss the view by a large amount. The warning of [`MeucciEntropyPoolingPrior`](@ref) states how to find such a result.
  - The method does not detect a missed feasible view. The search reads its own residual, not how closely the inner [`entropy_pooling`](@ref) solve meets the constraint of the view. With more than one view, `Optim.converged` accepts a solve that stopped on the step in `x` and not on stationarity. The warning of [`MeucciEntropyPoolingPrior`](@ref) states the mechanism, the measured sizes, and how to read the result.

# Returns

  - `pw::ScenarioProbabilities`: Posterior probability weights that meet the CVaR views, in the kind of `w`.

# Related

  - [`ep_cvar_views_setup`](@ref): resolves the targets that this search reads.
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref): its stopping rule sets how closely each inner solve meets the constraint that this search solves again. Its tip states the size of the miss.
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`entropy_pooling`](@ref)
  - [`norm_error`](@ref): scores the residual of a search over more than one view.
  - [`ConditionalValueatRisk`](@ref): reads the posterior statistic that residual scores.

# References

  - $(ref_dict[:meucciardiakeel2011]) Pages 39–49. $(ref_dict[:no_original_source])
"""
function ep_cvar_views_solve!(cvv::NamedTuple, epc::AbstractDict, w::ScenarioProbabilities,
                              opt::AbstractEntropyPoolingOptimiser)
    (; X, B, alphas, d_opt) = cvv
    N = length(B)
    function func(etas)
        delete!(epc, :cvar_eq)
        @argcheck(all(zero(eltype(etas)) .<= etas .<= B),
                  DomainError(etas, "all elements of etas must be in [0, B] where B = $B"))
        pos_part = max.(-X .- transpose(etas), zero(eltype(X)))
        add_ep_constraint!(epc, transpose(pos_part ./ transpose(alphas)), B .- etas,
                           :cvar_eq)
        wi = entropy_pooling(w, epc, opt)
        err = if N == 1
            sum(wi[.!iszero.(pos_part)]) - alphas[1]
        else
            norm_error(d_opt.err,
                       [ConditionalValueatRisk(; alpha = alphas[i], w = wi)(view(X, :, i)) -
                        B[i] for i in 1:N], N)
        end
        return wi, err
    end
    res = if N == 1
        #! The constraint at `eta = B[1]` reads `E_p[(-x - eta)^+] / alpha = 0`, and the
        #! positive part is non-zero on the observations worse than `-B[1]`, a non-empty set
        #! under the guard in `ep_cvar_views_setup`. No interior posterior carries a tail
        #! contribution of exactly zero, so the problem there is degenerate: its dual is
        #! unbounded, and a tighter stopping rule only runs the dual further. `Roots`
        #! evaluates both ends of a bracket, so every single-view search would run that
        #! problem once. `sqrt(eps)` is the smallest shrink that bounds it. Over the twelve
        #! cases of `test_12a_entropy_pooling.jl` two stopping rules stop the dual a factor
        #! of 1.4 to 2.4 apart at `B[1]`, and still 1.2 to 1.9 apart at `B[1] * (1 - eps)`,
        #! where at `B[1] * (1 - sqrt(eps))` they agree to four significant digits. The root
        #! sits between 0.32 and 0.64 of `B[1]`, so the shrink does not move it. See issue
        #! #574.
        hi = B[1] * (one(eltype(B)) - sqrt(eps(eltype(B))))
        try
            [Roots.find_zero(x -> func(x)[2], (zero(eltype(B)), hi), d_opt.args...;
                             d_opt.kwargs...)]
        catch e
            throw(ErrorException("CVaR entropy pooling optimisation failed. Relax the view, increase alpha, use different solver parameters, use VaR views instead, or use a different prior.\n$(e)"))
        end
    else
        res = Optim.optimize(x -> func(x)[2], zeros(N), B, 0.5 * B, d_opt.args...;
                             d_opt.kwargs...)
        @argcheck(Optim.converged(res),
                  ErrorException("CVaR entropy pooling optimisation failed. Relax the view, increase alpha, use different solver parameters, use VaR views instead, reduce the number of CVaR views, or use a different prior."))
        Optim.minimizer(res)
    end
    return func(res)[1]
end
"""
    prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute the entropy pooling prior moments of the asset returns.

`prior` orients the data along `dims` and calls [`ep_prior`](@ref), which dispatches on the entropy pooling algorithm `pe.alg`. [`H0_EntropyPooling`](@ref) applies every view in one optimisation. [`StagedEP`](@ref), the union of [`H1_EntropyPooling`](@ref) and [`H2_EntropyPooling`](@ref), applies the views in stages, from the lower moments to the higher moments.

# Algorithm

 1. Orient `X` and `F` to observations by assets with [`dims_oriented`](@ref), so every step below reads one observation in each row.
 2. Dispatch on `pe.alg` through [`ep_prior`](@ref), and return the [`LowOrderPrior`](@ref) that it makes.

# Arguments

  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, with its observations along `dims`.
  - `F`: Optional factor returns matrix, with its observations along `dims`.
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - `kwargs...`: More keyword arguments, which go to the wrapped prior estimator.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Prior result that holds the asset returns, the posterior mean and covariance, the posterior probability weights, the effective number of scenarios, the Kullback-Leibler divergence and the factor block of the last refit.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`ep_prior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    X, F = dims_oriented(dims, X, F)
    return ep_prior(pe.alg, pe, X, F, pnl; strict = strict, kwargs...)
end
"""
    ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing;
             strict::Bool = false, kwargs...)

Compute the entropy pooling prior moments of the asset returns, with the views applied in stages.

`ep_prior` calculates the mean and the covariance of the asset returns with entropy pooling, under the staged algorithms [`H1_EntropyPooling`](@ref) and [`H2_EntropyPooling`](@ref). It takes views on the mean, the value at risk, the conditional value at risk, the variance, the covariance, the skewness, the kurtosis and the correlation. Each stage adds the views of one group of moments, solves, and refits the wrapped estimator under the new weights. A later stage holds the moments of the earlier stages fixed, so a view on a higher moment does not move a lower moment.

Each stage refits the wrapped estimator after its solve, and the next stage parses its views against that refit. So a `prior(...)` reference in a later stage resolves against the posterior of the stage before, not against the prior of the estimator. Under a `mu_views` or `cvar_views` view, a `sigma_views` target `prior(A)*1.3` asks for 1.3 times the variance that stage one leaves, not 1.3 times the empirical variance. When the reference must be the empirical value, write the target as a number.

# Mathematical definition

The staged posterior is a chain of Kullback-Leibler projections, not one projection. Stage ``k`` holds the constraint set ``\\mathcal{C}_{k}``, which has the rows of stage ``k`` and of every stage before it. It projects a reference ``\\boldsymbol{r}^{(k)}`` that the algorithm tag fixes:

```math
\\begin{align}
\\boldsymbol{p}^{(k)} &= \\underset{\\boldsymbol{p} \\in \\mathcal{C}_{k}}{\\arg\\min} \\sum_{t=1}^{T} p_{t} \\ln\\!\\frac{p_{t}}{r_{t}^{(k)}}\\,, \\\\
\\mathcal{C}_{k} &= \\left\\{ \\boldsymbol{p} : \\mathbf{A}_{k} \\boldsymbol{p} = \\boldsymbol{B}_{k},\\; \\boldsymbol{p} \\geq \\boldsymbol{0},\\; \\boldsymbol{1}^\\intercal \\boldsymbol{p} = 1 \\right\\}\\,, \\\\
\\mathcal{C}_{1} &\\supseteq \\mathcal{C}_{2} \\supseteq \\mathcal{C}_{3}\\,, \\\\
\\boldsymbol{r}^{(k)} &= \\begin{cases} \\boldsymbol{q} & \\text{under } \\texttt{H1\\_EntropyPooling} \\\\ \\boldsymbol{p}^{(k-1)} & \\text{under } \\texttt{H2\\_EntropyPooling} \\end{cases}\\,, \\\\
\\boldsymbol{p}^{*} &= \\boldsymbol{p}^{(K)}\\,.
\\end{align}
```

The first stage holds the mean and value at risk views. The second stage holds the variance and covariance views, and the third stage holds the correlation, skewness and kurtosis views. The sets nest, so the posterior of the last stage meets every view. The two references differ only in the start point of the chain. ``\\texttt{H1\\_EntropyPooling}`` gives the projection of the prior onto the whole view set, and ``\\texttt{H2\\_EntropyPooling}`` gives the projection of each stage onto the next. A conditional value at risk view is not a row of ``\\mathbf{A}_{k}``. It enters every stage as the fixed point of [`ep_cvar_views_solve!`](@ref), whose section states it.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:T])
  - ``\\boldsymbol{p}^{(k)}``: Posterior probabilities of stage ``k``. ``\\boldsymbol{p}^{*}`` is the posterior of the last stage ``K``.
  - ``\\boldsymbol{r}^{(k)}``: Reference probabilities that stage ``k`` projects.
  - ``\\mathcal{C}_{k}``: Constraint set of stage ``k``, which holds the rows of every stage up to stage ``k``.
  - ``\\mathbf{A}_{k}``, ``\\boldsymbol{B}_{k}``: Rows and right-hand side of ``\\mathcal{C}_{k}``.

The posterior moments are the sample statistics under the weights ``\\boldsymbol{p}^{*}``.

# Algorithm

 1. Fit the wrapped prior estimator with [`ep_wrapped_prior`](@ref). The result is `pr` and the prior probabilities `w0` on its rows. The fit sets the observation axis. `T` is `size(pr.X, 1)`, which is smaller than `size(X, 1)` when a nested prior drops rows. The probabilities are `pe.w` when the caller gives one, `pr.w` when the fit returns one, and the uniform `1/T` otherwise. [`prior_probability_factory`](@ref) puts the `pe.w` of the caller on the rows that each weighted field of the wrapped estimator reads, and the fit of `pr` uses it. Before that fit, [`scenario_count`](@ref) reads `T` with only the steps that drop rows, so most wrapped priors need no fit for it. When the fit under `pe.w` returns its own `pr.w`, a nested prior solved its views from `pe.w`. Then `w0` is `pr.w`, the posterior of the nested prior.
 2. Pin each choice of `pe.pe` under a [`PinnedChoice`](@ref) at this fit with [`pin_prior_choice`](@ref), so every refit reads the choice that the views read. Every refit goes through [`scenario_prior`](@ref), which holds the model of this fit, with its regression and its residual block. So the refit reads the scenarios that the views read.
 3. Get the Investable Mask of the fit and the universe of the views with [`investable_views`](@ref). Every view row uses only the investable columns, and the mask is the same at each stage.
 4. Make the empty constraint dictionary `epc`, the matrix `fixed` of the pinned moments, and the Departure Ledger. Resolve the `cvar_views` against that fit one time, through [`ep_cvar_views_setup`](@ref) into `cvv`. Every stage searches the same targets, so a `prior(...)` reference gives one number for the whole chain.
 5. Stage one, the mean and the value at risk. Write the `mu_views` and `var_views` rows into `epc`. When `epc` holds a row or `cvv` holds a search, solve from `w0` through [`ep_cvar_views_solve!`](@ref) into `w1`, and refit `pr` under `w1`.
 6. Stage two, the variance and the covariance. Write the `sigma_views` and `cov_views` rows into `epc`. Pin the mean of every asset that those rows read with [`fix_mu!`](@ref), so the stage cannot move a moment that an earlier stage set. Under the same test, solve into `w1`, and refit `pr` under `w1`.
 7. Stage three, the correlation, the skewness and the kurtosis. Write the `sk_views`, `kt_views` and `rho_views` rows into `epc`. Pin the mean and the variance of every asset that those rows read with [`fix_mu!`](@ref) and [`fix_sigma!`](@ref). Under the same test, solve into `w1`, and refit `pr` under `w1`.
 8. Steps 6 and 7 solve from `w0` under [`H1_EntropyPooling`](@ref), and from the `w1` of the stage before under [`H2_EntropyPooling`](@ref).
 9. Report the departed assets and the views that they removed, with [`announce_ep_departures`](@ref).
10. Calculate the effective number of scenarios `ens`, the exponential of the entropy of `w1`. Calculate `kld`, the Kullback-Leibler divergence of `w1` from `w0`.
11. Return a [`LowOrderPrior`](@ref) that holds the moments of the last refit, `w1`, `ens` and `kld`. The factor block `fpr` comes from that refit with no change.

# Arguments

  - `alg`: Staged entropy pooling algorithm, which [`prior`](@ref) reads from `pe.alg`.
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, with one observation in each row, as [`prior`](@ref) orients it.
  - `F`: Optional factor returns matrix, with one observation in each row, as [`prior`](@ref) orients it.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - `kwargs...`: More keyword arguments, which go to the wrapped prior estimator.

# Validation

  - If a view field is not `nothing`, `pe.sets` must not be `nothing`.
  - If the caller gives prior weights `pe.w`, `length(pe.w) == size(pr.X, 1)` must hold, where `size(pr.X, 1)` is the number of observations that the wrapped estimator returns. A different length raises a `DimensionMismatch`. The message names that number, the number of observations that the estimator got, and the rule.
  - Every view equation has a comparison operator that its family accepts. An unsupported operator raises a `Meta.ParseError` that names the operators of that family.

# Returns

  - `pr::LowOrderPrior`: Prior result that holds the asset returns, the posterior mean and covariance, the posterior probability weights, the effective number of scenarios, the Kullback-Leibler divergence and the factor block of the last refit.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`prior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`StagedEP`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`ep_mu_views!`](@ref)
  - [`ep_var_views!`](@ref)
  - [`ep_cvar_views_setup`](@ref)
  - [`ep_cvar_views_solve!`](@ref)
  - [`ep_sigma_views!`](@ref)
  - [`ep_sk_views!`](@ref)
  - [`ep_kt_views!`](@ref)
  - [`ep_cov_views!`](@ref)
  - [`ep_rho_views!`](@ref)
  - [`fix_mu!`](@ref)
  - [`fix_sigma!`](@ref)
"""
function ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum,
                  F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing;
                  strict::Bool = false, kwargs...)
    # The nested prior is fitted first, and the prior probabilities are read on the rows
    # it answered. See the note at the same step in `EntropyPoolingPrior`'s staged
    # `ep_prior`. ADR 0116.
    pe, pr, w0 = ep_wrapped_prior(pe, X, F, pnl; strict = strict, kwargs...)
    # Every refit reads the cache of the fit that the views read (#1751).
    pr0 = pr
    w1 = w0
    # The views read the regressions of this fit. A choice under `PinnedChoice` is pinned
    # here, so every refit under the posterior weights reads the same choice (#1641).
    pe = pin_prior_choice(pe, X, F, nothing, pnl; strict = strict, kwargs...)
    # See the note at the same step in `EntropyPoolingPrior`'s staged `ep_prior`: every row
    # is built on the investable columns, because `0 * NaN` is `NaN`, and the mask does not
    # move between stages. ADR 0115 and ADR 0125.
    imsk, vsets, ni = investable_views(pr, pe.sets)
    led = String[]
    vpr = investable_prior(imsk, pr)
    fixed = falses(size(vpr.X, 2), 2)
    epc = Dict{Symbol, Tuple{MatNum, VecNum}}()
    # mu and VaR
    # Every `prior(...)` reference resolves against the fit above. The CVaR
    # search runs once per stage against a refit `pr`, so resolving inside it would state a
    # different target at each stage. It is resolved once, here, and the stages read it.
    cvv = ep_cvar_views_setup(pe.cvar_views, vpr, vsets, w0, pe.ds_opt, pe.dm_opt;
                              strict = strict, ledger = led)
    ep_mu_views!(pe.mu_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    ep_var_views!(pe.var_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    # Every row of every family can drop under `strict = false`, and the stage then states
    # no view. The prior is the answer, so neither the solve nor the refit runs. See issue
    # #852.
    if !isempty(epc) || !isnothing(cvv)
        # See the note at the same step in `EntropyPoolingPrior`'s staged `ep_prior` (#1649).
        ep_diagnose_views(pe.diagnose, w0, epc, AbstractEntropyPoolingTailView[], pe,
                          (:mu_views, :var_views))
        w1 = LinearAlgebra.normalize!(ep_cvar_views_solve!(cvv, epc, w0, pe.opt), 1)
        pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                            cache_reg = true)
        vpr = investable_prior(imsk, pr)
    end
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            to_fix = ep_sigma_views!(pe.sigma_views, epc, vpr, vsets, w0; strict = strict,
                                     ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
        end
        # cov
        if !isnothing(pe.cov_views)
            to_fix = ep_cov_views!(pe.cov_views, epc, vpr, vsets, w0; strict = strict,
                                   ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
        end
        # See the twin note one stage up: a stage that states no view does not solve.
        if !isempty(epc) || !isnothing(cvv)
            ws = ifelse(isa(alg, H1_EntropyPooling), w0, w1)
            ep_diagnose_views(pe.diagnose, ws, epc, AbstractEntropyPoolingTailView[], pe,
                              (:mu_views, :var_views, :sigma_views, :cov_views))
            w1 = LinearAlgebra.normalize!(ep_cvar_views_solve!(cvv, epc, ws, pe.opt), 1)
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
            to_fix = ep_rho_views!(pe.rho_views, epc, vpr, vsets, w0; strict = strict,
                                   ledger = led)
            fix_mu!(epc, view(fixed, :, 1), to_fix, vpr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, vpr)
        end
        # See the twin note two stages up: a stage that states no view does not solve.
        if !isempty(epc) || !isnothing(cvv)
            ws = ifelse(isa(alg, H1_EntropyPooling), w0, w1)
            ep_diagnose_views(pe.diagnose, ws, epc, AbstractEntropyPoolingTailView[], pe,
                              (:mu_views, :var_views, :sigma_views, :cov_views, :sk_views,
                               :kt_views, :rho_views))
            w1 = LinearAlgebra.normalize!(ep_cvar_views_solve!(cvv, epc, ws, pe.opt), 1)
            pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                                cache_reg = true)
        end
    end
    announce_ep_departures(ni, led, isempty(epc) && isnothing(cvv) && !isempty(led))
    # The factor block is the refit prior's, forwarded whole. It is *not* stamped with the
    # pooled weights: `w1` is threaded into the wrapped estimator by `factory` above, so a
    # factor prior that records its own weighting will report it here, and one that does not
    # (`EmpiricalPrior` never sets `w`) is a gap to close at that producer rather than to
    # paper over from out here — see #217. Writing `w1` on would also owe the factor block
    # `ens`/`kld` under ADR 0046's binding, which is the coupling that made the flat `f_w` a
    # duplicate of `w` at five of six producers in the first place.
    # The refit reads `w1` through its estimators, so its `eoc` is the count of the moments
    # under `w1`.
    (; X, o_X, mu, sigma, chol, eoc, rr, fpr, ss_gaps) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, eoc = eoc, kld = kld, rr = rr, fpr = fpr,
                         ss_gaps = ss_gaps, pc = opinion_pooling_cache(pr))
end
"""
    ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing;
             strict::Bool = false, kwargs...)

Compute the entropy pooling prior moments of the asset returns, with every view applied in one solve.

`ep_prior` calculates the mean and the covariance of the asset returns with entropy pooling, under the [`H0_EntropyPooling`](@ref) algorithm. It applies every view in one optimisation. This route solves the entropy pooling problem one time. But a view on a higher moment can move a lower moment, because no stage holds the lower moments fixed.

# Mathematical definition

The single-shot posterior is one Kullback-Leibler projection of the prior onto the intersection of the constraint sets of all the views. It is the staged chain of [`ep_prior`](@ref) with one stage:

```math
\\begin{align}
\\boldsymbol{p}^{*} &= \\underset{\\boldsymbol{p} \\in \\mathcal{C}}{\\arg\\min} \\sum_{t=1}^{T} p_{t} \\ln\\!\\frac{p_{t}}{q_{t}}\\,, \\\\
\\mathcal{C} &= \\mathcal{C}_{1} \\cap \\mathcal{C}_{2} \\cap \\mathcal{C}_{3}\\,.
\\end{align}
```

One reference and one feasible set hold every view, so no stage can pin a moment. A view on a higher moment can move a lower moment of the same asset. That is the only difference from the staged chain, whose sets ``\\mathcal{C}_{k}`` this set ``\\mathcal{C}`` intersects. A conditional value at risk view is not a row of ``\\mathcal{C}``. It enters as the fixed point of [`ep_cvar_views_solve!`](@ref), whose section states it.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:T])
  - ``\\boldsymbol{p}^{*}``: Posterior probabilities of the one solve.
  - ``\\mathcal{C}``, ``\\mathcal{C}_{k}``: Feasible set of the one solve, and the stage sets that it intersects.

The posterior moments are the sample statistics under the weights ``\\boldsymbol{p}^{*}``.

# Algorithm

 1. Fit the wrapped prior estimator with [`ep_wrapped_prior`](@ref). The result is `pr` and the prior probabilities `w0` on its rows. The fit sets the observation axis. `T` is `size(pr.X, 1)`, which is smaller than `size(X, 1)` when a nested prior drops rows. The probabilities are `pe.w` when the caller gives one, `pr.w` when the fit returns one, and the uniform `1/T` otherwise. [`prior_probability_factory`](@ref) puts the `pe.w` of the caller on the rows that each weighted field of the wrapped estimator reads, and the fit of `pr` uses it. Before that fit, [`scenario_count`](@ref) reads `T` with only the steps that drop rows, so most wrapped priors need no fit for it. When the fit under `pe.w` returns its own `pr.w`, a nested prior solved its views from `pe.w`. Then `w0` is `pr.w`, the posterior of the nested prior.
 2. Pin each choice of `pe.pe` under a [`PinnedChoice`](@ref) at this fit with [`pin_prior_choice`](@ref), so every refit reads the choice that the views read. Every refit goes through [`scenario_prior`](@ref), which holds the model of this fit, with its regression and its residual block. So the refit reads the scenarios that the views read.
 3. Get the Investable Mask of the fit and the universe of the views with [`investable_views`](@ref). Every view row uses only the investable columns.
 4. Make the empty constraint dictionary `epc` and the Departure Ledger. Resolve the `cvar_views` against that fit one time, through [`ep_cvar_views_setup`](@ref) into `cvv`.
 5. Write the `mu_views` and `var_views` rows into `epc`.
 6. Write the `sigma_views` and `cov_views` rows into `epc`. This route pins no moment, so it does not call [`fix_mu!`](@ref) or [`fix_sigma!`](@ref).
 7. Write the `sk_views`, `kt_views` and `rho_views` rows into `epc`.
 8. Report the departed assets and the views that they removed, with [`announce_ep_departures`](@ref).
 9. When `epc` holds a row or `cvv` holds a search, solve the whole set one time through [`ep_cvar_views_solve!`](@ref) into `w1`, and refit `pr` under `w1`. Under `strict = false`, every row of every family can drop, and the view set then states no view. Then `w1` is `w0`, `kld` is zero, and no refit runs.
10. Calculate the effective number of scenarios `ens`, the exponential of the entropy of `w1`. Calculate `kld`, the Kullback-Leibler divergence of `w1` from `w0`.
11. Return a [`LowOrderPrior`](@ref) that holds the moments of the refit, `w1`, `ens` and `kld`. The factor block `fpr` comes from that refit with no change.

# Arguments

  - `alg`: Single-shot entropy pooling algorithm, which [`prior`](@ref) reads from `pe.alg`.
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, with one observation in each row, as [`prior`](@ref) orients it.
  - `F`: Optional factor returns matrix, with one observation in each row, as [`prior`](@ref) orients it.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - `kwargs...`: More keyword arguments, which go to the wrapped prior estimator.

# Validation

  - If a view field is not `nothing`, `pe.sets` must not be `nothing`.
  - If the caller gives prior weights `pe.w`, `length(pe.w) == size(pr.X, 1)` must hold, where `size(pr.X, 1)` is the number of observations that the wrapped estimator returns. A different length raises a `DimensionMismatch`. The message names that number, the number of observations that the estimator got, and the rule.
  - Every view equation has a comparison operator that its family accepts. An unsupported operator raises a `Meta.ParseError` that names the operators of that family.

# Returns

  - `pr::LowOrderPrior`: Prior result that holds the asset returns, the posterior mean and covariance, the posterior probability weights, the effective number of scenarios, the Kullback-Leibler divergence and the factor block of the refit.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`prior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`ep_mu_views!`](@ref)
  - [`ep_var_views!`](@ref)
  - [`ep_cvar_views_setup`](@ref)
  - [`ep_cvar_views_solve!`](@ref)
  - [`ep_sigma_views!`](@ref)
  - [`ep_sk_views!`](@ref)
  - [`ep_kt_views!`](@ref)
  - [`ep_cov_views!`](@ref)
  - [`ep_rho_views!`](@ref)
"""
function ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum,
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
    epc = Dict{Symbol, Tuple{MatNum, VecNum}}()
    # mu and VaR
    # Every `prior(...)` reference resolves against the fit above.
    cvv = ep_cvar_views_setup(pe.cvar_views, vpr, vsets, w0, pe.ds_opt, pe.dm_opt;
                              strict = strict, ledger = led)
    ep_mu_views!(pe.mu_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    ep_var_views!(pe.var_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            ep_sigma_views!(pe.sigma_views, epc, vpr, vsets, w0; strict = strict,
                            ledger = led)
        end
        # cov
        if !isnothing(pe.cov_views)
            ep_cov_views!(pe.cov_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
        end
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
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
            ep_rho_views!(pe.rho_views, epc, vpr, vsets, w0; strict = strict, ledger = led)
        end
    end
    announce_ep_departures(ni, led, isempty(epc) && isnothing(cvv) && !isempty(led))
    w1 = w0
    # Every row of every family can drop under `strict = false`, and the view set then
    # states nothing. The prior is the answer, so neither the solve nor the refit runs. See
    # issue #852.
    if !isempty(epc) || !isnothing(cvv)
        ep_diagnose_views(pe.diagnose, w0, epc, AbstractEntropyPoolingTailView[], pe,
                          (:mu_views, :var_views, :sigma_views, :cov_views, :sk_views,
                           :kt_views, :rho_views))
        w1 = LinearAlgebra.normalize!(ep_cvar_views_solve!(cvv, epc, w0, pe.opt), 1)
        pr = scenario_prior(pe.pe, w1, pr0, X, F, pnl; strict = strict, kwargs...,
                            cache_reg = true)
    end
    # The factor block is the refit prior's, forwarded whole. It is *not* stamped with the
    # pooled weights: `w1` is threaded into the wrapped estimator by `factory` above, so a
    # factor prior that records its own weighting will report it here, and one that does not
    # (`EmpiricalPrior` never sets `w`) is a gap to close at that producer rather than to
    # paper over from out here — see #217. Writing `w1` on would also owe the factor block
    # `ens`/`kld` under ADR 0046's binding, which is the coupling that made the flat `f_w` a
    # duplicate of `w` at five of six producers in the first place.
    # The refit reads `w1` through its estimators, so its `eoc` is the count of the moments
    # under `w1`.
    (; X, o_X, mu, sigma, chol, eoc, rr, fpr, ss_gaps) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, eoc = eoc, kld = kld, rr = rr, fpr = fpr,
                         ss_gaps = ss_gaps, pc = opinion_pooling_cache(pr))
end

function factor_residual_config(pe::MeucciEntropyPoolingPrior)
    return factor_residual_config(pe.pe)
end

export MeucciEntropyPoolingPrior
