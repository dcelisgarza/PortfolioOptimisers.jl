"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments meet a set of views, and root-finds a CVaR view.

`MeucciEntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports views on the mean, the value at risk, the conditional value at risk, the variance, the covariance, the correlation, the skewness and the kurtosis, and it takes custom prior weights and solver configuration.

This is the earlier of the library's two entropy pooling estimators, and it is kept because its CVaR route is a different *algorithm*, not a different formulation of the same problem: a CVaR view is a target hunted by the recursive algorithm of Meucci, Ardia and Keel, where [`ConditionalValueatRiskEntropyPooling`](@ref) root-finds the value at risk level and re-solves the whole entropy pooling problem at each candidate. That route takes equality CVaR views alone, one asset per view.

Reach for [`EntropyPoolingPrior`](@ref) instead where a tail view has to be an inequality, name two assets, or land on the entropic value at risk: there each tail view is a constraint of the single entropy pooling problem, so one solve answers every view.

The comparison operator a view accepts depends on the moment it constrains: `var_views` takes `==` and `>=`, `cvar_views` takes `==` alone, and every other view family takes `==`, `>=` and `<=`. An unsupported operator raises a `Meta.ParseError` naming the operators that view accepts.

!!! warning

    An infeasible view set is never detected. Neither the entropy pooling solve nor the CVaR search reads one: the dual of an infeasible set is unbounded, so the minimiser runs away, the posterior collapses onto one observation, and `Optim` reports the solve as converged. The residual the CVaR search minimises is small on such a posterior, so the search reports success too. Read the result rather than the flag: `ens` falls to a handful out of the number of observations, one weight sits near one, `kld` is large, and the posterior statistic the view named is far from its target. Views that pull one asset in two directions at once are the common way to reach it, such as a `sigma_views` row that shrinks an asset written beside a `cvar_views` row that fattens the same asset's tail. The same pair on two different assets is feasible and solves normally, so it is the direction and not the pairing. [`entropy_pooling`](@ref) states the mechanism.

    A runaway dual sometimes overflows before it settles, and the view set then raises rather than answering. The staged route reaches the moment estimators with non-finite weights, which raise an `ArgumentError` naming Infs or NaNs. The single-shot route raises the CVaR search's own `ErrorException`. Neither raise detects the infeasibility. Both are the same runaway dual met further along, and which of the three a given view set gives is not stable: the search over an infeasible set is chaotic, so a change in the sequence of candidate value at risk levels moves the answer between them. Treat any of the three as the same finding, and read the views rather than the message.

# Algorithm

The constructor derives the prior probabilities, and validates everything else.

 1. When `w` is `nothing`, derive nothing. [`prior`](@ref) builds the uniform weights `1/T` at solve time, one per observation.
 2. When `w` is not `nothing`, normalise it to sum to one, giving the prior probabilities the pooling starts from. A mutable `w.values` is normalised in place with `LinearAlgebra.normalize!`, and an immutable one is replaced by a new `StatsBase.pweights` over the normalised values.

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
        w::Option{<:StatsBase.ProbabilityWeights} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling()
    ) -> MeucciEntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If any view constraint is not `nothing`, `sets` must not be `nothing`.
  - If `var_views` is a vector, `!isempty(var_views)`.
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
              │        me ┼ SimpleExpectedReturns
              │           │   w ┴ nothing
              │   horizon ┴ nothing
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
              │    xkey ┼ String: "nx"
              │   uxkey ┼ String: "ux"
              │    fkey ┼ String: "nf"
              │   ufkey ┼ String: "uf"
              │    zkey ┼ String: "nz"
              │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
       ds_opt ┼ nothing
       dm_opt ┼ nothing
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

  - $(ref_dict[:meucci2008])
  - $(ref_dict[:meucciardiakeel2011])
  - $(ref_dict[:vorobets2021])
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
    $(field_dict[:ep_w])
    """
    @wprop w
    """
    $(field_dict[:epalg])
    """
    alg
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
                                       opt::NonCVaREP,
                                       w::Option{<:StatsBase.ProbabilityWeights},
                                       alg::AbstractEntropyPoolingAlgorithm)
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
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(sigma_views), typeof(sk_views), typeof(kt_views),
                   typeof(cov_views), typeof(rho_views), typeof(sets), typeof(ds_opt),
                   typeof(dm_opt), typeof(opt), typeof(w), typeof(alg)}(pe, mu_views,
                                                                        var_views,
                                                                        cvar_views,
                                                                        sigma_views,
                                                                        sk_views, kt_views,
                                                                        cov_views,
                                                                        rho_views, sets,
                                                                        ds_opt, dm_opt, opt,
                                                                        w, alg)
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
                                   w::Option{<:StatsBase.ProbabilityWeights} = nothing,
                                   alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())::MeucciEntropyPoolingPrior
    return MeucciEntropyPoolingPrior(pe, mu_views, var_views, cvar_views, sigma_views,
                                     sk_views, kt_views, cov_views, rho_views, sets, ds_opt,
                                     dm_opt, opt, w, alg)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties MeucciEntropyPoolingPrior begin
    forward(pe, me, ce)
end
"""
    const VecMeucciEP = AbstractVector{<:MeucciEntropyPoolingPrior}

Groups the vector forms of [`MeucciEntropyPoolingPrior`](@ref) that a method can dispatch on.

The group exists because a caller that pools opinions holds many estimators rather than one, and a method that takes such a collection needs a single type to sign. The library's own opinion pooling signs the wider [`VecEP`](@ref) instead, which admits [`EntropyPoolingPrior`](@ref) beside this estimator, so `VecMeucciEP` is the narrower name for a method that takes the recursive CVaR route alone.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref): the element type the vector holds.
  - [`VecEP`](@ref): the wider group [`OpinionPoolingPrior`](@ref) signs, which admits both entropy pooling estimators.
"""
const VecMeucciEP = AbstractVector{<:MeucciEntropyPoolingPrior}
"""
    ep_cvar_views_setup(cvar_views::Nothing, args...; kwargs...)

No-op pass-through for conditional value at risk view constraints when none are specified.

# Arguments

  - `cvar_views::Nothing`: Indicates that no CVaR view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`: There is no search to prepare, which [`ep_cvar_views_solve!`](@ref) dispatches on.

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
                        sets::UniverseSets, w::StatsBase.ProbabilityWeights,
                        ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                        dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false)

Resolve the **conditional value at risk** views once, into the data one search reads.

`ep_cvar_views_setup` parses and validates the CVaR view groups, resolves every `prior(...)` reference against `pr` under `w`, and flattens the groups into one loss matrix, one target vector and one level vector. It also chooses the search the targets need. [`ep_cvar_views_solve!`](@ref) reads that answer and adds nothing to it.

The staged route of [`ep_prior`](@ref) searches up to three times, once per stage, and the reference each stage reads must be the same number. This verb is therefore called **once**, before any solve, against the first fit of the wrapped estimator and the prior probabilities that fit was read at. A later stage carries a refit `pr` whose weights are the previous stage's posterior, so a reference resolved there would state a different target at each stage, and the view the caller wrote would move under it. Resolving once also lifts the whole parse out of the search, which repeated it at every stage. Issue #628 owns that rule.

# Algorithm

 1. Wrap `cvar_views` in a vector when it is a single view, so one loop serves both shapes.
 2. For each view group, read its significance level into `alpha`, and parse its equations accepting `==` alone.
 3. Replace every group name by the assets it spans, and every `prior(...)` reference by the prior conditional value at risk at `alpha` under `w`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the equality block `lcs`, and check the two preconditions of the section below that read it.
 5. For each row of the block, read the asset it names into `cols`, its target into `B`, its level into `alphas` and its text into `eqns`. The groups flatten into one search: each level enters only as the divisor of its own view's positive part.
 6. Read the worst realisation of every named asset into `min_X`, and raise when any target reaches it.
 7. Choose the search `d_opt`. One view takes `ds_opt`, or a default [`ConditionalValueatRiskEntropyPooling`](@ref). More than one takes `dm_opt`, or a default [`OptimEntropyPooling`](@ref) over `Optim.Fminbox`.
 8. Return the loss columns `X`, the targets `B`, the levels `alphas` and the search `d_opt`.

# Arguments

  - `cvar_views`: CVaR view constraints.
  - `pr`: Prior result the `prior(...)` references are read from. It is the first fit of the wrapped estimator, not a refit of a later stage.
  - `sets`: Asset set mapping asset names to indices.
  - `w`: Observation weights `pr` was read at, which the `prior(...)` references resolve under.
  - `ds_opt`: CVaR-specific optimiser (for single view).
  - `dm_opt`: General optimiser (for multiple views).
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Validation

  - No view group declares a view formulation. A group whose `alg` is not `nothing` raises an `ArgumentError`: this route writes no constraint formulation, so it has nothing to apply the declaration to.
  - Every view names one asset. A view over more than one asset raises an `ArgumentError`.
  - Every target is non-negative. A negative target raises a `DomainError`.
  - Every target stays below the worst realisation of the asset it names. A target that reaches it raises an `ArgumentError` naming every offending view beside the largest target its asset admits.

# Returns

  - `cvv::NamedTuple`: The search data, carrying the loss columns `X`, the targets `B`, the levels `alphas` and the search `d_opt`.

# Related

  - [`ep_cvar_views_solve!`](@ref): runs the search this verb prepares.
  - [`replace_prior_views`](@ref): resolves the `prior(...)` reference a target may carry.
  - [`get_pr_value`](@ref): reads the prior conditional value at risk that reference resolves to.
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucciardiakeel2011])
"""
function ep_cvar_views_setup(cvar_views::CVV_VecCVV, pr::AbstractPriorResult,
                             sets::UniverseSets, w::StatsBase.ProbabilityWeights,
                             ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                             dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false)
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
        views = replace_group_by_assets(views, sets, false, true, false)
        views = replace_prior_views(views, pr, sets, :cvar, alpha, w; strict = strict)
        lcs = get_linear_constraints(views, sets; datatype = eltype(X0), strict = strict)
        @argcheck(!any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2)),
                  ArgumentError("Cannot mix multiple assets in a single cvar_view.\n$(views)"))
        @argcheck(!any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq),
                  DomainError("cvar_views cannot be negative.\n$(views)"))
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
    X = view(X0, :, cols)
    min_X = dropdims(-minimum(X; dims = 1); dims = 1)
    invalid = B .>= min_X
    if any(invalid)
        msg = "The following views are too extreme, the maximum viable view for a given asset is its worst realisation:"
        for (v, m) in zip(eqns[invalid], min_X[invalid])
            msg *= "\n$v\t(> $m)."
        end
        msg *= "\nPlease lower the views or use a different prior with fatter tails."
        throw(ArgumentError(msg))
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
                         w::StatsBase.ProbabilityWeights,
                         opt::AbstractEntropyPoolingOptimiser)

Solve the entropy pooling problem when no CVaR views are specified.

`ep_cvar_views_solve!` is an internal API compatibility method that solves the entropy pooling problem when [`ep_cvar_views_setup`](@ref) prepared no search, which happens when no conditional value at risk view constraint is present. It delegates to the main entropy pooling solver using the provided prior weights, constraint dictionary, and optimiser.

# Algorithm

 1. Call [`entropy_pooling`](@ref) with the prior weights `w`, the constraint dictionary `epc` and the optimiser `opt`, giving the posterior probability weights `pw`.
 2. Return `pw`. With no conditional value at risk view there is no value at risk to search over, so the whole staged solve collapses to the one call of step 1.

# Arguments

  - `cvv`: Indicates that [`ep_cvar_views_setup`](@ref) prepared no search.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `w`: Prior probability weights.
  - `opt`: Entropy pooling optimiser.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_cvar_views_setup`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_cvar_views_solve!(cvv::Nothing, epc::AbstractDict,
                              w::StatsBase.ProbabilityWeights,
                              opt::AbstractEntropyPoolingOptimiser)
    return entropy_pooling(w, epc, opt)
end
"""
    ep_cvar_views_solve!(cvv::NamedTuple, epc::AbstractDict,
                         w::StatsBase.ProbabilityWeights,
                         opt::AbstractEntropyPoolingOptimiser)

Solve the entropy pooling problem with Conditional Value-at-Risk (CVaR) view constraints.

`ep_cvar_views_solve!` reads the targets, the levels and the search that [`ep_cvar_views_setup`](@ref) resolved, and solves for the posterior probability weights by either root-finding (for a single CVaR view) or optimisation (for more than one). It parses nothing and resolves nothing: the staged route calls it once per stage, and every stage must read the same targets.

The search runs over the value at risk levels `etas`, one per view, each bounded by `[0, B]`. This is a continuous relaxation of the recursive algorithm, which searches over discrete tail sizes instead; it reaches the same target and it takes more than one view.

The single-view bracket stops a hair inside `B`, at `B * (1 - sqrt(eps))`. At `B` itself the constraint demands a posterior tail contribution of exactly zero, and no interior posterior carries one, so the problem there is degenerate and its dual is unbounded. `Roots` evaluates both ends of a bracket before it searches, so every single-view solve would run that problem once. The root sits near half of `B`, so the shrunk end holds it.

# Mathematical definition

The conditional value at risk of asset ``i`` is the value of the Rockafellar-Uryasev programme, whose minimiser is the value at risk. A view that pins it to ``\\bar{c}`` is therefore a pair of conditions on the posterior, one linear in ``\\boldsymbol{p}`` at a fixed ``\\eta`` and one that fixes ``\\eta``:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(X) &= \\underset{\\eta}{\\min}\\; \\left\\{ \\eta + \\frac{1}{\\alpha} \\sum_{t=1}^{T} p_{t} \\left(-x_{t,\\,i} - \\eta\\right)^{+} \\right\\}\\,, \\\\
\\frac{1}{\\alpha} \\sum_{t=1}^{T} p_{t} \\left(-x_{t,\\,i} - \\eta\\right)^{+} &= \\bar{c} - \\eta\\,, \\\\
\\sum_{t=1}^{T} p_{t} \\mathbb{1}\\left\\{-x_{t,\\,i} > \\eta\\right\\} &= \\alpha\\,.
\\end{align}
```

The second line is linear in ``\\boldsymbol{p}``, so one entropy pooling solve answers it at any candidate ``\\eta``. The third line holds exactly when that ``\\eta`` is the value at risk of the posterior it produced, so the two lines share one fixed point and the view is met only there.

Where:

  - $(math_dict[:cvar_stat])
  - $(math_dict[:cvar_target])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``\\eta``: Value at risk of asset ``i`` at level ``\\alpha``, the minimiser of the first line and the variable the outer search carries.
  - ``(\\cdot)^{+}``: Positive part, ``\\max(\\cdot,\\, 0)``.

# Algorithm

 1. Read the loss columns `X`, the targets `B`, the levels `alphas` and the search `d_opt` off `cvv`.
 2. Define `func(etas)`. It writes the second line above into `epc` under the key `:cvar_eq` at the candidate `etas`, solves the whole constraint set with [`entropy_pooling`](@ref) into `wi`, and returns `wi` beside the residual of the third line. One view residual is the posterior tail mass minus `alpha`. More than one is the [`norm_error`](@ref) of the posterior conditional value at risk minus the target, over the views.
 3. Search for the value at risk `res`. One view root-finds the residual of `func` over `[0, B[1] * (1 - sqrt(eps))]` with `Roots.find_zero`. More than one minimises it over the box `[0, B]` from the start `0.5 * B` with `Optim.optimize`.
 4. Call `func(res)` once more, and return the posterior probability weights it produces.

# Arguments

  - `cvv`: Search data, answered by [`ep_cvar_views_setup`](@ref).
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `w`: Prior probability weights the stage projects.
  - `opt`: Main entropy pooling optimiser.

# Validation

  - Every candidate value at risk stays in `[0, B]`. A candidate outside the box raises a `DomainError`.
  - The search must succeed. A `Roots.find_zero` that raises is rethrown as an `ErrorException`, and an `Optim.optimize` that `Optim.converged` reports as failed raises an `ErrorException`.
  - An infeasible view set is **not** caught. The residual this search minimises is the posterior tail mass minus `alpha`, and a posterior that sits on one observation carries a small residual while it misses the view by any margin. The summary paragraph of [`MeucciEntropyPoolingPrior`](@ref) states how to recognise that answer.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying CVaR view constraints.

# Related

  - [`ep_cvar_views_setup`](@ref): resolves the targets this search reads.
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref): its stopping rule sets how closely each inner solve meets the constraint this search re-solves, and its own tip states the size of that.
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`entropy_pooling`](@ref)
  - [`norm_error`](@ref): scores the residual of a search over more than one view.
  - [`ConditionalValueatRisk`](@ref): reads the posterior statistic that residual scores.

# References

  - $(ref_dict[:meucciardiakeel2011])
"""
function ep_cvar_views_solve!(cvv::NamedTuple, epc::AbstractDict,
                              w::StatsBase.ProbabilityWeights,
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
    prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute entropy pooling prior moments for asset returns.

`prior` orients the data with respect to `dims` and delegates to [`ep_prior`](@ref), which dispatches on the entropy pooling algorithm `pe.alg`. [`H0_EntropyPooling`](@ref) enforces every view in a single optimisation. [`StagedEP`](@ref), the union of [`H1_EntropyPooling`](@ref) and [`H2_EntropyPooling`](@ref), enforces the views in stages, from lower to higher moments.

# Algorithm

 1. Orient `X` and `F` to observations by assets with [`dims_oriented`](@ref), so every step below reads one observation per row.
 2. Dispatch on `pe.alg` through [`ep_prior`](@ref), and return the [`LowOrderPrior`](@ref) it produces.

# Arguments

  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - $(arg_dict[:dims])
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`ep_prior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    return ep_prior(pe.alg, pe, X, F; strict = strict, kwargs...)
end
"""
    ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
             strict::Bool = false, kwargs...)

Compute entropy pooling prior moments for asset returns with iterative constraint enforcement.

`ep_prior` estimates the mean and covariance of asset returns using the entropy pooling framework, supporting iterative constraint enforcement via the `H1_EntropyPooling` and `H2_EntropyPooling` algorithms. It integrates moment and view constraints (mean, variance, CVaR, skewness, kurtosis, correlation), flexible confidence specification, and composable optimisation algorithms. The method iteratively applies constraints, updating prior weights and moments at each step, and ensures that higher moment views do not inadvertently alter lower moments.

Each stage refits the wrapped estimator before it parses its own views, so a `prior(...)` reference in a later stage resolves against the **previous stage's posterior** rather than against the estimator's own prior. A `sigma_views` target written as `prior(A)*1.3` under a `mu_views` or `cvar_views` view therefore asks for 1.3 times the variance stage one left, which is not 1.3 times the empirical variance. Write the target as a number where the reference must be the empirical one.

# Mathematical definition

The staged posterior is a chain of Kullback-Leibler projections rather than one. Stage ``k`` carries the constraint set ``\\mathcal{C}_{k}`` of every stage up to and including itself, and it projects a reference ``\\boldsymbol{r}^{(k)}`` that the algorithm tag fixes:

```math
\\begin{align}
\\boldsymbol{p}^{(k)} &= \\underset{\\boldsymbol{p} \\in \\mathcal{C}_{k}}{\\arg\\min} \\sum_{t=1}^{T} p_{t} \\ln\\!\\frac{p_{t}}{r_{t}^{(k)}}\\,, \\\\
\\mathcal{C}_{k} &= \\left\\{ \\boldsymbol{p} : \\mathbf{A}_{k} \\boldsymbol{p} = \\boldsymbol{B}_{k},\\; \\boldsymbol{p} \\geq \\boldsymbol{0},\\; \\boldsymbol{1}^\\intercal \\boldsymbol{p} = 1 \\right\\}\\,, \\\\
\\mathcal{C}_{1} &\\supseteq \\mathcal{C}_{2} \\supseteq \\mathcal{C}_{3}\\,, \\\\
\\boldsymbol{r}^{(k)} &= \\begin{cases} \\boldsymbol{q} & \\text{under } \\texttt{H1\\_EntropyPooling} \\\\ \\boldsymbol{p}^{(k-1)} & \\text{under } \\texttt{H2\\_EntropyPooling} \\end{cases}\\,, \\\\
\\boldsymbol{p}^{*} &= \\boldsymbol{p}^{(K)}\\,.
\\end{align}
```

The three stages hold the mean and value at risk views, the variance and covariance views, and the correlation, skewness and kurtosis views. The sets nest, so the last stage's posterior meets every view, and the two references differ only in what the chain is measured from: ``\\texttt{H1\\_EntropyPooling}`` gives the projection of the prior onto the whole view set, and ``\\texttt{H2\\_EntropyPooling}`` gives the projection of each stage onto the next. A conditional value at risk view is not a row of ``\\mathbf{A}_{k}``: it enters every stage as the fixed point of [`ep_cvar_views_solve!`](@ref), whose own section states it.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:T])
  - ``\\boldsymbol{p}^{(k)}``: Posterior probabilities of stage ``k``, and ``\\boldsymbol{p}^{*}`` those of the last stage ``K``.
  - ``\\boldsymbol{r}^{(k)}``: Reference probabilities stage ``k`` projects.
  - ``\\mathcal{C}_{k}``: Constraint set of stage ``k``, carrying the rows of every stage up to it.
  - ``\\mathbf{A}_{k}``, ``\\boldsymbol{B}_{k}``: Rows and right-hand side that state ``\\mathcal{C}_{k}``.

Posterior moments are then read as probability-weighted sample statistics under ``\\boldsymbol{p}^{*}``.

# Algorithm

 1. Read the prior probabilities into `w0`. When `pe.w` is `nothing` they are the uniform `1/T`, and otherwise they are `pe.w`, whose length must match `T`.
 2. Build the empty constraint dictionary `epc`, the fixing ledger `fixed`, and fit the wrapped estimator under `w0` into the prior result `pr`.
 3. Resolve the `cvar_views` once against that first fit under `w0`, through [`ep_cvar_views_setup`](@ref) into `cvv`. Every stage searches the targets it holds, so a `prior(...)` reference states one number for the whole chain.
 4. Stage one, the mean and the value at risk. Write the `mu_views` and `var_views` rows into `epc`. When any of `mu_views`, `var_views` and `cvar_views` is present, solve through [`ep_cvar_views_solve!`](@ref) into `w1`, and refit `pr` under it.
 5. Stage two, the variance and the covariance. Write the `sigma_views` and `cov_views` rows into `epc`, and pin the mean of every asset those rows read with [`fix_mu!`](@ref), so the stage cannot move a moment an earlier stage set. Solve into `w1`, and refit `pr` under it.
 6. Stage three, the correlation, the skewness and the kurtosis. Write the `sk_views`, `kt_views` and `rho_views` rows into `epc`, and pin the mean and the variance of every asset those rows read with [`fix_mu!`](@ref) and [`fix_sigma!`](@ref). Solve into `w1`, and refit `pr` under it.
 7. Read the reference each of steps 5 and 6 solves from: `w0` under [`H1_EntropyPooling`](@ref), and the previous stage's `w1` under [`H2_EntropyPooling`](@ref).
 8. Read the effective number of scenarios `ens` as the exponential of the entropy of `w1`, and the divergence `kld` as the Kullback-Leibler divergence of `w1` from `w0`.
 9. Return a [`LowOrderPrior`](@ref) carrying the last refit's moments, `w1`, `ens` and `kld`. The feature matrix `Z` and the factor block `fpr` are forwarded from that refit unchanged.

# Arguments

  - `alg`: Staged entropy pooling algorithm, taken from `pe.alg` by [`prior`](@ref).
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets), oriented by [`prior`](@ref).
  - `F`: Optional factor matrix, oriented by [`prior`](@ref).
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Validation

  - If any view constraint is not `nothing`, `!isnothing(sets)`.
  - If prior weights `pe.w` are provided, `length(pe.w) == T`, where `T` is the number of observations. A length that does not match raises a `DimensionMismatch`.
  - Every view equation carries a comparison operator its own family accepts. An unsupported operator raises a `Meta.ParseError` naming the operators that family accepts.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

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
                  F::Option{<:MatNum}; strict::Bool = false, kwargs...)
    T, N = size(X)
    w1 = w0 = if isnothing(pe.w)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T,
                  DimensionMismatch("length(pe.w) ($(length(pe.w))) must match T ($T)"))
        pe.w
    end
    fixed = falses(N, 2)
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    # mu and VaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # Every `prior(...)` reference resolves against this first fit, under `w0`. The CVaR
    # search runs once per stage against a refit `pr`, so resolving inside it would state a
    # different target at each stage. It is resolved once, here, and the stages read it.
    cvv = ep_cvar_views_setup(pe.cvar_views, pr, pe.sets, w0, pe.ds_opt, pe.dm_opt;
                              strict = strict)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, w0; strict = strict)
    if !isnothing(pe.mu_views) || !isnothing(pe.var_views) || !isnothing(pe.cvar_views)
        w1 = ep_cvar_views_solve!(cvv, epc, w0, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            to_fix = ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        end
        # cov
        if !isnothing(pe.cov_views)
            to_fix = ep_cov_views!(pe.cov_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        end
        w1 = ep_cvar_views_solve!(cvv, epc, ifelse(isa(alg, H1_EntropyPooling), w0, w1),
                                  pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            to_fix = ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            to_fix = ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # rho
        if !isnothing(pe.rho_views)
            to_fix = ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        w1 = ep_cvar_views_solve!(cvv, epc, ifelse(isa(alg, H1_EntropyPooling), w0, w1),
                                  pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole. It is *not* stamped with the
    # pooled weights: `w1` is threaded into the wrapped estimator by `factory` above, so a
    # factor prior that records its own weighting will report it here, and one that does not
    # (`EmpiricalPrior` never sets `w`) is a gap to close at that producer rather than to
    # paper over from out here — see #217. Writing `w1` on would also owe the factor block
    # `ens`/`kld` under ADR 0046's binding, which is the coupling that made the flat `f_w` a
    # duplicate of `w` at five of six producers in the first place.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, Z = Z)
end
"""
    ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}; strict::Bool = false, kwargs...)

Compute entropy pooling prior moments for asset returns with single-shot constraint enforcement.

`ep_prior` estimates the mean and covariance of asset returns using the entropy pooling framework, enforcing all moment and view constraints in a single optimisation step via the `H0_EntropyPooling` algorithm. This approach is fast but may distort lower moments when higher moment views are present, as all constraints are applied simultaneously.

# Mathematical definition

The single-shot posterior is one Kullback-Leibler projection of the prior onto the intersection of every view's constraint set. It is the staged chain of [`ep_prior`](@ref) collapsed to one stage:

```math
\\begin{align}
\\boldsymbol{p}^{*} &= \\underset{\\boldsymbol{p} \\in \\mathcal{C}}{\\arg\\min} \\sum_{t=1}^{T} p_{t} \\ln\\!\\frac{p_{t}}{q_{t}}\\,, \\\\
\\mathcal{C} &= \\mathcal{C}_{1} \\cap \\mathcal{C}_{2} \\cap \\mathcal{C}_{3}\\,.
\\end{align}
```

One reference and one feasible set carry every view, so no moment can be pinned between stages: a higher moment view is free to move a lower moment of the same asset. That is the whole difference from the staged chain, whose ``\\mathcal{C}_{k}`` this ``\\mathcal{C}`` intersects. A conditional value at risk view is not a row of ``\\mathcal{C}``: it enters as the fixed point of [`ep_cvar_views_solve!`](@ref), whose own section states it.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:T])
  - ``\\boldsymbol{p}^{*}``: Posterior probabilities of the one solve.
  - ``\\mathcal{C}``, ``\\mathcal{C}_{k}``: Feasible set of the one solve, and the stage sets it intersects.

Posterior moments are then read as probability-weighted sample statistics under ``\\boldsymbol{p}^{*}``.

# Algorithm

 1. Read the prior probabilities into `w0`. When `pe.w` is `nothing` they are the uniform `1/T`, and otherwise they are `pe.w`, whose length must match `T`.
 2. Build the empty constraint dictionary `epc`, and fit the wrapped estimator under `w0` into the prior result `pr`.
 3. Resolve the `cvar_views` once against that fit under `w0`, through [`ep_cvar_views_setup`](@ref) into `cvv`.
 4. Write the `mu_views` and `var_views` rows into `epc`.
 5. Write the `sigma_views` and `cov_views` rows into `epc`. No moment is pinned, so [`fix_mu!`](@ref) is never called on this route.
 6. Write the `sk_views`, `kt_views` and `rho_views` rows into `epc`.
 7. Solve the whole accumulated set once through [`ep_cvar_views_solve!`](@ref) into `w1`, and refit `pr` under it.
 8. Read the effective number of scenarios `ens` as the exponential of the entropy of `w1`, and the divergence `kld` as the Kullback-Leibler divergence of `w1` from `w0`.
 9. Return a [`LowOrderPrior`](@ref) carrying the refit's moments, `w1`, `ens` and `kld`. The feature matrix `Z` and the factor block `fpr` are forwarded from that refit unchanged.

# Arguments

  - `alg`: Single-shot entropy pooling algorithm, taken from `pe.alg` by [`prior`](@ref).
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets), oriented by [`prior`](@ref).
  - `F`: Optional factor matrix, oriented by [`prior`](@ref).
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Validation

  - If any view constraint is not `nothing`, `!isnothing(pe.sets)`.
  - If prior weights `pe.w` are provided, `length(pe.w) == T`, where `T` is the number of observations. A length that does not match raises a `DimensionMismatch`.
  - Every view equation carries a comparison operator its own family accepts. An unsupported operator raises a `Meta.ParseError` naming the operators that family accepts.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

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
                  F::Option{<:MatNum}; strict::Bool = false, kwargs...)
    T = size(X, 1)
    w0 = if isnothing(pe.w)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T,
                  DimensionMismatch("length(pe.w) ($(length(pe.w))) must match T ($T)"))
        pe.w
    end
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    # mu and VaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # Every `prior(...)` reference resolves against this fit, under `w0`.
    cvv = ep_cvar_views_setup(pe.cvar_views, pr, pe.sets, w0, pe.ds_opt, pe.dm_opt;
                              strict = strict)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, w0; strict = strict)
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
        end
        # cov
        if !isnothing(pe.cov_views)
            ep_cov_views!(pe.cov_views, epc, pr, pe.sets; strict = strict)
        end
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
        end
        # rho
        if !isnothing(pe.rho_views)
            ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
        end
    end
    w1 = ep_cvar_views_solve!(cvv, epc, w0, pe.opt)
    pe = factory(pe, w1)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole. It is *not* stamped with the
    # pooled weights: `w1` is threaded into the wrapped estimator by `factory` above, so a
    # factor prior that records its own weighting will report it here, and one that does not
    # (`EmpiricalPrior` never sets `w`) is a gap to close at that producer rather than to
    # paper over from out here — see #217. Writing `w1` on would also owe the factor block
    # `ens`/`kld` under ADR 0046's binding, which is the coupling that made the flat `f_w` a
    # duplicate of `w` at five of six producers in the first place.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, Z = Z)
end

function factor_residual_config(pe::MeucciEntropyPoolingPrior)
    return factor_residual_config(pe.pe)
end

export MeucciEntropyPoolingPrior
