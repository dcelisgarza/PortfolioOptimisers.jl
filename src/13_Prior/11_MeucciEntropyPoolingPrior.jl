"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments meet a set of views, and root-finds a CVaR view.

`MeucciEntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports views on the mean, the value at risk, the conditional value at risk, the variance, the covariance, the correlation, the skewness and the kurtosis, and it takes custom prior weights and solver configuration.

This is the earlier of the library's two entropy pooling estimators, and it is kept because its CVaR route is a different *algorithm*, not a different formulation of the same problem: a CVaR view is a target hunted by the recursive algorithm of Meucci, Ardia and Keel, where [`ConditionalValueatRiskEntropyPooling`](@ref) root-finds the value at risk level and re-solves the whole entropy pooling problem at each candidate. That route takes equality CVaR views alone, one asset per view.

Reach for [`EntropyPoolingPrior`](@ref) instead where a tail view has to be an inequality, name two assets, or land on the entropic value at risk: there each tail view is a constraint of the single entropy pooling problem, so one solve answers every view.

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

# Details

  - If `w` is not `nothing`, it is normalised to sum to 1; otherwise, uniform weights are used when `prior` is called.

# View comparison operators

The comparison operators accepted in each view's constraint strings depend on the moment being constrained. An unsupported operator raises a `ParseError` listing the operators allowed for that view.

  - `mu_views`, `sigma_views`, `sk_views`, `kt_views`, `cov_views`, `rho_views` accept `==`, `>=` and `<=`.
  - `var_views` (Value at Risk) accepts only `==` and `>=`.
  - `cvar_views` (Conditional Value at Risk) accepts only `==`.

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

Alias for an abstract vector of [`MeucciEntropyPoolingPrior`](@ref) elements.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
"""
const VecMeucciEP = AbstractVector{<:MeucciEntropyPoolingPrior}
"""
    ep_cvar_views_solve!(cvar_views::Nothing, epc::AbstractDict, ::Any, ::Any,
                         w::StatsBase.ProbabilityWeights, opt::AbstractEntropyPoolingOptimiser, ::Any, ::Any;
                         kwargs...)

Solve the entropy pooling problem when no CVaR views are specified.

`ep_cvar_views_solve!` is an internal API compatibility method that solves the entropy pooling problem when no conditional value at risk (CVaR) view constraints are present (`cvar_views = nothing`). It delegates to the main entropy pooling solver using the provided prior weights, constraint dictionary, and optimiser.

# Arguments

  - `cvar_views`: Indicates that no CVaR view constraints are specified.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `::Any`: Prior result, ignored on this path.
  - `::Any`: Asset set, ignored on this path.
  - `w`: Prior probability weights.
  - `opt`: Entropy pooling optimiser.
  - `::Any`: CVaR-specific optimiser, ignored on this path.
  - `::Any`: General optimiser, ignored on this path.
  - `kwargs...`: Additional keyword arguments forwarded to the solver.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Details

  - This method is used for API compatibility when CVaR views are not present.
  - Calls [`entropy_pooling`](@ref) with the provided arguments.

# Related

  - [`entropy_pooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_cvar_views_solve!(cvar_views::Nothing, epc::AbstractDict, ::Any, ::Any,
                              w::StatsBase.ProbabilityWeights,
                              opt::AbstractEntropyPoolingOptimiser, ::Any, ::Any; kwargs...)
    return entropy_pooling(w, epc, opt)
end
"""
    ep_cvar_views_solve!(cvar_views::CVV_VecCVV, epc::AbstractDict,
                         pr::AbstractPriorResult, sets::UniverseSets,
                         w::StatsBase.ProbabilityWeights, opt::AbstractEntropyPoolingOptimiser,
                         ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                         dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false)

Solve the entropy pooling problem with Conditional Value-at-Risk (CVaR) view constraints.

`ep_cvar_views_solve!` parses and validates CVaR view constraints, replaces prior references, and constructs the corresponding entropy pooling constraint system. It then solves for posterior probability weights using either root-finding (for single CVaR view) or optimisation (for multiple views), depending on the number of constraints and the provided optimiser. Throws informative errors if views are infeasible or too extreme.

# Arguments

  - `cvar_views`: CVaR view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `w`: Prior probability weights.
  - `opt`: Main entropy pooling optimiser.
  - `ds_opt`: CVaR-specific optimiser (for single view).
  - `dm_opt`: General optimiser (for multiple views).
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying CVaR view constraints.

# Details

  - Parses CVaR view equations and replaces prior references.
  - Validates that only equality constraints are present and that each view targets a single asset.
  - Checks that views are not too extreme i.e. not greater than the worst realisation.
  - The search runs over the value at risk levels `etas`, one per view, each bounded by `[0, B]`. This is a continuous relaxation of the recursive algorithm, which searches over discrete tail sizes instead; it reaches the same target and it takes more than one view.
  - For a single CVaR view, uses root-finding via [`ConditionalValueatRiskEntropyPooling`](@ref).
  - For multiple CVaR views, uses optimisation via [`OptimEntropyPooling`](@ref).
  - Throws errors if optimisation fails or views are infeasible.

# Related

  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`entropy_pooling`](@ref)

# References

  - $(ref_dict[:meucciardiakeel2011])
"""
function ep_cvar_views_solve!(cvar_views::CVV_VecCVV, epc::AbstractDict,
                              pr::AbstractPriorResult, sets::UniverseSets,
                              w::StatsBase.ProbabilityWeights,
                              opt::AbstractEntropyPoolingOptimiser,
                              ds_opt::Option{<:ConditionalValueatRiskEntropyPooling},
                              dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false)
    X0 = pr.X
    if !isa(cvar_views, AbstractVector)
        cvar_views = [cvar_views]
    end
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
        views = replace_prior_views(views, pr, sets, :cvar, alpha; strict = strict)
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
    N = length(B)
    d_opt = if N == 1
        ifelse(!isnothing(ds_opt), ds_opt, ConditionalValueatRiskEntropyPooling())
    else
        ifelse(!isnothing(dm_opt), dm_opt,
               OptimEntropyPooling(;
                                   args = (Optim.Fminbox(),
                                           Optim.Options(; outer_x_abstol = 1e-4,
                                                         x_abstol = 1e-4))))
    end
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
        try
            [Roots.find_zero(x -> func(x)[2], (0, B[1]), d_opt.args...; d_opt.kwargs...)]
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

# Mathematical definition

Entropy pooling finds posterior weights ``\\boldsymbol{p}`` by minimising the Kullback-Leibler divergence from the prior ``\\boldsymbol{q}``:

```math
\\begin{align}
\\underset{\\boldsymbol{p}}{\\min} &\\sum_{t=1}^{T} p_t \\ln\\!\\frac{p_t}{q_t} \\quad \\text{s.t.} \\quad \\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} = \\boldsymbol{b}_{\\mathrm{eq}}, \\quad \\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} \\leq \\boldsymbol{b}_{\\mathrm{ineq}}, \\quad \\boldsymbol{p} \\geq \\boldsymbol{0}, \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{p} = 1\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}``: ``T \\times 1`` posterior weight vector.
  - ``\\boldsymbol{q}``: ``T \\times 1`` prior weight vector.
  - ``\\mathbf{A}_{\\mathrm{eq}}``, ``\\boldsymbol{b}_{\\mathrm{eq}}``: Equality constraint matrix and vector.
  - ``\\mathbf{A}_{\\mathrm{ineq}}``, ``\\boldsymbol{b}_{\\mathrm{ineq}}``: Inequality constraint matrix and vector.
  - $(math_dict[:T])

Posterior moments are then computed as probability-weighted sample statistics using ``\\boldsymbol{p}^*``.

# Arguments

  - `alg`: Staged entropy pooling algorithm, taken from `pe.alg` by [`prior`](@ref).
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets), oriented by [`prior`](@ref).
  - `F`: Optional factor matrix, oriented by [`prior`](@ref).
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Validation

  - If any view constraint is not `nothing`, `!isnothing(sets)`.
  - If prior weights `pe.w` are provided, `length(pe.w) == T`, where `T` is the number of observations.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Details

  - If `isnothing(pe.w)`, prior weights are initialised to `1/T` where `T` is the number of observations; otherwise, provided weights are normalised.
  - Constraints are enforced iteratively, from lower to higher moments.
  - Moment and view constraints are parsed and added to the constraint dictionary.
  - The initial weights for each stage is selected according to `pe.alg`.
  - At each stage, the prior weights are updated by solving the entropy pooling optimisation with the current set of constraints. If present, the CVaR views are also enforced at every stage.
  - Lower moments are fixed as needed to prevent distortion by higher moment views. If asset `i` has a view enforced on moment `N` that uses moments `n < N` to compute, then all moments `n` for asset `i` are fixed.
  - The final result includes the effective number of scenarios and Kullback-Leibler divergence between prior and posterior weights.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`prior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`StagedEP`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`ep_mu_views!`](@ref)
  - [`ep_var_views!`](@ref)
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
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets; strict = strict)
    if !isnothing(pe.mu_views) || !isnothing(pe.var_views) || !isnothing(pe.cvar_views)
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, w0, pe.opt, pe.ds_opt,
                                  pe.dm_opt; strict = strict)
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
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets,
                                  ifelse(isa(alg, H1_EntropyPooling), w0, w1), pe.opt,
                                  pe.ds_opt, pe.dm_opt; strict = strict)
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
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets,
                                  ifelse(isa(alg, H1_EntropyPooling), w0, w1), pe.opt,
                                  pe.ds_opt, pe.dm_opt; strict = strict)
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

Entropy pooling finds posterior weights ``\\boldsymbol{p}`` by minimising the Kullback-Leibler divergence from the prior ``\\boldsymbol{q}`` subject to all constraints simultaneously:

```math
\\begin{align}
\\underset{\\boldsymbol{p}}{\\min} &\\sum_{t=1}^{T} p_t \\ln\\!\\frac{p_t}{q_t} \\quad \\text{s.t.} \\quad \\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} = \\boldsymbol{b}_{\\mathrm{eq}}, \\quad \\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} \\leq \\boldsymbol{b}_{\\mathrm{ineq}}, \\quad \\boldsymbol{p} \\geq \\boldsymbol{0}, \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{p} = 1\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}``: ``T \\times 1`` posterior weight vector.
  - ``\\boldsymbol{q}``: ``T \\times 1`` prior weight vector.
  - ``\\mathbf{A}_{\\mathrm{eq}}``, ``\\boldsymbol{b}_{\\mathrm{eq}}``: Equality constraint matrix and vector.
  - ``\\mathbf{A}_{\\mathrm{ineq}}``, ``\\boldsymbol{b}_{\\mathrm{ineq}}``: Inequality constraint matrix and vector.
  - $(math_dict[:T])

# Arguments

  - `alg`: Single-shot entropy pooling algorithm, taken from `pe.alg` by [`prior`](@ref).
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets), oriented by [`prior`](@ref).
  - `F`: Optional factor matrix, oriented by [`prior`](@ref).
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Validation

  - If any view constraint is not `nothing`, `!isnothing(pe.sets)`.
  - If prior weights `pe.w` are provided, `length(pe.w) == T`, where `T` is the number of observations

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Details

  - If `isnothing(pe.w)`, prior weights are initialised to `1/T` where `T` is the number of observations; otherwise, provided weights are normalised.
  - All constraints are parsed and added to the constraint dictionary at once. This means that lower moments may be distorted by higher moment views, since they cannot be fixed at any point.
  - A single optimisation is performed to solve for the posterior weights, enforcing all constraints at once.
  - The final result includes the effective number of scenarios and Kullback-Leibler divergence between prior and posterior weights.

# Related

  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`prior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`ep_mu_views!`](@ref)
  - [`ep_var_views!`](@ref)
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
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets; strict = strict)
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
    w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, w0, pe.opt, pe.ds_opt,
                              pe.dm_opt; strict = strict)
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
