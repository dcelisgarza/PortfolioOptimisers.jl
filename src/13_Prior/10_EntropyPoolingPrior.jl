"""
$(DocStringExtensions.TYPEDEF)

Structured result for correlation view constraint equation parsing.

`RhoParsingResult` is produced when parsing correlation view constraints, such as those used in entropy pooling prior models. It extends the standard [`ParsingResult`](@ref) by including an `ij` field, which stores the tuple of asset pairs (indices) relevant for the correlation view.

# Fields

$(DocStringExtensions.FIELDS)

# Details

  - Produced by correlation view parsing routines, typically when the constraint involves asset pairs (e.g., `"(A, B) == 0.5"`).
  - The `ij` field enables downstream routines to map parsed correlation views to the appropriate entries in the correlation matrix.
  - Used internally for entropy pooling, Black-Litterman, and other advanced portfolio models that support correlation views.

# Related

  - [`AbstractParsingResult`](@ref)
  - [`ParsingResult`](@ref)
  - [`replace_prior_views`](@ref)
"""
@concrete struct RhoParsingResult <: AbstractParsingResult
    """
    $(field_dict[:vars])
    """
    vars
    """
    $(field_dict[:coef_c])
    """
    coef
    """
    $(field_dict[:op])
    """
    op
    """
    $(field_dict[:rhs])
    """
    rhs
    """
    $(field_dict[:eqn])
    """
    eqn
    """
    $(field_dict[:ij])
    """
    ij
    function RhoParsingResult(vars::VecStr, coef::VecNum, op::AbstractString, rhs::Number,
                              eqn::AbstractString,
                              ij::AbstractVector{<:Union{<:Tuple{<:Integer, <:Integer},
                                                         <:Tuple{<:VecInt, <:VecInt}}})
        @argcheck(length(vars) == length(coef), DimensionMismatch)
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn),
                   typeof(ij)}(vars, coef, op, rhs, eqn, ij)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for entropy pooling optimisers.

`AbstractEntropyPoolingOptimiser` is the base type for all optimisers that compute entropy pooling weights subject to moment and view constraints. All concrete entropy pooling optimisers should subtype this type to ensure a consistent interface for entropy pooling routines and integration with portfolio optimisation workflows.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
abstract type AbstractEntropyPoolingOptimiser <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for entropy pooling algorithms.

`AbstractEntropyPoolingAlgorithm` is the base type for all algorithms used in entropy pooling optimisation routines. All concrete entropy pooling algorithms should subtype this type to ensure a consistent interface for entropy pooling methods and integration with portfolio optimisation workflows.

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

One-shot entropy pooling. It sets and optimises all the constraints simultaneously. This introduces bias in the posterior probabilities, but is faster.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Uses the initial probabilities to optimise the posterior probabilities at every step. This reduces bias in the posterior probabilities, but is slower.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Uses the previous step's probabilities to optimise the next step's probabilities. This is faster but may introduce bias.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
"""
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
    const StagedEP = Union{<:H1_EntropyPooling, <:H2_EntropyPooling}

Alias for a union of staged entropy pooling algorithm types.

# Related

  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
const StagedEP = Union{<:H1_EntropyPooling, <:H2_EntropyPooling}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for entropy pooling optimisation algorithms.

`AbstractEntropyPoolingOptAlgorithm` is the base type for all optimisation algorithms used in entropy pooling routines. All concrete entropy pooling optimisation algorithms should subtype this type to ensure a consistent interface for entropy pooling optimisation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
abstract type AbstractEntropyPoolingOptAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Logarithmic entropy pooling optimisation algorithm.

`LogEntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptAlgorithm`](@ref) representing the logarithmic entropy pooling optimisation algorithm. This algorithm solves for posterior probabilities by minimising the Kullback-Leibler divergence between the prior and posterior weights, subject to moment and view constraints, using a logarithmic objective.

# Related

  - [`AbstractEntropyPoolingOptAlgorithm`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
struct LogEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Exponential entropy pooling optimisation algorithm.

`ExpEntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptAlgorithm`](@ref) representing the exponential entropy pooling optimisation algorithm. This algorithm solves for posterior probabilities by minimising the exponential divergence between the prior and posterior weights, subject to moment and view constraints, using an exponential objective.

# Related

  - [`AbstractEntropyPoolingOptAlgorithm`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
struct ExpEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Conditional Value-at-Risk (CVaR) entropy pooling optimiser.

`CVaREntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptimiser`](@ref) that uses root-finding algorithms from [`Roots.jl`](https://github.com/JuliaMath/Roots.jl) to solve entropy pooling problems with CVaR (Conditional Value-at-Risk) view constraints. This optimiser is designed for scenarios where CVaR views are specified and requires robust numerical methods to find the solution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CVaREntropyPooling(;
        args::Tuple = (Roots.Brent(),),
        kwargs::NamedTuple = (;)
    ) -> CVaREntropyPooling

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CVaREntropyPooling()
CVaREntropyPooling
    args ┼ Tuple{Roots.Brent}: (Roots.Brent(),)
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`Roots.jl`](https://github.com/JuliaMath/Roots.jl)
"""
@concrete struct CVaREntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:optargs])
    """
    args
    """
    $(field_dict[:optkwargs])
    """
    kwargs
    function CVaREntropyPooling(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function CVaREntropyPooling(; args::Tuple = (Roots.Brent(),),
                            kwargs::NamedTuple = (;))::CVaREntropyPooling
    return CVaREntropyPooling(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)-based entropy pooling optimiser.

`OptimEntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptimiser`](@ref) that uses [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) to solve entropy pooling problems. This optimiser supports both logarithmic and exponential entropy pooling objectives, and allows for flexible configuration of solver arguments, scaling parameters, and algorithm selection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimEntropyPooling(;
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        sc1::Number = 1,
        sc2::Number = 1e3,
        alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling()
        err::Option{<:NormError} = nothing
    ) -> OptimEntropyPooling

Keywords correspond to the struct's fields.

## Validation

  - `sc1 >= 0`
  - `sc2 >= 0`

# Examples

```jldoctest
julia> OptimEntropyPooling()
OptimEntropyPooling
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
     sc1 ┼ Int64: 1
     sc2 ┼ Float64: 1000.0
     alg ┼ ExpEntropyPooling()
     err ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`CVaREntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)
  - [`NormError`](@ref)
"""
@concrete struct OptimEntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:optargs])
    """
    args
    """
    $(field_dict[:optkwargs])
    """
    kwargs
    """
    $(field_dict[:sc1])
    """
    sc1
    """
    $(field_dict[:sc2])
    """
    sc2
    """
    $(field_dict[:epoptalg])
    """
    alg
    """
    $(field_dict[:err]) Only used when there are multiple cvar views. If `nothing`, the L2 norm is used.
    """
    err
    function OptimEntropyPooling(args::Tuple, kwargs::NamedTuple, sc1::Number, sc2::Number,
                                 alg::AbstractEntropyPoolingOptAlgorithm,
                                 err::Option{<:NormError})
        @argcheck(sc1 >= zero(sc1), DomainError(sc1, "sc1 must be >= 0"))
        return new{typeof(args), typeof(kwargs), typeof(sc1), typeof(sc2), typeof(alg),
                   typeof(err)}(args, kwargs, sc1, sc2, alg, err)
    end
end
function OptimEntropyPooling(; args::Tuple = (), kwargs::NamedTuple = (;), sc1::Number = 1,
                             sc2::Number = 1e3,
                             alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling(),
                             err::Option{<:NormError} = nothing)::OptimEntropyPooling
    return OptimEntropyPooling(args, kwargs, sc1, sc2, alg, err)
end
"""
$(DocStringExtensions.TYPEDEF)

[`JuMP.jl`](https://github.com/jump-dev/JuMP.jl)-based entropy pooling optimiser.

`JuMPEntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptimiser`](@ref) that uses [JuMP.jl](https://github.com/jump-dev/JuMP.jl) to solve entropy pooling problems. This optimiser supports both logarithmic and exponential entropy pooling objectives, and allows for flexible configuration of solver arguments, scaling parameters, and algorithm selection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPEntropyPooling(;
        slv::Slv_VecSlv,
        sc1::Number = 1,
        sc2::Number = 1e5,
        so::Number = 1,
        alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling()
    ) -> JuMPEntropyPooling

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a vector, `!isempty(slv)`.
  - `sc1 >= 0`
  - `sc2 >= 0`
  - `so >= 0`

# Examples

```jldoctest
julia> JuMPEntropyPooling(; slv = Solver(; name = :fake_solver, solver = :MySolver))
JuMPEntropyPooling
  slv ┼ Solver
      │          name ┼ Symbol: :fake_solver
      │        solver ┼ Symbol: :MySolver
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
  sc1 ┼ Int64: 1
  sc2 ┼ Float64: 100000.0
   so ┼ Int64: 1
  alg ┴ ExpEntropyPooling()
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`CVaREntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl)
"""
@concrete struct JuMPEntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc1])
    """
    sc1
    """
    $(field_dict[:sc2])
    """
    sc2
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:epoptalg])
    """
    alg
    function JuMPEntropyPooling(slv::Slv_VecSlv, sc1::Number, sc2::Number, so::Number,
                                alg::AbstractEntropyPoolingOptAlgorithm)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc1 >= zero(sc1), DomainError(sc1, "sc1 must be >= 0"))
        @argcheck(sc2 >= zero(sc2), DomainError(sc2, "sc2 must be >= 0"))
        @argcheck(so >= zero(so), DomainError(so, "so must be >= 0"))
        return new{typeof(slv), typeof(sc1), typeof(sc2), typeof(so), typeof(alg)}(slv, sc1,
                                                                                   sc2, so,
                                                                                   alg)
    end
end
function JuMPEntropyPooling(; slv::Slv_VecSlv, sc1::Number = 1, sc2::Number = 1e5,
                            so::Number = 1,
                            alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())::JuMPEntropyPooling
    return JuMPEntropyPooling(slv, sc1, sc2, so, alg)
end
"""
    const NonCVaREP = Union{<:OptimEntropyPooling, <:JuMPEntropyPooling}

Alias for a union of non-CVaR entropy pooling algorithm types.

# Related

  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
const NonCVaREP = Union{<:OptimEntropyPooling, <:JuMPEntropyPooling}
"""
$(DocStringExtensions.TYPEDEF)

Entropy pooling prior estimator for asset returns.

`EntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports moment and view constraints (mean, variance, CVaR, covariance, skewness, kurtosis, correlation), flexible confidence specification, and composable optimisation algorithms. The estimator integrates asset sets, view constraints, and multiple entropy pooling algorithms (Optim.jl, JuMP.jl, CVaR root-finding), and allows for custom prior weights and solver configuration.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyPoolingPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        mu_views::Option{<:LinearConstraintEstimator} = nothing,
        var_views::Option{<:LinearConstraintEstimator} = nothing,
        cvar_views::Option{<:LinearConstraintEstimator} = nothing,
        sigma_views::Option{<:LinearConstraintEstimator} = nothing,
        sk_views::Option{<:LinearConstraintEstimator} = nothing,
        kt_views::Option{<:LinearConstraintEstimator} = nothing,
        cov_views::Option{<:LinearConstraintEstimator} = nothing,
        rho_views::Option{<:LinearConstraintEstimator} = nothing,
        var_alpha::Number = 0.05,
        cvar_alpha::Number = 0.05,
        sets::Option{<:AssetSets} = nothing,
        ds_opt::Option{<:CVaREntropyPooling} = nothing,
        dm_opt::Option{<:OptimEntropyPooling} = nothing,
        opt::NonCVaREP = OptimEntropyPooling(),
        w::Option{<:StatsBase.ProbabilityWeights} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling()
    ) -> EntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If any view constraint is not `nothing`, `sets` must not be `nothing`.
  - If not `nothing`, `0 < var_alpha < 1`.
  - If not `nothing`, `0 < cvar_alpha < 1`.
  - If `w` is not `nothing`, it must be non-empty and match the number of observations.

# Details

  - If `w` is not `nothing`, it is normalised to sum to 1; otherwise, uniform weights are used when `prior` is called.
  - If `var_views` is not `nothing` without `var_alpha`, defaults to `0.05`.
  - If `cvar_views` is not `nothing` without `cvar_alpha`, defaults to `0.05`.

# View comparison operators

The comparison operators accepted in each view's constraint strings depend on the moment being constrained. An unsupported operator raises a `ParseError` listing the operators allowed for that view.

  - `mu_views`, `sigma_views`, `sk_views`, `kt_views`, `cov_views`, `rho_views` accept `==`, `>=` and `<=`.
  - `var_views` (Value at Risk) accepts only `==` and `>=`.
  - `cvar_views` (Conditional Value at Risk) accepts only `==`.

# Examples

```jldoctest
julia> EntropyPoolingPrior(; sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"])),
                           mu_views = LinearConstraintEstimator(;
                                                                val = ["A == 0.03",
                                                                       "B + C == 0.04"]))
EntropyPoolingPrior
           pe ┼ EmpiricalPrior
              │        ce ┼ PortfolioOptimisersCovariance
              │           │   ce ┼ Covariance
              │           │      │    me ┼ SimpleExpectedReturns
              │           │      │       │   w ┴ nothing
              │           │      │    ce ┼ GeneralCovariance
              │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
              │           │      │       │    w ┴ nothing
              │           │      │   alg ┴ FullMoment()
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
    var_alpha ┼ nothing
   cvar_alpha ┼ nothing
         sets ┼ AssetSets
              │    key ┼ String: "nx"
              │   ukey ┼ String: "ux"
              │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
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
  - [`AssetSets`](@ref)
  - [`CVaREntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`AbstractEntropyPoolingAlgorithm`](@ref)
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
    $(field_dict[:var_alpha])
    """
    var_alpha
    """
    $(field_dict[:cvar_alpha])
    """
    cvar_alpha
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
    function EntropyPoolingPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mu_views::Option{<:LinearConstraintEstimator},
                                 var_views::Option{<:LinearConstraintEstimator},
                                 cvar_views::Option{<:LinearConstraintEstimator},
                                 sigma_views::Option{<:LinearConstraintEstimator},
                                 sk_views::Option{<:LinearConstraintEstimator},
                                 kt_views::Option{<:LinearConstraintEstimator},
                                 cov_views::Option{<:LinearConstraintEstimator},
                                 rho_views::Option{<:LinearConstraintEstimator},
                                 var_alpha::Option{<:Number}, cvar_alpha::Option{<:Number},
                                 sets::Option{<:AssetSets},
                                 ds_opt::Option{<:CVaREntropyPooling},
                                 dm_opt::Option{<:OptimEntropyPooling}, opt::NonCVaREP,
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
        if !isnothing(var_views)
            if !isnothing(var_alpha)
                @argcheck(zero(var_alpha) < var_alpha < one(var_alpha),
                          DomainError(var_alpha, "var_alpha must be in (0, 1)"))
            else
                var_alpha = 0.05
            end
        end
        if !isnothing(cvar_views)
            if !isnothing(cvar_alpha)
                @argcheck(zero(cvar_alpha) < cvar_alpha < one(cvar_alpha),
                          DomainError(cvar_alpha, "cvar_alpha must be in (0, 1)"))
            else
                cvar_alpha = 0.05
            end
        end
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(sigma_views), typeof(sk_views), typeof(kt_views),
                   typeof(cov_views), typeof(rho_views), typeof(var_alpha),
                   typeof(cvar_alpha), typeof(sets), typeof(ds_opt), typeof(dm_opt),
                   typeof(opt), typeof(w), typeof(alg)}(pe, mu_views, var_views, cvar_views,
                                                        sigma_views, sk_views, kt_views,
                                                        cov_views, rho_views, var_alpha,
                                                        cvar_alpha, sets, ds_opt, dm_opt,
                                                        opt, w, alg)
    end
end
function EntropyPoolingPrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             mu_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_views::Option{<:LinearConstraintEstimator} = nothing,
                             cvar_views::Option{<:LinearConstraintEstimator} = nothing,
                             sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                             sk_views::Option{<:LinearConstraintEstimator} = nothing,
                             kt_views::Option{<:LinearConstraintEstimator} = nothing,
                             cov_views::Option{<:LinearConstraintEstimator} = nothing,
                             rho_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_alpha::Option{<:Number} = nothing,
                             cvar_alpha::Option{<:Number} = nothing,
                             sets::Option{<:AssetSets} = nothing,
                             ds_opt::Option{<:CVaREntropyPooling} = nothing,
                             dm_opt::Option{<:OptimEntropyPooling} = nothing,
                             opt::NonCVaREP = OptimEntropyPooling(),
                             w::Option{<:StatsBase.ProbabilityWeights} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())::EntropyPoolingPrior
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, sigma_views, sk_views,
                               kt_views, cov_views, rho_views, var_alpha, cvar_alpha, sets,
                               ds_opt, dm_opt, opt, w, alg)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties EntropyPoolingPrior begin
    forward(pe, me, ce)
end
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of [`EntropyPoolingPrior`](@ref) elements.

# Related

  - [`EntropyPoolingPrior`](@ref)
"""
const VecEP = AbstractVector{<:EntropyPoolingPrior}
"""
    add_ep_constraint!(epc::AbstractDict, lhs::MatNum, rhs::VecNum, key::Symbol)

Add an entropy pooling view constraint to the constraint dictionary.

`add_ep_constraint!` normalises and adds a constraint to the entropy pooling constraint dictionary `epc`. If a constraint with the same key already exists, it concatenates the new constraint to the existing one. This function is used internally to build the set of linear constraints for entropy pooling optimisation.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `lhs`: Left-hand side constraint matrix.
  - `rhs`: Right-hand side constraint vector.
  - `key`: Constraint type key (`:eq`, `:ineq`, `:feq`, `:cvar_eq`).

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Related

  - [`entropy_pooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function add_ep_constraint!(epc::AbstractDict, lhs::MatNum, rhs::VecNum, key::Symbol)
    sc = LinearAlgebra.norm(lhs)
    lhs /= sc
    rhs /= sc
    epc[key] = if !haskey(epc, key)
        (lhs, rhs)
    else
        (vcat(epc[key][1], lhs), append!(epc[key][2], rhs))
    end
    return nothing
end
"""
    replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets,
                        key::Symbol; alpha::Option{<:Number} = nothing,
                        strict::Bool = false)

Replace prior references in view parsing results with their corresponding prior values.

`replace_prior_views` scans a parsed view constraint [`ParsingResult`](@ref) for references to prior values (e.g., `prior(A)`), and replaces them with the actual prior value from the provided prior result object. This ensures that prior-based terms in view constraints are treated as constants and not as variables in the optimisation. If an asset referenced in a prior is not found in the asset set, a warning is issued (or an error if `strict=true`). If all variables in the view are prior references, an error is thrown.

# Arguments

  - `res`: Parsed view constraint containing variables and coefficients.

  - `pr`: Prior result object containing prior values.

  - `sets`: Asset set mapping asset names to indices.

  - `key`: Moment type key (`:mu`, `:var`, `:cvar`, etc.).

  - `alpha`: Optional confidence level for VaR/CVaR views.

  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `res::ParsingResult`: Updated parsing result with prior references replaced by their values.

# Details

  - Prior references are matched using the pattern `prior(<asset>)`.
  - The right-hand side of the constraint is adjusted by subtracting the prior value times its coefficient.
  - Variables corresponding to prior references are removed from the constraint.
  - Throws an error if no non-prior variables remain.

# Related

  - [`ParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`AssetSets`](@ref)
  - [`prior`](@ref)
"""
function replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets,
                             key::Symbol, alpha::Option{<:Number} = nothing;
                             strict::Bool = false)
    prior_pattern = r"prior\(([^()]*)\)"
    nx = sets.dict[sets.key]
    variables, coeffs = res.vars, res.coef
    idx_rm = Vector{Int}(undef, 0)
    rhs::typeof(res.rhs) = res.rhs
    non_prior = false
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            non_prior = true
            continue
        end
        j = findfirst(x -> x == m.captures[1], nx)
        if isnothing(j)
            msg = "Asset $(m.captures[1]) not found in $nx."
            strict ? throw(ArgumentError(msg)) : @warn(msg)
            push!(idx_rm, i)
            continue
        end
        rhs -= get_pr_value(pr, j, Val(key), alpha) * c
        push!(idx_rm, i)
    end
    if isempty(idx_rm)
        return res
    end
    @argcheck(non_prior,
              ArgumentError("Priors in views are replaced by their prior value, thus they are essentially part of the constant of the view, so you need a non-prior view to serve as the variable.\n$(res)"))
    idx = setdiff(1:length(variables), idx_rm)
    variables_new = variables[idx]
    coeffs_new = coeffs[idx]
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "))
    return ParsingResult(variables_new, coeffs_new, res.op, rhs, "$(eqn) $(res.op) $(rhs)")
end
"""
    replace_prior_views(res::VecPR, args...; kwargs...)

Broadcast prior reference replacement across multiple view constraints.

`replace_prior_views` applies [`replace_prior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values.

# Arguments

  - `res:`: Vector of parsed view constraints.
  - `args...`: Additional positional arguments forwarded to [`replace_prior_views`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`replace_prior_views`](@ref).

# Returns

  - `res::Vector{<:ParsingResult}`: Vector of updated parsing results with prior references replaced by their values.

# Related

  - [`ParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`AssetSets`](@ref)
"""
function replace_prior_views(res::VecPR, args...; kwargs...)
    return replace_prior_views.(res, args...; kwargs...)
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:mu}, args...)

Extract the mean (expected return) for asset `i` from a prior result.

`get_pr_value` returns the mean value for the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:mu}`: Dispatch tag for mean extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `mu::Number`: Mean (expected return) for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:mu}, args...)
    return pr.mu[i]
end
"""
    ep_mu_views!(mu_views::Nothing, args...; kwargs...)

No-op pass-through for mean view constraints when none are specified.

`ep_mu_views!` is an internal API compatibility method that does nothing when mean view constraints (`mu_views`) are not provided (`mu_views = nothing`). This allows higher-level entropy pooling routines to uniformly call `ep_mu_views!` without special-casing the absence of mean views.

# Arguments

  - `mu_views::Nothing`: Indicates that no mean view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_mu_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_mu_views!(mu_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_mu_views!(mu_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)

Parse and add mean (expected return) view constraints to the entropy pooling constraint dictionary.

`ep_mu_views!` parses mean view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method is used internally by entropy pooling routines to enforce mean views in the optimisation.

# Arguments

  - `mu_views`: Mean view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Supports both equality and fixed equality constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_mu_views!(mu_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    X = pr.X
    mu_views = parse_equation(mu_views.val; datatype = eltype(X))
    mu_views = replace_group_by_assets(mu_views, sets, false, true, false)
    mu_views = replace_prior_views(mu_views, pr, sets, :mu; strict = strict)
    lcs = get_linear_constraints(mu_views, sets; datatype = eltype(X), strict = strict)
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        add_ep_constraint!(epc, getproperty(lcs, p).A * transpose(X), getproperty(lcs, p).B,
                           p)
    end
    return nothing
end
"""
    fix_mu!(epc::AbstractDict, fixed::BitVector, to_fix::BitVector,
            pr::AbstractPriorResult)

Add constraints to fix the mean of specified assets in entropy pooling.

`fix_mu!` identifies assets in `to_fix` that are not yet fixed (i.e., not present in `fixed`), and adds constraints to the entropy pooling constraint dictionary `epc` to fix their mean to the prior value. This ensures that higher moment views (e.g., variance, skewness, kurtosis, correlation) do not inadvertently alter the mean of these assets. The function updates `fixed` in-place to reflect the newly fixed assets.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `fixed`: Boolean vector indicating which assets have their mean fixed.
  - `to_fix`: Boolean vector indicating which assets should have their mean fixed.
  - `pr`: Prior result containing asset return information.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in-place.

# Details

  - Adds a fixed equality constraint (`:feq`) for each asset in `to_fix` that is not yet fixed.
  - Uses the prior mean values from `pr.mu` for the constraint right-hand side.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function fix_mu!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
                 pr::AbstractPriorResult)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(view(pr.X, :, fix)), pr.mu[fix], :feq)
        fixed .= fixed .| fix
    end
    return nothing
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Number)

Extract the Value-at-Risk (VaR) for asset `i` from a prior result.

`get_pr_value` computes the VaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`. This method uses the asset return samples in `pr` and applies the VaR calculation, typically using the empirical quantile.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:var}`: Dispatch tag for VaR extraction.
  - `alpha`: Confidence level (e.g., `0.05` for 5% VaR).

# Returns

  - `var::Number`: Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Number)
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ValueatRisk(; alpha = alpha)(view(pr.X, :, i))
end
"""
    ep_var_views!(var_views::Nothing, args...; kwargs...)

No-op pass-through for variance view constraints when none are specified.

`ep_var_views!` is an internal API compatibility method that does nothing when variance view constraints (`var_views`) are not provided (`var_views = nothing`). This allows higher-level entropy pooling routines to uniformly call `ep_var_views!` without special-casing the absence of variance views.

# Arguments

  - `var_views::Nothing`: Indicates that no variance view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_var_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::AssetSets, alpha::Number; strict::Bool = false)

Parse and add variance (VaR) view constraints to the entropy pooling constraint dictionary.

`ep_var_views!` parses variance (VaR) view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method validates that only single-asset, non-negative, and unit-coefficient views are allowed, and throws informative errors for invalid or extreme views.

# Arguments

  - `var_views`: Variance (VaR) view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `alpha`: Confidence level for VaR.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Validates that only equality and inequality constraints with unit coefficients are present.
  - Throws errors for negative or multi-asset views, or if the view is more extreme than the worst realisation.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::AssetSets, alpha::Number;
                       strict::Bool = false)
    X = pr.X
    var_views = parse_equation(var_views.val; ops1 = ("==", ">="),
                               ops2 = (:call, :(==), :(>=)), datatype = eltype(X))
    var_views = replace_group_by_assets(var_views, sets, false, true, false)
    var_views = replace_prior_views(var_views, pr, sets, :var, alpha; strict = strict)
    lcs = get_linear_constraints(var_views, sets; datatype = eltype(X), strict = strict)
    @argcheck(!(!isnothing(lcs.ineq) && !any(x -> (iszero(x) || isone(x)), lcs.A_ineq) ||
                !isnothing(lcs.eq) && !any(x -> (iszero(x) || isone(x)), lcs.A_eq)),
              ArgumentError("var_view only supports coefficients of 1.\n$var_views"))
    @argcheck(!(!isnothing(lcs.ineq) &&
                any(x -> x != 1, count(!iszero, lcs.A_ineq; dims = 2)) ||
                !isnothing(lcs.eq) && any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2))),
              ArgumentError("Cannot mix multiple assets in a single var_view.\n$var_views"))
    @argcheck(!(!isnothing(lcs.eq) && any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq) ||
                !isnothing(lcs.ineq) &&
                any(x -> x < zero(eltype(x)), lcs.A_ineq .* lcs.B_ineq)),
              DomainError("var_views cannot be negative.\n$var_views"))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        B = getproperty(lcs, p).B
        for i in eachindex(B)
            j = .!iszero.(A[i, :])
            #! Figure out a way to include pr.w, probably see how it's implemented in ValueatRisk.
            idx = findall(x -> x <= -abs(B[i]), view(X, :, j))
            @argcheck(!isempty(idx),
                      DomainError("View $(i) = $(var_views[i].eqn) is too extreme, the maximum viable for asset $(findfirst(x -> x == true, j)) is $(-minimum(X[:,j])). Please lower it or use a different prior with fatter tails."))
            sign = ifelse(p == :eq || B[i] >= zero(eltype(B)), one(eltype(B)),
                          -one(eltype(B)))
            Ai = zeros(eltype(X), 1, size(X, 1))
            Ai[1, idx] .= sign
            add_ep_constraint!(epc, Ai, [sign * alpha], p)
        end
    end
    return nothing
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::OptimEntropyPooling)

Solve the dual of the exponential entropy pooling formulation using Optim.jl.

`entropy_pooling` computes posterior probabilities by minimising the exponential divergence between prior and posterior weights, subject to moment and view constraints. The optimisation is performed using [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), supporting box constraints and slack variables for relaxed equality constraints. This method is used internally by [`EntropyPoolingPrior`](@ref) when the optimiser is an [`OptimEntropyPooling`](@ref).

# Mathematical definition

The dual of the entropy pooling KL divergence problem is solved for Lagrange multipliers ``\\boldsymbol{x}``. The dual objective (for `ExpEntropyPooling`) is:

```math
\\begin{align}
\\underset{\\boldsymbol{x}}{\\min} &\\; \\boldsymbol{x}^\\intercal \\boldsymbol{b} + \\sum_{t=1}^{T} q_t \\exp\\!\\left(-\\boldsymbol{x}^\\intercal \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

The optimal posterior weights recover as:

```math
\\begin{align}
p_t^* &= q_t \\exp\\!\\left(-\\boldsymbol{x}^{*\\intercal} \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Lagrange multipliers (dual variables).
  - ``\\boldsymbol{b}``: Right-hand side constraint vector.
  - ``\\mathbf{A}_{\\cdot t}``: ``t``-th column of the constraint matrix ``\\mathbf{A}``.
  - ``q_t``: Prior weight for scenario ``t``.
  - ``p_t^*``: Optimal posterior weight for scenario ``t``.
  - $(math_dict[:T])

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `opt: Optim.jl-based entropy pooling optimiser with exponential objective.

      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Use the exponential formulation.
      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Use the logarithmic formulation.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Details

  - Constructs the constraint matrix and bounds from `epc`.
  - Relaxes fixed equality constraints via slack variables to make the problem more tractable.
  - Throws an error if optimisation fails.

# Related

  - [`OptimEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:ExpEntropyPooling})
    T = length(w)
    factor = inv(sqrt(T))
    A = fill(factor, 1, T)
    B = [factor]
    wb = [typemin(eltype(w)) typemax(eltype(w))]
    for (key, val) in epc
        A = vcat(A, val[1])
        B = vcat(B, val[2])
        s = length(val[2])
        wb = if key == :eq || key == :cvar_eq
            vcat(wb, [fill(typemin(eltype(w)), s) fill(typemax(eltype(w)), s)])
        elseif key == :ineq || key == :cvar_ineq
            vcat(wb, [zeros(eltype(w), s) fill(typemax(eltype(w)), s)])
        elseif key == :feq
            vcat(wb, [fill(-opt.sc2, s) fill(opt.sc2, s)])
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
    end
    x0 = fill(factor, size(A, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    y = similar(w)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            y .= w .* exp.(-transpose(A) * x .- one(eltype(w)))
            grad .= B - A * y
        end
    end
    function f(x)
        common_op(x)
        return opt.sc1 * (sum(y) + LinearAlgebra.dot(x, B))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    #! Start: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @static if v"2.0.1" <= pkgversion(Optim) < v"2.3.0"
        args = ifelse(isempty(opt.args), (Optim.Fminbox(; mu0 = 1e-5),), opt.args)
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, args...;
                                opt.kwargs...)
    else
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                                opt.kwargs...)
    end
    #! End: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return StatsBase.pweights(w .* exp.(-transpose(A) * x .- one(eltype(w))))
end
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:LogEntropyPooling})
    T = length(w)
    factor = inv(sqrt(T))
    A = fill(factor, 1, T)
    B = [factor]
    wb = [typemin(eltype(w)) typemax(eltype(w))]
    for (key, val) in epc
        A = vcat(A, val[1])
        B = vcat(B, val[2])
        s = length(val[2])
        wb = if key == :eq || key == :cvar_eq
            vcat(wb, [fill(typemin(eltype(w)), s) fill(typemax(eltype(w)), s)])
        elseif key == :ineq
            vcat(wb, [zeros(eltype(w), s) fill(typemax(eltype(w)), s)])
        elseif key == :feq
            vcat(wb, [fill(-opt.sc2, s) fill(opt.sc2, s)])
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
    end
    log_p = log.(w)
    x0 = fill(factor, size(A, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    log_x = similar(log_p)
    y = similar(log_p)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            log_x .= log_p - (one(eltype(log_p)) .+ transpose(A) * x)
            y .= exp.(log_x)
            grad .= B - A * y
        end
    end
    function f(x)
        common_op(x)
        return opt.sc1 * (LinearAlgebra.dot(x, grad) - LinearAlgebra.dot(y, log_x - log_p))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    #! Start: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @static if v"2.0.1" <= pkgversion(Optim) < v"2.3.0"
        args = ifelse(isempty(opt.args), (Optim.Fminbox(; mu0 = 1e-5),), opt.args)
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, args...;
                                opt.kwargs...)
    else
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                                opt.kwargs...)
    end
    #! End: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return StatsBase.pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(A) * x)))
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::JuMPEntropyPooling)

Solve the primal of the exponential entropy pooling formulation using JuMP.jl.

`entropy_pooling` computes posterior probabilities by minimising the exponential divergence between prior and posterior weights, subject to moment and view constraints. The optimisation is performed using [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl), supporting relative entropy cones and slack variables for relaxed equality constraints. This method is used internally by [`EntropyPoolingPrior`](@ref) when the optimiser is a [`JuMPEntropyPooling`](@ref).

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `opt`: JuMP.jl-based entropy pooling optimiser with exponential objective.

      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Use the exponential formulation.
      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Use the logarithmic formulation.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Details

  - Constructs the JuMP model with exponential objective and constraints from `epc`.
  - Relaxes fixed equality constraints by adding norm one cone bounded slack variables to make the problem more tractable.
  - Throws an error if optimisation fails.

# Related

  - [`JuMPEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`OptimEntropyPooling`](@ref)
"""
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                 <:ExpEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    T = length(w)
    model = JuMP.Model()
    JuMP.@variables(model, begin
                        t
                        x[1:T] >= 0
                    end)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(x) - one(eltype(w))) == 0
                          [sc1 * t; sc1 * w; sc1 * x] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    JuMP.@expression(model, obj_expr, so * t)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        JuMP.@constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        JuMP.@constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        JuMP.@constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        JuMP.@variables(model, begin
                            tc
                            c[1:N]
                        end)
        JuMP.@constraints(model, begin
                              cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                              [sc1 * tc; sc1 * c] in JuMP.MOI.NormOneCone(N + 1)
                          end)
        JuMP.add_to_expression!(obj_expr, so * sc2 * tc)
    end
    JuMP.@objective(model, Min, obj_expr)
    @argcheck(optimise_JuMP_model!(model, slv).success,
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    return StatsBase.pweights(JuMP.value.(x))
end
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                 <:LogEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    model = JuMP.Model()
    T = length(w)
    log_p = log.(w)
    # Decision variables (posterior probabilities)
    JuMP.@variables(model, begin
                        x[1:T]
                        t
                    end)
    JuMP.@expression(model, obj_expr, so * t)
    # Equality constraints from A_eq and B_eq and probabilities equal to 1
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(x) - one(eltype(w))) == 0
                          [sc1 * t; fill(sc1, T); sc1 * x] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        JuMP.@constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        JuMP.@constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        JuMP.@constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        JuMP.@variables(model, begin
                            tc
                            c[1:N]
                        end)
        JuMP.@constraints(model, begin
                              cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                              [sc1 * tc; sc1 * c] in JuMP.MOI.NormOneCone(N + 1)
                          end)
        JuMP.add_to_expression!(obj_expr, so * sc2 * tc)
    end
    JuMP.@objective(model, Min, obj_expr - so * LinearAlgebra.dot(x, log_p))
    # Solve the optimization problem
    @argcheck(optimise_JuMP_model!(model, slv).success,
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    return StatsBase.pweights(JuMP.value.(x))
end
"""
    ep_cvar_views_solve!(cvar_views::Nothing, epc::AbstractDict, ::Any, ::Any, ::Number,
                         w::StatsBase.ProbabilityWeights, opt::AbstractEntropyPoolingOptimiser, ::Any, ::Any;
                         kwargs...)

Solve entropy pooling views when no CVaR views are specified.

`ep_cvar_views_solve!` is an internal API compatibility method that solves the entropy pooling problem when no Conditional Value-at-Risk (CVaR) view constraints are present (`cvar_views = nothing`). It simply delegates to the main entropy pooling solver using the provided prior weights, constraint dictionary, and optimiser.

# Arguments

  - `cvar_views`: Indicates that no CVaR view constraints are specified.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `w`: Prior probability weights.
  - `opt`: Entropy pooling optimiser.
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
  - [`CVaREntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_cvar_views_solve!(cvar_views::Nothing, epc::AbstractDict, ::Any, ::Any, ::Any,
                              w::StatsBase.ProbabilityWeights,
                              opt::AbstractEntropyPoolingOptimiser, ::Any, ::Any; kwargs...)
    return entropy_pooling(w, epc, opt)
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Number)

Compute the Conditional Value-at-Risk (CVaR) for asset `i` from a prior result.

`get_pr_value` extracts the CVaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`. This method assumes the prior result contains the necessary asset return information (mean, covariance, or samples) to compute CVaR, typically under a normality assumption.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:cvar}`: Dispatch tag for CVaR computation.
  - `alpha`: Confidence level.

# Returns

  - `cvar::Number`: Conditional Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Number)
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ConditionalValueatRisk(; alpha = alpha)(view(pr.X, :, i))
end
"""
    ep_cvar_views_solve!(cvar_views::LinearConstraintEstimator, epc::AbstractDict,
                         pr::AbstractPriorResult, sets::AssetSets, alpha::Number,
                         w::StatsBase.ProbabilityWeights, opt::AbstractEntropyPoolingOptimiser,
                         ds_opt::Option{<:CVaREntropyPooling},
                         dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false)

Solve the entropy pooling problem with Conditional Value-at-Risk (CVaR) view constraints.

`ep_cvar_views_solve!` parses and validates CVaR view constraints, replaces prior references, and constructs the corresponding entropy pooling constraint system. It then solves for posterior probability weights using either root-finding (for single CVaR view) or optimisation (for multiple views), depending on the number of constraints and the provided optimiser. Throws informative errors if views are infeasible or too extreme.

# Arguments

  - `cvar_views`: CVaR view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `alpha`: Confidence level for CVaR.
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
  - For a single CVaR view, uses root-finding via [`CVaREntropyPooling`](@ref).
  - For multiple CVaR views, uses optimisation via [`OptimEntropyPooling`](@ref).
  - Throws errors if optimisation fails or views are infeasible.

# Related

  - [`CVaREntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_cvar_views_solve!(cvar_views::LinearConstraintEstimator, epc::AbstractDict,
                              pr::AbstractPriorResult, sets::AssetSets, alpha::Number,
                              w::StatsBase.ProbabilityWeights,
                              opt::AbstractEntropyPoolingOptimiser,
                              ds_opt::Option{<:CVaREntropyPooling},
                              dm_opt::Option{<:OptimEntropyPooling}; strict::Bool = false)
    X = pr.X
    cvar_views = parse_equation(cvar_views.val; ops1 = ("==",), ops2 = (:call, :(==)),
                                datatype = eltype(X))
    cvar_views = replace_group_by_assets(cvar_views, sets, false, true, false)
    cvar_views = replace_prior_views(cvar_views, pr, sets, :cvar, alpha; strict = strict)
    lcs = get_linear_constraints(cvar_views, sets; datatype = eltype(X), strict = strict)
    @argcheck(!any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2)),
              ArgumentError("Cannot mix multiple assets in a single cvar_view.\n$(cvar_views)"))
    @argcheck(!any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq),
              DomainError("cvar_views cannot be negative.\n$(cvar_views)"))
    idx = dropdims(.!iszero.(sum(lcs.A_eq; dims = 1)); dims = 1)
    idx2 = .!iszero.(lcs.A_eq)
    B = lcs.B_eq ./ view(lcs.A_eq, idx2)
    X = view(X, :, idx)
    min_X = dropdims(-minimum(X; dims = 1); dims = 1)
    invalid = B .>= min_X
    if any(invalid)
        if !isa(cvar_views, AbstractVector)
            cvar_views = [cvar_views]
        end
        msg = "The following views are too extreme, the maximum viable view for a given asset is its worst realisation:"
        arr = [(v.eqn, m) for (v, m) in zip(cvar_views[invalid], min_X[invalid])]
        for (v, m) in arr
            msg *= "\n$v\t(> $m)."
        end
        msg *= "\nPlease lower the views or use a different prior with fatter tails."
        throw(ArgumentError(msg))
    end
    N = length(B)
    d_opt = if N == 1
        ifelse(!isnothing(ds_opt), ds_opt, CVaREntropyPooling())
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
        add_ep_constraint!(epc, transpose(pos_part / alpha), B .- etas, :cvar_eq)
        wi = entropy_pooling(w, epc, opt)
        err = if N == 1
            sum(wi[.!iszero.(pos_part)]) - alpha
        else
            norm_error(d_opt.err,
                       [ConditionalValueatRisk(; alpha = alpha, w = wi)(view(X, :, i)) -
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
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:sigma}, args...)

Extract the variance for asset `i` from a prior result.

`get_pr_value` returns the variance (diagonal element of the covariance matrix) for the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:sigma}`: Dispatch tag for variance extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `sigma::Number`: Variance for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:sigma}, args...)
    return LinearAlgebra.diag(pr.sigma)[i]
end
"""
    ep_sigma_views!(sigma_views::LinearConstraintEstimator, epc::AbstractDict,
                    pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)

Parse and add variance (sigma) view constraints to the entropy pooling constraint dictionary.

`ep_sigma_views!` parses variance view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding quadratic constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean to be fixed to the prior value, ensuring that variance views do not inadvertently alter the mean.

# Arguments

  - `sigma_views`: Variance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior values.
  - Converts parsed views to quadratic constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean fixed due to variance constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_sigma_views!(sigma_views::LinearConstraintEstimator, epc::AbstractDict,
                         pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    X = pr.X
    sigma_views = parse_equation(sigma_views.val; datatype = eltype(X))
    sigma_views = replace_group_by_assets(sigma_views, sets, false, true, false)
    sigma_views = replace_prior_views(sigma_views, pr, sets, :sigma; strict = strict)
    lcs = get_linear_constraints(sigma_views, sets; datatype = eltype(X), strict = strict)
    tmp = transpose((X .- transpose(pr.mu)) .^ 2)
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
"""
    fix_sigma!(epc::AbstractDict, fixed::BitVector, to_fix::BitVector,
               pr::AbstractPriorResult)

Add constraints to fix the variance of specified assets in entropy pooling.

`fix_sigma!` identifies assets in `to_fix` that are not yet fixed (i.e., not present in `fixed`), and adds constraints to the entropy pooling constraint dictionary `epc` to fix their variance to the prior value. This ensures that higher moment views (e.g., skewness, kurtosis, correlation) do not inadvertently alter the variance of these assets. The function updates `fixed` in-place to reflect the newly fixed assets.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `fixed`: Boolean vector indicating which assets have their variance fixed.
  - `to_fix`: Boolean vector indicating which assets should have their variance fixed.
  - `pr`: Prior result containing asset return information.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in-place.

# Details

  - Adds a fixed equality constraint (`:feq`) for each asset in `to_fix` that is not yet fixed.
  - Uses the prior variance values from `LinearAlgebra.diag(pr.sigma)` for the constraint right-hand side.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
                    pr::AbstractPriorResult)
    sigma = LinearAlgebra.diag(pr.sigma)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(view(pr.X, :, fix) .- transpose(pr.mu[fix])) .^ 2,
                           sigma[fix], :feq)
        fixed .= fixed .| fix
    end
    return nothing
end
"""
    replace_coprior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets, key::Symbol;
                          strict::Bool = false)

Replace correlation prior references in view parsing results with their corresponding prior values.

`replace_coprior_views` scans a parsed correlation view constraint (`ParsingResult`) for references to prior values (e.g., `prior(A, B)`), and replaces them with the actual prior correlation value from the provided prior result object. This ensures that prior-based terms in correlation view constraints are treated as constants and not as variables in the optimisation. If an asset referenced in a prior is not found in the asset set, a warning is issued (or an error if `strict=true`). If all variables in the view are prior references, an error is thrown.

# Arguments

  - `res`: Parsed correlation view constraint containing variables and coefficients.
  - `pr`: Prior result object containing prior correlation values.
  - `sets`: Asset set mapping asset names to indices.
  - `key`: Symbol representing whether it's a correlation or covariance view.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `res::RhoParsingResult`: Updated parsing result with prior references replaced by their values and correlation indices.

# Details

  - Prior references are matched using the pattern `prior(<asset1>, <asset2>)`.
  - The right-hand side of the constraint is adjusted by subtracting the prior correlation value times its coefficient.
  - Variables corresponding to prior references are removed from the constraint.
  - Throws an error if no non-prior variables remain.
  - Returns a `RhoParsingResult` containing the updated variables, coefficients, operator, right-hand side, equation string, and correlation indices.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`AssetSets`](@ref)
  - [`prior`](@ref)
"""
function replace_coprior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets,
                               key::Symbol; strict::Bool = false)
    prior_pattern = r"prior\(([^()]*)\)"
    prior_corr_pattern = r"prior\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    nx = sets.dict[sets.key]
    variables, coeffs = res.vars, res.coef
    jk_idx = Vector{Union{Tuple{Int, Int}, Tuple{Vector{Int}, Vector{Int}}}}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    rhs::typeof(res.rhs) = res.rhs
    non_prior = false
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            non_prior = true
            n = match(corr_pattern, v)
            @argcheck(!isnothing(n),
                      ArgumentError("Correlation view $(v) must be of the form `(a, b)`."))
            asset1 = n.captures[1]
            asset2 = n.captures[2]
            if startswith(asset1, "[") && endswith(asset1, "]")
                asset1 = split(n.captures[1][2:(end - 1)], ", ")
                asset2 = split(n.captures[2][2:(end - 1)], ", ")
                j = [findfirst(x -> x == a1, nx) for a1 in asset1]
                k = [findfirst(x -> x == a2, nx) for a2 in asset2]
            else
                j = findfirst(x -> x == asset1, nx)
                k = findfirst(x -> x == asset2, nx)
                if isnothing(j)
                    msg = "Asset $(asset1) not found in $nx."
                    strict ? throw(ArgumentError(msg)) : @warn(msg)
                end
                if isnothing(k)
                    msg = "Asset $(asset2) not found in $nx."
                    strict ? throw(ArgumentError(msg)) : @warn(msg)
                end
                if isnothing(j) || isnothing(k)
                    push!(idx_rm, i)
                    continue
                end
            end
            push!(jk_idx, (j, k))
            continue
        end
        n = match(prior_corr_pattern, v)
        @argcheck(!isnothing(n),
                  ArgumentError("Correlation prior view $(v) must be of the form `prior(a, b)`."))
        asset1 = n.captures[1]
        asset2 = n.captures[2]
        if startswith(asset1, "[") && endswith(asset1, "]")
            asset1 = split(n.captures[1][2:(end - 1)], ", ")
            asset2 = split(n.captures[2][2:(end - 1)], ", ")
            j = [findfirst(x -> x == a1, nx) for a1 in asset1]
            k = [findfirst(x -> x == a2, nx) for a2 in asset2]
        else
            j = findfirst(x -> x == asset1, nx)
            k = findfirst(x -> x == asset2, nx)
            if isnothing(j)
                msg = "Asset $(asset1) not found in $nx."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
            end
            if isnothing(k)
                msg = "Asset $(asset2) not found in $nx."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
            end
            if isnothing(j) || isnothing(k)
                push!(idx_rm, i)
                continue
            end
        end
        rhs -= get_pr_value(pr, j, k, Val(key)) * c
        push!(idx_rm, i)
    end
    if isempty(idx_rm)
        return RhoParsingResult(res.vars, res.coef, res.op, res.rhs, res.eqn, jk_idx)
    end
    @argcheck(non_prior,
              ArgumentError("Priors in views are replaced by their prior value, thus they are essentially part of the constant of the view, so you need a non-prior view to serve as the variable.\n$(res)"))
    idx = setdiff(1:length(variables), idx_rm)
    variables_new = variables[idx]
    coeffs_new = coeffs[idx]
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "))
    return RhoParsingResult(variables_new, coeffs_new, res.op, rhs,
                            "$(eqn) $(res.op) $(rhs)", jk_idx)
end
"""
    replace_coprior_views(res::VecPR, args...; kwargs...)

Broadcast prior reference replacement across multiple view constraints.

`replace_coprior_views` applies [`replace_coprior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values.

# Arguments

  - `res:`: Vector of parsed view constraints.
  - `args...`: Additional positional arguments forwarded to [`replace_coprior_views`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`replace_coprior_views`](@ref).

# Returns

  - `res::Vector{<:ParsingResult}`: Vector of updated parsing results with prior references replaced by their values.

# Related

  - [`ParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`AssetSets`](@ref)
"""
function replace_coprior_views(res::VecPR, args...; kwargs...)
    return replace_coprior_views.(res, args...; kwargs...)
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, args...)

Extract the prior correlation value between assets `i` and `j` from a prior result.

`get_pr_value` returns the correlation coefficient between the assets indexed by `i` and `j` from the prior result object `pr`. This method is used internally to replace prior references in correlation view constraints and for moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the first asset.
  - `j`: Index of the second asset.
  - `args...`: Additional arguments (ignored).

# Returns

  - `rho::Number`: Correlation coefficient between assets `i` and `j`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, ::Val{:rho}, args...)
    return StatsBase.cov2cor(pr.sigma)[i, j]
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, ::Val{:cov}, args...)
    return pr.sigma[i, j]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the normalised average correlation between asset groups `i` and `j` from a prior result.

`get_pr_value` returns the Frobenius norm of the correlation sub-matrix divided by the group length for vector index sets `i` and `j`. This variant handles grouped asset correlation views.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Vector of indices for the first asset group.
  - `j`: Vector of indices for the second asset group.
  - `args...`: Additional arguments (ignored).

# Returns

  - `rho::Number`: Normalised average correlation between asset groups `i` and `j`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::VecInt, j::VecInt, args...)
    return LinearAlgebra.norm(StatsBase.cov2cor(pr.sigma)[i, j]) / length(i)
end
"""
    ep_cov_views!(cov_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)

Parse and add correlation view constraints to the entropy pooling constraint dictionary.

`ep_cov_views!` parses correlation view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that correlation views do not inadvertently alter lower moments.

# Arguments

  - `cov_views`: Correlation view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior correlation values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to correlation constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_cov_views!(cov_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    X = pr.X
    cov_views = parse_equation(cov_views.val; datatype = eltype(X))
    cov_views = replace_group_by_assets(cov_views, sets, false, true, true)
    cov_views = replace_coprior_views(cov_views, pr, sets, :cov; strict = strict)
    to_fix = falses(size(X, 2))
    for cov_view in cov_views
        @argcheck(length(cov_view.vars) == 1,
                  "Cannot mix multiple covariance pairs in a single view `$(cov_view.eqn)`.")
        d = ifelse(cov_view.op == ">=", -1, 1)
        i, j = cov_view.ij[1]
        Ai = d * cov_view.coef[1] * view(X, :, i) .* view(X, :, j)
        Bi = d * pr.mu[i] ⊙ pr.mu[j] ⊕ cov_view.rhs
        if !isa(i, AbstractVector)
            Bi = [Bi]
        end
        add_ep_constraint!(epc, transpose(Ai), Bi, ifelse(cov_view.op == "==", :eq, :ineq))
        to_fix[union(i, j)] .= true
    end
    return to_fix
end
"""
    ep_rho_views!(rho_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)

Parse and add correlation view constraints to the entropy pooling constraint dictionary.

`ep_rho_views!` parses correlation view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that correlation views do not inadvertently alter lower moments.

# Arguments

  - `rho_views`: Correlation view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior correlation values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to correlation constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_rho_views!(rho_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    X = pr.X
    rho_views = parse_equation(rho_views.val; datatype = eltype(X))
    rho_views = replace_group_by_assets(rho_views, sets, false, true, true)
    rho_views = replace_coprior_views(rho_views, pr, sets, :rho; strict = strict)
    to_fix = falses(size(X, 2))
    sigma = LinearAlgebra.diag(pr.sigma)
    for rho_view in rho_views
        @argcheck(length(rho_view.vars) == 1,
                  "Cannot mix multiple correlation pairs in a single view `$(rho_view.eqn)`.")
        @argcheck(-one(eltype(X)) <= rho_view.rhs <= one(eltype(X)),
                  "Correlation prior rho_view `$(rho_view.eqn)` must be in [-1, 1].")
        d = ifelse(rho_view.op == ">=", -1, 1)
        i, j = rho_view.ij[1]
        sigma_ij = if !isa(i, AbstractVector)
            sqrt(sigma[i] * sigma[j])
        else
            LinearAlgebra.norm(sigma[i] .* sigma[j])
        end
        Ai = d * rho_view.coef[1] * view(X, :, i) .* view(X, :, j)
        Bi = d * pr.mu[i] ⊙ pr.mu[j] ⊕ rho_view.rhs ⊙ sigma_ij
        if !isa(i, AbstractVector)
            Bi = [Bi]
        end
        add_ep_constraint!(epc, transpose(Ai), Bi, ifelse(rho_view.op == "==", :eq, :ineq))
        to_fix[union(i, j)] .= true
    end
    return to_fix
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew}, args...)

Extract the skewness for asset `i` from a prior result.

`get_pr_value` returns the skewness of the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for higher moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:skew}`: Dispatch tag for skewness extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `skew::Number`: Skewness for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew}, args...)
    #! Think about how to include pr.w
    return Skewness()(view(pr.X, :, i))
end
"""
    ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)

Parse and add skewness view constraints to the entropy pooling constraint dictionary.

`ep_sk_views!` parses skewness view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that skewness views do not inadvertently alter lower moments.

# Arguments

  - `skew_views`: Skewness view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior skewness values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to skewness constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    X = pr.X
    skew_views = parse_equation(skew_views.val; datatype = eltype(X))
    skew_views = replace_group_by_assets(skew_views, sets, false, true, false)
    skew_views = replace_prior_views(skew_views, pr, sets, :skew; strict = strict)
    lcs = get_linear_constraints(skew_views, sets; datatype = eltype(X), strict = strict)
    sigma = LinearAlgebra.diag(pr.sigma)
    tmp = transpose((X .^ 3 .- transpose(pr.mu) .^ 3 .- 3 * transpose(pr.mu .* sigma)) ./
                    transpose(sigma .* sqrt.(sigma)))
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis}, args...)

Extract the kurtosis for asset `i` from a prior result.

`get_pr_value` returns the kurtosis of the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for higher moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:kurtosis}`: Dispatch tag for kurtosis extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `kurtosis::Number`: Kurtosis for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis}, args...)
    #! Think about how to include pr.w
    return HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))(view(pr.X,
                                                                                             :,
                                                                                             i))
end
"""
    ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)

Parse and add kurtosis view constraints to the entropy pooling constraint dictionary.

`ep_kt_views!` parses kurtosis view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that kurtosis views do not inadvertently alter lower moments.

# Arguments

  - `kurtosis_views`: Kurtosis view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior kurtosis values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to kurtosis constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    X = pr.X
    kurtosis_views = parse_equation(kurtosis_views.val; datatype = eltype(X))
    kurtosis_views = replace_group_by_assets(kurtosis_views, sets, false, true, false)
    kurtosis_views = replace_prior_views(kurtosis_views, pr, sets, :kurtosis;
                                         strict = strict)
    lcs = get_linear_constraints(kurtosis_views, sets; datatype = eltype(X),
                                 strict = strict)
    X_sq = X .^ 2
    mu_sq = pr.mu .^ 2
    tmp = transpose((X_sq .* X_sq .- 4 * transpose(pr.mu) .* X_sq .* X .+
                     6 * transpose(mu_sq) .* X_sq .- 3 * transpose(mu_sq .* mu_sq)) ./
                    transpose(LinearAlgebra.diag(pr.sigma)) .^ 2)
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
"""
    prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:StagedEP},
          X::MatNum; F::Option{<:MatNum} = nothing, dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute entropy pooling prior moments for asset returns with iterative constraint enforcement.

`prior` estimates the mean and covariance of asset returns using the entropy pooling framework, supporting iterative constraint enforcement via the `H1_EntropyPooling` and `H2_EntropyPooling` algorithms. It integrates moment and view constraints (mean, variance, CVaR, skewness, kurtosis, correlation), flexible confidence specification, and composable optimisation algorithms. The method iteratively applies constraints, updating prior weights and moments at each step, and ensures that higher moment views do not inadvertently alter lower moments.

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

  - `pe`: Entropy pooling prior estimator with iterative algorithm .
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix (default: `nothing`).
  - $(arg_dict[:dims])
  - If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Validation

  - `dims in (1, 2)`.
  - If any view constraint is not `nothing`, `!isnothing(sets)`.
  - If prior weights `pe.w` are provided, `length(pe.w) == T`, where `T` is the number of observations.

# Details

  - If `isnothing(pe.w)`, prior weights are initialised to `1/T` where `T` is the number of observations; otherwise, provided weights are normalised.
  - Constraints are enforced iteratively, from lower to higher moments.
  - Moment and view constraints are parsed and added to the constraint dictionary.
  - The initial weights for each stage is selected according to `pe.alg`.
  - At each stage, the prior weights are updated by solving the entropy pooling optimisation with the current set of constraints. If present, the CVaR views are also enforced at every stage.
  - Lower moments are fixed as needed to prevent distortion by higher moment views. If asset `i` has a view enforced on moment `N` that uses moments `n < N` to compute, then all moments `n` for asset `i` are fixed.
  - The final result includes the effective number of scenarios and Kullback-Leibler divergence between prior and posterior weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`LowOrderPrior`](@ref)
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
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:StagedEP}, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be in (1, 2)"))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
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
    ep_var_views!(pe.var_views, epc, pr, pe.sets, pe.var_alpha; strict = strict)
    if !isnothing(pe.mu_views) || !isnothing(pe.var_views) || !isnothing(pe.cvar_views)
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha, w0,
                                  pe.opt, pe.ds_opt, pe.dm_opt; strict = strict)
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
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha,
                                  ifelse(isa(pe.alg, H1_EntropyPooling), w0, w1), pe.opt,
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
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha,
                                  ifelse(isa(pe.alg, H1_EntropyPooling), w0, w1), pe.opt,
                                  pe.ds_opt, pe.dm_opt; strict = strict)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w1, ens = ens,
                         kld = kld, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(rr) ? w1 : nothing)
end
"""
    prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:H0_EntropyPooling}, X::MatNum;
          F::Option{<:MatNum} = nothing, dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute entropy pooling prior moments for asset returns with single-shot constraint enforcement.

`prior` estimates the mean and covariance of asset returns using the entropy pooling framework, enforcing all moment and view constraints in a single optimisation step via the `H0_EntropyPooling` algorithm. This approach is fast but may distort lower moments when higher moment views are present, as all constraints are applied simultaneously.

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

  - `pe`: Entropy pooling prior estimator with single-shot algorithm.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - $(arg_dict[:dims])
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Validation

  - `dims in (1, 2)`.
  - If any view constraint is not `nothing`, `!isnothing(pe.sets)`.
  - If prior weights `pe.w` are provided, `length(pe.w) == T`, where `T` is the number of observations

# Details

  - If `isnothing(pe.w)`, prior weights are initialised to `1/T` where `T` is the number of observations; otherwise, provided weights are normalised.
  - All constraints are parsed and added to the constraint dictionary at once. This means that lower moments may be distorted by higher moment views, since they cannot be fixed at any point.
  - A single optimisation is performed to solve for the posterior weights, enforcing all constraints at once.
  - The final result includes the effective number of scenarios and Kullback-Leibler divergence between prior and posterior weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
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
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:H0_EntropyPooling}, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be in (1, 2)"))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
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
    ep_var_views!(pe.var_views, epc, pr, pe.sets, pe.var_alpha; strict = strict)
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
    w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha, w0, pe.opt,
                              pe.ds_opt, pe.dm_opt; strict = strict)
    pe = factory(pe, w1)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w1, ens = ens,
                         kld = kld, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(rr) ? w1 : nothing)
end

export RhoParsingResult, LogEntropyPooling, ExpEntropyPooling, EntropyPoolingPrior,
       H0_EntropyPooling, H1_EntropyPooling, H2_EntropyPooling, JuMPEntropyPooling,
       OptimEntropyPooling, CVaREntropyPooling
