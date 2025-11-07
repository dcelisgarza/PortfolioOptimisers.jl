"""
    abstract type AbstractEntropyPoolingOptimiser <: AbstractEstimator end

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
    abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end

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
    struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end

One-shot entropy pooling. It sets and optimises all the constraints simultaneously. This introduces bias in the posterior probabilities, but is faster.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
    struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end

Uses the initial probabilities to optimise the posterior probabilities at every step. This reduces bias in the posterior probabilities, but is slower.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
    struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end

Uses the previous step's probabilities to optimise the next step's probabilities. This is faster but may introduce bias.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
"""
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
    abstract type AbstractEntropyPoolingOptAlgorithm <: AbstractAlgorithm end

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
    struct LogEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end

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
    struct ExpEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end

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
    get_epw(alg::Union{<:H0_EntropyPooling, <:H1_EntropyPooling}, w0::AbstractWeights,
            wi::AbstractWeights)

Select entropy pooling weights according to the specified algorithm.

`get_epw` returns the appropriate weights for entropy pooling based on the chosen algorithm. For `H1_EntropyPooling`, it returns the initial prior weights `w0`. For `H2_EntropyPooling`, it returns the updated weights `wi`. This function is used internally to manage the flow of weights in multi-stage entropy pooling routines.

# Arguments

  - `alg`: Entropy pooling algorithm .
  - `w0`: Initial prior weights.
  - `wi`: Updated weights from previous step.

# Returns

  - `w::AbstractWeights`: Selected weights for the current entropy pooling step.

# Examples

```jldoctest
julia> using StatsBase

julia> w0 = pweights([0.25, 0.25, 0.25, 0.25]);

julia> wi = pweights([0.1, 0.2, 0.3, 0.4]);

julia> PortfolioOptimisers.get_epw(H1_EntropyPooling(), w0, wi)
4-element ProbabilityWeights{Float64, Float64, Vector{Float64}}:
 0.25
 0.25
 0.25
 0.25

julia> PortfolioOptimisers.get_epw(H2_EntropyPooling(), w0, wi)
4-element ProbabilityWeights{Float64, Float64, Vector{Float64}}:
 0.1
 0.2
 0.3
 0.4
```

# Related

  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function get_epw(::H1_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return w0
end
function get_epw(::H2_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return wi
end
"""
    struct CVaREntropyPooling{T1, T2} <: AbstractEntropyPoolingOptimiser
        args::T1
        kwargs::T2
    end

Conditional Value-at-Risk (CVaR) entropy pooling optimiser.

`CVaREntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptimiser`](@ref) that uses root-finding algorithms from [`Roots.jl`](https://github.com/JuliaMath/Roots.jl) to solve entropy pooling problems with CVaR (Conditional Value-at-Risk) view constraints. This optimiser is designed for scenarios where CVaR views are specified and requires robust numerical methods to find the solution.

# Fields

  - `args`: Tuple of arguments passed to the root-finding algorithm (e.g., `Roots.Brent()`).
  - `kwargs`: Named tuple of keyword arguments for the root-finding algorithm.

# Constructor

    CVaREntropyPooling(; args::Tuple = (Roots.Brent(),), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct CVaREntropyPooling{T1, T2} <: AbstractEntropyPoolingOptimiser
    args::T1
    kwargs::T2
    function CVaREntropyPooling(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function CVaREntropyPooling(; args::Tuple = (Roots.Brent(),), kwargs::NamedTuple = (;))
    return CVaREntropyPooling(args, kwargs)
end
"""
    struct OptimEntropyPooling{T1, T2, T3, T4, T5} <: AbstractEntropyPoolingOptimiser
        args::T1
        kwargs::T2
        sc1::T3
        sc2::T4
        alg::T5
    end

[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)-based entropy pooling optimiser.

`OptimEntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptimiser`](@ref) that uses [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) to solve entropy pooling problems. This optimiser supports both logarithmic and exponential entropy pooling objectives, and allows for flexible configuration of solver arguments, scaling parameters, and algorithm selection.

# Fields

  - `args`: Tuple of arguments passed to the [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) solver.
  - `kwargs`: Named tuple of keyword arguments for the [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) solver.
  - `sc1`: Scaling parameter for the objective function.
  - `sc2`: Slack parameter for relaxing fixed equality constraint penalties so that they can be satisfied more easily.
  - `alg`: Entropy pooling optimisation algorithm.

# Constructor

    OptimEntropyPooling(; args::Tuple = (), kwargs::NamedTuple = (;), sc1::Number = 1,
                        sc2::Number = 1e3,
                        alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())

Keyword arguments correspond to the fields above.

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
     alg ┴ ExpEntropyPooling()
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`CVaREntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)
"""
struct OptimEntropyPooling{T1, T2, T3, T4, T5} <: AbstractEntropyPoolingOptimiser
    args::T1
    kwargs::T2
    sc1::T3
    sc2::T4
    alg::T5
    function OptimEntropyPooling(args::Tuple, kwargs::NamedTuple, sc1::Number, sc2::Number,
                                 alg::AbstractEntropyPoolingOptAlgorithm)
        @argcheck(sc1 >= zero(sc1))
        return new{typeof(args), typeof(kwargs), typeof(sc1), typeof(sc2), typeof(alg)}(args,
                                                                                        kwargs,
                                                                                        sc1,
                                                                                        sc2,
                                                                                        alg)
    end
end
function OptimEntropyPooling(; args::Tuple = (), kwargs::NamedTuple = (;), sc1::Number = 1,
                             sc2::Number = 1e3,
                             alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())
    return OptimEntropyPooling(args, kwargs, sc1, sc2, alg)
end
"""
    struct JuMPEntropyPooling{T1, T2, T3, T4, T5} <: AbstractEntropyPoolingOptimiser
        slv::T1
        sc1::T2
        sc2::T3
        so::T4
        alg::T5
    end

[`JuMP.jl`](https://github.com/jump-dev/JuMP.jl)-based entropy pooling optimiser.

`JuMPEntropyPooling` is a concrete subtype of [`AbstractEntropyPoolingOptimiser`](@ref) that uses [JuMP.jl](https://github.com/jump-dev/JuMP.jl) to solve entropy pooling problems. This optimiser supports both logarithmic and exponential entropy pooling objectives, and allows for flexible configuration of solver arguments, scaling parameters, and algorithm selection.

# Fields

  - `slv`: Solver object or vector of solvers for JuMP.jl.
  - `sc1`: Scaling parameter for the objective function.
  - `sc2`: Scaling parameter for constraint penalties.
  - `so`: Scaling parameter for the objective expression.
  - `alg`: Entropy pooling optimisation algorithm.

# Constructor

    JuMPEntropyPooling(; slv::USolverVec, sc1::Number = 1,
                       sc2::Number = 1e5, so::Number = 1,
                       alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())

Keyword arguments correspond to the fields above.

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
struct JuMPEntropyPooling{T1, T2, T3, T4, T5} <: AbstractEntropyPoolingOptimiser
    slv::T1
    sc1::T2
    sc2::T3
    so::T4
    alg::T5
    function JuMPEntropyPooling(slv::USolverVec, sc1::Number, sc2::Number, so::Number,
                                alg::AbstractEntropyPoolingOptAlgorithm)
        if isa(slv, VecSolver)
            @argcheck(!isempty(slv))
        end
        @argcheck(sc1 >= zero(sc1))
        @argcheck(sc2 >= zero(sc2))
        @argcheck(so >= zero(so))
        return new{typeof(slv), typeof(sc1), typeof(sc2), typeof(so), typeof(alg)}(slv, sc1,
                                                                                   sc2, so,
                                                                                   alg)
    end
end
function JuMPEntropyPooling(; slv::USolverVec, sc1::Number = 1, sc2::Number = 1e5,
                            so::Number = 1,
                            alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())
    return JuMPEntropyPooling(slv, sc1, sc2, so, alg)
end
"""
    struct EntropyPoolingPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
                               T16} <: AbstractLowOrderPriorEstimator_AF
        pe::T1
        mu_views::T2
        var_views::T3
        cvar_views::T4
        sigma_views::T5
        sk_views::T6
        kt_views::T7
        rho_views::T8
        var_alpha::T9
        cvar_alpha::T10
        sets::T11
        ds_opt::T12
        dm_opt::T13
        opt::T14
        w::T15
        alg::T16
    end

Entropy pooling prior estimator for asset returns.

`EntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports moment and view constraints (mean, variance, CVaR, covariance, skewness, kurtosis, correlation), flexible confidence specification, and composable optimisation algorithms. The estimator integrates asset sets, view constraints, and multiple entropy pooling algorithms (Optim.jl, JuMP.jl, CVaR root-finding), and allows for custom prior weights and solver configuration.

# Fields

  - `pe`: Prior estimator for asset returns.
  - `mu_views`: Mean view constraints.
  - `sigma_views`: Variance view constraints.
  - `cvar_views`: CVaR view constraints.
  - `sigma_views`: Covariance view constraints.
  - `sk_views`: Skewness view constraints.
  - `kt_views`: Kurtosis view constraints.
  - `rho_views`: Correlation view constraints.
  - `var_alpha`: Confidence level for VaR (Value at Risk) views.
  - `cvar_alpha`: Confidence level for CVaR (Conditional Value at Risk) views.
  - `sets`: Asset sets.
  - `ds_opt`: CVaR entropy pooling optimiser.
  - `dm_opt`: Optim.jl-based entropy pooling optimiser.
  - `opt`: Main entropy pooling optimiser.
  - `w`: Prior weights.
  - `alg`: Entropy pooling algorithm.

# Constructor

    EntropyPoolingPrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                        mu_views::Option{<:LinearConstraintEstimator} = nothing,
                        var_views::Option{<:LinearConstraintEstimator} = nothing,
                        cvar_views::Option{<:LinearConstraintEstimator} = nothing,
                        sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                        sk_views::Option{<:LinearConstraintEstimator} = nothing,
                        kt_views::Option{<:LinearConstraintEstimator} = nothing,
                        rho_views::Option{<:LinearConstraintEstimator} = nothing,
                        var_alpha::Number = 0.05, cvar_alpha::Number = 0.05,
                        sets::Option{<:AssetSets} = nothing,
                        ds_opt::Union{Nothing, <:CVaREntropyPooling} = nothing,
                        dm_opt::Union{Nothing, <:OptimEntropyPooling} = nothing,
                        opt::Union{<:OptimEntropyPooling, <:JuMPEntropyPooling} = OptimEntropyPooling(),
                        w::Option{<:ProbabilityWeights} = nothing,
                        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())

Keyword arguments correspond to the fields above.

## Validation

  - If any view constraint is provided, `sets` must not be `nothing`.
  - If not `nothing`, `0 < var_alpha < 1`.
  - If not `nothing`, `0 < cvar_alpha < 1`.
  - If `w` is provided, it must be non-empty and match the number of observations.

# Details

  - If `w` is provided, it is normalised to sum to 1; otherwise, uniform weights are used when `prior` is called.
  - If `var_views` is provided without `var_alpha`, defaults to `0.05`.
  - If `cvar_views` is provided without `cvar_alpha`, defaults to `0.05`.

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
              │           │      │   alg ┴ Full()
              │           │   mp ┼ DefaultMatrixProcessing
              │           │      │       pdm ┼ Posdef
              │           │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
              │           │      │   denoise ┼ nothing
              │           │      │    detone ┼ nothing
              │           │      │       alg ┴ nothing
              │        me ┼ SimpleExpectedReturns
              │           │   w ┴ nothing
              │   horizon ┴ nothing
     mu_views ┼ LinearConstraintEstimator
              │   val ┴ Vector{String}: ["A == 0.03", "B + C == 0.04"]
    var_views ┼ nothing
   cvar_views ┼ nothing
  sigma_views ┼ nothing
     sk_views ┼ nothing
     kt_views ┼ nothing
    rho_views ┼ nothing
    var_alpha ┼ nothing
   cvar_alpha ┼ nothing
         sets ┼ AssetSets
              │    key ┼ String: "nx"
              │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
       ds_opt ┼ nothing
       dm_opt ┼ nothing
          opt ┼ OptimEntropyPooling
              │     args ┼ Tuple{}: ()
              │   kwargs ┼ @NamedTuple{}: NamedTuple()
              │      sc1 ┼ Int64: 1
              │      sc2 ┼ Float64: 1000.0
              │      alg ┴ ExpEntropyPooling()
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
struct EntropyPoolingPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
                           T16} <: AbstractLowOrderPriorEstimator_AF
    pe::T1
    mu_views::T2
    var_views::T3
    cvar_views::T4
    sigma_views::T5
    sk_views::T6
    kt_views::T7
    rho_views::T8
    var_alpha::T9
    cvar_alpha::T10
    sets::T11
    ds_opt::T12
    dm_opt::T13
    opt::T14
    w::T15
    alg::T16
    function EntropyPoolingPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mu_views::Option{<:LinearConstraintEstimator},
                                 var_views::Option{<:LinearConstraintEstimator},
                                 cvar_views::Option{<:LinearConstraintEstimator},
                                 sigma_views::Option{<:LinearConstraintEstimator},
                                 sk_views::Option{<:LinearConstraintEstimator},
                                 kt_views::Option{<:LinearConstraintEstimator},
                                 rho_views::Option{<:LinearConstraintEstimator},
                                 var_alpha::Option{<:Number}, cvar_alpha::Option{<:Number},
                                 sets::Option{<:AssetSets},
                                 ds_opt::Union{Nothing, <:CVaREntropyPooling},
                                 dm_opt::Union{Nothing, <:OptimEntropyPooling},
                                 opt::Union{<:OptimEntropyPooling, <:JuMPEntropyPooling},
                                 w::Option{<:ProbabilityWeights},
                                 alg::AbstractEntropyPoolingAlgorithm)
        if !isnothing(w)
            @argcheck(!isempty(w))
            if ismutable(w.values)
                normalize!(w, 1)
            else
                w = pweights(normalize(w, 1))
            end
        end
        if !isnothing(mu_views) ||
           !isnothing(var_views) ||
           !isnothing(cvar_views) ||
           !isnothing(sigma_views) ||
           !isnothing(sk_views) ||
           !isnothing(kt_views) ||
           !isnothing(rho_views)
            @argcheck(!isnothing(sets))
        end
        if !isnothing(var_views)
            if !isnothing(var_alpha)
                @argcheck(zero(var_alpha) < var_alpha < one(var_alpha))
            else
                var_alpha = 0.05
            end
        end
        if !isnothing(cvar_views)
            if !isnothing(cvar_alpha)
                @argcheck(zero(cvar_alpha) < cvar_alpha < one(cvar_alpha))
            else
                cvar_alpha = 0.05
            end
        end
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(sigma_views), typeof(sk_views), typeof(kt_views),
                   typeof(rho_views), typeof(var_alpha), typeof(cvar_alpha), typeof(sets),
                   typeof(ds_opt), typeof(dm_opt), typeof(opt), typeof(w), typeof(alg)}(pe,
                                                                                        mu_views,
                                                                                        var_views,
                                                                                        cvar_views,
                                                                                        sigma_views,
                                                                                        sk_views,
                                                                                        kt_views,
                                                                                        rho_views,
                                                                                        var_alpha,
                                                                                        cvar_alpha,
                                                                                        sets,
                                                                                        ds_opt,
                                                                                        dm_opt,
                                                                                        opt,
                                                                                        w,
                                                                                        alg)
    end
end
function EntropyPoolingPrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             mu_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_views::Option{<:LinearConstraintEstimator} = nothing,
                             cvar_views::Option{<:LinearConstraintEstimator} = nothing,
                             sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                             sk_views::Option{<:LinearConstraintEstimator} = nothing,
                             kt_views::Option{<:LinearConstraintEstimator} = nothing,
                             rho_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_alpha::Option{<:Number} = nothing,
                             cvar_alpha::Option{<:Number} = nothing,
                             sets::Option{<:AssetSets} = nothing,
                             ds_opt::Union{Nothing, <:CVaREntropyPooling} = nothing,
                             dm_opt::Union{Nothing, <:OptimEntropyPooling} = nothing,
                             opt::Union{<:OptimEntropyPooling, <:JuMPEntropyPooling} = OptimEntropyPooling(),
                             w::Option{<:ProbabilityWeights} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, sigma_views, sk_views,
                               kt_views, rho_views, var_alpha, cvar_alpha, sets, ds_opt,
                               dm_opt, opt, w, alg)
end
function Base.getproperty(obj::EntropyPoolingPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function factory(pe::EntropyPoolingPrior, w::Option{<:AbstractWeights} = nothing)
    return EntropyPoolingPrior(; pe = factory(pe.pe, w), mu_views = pe.mu_views,
                               var_views = pe.var_views, cvar_views = pe.cvar_views,
                               sigma_views = pe.sigma_views, sk_views = pe.sk_views,
                               kt_views = pe.kt_views, rho_views = pe.rho_views,
                               var_alpha = pe.var_alpha, cvar_alpha = pe.cvar_alpha,
                               sets = pe.sets, ds_opt = pe.ds_opt, dm_opt = pe.dm_opt,
                               opt = pe.opt, w = nothing_scalar_array_factory(pe.w, w),
                               alg = pe.alg)
end
"""
    add_ep_constraint!(epc::AbstractDict, lhs::NumMat, rhs::NumVec, key::Symbol)

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
function add_ep_constraint!(epc::AbstractDict, lhs::NumMat, rhs::NumVec, key::Symbol)
    sc = norm(lhs)
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
    replace_prior_views(res::AbstractVector{<:ParsingResult}, args...; kwargs...)

Broadcast prior reference replacement across multiple view constraints.

`replace_prior_views` applies [`replace_prior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values. This enables efficient batch processing of multiple view constraints in entropy pooling routines.

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
function replace_prior_views(res::AbstractVector{<:ParsingResult}, args...; kwargs...)
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

  - `nothing`: No operation is performed.

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
    mu_views = parse_equation(mu_views.val; datatype = eltype(pr.X))
    mu_views = replace_group_by_assets(mu_views, sets, false, true, false)
    mu_views = replace_prior_views(mu_views, pr, sets, :mu; strict = strict)
    lcs = get_linear_constraints(mu_views, sets; datatype = eltype(pr.X), strict = strict)
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        add_ep_constraint!(epc, getproperty(lcs, p).A * transpose(pr.X),
                           getproperty(lcs, p).B, p)
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
        add_ep_constraint!(epc, transpose(pr.X[:, fix]), pr.mu[fix], :feq)
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
    #! Don't use a view, use a copy, value at risk uses partialsort!
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ValueatRisk(; alpha = alpha)(pr.X[:, i])
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

  - `nothing`: No operation is performed.

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
    var_views = parse_equation(var_views.val; ops1 = ("==", ">="),
                               ops2 = (:call, :(==), :(>=)), datatype = eltype(pr.X))
    var_views = replace_group_by_assets(var_views, sets, false, true, false)
    var_views = replace_prior_views(var_views, pr, sets, :var, alpha; strict = strict)
    lcs = get_linear_constraints(var_views, sets; datatype = eltype(pr.X), strict = strict)
    @argcheck(!(!isnothing(lcs.ineq) && !any(x -> (iszero(x) || isone(x)), lcs.A_ineq) ||
                !isnothing(lcs.eq) && !any(x -> (iszero(x) || isone(x)), lcs.A_eq)),
              ArgumentError("`var_view` only supports coefficients of 1.\n$var_views"))
    @argcheck(!(!isnothing(lcs.ineq) &&
                any(x -> x != 1, count(!iszero, lcs.A_ineq; dims = 2)) ||
                !isnothing(lcs.eq) && any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2))),
              ArgumentError("Cannot mix multiple assets in a single `var_view`.\n$var_views"))
    @argcheck(!(!isnothing(lcs.eq) && any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq) ||
                !isnothing(lcs.ineq) &&
                any(x -> x < zero(eltype(x)), lcs.A_ineq .* lcs.B_ineq)),
              ArgumentError("`var_view` cannot be negative.\n$var_views"))
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        B = getproperty(lcs, p).B
        for i in eachindex(B)
            j = .!iszero.(A[i, :])
            #! Figure out a way to include pr.w, probably see how it's implemented in ValueatRisk.
            idx = findall(x -> x <= -abs(B[i]), view(pr.X, :, j))
            @argcheck(!isempty(idx),
                      ArgumentError("View `$(var_views[i].eqn)` is too extreme, the maximum viable for asset $(findfirst(x -> x == true, j)) is $(-minimum(pr.X[:,j])). Please lower it or use a different prior with fatter tails."))
            sign = ifelse(p == :eq || B[i] >= zero(eltype(B)), one(eltype(B)),
                          -one(eltype(B)))
            Ai = zeros(eltype(pr.X), 1, size(pr.X, 1))
            Ai[1, idx] .= sign
            add_ep_constraint!(epc, Ai, [sign * alpha], p)
        end
    end
    return nothing
end
"""
    entropy_pooling(w::NumVec, epc::AbstractDict, opt::OptimEntropyPooling)

Solve the dual of the exponential entropy pooling formulation using Optim.jl.

`entropy_pooling` computes posterior probabilities by minimising the exponential divergence between prior and posterior weights, subject to moment and view constraints. The optimisation is performed using [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), supporting box constraints and slack variables for relaxed equality constraints. This method is used internally by [`EntropyPoolingPrior`](@ref) when the optimiser is an [`OptimEntropyPooling`](@ref).

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `opt: Optim.jl-based entropy pooling optimiser with exponential objective.

      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: use the exponential formulation.
      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: use the logarithmic formulation.

# Returns

  - `pw::ProbabilityWeights`: Posterior probability weights satisfying the constraints.

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
function entropy_pooling(w::NumVec, epc::AbstractDict,
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
        return opt.sc1 * sum(y) + dot(x, B)
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                            opt.kwargs...)
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return pweights(w .* exp.(-transpose(A) * x .- one(eltype(w))))
end
function entropy_pooling(w::NumVec, epc::AbstractDict,
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
        return opt.sc1 * (dot(x, grad) - dot(y, log_x - log_p))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                            opt.kwargs...)
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(A) * x)))
end
"""
    entropy_pooling(w::NumVec, epc::AbstractDict, opt::JuMPEntropyPooling)

Solve the primal of the exponential entropy pooling formulation using JuMP.jl.

`entropy_pooling` computes posterior probabilities by minimising the exponential divergence between prior and posterior weights, subject to moment and view constraints. The optimisation is performed using [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl), supporting relative entropy cones and slack variables for relaxed equality constraints. This method is used internally by [`EntropyPoolingPrior`](@ref) when the optimiser is a [`JuMPEntropyPooling`](@ref).

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `opt`: JuMP.jl-based entropy pooling optimiser with exponential objective.

      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: use the exponential formulation.
      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: use the logarithmic formulation.

# Returns

  - `pw::ProbabilityWeights`: Posterior probability weights satisfying the constraints.

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
function entropy_pooling(w::NumVec, epc::AbstractDict,
                         opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                 <:ExpEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    T = length(w)
    model = Model()
    @variables(model, begin
                   t
                   x[1:T] >= 0
               end)
    @constraints(model,
                 begin
                     sc1 * (sum(x) - one(eltype(w))) == 0
                     [sc1 * t; sc1 * w; sc1 * x] in MOI.RelativeEntropyCone(2 * T + 1)
                 end)
    @expression(model, obj_expr, so * t)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        @constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        @constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        @constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        @variables(model, begin
                       tc
                       c[1:N]
                   end)
        @constraints(model, begin
                         cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                         [sc1 * tc; sc1 * c] in MOI.NormOneCone(N + 1)
                     end)
        add_to_expression!(obj_expr, so * sc2 * tc)
    end
    @objective(model, Min, obj_expr)
    @argcheck(optimise_JuMP_model!(model, slv).success,
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    return pweights(value.(x))
end
function entropy_pooling(w::NumVec, epc::AbstractDict,
                         opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                 <:LogEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    model = Model()
    T = length(w)
    log_p = log.(w)
    # Decision variables (posterior probabilities)
    @variables(model, begin
                   x[1:T]
                   t
               end)
    @expression(model, obj_expr, so * t)
    # Equality constraints from A_eq and B_eq and probabilities equal to 1
    @constraints(model,
                 begin
                     sc1 * (sum(x) - one(eltype(w))) == 0
                     [sc1 * t; fill(sc1, T); sc1 * x] in MOI.RelativeEntropyCone(2 * T + 1)
                 end)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        @constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        @constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        @constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        @variables(model, begin
                       tc
                       c[1:N]
                   end)
        @constraints(model, begin
                         cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                         [sc1 * tc; sc1 * c] in MOI.NormOneCone(N + 1)
                     end)
        add_to_expression!(obj_expr, so * sc2 * tc)
    end
    @objective(model, Min, obj_expr - so * dot(x, log_p))
    # Solve the optimization problem
    @argcheck(optimise_JuMP_model!(model, slv).success,
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    return pweights(value.(x))
end
"""
    ep_cvar_views_solve!(cvar_views::Nothing, epc::AbstractDict, ::Any, ::Any, ::Number,
                         w::AbstractWeights, opt::AbstractEntropyPoolingOptimiser, ::Any, ::Any;
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

  - `pw::ProbabilityWeights`: Posterior probability weights satisfying the constraints.

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
                              w::AbstractWeights, opt::AbstractEntropyPoolingOptimiser,
                              ::Any, ::Any; kwargs...)
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
    #! Don't use a view, use a copy, value at risk uses partialsort!
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ConditionalValueatRisk(; alpha = alpha)(pr.X[:, i])
end
"""
    ep_cvar_views_solve!(cvar_views::LinearConstraintEstimator, epc::AbstractDict,
                         pr::AbstractPriorResult, sets::AssetSets, alpha::Number,
                         w::AbstractWeights, opt::AbstractEntropyPoolingOptimiser,
                         ds_opt::Union{Nothing, <:CVaREntropyPooling},
                         dm_opt::Union{Nothing, <:OptimEntropyPooling}; strict::Bool = false)

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

  - `pw::ProbabilityWeights`: Posterior probability weights satisfying CVaR view constraints.

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
                              w::AbstractWeights, opt::AbstractEntropyPoolingOptimiser,
                              ds_opt::Union{Nothing, <:CVaREntropyPooling},
                              dm_opt::Union{Nothing, <:OptimEntropyPooling};
                              strict::Bool = false)
    cvar_views = parse_equation(cvar_views.val; ops1 = ("==",), ops2 = (:call, :(==)),
                                datatype = eltype(pr.X))
    cvar_views = replace_group_by_assets(cvar_views, sets, false, true, false)
    cvar_views = replace_prior_views(cvar_views, pr, sets, :cvar, alpha; strict = strict)
    lcs = get_linear_constraints(cvar_views, sets; datatype = eltype(pr.X), strict = strict)
    @argcheck(isnothing(lcs.ineq), "`cvar_view` can only have equality constraints.")
    @argcheck(!any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2)),
              ArgumentError("Cannot mix multiple assets in a single `cvar_view`."))
    @argcheck(!any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq),
              ArgumentError("`cvar_view` cannot be negative."))
    idx = dropdims(.!iszero.(sum(lcs.A_eq; dims = 1)); dims = 1)
    idx2 = .!iszero.(lcs.A_eq)
    B = lcs.B_eq ./ view(lcs.A_eq, idx2)
    X = view(pr.X, :, idx)
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
        @argcheck(all(zero(eltype(etas)) .<= etas .<= B))
        pos_part = max.(-X .- transpose(etas), zero(eltype(X)))
        add_ep_constraint!(epc, transpose(pos_part / alpha), B .- etas, :cvar_eq)
        wi = entropy_pooling(w, epc, opt)
        err = if N == 1
            sum(wi[.!iszero.(pos_part)]) - alpha
        else
            norm([ConditionalValueatRisk(; alpha = alpha, w = wi)(X[:, i]) - B[i]
                  for i in 1:N]) / sqrt(N)
        end
        return wi, err
    end
    res = if N == 1
        try
            [find_zero(x -> func(x)[2], (0, B[1]), d_opt.args...; d_opt.kwargs...)]
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
    return diag(pr.sigma)[i]
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
    sigma_views = parse_equation(sigma_views.val; datatype = eltype(pr.X))
    sigma_views = replace_group_by_assets(sigma_views, sets, false, true, false)
    sigma_views = replace_prior_views(sigma_views, pr, sets, :sigma; strict = strict)
    lcs = get_linear_constraints(sigma_views, sets; datatype = eltype(pr.X),
                                 strict = strict)
    tmp = transpose((pr.X .- transpose(pr.mu)) .^ 2)
    to_fix = falses(size(pr.X, 2))
    for p in propertynames(lcs)
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
  - Uses the prior variance values from `diag(pr.sigma)` for the constraint right-hand side.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
                    pr::AbstractPriorResult)
    sigma = diag(pr.sigma)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(pr.X[:, fix] .- transpose(pr.mu[fix])) .^ 2,
                           sigma[fix], :feq)
        fixed .= fixed .| fix
    end
    return nothing
end
"""
    replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets;
                        strict::Bool = false)

Replace correlation prior references in view parsing results with their corresponding prior values.

`replace_prior_views` scans a parsed correlation view constraint (`ParsingResult`) for references to prior values (e.g., `prior(A, B)`), and replaces them with the actual prior correlation value from the provided prior result object. This ensures that prior-based terms in correlation view constraints are treated as constants and not as variables in the optimisation. If an asset referenced in a prior is not found in the asset set, a warning is issued (or an error if `strict=true`). If all variables in the view are prior references, an error is thrown.

# Arguments

  - `res`: Parsed correlation view constraint containing variables and coefficients.
  - `pr`: Prior result object containing prior correlation values.
  - `sets`: Asset set mapping asset names to indices.
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
function replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets;
                             strict::Bool = false)
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
                      ArgumentError("Correlation prior view $(v) must be of the form `(a, b)`."))
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
        rhs -= get_pr_value(pr, j, k) * c
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
function get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, args...)
    return cov2cor(pr.sigma)[i, j]
end
function get_pr_value(pr::AbstractPriorResult, i::IntVec, j::IntVec, args...)
    return norm(cov2cor(pr.sigma)[i, j]) / length(i)
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
    rho_views = parse_equation(rho_views.val; datatype = eltype(pr.X))
    rho_views = replace_group_by_assets(rho_views, sets, false, true, true)
    rho_views = replace_prior_views(rho_views, pr, sets; strict = strict)
    to_fix = falses(size(pr.X, 2))
    sigma = diag(pr.sigma)
    for rho_view in rho_views
        @argcheck(length(rho_view.vars) == 1,
                  "Cannot mix multiple correlation pairs in a single view `$(rho_view.eqn)`.")
        @argcheck(-one(eltype(pr.X)) <= rho_view.rhs <= one(eltype(pr.X)),
                  "Correlation prior rho_view `$(rho_view.eqn)` must be in [-1, 1].")
        d = ifelse(rho_view.op == ">=", -1, 1)
        i, j = rho_view.ij[1]
        sigma_ij = if !isa(i, AbstractVector)
            sqrt(sigma[i] * sigma[j])
        else
            norm(sigma[i] .* sigma[j])
        end
        Ai = d * rho_view.coef[1] * view(pr.X, :, i) .* view(pr.X, :, j)
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
    return Skewness()([1], reshape(pr.X[:, i], :, 1))
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
    skew_views = parse_equation(skew_views.val; datatype = eltype(pr.X))
    skew_views = replace_group_by_assets(skew_views, sets, false, true, false)
    skew_views = replace_prior_views(skew_views, pr, sets, :skew; strict = strict)
    lcs = get_linear_constraints(skew_views, sets; datatype = eltype(pr.X), strict = strict)
    sigma = diag(pr.sigma)
    tmp = transpose((pr.X .^ 3 .- transpose(pr.mu) .^ 3 .- 3 * transpose(pr.mu .* sigma)) ./
                    transpose(sigma .* sqrt.(sigma)))
    to_fix = falses(size(pr.X, 2))
    for p in propertynames(lcs)
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
    return HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                        reshape(pr.X[:,
                                                                                                     i],
                                                                                                :,
                                                                                                1))
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
    kurtosis_views = parse_equation(kurtosis_views.val; datatype = eltype(pr.X))
    kurtosis_views = replace_group_by_assets(kurtosis_views, sets, false, true, false)
    kurtosis_views = replace_prior_views(kurtosis_views, pr, sets, :kurtosis;
                                         strict = strict)
    lcs = get_linear_constraints(kurtosis_views, sets; datatype = eltype(pr.X),
                                 strict = strict)
    X_sq = pr.X .^ 2
    mu_sq = pr.mu .^ 2
    tmp = transpose((X_sq .* X_sq .- 4 * transpose(pr.mu) .* X_sq .* pr.X .+
                     6 * transpose(mu_sq) .* X_sq .- 3 * transpose(mu_sq .* mu_sq)) ./
                    transpose(diag(pr.sigma)) .^ 2)
    to_fix = falses(size(pr.X, 2))
    for p in propertynames(lcs)
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
                                  <:Union{<:H1_EntropyPooling, <:H2_EntropyPooling}},
          X::NumMat; F::Option{<:NumMat} = nothing, dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute entropy pooling prior moments for asset returns with iterative constraint enforcement.

`prior` estimates the mean and covariance of asset returns using the entropy pooling framework, supporting iterative constraint enforcement via the `H1_EntropyPooling` and `H2_EntropyPooling` algorithms. It integrates moment and view constraints (mean, variance, CVaR, skewness, kurtosis, correlation), flexible confidence specification, and composable optimisation algorithms. The method iteratively applies constraints, updating prior weights and moments at each step, and ensures that higher moment views do not inadvertently alter lower moments.

# Arguments

  - `pe`: Entropy pooling prior estimator with iterative algorithm .
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix (default: `nothing`).
  - `dims`: Dimension along which to compute moments.
  - If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Validation

  - `dims in (1, 2)`.
  - If any view constraint is provided, `!isnothing(sets)`.
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
  - [`ep_rho_views!`](@ref)
  - [`fix_mu!`](@ref)
  - [`fix_sigma!`](@ref)
"""
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any,
                                       <:Union{<:H1_EntropyPooling, <:H2_EntropyPooling}},
               X::NumMat, F::Option{<:NumMat} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    T, N = size(X)
    w1 = w0 = if isnothing(pe.w)
        iT = inv(T)
        pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T)
        pe.w
    end
    fixed = falses(N, 2)
    epc = Dict{Symbol, Tuple{<:NumMat, <:NumVec}}()
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
    # sigma
    if !isnothing(pe.sigma_views)
        to_fix = ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
        fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha,
                                  get_epw(pe.alg, w0, w1), pe.opt, pe.ds_opt, pe.dm_opt;
                                  strict = strict)
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
                                  get_epw(pe.alg, w0, w1), pe.opt, pe.ds_opt, pe.dm_opt;
                                  strict = strict)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = pr
    ens = exp(entropy(w1))
    kld = kldivergence(w1, w0)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w1, ens = ens,
                         kld = kld, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(rr) ? w1 : nothing)
end
"""
    prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:H0_EntropyPooling}, X::NumMat;
          F::Option{<:NumMat} = nothing, dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute entropy pooling prior moments for asset returns with single-shot constraint enforcement.

`prior` estimates the mean and covariance of asset returns using the entropy pooling framework, enforcing all moment and view constraints in a single optimisation step via the `H0_EntropyPooling` algorithm. This approach is fast but may distort lower moments when higher moment views are present, as all constraints are applied simultaneously.

# Arguments

  - `pe`: Entropy pooling prior estimator with single-shot algorithm.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - `dims`: Dimension along which to compute moments,
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, weights, effective number of scenarios, Kullback-Leibler divergence, and optional factor moments.

# Validation

  - `dims in (1, 2)`.
  - If any view constraint is provided, `!isnothing(pe.sets)`.
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
  - [`ep_rho_views!`](@ref)
"""
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:H0_EntropyPooling}, X::NumMat,
               F::Option{<:NumMat} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    T = size(X, 1)
    w0 = if isnothing(pe.w)
        iT = inv(T)
        pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T)
        pe.w
    end
    epc = Dict{Symbol, Tuple{<:NumMat, <:NumVec}}()
    # mu and VaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, pe.var_alpha; strict = strict)
    # sigma
    if !isnothing(pe.sigma_views)
        ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
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
    ens = exp(entropy(w1))
    kld = kldivergence(w1, w0)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w1, ens = ens,
                         kld = kld, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(rr) ? w1 : nothing)
end

export LogEntropyPooling, ExpEntropyPooling, EntropyPoolingPrior, H0_EntropyPooling,
       H1_EntropyPooling, H2_EntropyPooling, JuMPEntropyPooling, OptimEntropyPooling,
       CVaREntropyPooling
