"""
    abstract type AbstractOrderedWeightsArrayEstimator <: AbstractEstimator end

Abstract supertype for all Ordered Weights Array (OWA) estimator types in `PortfolioOptimisers.jl`.

All concrete types implementing OWA estimation algorithms should subtype `AbstractOrderedWeightsArrayEstimator`. This enables a consistent interface for OWA-based estimators throughout the package.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
"""
abstract type AbstractOrderedWeightsArrayEstimator <: AbstractEstimator end
"""
    abstract type AbstractOrderedWeightsArrayAlgorithm <: AbstractAlgorithm end

Abstract supertype for all Ordered Weights Array (OWA) algorithm types in `PortfolioOptimisers.jl`.

All concrete types implementing specific OWA algorithms should subtype `AbstractOrderedWeightsArrayAlgorithm`. This enables flexible extension and dispatch of OWA routines.

# Related

  - [`MaximumEntropy`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)
"""
abstract type AbstractOrderedWeightsArrayAlgorithm <: AbstractAlgorithm end
"""
    struct MaximumEntropy <: AbstractOrderedWeightsArrayAlgorithm end

Represents the Maximum Entropy algorithm for Ordered Weights Array (OWA) estimation.

The Maximum Entropy algorithm seeks the OWA weights that maximize entropy, resulting in the most "uninformative" or uniform distribution of weights subject to the imposed constraints.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
"""
struct MaximumEntropy <: AbstractOrderedWeightsArrayAlgorithm end
"""
    struct MinimumSquaredDistance <: AbstractOrderedWeightsArrayAlgorithm end

Represents the Minimum Squared Distance algorithm for Ordered Weights Array (OWA) estimation.

The Minimum Squared Distance algorithm finds OWA weights that minimize the squared distance between adjacent entries in the array, subject to the OWA constraints. This approach promotes smoothness in the resulting weights.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
"""
struct MinimumSquaredDistance <: AbstractOrderedWeightsArrayAlgorithm end
"""
    struct MinimumSumSquares <: AbstractOrderedWeightsArrayAlgorithm end

Represents the Minimum Sum of Squares algorithm for Ordered Weights Array (OWA) estimation.

The Minimum Sum of Squares algorithm minimizes the sum of squared OWA weights, subject to the OWA constraints. This promotes sparsity or concentration in the resulting weights. This can be used to emphasize extreme order statistics in OWA-based risk measures.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
"""
struct MinimumSumSquares <: AbstractOrderedWeightsArrayAlgorithm end
"""
    struct NormalisedConstantRelativeRiskAversion{T1} <: AbstractOrderedWeightsArrayEstimator
        g::T1
    end

Estimator type for normalised constant relative risk aversion (CRRA) OWA weights.

This struct represents an estimator for Ordered Weights Array (OWA) weights based on a normalised constant relative risk aversion parameter `g`. The CRRA approach generates OWA weights that interpolate between risk-neutral and risk-averse profiles, controlled by the parameter `g`.

# Fields

  - `g`: Risk aversion parameter.

# Constructor

    NormalisedConstantRelativeRiskAversion(; g::Number = 0.5)

Keyword arguments correspond to the fields above.

## Validation

  - `0 < g < 1`.

# Examples

```jldoctest
julia> NormalisedConstantRelativeRiskAversion()
NormalisedConstantRelativeRiskAversion
  g ┴ Float64: 0.5
```

# Related

  - [`AbstractOrderedWeightsArrayEstimator`](@ref)
  - [`owa_l_moment_crm`](@ref)
"""
struct NormalisedConstantRelativeRiskAversion{T1} <: AbstractOrderedWeightsArrayEstimator
    g::T1
    function NormalisedConstantRelativeRiskAversion(g::Number)
        @argcheck(zero(g) < g < one(g), DomainError("0 < g < 1 must hold. Got\ng => $g"))
        return new{typeof(g)}(g)
    end
end
function NormalisedConstantRelativeRiskAversion(; g::Number = 0.5)
    return NormalisedConstantRelativeRiskAversion(g)
end
"""
    struct OWAJuMP{T1, T2, T3, T4, T5} <: AbstractOrderedWeightsArrayEstimator
        slv::T1
        max_phi::T2
        sc::T3
        so::T4
        alg::T5
    end

Estimator type for OWA weights using JuMP-based optimization.

`OWAJuMP` encapsulates all configuration required to estimate OWA weights via mathematical programming using JuMP. It supports multiple algorithms and solver backends, and allows fine control over constraints and scaling.

# Fields

  - `slv`: Solver or vector of solvers to use.
  - `max_phi`: Maximum allowed value for any OWA weight.
  - `sc`: Scaling parameter for constraints.
  - `so`: Scaling parameter for the objective.
  - `alg`: Algorithm for OWA weight estimation.

# Constructor

    OWAJuMP(; slv::Slv_VecSlv = Solver(), max_phi::Number = 0.5, sc::Number = 1.0,
            so::Number = 1.0, alg::AbstractOrderedWeightsArrayAlgorithm = MaximumEntropy())

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(slv)`.
  - `0 < max_phi < 1`.
  - `isfinite(sc)` and `sc > 0`.
  - `isfinite(so)` and `so > 0`.

# Examples

```jldoctest
julia> OWAJuMP()
OWAJuMP
      slv ┼ Solver
          │          name ┼ String: ""
          │        solver ┼ nothing
          │      settings ┼ nothing
          │     check_sol ┼ @NamedTuple{}: NamedTuple()
          │   add_bridges ┴ Bool: true
  max_phi ┼ Float64: 0.5
       sc ┼ Float64: 1.0
       so ┼ Float64: 1.0
      alg ┴ MaximumEntropy()
```

# Related

  - [`AbstractOrderedWeightsArrayEstimator`](@ref)
  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`Solver`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
struct OWAJuMP{T1, T2, T3, T4, T5} <: AbstractOrderedWeightsArrayEstimator
    slv::T1
    max_phi::T2
    sc::T3
    so::T4
    alg::T5
    function OWAJuMP(slv::Slv_VecSlv, max_phi::Number, sc::Number, so::Number,
                     alg::AbstractOrderedWeightsArrayAlgorithm)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError)
        end
        @argcheck(zero(max_phi) < max_phi < one(max_phi),
                  DomainError("0 < max_phi < 1 must hold. Got\nmax_phi => $max_phi"))
        assert_nonempty_gt0_finite_val(sc, :sc)
        assert_nonempty_gt0_finite_val(so, :so)
        return new{typeof(slv), typeof(max_phi), typeof(sc), typeof(so), typeof(alg)}(slv,
                                                                                      max_phi,
                                                                                      sc,
                                                                                      so,
                                                                                      alg)
    end
end
function OWAJuMP(; slv::Slv_VecSlv = Solver(), max_phi::Number = 0.5, sc::Number = 1.0,
                 so::Number = 1.0,
                 alg::AbstractOrderedWeightsArrayAlgorithm = MaximumEntropy())
    return OWAJuMP(slv, max_phi, sc, so, alg)
end
"""
    ncrra_weights(weights::MatNum; g::Number = 0.5)

Compute normalised constant relative risk aversion (CRRA) Ordered Weights Array (OWA) weights.

This function generates OWA weights using a normalised CRRA scheme, parameterised by `g`. The CRRA approach interpolates between risk-neutral and risk-averse weighting profiles, controlled by the risk aversion parameter `g`. The resulting weights are normalised to sum to one and are suitable for use in OWA-based risk measures.

# Arguments

  - `weights`: Matrix of weights (typically order statistics or moment weights).
  - `g`: Risk aversion parameter.

# Validation

  - `0 < g < 1`.

# Returns

  - `w::Vector`: Vector of OWA weights, normalised to sum to one.

# Details

The function computes the OWA weights as follows:

 1. For each order statistic, recursively compute the CRRA weight using the formula:

    e *= g + i - 1
    phis[i] = e / factorial(i + 1)

 2. The vector `phis` is normalised to sum to one.
 3. The final OWA weights are computed as a weighted sum of the input `weights` and `phis`, with monotonicity enforced by taking the maximum up to each index.

# Examples

```jldoctest
julia> w = [1.0 0.5; 0.5 1.0]
2×2 Matrix{Float64}:
 1.0  0.5
 0.5  1.0

julia> PortfolioOptimisers.ncrra_weights(w, 0.5)
2-element Vector{Float64}:
 0.8333333333333333
 0.8333333333333333
```

# Related

  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`MatNum`](@ref)
"""
function ncrra_weights(weights::MatNum, g::Number = 0.5)
    N = size(weights, 2)
    phis = Vector{eltype(weights)}(undef, N)
    e = 1
    for i in eachindex(phis)
        e *= g + i - 1
        phis[i] = e / factorial(i + 1)
    end
    phis ./= sum(phis)
    a = weights * phis
    w = similar(a)
    w[1] = a[1]
    for i in 2:length(a)
        w[i] = maximum(a[1:i])
    end
    return w
end
"""
    owa_model_setup(method::OWAJuMP, weights::MatNum)

Construct a JuMP model for Ordered Weights Array (OWA) weight estimation.

This function sets up a JuMP optimization model for OWA weights, given an `OWAJuMP` estimator and a matrix of weights (e.g., order statistics or moment weights). The model includes variables for the OWA weights (`phi`) and auxiliary variables (`theta`), and enforces constraints for non-negativity, upper bounds, sum-to-one, monotonicity, and consistency with the input weights.

# Arguments

  - `method`: OWA estimator containing solver, scaling, and algorithm configuration.
  - `weights`: Matrix of weights (typically order statistics or moment weights).

# Returns

  - `model::JuMP.Model`: Configured JuMP model with variables and constraints for OWA weight estimation.

# Constraints

  - `phi` (OWA weights) are non-negative and bounded above by `max_phi`.
  - The sum of `phi` is 1.
  - `theta` is constrained to be equal to the weighted sum of the input weights and `phi`.
  - Monotonicity is enforced on `phi` and `theta`.

# Related

  - [`OWAJuMP`](@ref)
  - [`MatNum`](@ref)
  - [`owa_l_moment_crm`](@ref)
"""
function owa_model_setup(method::OWAJuMP, weights::MatNum)
    T, N = size(weights)
    model = JuMP.Model()
    max_phi = method.max_phi
    sc = method.sc
    @variables(model, begin
                   theta[1:T]
                   phi[1:N]
               end)
    @constraints(model, begin
                     sc * phi >= 0
                     sc * (phi .- max_phi) <= 0
                     sc * (sum(phi) - 1) == 0
                     sc * (theta - weights * phi) == 0
                     sc * (phi[2:end] - phi[1:(end - 1)]) <= 0
                     sc * (theta[2:end] - theta[1:(end - 1)]) >= 0
                 end)
    return model
end
"""
    owa_model_solve(model::JuMP.Model, method::OWAJuMP, weights::MatNum)

Solve a JuMP model for OWA weight estimation and extract the resulting OWA weights.

This function solves the provided JuMP model using the solver(s) specified in the `OWAJuMP` estimator. If the optimization is successful, it extracts the OWA weights (`phi`), normalises them to sum to one, and computes the final OWA weights as a weighted sum with the input `weights`. If the optimization fails, a warning is issued and a fallback to `ncrra_weights` is used.

# Arguments

  - `model`: JuMP model for OWA weight estimation.
  - `method`: OWA estimator containing solver configuration.
  - `weights`: Matrix of weights (typically order statistics or moment weights).

# Returns

  - `w::Vector`: Vector of OWA weights, normalised to sum to one.

# Details

  - If the solver succeeds, the solution is extracted from the `phi` variable and normalised.
  - If the solver fails, a warning is issued and the fallback `ncrra_weights(weights, 0.5)` is returned.

# Related

  - [`OWAJuMP`](@ref)
  - [`MatNum`](@ref)
  - [`owa_model_setup`](@ref)
  - [`ncrra_weights`](@ref)
"""
function owa_model_solve(model::JuMP.Model, method::OWAJuMP, weights::MatNum)
    slv = method.slv
    return if optimise_JuMP_model!(model, slv).success
        phi = model[:phi]
        phis = value.(phi)
        phis ./= sum(phis)
        w = weights * phis
    else
        @warn("Type: $method\nReverting to ncrra_weights.")
        w = ncrra_weights(weights, 0.5)
    end
end
"""
    owa_l_moment_crm(method::AbstractOrderedWeightsArrayEstimator, weights::MatNum)

Compute Ordered Weights Array (OWA) linear moment convex risk measure (CRM) weights using various estimation methods.

This function dispatches on the estimator `method` to compute OWA weights from a matrix of moment or order-statistic weights. It supports several OWA estimation approaches, including normalised constant relative risk aversion (CRRA) and JuMP-based optimization with different algorithms.

# Arguments

  - `method::NormalisedConstantRelativeRiskAversion`: Computes OWA weights using the normalised CRRA scheme, parameterised by the risk aversion parameter `g` in `method`. The resulting weights interpolate between risk-neutral and risk-averse profiles and are normalised to sum to one.
  - `method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MaximumEntropy}`: Computes OWA weights by solving a maximum entropy optimization problem using JuMP. This yields the most "uninformative" or uniform OWA weights subject to the imposed constraints.
  - `method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MinimumSquaredDistance}`: Computes OWA weights by minimizing the squared distance from a target or reference vector, regularizing the OWA weights towards a desired profile.
  - `method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MinimumSumSquares}`: Computes OWA weights by minimizing the sum of squared OWA weights, promoting sparsity or concentration in the resulting weights.
  - `weights`: Matrix of weights (e.g., order statistics or moment weights).

# Returns

  - `w::Vector`: Vector of OWA weights, normalised to sum to one.

# Related

  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`MaximumEntropy`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)
  - [`MatNum`](@ref)
  - [`ncrra_weights`](@ref)
"""
function owa_l_moment_crm(method::NormalisedConstantRelativeRiskAversion, weights::MatNum)
    return ncrra_weights(weights, method.g)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MaximumEntropy},
                          weights::MatNum)
    T = size(weights, 1)
    sc = method.sc
    so = method.so
    ovec = range(sc, sc; length = T)
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variables(model, begin
                   t
                   x[1:T]
               end)
    @constraints(model, begin
                     sc * (sum(x) - 1) == 0
                     [sc * t; ovec; sc * x] in MOI.RelativeEntropyCone(2 * T + 1)
                     [i = 1:T], [sc * x[i]; sc * theta[i]] in MOI.NormOneCone(2)
                 end)
    @objective(model, Max, -so * t)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                          <:MinimumSquaredDistance}, weights::MatNum)
    sc = method.sc
    so = method.so
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model,
                [sc * t; sc * (theta[2:end] - theta[1:(end - 1)])] in SecondOrderCone())
    @objective(model, Min, so * t)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MinimumSumSquares},
                          weights::MatNum)
    sc = method.sc
    so = method.so
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model, [sc * t; sc * theta] in SecondOrderCone())
    @objective(model, Min, so * t)
    return owa_model_solve(model, method, weights)
end
"""
    owa_gmd(T::Integer)

Compute the Ordered Weights Array (OWA) of the Gini Mean Difference (GMD) risk measure.

# Arguments

  - `T`: Number of observations.

# Returns

  - `w::Range`: Vector of OWA weights of length `T`.
"""
function owa_gmd(T::Integer)
    return (4 * (1:T) .- 2 * (T + 1)) / (T * (T - 1))
end
"""
    owa_cvar(T::Integer; alpha::Number = 0.05)

Compute the Ordered Weights Array (OWA) weights for the Conditional Value at Risk.

# Arguments

  - `T`: Number of observations.
  - `alpha`: Confidence level for CVaR.

# Validation

  - `0 < alpha < 1`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_wcvar`](@ref)
  - [`owa_tg`](@ref)
  - [`VecNum`](@ref)
"""
function owa_cvar(T::Integer, alpha::Number = 0.05)
    @argcheck(zero(alpha) < alpha < one(alpha),
              DomainError("0 < alpha < 1 must hold. Got\nalpha => $alpha"))
    k = floor(Int, T * alpha)
    w = zeros(typeof(alpha), T)
    w[1:k] .= -one(alpha) / (T * alpha)
    w[k + 1] = -one(alpha) - sum(w[1:k])
    return w
end
"""
    owa_wcvar(T::Integer, alphas::VecNum, weights::VecNum)

Compute the Ordered Weights Array (OWA) weights for a weighted combination of Conditional Value at Risk measures.

# Arguments

  - `T`: Number of observations.
  - `alphas`: Vector of confidence levels.
  - `weights`: Vector of weights.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_tg`](@ref)
  - [`VecNum`](@ref)
"""
function owa_wcvar(T::Integer, alphas::VecNum, weights::VecNum)
    w = zeros(promote_type(eltype(alphas), eltype(weights)), T)
    for (i, j) in zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end
    return w
end
"""
    owa_tg(T::Integer; alpha_i::Number = 1e-4, alpha::Number = 0.05, a_sim::Integer = 100)

Compute the Ordered Weights Array (OWA) weights for the tail Gini risk measure.

This function approximates the tail Gini risk measure by integrating over a range of CVaR levels from `alpha_i` to `alpha`, using `a_sim` points. The resulting weights are suitable for tail risk assessment.

# Arguments

  - `T`: Number of observations.
  - `alpha_i`: Lower bound for CVaR integration.
  - `alpha`: Upper bound for CVaR integration.
  - `a_sim`: Number of integration points.

# Validation

  - `0 < alpha_i < alpha < 1`.
  - `a_sim > 0`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_wcvar`](@ref)
  - [`VecNum`](@ref)
"""
function owa_tg(T::Integer; alpha_i::Number = 1e-4, alpha::Number = 0.05,
                a_sim::Integer = 100)
    @argcheck(zero(alpha) < alpha_i < alpha < one(alpha),
              DomainError("0 < alpha_i < alpha < 1 must hold. Got\nalpha_i => $alpha_i\nalpha => $alpha"))
    @argcheck(zero(a_sim) < a_sim, DomainError)
    alphas = range(alpha_i, alpha; length = a_sim)
    n = length(alphas)
    w = Vector{typeof(alpha)}(undef, n)
    w[1] = alphas[2] * alphas[1] / alphas[n]^2
    for i in 2:(n - 1)
        w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
    end
    w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]
    return owa_wcvar(T, alphas, w)
end
"""
    owa_wr(T::Integer)

Compute the Ordered Weights Array (OWA) weights for the worst realisation risk measure.

This function returns a vector of OWA weights that select the minimum (worst) value among `T` observations.

# Arguments

  - `T`: Number of observations.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_rg`](@ref)
  - [`VecNum`](@ref)
"""
function owa_wr(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    return w
end
"""
    owa_rg(T::Integer)

Compute the Ordered Weights Array (OWA) weights for the range risk measure.

This function returns a vector of OWA weights corresponding to the range (difference between maximum and minimum) returns among `T` observations.

# Arguments

  - `T`: Number of observations.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_wr`](@ref)
  - [`VecNum`](@ref)
"""
function owa_rg(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    w[T] = 1
    return w
end
"""
    owa_cvarrg(T::Integer; alpha::Number = 0.05, beta::Number = alpha)

Compute the Ordered Weights Array (OWA) weights for the Conditional Value at Risk Range risk measure.

This function returns a vector of OWA weights corresponding to the difference between CVaR at level `alpha` (lower tail) and the reversed CVaR at level `beta` (upper tail).

# Arguments

  - `T`: Number of observations.
  - `alpha`: CVaR confidence level for the lower tail.
  - `beta`: CVaR confidence level for the upper tail.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_rg`](@ref)
  - [`VecNum`](@ref)
"""
function owa_cvarrg(T::Integer; alpha::Number = 0.05, beta::Number = alpha)
    return owa_cvar(T, alpha) - reverse(owa_cvar(T, beta))
end
"""
    owa_wcvarrg(T::Integer, alphas::VecNum, weights_a::VecNum;
                betas::VecNum = alphas,
                weights_b::VecNum = weights_a)

Compute the Ordered Weights Array (OWA) weights for the weighted Conditional Value at Risk Range risk measure.

This function returns a vector of OWA weights corresponding to the difference between a weighted sum of CVaR measures at levels `alphas` with weights `weights_a` and the reversed weighted sum of CVaR measures at levels `betas` with weights `weights_b`.

# Arguments

  - `T`: Number of observations.
  - `alphas`: Vector of lower tail CVaR confidence levels.
  - `weights_a`: Vector of weights for lower tail CVaR.
  - `betas`: Vector of upper tail CVaR confidence levels.
  - `weights_b`: Vector of weights for upper tail CVaR.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_wcvar`](@ref)
  - [`owa_cvarrg`](@ref)
  - [`VecNum`](@ref)
"""
function owa_wcvarrg(T::Integer, alphas::VecNum, weights_a::VecNum, betas::VecNum = alphas,
                     weights_b::VecNum = weights_a)
    w = owa_wcvar(T, alphas, weights_a) - reverse(owa_wcvar(T, betas, weights_b))
    return w
end
"""
    owa_tgrg(T::Integer; alpha_i::Number = 0.0001, alpha::Number = 0.05, a_sim::Integer = 100,
             beta_i::Number = alpha_i, beta::Number = alpha, b_sim::Integer = a_sim)

Compute the Ordered Weights Array (OWA) weights for the tail Gini range risk measure.

This function returns a vector of OWA weights corresponding to the difference between tail Gini measures for the lower and upper tails, each approximated by integrating over a range of CVaR levels.

# Arguments

  - `T`: Number of observations.
  - `alpha_i`: Lower bound for lower tail CVaR integration.
  - `alpha`: Upper bound for lower tail CVaR integration.
  - `a_sim`: Number of integration points for lower tail.
  - `beta_i`: Lower bound for upper tail CVaR integration.
  - `beta`: Upper bound for upper tail CVaR integration.
  - `b_sim`: Number of integration points for upper tail.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_tg`](@ref)
  - [`owa_rg`](@ref)
  - [`VecNum`](@ref)
"""
function owa_tgrg(T::Integer; alpha_i::Number = 0.0001, alpha::Number = 0.05,
                  a_sim::Integer = 100, beta_i::Number = alpha_i, beta::Number = alpha,
                  b_sim::Integer = a_sim)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) -
        reverse(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end
"""
    owa_l_moment(T::Integer; k::Integer = 2)

Compute the linear moment weights for the linear moments convex risk measure (CRM).

This function returns the vector of weights for the OWA linear moment of order `k` for `T` observations. The weights are derived from combinatorial expressions and are used to construct higher-order moment risk measures.

# Arguments

  - `T`: Number of observations.
  - `k`: Moment order.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_l_moment_crm`](@ref)
  - [`VecNum`](@ref)
"""
function owa_l_moment(T::Integer, k::Integer = 2)
    T, k = promote(T, k)
    w = Vector{typeof(inv(T * k))}(undef, T)
    for i in eachindex(w)
        a = zero(k)
        for j in 0:(k - 1)
            a += (-1)^j *
                 binomial(k - 1, j) *
                 binomial(i - 1, k - 1 - j) *
                 binomial(T - i, j)
        end
        a *= 1 / (k * binomial(T, k))
        w[i] = a
    end
    return w
end
"""
    owa_l_moment_crm(T::Integer; k::Integer = 2,
                     method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion())

Compute the ordered weights array (OWA) linear moments convex risk measure (CRM) weights for a given number of observations and moment order.

This function constructs the OWA linear moment CRM weights matrix for order statistics of size `T` and moment orders from 2 up to `k`, and then applies the specified OWA estimation method to produce the final OWA weights.

# Arguments

  - `T`: Number of observations.
  - `k`: Highest moment order to include.
  - `method`: OWA estimator.

# Validation

  - `k >= 2`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`, normalised to sum to one.

# Details

  - Constructs a matrix of OWA moment weights for each moment order from 2 to `k`.
  - Applies the specified OWA estimation method to aggregate the moment weights into a single OWA weight vector.

# Related

  - [`owa_l_moment`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`VecNum`](@ref)
"""
function owa_l_moment_crm(T::Integer; k::Integer = 2,
                          method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion())
    @argcheck(2 <= k, DomainError)
    rg = 2:k
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(rg))
    for i in rg
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] = wi
    end
    return owa_l_moment_crm(method, weights)
end

export MaximumEntropy, MinimumSquaredDistance, MinimumSumSquares,
       NormalisedConstantRelativeRiskAversion, OWAJuMP, owa_gmd, owa_cvar, owa_wcvar,
       owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_wcvarrg, owa_tgrg, owa_l_moment,
       owa_l_moment_crm
