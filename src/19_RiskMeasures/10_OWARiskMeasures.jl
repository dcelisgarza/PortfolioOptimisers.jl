"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Ordered Weights Array (OWA) estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing OWA estimation algorithms should be subtypes of `AbstractOrderedWeightsArrayEstimator`.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
"""
abstract type AbstractOrderedWeightsArrayEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable OWA weight function estimators in `PortfolioOptimisers.jl`.

All concrete subtypes implementing callable OWA weight functions should subtype `AbstractOrderedWeightsArrayFunction`.

# Interfaces

In order to implement a new callable OWA weight function that works seamlessly with the library, subtype `AbstractOrderedWeightsArrayFunction`, ensuring that the structure contains all necessary parameters, and implement the following method:

## Callable interface

  - `(r::ConcreteType)(T::Integer) -> VecNum`: Computes and returns the OWA weight vector for `T` observations.

### Arguments

  - `r`: Callable OWA weight function instance.
  - `T::Integer`: Number of observations.

### Returns

  - `w::VecNum`: OWA weight vector of length `T`.

### Examples

```jldoctest
julia> struct MyOWAFunction <: PortfolioOptimisers.AbstractOrderedWeightsArrayFunction end

julia> function (r::MyOWAFunction)(T::Integer)
           return fill(inv(T), T)
       end

julia> MyOWAFunction()(4)
4-element Vector{Float64}:
 0.25
 0.25
 0.25
 0.25
```

## Related

  - [`LinearMoment`](@ref)
  - [`OWA_Func_VecNum`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
"""
abstract type AbstractOrderedWeightsArrayFunction <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Ordered Weights Array (OWA) algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific OWA algorithms should be subtypes of `AbstractOrderedWeightsArrayAlgorithm`.

# Related

  - [`MaximumEntropy`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
abstract type AbstractOrderedWeightsArrayAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for entropy formulations used in the [`MaximumEntropy`](@ref) OWA algorithm.

# Related

  - [`ExponentialConeEntropy`](@ref)
  - [`RelativeEntropy`](@ref)
  - [`MaximumEntropy`](@ref)
"""
abstract type EntropyFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Entropy formulation for [`MaximumEntropy`](@ref) OWA that uses the exponential cone entropy constraint in JuMP.

# Related

  - [`EntropyFormulation`](@ref)
  - [`RelativeEntropy`](@ref)
  - [`MaximumEntropy`](@ref)
"""
struct ExponentialConeEntropy <: EntropyFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Entropy formulation for [`MaximumEntropy`](@ref) OWA that uses the relative entropy cone constraint in JuMP. This is the default entropy formulation.

# Related

  - [`EntropyFormulation`](@ref)
  - [`ExponentialConeEntropy`](@ref)
  - [`MaximumEntropy`](@ref)
"""
struct RelativeEntropy <: EntropyFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Maximum Entropy algorithm for Ordered Weights Array (OWA) estimation.

The Maximum Entropy algorithm seeks the OWA weights that maximize entropy, resulting in the most "uninformative" or uniform distribution of weights subject to the imposed constraints.

```math
\\begin{align}
\\underset{\\boldsymbol{\\phi},\\, \\boldsymbol{\\theta}}{\\max} -\\sum\\limits_{t=1}^{T}\\psi_{t} \\log\\left(\\psi_{t}\\right)\\\\
\\text{s.t.} \\quad & \\left(\\psi_{i},\\,\\theta_{i}\\right) \\in \\mathcal{K}_{noc} \\quad \\forall i = 1, \\ldots,\\, T \\\\
 & \\sum\\limits_{k=1}^K \\phi_{k} = 1 \\\\
 & \\boldsymbol{\\phi} \\leq \\phi_{\\text{max}} \\\\
 & \\boldsymbol{\\phi} \\geq 0 \\\\
 & \\phi_{k+1} \\leq \\phi_{k} \\quad \\forall k = 1, \\ldots,\\, K-1 \\\\
 & \\boldsymbol{w}_{k} = \\dfrac{1}{k} \\binom{T}{k}^{-1} \\sum\\limits_{i=0}^{k-1} (-1)^{i} \\binom{k-1}{i} \\binom{t-1}{k-1-i} \\binom{T-t}{i}  \\quad \\forall t = 1,\\ldots,\\, T \\\\
 & \\mathbf{w} = \\left[(-1)^k\\boldsymbol{w}_{k} \\quad \\forall k = 2,\\ldots,\\, K\\right] \\\\
 & \\boldsymbol{\\theta} = \\mathbf{w} \\boldsymbol{\\phi} \\\\
 & \\theta_{t+1} \\geq \\theta_{t} \\quad \\forall t = 1, \\ldots,\\, T-1 \\\\
\\end{align}
```

Where:

  - ``\\mathcal{K}_{\\text{noc}} \\coloneqq \\left\\{\\left(t,\\,x\\right) \\in \\mathbb{R}^n : t \\geq \\lVert x \\rVert_{1} = \\sum\\limits_{i} \\lvert x_{i} \\rvert\\right\\}``: Is the norm one cone, which enforces each entry of ``\\boldsymbol{\\psi}`` is the absolute value of each entry of ``\\boldsymbol{\\theta}``.
  - ``\\phi_{k}``: Is the risk aversion coefficient for the `k`-th order moment.
  - ``\\phi_{\\text{max}}``: Is the maximum risk aversion coefficient.
  - ``T``: Is the total number of observations.
  - ``\\boldsymbol{w}_{k}``: Is the `T × 1` OWA weights vector for the `k`-th order moment.
  - ``\\mathbf{w}``: Is the `T × K` matrix of OWA weights for all order moments where each column `k` corresponds to weights of the `k`-th order moment, each row corresponds to the weights for the `t`-th observation.
  - ``\\boldsymbol{\\theta}``: Is the final `T × 1` OWA weights vector after enforcing non-decreasing monotonicity and incorporating the user-defined risk aversion.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumEntropy(;
        alg::EntropyFormulation = RelativeEntropy(),
    ) -> MaximumEntropy

Keywords correspond to the struct's fields.

# Details

The `MaximumEntropy` algorithm can be configured to use different entropy formulations via the `alg` field. The default is `RelativeEntropy`, but other formulations such as `ExponentialConeEntropy` can also be used.

# Examples

```jldoctest
julia> MaximumEntropy()
MaximumEntropy
  alg ┴ RelativeEntropy()
```

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
@concrete struct MaximumEntropy <: AbstractOrderedWeightsArrayAlgorithm
    """
    $(field_dict[:alg])
    """
    alg
    function MaximumEntropy(alg::EntropyFormulation)
        return new{typeof(alg)}(alg)
    end
end
function MaximumEntropy(; alg::EntropyFormulation = RelativeEntropy())::MaximumEntropy
    return MaximumEntropy(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for squared OWA weight optimisation algorithms.

Subtypes find OWA weights by minimising a squared-distance or squared-sum objective subject to OWA constraints, and are parameterised by the optimisation algorithm type `T`.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)
"""
abstract type SquaredOrderedWeightsArrayAlgorithm{T} <: AbstractOrderedWeightsArrayAlgorithm end
"""
    const UnionAllSOCRiskExpr = Union{<:SquaredSOCRiskExpr, <:RSOCRiskExpr, <:SOCRiskExpr}

Union of all second-order cone risk expression formulation types.

# Related

  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`UnionSOCRiskExpr`](@ref)
  - [`UnionRSOCSOCRiskExpr`](@ref)
"""
const UnionAllSOCRiskExpr = Union{<:SquaredSOCRiskExpr, <:RSOCRiskExpr, <:SOCRiskExpr}
"""
    const UnionSOCRiskExpr = Union{<:SquaredSOCRiskExpr, <:SOCRiskExpr}

Union of squared and plain SOC risk expression formulation types (excludes RSOC).

# Related

  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`UnionAllSOCRiskExpr`](@ref)
"""
const UnionSOCRiskExpr = Union{<:SquaredSOCRiskExpr, <:SOCRiskExpr}
"""
    const UnionRSOCSOCRiskExpr = Union{<:RSOCRiskExpr, <:SOCRiskExpr}

Union of RSOC and plain SOC risk expression formulation types (excludes squared SOC).

# Related

  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`UnionAllSOCRiskExpr`](@ref)
"""
const UnionRSOCSOCRiskExpr = Union{<:RSOCRiskExpr, <:SOCRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Represents the Minimum Squared Distance algorithm for Ordered Weights Array (OWA) estimation.

The Minimum Squared Distance algorithm finds OWA weights that minimize the squared distance between adjacent entries in the array, subject to the OWA constraints. This approach promotes smoothness in the resulting weights.

```math
\\begin{align}
\\underset{\\boldsymbol{\\theta}}{\\min} \\sum\\limits_{t=1}^{T-1}\\left(\\boldsymbol{\\theta}_{t+1} - \\boldsymbol{\\theta}_{t} \\right)^2 \\\\
\\text{s.t.} \\quad & \\sum\\limits_{k=1}^K \\phi_{k} = 1 \\\\
 & \\boldsymbol{\\phi} \\leq \\phi_{\\text{max}} \\\\
 & \\boldsymbol{\\phi} \\geq 0 \\\\
 & \\phi_{k+1} \\leq \\phi_{k} \\quad \\forall k = 1, \\ldots,\\, K-1 \\\\
 & \\boldsymbol{w}_{k} = \\dfrac{1}{k} \\binom{T}{k}^{-1} \\sum\\limits_{i=0}^{k-1} (-1)^{i} \\binom{k-1}{i} \\binom{t-1}{k-1-i} \\binom{T-t}{i}  \\quad \\forall t = 1,\\ldots,\\, T \\\\
 & \\mathbf{w} = \\left[(-1)^k\\boldsymbol{w}_{k} \\quad \\forall k = 2,\\ldots,\\, K\\right] \\\\
 & \\boldsymbol{\\theta} = \\mathbf{w} \\boldsymbol{\\phi} \\\\
 & \\theta_{t+1} \\geq \\theta_{t} \\quad \\forall t = 1, \\ldots,\\, T-1 \\\\
\\end{align}
```

Where:

  - ``\\phi_{k}``: Is the risk aversion coefficient for the `k`-th order moment.
  - ``\\phi_{\\text{max}}``: Is the maximum risk aversion coefficient.
  - ``T``: Is the total number of observations.
  - ``\\boldsymbol{w}_{k}``: Is the `T × 1` OWA weights vector for the `k`-th order moment.
  - ``\\mathbf{w}``: Is the `T × K` matrix of OWA weights for all order moments where each column `k` corresponds to weights of the `k`-th order moment, each row corresponds to the weights for the `t`-th observation.
  - ``\\boldsymbol{\\theta}``: Is the final `T × 1` OWA weights vector after enforcing non-decreasing monotonicity and incorporating the user-defined risk aversion.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MinimumSquaredDistance(;
        alg::UnionAllSOCRiskExpr = SOCRiskExpr(),
    ) -> MinimumSquaredDistance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MinimumSquaredDistance()
MinimumSquaredDistance
  alg ┴ SOCRiskExpr()
```

# Details

The `MinimumSquaredDistance` algorithm can be configured to use different second-order cone risk expressions via the `alg` field. The default is `SOCRiskExpr`, but other formulations such as `SquaredSOCRiskExpr` or `RSOCRiskExpr` can also be used.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
struct MinimumSquaredDistance{__T_alg} <: SquaredOrderedWeightsArrayAlgorithm{__T_alg}
    """
    $(field_dict[:alg])
    """
    alg::__T_alg
    function MinimumSquaredDistance(alg::UnionAllSOCRiskExpr)
        return new{typeof(alg)}(alg)
    end
end
function MinimumSquaredDistance(;
                                alg::UnionAllSOCRiskExpr = SOCRiskExpr())::MinimumSquaredDistance
    return MinimumSquaredDistance(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Minimum Sum of Squares algorithm for Ordered Weights Array (OWA) estimation.

The Minimum Sum of Squares algorithm minimizes the sum of squared OWA weights, subject to the OWA constraints. This promotes sparsity or concentration in the resulting weights. This can be used to emphasize extreme order statistics in OWA-based risk measures.

```math
\\begin{align}
\\underset{\\boldsymbol{\\theta}}{\\min} \\sum\\limits_{t=1}^{T} \\boldsymbol{\\theta}_{t}^2 \\\\
\\text{s.t.} \\quad & \\sum\\limits_{k=1}^K \\phi_{k} = 1 \\\\
 & \\boldsymbol{\\phi} \\leq \\phi_{\\text{max}} \\\\
 & \\boldsymbol{\\phi} \\geq 0 \\\\
 & \\phi_{k+1} \\leq \\phi_{k} \\quad \\forall k = 1, \\ldots,\\, K-1 \\\\
 & \\boldsymbol{w}_{k} = \\dfrac{1}{k} \\binom{T}{k}^{-1} \\sum\\limits_{i=0}^{k-1} (-1)^{i} \\binom{k-1}{i} \\binom{t-1}{k-1-i} \\binom{T-t}{i}  \\quad \\forall t = 1,\\ldots,\\, T \\\\
 & \\mathbf{w} = \\left[(-1)^k\\boldsymbol{w}_{k} \\quad \\forall k = 2,\\ldots,\\, K\\right] \\\\
 & \\boldsymbol{\\theta} = \\mathbf{w} \\boldsymbol{\\phi} \\\\
 & \\theta_{t+1} \\geq \\theta_{t} \\quad \\forall t = 1, \\ldots,\\, T-1 \\\\
\\end{align}
```

Where:

  - ``\\phi_{k}``: Is the risk aversion coefficient for the `k`-th order moment.
  - ``\\phi_{\\text{max}}``: Is the maximum risk aversion coefficient.
  - ``T``: Is the total number of observations.
  - ``\\boldsymbol{w}_{k}``: Is the `T × 1` OWA weights vector for the `k`-th order moment.
  - ``\\mathbf{w}``: Is the `T × K` matrix of OWA weights for all order moments where each column `k` corresponds to weights of the `k`-th order moment, each row corresponds to the weights for the `t`-th observation.
  - ``\\boldsymbol{\\theta}``: Is the final `T × 1` OWA weights vector after enforcing non-decreasing monotonicity and incorporating the user-defined risk aversion.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MinimumSumSquares(;
        alg::UnionAllSOCRiskExpr = SOCRiskExpr(),
    ) -> MinimumSumSquares

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MinimumSumSquares()
MinimumSumSquares
  alg ┴ SOCRiskExpr()
```

# Details

The `MinimumSumSquares` algorithm can be configured to use different second-order cone risk expressions via the `alg` field. The default is `SOCRiskExpr`, but other formulations such as `SquaredSOCRiskExpr` or `RSOCRiskExpr` can also be used.

# Related

  - [`SquaredOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
struct MinimumSumSquares{__T_alg} <: SquaredOrderedWeightsArrayAlgorithm{__T_alg}
    """
    $(field_dict[:alg])
    """
    alg::__T_alg
    function MinimumSumSquares(alg::UnionAllSOCRiskExpr)
        return new{typeof(alg)}(alg)
    end
end
function MinimumSumSquares(; alg::UnionAllSOCRiskExpr = SOCRiskExpr())::MinimumSumSquares
    return MinimumSumSquares(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator type for normalised constant relative risk aversion (CRRA) OWA weights.

This struct represents an estimator for Ordered Weights Array (OWA) weights based on a normalised constant relative risk aversion parameter `g`. The CRRA approach generates OWA weights that interpolate between risk-neutral and risk-averse profiles, controlled by the parameter `g`.

```math
\\begin{align}
\\phi_{1} &\\coloneqq 1 \\\\
\\boldsymbol{\\phi} &= \\phi_{k-1} \\dfrac{\\gamma + k - 2}{k!} (k-1)! \\quad \\forall k = 2,\\ldots,\\, K \\\\
\\sum\\limits_{k=2}^K \\phi_{k} &= 1 \\\\
\\boldsymbol{w}_{k} &= \\dfrac{1}{k} \\binom{T}{k}^{-1} \\sum\\limits_{i=0}^{k-1} (-1)^{i} \\binom{k-1}{i} \\binom{t-1}{k-1-i} \\binom{T-t}{i}  \\quad \\forall t = 1,\\ldots,\\, T \\\\
\\mathbf{w} &= \\left[(-1)^k\\boldsymbol{w}_{k} \\quad \\forall k = 2,\\ldots,\\, K\\right] \\\\
\\boldsymbol{\\vartheta} &= \\mathbf{w} \\boldsymbol{\\phi} \\\\
\\theta_{i} &= \\max \\left(\\vartheta_{j} \\quad \\forall j = 1, \\ldots,\\, i\\right) \\quad \\forall i = 1,\\ldots,\\, T
\\end{align}
```

Where:

  - ``\\phi_{k}``: Is the risk aversion coefficient for the `k`-th order moment.
  - ``\\gamma``: Is the risk aversion parameter `g`.
  - ``T``: Is the total number of observations.
  - ``\\boldsymbol{w}_{k}``: Is the `T × 1` OWA weights vector for the `k`-th order moment.
  - ``\\mathbf{w}``: Is the `T × K` matrix of OWA weights for all order moments where each column `k` corresponds to weights of the `k`-th order moment, each row corresponds to the weights for the `t`-th observation.
  - ``\\boldsymbol{\\vartheta}``: Is the intermediate `T × 1` OWA weights vector incorporating the user-defined risk aversion before enforcing non-decreasing monotonicity.
  - ``\\boldsymbol{\\theta}``: Is the final `T × 1` OWA weights vector incorporating the user-defined risk aversion after enforcing non-decreasing monotonicity.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormalisedConstantRelativeRiskAversion(;
        g::Number = 0.5,
    ) -> NormalisedConstantRelativeRiskAversion

Keywords correspond to the struct's fields.

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

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
@concrete struct NormalisedConstantRelativeRiskAversion <:
                 AbstractOrderedWeightsArrayEstimator
    """
    $(field_dict[:g_rm])
    """
    g
    function NormalisedConstantRelativeRiskAversion(g::Number)
        @argcheck(zero(g) < g < one(g), DomainError("0 < g < 1 must hold. Got\ng => $g"))
        return new{typeof(g)}(g)
    end
end
function NormalisedConstantRelativeRiskAversion(;
                                                g::Number = 0.5)::NormalisedConstantRelativeRiskAversion
    return NormalisedConstantRelativeRiskAversion(g)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator type for OWA weights using JuMP-based optimization.

`OWAJuMP` encapsulates all configuration required to estimate OWA weights via mathematical programming using JuMP. It supports multiple algorithms and solver backends, and allows fine control over constraints and scaling.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OWAJuMP(;
        slv::Slv_VecSlv,
        max_phi::Number = 0.5,
        sc::Number = 1.0,
        so::Number = 1.0,
        alg::AbstractOrderedWeightsArrayAlgorithm = MaximumEntropy(),
    ) -> OWAJuMP

Keyword arguments correspond to the struct's fields.

## Validation

  - `!isempty(slv)`.
  - `0 < max_phi < 1`.
  - `isfinite(sc)` and `sc > 0`.
  - `isfinite(so)` and `so > 0`.

# Examples

```jldoctest
julia> OWAJuMP(; slv = Solver(; solver = nothing))
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
      alg ┼ MaximumEntropy
          │   alg ┴ RelativeEntropy()
```

# Related

  - [`AbstractOrderedWeightsArrayEstimator`](@ref)
  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`Solver`](@ref)
  - [`Slv_VecSlv`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
@concrete struct OWAJuMP <: AbstractOrderedWeightsArrayEstimator
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:max_phi])
    """
    max_phi
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:alg])
    """
    alg
    function OWAJuMP(slv::Slv_VecSlv, max_phi::Number, sc::Number, so::Number,
                     alg::AbstractOrderedWeightsArrayAlgorithm)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
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
function OWAJuMP(; slv::Slv_VecSlv, max_phi::Number = 0.5, sc::Number = 1.0,
                 so::Number = 1.0,
                 alg::AbstractOrderedWeightsArrayAlgorithm = MaximumEntropy())::OWAJuMP
    return OWAJuMP(slv, max_phi, sc, so, alg)
end
"""
    ncrra_weights(weights::MatNum, g::Number = 0.5)

Compute normalised constant relative risk aversion (CRRA) Ordered Weights Array (OWA) weights.

This function generates OWA weights using a normalised CRRA scheme, parameterised by `g`. The CRRA approach interpolates between risk-neutral and risk-averse weighting profiles, controlled by the risk aversion parameter `g`. The resulting weights are normalised to sum to one and are suitable for use in OWA-based risk measures.

# Arguments

  - `weights`: Matrix of weights (typically order statistics or moment weights).
  - `g`: Risk aversion parameter.

# Validation

  - `0 < g < 1`.

# Returns

  - `w::VecNum`: Vector of OWA weights, normalised to sum to one.

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

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
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
        w[i] = maximum(view(a, 1:i))
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

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
function owa_model_setup(method::OWAJuMP, weights::MatNum)
    T, N = size(weights)
    model = JuMP.Model()
    max_phi = method.max_phi
    sc = method.sc
    JuMP.@variables(model, begin
                        theta[1:T]
                        phi[1:N]
                    end)
    JuMP.@constraints(model, begin
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

  - `w::VecNum`: Vector of OWA weights, normalised to sum to one.

# Details

  - If the solver succeeds, the solution is extracted from the `phi` variable and normalised.
  - If the solver fails, a warning is issued and the fallback `ncrra_weights(weights, 0.5)` is returned.

# Related

  - [`OWAJuMP`](@ref)
  - [`MatNum`](@ref)
  - [`owa_model_setup`](@ref)
  - [`ncrra_weights`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
function owa_model_solve(model::JuMP.Model, method::OWAJuMP, weights::MatNum)
    slv = method.slv
    return if optimise_JuMP_model!(model, slv).success
        w = JuMP.value.(model[:theta])
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

  - `w::VecNum`: Vector of OWA weights, normalised to sum to one.

# Related

  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`MaximumEntropy`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)
  - [`MatNum`](@ref)
  - [`ncrra_weights`](@ref)

# References

  - [owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).
"""
function owa_l_moment_crm(method::NormalisedConstantRelativeRiskAversion, weights::MatNum)
    return ncrra_weights(weights, method.g)
end
"""
    owa_l_moment_crm_entropy(method, ...)

Compute OWA L-moment CRM weights using entropy maximisation.

Internal helper for the OWA (Ordered Weighted Average) L-moment constant relative risk measure, computing weights that maximise entropy subject to moment constraints.

# Arguments

  - `method`: OWA JuMP method configuration.
  - Additional parameters.

# Returns

  - OWA weight vector.

# Related

  - [`owa_l_moment_crm_sumsq_obj`](@ref)
"""
function owa_l_moment_crm_entropy(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                                  <:MaximumEntropy{<:RelativeEntropy}},
                                  model::JuMP.Model)
    sc = method.sc
    so = method.so
    x = model[:x]
    T = length(x)
    ovec = range(sc, sc; length = T)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; ovec; sc * x] in JuMP.MOI.RelativeEntropyCone(2 * T + 1))
    JuMP.@objective(model, Max, -so * t)
    return nothing
end
function owa_l_moment_crm_entropy(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                                  <:MaximumEntropy{<:ExponentialConeEntropy}},
                                  model::JuMP.Model)
    sc = method.sc
    so = method.so
    x = model[:x]
    T = length(x)
    JuMP.@variable(model, t[1:T])
    JuMP.@constraint(model, [i = 1:T],
                     [sc * t[i], sc * x[i], 1] in JuMP.MOI.ExponentialCone())
    JuMP.@objective(model, Max, so * sum(t))
    return nothing
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MaximumEntropy},
                          weights::MatNum)
    T = size(weights, 1)
    sc = method.sc
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    JuMP.@variable(model, x[1:T])
    JuMP.@constraints(model,
                      begin
                          sc * (sum(x) - 1) == 0
                          [i = 1:T], [sc * x[i]; sc * theta[i]] in JuMP.MOI.NormOneCone(2)
                      end)
    owa_l_moment_crm_entropy(method, model)
    return owa_model_solve(model, method, weights)
end
"""
    owa_l_moment_crm_sumsq_obj(method, ...)

Compute OWA L-moment CRM weights by minimising sum of squared deviations.

Internal helper for the OWA L-moment constant relative risk measure using a sum-of-squares objective.

# Arguments

  - `method`: OWA JuMP method configuration.
  - Additional parameters.

# Returns

  - OWA weight vector.

# Related

  - [`owa_l_moment_crm_entropy`](@ref)
"""
function owa_l_moment_crm_sumsq_obj(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                                    <:SquaredOrderedWeightsArrayAlgorithm{<:UnionRSOCSOCRiskExpr}},
                                    model::JuMP.Model)
    so = method.so
    t = model[:t]
    JuMP.@objective(model, Min, so * t)
end
function owa_l_moment_crm_sumsq_obj(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                                    <:SquaredOrderedWeightsArrayAlgorithm{<:SquaredSOCRiskExpr}},
                                    model::JuMP.Model)
    so = method.so
    t = model[:t]
    JuMP.@objective(model, Min, so * t^2)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                          <:MinimumSquaredDistance{<:UnionSOCRiskExpr}},
                          weights::MatNum)
    sc = method.sc
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; sc * (theta[2:end] - theta[1:(end - 1)])] in
                     JuMP.SecondOrderCone())
    owa_l_moment_crm_sumsq_obj(method, model)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                          <:MinimumSquaredDistance{<:RSOCRiskExpr}},
                          weights::MatNum)
    sc = method.sc
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; 0.5; sc * (theta[2:end] - theta[1:(end - 1)])] in
                     JuMP.RotatedSecondOrderCone())
    owa_l_moment_crm_sumsq_obj(method, model)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                          <:MinimumSumSquares{<:UnionSOCRiskExpr}},
                          weights::MatNum)
    sc = method.sc
    so = method.so
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * theta] in JuMP.SecondOrderCone())
    owa_l_moment_crm_sumsq_obj(method, model)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                          <:MinimumSumSquares{<:RSOCRiskExpr}},
                          weights::MatNum)
    sc = method.sc
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; 0.5; sc * theta] in JuMP.RotatedSecondOrderCone())
    owa_l_moment_crm_sumsq_obj(method, model)
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
$(DocStringExtensions.TYPEDEF)

Callable OWA weight estimator for the Conditional Value at Risk (CVaR) risk measure.

When called as `r(T)`, returns the OWA weight vector for CVaR at confidence level `alpha` for `T` observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayConditionalValueatRisk(;
        alpha::Number = 0.05
    ) -> OrderedWeightsArrayConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha])

# Examples

```jldoctest
julia> OrderedWeightsArrayConditionalValueatRisk()
OrderedWeightsArrayConditionalValueatRisk
  alpha ┴ Float64: 0.05
```

## Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_cvar`](@ref)
  - [`OWA_Func_VecNum`](@ref)
"""
@concrete struct OrderedWeightsArrayConditionalValueatRisk <:
                 AbstractOrderedWeightsArrayFunction
    """
    $(field_dict[:alpha])
    """
    alpha
    function OrderedWeightsArrayConditionalValueatRisk(alpha::Number)
        @argcheck(0 < alpha < 1,
                  DomainError("0 < alpha < 1 must hold. Got\nalpha => $alpha"))
        return new{typeof(alpha)}(alpha)
    end
end
function OrderedWeightsArrayConditionalValueatRisk(; alpha::Number = 0.05)
    return OrderedWeightsArrayConditionalValueatRisk(alpha)
end
function (r::OrderedWeightsArrayConditionalValueatRisk)(T::Integer)
    return owa_cvar(T, r.alpha)
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
$(DocStringExtensions.TYPEDEF)

Callable OWA weight estimator for the tail Gini risk measure.

When called as `r(T)`, returns the OWA weight vector approximating the tail Gini measure by integrating over CVaR levels from `alpha_i` to `alpha` using `a_sim` points.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayTailGini(;
        alpha_i::Number = 1e-4,
        alpha::Number = 0.05,
        a_sim::Integer = 100
    ) -> OrderedWeightsArrayTailGini

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_i_alpha])
  - $(val_dict[:a_sim_pos])

# Examples

```jldoctest
julia> OrderedWeightsArrayTailGini()
OrderedWeightsArrayTailGini
  alpha_i ┼ Float64: 0.0001
    alpha ┼ Float64: 0.05
    a_sim ┴ Int64: 100
```

## Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_tg`](@ref)
  - [`OWA_Func_VecNum`](@ref)
"""
@concrete struct OrderedWeightsArrayTailGini <: AbstractOrderedWeightsArrayFunction
    """
    $(field_dict[:alpha_i])
    """
    alpha_i
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:a_sim])
    """
    a_sim
    function OrderedWeightsArrayTailGini(alpha_i::Number, alpha::Number, a_sim::Integer)
        @argcheck(0 < alpha_i < alpha < 1,
                  DomainError("0 < alpha_i < alpha < 1 must hold. Got\nalpha_i => $alpha_i\nalpha => $alpha"))
        @argcheck(0 < a_sim, DomainError("a_sim must be positive. Got\n a_sim => $a_sim"))
        return new{typeof(alpha_i), typeof(alpha), typeof(a_sim)}(alpha_i, alpha, a_sim)
    end
end
function OrderedWeightsArrayTailGini(; alpha_i::Number = 1e-4, alpha::Number = 0.05,
                                     a_sim::Integer = 100)
    return OrderedWeightsArrayTailGini(alpha_i, alpha, a_sim)
end
function (r::OrderedWeightsArrayTailGini)(T::Integer)
    return owa_tg(T; alpha_i = r.alpha_i, alpha = r.alpha, a_sim = r.a_sim)
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
    return owa_cvar(T, alpha) - reverse!(owa_cvar(T, beta))
end
"""
$(DocStringExtensions.TYPEDEF)

Callable OWA weight estimator for the Conditional Value at Risk Range risk measure.

When called as `r(T)`, returns the OWA weight vector for the CVaR range at lower confidence level `alpha` and upper confidence level `beta` for `T` observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayConditionalValueatRiskRange(;
        alpha::Number = 0.05,
        beta::Number = alpha
    ) -> OrderedWeightsArrayConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha])
  - $(val_dict[:beta])

# Examples

```jldoctest
julia> OrderedWeightsArrayConditionalValueatRiskRange()
OrderedWeightsArrayConditionalValueatRiskRange
  alpha ┼ Float64: 0.05
   beta ┴ Float64: 0.05
```

## Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_cvarrg`](@ref)
  - [`OWA_Func_VecNum`](@ref)
"""
@concrete struct OrderedWeightsArrayConditionalValueatRiskRange <:
                 AbstractOrderedWeightsArrayFunction
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:beta])
    """
    beta
    function OrderedWeightsArrayConditionalValueatRiskRange(alpha::Number, beta::Number)
        @argcheck(0 < alpha < 1,
                  DomainError("0 < alpha < 1 must hold. Got\nalpha => $alpha"))
        @argcheck(0 < beta < 1, DomainError("0 < beta < 1 must hold. Got\nbeta => $beta"))
        return new{typeof(alpha), typeof(beta)}(alpha, beta)
    end
end
function OrderedWeightsArrayConditionalValueatRiskRange(; alpha::Number = 0.05,
                                                        beta::Number = alpha)
    return OrderedWeightsArrayConditionalValueatRiskRange(alpha, beta)
end
function (r::OrderedWeightsArrayConditionalValueatRiskRange)(T::Integer)
    return owa_cvarrg(T; alpha = r.alpha, beta = r.beta)
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
    w = owa_wcvar(T, alphas, weights_a) - reverse!(owa_wcvar(T, betas, weights_b))
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
        reverse!(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end
"""
$(DocStringExtensions.TYPEDEF)

Callable OWA weight estimator for the tail Gini range risk measure.

When called as `r(T)`, returns the OWA weight vector for the difference between the lower and upper tail Gini measures, each approximated by integrating over CVaR levels.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayTailGiniRange(;
        alpha_i::Number = 1e-4,
        alpha::Number = 0.05,
        a_sim::Integer = 100,
        beta_i::Number = alpha_i,
        beta::Number = alpha,
        b_sim::Integer = a_sim
    ) -> OrderedWeightsArrayTailGiniRange

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_i_alpha])
  - $(val_dict[:a_sim_pos])
  - $(val_dict[:beta_i_beta])
  - $(val_dict[:b_sim_pos])

# Examples

```jldoctest
julia> OrderedWeightsArrayTailGiniRange()
OrderedWeightsArrayTailGiniRange
  alpha_i ┼ Float64: 0.0001
    alpha ┼ Float64: 0.05
    a_sim ┼ Int64: 100
   beta_i ┼ Float64: 0.0001
     beta ┼ Float64: 0.05
    b_sim ┴ Int64: 100
```

## Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_tgrg`](@ref)
  - [`OWA_Func_VecNum`](@ref)
"""
@concrete struct OrderedWeightsArrayTailGiniRange <: AbstractOrderedWeightsArrayFunction
    """
    $(field_dict[:alpha_i])
    """
    alpha_i
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:a_sim])
    """
    a_sim
    """
    $(field_dict[:beta_i])
    """
    beta_i
    """
    $(field_dict[:beta])
    """
    beta
    """
    $(field_dict[:b_sim])
    """
    b_sim
    function OrderedWeightsArrayTailGiniRange(alpha_i::Number, alpha::Number,
                                              a_sim::Integer, beta_i::Number, beta::Number,
                                              b_sim::Integer)
        @argcheck(0 < alpha_i < alpha < 1,
                  DomainError("0 < alpha_i < alpha < 1 must hold. Got\nalpha_i => $alpha_i\nalpha => $alpha"))
        @argcheck(0 < a_sim, DomainError("a_sim must be positive. Got\n a_sim => $a_sim"))
        @argcheck(0 < beta_i < beta < 1,
                  DomainError("0 < beta_i < beta < 1 must hold. Got\nbeta_i => $beta_i\nbeta => $beta"))
        @argcheck(0 < b_sim, DomainError("b_sim must be positive. Got\n b_sim => $b_sim"))
        return new{typeof(alpha_i), typeof(alpha), typeof(a_sim), typeof(beta_i),
                   typeof(beta), typeof(b_sim)}(alpha_i, alpha, a_sim, beta_i, beta, b_sim)
    end
end
function OrderedWeightsArrayTailGiniRange(; alpha_i::Number = 1e-4, alpha::Number = 0.05,
                                          a_sim::Integer = 100, beta_i::Number = alpha_i,
                                          beta::Number = alpha, b_sim::Integer = a_sim)
    return OrderedWeightsArrayTailGiniRange(alpha_i, alpha, a_sim, beta_i, beta, b_sim)
end
function (r::OrderedWeightsArrayTailGiniRange)(T::Integer)
    return owa_tgrg(T; alpha_i = r.alpha_i, alpha = r.alpha, a_sim = r.a_sim,
                    beta_i = r.beta_i, beta = r.beta, b_sim = r.b_sim)
end
"""
    owa_l_moment(T::Integer, k::Integer = 2)

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
    owa_l_moment_crm(T::Integer,
                     method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion();
                     k::Integer = 2)

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
function owa_l_moment_crm(T::Integer,
                          method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion();
                          k::Integer = 2)
    @argcheck(2 <= k, DomainError)
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(2:k))
    for i in 2:k
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] = wi
    end
    return owa_l_moment_crm(method, weights)
end
"""
$(DocStringExtensions.TYPEDEF)

Callable estimator that generates OWA linear moment convex risk measure (CRM) weights for a given number of observations.

When called as `lm(T)`, returns the OWA weight vector produced by [`owa_l_moment_crm`](@ref) using the configured `method` and moment order `k`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearMoment(;
        method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion(),
        k::Integer = 2
    ) -> LinearMoment

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:lm_k])

# Examples

```jldoctest
julia> LinearMoment()
LinearMoment
  method ┼ NormalisedConstantRelativeRiskAversion
         │   g ┴ Float64: 0.5
       k ┴ Int64: 2
```

## Related

  - [`AbstractOrderedWeightsArrayEstimator`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`OWA_Func_VecNum`](@ref)
"""
@concrete struct LinearMoment <: AbstractOrderedWeightsArrayFunction
    """
    $(field_dict[:owa_method])
    """
    method
    """
    $(field_dict[:lm_k])
    """
    k
    function LinearMoment(method::AbstractOrderedWeightsArrayEstimator, k::Integer)
        @argcheck(2 <= k, DomainError)
        return new{typeof(method), typeof(k)}(method, k)
    end
end
function LinearMoment(;
                      method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion(),
                      k::Integer = 2)
    return LinearMoment(method, k)
end
function (r::LinearMoment)(T::Integer)
    return owa_l_moment_crm(T, r.method; k = r.k)
end
"""
    const OWA_Func_VecNum = Union{<:Func_VecNum, <:AbstractOrderedWeightsArrayFunction}

Union type for OWA weight specifications: a function, a numeric vector, or an [`AbstractOrderedWeightsArrayFunction`](@ref) callable.

# Related

  - [`Func_VecNum`](@ref)
  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
"""
const OWA_Func_VecNum = Union{<:Func_VecNum, <:AbstractOrderedWeightsArrayFunction}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for ordered weights array (OWA) formulation types.

Determines whether OWA weights are computed exactly or approximately.

# Related

  - [`ExactOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
"""
abstract type OrderedWeightsArrayFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

OWA formulation that computes exact OWA weights by solving a linear programme.

# Related

  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
"""
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
"""
$(DocStringExtensions.TYPEDEF)

OWA formulation that approximates OWA weights using a set of p-norm parameters.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ApproxOrderedWeightsArray(;
        p::VecNum = Float64[2, 3, 4, 10, 50]
    ) -> ApproxOrderedWeightsArray

# Related

  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
"""
@concrete struct ApproxOrderedWeightsArray <: OrderedWeightsArrayFormulation
    """
    $(field_dict[:p_rm])
    """
    p
    function ApproxOrderedWeightsArray(p::VecNum)
        @argcheck(!isempty(p), IsEmptyError("p cannot be empty"))
        @argcheck(all(x -> x > one(x), p), DomainError(p, "all elements of p must be > 1"))
        return new{typeof(p)}(p)
    end
end
function ApproxOrderedWeightsArray(; p::VecNum = Float64[2, 3, 4, 10, 50])
    return ApproxOrderedWeightsArray(p)
end
"""
$(DocStringExtensions.TYPEDEF)

Ordered Weights Array (OWA) risk measure.

Computes portfolio risk as a linear combination of sorted portfolio returns using OWA weights. The OWA weights can be provided directly or computed from an OWA algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArray(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::OWA_Func_VecNum = owa_gmd,
        alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()
    ) -> OrderedWeightsArray

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
@concrete struct OrderedWeightsArray <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:owa_w])
    """
    w
    """
    $(field_dict[:alg])
    """
    alg
    function OrderedWeightsArray(settings::RiskMeasureSettings, w::OWA_Func_VecNum,
                                 alg::OrderedWeightsArrayFormulation)
        if isa(w, VecNum)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        end
        return new{typeof(settings), typeof(w), typeof(alg)}(settings, w, alg)
    end
end
function OrderedWeightsArray(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::OWA_Func_VecNum = owa_gmd,
                             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(settings, w, alg)
end
function (r::OrderedWeightsArray)(x::VecNum)
    w = isa(r.w, VecNum) ? r.w : r.w(length(x))
    return LinearAlgebra.dot(w, sort(x))
end
"""
$(DocStringExtensions.TYPEDEF)

Ordered Weights Array Range (OWA Range) risk measure.

Computes portfolio risk as the difference between two OWA linear combinations of sorted portfolio returns, providing a range-based risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w1::OWA_Func_VecNum = owa_gmd,
        w2::OWA_Func_VecNum = owa_gmd,
        alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
        rev::Bool = false
    ) -> OrderedWeightsArrayRange

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
@concrete struct OrderedWeightsArrayRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w1_owa])
    """
    w1
    """
    $(field_dict[:w2_owa])
    """
    w2
    """
    $(field_dict[:alg])
    """
    alg
    function OrderedWeightsArrayRange(settings::RiskMeasureSettings, w1::OWA_Func_VecNum,
                                      w2::OWA_Func_VecNum,
                                      alg::OrderedWeightsArrayFormulation, rev::Bool)
        w1_flag = isa(w1, VecNum)
        w2_flag = isa(w2, VecNum)
        if w1_flag
            @argcheck(!isempty(w1), IsEmptyError("w1 cannot be empty"))
        end
        if w2_flag
            @argcheck(!isempty(w2), IsEmptyError("w2 cannot be empty"))
            if !rev
                if w1 === w2
                    w2 = reverse(w2)
                else
                    reverse!(w2)
                end
            end
        else
            if !rev
                w2 = reverse! ∘ w2
            end
        end
        if w1_flag && w2_flag
            @argcheck(length(w1) == length(w2),
                      DimensionMismatch("w1 ($(length(w1))) must match w2 ($(length(w2)))"))
        end
        return new{typeof(settings), typeof(w1), typeof(w2), typeof(alg)}(settings, w1, w2,
                                                                          alg)
    end
end
function OrderedWeightsArrayRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  w1::OWA_Func_VecNum = owa_tg,
                                  w2::OWA_Func_VecNum = owa_tg,
                                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                                  rev::Bool = false)
    return OrderedWeightsArrayRange(settings, w1, w2, alg, rev)
end
function (r::OrderedWeightsArrayRange)(x::VecNum)
    w1 = isa(r.w1, VecNum) ? r.w1 : r.w1(length(x))
    w2 = isa(r.w2, VecNum) ? r.w2 : r.w2(length(x))
    w = w1 - w2
    return LinearAlgebra.dot(w, sort(x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::OrderedWeightsArray) = NetReturnsInput()
risk_input_kind(::OrderedWeightsArrayRange) = NetReturnsInput()

export MaximumEntropy, ExponentialConeEntropy, RelativeEntropy, MinimumSquaredDistance,
       MinimumSumSquares, NormalisedConstantRelativeRiskAversion, OWAJuMP, owa_gmd,
       owa_cvar, owa_wcvar, owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_wcvarrg, owa_tgrg,
       owa_l_moment, owa_l_moment_crm, ExactOrderedWeightsArray, ApproxOrderedWeightsArray,
       OrderedWeightsArray, OrderedWeightsArrayRange, LinearMoment,
       OrderedWeightsArrayConditionalValueatRisk, OrderedWeightsArrayTailGini,
       OrderedWeightsArrayConditionalValueatRiskRange, OrderedWeightsArrayTailGiniRange
