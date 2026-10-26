"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators of the risk aversion coefficients of an L-moment OWA weight vector.

[`owa_l_moment_crm`](@ref) dispatches on the subtype. [`NormalisedConstantRelativeRiskAversion`](@ref) states the coefficients in closed form, and [`OWAJuMP`](@ref) solves a programme for them.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`owa_l_moment_crm`](@ref)

# References

  - $(ref_dict[:owa2])
"""
abstract type AbstractOrderedWeightsArrayEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the callable builders of an OWA weight vector.

A builder holds the parameters of one weight vector, such as a CVaR level, and computes the vector for a sample size when it is called. [`OrderedWeightsArray`](@ref) and [`OrderedWeightsArrayRange`](@ref) call it at the number of observations of the sample, so one builder serves a sample of any length.

# Interfaces

To implement a new weight builder, subtype `AbstractOrderedWeightsArrayFunction`. Put every parameter of the weight vector in a field, and implement the following method:

## Callable interface

  - `(r::MyOWAFunction)(T::Integer) -> VecNum`: Computes the OWA weight vector for `T` observations. A nondecreasing vector gives a convex risk measure, and [`SortingNetworkOrderedWeightsArray`](@ref) refuses any other vector.

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

# Related

  - [`LinearMoment`](@ref)
  - [`OWA_Func_VecNum`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)

# References

  - $(ref_dict[:owaog])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.1.5.
"""
abstract type AbstractOrderedWeightsArrayFunction <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the objectives of the programme that [`OWAJuMP`](@ref) solves for an L-moment OWA weight vector.

Every subtype solves over the same feasible set, which [`owa_l_moment_crm`](@ref) states. The subtype selects the objective alone.

# Related

  - [`MaximumEntropy`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)

# References

  - $(ref_dict[:owa2])
"""
abstract type AbstractOrderedWeightsArrayAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the conic encodings of the entropy objective of [`MaximumEntropy`](@ref).

The encoding changes the cones of the model, not the problem. [`owa_l_moment_crm_entropy`](@ref) states the rows of each encoding.

# Related

  - [`ExponentialConeEntropy`](@ref)
  - [`RelativeEntropy`](@ref)
  - [`MaximumEntropy`](@ref)

# References

  - $(ref_dict[:owa2])
"""
abstract type EntropyFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the entropy objective of [`MaximumEntropy`](@ref) with one exponential cone for each risk aversion coefficient.

It selects the branch of [`owa_l_moment_crm_entropy`](@ref) that writes `K - 1` exponential cones.

# Related

  - [`EntropyFormulation`](@ref)
  - [`RelativeEntropy`](@ref)
  - [`MaximumEntropy`](@ref)

# References

  - $(ref_dict[:owa2])
"""
struct ExponentialConeEntropy <: EntropyFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the entropy objective of [`MaximumEntropy`](@ref) with one relative entropy cone.

It selects the branch of [`owa_l_moment_crm_entropy`](@ref) that writes one cone of dimension `2K - 1`. It is the default encoding.

# Related

  - [`EntropyFormulation`](@ref)
  - [`ExponentialConeEntropy`](@ref)
  - [`MaximumEntropy`](@ref)

# References

  - $(ref_dict[:owa2])
"""
struct RelativeEntropy <: EntropyFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Selects the risk aversion coefficients of maximum entropy in the L-moment programme of [`OWAJuMP`](@ref).

The objective is the Shannon entropy of the coefficients ``\\boldsymbol{\\phi}``. The coefficients are a probability vector, so the objective spreads the weight over the L-moments as evenly as the rows let it. The objective is strictly concave, so the programme has one maximiser, and the OWA weights ``\\mathbf{W} \\boldsymbol{\\phi}`` are unique.

`alg` selects the conic encoding of the entropy, not the problem. [`RelativeEntropy`](@ref) writes one [`JuMP.MOI.RelativeEntropyCone`](https://jump.dev/JuMP.jl/stable/moi/reference/standard_form/#MathOptInterface.RelativeEntropyCone) of dimension `2K - 1`, and [`ExponentialConeEntropy`](@ref) writes `K - 1` [`JuMP.MOI.ExponentialCone`](https://jump.dev/JuMP.jl/stable/moi/reference/standard_form/#MathOptInterface.ExponentialCone) rows.

Equation 13 of [owa2](@cite) maximises ``\\sum_{t} \\ln \\psi_{t}`` over a vector ``\\boldsymbol{\\psi}`` subject to ``\\psi_{t} \\geq \\lvert \\theta_{t} \\rvert`` alone, and that objective has no maximum. Its intent, the entropy of the absolute OWA weights, does not select one vector either, and it is not a convex programme in the normalised form. The L-moment weight vector of order ``k`` has entries of absolute value at most ``1/T`` once ``T > k (k - 1) / 2``, measured for every order up to `10` and every ``T`` up to 600. So every feasible ``\\boldsymbol{\\theta}`` has ``\\lvert \\theta_{t} \\rvert \\leq 1/T``, and the uniform ``\\psi_{t} = 1/T`` bounds it. A probability vector ``\\boldsymbol{\\psi}`` that bounds ``\\lvert \\boldsymbol{\\theta} \\rvert`` from above then has the largest entropy ``\\log T`` at every feasible point. The library puts the entropy on the coefficients for this reason.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{\\phi},\\, \\boldsymbol{\\theta}}{\\max} \\quad & -\\sum\\limits_{k=2}^{K} \\phi_{k} \\log \\phi_{k}\\\\
\\text{s.t.} \\quad & \\boldsymbol{\\theta} = \\mathbf{W} \\boldsymbol{\\phi}\\\\
 & \\sum\\limits_{k=2}^{K} \\phi_{k} = 1\\\\
 & 0 \\leq \\phi_{k} \\leq \\phi_{\\max} \\quad \\forall k = 2,\\, \\ldots,\\, K\\\\
 & \\phi_{k+1} \\leq \\phi_{k} \\quad \\forall k = 2,\\, \\ldots,\\, K - 1\\\\
 & \\theta_{t+1} \\geq \\theta_{t} \\quad \\forall t = 1,\\, \\ldots,\\, T - 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:theta_owa])
  - $(math_dict[:W_lmom])
  - $(math_dict[:omega_ik_lmom])
  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:phi_max_owa])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumEntropy(;
        alg::EntropyFormulation = RelativeEntropy(),
    ) -> MaximumEntropy

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MaximumEntropy()
MaximumEntropy
  alg ┴ RelativeEntropy()
```

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`owa_l_moment_crm_entropy`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equation 13.
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

Abstract supertype for the quadratic objectives of the L-moment programme of [`OWAJuMP`](@ref).

Each subtype minimises a sum of squares of the OWA weights. The parameter `T` is the type of the conic encoding of that sum, a second-order cone risk expression. [`owa_l_moment_crm_sumsq_obj`](@ref) dispatches on it.

# Related

  - [`AbstractOrderedWeightsArrayAlgorithm`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)

# References

  - $(ref_dict[:owa2])
"""
abstract type SquaredOrderedWeightsArrayAlgorithm{T} <: AbstractOrderedWeightsArrayAlgorithm end
"""
    const UnionAllSOCRiskExpr = Union{SquaredSOCRiskExpr, RSOCRiskExpr, SOCRiskExpr}

The three conic encodings of the objective of a [`SquaredOrderedWeightsArrayAlgorithm`](@ref).

[`MinimumSquaredDistance`](@ref) and [`MinimumSumSquares`](@ref) accept each of them in `alg`. The three encodings give one minimiser, so the choice changes the model and not the OWA weights.

# Related

  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`UnionSOCRiskExpr`](@ref)
  - [`UnionRSOCSOCRiskExpr`](@ref)
"""
const UnionAllSOCRiskExpr = Union{SquaredSOCRiskExpr, RSOCRiskExpr, SOCRiskExpr}
"""
    const UnionSOCRiskExpr = Union{SquaredSOCRiskExpr, SOCRiskExpr}

The encodings that bound the objective norm with one second-order cone.

The methods of [`owa_l_moment_crm`](@ref) for these encodings write the row `t ≥ ‖v‖₂`. [`RSOCRiskExpr`](@ref) is not a member, because it writes a rotated cone.

# Related

  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`UnionAllSOCRiskExpr`](@ref)
"""
const UnionSOCRiskExpr = Union{SquaredSOCRiskExpr, SOCRiskExpr}
"""
    const UnionRSOCSOCRiskExpr = Union{RSOCRiskExpr, SOCRiskExpr}

The encodings whose objective is linear in the cone variable `t`.

[`owa_l_moment_crm_sumsq_obj`](@ref) minimises `so * t` for these encodings. [`SquaredSOCRiskExpr`](@ref) is not a member, because it minimises `so * t^2`.

# Related

  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`UnionAllSOCRiskExpr`](@ref)
"""
const UnionRSOCSOCRiskExpr = Union{RSOCRiskExpr, SOCRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Selects the smoothest OWA weights in the L-moment programme of [`OWAJuMP`](@ref).

The objective is the sum of the squared differences of adjacent OWA weights, so the weights change little from one order statistic to the next. Use [`MinimumSumSquares`](@ref) to keep the weights small instead.

`alg` selects the conic encoding of the objective, not the problem. [`SOCRiskExpr`](@ref) minimises ``\\lVert \\boldsymbol{\\theta}_{2:T} - \\boldsymbol{\\theta}_{1:T-1} \\rVert_{2}``, and [`SquaredSOCRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref) minimise its square. The square root is strictly increasing, so the three encodings give one minimiser and return the same OWA weights to solver tolerance. In [`SecondMomentFormulation`](@ref), `alg` changes the units of the reported risk instead.

Equation 15 of [owa2](@cite) prints a maximum of this sum. The library minimises it, which is the convex problem that the name of the type states.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{\\phi},\\, \\boldsymbol{\\theta}}{\\min} \\quad & \\sum\\limits_{t=1}^{T-1} \\left(\\theta_{t+1} - \\theta_{t}\\right)^{2}\\\\
\\text{s.t.} \\quad & \\boldsymbol{\\theta} = \\mathbf{W} \\boldsymbol{\\phi}\\\\
 & \\sum\\limits_{k=2}^{K} \\phi_{k} = 1\\\\
 & 0 \\leq \\phi_{k} \\leq \\phi_{\\max} \\quad \\forall k = 2,\\, \\ldots,\\, K\\\\
 & \\phi_{k+1} \\leq \\phi_{k} \\quad \\forall k = 2,\\, \\ldots,\\, K - 1\\\\
 & \\theta_{t+1} \\geq \\theta_{t} \\quad \\forall t = 1,\\, \\ldots,\\, T - 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:theta_owa])
  - $(math_dict[:W_lmom])
  - $(math_dict[:omega_ik_lmom])
  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:phi_max_owa])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

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

# Related

  - [`SquaredOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`owa_l_moment_crm_sumsq_obj`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equation 15.
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

Selects the OWA weights of least sum of squares in the L-moment programme of [`OWAJuMP`](@ref).

The objective is a ridge penalty on the OWA weights, so the solution spreads the weight mass over the order statistics and does not concentrate it. Use [`MinimumSquaredDistance`](@ref) to make the weights change smoothly instead.

`alg` selects the conic encoding of the objective, not the problem. [`SOCRiskExpr`](@ref) minimises ``\\lVert \\boldsymbol{\\theta} \\rVert_{2}``, and [`SquaredSOCRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref) minimise its square. The square root is strictly increasing, so the three encodings give one minimiser and return the same OWA weights to solver tolerance. In [`SecondMomentFormulation`](@ref), `alg` changes the units of the reported risk instead.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{\\phi},\\, \\boldsymbol{\\theta}}{\\min} \\quad & \\sum\\limits_{t=1}^{T} \\theta_{t}^{2}\\\\
\\text{s.t.} \\quad & \\boldsymbol{\\theta} = \\mathbf{W} \\boldsymbol{\\phi}\\\\
 & \\sum\\limits_{k=2}^{K} \\phi_{k} = 1\\\\
 & 0 \\leq \\phi_{k} \\leq \\phi_{\\max} \\quad \\forall k = 2,\\, \\ldots,\\, K\\\\
 & \\phi_{k+1} \\leq \\phi_{k} \\quad \\forall k = 2,\\, \\ldots,\\, K - 1\\\\
 & \\theta_{t+1} \\geq \\theta_{t} \\quad \\forall t = 1,\\, \\ldots,\\, T - 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:theta_owa])
  - $(math_dict[:W_lmom])
  - $(math_dict[:omega_ik_lmom])
  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:phi_max_owa])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

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

# Related

  - [`SquaredOrderedWeightsArrayAlgorithm`](@ref)
  - [`OWAJuMP`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`owa_l_moment_crm_sumsq_obj`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equation 14.
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

Sets the risk aversion coefficients of an L-moment OWA weight vector to the normalised constant relative risk aversion coefficients of `g`.

Equation 12 of [owa2](@cite) states the coefficients. The library divides them by their sum, which that equation suggests. It also replaces each OWA weight by the largest weight up to its position, so the vector is nondecreasing and the risk measure is convex. [`ncrra_weights`](@ref) computes the vector, and [`OWAJuMP`](@ref) falls back to it with `g = 0.5` when its programme fails.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\phi}_{k} &= \\dfrac{1}{k!} \\prod\\limits_{j=0}^{k-2} \\left(\\gamma + j\\right) \\quad \\forall k = 2,\\, \\ldots,\\, K\\\\
\\phi_{k} &= \\dfrac{\\tilde{\\phi}_{k}}{\\sum_{j=2}^{K} \\tilde{\\phi}_{j}}\\\\
\\boldsymbol{\\vartheta} &= \\mathbf{W} \\boldsymbol{\\phi}\\\\
\\eta_{t} &= \\max\\limits_{s \\leq t} \\vartheta_{s} \\quad \\forall t = 1,\\, \\ldots,\\, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:gamma_crra])
  - ``\\tilde{\\phi}_{k}``: Coefficient of the ``k``-th L-moment before the normalisation.
  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:W_lmom])
  - $(math_dict[:omega_ik_lmom])
  - ``\\boldsymbol{\\vartheta}``: Combination of the signed L-moment weights, ``T \\times 1``. It need not be nondecreasing.
  - $(math_dict[:eta_owa])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

The coefficients sum to one, and the OWA weights do not. Each L-moment weight vector of order two or more sums to zero, so ``\\boldsymbol{\\vartheta}`` sums to zero, and the running maximum can only raise that sum.

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
  - [`ncrra_weights`](@ref)
  - [`owa_l_moment_crm`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equation 12.
"""
@concrete struct NormalisedConstantRelativeRiskAversion <:
                 AbstractOrderedWeightsArrayEstimator
    """
    $(field_dict[:g_rm])
    """
    g
    function NormalisedConstantRelativeRiskAversion(g::Number)
        assert_unit_interval(g, :g)
        return new{typeof(g)}(g)
    end
end
function NormalisedConstantRelativeRiskAversion(;
                                                g::Number = 0.5)::NormalisedConstantRelativeRiskAversion
    return NormalisedConstantRelativeRiskAversion(g)
end
"""
$(DocStringExtensions.TYPEDEF)

Solves a programme for the risk aversion coefficients of an L-moment OWA weight vector.

`alg` selects the objective, and every objective solves over the same feasible set. [`owa_l_moment_crm`](@ref) states the programme. `sc` scales the rows and `so` scales the objective, and neither changes the solution.

When the solver fails, [`owa_model_solve`](@ref) warns and returns the weights of `NormalisedConstantRelativeRiskAversion(; g = 0.5)`. The programme is infeasible for some `max_phi` and some orders: the coefficients sum to one and each is at most `max_phi`, and the OWA weights must be nondecreasing. At `k = 2` it is always infeasible, because its one coefficient must equal one. The fallback then returns the weights of the second L-moment, which is the only answer whose coefficients sum to one. At `k = 3` with the default `max_phi = 0.5`, the rows leave the one choice ``\\boldsymbol{\\phi} = (0.5, 0.5)``, and its OWA weights decrease near the largest return, so the programme is infeasible too.

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
  - [`owa_model_setup`](@ref)
  - [`owa_model_solve`](@ref)
  - [`Solver`](@ref)
  - [`Slv_VecSlv`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equations 13 to 15.
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
        assert_unit_interval(max_phi, :max_phi)
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

Compute the OWA weight vector of the normalised constant relative risk aversion coefficients of `g`.

Column `j` of `weights` takes the coefficient of the L-moment of order `j + 1`, so the columns of the signed L-moment weight matrix give the vector that [`NormalisedConstantRelativeRiskAversion`](@ref) states. The coefficients sum to one, and the returned weights do not. The returned vector is the running maximum of a combination of the columns, so its sum is at least the same combination of the column sums.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\phi}_{j} &= \\dfrac{1}{(j + 1)!} \\prod\\limits_{i=0}^{j-1} \\left(\\gamma + i\\right) \\quad \\forall j = 1,\\, \\ldots,\\, M\\\\
\\phi_{j} &= \\dfrac{\\tilde{\\phi}_{j}}{\\sum_{i=1}^{M} \\tilde{\\phi}_{i}}\\\\
\\boldsymbol{\\vartheta} &= \\mathbf{V} \\boldsymbol{\\phi}\\\\
\\eta_{t} &= \\max\\limits_{s \\leq t} \\vartheta_{s} \\quad \\forall t = 1,\\, \\ldots,\\, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:gamma_crra])
  - ``\\mathbf{V}``: The matrix `weights`, ``T \\times M``.
  - ``\\tilde{\\phi}_{j}``, ``\\phi_{j}``: Coefficient of column ``j`` before and after the normalisation.
  - ``\\boldsymbol{\\vartheta}``: Combination of the columns of ``\\mathbf{V}``, ``T \\times 1``.
  - $(math_dict[:eta_owa])

# Algorithm

 1. Set the running product `e = 1`. For each column `i` of `weights`, multiply `e` by `g + i - 1`, and set `phis[i] = e / factorial(i + 1)`.
 2. Divide `phis` by its sum.
 3. Compute the combination `a = weights * phis`.
 4. Set `w[i]` to the largest entry of `a[1:i]`, for each `i`.

# Arguments

  - `weights`: Matrix ``T \\times M`` of signed L-moment weights, one column for each order from two up.
  - `g`: Relative risk aversion.

# Returns

  - `w::VecNum`: Nondecreasing vector of OWA weights of length `size(weights, 1)`.

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

  - $(ref_dict[:owa2]) Section 3.2, Equation 12.
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
        w[i] = max(w[i - 1], a[i])
    end
    return w
end
"""
    owa_model_setup(method::OWAJuMP, weights::MatNum)

Build the feasible set of the L-moment programme of [`OWAJuMP`](@ref) in a new JuMP model.

The model holds the risk aversion coefficients `phi` and the OWA weights `theta`, and the rows that every objective of [`AbstractOrderedWeightsArrayAlgorithm`](@ref) shares. It has no objective and no optimiser. [`owa_l_moment_crm`](@ref) adds the objective, and [`owa_model_solve`](@ref) solves the model.

# JuMP formulation

## Variables

  - `theta`: Created, the OWA weights ``\\boldsymbol{\\theta}``, ``T \\times 1``.
  - `phi`: Created, the risk aversion coefficients ``\\boldsymbol{\\phi}``, ``(K - 1) \\times 1``.

## Constraints

Every row is registered under no name.

  - ``s_c \\boldsymbol{\\phi} \\geq 0``.
  - ``s_c (\\boldsymbol{\\phi} - \\phi_{\\max}) \\leq 0``.
  - ``s_c \\left(\\sum_{k=2}^{K} \\phi_{k} - 1\\right) = 0``.
  - ``s_c (\\boldsymbol{\\theta} - \\mathbf{W} \\boldsymbol{\\phi}) = 0``.
  - ``s_c (\\phi_{k+1} - \\phi_{k}) \\leq 0`` for ``k = 2, \\ldots, K - 1``.
  - ``s_c (\\theta_{t+1} - \\theta_{t}) \\geq 0`` for ``t = 1, \\ldots, T - 1``.

Where:

  - $(math_dict[:sc_scale]) It is the `sc` field of `method`.
  - $(math_dict[:theta_owa])
  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:phi_max_owa])
  - $(math_dict[:W_lmom]) It is the matrix `weights`.
  - $(math_dict[:omega_ik_lmom])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

# Arguments

  - `method`: Programme of the coefficients. The function reads `max_phi` and `sc`.
  - `weights`: Signed L-moment weight matrix ``T \\times (K - 1)``.

# Returns

  - `model::JuMP.Model`: The model of the feasible set.

# Related

  - [`OWAJuMP`](@ref)
  - [`MatNum`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`owa_model_solve`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equations 13 to 15.
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

Solve the L-moment programme in `model`, and return its OWA weights.

The returned weights are the value of the variable `theta`, which [`owa_model_setup`](@ref) holds equal to `weights * phi`. The model holds the coefficients `phi` to a sum of one, not `theta`.

The programme is infeasible for some combinations of the sample size, the highest order and `max_phi`, and the function then returns the fallback. These are the cases with the default `max_phi = 0.5`, measured with a linear programme that maximises the smallest step of the OWA weights:

  - At the highest order `2` it is always infeasible, because the one coefficient must equal one, and `max_phi < 1`.
  - At the highest order `3` the rows leave the one choice ``\\boldsymbol{\\phi} = (0.5, 0.5)``, and its OWA weights decrease near the largest return. It is infeasible at every sample size measured: 3 to 300, 1000 and 5000 observations.
  - At the highest order `4` it is infeasible up to 21 observations, and feasible at every sample size measured from 22 to 80, 100 and 1000 observations.
  - At each highest order `k` from `5` to `8` it is feasible at every sample size measured, from `k` to 80 observations.

# Algorithm

 1. Optimise `model` with the solvers of `method.slv`, through [`optimise_JuMP_model!`](@ref).
 2. If the solve succeeds, return `w`, the value of `model[:theta]`.
 3. Otherwise, warn, and return `w = ncrra_weights(weights, 0.5)`.

# Arguments

  - `model`: Model of the programme, with its objective set.
  - `method`: Programme of the coefficients. The function reads `slv`.
  - `weights`: Signed L-moment weight matrix ``T \\times (K - 1)``, for the fallback.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `size(weights, 1)`.

# Related

  - [`OWAJuMP`](@ref)
  - [`MatNum`](@ref)
  - [`owa_model_setup`](@ref)
  - [`ncrra_weights`](@ref)

# References

  - $(ref_dict[:owa2])
"""
function owa_model_solve(model::JuMP.Model, method::OWAJuMP, weights::MatNum)
    slv = method.slv
    return if optimise_JuMP_model!(model, slv).success
        w = JuMP.value.(model[:theta])
    else
        @warn("Type: $method\nReverting to ncrra_weights.", maxlog = log_limit(:warn),
              _id = @log_id(typeof(method)))
        w = ncrra_weights(weights, 0.5)
    end
end
"""
    owa_l_moment_crm(method::AbstractOrderedWeightsArrayEstimator, weights::MatNum)

Combine the columns of a signed L-moment weight matrix into one OWA weight vector, with the risk aversion coefficients that `method` finds.

[`NormalisedConstantRelativeRiskAversion`](@ref) states the coefficients in closed form, through [`ncrra_weights`](@ref). [`OWAJuMP`](@ref) builds the feasible set with [`owa_model_setup`](@ref), adds the rows and the objective of its `alg`, and solves the model with [`owa_model_solve`](@ref). The docstrings of [`MaximumEntropy`](@ref), [`MinimumSquaredDistance`](@ref) and [`MinimumSumSquares`](@ref) state the three programmes.

# Algorithm

For an [`OWAJuMP`](@ref) `method`:

 1. Build `model`, the feasible set, with [`owa_model_setup`](@ref).
 2. Add the variables and the rows of the objective of `method.alg`, as `# JuMP formulation` states. Under [`MaximumEntropy`](@ref), [`owa_l_moment_crm_entropy`](@ref) adds them.
 3. Set the objective, with [`owa_l_moment_crm_entropy`](@ref) for [`MaximumEntropy`](@ref) and with [`owa_l_moment_crm_sumsq_obj`](@ref) otherwise.
 4. Solve `model` and return its OWA weights, with [`owa_model_solve`](@ref).

# JuMP formulation

## Variables

  - `theta`: Read, the OWA weights that [`owa_model_setup`](@ref) creates.
  - `t`: Created under [`MinimumSquaredDistance`](@ref) and [`MinimumSumSquares`](@ref), the scalar that the cone bounds. Under [`MaximumEntropy`](@ref), [`owa_l_moment_crm_entropy`](@ref) creates it.

## Constraints

Every row is registered under no name. Under [`MaximumEntropy`](@ref) the method adds no row, and [`owa_l_moment_crm_entropy`](@ref) adds the cone rows of the entropy.

  - Under [`MinimumSquaredDistance`](@ref) with [`SOCRiskExpr`](@ref) or [`SquaredSOCRiskExpr`](@ref): ``(s_c t,\\, s_c \\boldsymbol{d}) \\in \\mathcal{K}_{2}``, which states ``t \\geq \\lVert \\boldsymbol{d} \\rVert_{2}``.
  - Under [`MinimumSquaredDistance`](@ref) with [`RSOCRiskExpr`](@ref): ``(s_c t,\\, s_c / 2,\\, s_c \\boldsymbol{d}) \\in \\mathcal{Q}_{r}``, which states ``t \\geq \\lVert \\boldsymbol{d} \\rVert_{2}^{2}``.
  - Under [`MinimumSumSquares`](@ref) with [`SOCRiskExpr`](@ref) or [`SquaredSOCRiskExpr`](@ref): ``(s_c t,\\, s_c \\boldsymbol{\\theta}) \\in \\mathcal{K}_{2}``, which states ``t \\geq \\lVert \\boldsymbol{\\theta} \\rVert_{2}``.
  - Under [`MinimumSumSquares`](@ref) with [`RSOCRiskExpr`](@ref): ``(s_c t,\\, s_c / 2,\\, s_c \\boldsymbol{\\theta}) \\in \\mathcal{Q}_{r}``, which states ``t \\geq \\lVert \\boldsymbol{\\theta} \\rVert_{2}^{2}``.

Where:

  - $(math_dict[:sc_scale]) It is the `sc` field of `method`.
  - $(math_dict[:theta_owa])
  - ``\\boldsymbol{d}``: Differences of adjacent OWA weights, ``d_{t} = \\theta_{t+1} - \\theta_{t}`` for ``t = 1, \\ldots, T - 1``.
  - $(math_dict[:K_q_norm])
  - ``\\mathcal{Q}_{r} = \\{(u, v, \\boldsymbol{z}) : 2 u v \\geq \\lVert \\boldsymbol{z} \\rVert_{2}^{2},\\, u, v \\geq 0\\}``: Rotated second-order cone.
  - $(math_dict[:T])

# Arguments

  - `method`: Estimator of the risk aversion coefficients.
  - `weights`: Signed L-moment weight matrix ``T \\times (K - 1)``. Column `k - 1` holds the weights of the L-moment of order `k`, times ``(-1)^{k}``.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `size(weights, 1)`.

# Related

  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`MaximumEntropy`](@ref)
  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)
  - [`MatNum`](@ref)
  - [`ncrra_weights`](@ref)
  - [`owa_model_setup`](@ref)
  - [`owa_model_solve`](@ref)

# References

  - $(ref_dict[:owa2]) Section 3.2, Equations 11 to 15.
"""
function owa_l_moment_crm(method::NormalisedConstantRelativeRiskAversion, weights::MatNum)
    return ncrra_weights(weights, method.g)
end
"""
    owa_l_moment_crm_entropy(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MaximumEntropy},
                             model::JuMP.Model)

Add the entropy objective of the [`MaximumEntropy`](@ref) programme to `model`.

The method dispatches on the [`EntropyFormulation`](@ref) in `method.alg.alg`. It reads the risk aversion coefficients `phi`, adds the variable and the cone rows that bound their entropy, and sets the objective. It does not solve the model.

The two encodings state one problem. Under [`ExponentialConeEntropy`](@ref) the objective is the entropy less ``\\log s_c``, because the cones hold ``s_c \\phi_{k}`` and the coefficients sum to one. So the objective values differ when ``s_c \\neq 1``, and the maximiser does not.

The objective carries a factor of ``10^{-3}``, which does not change the maximiser. The steps of ``\\boldsymbol{\\theta}`` are of order ``1/T^{2}``, the entropy is of order one, and the maximiser touches the rows ``\\theta_{t+1} \\geq \\theta_{t}`` tangentially. Without the factor, Clarabel stopped short of optimal on 31 of 39 programmes, with ``T`` from 30 to 5000, the highest order from 4 to 10 and `max_phi = 0.75`. With it, Clarabel solved every one of them.

# JuMP formulation

## Variables

  - `phi`: Read, the risk aversion coefficients ``\\boldsymbol{\\phi}`` that [`owa_model_setup`](@ref) creates.
  - `t`: Created. Under [`RelativeEntropy`](@ref) it is a scalar, and under [`ExponentialConeEntropy`](@ref) it is a vector ``(K - 1) \\times 1``.

## Constraints

Every row is registered under no name.

  - Under [`RelativeEntropy`](@ref): ``(s_c t,\\, s_c \\boldsymbol{1},\\, s_c \\boldsymbol{\\phi}) \\in \\mathcal{K}_{\\mathrm{rel}}``, which states ``t \\geq \\sum_{k=2}^{K} \\phi_{k} \\log \\phi_{k}``.
  - Under [`ExponentialConeEntropy`](@ref): ``(s_c t_{k},\\, s_c \\phi_{k},\\, 1) \\in \\mathcal{K}_{\\exp}`` for each ``k``, which states ``t_{k} \\leq -\\phi_{k} \\log(s_c \\phi_{k})``.

## Objective

  - Under [`RelativeEntropy`](@ref): `Max` of ``-10^{-3} s_o t``.
  - Under [`ExponentialConeEntropy`](@ref): `Max` of ``10^{-3} s_o \\sum_{k=2}^{K} t_{k}``.

Where:

  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:theta_owa])
  - $(math_dict[:sc_scale]) It is the `sc` field of `method`.
  - $(math_dict[:so_scale]) It is the `so` field of `method`.
  - ``\\mathcal{K}_{\\mathrm{rel}} = \\{(u, \\boldsymbol{v}, \\boldsymbol{z}) : u \\geq \\sum_{i} z_{i} \\log(z_{i} / v_{i}),\\, \\boldsymbol{v}, \\boldsymbol{z} \\geq 0\\}``: Relative entropy cone.
  - $(math_dict[:K_exp])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

# Arguments

  - `method`: Programme of the coefficients, whose `alg` is a [`MaximumEntropy`](@ref).
  - `model::JuMP.Model`: Model of [`owa_l_moment_crm`](@ref), which holds the variable `phi`.

# Returns

  - `nothing`. The function changes `model` in place.

# Related

  - [`MaximumEntropy`](@ref)
  - [`EntropyFormulation`](@ref)
  - [`owa_model_setup`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`owa_l_moment_crm_sumsq_obj`](@ref)

# References

  - $(ref_dict[:owa2])
"""
function owa_l_moment_crm_entropy(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                                  <:MaximumEntropy{<:RelativeEntropy}},
                                  model::JuMP.Model)
    sc = method.sc
    so = method.so
    phi = model[:phi]
    N = length(phi)
    ovec = range(sc, sc; length = N)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; ovec; sc * phi] in JuMP.MOI.RelativeEntropyCone(2 * N + 1))
    # The steps of `theta` are of order 1/T^2 and the entropy is of order one, so the
    # solver stalls short of optimal unless the objective is scaled down.
    JuMP.@objective(model, Max, -so * t / 1000)
    return nothing
end
function owa_l_moment_crm_entropy(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                                  <:MaximumEntropy{<:ExponentialConeEntropy}},
                                  model::JuMP.Model)
    sc = method.sc
    so = method.so
    phi = model[:phi]
    N = length(phi)
    JuMP.@variable(model, t[1:N])
    JuMP.@constraint(model, [i = 1:N],
                     [sc * t[i], sc * phi[i], 1] in JuMP.MOI.ExponentialCone())
    JuMP.@objective(model, Max, so * sum(t) / 1000)
    return nothing
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any, <:MaximumEntropy},
                          weights::MatNum)
    model = owa_model_setup(method, weights)
    owa_l_moment_crm_entropy(method, model)
    return owa_model_solve(model, method, weights)
end
"""
    owa_l_moment_crm_sumsq_obj(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                               <:SquaredOrderedWeightsArrayAlgorithm},
                               model::JuMP.Model)

Set the objective of a [`SquaredOrderedWeightsArrayAlgorithm`](@ref) programme on `model`.

[`owa_l_moment_crm`](@ref) adds the cone row that bounds the variable `t`. This method sets the objective alone, and it dispatches on the [`SecondMomentFormulation`](@ref) that parameterises `method.alg`. It does not solve the model. The three encodings give one minimiser, so they return the same OWA weights to solver tolerance.

# JuMP formulation

## Variables

  - `t`: Read, the scalar that the cone row of the caller bounds.

## Objective

  - Under [`SOCRiskExpr`](@ref) or [`RSOCRiskExpr`](@ref): `Min` of ``s_o t``.
  - Under [`SquaredSOCRiskExpr`](@ref): `Min` of ``s_o t^{2}``, because the cone row of the caller bounds ``t`` by a norm, not by its square.

Where:

  - $(math_dict[:so_scale]) It is the `so` field of `method`.

# Arguments

  - `method`: Programme of the coefficients, whose `alg` is a [`MinimumSquaredDistance`](@ref) or a [`MinimumSumSquares`](@ref).
  - `model::JuMP.Model`: Model of [`owa_l_moment_crm`](@ref), which holds the variable `t` and its cone row.

# Returns

  - The objective function that the method sets on `model`. The function changes `model` in place.

# Related

  - [`MinimumSquaredDistance`](@ref)
  - [`MinimumSumSquares`](@ref)
  - [`SquaredOrderedWeightsArrayAlgorithm`](@ref)
  - [`owa_model_setup`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`owa_l_moment_crm_entropy`](@ref)

# References

  - $(ref_dict[:owa2])
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
                     [sc * t; sc * 0.5; sc * (theta[2:end] - theta[1:(end - 1)])] in
                     JuMP.RotatedSecondOrderCone())
    owa_l_moment_crm_sumsq_obj(method, model)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMP{<:Any, <:Any, <:Any, <:Any,
                                          <:MinimumSumSquares{<:UnionSOCRiskExpr}},
                          weights::MatNum)
    sc = method.sc
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
    JuMP.@constraint(model, [sc * t; sc * 0.5; sc * theta] in JuMP.RotatedSecondOrderCone())
    owa_l_moment_crm_sumsq_obj(method, model)
    return owa_model_solve(model, method, weights)
end
"""
    owa_gmd(T::Integer)

Compute the OWA weight vector of the Gini mean difference.

The weights rise linearly with the rank and sum to zero, so the risk is a dispersion and does not depend on the mean of the returns. [gmd](@citet) states the Gini mean difference, and Section 2.1 of [owa3](@cite) states its OWA weights.

# Mathematical definition

```math
\\begin{align}
\\eta_{t} &= \\dfrac{2 \\left(2 t - 1 - T\\right)}{T \\left(T - 1\\right)} \\quad \\forall t = 1,\\, \\ldots,\\, T\\\\
\\sum\\limits_{t=1}^{T} \\eta_{t} x_{(t)} &= \\dfrac{1}{T \\left(T - 1\\right)} \\sum\\limits_{i=1}^{T} \\sum\\limits_{j=1}^{T} \\left\\lvert x_{i} - x_{j} \\right\\rvert\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - $(math_dict[:x_t_asc])
  - $(math_dict[:xret])
  - $(math_dict[:T])

# Arguments

  - `T`: Number of observations, `T >= 2`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`. It is a lazy range, and its entries sum to zero.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_tg`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:gmd])
  - $(ref_dict[:owa1])
  - $(ref_dict[:owa3])
  - $(ref_dict[:cajas2025]) Section 7.2.1.2.
"""
function owa_gmd(T::Integer)
    return (4 * (1:T) .- 2 * (T + 1)) / (T * (T - 1))
end
"""
    owa_cvar(T::Integer, alpha::Number = 0.05)

Compute the OWA weight vector of the conditional value at risk.

The weights put the mass ``-1`` on the ``\\alpha T`` smallest returns, so the risk is the mean loss of the worst ``\\alpha`` share of the sample. When ``\\alpha T`` is not an integer, the order statistic at the boundary takes the remaining fraction. The vector is nondecreasing and sums to ``-1``, and it gives the value of [`ConditionalValueatRisk`](@ref) with no observation weights.

# Mathematical definition

```math
\\begin{align}
k &= \\lfloor \\alpha T \\rfloor\\\\
\\eta_{t} &= \\begin{cases}
-\\dfrac{1}{\\alpha T} & t \\leq k\\\\
-1 + \\dfrac{k}{\\alpha T} & t = k + 1\\\\
0 & t > k + 1
\\end{cases}\\\\
\\sum\\limits_{t=1}^{T} \\eta_{t} x_{(t)} &= -\\dfrac{1}{\\alpha T} \\left(\\sum\\limits_{t=1}^{k} x_{(t)} + \\left(\\alpha T - k\\right) x_{(k+1)}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_rm])
  - ``k``: Number of order statistics that the tail holds in full.
  - $(math_dict[:eta_owa])
  - $(math_dict[:x_t_asc])
  - $(math_dict[:xret])
  - $(math_dict[:T])

# Arguments

  - `T`: Number of observations.
  - `alpha`: Significance level, the share of the sample in the tail.

# Validation

  - `0 < alpha < 1`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_wcvar`](@ref)
  - [`owa_tg`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.2.2.4.
"""
function owa_cvar(T::Integer, alpha::Number = 0.05)
    assert_unit_interval(alpha, :alpha)
    k = floor(Int, T * alpha)
    w = zeros(typeof(alpha), T)
    w[1:k] .= -one(alpha) / (T * alpha)
    w[k + 1] = -one(alpha) - sum(@view(w[1:k]))
    return w
end
"""
$(DocStringExtensions.TYPEDEF)

Builds the OWA weight vector of the conditional value at risk at level `alpha`.

Called as `r(T)`, it returns [`owa_cvar`](@ref)`(T, alpha)`, whose docstring states the closed form. A Calibration Rule in `alpha` resolves against the prior before the call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayConditionalValueatRisk(;
        alpha::Num_SigCal = 0.05
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

# Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_cvar`](@ref)
  - [`OWA_Func_VecNum`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.2.2.4.
"""
@concrete struct OrderedWeightsArrayConditionalValueatRisk <:
                 AbstractOrderedWeightsArrayFunction
    """
    $(field_dict[:alpha])
    """
    alpha
    function OrderedWeightsArrayConditionalValueatRisk(alpha::Num_SigCal)
        assert_unit_interval(alpha, :alpha)
        return new{typeof(alpha)}(alpha)
    end
end
function OrderedWeightsArrayConditionalValueatRisk(; alpha::Num_SigCal = 0.05)
    return OrderedWeightsArrayConditionalValueatRisk(alpha)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::OrderedWeightsArrayConditionalValueatRisk) = (; alpha = x.alpha)
function (r::OrderedWeightsArrayConditionalValueatRisk)(T::Integer)
    return owa_cvar(T, r.alpha)
end
"""
    owa_wcvar(T::Integer, alphas::VecNum, weights::VecNum)

Compute the OWA weight vector of a weighted sum of conditional values at risk.

The vector sums to minus the sum of `weights`. Section 2.1 of [owa3](@cite) states it with weights that sum to one, and the function does not check that sum. A negative weight can make the vector decrease, and the risk measure is then not convex.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\eta} &= \\sum\\limits_{j=1}^{J} p_{j} \\boldsymbol{\\eta}^{\\mathrm{CVaR}}(\\alpha_{j})\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - $(math_dict[:eta_cvar])
  - ``\\alpha_{j}``, ``p_{j}``: The ``j``-th entries of `alphas` and `weights`.
  - ``J``: Number of levels, `length(alphas)`.

# Arguments

  - `T`: Number of observations.
  - `alphas`: Vector of significance levels, each in ``(0, 1)``.
  - `weights`: Vector of the weight of each level.

# Validation

  - `length(alphas) == length(weights)`. Raises a `DimensionMismatch`.
  - Each entry of `alphas` satisfies `0 < alpha < 1`, through [`owa_cvar`](@ref).

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_tg`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa3]) Section 2.1.
"""
function owa_wcvar(T::Integer, alphas::VecNum, weights::VecNum)
    @argcheck(length(alphas) == length(weights),
              DimensionMismatch("`alphas` has $(length(alphas)) levels and `weights` has $(length(weights)) entries. Each level needs one weight."))
    w = zeros(promote_type(eltype(alphas), eltype(weights)), T)
    for (i, j) in zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end
    return w
end
"""
    owa_tg(T::Integer; alpha_i::Number = 1e-4, alpha::Number = 0.05, a_sim::Integer = 100)

Compute the OWA weight vector of the tail Gini, as a weighted sum of conditional values at risk.

Section 2.1 of [owa3](@cite) states the tail Gini weights as this weighted sum, over a grid of levels from zero to `alpha`. The library starts the grid at `alpha_i > 0`, because the conditional value at risk has no level zero. The quadrature weights sum to one, so the vector sums to ``-1``.

# Mathematical definition

```math
\\begin{align}
\\alpha_{j} &= \\alpha_{i} + \\dfrac{j - 1}{m - 1} \\left(\\alpha - \\alpha_{i}\\right) \\quad \\forall j = 1,\\, \\ldots,\\, m\\\\
p_{1} &= \\dfrac{\\alpha_{1} \\alpha_{2}}{\\alpha_{m}^{2}}\\\\
p_{j} &= \\dfrac{\\left(\\alpha_{j+1} - \\alpha_{j-1}\\right) \\alpha_{j}}{\\alpha_{m}^{2}} \\quad \\forall j = 2,\\, \\ldots,\\, m - 1\\\\
p_{m} &= \\dfrac{\\alpha_{m} - \\alpha_{m-1}}{\\alpha_{m}}\\\\
\\boldsymbol{\\eta} &= \\sum\\limits_{j=1}^{m} p_{j} \\boldsymbol{\\eta}^{\\mathrm{CVaR}}(\\alpha_{j})\\,.
\\end{align}
```

Where:

  - ``\\alpha_{i}``, ``\\alpha``: The levels `alpha_i` and `alpha`, the two ends of the grid.
  - ``m``: Number of grid points, `a_sim`.
  - ``\\alpha_{j}``, ``p_{j}``: The ``j``-th level of the grid and its quadrature weight.
  - $(math_dict[:eta_owa])
  - $(math_dict[:eta_cvar])

# Arguments

  - `T`: Number of observations.
  - `alpha_i`: Inner level, the first point of the grid.
  - `alpha`: Outer level, the last point of the grid.
  - `a_sim`: Number of grid points.

# Validation

  - `0 < alpha_i < alpha < 1`. Raises a `DomainError`.
  - `a_sim >= 2`. Raises a `DomainError`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_wcvar`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:tgini])
  - $(ref_dict[:owa1])
  - $(ref_dict[:owa3]) Section 2.1.
  - $(ref_dict[:cajas2025]) Section 7.2.2.5.
"""
function owa_tg(T::Integer; alpha_i::Number = 1e-4, alpha::Number = 0.05,
                a_sim::Integer = 100)
    @argcheck(zero(alpha) < alpha_i < alpha < one(alpha),
              DomainError((alpha_i, alpha),
                          "`alpha_i` is $alpha_i and `alpha` is $alpha. The tail Gini integrates the CVaR from the inner level `alpha_i` to the outer level `alpha`, so `0 < alpha_i < alpha < 1` must hold."))
    @argcheck(2 <= a_sim,
              DomainError(a_sim, "`a_sim` is $a_sim, and the grid needs `a_sim >= 2`."))
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

Builds the OWA weight vector of the tail Gini at level `alpha`.

Called as `r(T)`, it returns [`owa_tg`](@ref)`(T; alpha_i, alpha, a_sim)`, whose docstring states the quadrature over the conditional values at risk. A Calibration Rule in `alpha` resolves against the prior before the call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayTailGini(;
        alpha_i::Number = 1e-4,
        alpha::Num_SigCal = 0.05,
        a_sim::Integer = 100
    ) -> OrderedWeightsArrayTailGini

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_i_alpha])
  - $(val_dict[:a_sim_ge2])

# Examples

```jldoctest
julia> OrderedWeightsArrayTailGini()
OrderedWeightsArrayTailGini
  alpha_i ┼ Float64: 0.0001
    alpha ┼ Float64: 0.05
    a_sim ┴ Int64: 100
```

# Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_tg`](@ref)
  - [`OWA_Func_VecNum`](@ref)

# References

  - $(ref_dict[:tgini])
  - $(ref_dict[:owa1])
  - $(ref_dict[:owa3])
  - $(ref_dict[:cajas2025]) Section 7.2.2.5.
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
    function OrderedWeightsArrayTailGini(alpha_i::Number, alpha::Num_SigCal, a_sim::Integer)
        # The joint bound is the whole of the ordering validation, and it can only run
        # where `alpha` is a number. A rule states no value at construction, so `alpha_i`
        # is checked on its own here and the pair is checked again when the rebuild runs at
        # fold time, against the number the rule returned.
        if isa(alpha, Number)
            @argcheck(0 < alpha_i < alpha < 1,
                      DomainError((alpha_i, alpha),
                                  "`alpha_i` is $alpha_i and `alpha` is $alpha. The tail Gini integrates the CVaR from the inner level `alpha_i` to the outer level `alpha`, so `0 < alpha_i < alpha < 1` must hold."))
        else
            assert_unit_interval(alpha_i, :alpha_i)
        end
        @argcheck(2 <= a_sim,
                  DomainError(a_sim, "`a_sim` is $a_sim, and the grid needs `a_sim >= 2`."))
        return new{typeof(alpha_i), typeof(alpha), typeof(a_sim)}(alpha_i, alpha, a_sim)
    end
end
function OrderedWeightsArrayTailGini(; alpha_i::Number = 1e-4, alpha::Num_SigCal = 0.05,
                                     a_sim::Integer = 100)
    return OrderedWeightsArrayTailGini(alpha_i, alpha, a_sim)
end
# Calibration slots — see `calibration_slots`. `alpha_i` is a starting point of the inner
# integration, not a quantity to estimate, so it is not one of them.
calibration_slots(x::OrderedWeightsArrayTailGini) = (; alpha = x.alpha)
function (r::OrderedWeightsArrayTailGini)(T::Integer)
    return owa_tg(T; alpha_i = r.alpha_i, alpha = r.alpha, a_sim = r.a_sim)
end
"""
    owa_wr(T::Integer)

Compute the OWA weight vector of the worst realisation.

The vector selects the smallest return with the weight ``-1``, so the risk is the largest loss of the sample. It gives the value of [`WorstRealisation`](@ref).

# Mathematical definition

```math
\\begin{align}
\\eta_{t} &= \\begin{cases}
-1 & t = 1\\\\
0 & t > 1
\\end{cases}\\\\
\\sum\\limits_{t=1}^{T} \\eta_{t} x_{(t)} &= -x_{(1)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - $(math_dict[:x_t_asc])
  - $(math_dict[:xret])
  - $(math_dict[:T])

# Arguments

  - `T`: Number of observations.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_rg`](@ref)
  - [`WorstRealisation`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa3]) Section 2.1.
"""
function owa_wr(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    return w
end
"""
    owa_rg(T::Integer)

Compute the OWA weight vector of the range of the returns.

The vector weights the largest return by ``1`` and the smallest by ``-1``, so the risk is the largest return less the smallest. It gives the value of [`Range`](@ref). A sample of one observation has a range of zero, and its vector is ``(0)``.

# Mathematical definition

```math
\\begin{align}
\\eta_{t} &= \\begin{cases}
-1 & t = 1\\\\
1 & t = T\\\\
0 & 1 < t < T
\\end{cases}\\\\
\\sum\\limits_{t=1}^{T} \\eta_{t} x_{(t)} &= x_{(T)} - x_{(1)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - $(math_dict[:x_t_asc])
  - $(math_dict[:xret])
  - $(math_dict[:T])

# Arguments

  - `T`: Number of observations.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_wr`](@ref)
  - [`Range`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa3]) Section 2.1.
"""
function owa_rg(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    # Add, so that one observation gives a zero weight and a zero range.
    w[T] += 1
    return w
end
"""
    owa_cvarrg(T::Integer; alpha::Number = 0.05, beta::Number = alpha)

Compute the OWA weight vector of the conditional value at risk range.

The vector is the loss-tail vector at level `alpha` less the reversed loss-tail vector at level `beta`. The reversal moves the second tail to the largest returns, and the subtraction makes its weights positive. So the risk is the conditional value at risk of the losses plus the mean of the `beta` share of the largest returns, a sum of the two tails. The vector sums to zero.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\eta} &= \\boldsymbol{\\eta}^{\\mathrm{CVaR}}(\\alpha) - \\mathbf{A} \\boldsymbol{\\eta}^{\\mathrm{CVaR}}(\\beta)\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - $(math_dict[:eta_cvar])
  - $(math_dict[:A_rev])
  - ``\\alpha``, ``\\beta``: Significance levels of the loss tail and of the gain tail.

# Arguments

  - `T`: Number of observations.
  - `alpha`: Significance level of the loss tail.
  - `beta`: Significance level of the gain tail.

# Validation

  - `0 < alpha < 1` and `0 < beta < 1`, through [`owa_cvar`](@ref).

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_cvar`](@ref)
  - [`owa_rg`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa3]) Section 2.1.
"""
function owa_cvarrg(T::Integer; alpha::Number = 0.05, beta::Number = alpha)
    return owa_cvar(T, alpha) - reverse!(owa_cvar(T, beta))
end
"""
$(DocStringExtensions.TYPEDEF)

Builds the OWA weight vector of the conditional value at risk range at the levels `alpha` and `beta`.

Called as `r(T)`, it returns [`owa_cvarrg`](@ref)`(T; alpha, beta)`, whose docstring states the closed form. `alpha` is the level of the loss tail and `beta` the level of the gain tail. A Calibration Rule in either slot resolves against the prior before the call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayConditionalValueatRiskRange(;
        alpha::Num_SigCal = 0.05,
        beta::Num_SigCal = alpha
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

# Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_cvarrg`](@ref)
  - [`OWA_Func_VecNum`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.2.3.
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
    function OrderedWeightsArrayConditionalValueatRiskRange(alpha::Num_SigCal,
                                                            beta::Num_SigCal)
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        return new{typeof(alpha), typeof(beta)}(alpha, beta)
    end
end
function OrderedWeightsArrayConditionalValueatRiskRange(; alpha::Num_SigCal = 0.05,
                                                        beta::Num_SigCal = alpha)
    return OrderedWeightsArrayConditionalValueatRiskRange(alpha, beta)
end
# Calibration slots — see `calibration_slots`. One slot per tail, each with its own role.
function calibration_slots(x::OrderedWeightsArrayConditionalValueatRiskRange)
    return (; alpha = x.alpha, beta = x.beta)
end
function (r::OrderedWeightsArrayConditionalValueatRiskRange)(T::Integer)
    return owa_cvarrg(T; alpha = r.alpha, beta = r.beta)
end
"""
    owa_wcvarrg(T::Integer, alphas::VecNum, weights_a::VecNum,
                betas::VecNum = alphas,
                weights_b::VecNum = weights_a)

Compute the OWA weight vector of the range of two weighted sums of conditional values at risk.

The vector is the loss-tail vector of [`owa_wcvar`](@ref) at `alphas` and `weights_a`, less the reversed vector at `betas` and `weights_b`. As in [`owa_cvarrg`](@ref), the reversal and the subtraction make the risk a sum of the two tails.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\eta} &= \\sum\\limits_{j=1}^{J} p_{j} \\boldsymbol{\\eta}^{\\mathrm{CVaR}}(\\alpha_{j}) - \\mathbf{A} \\sum\\limits_{l=1}^{L} q_{l} \\boldsymbol{\\eta}^{\\mathrm{CVaR}}(\\beta_{l})\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - $(math_dict[:eta_cvar])
  - $(math_dict[:A_rev])
  - ``\\alpha_{j}``, ``p_{j}``: The ``j``-th entries of `alphas` and `weights_a`, and ``J`` their length.
  - ``\\beta_{l}``, ``q_{l}``: The ``l``-th entries of `betas` and `weights_b`, and ``L`` their length.

# Arguments

  - `T`: Number of observations.
  - `alphas`: Vector of significance levels of the loss tail.
  - `weights_a`: Vector of the weight of each level of the loss tail.
  - `betas`: Vector of significance levels of the gain tail.
  - `weights_b`: Vector of the weight of each level of the gain tail.

# Validation

  - `length(alphas) == length(weights_a)` and `length(betas) == length(weights_b)`, through [`owa_wcvar`](@ref). Raises a `DimensionMismatch`.
  - Each level is in ``(0, 1)``, through [`owa_cvar`](@ref).

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_wcvar`](@ref)
  - [`owa_cvarrg`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa3]) Section 2.1.
"""
function owa_wcvarrg(T::Integer, alphas::VecNum, weights_a::VecNum, betas::VecNum = alphas,
                     weights_b::VecNum = weights_a)
    w = owa_wcvar(T, alphas, weights_a) - reverse!(owa_wcvar(T, betas, weights_b))
    return w
end
"""
    owa_tgrg(T::Integer; alpha_i::Number = 0.0001, alpha::Number = 0.05, a_sim::Integer = 100,
             beta_i::Number = alpha_i, beta::Number = alpha, b_sim::Integer = a_sim)

Compute the OWA weight vector of the tail Gini range.

The vector is the tail Gini vector of [`owa_tg`](@ref) at `alpha_i`, `alpha` and `a_sim`, less the reversed tail Gini vector at `beta_i`, `beta` and `b_sim`. As in [`owa_cvarrg`](@ref), the reversal and the subtraction make the risk a sum of the two tails. The vector sums to zero.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\eta} &= \\boldsymbol{\\eta}^{\\mathrm{TG}}(\\alpha_{i}, \\alpha, m_{a}) - \\mathbf{A} \\boldsymbol{\\eta}^{\\mathrm{TG}}(\\beta_{i}, \\beta, m_{b})\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_owa])
  - ``\\boldsymbol{\\eta}^{\\mathrm{TG}}(\\alpha_{i}, \\alpha, m)``: Tail Gini OWA weight vector, the value of [`owa_tg`](@ref)`(T; alpha_i, alpha, a_sim = m)`.
  - $(math_dict[:A_rev])
  - ``\\alpha_{i}``, ``\\alpha``, ``m_{a}``: Inner level, outer level and grid size of the loss tail.
  - ``\\beta_{i}``, ``\\beta``, ``m_{b}``: Inner level, outer level and grid size of the gain tail.

# Arguments

  - `T`: Number of observations.
  - `alpha_i`: Inner level of the loss tail.
  - `alpha`: Outer level of the loss tail.
  - `a_sim`: Number of grid points of the loss tail.
  - `beta_i`: Inner level of the gain tail.
  - `beta`: Outer level of the gain tail.
  - `b_sim`: Number of grid points of the gain tail.

# Validation

  - `0 < alpha_i < alpha < 1` and `0 < beta_i < beta < 1`, through [`owa_tg`](@ref). Raises a `DomainError`.
  - `a_sim >= 2` and `b_sim >= 2`, through [`owa_tg`](@ref). Raises a `DomainError`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_tg`](@ref)
  - [`owa_rg`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa3]) Section 2.1.
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

Builds the OWA weight vector of the tail Gini range.

Called as `r(T)`, it returns [`owa_tgrg`](@ref)`(T; alpha_i, alpha, a_sim, beta_i, beta, b_sim)`, whose docstring states the closed form. The `alpha` fields set the loss tail and the `beta` fields the gain tail. A Calibration Rule in `alpha` or `beta` resolves against the prior before the call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayTailGiniRange(;
        alpha_i::Number = 1e-4,
        alpha::Num_SigCal = 0.05,
        a_sim::Integer = 100,
        beta_i::Number = alpha_i,
        beta::Num_SigCal = alpha,
        b_sim::Integer = a_sim
    ) -> OrderedWeightsArrayTailGiniRange

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_i_alpha])
  - $(val_dict[:a_sim_ge2])
  - $(val_dict[:beta_i_beta])
  - $(val_dict[:b_sim_ge2])

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

# Related

  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`owa_tgrg`](@ref)
  - [`OWA_Func_VecNum`](@ref)

# References

  - $(ref_dict[:tgini])
  - $(ref_dict[:owa1])
  - $(ref_dict[:owa3])
  - $(ref_dict[:cajas2025]) Section 7.2.3.
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
    function OrderedWeightsArrayTailGiniRange(alpha_i::Number, alpha::Num_SigCal,
                                              a_sim::Integer, beta_i::Number,
                                              beta::Num_SigCal, b_sim::Integer)
        # Each joint bound runs where its outer level is a number, on the terms the scalar
        # tail-Gini type states. A rule states no value at construction, so its inner bound
        # is checked on its own here and the pair is checked at fold time.
        if isa(alpha, Number)
            @argcheck(0 < alpha_i < alpha < 1,
                      DomainError((alpha_i, alpha),
                                  "`alpha_i` is $alpha_i and `alpha` is $alpha. The tail Gini integrates the CVaR from the inner level `alpha_i` to the outer level `alpha`, so `0 < alpha_i < alpha < 1` must hold."))
        else
            assert_unit_interval(alpha_i, :alpha_i)
        end
        @argcheck(2 <= a_sim,
                  DomainError(a_sim, "`a_sim` is $a_sim, and the grid needs `a_sim >= 2`."))
        if isa(beta, Number)
            @argcheck(0 < beta_i < beta < 1,
                      DomainError((beta_i, beta),
                                  "`beta_i` is $beta_i and `beta` is $beta. The upper tail Gini integrates the CVaR from the inner level `beta_i` to the outer level `beta`, so `0 < beta_i < beta < 1` must hold."))
        else
            assert_unit_interval(beta_i, :beta_i)
        end
        @argcheck(2 <= b_sim,
                  DomainError(b_sim, "`b_sim` is $b_sim, and the grid needs `b_sim >= 2`."))
        return new{typeof(alpha_i), typeof(alpha), typeof(a_sim), typeof(beta_i),
                   typeof(beta), typeof(b_sim)}(alpha_i, alpha, a_sim, beta_i, beta, b_sim)
    end
end
function OrderedWeightsArrayTailGiniRange(; alpha_i::Number = 1e-4,
                                          alpha::Num_SigCal = 0.05, a_sim::Integer = 100,
                                          beta_i::Number = alpha_i,
                                          beta::Num_SigCal = alpha, b_sim::Integer = a_sim)
    return OrderedWeightsArrayTailGiniRange(alpha_i, alpha, a_sim, beta_i, beta, b_sim)
end
# Calibration slots — see `calibration_slots`. The two inner starting points are not
# quantities to estimate, so neither is one of them.
calibration_slots(x::OrderedWeightsArrayTailGiniRange) = (; alpha = x.alpha, beta = x.beta)
function (r::OrderedWeightsArrayTailGiniRange)(T::Integer)
    return owa_tgrg(T; alpha_i = r.alpha_i, alpha = r.alpha, a_sim = r.a_sim,
                    beta_i = r.beta_i, beta = r.beta, b_sim = r.b_sim)
end
"""
    owa_l_moment(T::Integer, k::Integer = 2)

Compute the OWA weight vector of the sample L-moment of order `k`.

The `k`-th sample L-moment is a weighted sum of the order statistics of the sample, so it is an OWA operator, and this function returns its weights. For `k >= 2` the weights sum to zero, and for `k = 1` they are ``1/T``, the weights of the mean.

The binomial coefficients that hold ``T`` are computed in the element type of the weights, because ``\\binom{T}{k}`` exceeds the largest `Int` at `T = 1000` and `k = 8`. On a `Float64` vector the weights agree with exact rational arithmetic to a relative error below `1e-14` up to `k = 10`.

# Mathematical definition

```math
\\begin{align}
\\omega_{i}^{k} &= \\dfrac{1}{k} \\binom{T}{k}^{-1} \\sum\\limits_{j=0}^{k-1} \\left(-1\\right)^{j} \\binom{k-1}{j} \\binom{i-1}{k-1-j} \\binom{T-i}{j}\\\\
\\lambda_{k} &= \\sum\\limits_{i=1}^{T} \\omega_{i}^{k} y_{[i]}\\,.
\\end{align}
```

Where:

  - $(math_dict[:omega_ik_lmom])
  - ``\\lambda_{k}``: The ``k``-th sample L-moment.
  - ``y_{[i]}``: The ``i``-th order statistic of the sample, in ascending order.
  - $(math_dict[:T])
  - ``k``: The order of the L-moment.

# Arguments

  - `T`: Number of observations.
  - `k`: Order of the L-moment.

# Validation

  - `1 <= k <= T`. Raises a `DomainError`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`. For `k >= 2` it sums to zero. Its first entry is ``(-1)^{k-1}/T`` and its last entry is ``1/T``.

# Related

  - [`owa_l_moment_crm`](@ref)
  - [`LinearMoment`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa2]) Equation 3.
  - $(ref_dict[:cajas2025]) Section 3.1.5.
"""
function owa_l_moment(T::Integer, k::Integer = 2)
    @argcheck(1 <= k <= T,
              DomainError(k,
                          "`k` is $k and `T` is $T. The sample L-moment of order `k` weighs `k` order statistics at a time, so `1 <= k <= T` must hold."))
    T, k = promote(T, k)
    F = typeof(inv(T * k))
    w = Vector{F}(undef, T)
    # The binomials of `T` overflow an integer at a moderate size, `binomial(1000, 8)`
    # among them, so the sample-size binomials are computed in the type of the weights.
    d = k * binomial(convert(F, T), k)
    for i in eachindex(w)
        a = zero(F)
        for j in 0:(k - 1)
            a += (-1)^j *
                 binomial(k - 1, j) *
                 binomial(convert(F, i - 1), k - 1 - j) *
                 binomial(convert(F, T - i), j)
        end
        w[i] = a / d
    end
    return w
end
"""
    owa_l_moment_crm(T::Integer,
                     method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion();
                     k::Integer = 2)

Compute the OWA weight vector of the L-moment convex risk measure of the orders `2` to `k`.

The risk measure is a signed combination of the L-moments of the portfolio returns, with the risk aversion coefficients that `method` finds. The coefficients sum to one, and the OWA weights do not, because each L-moment weight vector of order two or more sums to zero.

Each signed L-moment weight vector starts at ``-1/T``, so the first OWA weight is ``-1/T`` at every order and for every `method`. A higher `k` moves the other entries. Under the default `method` on 60 observations, the largest move is ``0.61/T`` from `k = 2` to `k = 3`, and ``0.053/T`` from `k = 7` to `k = 8`.

# Mathematical definition

```math
\\begin{align}
\\rho(\\boldsymbol{w}) &= \\sum\\limits_{k=2}^{K} \\left(-1\\right)^{k} \\phi_{k} \\lambda_{k}\\left(\\hat{\\boldsymbol{r}}\\right)\\\\
&= \\sum\\limits_{i=1}^{T} \\eta_{i} \\hat{r}_{[i]}\\\\
\\eta_{i} &= \\sum\\limits_{k=2}^{K} \\left(-1\\right)^{k} \\phi_{k} \\omega_{i}^{k}\\,.
\\end{align}
```

Where:

  - ``\\rho(\\boldsymbol{w})``: L-moment convex risk measure of the portfolio.
  - ``\\lambda_{k}(\\hat{\\boldsymbol{r}})``: The ``k``-th sample L-moment of the net portfolio returns.
  - $(math_dict[:rhat_net_vec]) ``\\hat{r}_{[i]}`` is its ``i``-th smallest entry.
  - $(math_dict[:phi_k_lmom])
  - $(math_dict[:omega_ik_lmom])
  - $(math_dict[:eta_owa])
  - $(math_dict[:K_lmom])
  - $(math_dict[:T])

# Algorithm

 1. Check that `k >= 2`.
 2. Build `weights`, the signed L-moment weight matrix ``T \\times (k - 1)``. Its column `i - 1` is ``(-1)^{i}`` times [`owa_l_moment`](@ref)`(T, i)`, for `i` from `2` to `k`.
 3. Return `owa_l_moment_crm(method, weights)`, which finds the coefficients and combines the columns.

# Arguments

  - `T`: Number of observations.
  - `method`: Estimator of the risk aversion coefficients.
  - `k`: Highest L-moment order.

# Validation

  - `k >= 2`. Raises a `DomainError`.
  - `k <= T`, through [`owa_l_moment`](@ref). Raises a `DomainError`.

# Returns

  - `w::VecNum`: Vector of OWA weights of length `T`.

# Related

  - [`owa_l_moment`](@ref)
  - [`LinearMoment`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:owa2]) Equation 11, Section 3.2.
  - $(ref_dict[:cajas2025]) Section 3.1.5.
"""
function owa_l_moment_crm(T::Integer,
                          method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion();
                          k::Integer = 2)
    @argcheck(2 <= k,
              DomainError(k,
                          "`k` is $k, and the L-moment risk measure spans the orders `2:k`, so it needs at least the second. State an integer greater than or equal to `2`."))
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(2:k))
    for i in 2:k
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] = wi
    end
    return owa_l_moment_crm(method, weights)
end
"""
$(DocStringExtensions.TYPEDEF)

Builds the OWA weight vector of the L-moment convex risk measure of the orders `2` to `k`.

Called as `lm(T)`, it returns [`owa_l_moment_crm`](@ref)`(T, method; k)`, whose docstring states the measure. `method` finds the risk aversion coefficients of the L-moments.

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

# Related

  - [`AbstractOrderedWeightsArrayEstimator`](@ref)
  - [`NormalisedConstantRelativeRiskAversion`](@ref)
  - [`OWAJuMP`](@ref)
  - [`owa_l_moment_crm`](@ref)
  - [`OWA_Func_VecNum`](@ref)

# References

  - $(ref_dict[:owa2])
  - $(ref_dict[:cajas2025]) Section 3.1.5.
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
        @argcheck(2 <= k,
                  DomainError(k,
                              "`LinearMoment.k` is $k, and the L-moment risk measure spans the orders `2:k`, so it needs at least the second. State an integer greater than or equal to `2`."))
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
    const OWA_Func_VecNum = Union{Func_VecNum, AbstractOrderedWeightsArrayFunction}

The values that a weight slot of [`OrderedWeightsArray`](@ref) or [`OrderedWeightsArrayRange`](@ref) accepts.

A weight vector fixes the sample size. A function or an [`AbstractOrderedWeightsArrayFunction`](@ref) is called at the sample size, so one measure serves a sample of any length. Only a builder can hold a Calibration Rule.

# Related

  - [`Func_VecNum`](@ref)
  - [`AbstractOrderedWeightsArrayFunction`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
"""
const OWA_Func_VecNum = Union{Func_VecNum, AbstractOrderedWeightsArrayFunction}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the JuMP encodings of an OWA risk measure.

The OWA risk sorts the returns, and a convex programme cannot sort its variables. [`ExactOrderedWeightsArray`](@ref) and [`SortingNetworkOrderedWeightsArray`](@ref) encode the sort exactly, at two different costs. [`ApproxOrderedWeightsArray`](@ref) drops the sort and gives an upper bound. The functor of the measure sorts the returns itself and does not read the formulation.

# Related

  - [`ExactOrderedWeightsArray`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)

# References

  - $(ref_dict[:owaog])
  - $(ref_dict[:owa1])
  - $(ref_dict[:owa3])
  - $(ref_dict[:cajas2025]) Section 7.1.5.
"""
abstract type OrderedWeightsArrayFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the exact OWA risk with the linear programme that sorts the sample.

The programme is the dual of the assignment problem that pairs each weight with an order statistic. So it sorts the sample inside the model, and the sort costs `T^2` rows. It adds two vector variables `a` and `b` of length `T` and the `T × T` block of rows ``y_{i} w_{j} \\leq a_{j} + b_{i}``, and the risk is ``\\sum_{t} (a_{t} + b_{t})``. The method of [`set_risk_constraints!`](@ref) for this formulation states the model. For a long sample, [`SortingNetworkOrderedWeightsArray`](@ref) gives the same exact risk at ``O(T \\log^{2} T)`` rows, and [`ApproxOrderedWeightsArray`](@ref) gives an upper bound at a lower cost.

The programme pairs the largest weight with the largest sorted return, so it attains ``\\mathrm{sort}(\\boldsymbol{\\omega})^{\\intercal} \\mathrm{sort}(\\hat{\\boldsymbol{r}})``. This is the OWA risk when, and only when, the weight vector is nondecreasing, which is also the condition for the risk measure to be convex. Every weight builder in this package returns such a vector. The programme sorts a weight vector that is out of order, so the model and the functor then disagree.

# Related

  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:owaog])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.1.5.
"""
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Encodes an upper bound on the OWA risk with one power cone block for each p-norm order in `p`.

It relaxes the dual representation of the OWA risk, and does not sort. It replaces the `T × T` block of [`ExactOrderedWeightsArray`](@ref) by one p-norm row for each entry of `p`. It keeps only the properties of the weight vector that a reordering leaves unchanged: the minimum, the maximum, the sum, and one p-norm for each entry of `p`. Every reordering of the weight vector meets those properties, so the model risk lies at or above the exact OWA risk of the same portfolio. The method of [`set_risk_constraints!`](@ref) for this formulation states the model and its `## Relaxation`.

The bound is exact when the weight vector takes two values only, its minimum and its maximum, as the vectors of [`owa_wr`](@ref) and of [`owa_cvar`](@ref) at an integer ``\\alpha T`` do. The box and the sum then describe the convex hull of the reorderings. For other vectors the gap closes as the weight vector approaches a line. [owa3](@citet) studies the linear case. For the Gini mean difference and the tail Gini over samples of 500 to 10,000 observations, [owa3](@cite) reports the same objective value as the exact formulation, to the printed precision. Measured against the functor at `T = 100`, `N = 8` with the default `p`, the Gini mean difference is 0.06 % high, the tail Gini is 1.7e-5 % high, and the tail Gini range is 0.37 % high. A fourth-order L-moment weight vector, which is not linear, is 4.5 % high. This formulation is the fastest of the three. For a weight vector that is far from a line, prefer an exact formulation: [`SortingNetworkOrderedWeightsArray`](@ref) for a long sample, and [`ExactOrderedWeightsArray`](@ref) for a short one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ApproxOrderedWeightsArray(;
        p::VecNum = Float64[2, 3, 4, 10, 50]
    ) -> ApproxOrderedWeightsArray

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:p_owa])

# Examples

```jldoctest
julia> ApproxOrderedWeightsArray()
ApproxOrderedWeightsArray
  p ┴ Vector{Float64}: [2.0, 3.0, 4.0, 10.0, 50.0]
```

# Related

  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:owa3]) Sections 2.3 and 2.4, Equations 4 and 5.
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.1.5.
"""
@concrete struct ApproxOrderedWeightsArray <: OrderedWeightsArrayFormulation
    """
    $(field_dict[:p_owa])
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
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse an OWA weight vector `w` that the formulation `alg` cannot optimise.

The exact and the approximate formulations accept every weight vector, so this method returns `nothing`. A formulation that needs a property of the vector adds a method that checks it.

# Arguments

  - `alg`: OWA formulation of the risk measure.
  - `w`: OWA weight vector, after the range measure reverses its second vector.

# Returns

  - `nothing`.

# Related

  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
"""
assert_owa_formulation_weights(::OrderedWeightsArrayFormulation, ::VecNum) = nothing
"""
$(DocStringExtensions.TYPEDEF)

Measures risk as a weighted sum of the sorted portfolio returns, the ordered weights array (OWA) risk.

`w` holds the OWA weights, as a vector, a function of the sample size or a weight builder. The weight builders of this library give the Gini mean difference, the conditional value at risk, the tail Gini, the worst realisation and the L-moment convex risk measure. A nondecreasing weight vector gives a convex risk measure. `alg` selects the JuMP encoding, and the functor does not read it.

# Mathematical definition

```math
\\begin{align}
\\mathrm{OWA}_{\\boldsymbol{w}}(\\boldsymbol{x}) &= \\sum_{t=1}^{T} w_{t} x_{(t)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{OWA}_{\\boldsymbol{w}}(\\boldsymbol{x})``: Ordered weights array risk.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\boldsymbol{w}``: OWA weight vector ``T \\times 1``, the field `w` or its value at ``T``.
  - $(math_dict[:x_t_asc])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArray(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::OWA_Func_VecNum = owa_gmd,
        alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()
    ) -> OrderedWeightsArray

Keywords correspond to the struct's fields.

## Validation

  - If `w` is a `VecNum`: `!isempty(w)`. Raises an `IsEmptyError`.
  - If `w` is a `VecNum` and `alg` is a [`SortingNetworkOrderedWeightsArray`](@ref): `issorted(w)`. Raises a `DomainError`. A callable `w` is checked when the model is built.

# Functor

    (r::OrderedWeightsArray)(x::VecNum)

Computes the OWA risk of a portfolio returns vector `x`. A `w` that is a callable is evaluated at `length(x)` first, so the weight vector always matches the sample.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> OrderedWeightsArray()
OrderedWeightsArray
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ typeof(owa_gmd): PortfolioOptimisers.owa_gmd
       alg ┼ ApproxOrderedWeightsArray
           │   p ┴ Vector{Float64}: [2.0, 3.0, 4.0, 10.0, 50.0]
```

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`RiskMeasureSettings`](@ref)

# References

  - $(ref_dict[:owaog])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.1.5.
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
            assert_owa_formulation_weights(alg, w)
        end
        return new{typeof(settings), typeof(w), typeof(alg)}(settings, w, alg)
    end
end
function OrderedWeightsArray(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::OWA_Func_VecNum = owa_gmd,
                             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(settings, w, alg)
end
# Deferrable slots — see `deferred_slots`. The weight builder carries them, so both the
# check and the derived recursion in `resolve_deferred_quantities` reach a builder's own
# `alpha` through `w`. A weight vector and a plain function defer nothing, so the recursion
# is the identity for them.
deferred_slots(r::OrderedWeightsArray) = (; w = r.w)
"""
    const OWA_RevFunc = ComposedFunction{typeof(reverse),
                                         <:AbstractOrderedWeightsArrayFunction}

The reversal of an ordered-weights builder, which is what [`OrderedWeightsArrayRange`](@ref) stores in `w2` under the approximate formulation.

It is a `Function` and it computes no quantity, so it is the one occupant of a weight slot that a reader of the calibration channel must part from a **Calibration Rule**. The name is written once here, because the type is long and three sites read it.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`OWA_CalOccupant`](@ref)
  - [`AbstractOrderedWeightsArrayFunction`](@ref)
"""
const OWA_RevFunc = ComposedFunction{typeof(reverse), <:AbstractOrderedWeightsArrayFunction}
"""
    const OWA_CalOccupant = Union{AbstractOrderedWeightsArrayFunction, OWA_RevFunc}

The occupants of an [`OrderedWeightsArray`](@ref) weight slot that carry a calibration slot of their own: a weight builder, and the reversal of one that the Range container builds.

The slot also takes a weight vector and a caller's own plain function, and neither carries a slot, so [`calibration_slots`](@ref) names neither. Everywhere else in the library, a plain function in a calibration slot is a Calibration Rule. The rule is what [`resolve_calibration_slot`](@ref) calls and what [`assert_calibrated_slots`](@ref) refuses, so a weight builder written as a plain function must not reach either.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
  - [`OWA_RevFunc`](@ref)
  - [`calibration_slots`](@ref)
"""
const OWA_CalOccupant = Union{AbstractOrderedWeightsArrayFunction, OWA_RevFunc}
# A reversed builder is a `Function` and computes no quantity, so it withdraws itself from
# the rule marker — see `is_calibration_rule`.
is_calibration_rule(::OWA_CalOccupant) = false
# Calibration slots — see `calibration_slots`. The container states no quantity of its own;
# it names the child, and the child names its own slots. Only a builder carries a slot, so
# only a builder is named: a weight vector and a caller's own plain function carry none, and
# a plain function standing in a calibration slot IS a rule everywhere else in the library.
# The occupant's type is a type parameter, so the branch folds.
calibration_slots(r::OrderedWeightsArray) = isa(r.w, OWA_CalOccupant) ? (; w = r.w) : (;)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the weight builder in `w` against prior result `pr`, and return an [`OrderedWeightsArray`](@ref) whose builder holds numbers.

The measure carries no propagation tag, so it generates no prior [`factory`](@ref) method of its own and this is the channel the value-level entry points take. The `JuMP` route reaches the same recursion through [`set_risk_constraints!`](@ref), which calls [`resolve_deferred_quantities`](@ref) directly.

A builder holding numbers, a weight vector and a plain function are all carried through unchanged, so the common case allocates nothing.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
  - [`expected_risk`](@ref)
"""
function factory(x::OrderedWeightsArray, pr::AbstractPriorResult, args...; kwargs...)
    return resolve_deferred_quantities(x, pr)
end
function (r::OrderedWeightsArray)(x::VecNum)
    w = isa(r.w, VecNum) ? r.w : r.w(length(x))
    return LinearAlgebra.dot(w, sort(x))
end

export MaximumEntropy, ExponentialConeEntropy, RelativeEntropy, MinimumSquaredDistance,
       MinimumSumSquares, NormalisedConstantRelativeRiskAversion, OWAJuMP, owa_gmd,
       owa_cvar, owa_wcvar, owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_wcvarrg, owa_tgrg,
       owa_l_moment, owa_l_moment_crm, ExactOrderedWeightsArray, ApproxOrderedWeightsArray,
       OrderedWeightsArray, LinearMoment, OrderedWeightsArrayConditionalValueatRisk,
       OrderedWeightsArrayTailGini, OrderedWeightsArrayConditionalValueatRiskRange,
       OrderedWeightsArrayTailGiniRange
public AbstractOrderedWeightsArrayFunction
