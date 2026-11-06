"""
$(DocStringExtensions.TYPEDEF)

Computes the expected excess returns that a set of equilibrium weights implies, by reverse optimisation.

It holds a covariance estimator, the equilibrium weights and the risk aversion parameter. [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) build their prior mean with the same expression when their `l` is set.

The equilibrium risk premium vector ``\\mathbf{\\Pi} = \\delta \\mathbf{\\Sigma} \\mathbf{W}`` is step 6 of the Appendix of [black1991](@cite) and of [black1992](@cite), with the market weights ``\\mathbf{W}``. [sharpe1974](@cite) derives it earlier. It maximises ``E_p - \\lambda_S V_p`` under full investment, sets the gradient to zero, and finds the implied excess return ``E_i - \\pi = 2 \\lambda_S (\\mathbf{\\Sigma} \\boldsymbol{x})_i`` of every asset held, where ``\\pi`` is the riskless return. It also finds ``\\lambda_S = (E_p - \\pi) / (2 V_p)``. Equations 5.1 and 5.2 of [cajas2025](@cite) derive the same result by setting the gradient of the quadratic utility ``\\boldsymbol{x}^\\intercal \\boldsymbol{\\Pi} - \\frac{\\delta}{2} \\boldsymbol{x}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{x}`` to zero, so ``\\delta = 2 \\lambda_S``.

`l` is the risk aversion of the representative investor. It is the ``\\lambda`` of the equation below, which [black1991](@cite) and [black1992](@cite) write ``\\delta``. Both say that ``\\delta`` is a proportionality constant based on the formulas of [black1989universal](@cite). That work gives the universal hedging constant, on page 18, and no formula for ``\\delta``. This library takes `l` from the caller, and does not estimate it from the data. The result is linear in `l`. Twice the `l` gives twice every return, so a negative return moves down as `l` grows. An `l` of zero gives a zero mean.

`w` is the weight vector that the model assumes the market holds, such as the market capitalisation weights or the weights of a benchmark. The caller supplies it too, because the data hold returns and no holdings. If `w` is `nothing`, [`equilibrium_mu`](@ref) uses equal weights. It reads their number from the columns of the covariance matrix, not from `X`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EquilibriumExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        w::Option{<:VecNum} = nothing,
        l::Number = 1
    ) -> EquilibriumExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ce_cov])
  - $(val_dict[:oow])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> EquilibriumExpectedReturns()
EquilibriumExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┼ FullMoment()
     │      │     w ┴ nothing
     │   mp ┼ MatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
   w ┼ nothing
   l ┴ Int64: 1
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`equilibrium_mu`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:sharpe1974]) Sections II and III, pages 465 to 467.
  - $(ref_dict[:black1991]) Appendix, step 6.
  - $(ref_dict[:black1992]) Appendix, step 6, page 42.
  - $(ref_dict[:black1989universal]) The Universal Hedging Formula, page 18.
  - $(ref_dict[:cajas2025]) Section 5.1.1, Equations 5.1 and 5.2.
"""
@propagatable @concrete struct EquilibriumExpectedReturns <:
                               AbstractExpectedReturnsEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:eqw])
    """
    @vprop w
    """
    $(field_dict[:l])
    """
    l
    function EquilibriumExpectedReturns(ce::StatsBase.CovarianceEstimator,
                                        w::Option{<:VecNum}, l::Number)
        assert_covariance_estimator(ce, :ce)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(ce), typeof(w), typeof(l)}(ce, w, l)
    end
end
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    w::Option{<:VecNum} = nothing,
                                    l::Number = 1)::EquilibriumExpectedReturns
    return EquilibriumExpectedReturns(ce, w, l)
end
"""
    equilibrium_mu(l::Number, sigma::MatNum, w::Option{<:VecNum})

Compute equilibrium expected returns from a risk aversion parameter, a covariance block, and equilibrium weights.

This function is the one place in the library that computes ``\\lambda \\mathbf{\\Sigma} \\boldsymbol{w}`` and its equal-weight fallback. [`EquilibriumExpectedReturns`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) all call it, so the three share one fallback and one length check.

The result is an excess return. Reverse optimisation implies a risk premium, so the result holds no risk-free rate. For this reason the two Black-Litterman priors add the rate to the result of this function with [`apply_rf`](@ref). They add it only when they call this function. A mean that they take from a wrapped prior estimator is a total return already.

`sigma` is a covariance block, which can be rectangular. Its columns are the assets of the weights, so `w` must have the length `size(sigma, 2)`. A square covariance matrix gives the equilibrium returns of the assets. A rectangular block gives one equilibrium return for each of its rows. [`FactorBlackLittermanPrior`](@ref) passes a `factors × assets` block, and gets a prior mean of the factors from asset weights. [`AugmentedBlackLittermanPrior`](@ref) puts the asset covariance matrix on top of that block, and gets the assets and the factors in one vector.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{\\text{eq}} &= \\lambda \\, \\mathbf{\\Sigma} \\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\lambda``: The risk aversion parameter.
  - ``\\mathbf{\\Sigma}``: The ``M \\times N`` covariance block.
  - ``\\boldsymbol{w}``: The ``N \\times 1`` equilibrium portfolio weights.

# Algorithm

 1. When `w` is `nothing`, read the asset count `N` from `size(sigma, 2)`, and build the equal-weight vector `fill(inv(N), N)`.
 2. When `w` is a vector, check that `length(w)` equals `size(sigma, 2)`.
 3. Apply the expression above to `l`, `sigma` and the weights of step 1 or step 2. The result is `mu`.

# Arguments

  - `l`: The risk aversion parameter.
  - `sigma`: The covariance block. Its columns are the assets.
  - `w`: The equilibrium weights, or `nothing` for equal weights.

# Validation

  - If `w` is a vector, `length(w) == size(sigma, 2)`.

# Returns

  - `mu::VecNum`: The equilibrium expected returns, a vector of length `size(sigma, 1)`.

# Related

  - [`EquilibriumExpectedReturns`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
"""
function equilibrium_mu(l::Number, sigma::MatNum, w::Nothing)
    N = size(sigma, 2)
    return l * sigma * fill(inv(N), N)
end
function equilibrium_mu(l::Number, sigma::MatNum, w::VecNum)
    @argcheck(length(w) == size(sigma, 2),
              DimensionMismatch("length(w) ($(length(w))) must match the number of assets, size(sigma, 2) ($(size(sigma, 2)))"))
    return l * sigma * w
end
"""
    Statistics.mean(me::EquilibriumExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute equilibrium expected returns from a covariance estimator, weights, and risk aversion.

The method returns `λ * Σ * w`. `λ` is the risk aversion parameter `me.l`, `Σ` is the covariance matrix that `me.ce` estimates from `X`, and `w` is the vector of equilibrium weights `me.w`. [`equilibrium_mu`](@ref) computes the expression, and uses equal weights if `me.w` is `nothing`.

The result is an excess return. Reverse optimisation implies a risk premium, so the result holds no risk-free rate, and the method subtracts none.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{\\text{eq}} &= \\lambda \\, \\hat{\\mathbf{\\Sigma}} \\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\lambda``: The risk aversion parameter `me.l`.
  - ``\\hat{\\mathbf{\\Sigma}}``: The `N × N` covariance matrix that `me.ce` estimates from the data.
  - ``\\boldsymbol{w}``: The `N × 1` equilibrium portfolio weights `me.w`, or equal weights if `me.w` is `nothing`.

# Algorithm

 1. Estimate the covariance matrix of `X` with `me.ce`, through [`library_covariance_estimator`](@ref). The result is `sigma`.
 2. Pass `me.l`, `sigma` and `me.w` to [`equilibrium_mu`](@ref). It selects the equal-weight fallback when `me.w` is `nothing`. Its result is `mu`.

# Arguments

  - `me`: The equilibrium expected returns estimator.
  - `X`: The data matrix, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method passes to the covariance estimator.

# Validation

  - $(val_dict[:dims]) This method does not check `dims`. The covariance estimator `me.ce` raises the `DomainError`.

# Returns

  - `mu::VecNum`: The equilibrium expected returns, a vector of length `N`. The method returns a vector for both values of `dims`, where most expected returns estimators return a `(1, N)` or an `(N, 1)` matrix. [`equilibrium_mu`](@ref) multiplies the covariance matrix by the weight vector, and the product is a vector. `dims` reaches the covariance estimator alone, and the covariance matrix is `N × N` for both values of `dims`.

# Related

  - [`EquilibriumExpectedReturns`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function Statistics.mean(me::EquilibriumExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    sigma = Statistics.cov(library_covariance_estimator(me.ce), X; dims = dims, kwargs...)
    return equilibrium_mu(me.l, sigma, me.w)
end
# The mean is a linear map of the covariance of `me.ce`, so it is worth the count of that
# covariance.
function effective_observation_count(me::EquilibriumExpectedReturns, X::MatNum; kwargs...)
    return effective_observation_count(me.ce, X; kwargs...)
end

export EquilibriumExpectedReturns
