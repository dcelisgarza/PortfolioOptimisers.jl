"""
    kappa_log(u::Number, kappa::Number)

Evaluate the Kaniadakis logarithm of [kaniadakis2001](@cite), the inverse of the κ-deformed exponential.

[rlvar](@cite) defines the relativistic value at risk with this logarithm. [`RRM`](@ref), [`set_relativistic_risk_constraints!`](@ref) and the relativistic value at risk views of entropy pooling multiply a dual variable by `kappa_log(inv(alpha * T), kappa)`, where `alpha` is the significance level and `T` is the sample size. [`EntropyBudget`](@ref) finds the `kappa` that gives a target value of this coefficient.

The function checks neither argument. Each relativistic risk measure and each relativistic value at risk view runs [`assert_unit_interval`](@ref) on `kappa` in its constructor. When a calibration rule gives `kappa`, `rebuild_with_slots` runs the same constructor on the new value. [`EntropyBudget`](@ref) passes the midpoints of a bracket inside ``(0, 1)``. Every caller passes ``u = 1 / (\\alpha T)``, which is positive.

# Mathematical definition

```math
\\begin{align}
\\ln_{\\kappa}(u) &= \\dfrac{u^{\\kappa} - u^{-\\kappa}}{2 \\kappa}\\,.
\\end{align}
```

Where:

  - $(math_dict[:ln_kappa])
  - ``u > 0``: Argument of the logarithm.
  - $(math_dict[:kappa_rm])

The value is negative for every ``u < 1``, and ``\\ln_{\\kappa}(u) \\to \\ln(u)`` as ``\\kappa \\to 0``.

# Arguments

  - `u`: Argument of the logarithm. It must be positive.
  - `kappa`: Kaniadakis deformation parameter. It must lie in ``(0, 1)``.

# Returns

  - `lnk::Number`: Value of the Kaniadakis logarithm.

# Examples

```jldoctest
julia> PortfolioOptimisers.kappa_log(2, 0.3)
0.6981533616478014

julia> PortfolioOptimisers.kappa_log(0.5, 0.3)
-0.6981533616478014
```

# Related

  - [`RRM`](@ref)
  - [`set_relativistic_risk_constraints!`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`EntropyBudget`](@ref)
  - [`assert_unit_interval`](@ref): the check that the constructors run on `kappa`.

# References

  - $(ref_dict[:kaniadakis2001]) Section VI, Equation 63.
  - $(ref_dict[:rlvar]) Section 2.2, Equation 5.
"""
function kappa_log(u::Number, kappa::Number)
    return (u^kappa - u^(-kappa)) / (2 * kappa)
end
