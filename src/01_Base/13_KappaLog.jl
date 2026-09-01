"""
    kappa_log(u::Number, kappa::Number)

Evaluate the Kaniadakis logarithm.

The relativistic risk measures, their JuMP constraint layer and the entropy pooling views of the relativistic value at risk all scale a dual variable by this quantity, so the library states it once here. The function checks neither argument, because every caller holds `kappa` in a field that its constructor has already passed to [`assert_unit_interval`](@ref), and every caller passes the reciprocal of a tail mass, which is positive.

# Mathematical definition

```math
\\begin{align}
\\ln_{\\kappa}(u) &= \\dfrac{u^{\\kappa} - u^{-\\kappa}}{2 \\kappa}\\,.
\\end{align}
```

Where:

  - $(math_dict[:ln_kappa])
  - ``u > 0``: Argument of the logarithm.
  - ``\\kappa \\in (0, 1)``: Deformation parameter.

The value is negative for every ``u < 1``, and ``\\ln_{\\kappa}(u) \\to \\ln(u)`` as ``\\kappa \\to 0``.

# Arguments

  - `u`: Argument of the logarithm. Must be positive.
  - $(arg_dict[:kappa]) Must lie in ``(0, 1)``.

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

  - [`RRM`](@ref): every relativistic risk measure reaches the logarithm through it.
  - [`RelativisticValueatRisk`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`assert_unit_interval`](@ref): the check that every caller's constructor runs on `kappa`.

# References

  - $(ref_dict[:rlvar])
"""
function kappa_log(u::Number, kappa::Number)
    return (u^kappa - u^(-kappa)) / (2 * kappa)
end
