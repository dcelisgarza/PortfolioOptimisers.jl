"""
    cross_sectional_systematic(f::MatNum, b::Option{<:VecNum}, Z::Arr3Num) -> MatNum

Return the systematic part of a cross-sectional regression, `observations × assets`.

# Mathematical definition

```math
\\begin{align}
\\hat{x}_{t,i} &= b_{t} + \\boldsymbol{z}_{t,i}^{\\intercal} \\boldsymbol{f}_{t}\\,.
\\end{align}
```

Where:

  - ``\\hat{x}_{t,i}``: Systematic return of asset ``i`` at observation ``t``.
  - ``\\boldsymbol{z}_{t,i}``: Exposures of asset ``i`` at observation ``t``.
  - ``\\boldsymbol{f}_{t}``: Factor returns of observation ``t``.
  - ``b_{t}``: Intercept of observation ``t``. The term is zero when no intercept was fitted.

# Arguments

  - `f::MatNum`: Factor returns matrix `observations × factors`.
  - `b::Option{<:VecNum}`: Intercept vector, or `nothing` when none was fitted.
  - `Z::Arr3Num`: Exposure tensor `assets × factors × observations`. The asset axis may differ from the one the fit saw. The observation axis and the factor axis must match `f`.

# Validation

  - `size(Z, 3) == size(f, 1)` and `size(Z, 2) == size(f, 2)`.

# Returns

  - `Xh::MatNum`: Systematic returns, `observations × assets`.

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_systematic(f::MatNum, b::Option{<:VecNum}, Z::Arr3Num)::MatNum
    @argcheck(size(Z, 3) == size(f, 1) && size(Z, 2) == size(f, 2),
              DimensionMismatch("Z ($(size(Z, 1))×$(size(Z, 2))×$(size(Z, 3))) must match f ($(size(f, 1))×$(size(f, 2))) on the factor and observation axes"))
    Xh = similar(f, size(Z, 3), size(Z, 1))
    for t in axes(Z, 3), i in axes(Z, 1)
        Xh[t, i] = LinearAlgebra.dot(view(Z, i, :, t), view(f, t, :))
    end
    return isnothing(b) ? Xh : Xh .+ b
end
"""
    cross_sectional_residuals(X::MatNum, f::MatNum, b::Option{<:VecNum}, Z::Arr3Num) -> MatNum

Return the residuals of a cross-sectional regression, `observations × assets`: the returns `X` less the systematic part of [`cross_sectional_systematic`](@ref), in one pass.

Each element takes the roundings of `X - cross_sectional_systematic(f, b, Z)` in the same order, so the two give the same bits. The pass allocates the residuals alone, where the two-step form also allocates the systematic part and, with an intercept, its shifted copy.

# Algorithm

 1. For each observation `t` and asset `i`, take the dot product of the exposures `Z[i, :, t]` and the factor returns `f[t, :]`, in the element type of `f`.
 2. Add the intercept `b[t]` when one was fitted.
 3. Subtract the sum from `X[t, i]`.

# Arguments

  - `X::MatNum`: Returns matrix `observations × assets`.
  - `f::MatNum`: Factor returns matrix `observations × factors`.
  - `b::Option{<:VecNum}`: Intercept vector, or `nothing` when none was fitted.
  - `Z::Arr3Num`: Exposure tensor `assets × factors × observations`.

# Validation

  - The rules of [`cross_sectional_systematic`](@ref), and `size(X) == (size(Z, 3), size(Z, 1))`.

# Returns

  - `eps::MatNum`: Residuals, `observations × assets`.

# Related

  - [`cross_sectional_systematic`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_residuals(X::MatNum, f::MatNum, b::Option{<:VecNum},
                                   Z::Arr3Num)::MatNum
    # `f` is `observations × factors` and `X` is `observations × assets`.
    @argcheck((size(f), size(X)) == ((size(Z, 3), size(Z, 2)), (size(Z, 3), size(Z, 1))),
              DimensionMismatch("X ($(size(X, 1))×$(size(X, 2))), f ($(size(f, 1))×$(size(f, 2))) and Z ($(size(Z, 1))×$(size(Z, 2))×$(size(Z, 3))) must agree on the asset, factor and observation axes"))
    # The element types of the two-step form: the systematic part is stored in the type of
    # `f`, the intercept promotes it, and the difference promotes again.
    Ts = isnothing(b) ? eltype(f) : Base.promote_op(+, eltype(f), eltype(b))
    eps = similar(X, Base.promote_op(-, eltype(X), Ts))
    for t in axes(Z, 3), i in axes(Z, 1)
        s = convert(eltype(f), LinearAlgebra.dot(view(Z, i, :, t), view(f, t, :)))
        eps[t, i] = X[t, i] - (isnothing(b) ? s : s + b[t])
    end
    return eps
end
"""
    StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num) -> MatNum

Return the systematic part of a fitted cross-sectional regression, `observations × assets`.

For the `X` that the fit saw, the residuals of the result are `X - predict(csr, Z)` already. The method gives the systematic part on an exposure tensor that the fit did not see.

# Algorithm

 1. Call [`cross_sectional_systematic`](@ref) with `csr.f`, `csr.b` and `Z`.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `assets × factors × observations`. The asset axis may differ from the one the fit saw. The observation axis and the factor axis must match `csr.f`.

# Validation

  - The rules of [`cross_sectional_systematic`](@ref).

# Returns

  - `Xh::MatNum`: Systematic returns, `observations × assets`.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 3, 2, 1);

julia> csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5],
                                        ones(1, 3));

julia> predict(csr, Z)
1×3 Matrix{Float64}:
 1.0  2.0  1.5
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_systematic`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_r2`](@ref)
"""
function StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)::MatNum
    return cross_sectional_systematic(csr.f, csr.b, Z)
end
