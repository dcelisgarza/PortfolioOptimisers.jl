"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the names of the reduced factor axis.

The reduced axis follows the raw order with the dropped factor of every constrained family removed, so a retained factor keeps its raw name and its economic meaning.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `nf::VecStr`: Names of the raw factor axis, of length `fcb.K`.

# Validation

  - `length(nf) == fcb.K`.

# Returns

  - `nf::Vector{String}`: The retained names, in reduced-axis order.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"ind\"], fi = [[2, 3]], di = [2],
                               ratios = reshape([0.5], 1, 1), K = 3);

julia> PortfolioOptimisers.reduce_factor_names(fcb, [\"mkt\", \"ind=a\", \"ind=b\"])
2-element Vector{String}:
 "mkt"
 "ind=a"
```

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`dropped_factor_names`](@ref)
"""
function reduce_factor_names(fcb::FactorFamilyBasis, nf::VecStr)::Vector{String}
    assert_factor_axis_length(length(nf), fcb.K, :nf)
    return [String(nf[i]) for i in retained_factor_indices(fcb)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Map an exposure history onto the reduced factor axis.

# Mathematical definition

For a family that drops member ``k``, each retained member ``j`` becomes

```math
z_{t,j} = x_{t,j} - \\frac{c_t(j)}{c_t(k)} \\, x_{t,k},
```

and a factor outside every constrained family is copied unchanged.

Where:

  - ``x_{t,j}``: raw exposure of the assets to factor ``j`` at observation ``t``.
  - ``c_t(j) / c_t(k)``: the entry of `fcb.ratios` for member ``j``.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `Ms::Arr3Num`: Exposure history on the raw axis, `observations × assets × factors`.

# Validation

  - `size(Ms, 3) == fcb.K`, and `size(Ms, 1)` matches the observation axis of the basis.

# Returns

  - `Ms::Array{<:Real, 3}`: The exposure history on the reduced axis, `observations × assets × reduced factors`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_loadings`](@ref)
"""
function reduce_exposures(fcb::FactorFamilyBasis, Ms::Arr3Num)
    assert_factor_axis_length(size(Ms, 3), fcb.K, :Ms)
    assert_factor_basis_obs(size(Ms, 1), fcb, :Ms)
    Tf = promote_type(float(real(eltype(Ms))), float(real(eltype(fcb.ratios))))
    T, N, _ = size(Ms)
    ret = retained_factor_indices(fcb)
    Y = Array{Tf, 3}(undef, T, N, length(ret))
    for k in eachindex(ret), i in 1:N, t in 1:T
        Y[t, i, k] = Tf(Ms[t, i, ret[k]])
    end
    for j in eachindex(fcb.fnm)
        raw, red, col = family_retained_indices(fcb, j)
        d = fcb.fi[j][fcb.di[j]]
        for p in eachindex(raw), i in 1:N, t in 1:T
            Y[t, i, red[p]] = Tf(Ms[t, i, raw[p]]) -
                              Tf(fcb.ratios[t, col[p]]) * Tf(Ms[t, i, d])
        end
    end
    return Y
end
"""
    reduce_loadings(fcb::FactorFamilyBasis, M::MatNum, t::Integer = size(fcb.ratios, 1))

Map a point-in-time loading matrix onto the reduced factor axis.

This is [`reduce_exposures`](@ref) at one observation, so it applies the ratios of observation `t` to a matrix of assets by raw factors.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `M::MatNum`: Loading matrix on the raw axis, `assets × factors`.
  - `t::Integer`: Observation whose ratios are applied. It defaults to the last observation of the basis.

# Validation

  - `size(M, 2) == fcb.K`, and `t` indexes the observation axis of the basis.

# Returns

  - `L::Matrix{<:Real}`: The loading matrix on the reduced axis, `assets × reduced factors`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_exposures`](@ref)
"""
function reduce_loadings(fcb::FactorFamilyBasis, M::MatNum,
                         t::Integer = size(fcb.ratios, 1))
    assert_factor_axis_length(size(M, 2), fcb.K, :M)
    assert_factor_basis_index(t, fcb)
    Tf = promote_type(float(real(eltype(M))), float(real(eltype(fcb.ratios))))
    N = size(M, 1)
    ret = retained_factor_indices(fcb)
    L = Matrix{Tf}(undef, N, length(ret))
    for k in eachindex(ret), i in 1:N
        L[i, k] = Tf(M[i, ret[k]])
    end
    for j in eachindex(fcb.fnm)
        raw, red, col = family_retained_indices(fcb, j)
        d = fcb.fi[j][fcb.di[j]]
        for p in eachindex(raw), i in 1:N
            L[i, red[p]] = Tf(M[i, raw[p]]) - Tf(fcb.ratios[t, col[p]]) * Tf(M[i, d])
        end
    end
    return L
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse an observation index that does not lie on the observation axis of the basis.

# Arguments

  - `t::Integer`: Observation index.
  - `fcb`: A Factor Family Basis.

# Validation

  - `1 <= t <= size(fcb.ratios, 1)`.

# Returns

  - `nothing`.

# Related

  - [`FactorFamilyBasis`](@ref)
"""
function assert_factor_basis_index(t::Integer, fcb::FactorFamilyBasis)::Nothing
    T = size(fcb.ratios, 1)
    @argcheck(1 <= t <= T,
              DomainError(t,
                          "the observation index must lie in 1:$T, the observation axis of the basis"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Drop the redundant factor returns, giving the reduced-axis factor returns.

Factor returns are coordinates in factor-return space, so the reduction keeps the retained columns and applies no ratio. It is the inverse of [`expand_factor_returns`](@ref).

# Arguments

  - `fcb`: A Factor Family Basis.
  - `f::VecNum_MatNum`: Factor returns on the raw axis, either one observation per row or one observation alone.

# Validation

  - The factor axis of `f` is `fcb.K`.

# Returns

  - `f::Array{<:Real}`: The factor returns on the reduced axis, of the same number of dimensions as the input.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`expand_factor_returns`](@ref)
"""
function reduce_factor_returns(fcb::FactorFamilyBasis, f::MatNum)
    assert_factor_axis_length(size(f, 2), fcb.K, :f)
    return f[:, retained_factor_indices(fcb)]
end
function reduce_factor_returns(fcb::FactorFamilyBasis, f::VecNum)
    assert_factor_axis_length(length(f), fcb.K, :f)
    return f[retained_factor_indices(fcb)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Drop the redundant entries of a factor mean, giving the reduced-axis mean.

It is the inverse of [`expand_factor_mu`](@ref), and it applies no ratio.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `mu::VecNum`: Factor mean on the raw axis.

# Validation

  - `length(mu) == fcb.K`.

# Returns

  - `mu::Vector{<:Real}`: The factor mean on the reduced axis.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`expand_factor_mu`](@ref)
"""
function reduce_factor_mu(fcb::FactorFamilyBasis, mu::VecNum)
    assert_factor_axis_length(length(mu), fcb.K, :mu)
    return mu[retained_factor_indices(fcb)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Take the full-rank block of a factor covariance, giving the reduced-axis covariance.

The reduced factor returns are the retained raw ones, so the reduced covariance is the submatrix of the retained indices and no ratio is applied. It is the inverse of [`expand_factor_covariance`](@ref).

# Arguments

  - `fcb`: A Factor Family Basis.
  - `sigma::MatNum`: Factor covariance on the raw axis, `factors × factors`.

# Validation

  - `size(sigma) == (fcb.K, fcb.K)`.

# Returns

  - `sigma::Matrix{<:Real}`: The factor covariance on the reduced axis.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`expand_factor_covariance`](@ref)
"""
function reduce_factor_covariance(fcb::FactorFamilyBasis, sigma::MatNum)
    assert_factor_axis_length(size(sigma, 1), fcb.K, :sigma)
    assert_factor_axis_length(size(sigma, 2), fcb.K, :sigma)
    ret = retained_factor_indices(fcb)
    return sigma[ret, ret]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the reduced-axis weights that reconstruct the dropped factors at one observation.

Row `j` holds the coefficients of the zero-sum condition of family `j`, so the dropped factor of that family is the row applied to a reduced-axis quantity. The dense change of basis is never formed.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `t::Integer`: Observation whose ratios are read.

# Validation

  - `t` indexes the observation axis of the basis.

# Returns

  - `W::Matrix{<:Real}`: The reconstruction weights, `constrained families × reduced factors`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`expand_factor_covariance`](@ref)
"""
function dropped_factor_weights(fcb::FactorFamilyBasis, t::Integer)
    assert_factor_basis_index(t, fcb)
    Tf = float(real(eltype(fcb.ratios)))
    W = zeros(Tf, length(fcb.fnm), reduced_factor_count(fcb))
    for j in eachindex(fcb.fnm)
        _, red, col = family_retained_indices(fcb, j)
        for p in eachindex(red)
            W[j, red[p]] = -Tf(fcb.ratios[t, col[p]])
        end
    end
    return W
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reconstruct the raw-axis factor returns from the reduced-axis ones.

# Mathematical definition

The retained returns pass through, and the dropped return of each family follows from the zero-sum condition:

```math
f_k(t) = -\\sum_{j \\ne k} \\frac{c_t(j)}{c_t(k)} \\, g_j(t).
```

Where:

  - ``g_j(t)``: reduced-axis factor return of retained member ``j`` at observation ``t``.
  - ``f_k(t)``: raw-axis factor return of the dropped member ``k``.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `g::VecNum_MatNum`: Factor returns on the reduced axis. A matrix carries one observation per row and expands each row with that row's ratios. A vector expands with the ratios of the last observation.

# Validation

  - The factor axis of `g` is the reduced factor count, and a matrix matches the observation axis of the basis.

# Returns

  - `f::Array{<:Real}`: The factor returns on the raw axis, of the same number of dimensions as the input.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_factor_returns`](@ref)
"""
function expand_factor_returns(fcb::FactorFamilyBasis, g::MatNum)
    assert_factor_axis_length(size(g, 2), reduced_factor_count(fcb), :g)
    assert_factor_basis_obs(size(g, 1), fcb, :g)
    Tf = promote_type(float(real(eltype(g))), float(real(eltype(fcb.ratios))))
    T = size(g, 1)
    f = zeros(Tf, T, fcb.K)
    ret = retained_factor_indices(fcb)
    for k in eachindex(ret), t in 1:T
        f[t, ret[k]] = Tf(g[t, k])
    end
    for j in eachindex(fcb.fnm)
        _, red, col = family_retained_indices(fcb, j)
        d = fcb.fi[j][fcb.di[j]]
        for t in 1:T
            s = zero(Tf)
            for p in eachindex(red)
                s += Tf(fcb.ratios[t, col[p]]) * Tf(g[t, red[p]])
            end
            f[t, d] = -s
        end
    end
    return f
end
function expand_factor_returns(fcb::FactorFamilyBasis, g::VecNum)
    return expand_factor_mu(fcb, g, size(fcb.ratios, 1))
end
"""
    expand_factor_mu(fcb::FactorFamilyBasis, mu::VecNum, t::Integer = size(fcb.ratios, 1))

Reconstruct the raw-axis factor mean from the reduced-axis one.

The retained entries pass through, and the dropped entry of each family is the zero-sum reconstruction at observation `t`. It is the inverse of [`reduce_factor_mu`](@ref).

# Arguments

  - `fcb`: A Factor Family Basis.
  - `mu::VecNum`: Factor mean on the reduced axis.
  - `t::Integer`: Observation whose ratios are applied. It defaults to the last observation of the basis.

# Validation

  - `length(mu)` is the reduced factor count, and `t` indexes the observation axis of the basis.

# Returns

  - `mu::Vector{<:Real}`: The factor mean on the raw axis.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_factor_mu`](@ref)
"""
function expand_factor_mu(fcb::FactorFamilyBasis, mu::VecNum,
                          t::Integer = size(fcb.ratios, 1))
    assert_factor_axis_length(length(mu), reduced_factor_count(fcb), :mu)
    assert_factor_basis_index(t, fcb)
    Tf = promote_type(float(real(eltype(mu))), float(real(eltype(fcb.ratios))))
    f = zeros(Tf, fcb.K)
    ret = retained_factor_indices(fcb)
    for k in eachindex(ret)
        f[ret[k]] = Tf(mu[k])
    end
    for j in eachindex(fcb.fnm)
        _, red, col = family_retained_indices(fcb, j)
        d = fcb.fi[j][fcb.di[j]]
        s = zero(Tf)
        for p in eachindex(red)
            s += Tf(fcb.ratios[t, col[p]]) * Tf(mu[red[p]])
        end
        f[d] = -s
    end
    return f
end
"""
    expand_factor_covariance(fcb::FactorFamilyBasis, sigma::MatNum,
                             t::Integer = size(fcb.ratios, 1))

Reconstruct the raw-axis factor covariance from the reduced-axis one.

# Mathematical definition

```math
\\Sigma = R_t \\, \\Sigma^{\\mathrm{red}} \\, R_t^{\\top}.
```

Where:

  - ``R_t``: the change of basis at observation ``t``, which is never formed. The retained block is copied, and the dropped rows and columns come from the reconstruction weights of [`dropped_factor_weights`](@ref).

The answer is singular by construction, because the raw axis is a linear image of a smaller one.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `sigma::MatNum`: Factor covariance on the reduced axis, `reduced factors × reduced factors`.
  - `t::Integer`: Observation whose ratios are applied. It defaults to the last observation of the basis.

# Validation

  - `size(sigma)` is the reduced factor count on both axes, and `t` indexes the observation axis of the basis.

# Returns

  - `sigma::Matrix{<:Real}`: The factor covariance on the raw axis.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_factor_covariance`](@ref)
  - [`dropped_factor_weights`](@ref)
"""
function expand_factor_covariance(fcb::FactorFamilyBasis, sigma::MatNum,
                                  t::Integer = size(fcb.ratios, 1))
    Kr = reduced_factor_count(fcb)
    assert_factor_axis_length(size(sigma, 1), Kr, :sigma)
    assert_factor_axis_length(size(sigma, 2), Kr, :sigma)
    W = dropped_factor_weights(fcb, t)
    Tf = promote_type(float(real(eltype(sigma))), eltype(W))
    ret = retained_factor_indices(fcb)
    drp = dropped_factor_indices(fcb)
    DR = W * sigma
    DD = DR * transpose(W)
    S = zeros(Tf, fcb.K, fcb.K)
    for b in eachindex(ret), a in eachindex(ret)
        S[ret[a], ret[b]] = Tf(sigma[a, b])
    end
    for b in eachindex(ret), a in eachindex(drp)
        S[drp[a], ret[b]] = Tf(DR[a, b])
        S[ret[b], drp[a]] = Tf(DR[a, b])
    end
    for b in eachindex(drp), a in eachindex(drp)
        S[drp[a], drp[b]] = Tf(DD[a, b])
    end
    return S
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Project raw factor-space coordinates into the reduced basis.

This applies the transpose of the change of basis, so it is not a column selection: the coordinate of a dropped factor contributes to the retained coordinates of its family. A portfolio's factor exposure is such a coordinate.

# Mathematical definition

```math
y_j = x_j - \\frac{c_t(j)}{c_t(k)} \\, x_k, \\qquad j \\ne k.
```

Where:

  - ``x_j``: raw coordinate of retained member ``j``.
  - ``x_k``: raw coordinate of the dropped member ``k``.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `x::VecNum_MatNum`: Coordinates on the raw axis. A matrix carries one observation per row and projects each row with that row's ratios. A vector projects with the ratios of the last observation.

# Validation

  - The factor axis of `x` is `fcb.K`, and a matrix matches the observation axis of the basis.

# Returns

  - `y::Array{<:Real}`: The coordinates on the reduced axis, of the same number of dimensions as the input.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_factor_returns`](@ref)
"""
function project_factor_coordinates(fcb::FactorFamilyBasis, x::MatNum)
    assert_factor_axis_length(size(x, 2), fcb.K, :x)
    assert_factor_basis_obs(size(x, 1), fcb, :x)
    Tf = promote_type(float(real(eltype(x))), float(real(eltype(fcb.ratios))))
    T = size(x, 1)
    ret = retained_factor_indices(fcb)
    y = Matrix{Tf}(undef, T, length(ret))
    for k in eachindex(ret), t in 1:T
        y[t, k] = Tf(x[t, ret[k]])
    end
    for j in eachindex(fcb.fnm)
        raw, red, col = family_retained_indices(fcb, j)
        d = fcb.fi[j][fcb.di[j]]
        for p in eachindex(raw), t in 1:T
            y[t, red[p]] = Tf(x[t, raw[p]]) - Tf(fcb.ratios[t, col[p]]) * Tf(x[t, d])
        end
    end
    return y
end
function project_factor_coordinates(fcb::FactorFamilyBasis, x::VecNum)
    assert_factor_axis_length(length(x), fcb.K, :x)
    t = size(fcb.ratios, 1)
    Tf = promote_type(float(real(eltype(x))), float(real(eltype(fcb.ratios))))
    ret = retained_factor_indices(fcb)
    y = Vector{Tf}(undef, length(ret))
    for k in eachindex(ret)
        y[k] = Tf(x[ret[k]])
    end
    for j in eachindex(fcb.fnm)
        raw, red, col = family_retained_indices(fcb, j)
        d = fcb.fi[j][fcb.di[j]]
        for p in eachindex(raw)
            y[red[p]] = Tf(x[raw[p]]) - Tf(fcb.ratios[t, col[p]]) * Tf(x[d])
        end
    end
    return y
end
