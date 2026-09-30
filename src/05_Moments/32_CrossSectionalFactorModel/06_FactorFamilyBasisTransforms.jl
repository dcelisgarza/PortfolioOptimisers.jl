"""
    reduce_factor_names(fcb::FactorFamilyBasis, nf::VecStr)

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
    reduce_exposures(fcb::FactorFamilyBasis, Ms::Arr3Num)

Map an exposure history onto the reduced factor axis.

The function applies the ratios of each observation to the exposures of that observation.

# Mathematical definition

```math
\\begin{align}
\\mathbf{B}^{\\mathrm{red}}_{t} &= \\mathbf{B}_{t} \\mathbf{R}_{t}\\,, \\\\
\\mathbf{B}^{\\mathrm{red}}_{t} \\boldsymbol{f}^{\\mathrm{red}}_{t} &= \\mathbf{B}_{t} \\boldsymbol{f}^{\\mathrm{raw}}_{t} \\quad \\text{when} \\quad \\boldsymbol{f}^{\\mathrm{raw}}_{t} = \\mathbf{R}_{t} \\boldsymbol{f}^{\\mathrm{red}}_{t}\\,.
\\end{align}
```

The column of a retained member ``j`` of a family that drops ``k`` is the column of ``j`` less ``r_{t}(j)`` times the column of ``k``, and a factor outside every constrained family keeps its column. The second line follows from the first, so the two bases give the same fitted values.

Where:

  - $(math_dict[:B_t_att])
  - ``\\mathbf{B}^{\\mathrm{red}}_{t}``: Exposure slice of observation ``t`` on the reduced axis, ``N \\times K_{r}``.
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])
  - $(math_dict[:f_t_fcb])

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
    Tf = promote_type(real(eltype(Ms)), real(eltype(fcb.ratios)))
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
    reduce_loadings(fcb::FactorFamilyBasis, M::Arr3Num)

Map a point-in-time loading matrix onto the reduced factor axis.

This is [`reduce_exposures`](@ref) at one observation. It applies the ratios of observation `t` to a matrix of assets by raw factors. A stack of loading matrices, one slice per observation of the basis, is an exposure history, and the function reduces it with [`reduce_exposures`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathbf{L} &= \\mathbf{M} \\mathbf{R}_{t}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}``: Loading matrix on the raw axis, ``N \\times K``.
  - ``\\mathbf{L}``: Loading matrix on the reduced axis, ``N \\times K_{r}``.
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])

# Arguments

  - `fcb`: A Factor Family Basis.
  - `M::MatNum_Arr3Num`: Loading matrix on the raw axis, `assets × factors`, or a stack of them, `observations × assets × factors`.
  - `t::Integer`: Observation whose ratios the function applies to a matrix. It defaults to the last observation of the basis.

# Validation

  - The factor axis of `M` is `fcb.K`, `t` indexes the observation axis of the basis, and a stack matches that axis.

# Returns

  - `L::Array{<:Real}`: The loading matrix on the reduced axis, `assets × reduced factors`, or the stack of them.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_exposures`](@ref)
"""
function reduce_loadings(fcb::FactorFamilyBasis, M::MatNum,
                         t::Integer = size(fcb.ratios, 1))
    assert_factor_axis_length(size(M, 2), fcb.K, :M)
    assert_factor_basis_index(t, fcb)
    Tf = promote_type(real(eltype(M)), real(eltype(fcb.ratios)))
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
function reduce_loadings(fcb::FactorFamilyBasis, M::Arr3Num)
    return reduce_exposures(fcb, M)
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
    reduce_factor_returns(fcb::FactorFamilyBasis, f::MatNum)
    reduce_factor_returns(fcb::FactorFamilyBasis, f::VecNum)

Drop the redundant factor returns, giving the reduced-axis factor returns.

The reduction keeps the retained columns and applies no ratio. It undoes [`expand_factor_returns`](@ref), and the converse holds only for raw returns that satisfy the zero-sum condition of every constrained Factor Family.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{f}^{\\mathrm{red}}_{t} &= \\mathbf{S}^{\\intercal} \\boldsymbol{f}^{\\mathrm{raw}}_{t}\\,, \\\\
\\mathbf{S}^{\\intercal} \\mathbf{R}_{t} &= \\mathbf{I}\\,.
\\end{align}
```

The second line follows from the definitions of ``\\mathbf{S}`` and ``\\mathbf{R}_{t}``.

Where:

  - $(math_dict[:f_t_fcb])
  - $(math_dict[:S_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])
  - $(math_dict[:I_identity])

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
    reduce_factor_mu(fcb::FactorFamilyBasis, mu::VecNum)

Drop the redundant entries of a factor mean, giving the reduced-axis mean.

The reduction applies no ratio. It undoes [`expand_factor_mu`](@ref), and the converse holds only for a raw mean that satisfies the zero-sum condition of every constrained Factor Family.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}^{\\mathrm{red}} &= \\mathbf{S}^{\\intercal} \\boldsymbol{\\mu}^{\\mathrm{raw}}\\,, \\\\
\\mathbf{S}^{\\intercal} \\mathbf{R}_{t} &= \\mathbf{I}\\,.
\\end{align}
```

The second line follows from the definitions of ``\\mathbf{S}`` and ``\\mathbf{R}_{t}``.

Where:

  - $(math_dict[:mu_fcb])
  - $(math_dict[:S_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])
  - $(math_dict[:I_identity])

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
    reduce_factor_covariance(fcb::FactorFamilyBasis, sigma::MatNum)
    reduce_factor_covariance(fcb::FactorFamilyBasis, sigma::Arr3Num)

Take the full-rank block of a factor covariance, giving the reduced-axis covariance.

The reduced factor returns are the retained raw ones, so the reduced covariance is the block of the retained factors and the reduction applies no ratio. It undoes [`expand_factor_covariance`](@ref), and the converse holds only for a raw covariance of the form ``\\mathbf{R}_{t} \\mathbf{\\Sigma}^{\\mathrm{red}} \\mathbf{R}_{t}^{\\intercal}``. A stack of covariances, one slice per observation, gives the stack of their blocks.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}^{\\mathrm{red}} &= \\mathbf{S}^{\\intercal} \\mathbf{\\Sigma}^{\\mathrm{raw}} \\mathbf{S}\\,, \\\\
\\mathbf{S}^{\\intercal} \\mathbf{R}_{t} &= \\mathbf{I}\\,.
\\end{align}
```

The second line follows from the definitions of ``\\mathbf{S}`` and ``\\mathbf{R}_{t}``.

Where:

  - $(math_dict[:Sigma_fcb])
  - $(math_dict[:S_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])
  - $(math_dict[:I_identity])

# Arguments

  - `fcb`: A Factor Family Basis.
  - `sigma::MatNum_Arr3Num`: Factor covariance on the raw axis, `factors × factors`, or a stack of them, `observations × factors × factors`.

# Validation

  - Both factor axes of `sigma` are `fcb.K`.

# Returns

  - `sigma::Array{<:Real}`: The factor covariance on the reduced axis, of the same number of dimensions as the input.

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
function reduce_factor_covariance(fcb::FactorFamilyBasis, sigma::Arr3Num)
    assert_factor_axis_length(size(sigma, 2), fcb.K, :sigma)
    assert_factor_axis_length(size(sigma, 3), fcb.K, :sigma)
    ret = retained_factor_indices(fcb)
    return sigma[:, ret, ret]
end
"""
    dropped_factor_weights(fcb::FactorFamilyBasis, t::Integer)
    dropped_factor_weights(fcb::FactorFamilyBasis)

Return the reduced-axis weights that reconstruct the dropped factors at one observation.

Row `j` holds the coefficients of the zero-sum condition of family `j`, so the row applied to a reduced-axis vector gives the entry of the factor that family drops. The function builds these rows from the ratios and never forms the change of basis. Without `t`, it gives the weights of every observation of the basis, one slice per observation.

# Mathematical definition

```math
\\begin{align}
\\mathbf{W}_{t} &= \\mathbf{D}^{\\intercal} \\mathbf{R}_{t}\\,.
\\end{align}
```

The row of a family that drops ``k`` holds ``-r_{t}(j)`` in the column of each retained member ``j``, and zero in every other column.

Where:

  - $(math_dict[:W_t_fcb])
  - $(math_dict[:D_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])

# Arguments

  - `fcb`: A Factor Family Basis.
  - `t::Integer`: Observation whose ratios the function reads.

# Validation

  - `t` indexes the observation axis of the basis.

# Returns

  - `W::Array{<:Real}`: The reconstruction weights, `constrained families × reduced factors`, or without `t` the stack of them, `observations × constrained families × reduced factors`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`expand_factor_covariance`](@ref)
"""
function dropped_factor_weights(fcb::FactorFamilyBasis, t::Integer)
    assert_factor_basis_index(t, fcb)
    Tf = real(eltype(fcb.ratios))
    W = zeros(Tf, length(fcb.fnm), reduced_factor_count(fcb))
    for j in eachindex(fcb.fnm)
        _, red, col = family_retained_indices(fcb, j)
        for p in eachindex(red)
            W[j, red[p]] = -Tf(fcb.ratios[t, col[p]])
        end
    end
    return W
end
function dropped_factor_weights(fcb::FactorFamilyBasis)
    T = size(fcb.ratios, 1)
    W = zeros(real(eltype(fcb.ratios)), T, length(fcb.fnm), reduced_factor_count(fcb))
    for t in 1:T
        W[t, :, :] = dropped_factor_weights(fcb, t)
    end
    return W
end
"""
    expand_factor_returns(fcb::FactorFamilyBasis, g::MatNum)
    expand_factor_returns(fcb::FactorFamilyBasis, g::VecNum)

Reconstruct the raw-axis factor returns from the reduced-axis ones.

The retained returns pass through, and the zero-sum condition of each family gives the return of the factor it drops.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{f}^{\\mathrm{raw}}_{t} &= \\mathbf{R}_{t} \\boldsymbol{f}^{\\mathrm{red}}_{t}\\,, \\\\
f^{\\mathrm{raw}}_{t,k} &= -\\sum_{j \\in \\mathcal{F} \\setminus \\{k\\}} r_{t}(j) \\, f^{\\mathrm{raw}}_{t,j}\\,.
\\end{align}
```

The second line follows from the first for every constrained Factor Family ``\\mathcal{F}`` that drops ``k``.

Where:

  - $(math_dict[:f_t_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:F_fam_att])
  - $(math_dict[:K_r_fcb])

# Arguments

  - `fcb`: A Factor Family Basis.
  - `g::VecNum_MatNum`: Factor returns on the reduced axis, either one observation per row or one observation alone. The function expands each row of a matrix with the ratios of that row, and a vector with the ratios of the last observation.

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
    Tf = promote_type(real(eltype(g)), real(eltype(fcb.ratios)))
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
    assert_factor_axis_length(length(g), reduced_factor_count(fcb), :g)
    return expand_factor_mu(fcb, g, size(fcb.ratios, 1))
end
"""
    expand_factor_mu(fcb::FactorFamilyBasis, mu::VecNum, t::Integer = size(fcb.ratios, 1))
    expand_factor_mu(fcb::FactorFamilyBasis, mu::MatNum)

Reconstruct the raw-axis factor mean from the reduced-axis one.

The retained entries pass through, and the zero-sum condition of each family at observation `t` gives the entry of the factor it drops. [`reduce_factor_mu`](@ref) undoes it. A matrix holds one mean per observation of the basis, one per row, and the function expands each row with the ratios of that row, as [`expand_factor_returns`](@ref) does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}^{\\mathrm{raw}} &= \\mathbf{R}_{t} \\boldsymbol{\\mu}^{\\mathrm{red}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])

# Arguments

  - `fcb`: A Factor Family Basis.
  - `mu::VecNum_MatNum`: Factor mean on the reduced axis, or one mean per observation, `observations × reduced factors`.
  - `t::Integer`: Observation whose ratios the function applies to a vector. It defaults to the last observation of the basis.

# Validation

  - The factor axis of `mu` is the reduced factor count, `t` indexes the observation axis of the basis, and a matrix matches that axis.

# Returns

  - `mu::Array{<:Real}`: The factor mean on the raw axis, of the same number of dimensions as the input.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_factor_mu`](@ref)
  - [`expand_factor_returns`](@ref)
"""
function expand_factor_mu(fcb::FactorFamilyBasis, mu::VecNum,
                          t::Integer = size(fcb.ratios, 1))
    assert_factor_axis_length(length(mu), reduced_factor_count(fcb), :mu)
    assert_factor_basis_index(t, fcb)
    Tf = promote_type(real(eltype(mu)), real(eltype(fcb.ratios)))
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
function expand_factor_mu(fcb::FactorFamilyBasis, mu::MatNum)
    return expand_factor_returns(fcb, mu)
end
"""
    expand_factor_covariance(fcb::FactorFamilyBasis, sigma::MatNum,
                             t::Integer = size(fcb.ratios, 1))
    expand_factor_covariance(fcb::FactorFamilyBasis, sigma::Arr3Num)

Reconstruct the raw-axis factor covariance from the reduced-axis one.

The answer is singular by construction, because the raw axis is a linear image of a smaller one. [`reduce_factor_covariance`](@ref) undoes it. A stack holds one covariance per observation of the basis, and the function expands slice `t` with the ratios of observation `t`. The standard errors of a realised attribution read such a stack, one covariance of the estimated factor returns per observation.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}^{\\mathrm{raw}} &= \\mathbf{R}_{t} \\mathbf{\\Sigma}^{\\mathrm{red}} \\mathbf{R}_{t}^{\\intercal}\\,, \\\\
\\mathbf{S}^{\\intercal} \\mathbf{\\Sigma}^{\\mathrm{raw}} \\mathbf{S} &= \\mathbf{\\Sigma}^{\\mathrm{red}}\\,, \\\\
\\mathbf{D}^{\\intercal} \\mathbf{\\Sigma}^{\\mathrm{raw}} \\mathbf{S} &= \\mathbf{W}_{t} \\mathbf{\\Sigma}^{\\mathrm{red}}\\,, \\\\
\\mathbf{D}^{\\intercal} \\mathbf{\\Sigma}^{\\mathrm{raw}} \\mathbf{D} &= \\mathbf{W}_{t} \\mathbf{\\Sigma}^{\\mathrm{red}} \\mathbf{W}_{t}^{\\intercal}\\,, \\\\
\\operatorname{rank} \\mathbf{\\Sigma}^{\\mathrm{raw}} &\\le K_{r} < K\\,.
\\end{align}
```

The last four lines follow from the first. The second, third and fourth give the retained block, the block of the dropped rows against the retained columns, and the dropped block.

Where:

  - $(math_dict[:Sigma_fcb])
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:S_fcb])
  - $(math_dict[:D_fcb])
  - $(math_dict[:W_t_fcb])
  - $(math_dict[:K_r_fcb])
  - $(math_dict[:K])

# Algorithm

 1. Check that `sigma` is ``K_{r} \\times K_{r}``.
 2. Build the reconstruction weights `W` of observation `t` with [`dropped_factor_weights`](@ref).
 3. Multiply `W` by `sigma`, giving `DR`, the block of the dropped rows against the retained columns.
 4. Multiply `DR` by the transpose of `W`, giving `DD`, the dropped block.
 5. Write `sigma`, `DR`, the transpose of `DR` and `DD` into their blocks of the ``K \\times K`` answer `raw`.
 6. For a stack, check that it holds one slice per observation of the basis, and do steps 1 to 5 on slice `t` with the ratios of observation `t`.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `sigma::MatNum_Arr3Num`: Factor covariance on the reduced axis, `reduced factors × reduced factors`, or a stack of them, `observations × reduced factors × reduced factors`.
  - `t::Integer`: Observation whose ratios the function applies to a matrix. It defaults to the last observation of the basis.

# Validation

  - Both factor axes of `sigma` are the reduced factor count, `t` indexes the observation axis of the basis, and a stack matches that axis.

# Returns

  - `sigma::Array{<:Real}`: The factor covariance on the raw axis, of the same number of dimensions as the input.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"ind\"], fi = [[2, 3]], di = [2],
                               ratios = reshape([0.5, 2.0], 2, 1), K = 3);

julia> V = cat([1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 4.0]; dims = 3);

julia> V = permutedims(V, (3, 1, 2));

julia> S = PortfolioOptimisers.expand_factor_covariance(fcb, V);

julia> S[1, :, :]
3×3 Matrix{Float64}:
 1.0   0.0   0.0
 0.0   1.0  -0.5
 0.0  -0.5   0.25

julia> S[2, :, :] == PortfolioOptimisers.expand_factor_covariance(fcb, V[2, :, :], 2)
true
```

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
    Tf = promote_type(real(eltype(sigma)), eltype(W))
    ret = retained_factor_indices(fcb)
    drp = dropped_factor_indices(fcb)
    DR = W * sigma
    DD = DR * transpose(W)
    raw = zeros(Tf, fcb.K, fcb.K)
    for b in eachindex(ret), a in eachindex(ret)
        raw[ret[a], ret[b]] = Tf(sigma[a, b])
    end
    for b in eachindex(ret), a in eachindex(drp)
        raw[drp[a], ret[b]] = Tf(DR[a, b])
        raw[ret[b], drp[a]] = Tf(DR[a, b])
    end
    for b in eachindex(drp), a in eachindex(drp)
        raw[drp[a], drp[b]] = Tf(DD[a, b])
    end
    return raw
end
function expand_factor_covariance(fcb::FactorFamilyBasis, sigma::Arr3Num)
    assert_factor_basis_obs(size(sigma, 1), fcb, :sigma)
    Tf = promote_type(real(eltype(sigma)), real(eltype(fcb.ratios)))
    T = size(sigma, 1)
    raw = Array{Tf, 3}(undef, T, fcb.K, fcb.K)
    for t in 1:T
        raw[t, :, :] = expand_factor_covariance(fcb, view(sigma, t, :, :), t)
    end
    return raw
end
"""
    project_factor_coordinates(fcb::FactorFamilyBasis, x::MatNum)
    project_factor_coordinates(fcb::FactorFamilyBasis, x::VecNum)

Project raw factor-space coordinates into the reduced basis.

The map applies the transpose of the change of basis, so it is not a column selection. The coordinate of a dropped factor contributes to the retained coordinates of its family. The factor exposure of a portfolio is such a coordinate.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{y} &= \\mathbf{R}_{t}^{\\intercal} \\boldsymbol{x}\\,, \\\\
\\mathbf{R}_{t}^{\\intercal} \\boldsymbol{g}_{t} &= (\\mathbf{B}_{t} \\mathbf{R}_{t})^{\\intercal} \\boldsymbol{w}_{t}\\,.
\\end{align}
```

The entry of a retained member ``j`` of a family that drops ``k`` is ``x_{j} - r_{t}(j) \\, x_{k}``, and the entry of a factor outside every constrained family is ``x_{j}``. The second line follows from the first, because ``\\boldsymbol{g}_{t} = \\mathbf{B}_{t}^{\\intercal} \\boldsymbol{w}_{t}``. The projected factor exposure of a portfolio is its exposure to the reduced factors of [`reduce_exposures`](@ref).

Where:

  - ``\\boldsymbol{x}``: Coordinates on the raw axis, ``K \\times 1``, with entry ``x_{j}`` for factor ``j``.
  - ``\\boldsymbol{y}``: Coordinates on the reduced axis, ``K_{r} \\times 1``.
  - $(math_dict[:R_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:K_r_fcb])
  - $(math_dict[:g_t_att])
  - $(math_dict[:B_t_att])
  - $(math_dict[:w_t_att])

# Arguments

  - `fcb`: A Factor Family Basis.
  - `x::VecNum_MatNum`: Coordinates on the raw axis, either one observation per row or one observation alone. The function projects each row of a matrix with the ratios of that row, and a vector with the ratios of the last observation.

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
    Tf = promote_type(real(eltype(x)), real(eltype(fcb.ratios)))
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
    Tf = promote_type(real(eltype(x)), real(eltype(fcb.ratios)))
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

public reduce_factor_names, reduce_exposures, reduce_loadings, reduce_factor_returns,
       reduce_factor_mu, reduce_factor_covariance, dropped_factor_weights,
       expand_factor_returns, expand_factor_mu, expand_factor_covariance,
       project_factor_coordinates
