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
 3. Multiply `W` by `sigma` with [`support_product`](@ref), giving `DR`, the block of the dropped rows against the retained columns. A `NaN` of `sigma` reaches only the dropped factors whose weight at it is not zero.
 4. Multiply `W`, `sigma` and the transpose of `W` with [`support_product`](@ref), giving `DD`, the dropped block.
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
  - [`support_product`](@ref)
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
    # A dropped factor reads the covariance of its own family alone, so a `NaN` of another
    # family must not reach it through a zero weight.
    DR = support_product(W, sigma)
    DD = support_product(W, sigma, W)
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
"""
    unseen_member_design(rule::AbstractUnseenMemberRule, fcb::Nothing, B::Arr3Num, Zl::Arr3Num,
                         W::MatNum)
    unseen_member_design(rule::SolvedUnseenMember, fcb::FactorFamilyBasis, B::Arr3Num,
                         Zl::Arr3Num, W::MatNum)
    unseen_member_design(rule::ZeroUnseenMember, fcb::FactorFamilyBasis, B::Arr3Num,
                         Zl::Arr3Num, W::MatNum)

Return the design that the regression of a [`CrossSectionalFactorPrior`](@ref) solves under an Unseen Member rule, and the change of each observation that maps its coefficients back.

Without a constrained Factor Family there is no Unseen Member. Under [`SolvedUnseenMember`](@ref) the regression solves the reduced design itself. In both cases the design is `Zl`, and no observation changes.

The exposures lag the returns, so the condition of an observation reads the benchmark weights of an earlier one. An asset that loads on a member there can leave the regression sample of the observation, for example when it delists. If no asset of the sample loads on the member, the data state nothing about its return, and its column of the raw design is zero. The member then fixes the condition by itself. The condition no longer identifies the other members of the family, so the reduced design is rank-deficient, and a pseudo-inverse answer depends on the member that the family drops.

Under [`ZeroUnseenMember`](@ref) the function gives the member a return of zero, as an Empty Factor has, and as the member has at an observation that gives it no benchmark weight. The condition then holds over the members that the sample sees, and over the whole family too, because the member adds a zero term to it. So the observation is identified again, and its answer does not depend on the dropped member. The residuals of the pairs of positive weight do not change when the members that the sample sees span the same fitted values, as a one-hot family does. The residual of an asset outside the sample reads the return of each Unseen Member that it loads on, which is zero.

# Mathematical definition

```math
\\begin{align}
\\mathcal{U}_{t} &= \\left\\{j \\in \\mathcal{F} : B_{tij} = 0 \\ \\forall i : u_{ti} > 0\\right\\}\\,, \\\\
f^{\\mathrm{raw}}_{t,j} &= 0\\,, \\quad j \\in \\mathcal{U}_{t}\\,, \\\\
\\sum_{j \\in \\mathcal{F} \\setminus \\mathcal{U}_{t}} c_{t}(j) \\, f^{\\mathrm{raw}}_{t,j} &= 0\\,, \\\\
\\boldsymbol{f}^{\\mathrm{red}}_{t} &= \\mathbf{P}_{t} \\boldsymbol{h}_{t}\\,, \\quad \\boldsymbol{h}_{t} = \\underset{\\boldsymbol{h}}{\\arg\\min} \\sum_{i} u_{ti} \\left(x_{t,\\,i} - \\mathbf{B}^{\\mathrm{red}}_{t,i} \\mathbf{P}_{t} \\boldsymbol{h}\\right)^{2}\\,.
\\end{align}
```

``\\mathbf{P}_{t}`` is the identity with a zero column at each retained member of ``\\mathcal{U}_{t}``. When the dropped member ``k`` is in ``\\mathcal{U}_{t}``, the condition moves onto the retained member ``q`` the sample sees with the largest ``|r_{t}(q)|``: the column of ``q`` is zero, and its row holds ``-r_{t}(j) / r_{t}(q)`` at each other retained member ``j`` the sample sees. So ``\\mathbf{B}^{\\mathrm{red}}_{t} \\mathbf{P}_{t} \\boldsymbol{h}_{t} = \\mathbf{B}^{\\mathrm{red}}_{t} \\boldsymbol{f}^{\\mathrm{red}}_{t}``. ``\\mathbf{P}_{t}`` is idempotent, ``\\mathbf{P}_{t} \\mathbf{P}_{t} = \\mathbf{P}_{t}``, so the changed design times the factor returns is the same product too, and a consumer that reads the changed design with the stored factor returns gets the fitted values of the fit. The covariance of the factor returns is ``\\mathbf{P}_{t} \\mathbf{V}_{h} \\mathbf{P}_{t}^{\\intercal}``, where ``\\mathbf{V}_{h}`` is the covariance of ``\\boldsymbol{h}_{t}``. So an Unseen Member has a variance of zero, as its return is stated, not estimated. And the expansion of ``\\boldsymbol{f}^{\\mathrm{red}}_{t}`` through ``\\mathbf{R}_{t}`` gives the raw returns above. Under [`ZeroUnseenMember`](@ref) the function changes no observation where every member of ``\\mathcal{U}_{t}`` has ``c_{t}(j) = 0`` and the sample sees ``k``, because there the zero column already states that answer.

Where:

  - $(math_dict[:F_fam_att])
  - ``\\mathcal{U}_{t}``: Members of ``\\mathcal{F}`` that no asset of positive weight at observation ``t`` loads on.
  - $(math_dict[:B_tik_cs])
  - $(math_dict[:u_ti_cs])
  - $(math_dict[:c_tj_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:f_t_fcb])
  - ``\\mathbf{B}^{\\mathrm{red}}_{t,i}``: Reduced exposures of asset ``i`` at observation ``t``, ``1 \\times K_{r}``.
  - $(math_dict[:x_ti_ret])
  - ``\\mathbf{P}_{t}``: Change of the reduced design of observation ``t``, ``K_{r} \\times K_{r}``.
  - ``\\boldsymbol{h}_{t}``: Coefficients of the regression on the changed design, ``K_{r} \\times 1``.

# Arguments

  - `rule`: The Unseen Member rule.
  - `fcb`: The Factor Family Basis of the lagged exposures, one row per observation of `Zl`, or `nothing` without a constrained Factor Family.
  - `B`: Lagged exposures on the raw axis, `observations × assets × factors`.
  - `Zl`: Lagged exposures on the reduced axis, which `fcb` gives from `B`.
  - `W`: Cross-sectional weights matrix `observations × assets`. The pairs of positive weight are the sample of each observation, as [`cross_sectional_design_mask`](@ref) states it. The fit passes the weights it regresses with, and a consumer of a [`CrossSectionalFactorModel`](@ref) passes the stored weights `rw`.

# Validation

  - The factor axis of `B` is `fcb.K`, and `B` matches the observation axis of the basis.
  - `W` has the observations and the assets of `B`.

# Returns

  - `Z::Arr3Num`: The design to regress on: `Zl`, with `Zl[t, :, :] * P` at each observation in `P`.
  - `P`: One `t => P_t` pair per observation that the function changes. It is empty when the function changes none, and `()` without a basis or under [`SolvedUnseenMember`](@ref).

# Related

  - [`AbstractUnseenMemberRule`](@ref)
  - [`ZeroUnseenMember`](@ref)
  - [`SolvedUnseenMember`](@ref)
  - [`unseen_member_returns`](@ref)
  - [`reduce_exposures`](@ref)
  - [`cross_sectional_live_regression`](@ref)
"""
function unseen_member_design(::AbstractUnseenMemberRule, ::Nothing, ::Arr3Num, Zl::Arr3Num,
                              ::MatNum)
    return (; Z = Zl, P = ())
end
function unseen_member_design(::SolvedUnseenMember, ::FactorFamilyBasis, ::Arr3Num,
                              Zl::Arr3Num, ::MatNum)
    return (; Z = Zl, P = ())
end
function unseen_member_design(::ZeroUnseenMember, fcb::FactorFamilyBasis, B::Arr3Num,
                              Zl::Arr3Num, W::MatNum)
    assert_factor_axis_length(size(B, 3), fcb.K, :B)
    assert_factor_basis_obs(size(B, 1), fcb, :B)
    @argcheck(size(W) == (size(B, 1), size(B, 2)),
              DimensionMismatch("W ($(size(W, 1))×$(size(W, 2))) must match B ($(size(B, 1))×$(size(B, 2))) on the observation and asset axes"))
    act = W .> zero(eltype(W))
    Tp = float_if_integer(real(eltype(fcb.ratios)))
    P = Pair{Int, Matrix{Tp}}[]
    for t in axes(B, 1)
        Pt = unseen_member_change(fcb, view(B, t, :, :), view(act, t, :), t)
        if !isnothing(Pt)
            push!(P, t => Pt)
        end
    end
    if isempty(P)
        return (; Z = Zl, P = P)
    end
    Z = similar(Zl, promote_type(eltype(Zl), Tp))
    copyto!(Z, Zl)
    # A product over views of an open element type is opaque to the analysis, so the rows that
    # change take a loop of scalar products.
    for (t, Pt) in P, k in axes(Pt, 2), i in axes(Z, 2)
        Z[t, i, k] = sum(j -> Zl[t, i, j] * Pt[j, k], axes(Pt, 1))
    end
    return (; Z = Z, P = P)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the change of the reduced design of one observation under [`ZeroUnseenMember`](@ref), or `nothing` when the observation needs none.

A family needs a change when the sample does not see its dropped member, or when it does not see a retained member that carries a ratio other than zero. [`unseen_member_design`](@ref) states the change and its mathematics.

# Arguments

  - `fcb`: The Factor Family Basis of the lagged exposures.
  - `Bt`: The lagged exposures of the observation on the raw axis, `assets × factors`.
  - `at`: `true` at each asset of positive weight at the observation.
  - `t`: The observation, a row of `fcb.ratios`.

# Returns

  - `Pt::Option{<:Matrix}`: The change, `reduced factors × reduced factors`, or `nothing`.

# Related

  - [`unseen_member_design`](@ref)
  - [`unseen_member_family!`](@ref)
"""
function unseen_member_change(fcb::FactorFamilyBasis, Bt::MatNum, at::AbstractVector{Bool},
                              t::Integer)
    Tp = float_if_integer(real(eltype(fcb.ratios)))
    seen(i) = any(n -> at[n] && !iszero(Bt[n, i]), eachindex(at))
    Pt = nothing
    for j in eachindex(fcb.fnm)
        raw, red, col = family_retained_indices(fcb, j)
        us = map(!seen, raw)
        ds = seen(fcb.fi[j][fcb.di[j]])
        r = view(fcb.ratios, t, col)
        if ds && all(iszero, view(r, us))
            continue
        end
        Pt = something(Pt,
                       Matrix{Tp}(LinearAlgebra.I, reduced_factor_count(fcb),
                                  reduced_factor_count(fcb)))
        unseen_member_family!(Pt, red, r, us, ds)
    end
    return Pt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Write the change of one constrained Factor Family into the change of the reduced design of one observation.

The column of each retained member that the sample does not see becomes zero. When the sample does not see the dropped member either, the zero-sum condition moves onto the retained member `q` that the sample sees with the largest ``|r_{t}(q)|``: its column becomes zero, and its row holds ``-r_{t}(j) / r_{t}(q)`` at each other retained member ``j`` that the sample sees. A family whose members the sample sees carry no ratio other than zero keeps the other columns, because the condition then reads the dropped member alone.

# Arguments

  - `Pt`: The change of the observation, modified in place.
  - `red`: The reduced-axis index of each retained member of the family.
  - `r`: The ratio of each retained member at the observation.
  - `us`: `true` at each retained member that the sample does not see.
  - `ds`: Whether the sample sees the dropped member.

# Returns

  - `Pt::Matrix`: The change, modified in place.

# Related

  - [`unseen_member_change`](@ref)
  - [`unseen_member_design`](@ref)
"""
function unseen_member_family!(Pt::Matrix, red::AbstractVector{<:Integer}, r::VecNum,
                               us::AbstractVector{Bool}, ds::Bool)
    Pt[:, view(red, us)] .= zero(eltype(Pt))
    live = findall(!, us)
    if ds || isempty(live)
        return Pt
    end
    q = argmax(p -> abs(r[p]), live)
    if iszero(r[q])
        return Pt
    end
    for p in live
        Pt[red[q], red[p]] = -r[p] / r[q]
    end
    Pt[red[q], red[q]] = zero(eltype(Pt))
    return Pt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Map the coefficients of a regression on the design of [`unseen_member_design`](@ref) back to the reduced factor returns.

The row of each changed observation `t` becomes `P_t` times the coefficients of that row. Every other row, the residuals, the counts, the intercept and the leverage-one mark stay as the regression gives them. The design times the coefficients equals the reduced exposures times the factor returns, so the residuals are those of the factor returns.

# Arguments

  - `csr`: The regression on the changed design.
  - `P`: The `t => P_t` pairs of [`unseen_member_design`](@ref).

# Returns

  - `csr::CrossSectionalRegression`: The regression with the reduced factor returns in `f`. It is `csr` itself when `P` is empty.

# Related

  - [`unseen_member_design`](@ref)
"""
function unseen_member_returns(csr::CrossSectionalRegression, P)
    if isempty(P)
        return csr
    end
    f = copy(csr.f)
    for (t, Pt) in P
        LinearAlgebra.mul!(view(f, t, :), Pt, view(csr.f, t, :))
    end
    return CrossSectionalRegression(; f = f, eps = csr.eps, n = csr.n, b = csr.b,
                                    h1 = csr.h1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the change of the reduced design of one observation from the pairs of [`unseen_member_design`](@ref), or `nothing` when the observation has none.

A consumer that rebuilds the design of each observation reads the change of that observation here, so an observation without a change takes its usual code.

# Arguments

  - `P`: The `t => P_t` pairs of [`unseen_member_design`](@ref).
  - `t`: The observation.

# Returns

  - `Pt::Option{<:AbstractMatrix}`: The change of observation `t`, or `nothing`.

# Related

  - [`unseen_member_design`](@ref)
  - [`cs_regression_t_stats`](@ref)
  - [`attribution_error_pass`](@ref)
"""
function unseen_member_change_at(P, t::Integer)
    for (s, Ps) in P
        if s == t
            return Ps
        end
    end
    return nothing
end

public reduce_factor_names, reduce_exposures, reduce_loadings, reduce_factor_returns,
       reduce_factor_mu, reduce_factor_covariance, dropped_factor_weights,
       expand_factor_returns, expand_factor_mu, expand_factor_covariance,
       project_factor_coordinates, unseen_member_design
