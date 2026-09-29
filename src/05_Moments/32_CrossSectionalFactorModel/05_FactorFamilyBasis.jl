"""
$(DocStringExtensions.TYPEDEF)

Compact change of basis between the raw factor axis and the reduced axis a re-based Factor Family is fitted in.

A Factor Family whose one-hot exposures are collinear with a global factor carries one redundant column. The re-basis drops one member of the family and rewrites the family in an equivalent basis of full column rank, in which the benchmark-weighted factor returns of the family sum to zero. The change of basis differs from one observation to the next. The result stores, for each observation, the ratio of the benchmark-weighted exposure of each retained member to that of the dropped member, and it never stores the dense basis matrix.

# Mathematical definition

For each constrained Factor Family ``\\mathcal{F}`` that drops the member ``k``, and at each observation ``t``,

```math
\\begin{align}
c_{t}(j) &= \\left(\\mathbf{B}_{t}^{\\intercal} \\bar{\\boldsymbol{w}}_{t}\\right)_{j}\\,, \\\\
r_{t}(j) &= \\frac{c_{t}(j)}{c_{t}(k)}\\,, \\quad j \\in \\mathcal{F} \\setminus \\{k\\}\\,, \\\\
\\sum_{j \\in \\mathcal{F}} c_{t}(j) \\, f^{\\mathrm{raw}}_{t,j} &= 0\\,, \\\\
f^{\\mathrm{raw}}_{t,k} &= -\\sum_{j \\in \\mathcal{F} \\setminus \\{k\\}} r_{t}(j) \\, f^{\\mathrm{raw}}_{t,j}\\,.
\\end{align}
```

The third line is the zero-sum condition, and the fourth line is the same condition solved for the return of the dropped factor. So the retained factor returns and the ratios fix the whole family, and the reduced axis holds one factor fewer for each family. A ratio is finite only where ``c_{t}(k) \\neq 0``.

Where:

  - $(math_dict[:F_fam_att])
  - $(math_dict[:c_tj_fcb])
  - $(math_dict[:B_t_att])
  - $(math_dict[:wbar_t_fcb])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:f_t_fcb])
  - $(math_dict[:K_r_fcb])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorFamilyBasis(; fnm::VecStr, fi::AbstractVector{<:AbstractVector{<:Integer}},
                      di::AbstractVector{<:Integer}, ratios::MatNum, K::Integer)

Keywords correspond to the struct's fields.

## Validation

  - `K > 0` and `!isempty(fnm)`.
  - `fnm`, `fi` and `di` have the same length, and `fnm` does not repeat a label.
  - Every family holds at least two members, its members are unique, and every member lies in `1:K`.
  - No factor belongs to two families.
  - `di[j]` indexes `fi[j]`.
  - `ratios` has one column per retained member of a constrained family, `sum(length(fi[j]) - 1)` in all.
  - `ratios` is not empty, and every entry of it is finite.
  - The reduced axis is not empty. The rules above imply it: the families are disjoint, each holds at least two members of `1:K`, so `K` is at least twice the number of families, and no separate check is needed.

# Examples

```jldoctest
julia> FactorFamilyBasis(; fnm = [\"industry\"], fi = [[1, 2]], di = [2],
                         ratios = reshape([0.5, 0.4], 2, 1), K = 3)
FactorFamilyBasis
     fnm ┼ Vector{String}: ["industry"]
      fi ┼ 1-element Vector{Vector{Int64}}
      di ┼ Vector{Int64}: [2]
  ratios ┼ 2×1 Matrix{Float64}
       K ┴ Int64: 3
```

# Related

  - [`AbstractFactorFamilyBasis`](@ref)
  - [`factor_family_basis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`has_family_rebasis`](@ref)
"""
@concrete struct FactorFamilyBasis <: AbstractFactorFamilyBasis
    """
    Label of each constrained Factor Family, one entry per family, in the order the caller requested them. That order fixes the column order of `ratios`.
    """
    fnm
    """
    Member indices of each constrained Factor Family in the raw factor axis, one vector per family. Two families never share a member.
    """
    fi
    """
    Position within `fi[j]` of the member family `j` drops, one entry per family. The dropped member is the one the zero-sum condition reconstructs.
    """
    di
    """
    Ratios of the benchmark-weighted exposures, `observations × C`, with `C = sum(length(fi[j]) - 1)`. The block of family `j` holds one column per retained member, in the member order of `fi[j]` with the dropped position removed.
    """
    ratios
    """
    Number of factors on the raw axis. The reduced axis holds `K - length(fnm)` of them.
    """
    K
    function FactorFamilyBasis(fnm::VecStr, fi::AbstractVector{<:AbstractVector{<:Integer}},
                               di::AbstractVector{<:Integer}, ratios::MatNum, K::Integer)
        assert_gt0(K, :K)
        @argcheck(!isempty(fnm), IsEmptyError("fnm cannot be empty"))
        @argcheck(length(fnm) == length(fi) == length(di),
                  DimensionMismatch("fnm ($(length(fnm))), fi ($(length(fi))) and di ($(length(di))) must have the same length"))
        @argcheck(allunique(fnm), ArgumentError("fnm must not repeat a family label"))
        seen = Set{Int}()
        C = 0
        for j in eachindex(fi)
            m = length(fi[j])
            @argcheck(m >= 2,
                      ArgumentError("family $(fnm[j]) holds $m factors, and a constrained family needs at least two"))
            @argcheck(allunique(fi[j]),
                      ArgumentError("family $(fnm[j]) repeats a factor index"))
            for i in fi[j]
                @argcheck(1 <= i <= K,
                          DomainError(i,
                                      "every factor index of family $(fnm[j]) must lie in 1:$K"))
                @argcheck(i ∉ seen,
                          ArgumentError("factor index $i belongs to more than one constrained family, and the families must be disjoint"))
                push!(seen, i)
            end
            @argcheck(1 <= di[j] <= m,
                      DomainError(di[j],
                                  "the dropped position of family $(fnm[j]) must lie in 1:$m"))
            C += m - 1
        end
        @argcheck(size(ratios, 2) == C,
                  DimensionMismatch("ratios ($(size(ratios, 2)) columns) must hold one column per retained member of a constrained family ($C)"))
        @argcheck(!isempty(ratios), IsEmptyError("ratios cannot be empty"))
        assert_all_finite(ratios, :ratios)
        return new{typeof(fnm), typeof(fi), typeof(di), typeof(ratios), typeof(K)}(fnm, fi,
                                                                                   di,
                                                                                   ratios,
                                                                                   K)
    end
end
function FactorFamilyBasis(; fnm::VecStr, fi::AbstractVector{<:AbstractVector{<:Integer}},
                           di::AbstractVector{<:Integer}, ratios::MatNum,
                           K::Integer)::FactorFamilyBasis
    return FactorFamilyBasis(fnm, fi, di, ratios, K)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the raw-axis index of the factor each constrained family drops.

# Arguments

  - `fcb`: A Factor Family Basis.

# Returns

  - `d::Vector{Int}`: One raw-axis index per constrained family, in family order.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"industry\"], fi = [[1, 3]], di = [2],
                               ratios = reshape([0.5], 1, 1), K = 4);

julia> PortfolioOptimisers.dropped_factor_indices(fcb)
1-element Vector{Int64}:
 3
```

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`retained_factor_indices`](@ref)
"""
function dropped_factor_indices(fcb::FactorFamilyBasis)::Vector{Int}
    return [Int(fcb.fi[j][fcb.di[j]]) for j in eachindex(fcb.fi)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the raw-axis indices the reduced axis keeps, in raw order.

The reduced axis is the raw axis with the dropped factor of every constrained family removed, so every other factor keeps its relative position.

# Arguments

  - `fcb`: A Factor Family Basis.

# Returns

  - `r::Vector{Int}`: The retained raw-axis indices, in increasing order.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"industry\"], fi = [[1, 3]], di = [2],
                               ratios = reshape([0.5], 1, 1), K = 4);

julia> PortfolioOptimisers.retained_factor_indices(fcb)
3-element Vector{Int64}:
 1
 2
 4
```

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`dropped_factor_indices`](@ref)
  - [`reduced_factor_count`](@ref)
"""
function retained_factor_indices(fcb::FactorFamilyBasis)::Vector{Int}
    keep = trues(fcb.K)
    for d in dropped_factor_indices(fcb)
        keep[d] = false
    end
    return findall(keep)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the number of factors on the reduced axis.

# Arguments

  - `fcb`: A Factor Family Basis.

# Returns

  - `K::Int`: `fcb.K` less one factor per constrained family.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"industry\"], fi = [[1, 3]], di = [2],
                               ratios = reshape([0.5], 1, 1), K = 4);

julia> PortfolioOptimisers.reduced_factor_count(fcb)
3
```

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`retained_factor_indices`](@ref)
"""
function reduced_factor_count(fcb::FactorFamilyBasis)::Int
    return Int(fcb.K) - length(fcb.fnm)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the map from a raw-axis index to its reduced-axis index.

A dropped factor has no reduced-axis index, and the map holds `0` for it.

# Arguments

  - `fcb`: A Factor Family Basis.

# Returns

  - `m::Vector{Int}`: One entry per raw factor, `0` where the factor is dropped.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`retained_factor_indices`](@ref)
"""
function raw_to_reduced_index(fcb::FactorFamilyBasis)::Vector{Int}
    out = zeros(Int, fcb.K)
    for (k, i) in enumerate(retained_factor_indices(fcb))
        out[i] = k
    end
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the raw-axis indices, the reduced-axis indices and the `ratios` columns of one constrained family.

The three vectors are aligned. Entry `p` of each names the same retained member of family `j`, in the raw axis, in the reduced axis and in `ratios`.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `j::Integer`: Position of the family in `fcb.fnm`.

# Returns

  - `raw::Vector{Int}`: Raw-axis indices of the retained members.
  - `red::Vector{Int}`: Reduced-axis indices of the same members.
  - `col::Vector{Int}`: Columns of `fcb.ratios` that hold their ratios.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`raw_to_reduced_index`](@ref)
"""
function family_retained_indices(fcb::FactorFamilyBasis, j::Integer)
    off = 0
    for l in 1:(j - 1)
        off += length(fcb.fi[l]) - 1
    end
    raw = Int[]
    col = Int[]
    p = 0
    for (q, i) in enumerate(fcb.fi[j])
        if q == fcb.di[j]
            continue
        end
        p += 1
        push!(raw, Int(i))
        push!(col, off + p)
    end
    m = raw_to_reduced_index(fcb)
    red = [m[i] for i in raw]
    return raw, red, col
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the basis restricted to a selection of observations.

The ratios keep the rows that `i` selects, and every other field is copied. So a caller reads the basis on a window of observations without building it again.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `i`: Indices of the observations to select.

# Returns

  - `fcb::FactorFamilyBasis`: A new basis whose `ratios` hold only the selected observations.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"industry\"], fi = [[1, 2]], di = [2],
                               ratios = reshape([0.5, 0.4, 0.3], 3, 1), K = 3);

julia> PortfolioOptimisers.factor_basis_slice(fcb, 2:3).ratios
2×1 Matrix{Float64}:
 0.4
 0.3
```

# Related

  - [`FactorFamilyBasis`](@ref)
"""
function factor_basis_slice(fcb::FactorFamilyBasis, i)::FactorFamilyBasis
    R = fcb.ratios[i, :]
    return FactorFamilyBasis(; fnm = fcb.fnm, fi = fcb.fi, di = fcb.di,
                             ratios = isa(R, AbstractMatrix) ? R : reshape(R, 1, :),
                             K = fcb.K)
end
"""
    append_passthrough_factors(fcb::Nothing, n::Integer) -> nothing
    append_passthrough_factors(fcb::FactorFamilyBasis, n::Integer) -> FactorFamilyBasis

Extend a Factor Family Basis with `n` factors that it passes through unchanged.

The new factors take the last `n` positions of the raw axis, and they belong to no constrained family. So the basis keeps them on the reduced axis, after every retained factor, and it never re-bases them: their row of the change of basis is an identity row. The families, the dropped members and the ratios do not change. The observed factors of a [`CrossSectionalFactorPrior`](@ref), such as its Currency Factors, join the basis this way, because their returns are observed and the zero-sum condition of a family does not reach them.

A prior that constrains no Factor Family has no basis, and the method over `Nothing` takes that case. Its raw axis is its reduced axis, so a pass-through factor needs no basis there.

# Algorithm

 1. Refuse a negative `n`.
 2. `fcb` is `nothing`: return `nothing`.
 3. `n` is zero: return `fcb` itself.
 4. Otherwise build the basis again with `K = fcb.K + n`, which runs every check of the constructor.

# Arguments

  - `fcb`: A Factor Family Basis, or `nothing`.
  - `n`: Number of pass-through factors to append.

# Validation

  - `n >= 0`. Raises a `DomainError`.

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The extended basis, or `nothing`.

# Examples

```jldoctest
julia> fcb = FactorFamilyBasis(; fnm = [\"industry\"], fi = [[2, 3]], di = [2],
                               ratios = reshape([0.5], 1, 1), K = 3);

julia> fcb2 = PortfolioOptimisers.append_passthrough_factors(fcb, 2);

julia> fcb2.K, PortfolioOptimisers.retained_factor_indices(fcb2)
(5, [1, 2, 4, 5])
```

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`retained_factor_indices`](@ref)
  - [`expand_factor_returns`](@ref)
  - [`AbstractObservedExposureEstimator`](@ref)
"""
function append_passthrough_factors(::Nothing, n::Integer)::Nothing
    assert_nonneg(n, :n)
    return nothing
end
function append_passthrough_factors(fcb::FactorFamilyBasis, n::Integer)::FactorFamilyBasis
    assert_nonneg(n, :n)
    if iszero(n)
        return fcb
    end
    return FactorFamilyBasis(; fnm = fcb.fnm, fi = fcb.fi, di = fcb.di, ratios = fcb.ratios,
                             K = fcb.K + n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the name of the factor each constrained family drops.

# Arguments

  - `fcb`: A Factor Family Basis.
  - `nf::VecStr`: Names of the raw factor axis, of length `fcb.K`.

# Validation

  - `length(nf) == fcb.K`.

# Returns

  - `nm::Vector{String}`: One dropped factor name per constrained family, in family order.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_factor_names`](@ref)
"""
function dropped_factor_names(fcb::FactorFamilyBasis, nf::VecStr)::Vector{String}
    assert_factor_axis_length(length(nf), fcb.K, :nf)
    return [String(nf[d]) for d in dropped_factor_indices(fcb)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse an argument whose factor axis is the wrong length.

# Arguments

  - `got::Integer`: Length the argument carries.
  - `want::Integer`: Length the basis needs.
  - `sym::Sym_Str`: Name of the argument, used in the message.

# Validation

  - `got == want`, otherwise a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`FactorFamilyBasis`](@ref)
"""
function assert_factor_axis_length(got::Integer, want::Integer, sym::Sym_Str)::Nothing
    @argcheck(got == want,
              DimensionMismatch("$sym carries a factor axis of $got, and the basis needs $want"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a time-varying argument whose observation axis does not match the basis.

# Arguments

  - `got::Integer`: Number of observations the argument carries.
  - `fcb`: A Factor Family Basis.
  - `sym::Sym_Str`: Name of the argument, used in the message.

# Validation

  - `got == size(fcb.ratios, 1)`, otherwise a `DimensionMismatch` that names [`factor_basis_slice`](@ref).

# Returns

  - `nothing`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`factor_basis_slice`](@ref)
"""
function assert_factor_basis_obs(got::Integer, fcb::FactorFamilyBasis,
                                 sym::Sym_Str)::Nothing
    want = size(fcb.ratios, 1)
    @argcheck(got == want,
              DimensionMismatch("$sym carries $got observations and the basis carries $want. Slice the basis with `factor_basis_slice` so both time axes agree"))
    return nothing
end
"""
    factor_family_basis(families::AbstractVector{<:Pair}, Ms::Arr3Num, bw::MatNum,
                        nf::VecStr, fam::VecStr) -> FactorFamilyBasis

Build the compact change of basis of the requested constrained Factor Families.

The element type of the ratios is the promoted type of `Ms` and `bw`, and an integer type becomes a floating-point type. So a `Rational` history gives exact ratios, and a `Float32` history gives `Float32` ratios. A dropped factor whose benchmark-weighted exposure is small but not zero gives large finite ratios. The builder accepts them, and they are ill-conditioned. To avoid them, state a dropped factor with a large exposure, or let the automatic choice pick one.

# Mathematical definition

For each requested family ``\\mathcal{F}``, and at each observation ``t``,

```math
\\begin{align}
\\bar{w}_{t,i} &= \\frac{b_{t,i}}{\\sum_{l=1}^{N} b_{t,l}}\\,, \\\\
c_{t}(j) &= \\left(\\mathbf{B}_{t}^{\\intercal} \\bar{\\boldsymbol{w}}_{t}\\right)_{j}\\,, \\\\
k &= \\underset{j \\in \\mathcal{F}}{\\arg\\max} \\sum_{t=1}^{T} \\lvert c_{t}(j) \\rvert\\,, \\\\
r_{t}(j) &= \\frac{c_{t}(j)}{c_{t}(k)}\\,, \\quad j \\in \\mathcal{F} \\setminus \\{k\\}\\,.
\\end{align}
```

The third line holds when the caller states no dropped factor. Otherwise ``k`` is the stated factor. The automatic choice makes ``\\lvert c_{t}(k) \\rvert`` large on average, so the ratios stay moderate.

Where:

  - ``b_{t,i}``: Benchmark weight of asset ``i`` at observation ``t``, with a non-finite weight read as zero.
  - $(math_dict[:wbar_t_fcb])
  - $(math_dict[:c_tj_fcb])
  - $(math_dict[:B_t_att])
  - $(math_dict[:F_fam_att])
  - $(math_dict[:r_tj_fcb])
  - $(math_dict[:N])
  - $(math_dict[:T])

# Algorithm

 1. Check the axes of `Ms`, `bw`, `nf` and `fam`.
 2. Compute `c`, the benchmark-weighted exposure of every raw factor at every observation, with [`weighted_family_exposures`](@ref).
 3. For each requested family, in the order of `families`, find its members `idx` in `fam`.
 4. Resolve the position `d` of the dropped member with [`resolve_dropped_member`](@ref). On a tie of the automatic choice, the first member in the order of `fam` wins.
 5. Refuse the family when the dropped member's column of `c` is zero at an observation.
 6. Divide the columns of `c` of the retained members by the column of the dropped member, giving the family's block of the ratios.
 7. Build the [`FactorFamilyBasis`](@ref) from the resolved families and the concatenated blocks, which runs every check of the constructor again.

# Arguments

  - `families`: Pairs of `family label => dropped factor name`, in the order the families take columns of `ratios`. A `nothing` on the right asks for the automatic choice of the dropped factor.
  - `Ms::Arr3Num`: Exposure history, `observations × assets × factors`.
  - `bw::MatNum`: Benchmark weight history, `observations × assets`.
  - `nf::VecStr`: Names of the raw factor axis, of length `size(Ms, 3)`.
  - `fam::VecStr`: Family label of each raw factor, of length `size(Ms, 3)`.

# Validation

  - `families` and `Ms` are not empty, and no family label appears twice.
  - `nf` and `fam` are as long as the factor axis of `Ms`, and `nf` does not repeat a name.
  - `bw` matches `Ms` on the observation and asset axes, and every finite weight is non-negative.
  - Every observation carries a strictly positive benchmark weight sum.
  - Every requested family label appears in `fam`, and holds at least two factors.
  - A stated dropped factor name appears in `nf`, and belongs to the family that names it.
  - The benchmark-weighted exposure of each dropped factor is not zero at any observation, otherwise an `IsNonFiniteError`, because the ratios would not be finite.
  - The rules of [`FactorFamilyBasis`](@ref).

# Returns

  - `fcb::FactorFamilyBasis`: The compact change of basis.

# Examples

```jldoctest
julia> Ms = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], 2, 2, 2);

julia> factor_family_basis([\"ind\" => nothing], Ms, [0.5 0.5; 0.5 0.5], [\"ind=a\", \"ind=b\"],
                           [\"ind\", \"ind\"])
FactorFamilyBasis
     fnm ┼ Vector{String}: ["ind"]
      fi ┼ 1-element Vector{Vector{Int64}}
      di ┼ Vector{Int64}: [1]
  ratios ┼ 2×1 Matrix{Float64}
       K ┴ Int64: 2
```

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`reduce_exposures`](@ref)
"""
function factor_family_basis(families::AbstractVector{<:Pair}, Ms::Arr3Num, bw::MatNum,
                             nf::VecStr, fam::VecStr)::FactorFamilyBasis
    @argcheck(!isempty(families), IsEmptyError("families cannot be empty"))
    @argcheck(!isempty(Ms), IsEmptyError("Ms cannot be empty"))
    T, N, K = size(Ms)
    @argcheck(size(bw) == (T, N),
              DimensionMismatch("bw ($(size(bw, 1))×$(size(bw, 2))) must match Ms ($T×$N) on the observation and asset axes"))
    assert_factor_axis_length(length(nf), K, :nf)
    assert_factor_axis_length(length(fam), K, :fam)
    @argcheck(allunique(nf), ArgumentError("nf must not repeat a factor name"))
    Tf = float_if_integer(promote_type(real(eltype(Ms)), real(eltype(bw))))
    c = weighted_family_exposures(Ms, bw, Tf)
    fnm = String[]
    fi = Vector{Int}[]
    di = Int[]
    blocks = Matrix{Tf}[]
    ret = Int[]
    for pr in families
        nm = String(first(pr))
        @argcheck(nm ∉ fnm, ArgumentError("family $nm appears more than once in families"))
        idx = findall(isequal(nm), fam)
        @argcheck(!isempty(idx),
                  ArgumentError("family $nm names no factor. The declared families are $(unique(fam))"))
        @argcheck(length(idx) >= 2,
                  ArgumentError("family $nm holds $(length(idx)) factor, and a constrained family needs at least two"))
        d = resolve_dropped_member(last(pr), nm, idx, nf, c)
        t0 = findfirst(iszero, view(c, :, idx[d]))
        @argcheck(isnothing(t0),
                  IsNonFiniteError("the dropped factor $(nf[idx[d]]) of family $nm has a zero benchmark-weighted exposure at observation $t0, so its ratios are not finite. Drop another member of the family"))
        append!(empty!(ret), (i for i in idx if i != idx[d]))
        push!(fnm, nm)
        push!(fi, idx)
        push!(di, d)
        push!(blocks, c[:, ret] ./ view(c, :, idx[d]))
    end
    return FactorFamilyBasis(; fnm = fnm, fi = fi, di = di, ratios = reduce(hcat, blocks),
                             K = K)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the benchmark-weighted exposure of every raw factor at every observation.

# Algorithm

 1. Read a non-finite benchmark weight as zero, and refuse a finite negative one.
 2. Normalise each observation's weights to sum to one, and refuse an observation whose sum is not positive.
 3. Read a non-finite exposure as zero, and take the weighted sum across the assets.

# Arguments

  - `Ms::Arr3Num`: Exposure history, `observations × assets × factors`.
  - `bw::MatNum`: Benchmark weight history, `observations × assets`.
  - `Tf::Type{<:Real}`: Element type of the answer.

# Validation

  - Every finite entry of `bw` is non-negative.
  - Every observation's benchmark weights sum to a strictly positive number, once the non-finite ones are read as zero.

# Returns

  - `c::Matrix{<:Real}`: The benchmark-weighted exposures, `observations × factors`.

# Related

  - [`factor_family_basis`](@ref)
"""
function weighted_family_exposures(Ms::Arr3Num, bw::MatNum, Tf::Type{<:Real})
    T, N, K = size(Ms)
    c = zeros(Tf, T, K)
    w = zeros(Tf, N)
    for t in 1:T
        s = zero(Tf)
        for i in 1:N
            b = bw[t, i]
            if isfinite(b)
                @argcheck(b >= zero(b),
                          DomainError(b, "every finite entry of bw must be >= 0"))
                w[i] = Tf(b)
            else
                w[i] = zero(Tf)
            end
            s += w[i]
        end
        @argcheck(s > zero(s),
                  ArgumentError("the benchmark weights of observation $t sum to $s, and they must sum to a strictly positive number once a non-finite weight is read as zero"))
        for k in 1:K, i in 1:N
            x = Ms[t, i, k]
            if isfinite(x)
                c[t, k] += w[i] * Tf(x) / s
            end
        end
    end
    return c
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the position within a family of the member the re-basis drops.

# Arguments

  - `drop`: Name of the member to drop, or `nothing` for the automatic choice.
  - `nm::AbstractString`: Label of the family, used in the messages.
  - `idx::AbstractVector{<:Integer}`: Raw-axis indices of the family's members.
  - `nf::VecStr`: Names of the raw factor axis.
  - `c::MatNum`: Benchmark-weighted exposures, `observations × factors`.

# Validation

  - A stated `drop` appears in `nf`, and belongs to the family that names it.

# Returns

  - `d::Int`: Position within `idx` of the dropped member. The automatic choice takes the member with the largest sum of absolute benchmark-weighted exposures over the observations, and the first such member on a tie.

# Related

  - [`factor_family_basis`](@ref)
  - [`weighted_family_exposures`](@ref)
"""
function resolve_dropped_member(drop::AbstractString, nm::AbstractString,
                                idx::AbstractVector{<:Integer}, nf::VecStr, ::MatNum)::Int
    k = findfirst(isequal(drop), nf)
    @argcheck(!isnothing(k),
              ArgumentError("the dropped factor $drop of family $nm names no factor on the raw axis"))
    d = findfirst(isequal(k), idx)
    @argcheck(!isnothing(d),
              ArgumentError("the dropped factor $drop does not belong to family $nm"))
    return d
end
function resolve_dropped_member(::Nothing, ::AbstractString, idx::AbstractVector{<:Integer},
                                ::VecStr, c::MatNum)::Int
    best = 1
    top = -one(eltype(c))
    for (p, k) in enumerate(idx)
        s = zero(eltype(c))
        for t in axes(c, 1)
            s += abs(c[t, k])
        end
        if s > top
            top = s
            best = p
        end
    end
    return best
end

export FactorFamilyBasis, factor_family_basis
