"""
$(DocStringExtensions.TYPEDEF)

Compact change of basis between the raw factor axis and the reduced axis a re-based Factor Family is fitted in.

A Factor Family whose one-hot exposures are collinear with a global factor carries one redundant column. The re-basis drops one member of the family and rewrites the family in an equivalent basis of full column rank, in which the benchmark-weighted factor returns of the family sum to zero. The change of basis is time-varying, and this result stores it compactly: per observation it holds the ratios of the benchmark-weighted exposures of the retained members to that of the dropped member, never the dense basis matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorFamilyBasis(; fnm::VecStr, fi::AbstractVector{<:AbstractVector{<:Integer}},
                      di::AbstractVector{<:Integer}, ratios::MatNum, K::Integer)

Keywords correspond to the struct's fields.

## Validation

  - `K > 0` and `!isempty(fnm)`.
  - `fnm`, `fi` and `di` have the same length.
  - Every family holds at least two members, its members are unique, and every member lies in `1:K`.
  - No factor belongs to two families.
  - `di[j]` indexes `fi[j]`.
  - `ratios` has one column per retained member of a constrained family, `sum(length(fi[j]) - 1)` in all.
  - Every entry of `ratios` is finite.
  - `K` is greater than the number of families, so the reduced axis is not empty.

# Mathematical definition

For a family of ``m`` members the benchmark-weighted exposure of member ``j`` at observation ``t`` is

```math
c_t(j) = \\sum_{i} w^{b}_{t,i} \\, B_{t,i,j},
```

and the family's factor returns satisfy the zero-sum condition ``c_t^{\\top} f_{\\mathrm{family}}(t) = 0``. Dropping member ``k`` and writing ``r_t(j) = c_t(j) / c_t(k)`` for ``j \\ne k`` parameterises the reduced basis, and `ratios` stores those ``r_t(j)``.

Where:

  - ``w^{b}_{t,i}``: benchmark weight of asset ``i`` at observation ``t``.
  - ``B_{t,i,j}``: exposure of asset ``i`` to factor ``j`` at observation ``t``.
  - ``f_{\\mathrm{family}}(t)``: factor returns of the family at observation ``t``.

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
    Label of each constrained Factor Family, one entry per family, in the order the families were requested. That order fixes the column order of `ratios`.
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
        @argcheck(K > length(fnm),
                  ArgumentError("K ($K) must exceed the number of constrained families ($(length(fnm))), because each family drops one factor and the reduced axis cannot be empty"))
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

A dropped factor has no reduced-axis index, and the map answers `0` for it.

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

The three vectors are aligned: entry `p` names the same retained member of family `j` in the raw axis, in the reduced axis and in `ratios`.

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

Attribution reads the basis on the observations of a window, and it drops the lag by slicing the observation axis rather than by rebuilding the basis.

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

# Algorithm

 1. Normalise `bw` so the benchmark weights of each observation sum to one, reading a non-finite weight as zero.
 2. Take the benchmark-weighted exposure `c_t(j)` of every raw factor, reading a non-finite exposure as zero.
 3. For each requested family, resolve the member indices from `fam`, and resolve the dropped member. A stated drop is looked up in `nf`. An unstated drop is the member with the largest time-average absolute benchmark-weighted exposure, which keeps the ratios moderate.
 4. Divide the benchmark-weighted exposures of the retained members by that of the dropped member, giving the family's block of `ratios`.
 5. Build the [`FactorFamilyBasis`](@ref) from the resolved families and the concatenated blocks, which re-runs every guard of the constructor.

# Arguments

  - `families`: Pairs of `family label => dropped factor name`, in the order the families take columns of `ratios`. A `nothing` on the right asks for the automatic choice of step 3.
  - `Ms::Arr3Num`: Exposure history, `observations × assets × factors`.
  - `bw::MatNum`: Benchmark weight history, `observations × assets`.
  - `nf::VecStr`: Names of the raw factor axis, of length `size(Ms, 3)`.
  - `fam::VecStr`: Family label of each raw factor, of length `size(Ms, 3)`.

# Validation

  - `families` is not empty, and no family label appears twice.
  - `nf` and `fam` are as long as the factor axis of `Ms`, and `nf` does not repeat a name.
  - `bw` matches `Ms` on the observation and asset axes, and every finite weight is non-negative.
  - Every observation carries a strictly positive benchmark weight sum.
  - Every requested family label appears in `fam`, and holds at least two factors.
  - A stated dropped factor name appears in `nf`, and belongs to the family that names it.
  - The rules of [`FactorFamilyBasis`](@ref), which refuse a non-finite ratio.

A dropped factor whose benchmark-weighted exposure is zero at some observation produces a non-finite ratio, and the constructor refuses it. One that is merely small produces a large but finite ratio, which is accepted and ill-conditioned: state the drop, or let step 3 choose it.

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
    Tf = promote_type(float(real(eltype(Ms))), float(real(eltype(bw))))
    c = weighted_family_exposures(Ms, bw, Tf)
    fnm = String[]
    fi = Vector{Int}[]
    di = Int[]
    blocks = Matrix{Tf}[]
    for pr in families
        nm = String(first(pr))
        @argcheck(nm ∉ fnm, ArgumentError("family $nm appears more than once in families"))
        idx = findall(isequal(nm), fam)
        @argcheck(!isempty(idx),
                  ArgumentError("family $nm names no factor. The declared families are $(unique(fam))"))
        @argcheck(length(idx) >= 2,
                  ArgumentError("family $nm holds $(length(idx)) factor, and a constrained family needs at least two"))
        d = resolve_dropped_member(last(pr), nm, idx, nf, c)
        ret = [i for i in idx if i != idx[d]]
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

  - `d::Int`: Position within `idx` of the dropped member.

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
