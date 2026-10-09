"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends the rows of a block to a history, or starts the history with them.

A history that a step appended to is a view of the first rows of a backing array, whose other rows are spare. The append writes the rows of the block into the spare rows in place, and gives a longer view of the same backing, so a step costs the rows of the block and no copy of the history. A backing with too few spare rows is replaced by one of twice the rows, as [`reserve_sample_buffer`](@ref) grows the backing of a [`SampleBufferState`](@ref). Each state keeps the view of its own rows, so an earlier state reads its own rows after a later step. The caller makes sure that the state is the newest one of its lineage, with [`cross_sectional_carry_own`](@ref), so no append writes a row that another state reads.

# Arguments

  - `a`: The history, or `nothing`.
  - `b`: The rows of the block.

# Returns

  - `h`: `a` followed by `b` along the observation axis.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_spare_backing`](@ref)
  - [`cross_sectional_fold_join`](@ref)
"""
function cross_sectional_fold_append(::Nothing, b)
    return b
end
function cross_sectional_fold_append(a::AbstractArray{<:Any, D},
                                     b::AbstractArray{<:Any, D}) where {D}
    n = size(a, 1)
    k = n + size(b, 1)
    T = promote_type(eltype(a), eltype(b))
    P = cross_sectional_spare_backing(a, T, k)
    if isnothing(P)
        P = similar(b, T, (2 * k, Base.tail(size(a))...))
        selectdim(P, 1, 1:n) .= a
    end
    selectdim(P, 1, (n + 1):k) .= b
    return view(P, Base.OneTo(k), ntuple(_ -> Colon(), D - 1)...)
end
function cross_sectional_fold_append(a::FactorFamilyBasis, b::FactorFamilyBasis)
    return FactorFamilyBasis(; fnm = a.fnm, fi = a.fi, di = a.di,
                             ratios = cross_sectional_fold_append(a.ratios, b.ratios),
                             K = a.K)
end
function cross_sectional_fold_append(a::CrossSectionalRegression,
                                     b::CrossSectionalRegression)
    return CrossSectionalRegression(; f = cross_sectional_fold_append(a.f, b.f),
                                    eps = cross_sectional_fold_append(a.eps, b.eps),
                                    n = cross_sectional_fold_append(a.n, b.n), b = nothing,
                                    h1 = cross_sectional_fold_append(a.h1, b.h1))
end
"""
    cross_sectional_fold_join(Ms, obs, b::AbstractArray, bo::Nothing) -> NamedTuple
    cross_sectional_fold_join(Ms::Nothing, obs::Nothing, b::AbstractArray{<:Any, 3},
                              bo::NamedTuple) -> NamedTuple
    cross_sectional_fold_join(Ms::AbstractArray{<:Any, 3}, obs::NamedTuple,
                              b::AbstractArray{<:Any, 3}, bo::NamedTuple) -> NamedTuple

Appends the rows of a block to the exposure history and to the observed factors that the carry fold of a Cross-Sectional Factor Prior carries.

The call with no data reads the exposure history with the observed exposures after the estimated ones, as [`cross_sectional_observed_append`](@ref) joins them. So under observed factors the two exposure histories are views of one backing, whose factor axis holds the estimated factors and then the observed ones. [`cross_sectional_joined_exposures`](@ref) then reads the joined history of the fitted observations as a view of the backing, and no step and no call with no data copies it. The join appends the joined rows of the block with [`cross_sectional_fold_append`](@ref), into the spare rows of the backing. Two histories that share no backing, as after the first block, join by a copy into a new backing, once. The observed returns append with [`cross_sectional_fold_append`](@ref), and the names and the family labels stay those of the first block.

# Arguments

  - `Ms`: The exposure history of the estimated factors, or `nothing` before the first block.
  - `obs`: The observed factors `(; Z, R, lv, nf, fam)` of the history, or `nothing`.
  - `b`: The exposures of the estimated factors of the block.
  - `bo`: The observed factors of the block, or `nothing` without an observed factor, when the method over `Nothing` appends `b` alone.

# Returns

  - `h::NamedTuple`: `(; Ms, obs)`, the two histories followed by the rows of the block.

# Related

  - [`cross_sectional_fold_append`](@ref)
  - [`cross_sectional_joined_exposures`](@ref)
  - [`cross_sectional_fold_histories`](@ref)
"""
function cross_sectional_fold_join(Ms, obs, b::AbstractArray, ::Nothing)
    return (; Ms = cross_sectional_fold_append(Ms, b), obs = obs)
end
function cross_sectional_fold_join(::Nothing, ::Nothing, b::AbstractArray{<:Any, 3},
                                   bo::NamedTuple)
    return cross_sectional_fold_split(cat(b, bo.Z; dims = 3), size(b, 3), bo.R, bo)
end
function cross_sectional_fold_join(Ms::AbstractArray{<:Any, 3}, obs::NamedTuple,
                                   b::AbstractArray{<:Any, 3}, bo::NamedTuple)
    J = cross_sectional_fold_append(cross_sectional_joined_exposures(Ms, obs.Z),
                                    cat(b, bo.Z; dims = 3))
    return cross_sectional_fold_split(J, size(b, 3),
                                      cross_sectional_fold_append(obs.R, bo.R), obs)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Splits a joined exposure history into the history of the estimated factors and the observed factors.

Both histories are views of the backing of `J` with a `Base.OneTo` row index, so the next append finds the spare rows of the backing, as [`cross_sectional_spare_backing`](@ref) states. A view of a view would carry a `UnitRange` row index, which marks no backing of the library.

# Arguments

  - `J`: The joined exposure history, `observations × assets × factors`, the estimated factors first.
  - `K`: The number of estimated factors.
  - `R`: The observed returns of the history.
  - `o`: The observed factors whose names and family labels the history keeps.

# Returns

  - `h::NamedTuple`: `(; Ms, obs)`, as [`cross_sectional_fold_join`](@ref) states it.

# Related

  - [`cross_sectional_fold_join`](@ref)
"""
function cross_sectional_fold_split(J::AbstractArray{<:Any, 3}, K::Integer,
                                    R::AbstractMatrix, o::NamedTuple)
    P = parent(J)
    k = Base.OneTo(size(J, 1))
    return (; Ms = view(P, k, :, 1:K),
            obs = (; Z = view(P, k, :, (K + 1):size(P, 3)), R = R, lv = o.lv, nf = o.nf,
                   fam = o.fam))
end
"""
    cross_sectional_spare_backing(a::AbstractArray, T::Type, k::Integer)
    cross_sectional_spare_backing(a::SubArray{<:Any, <:Any, <:Union{Array, BitArray},
                                               <:Tuple{Base.OneTo{Int}, Vararg{Base.Slice}}},
                                  T::Type, k::Integer)

Returns the backing array of a history when it holds `k` rows of element type `T`, or `nothing`.

[`cross_sectional_fold_append`](@ref) makes every backing, and gives a view of its first rows with a `Base.OneTo` row index. Only such a view has a backing that the append can write into. Any other array, a history that a fit of every observation made or a view of the caller's data, has none, so the append copies it into a new backing.

# Arguments

  - `a`: The history.
  - `T`: The element type of the history after the append.
  - `k`: The number of rows of the history after the append.

# Returns

  - `P::Option{<:AbstractArray}`: The backing array, or `nothing`.

# Related

  - [`cross_sectional_fold_append`](@ref)
"""
function cross_sectional_spare_backing(::AbstractArray, ::Type, ::Integer)
    return nothing
end
function cross_sectional_spare_backing(a::SubArray{<:Any, <:Any, <:Union{Array, BitArray},
                                                   <:Tuple{Base.OneTo{Int},
                                                           Vararg{Base.Slice}}}, ::Type{T},
                                       k::Integer) where {T}
    P = parent(a)
    return eltype(P) === T && size(P, 1) >= k ? P : nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the state of the carry fold of a Cross-Sectional Factor Prior the newest state of its lineage, before a step.

A step appends to the histories and to the buffer `buf` in place, into the spare rows of their backings, with [`cross_sectional_fold_append`](@ref). A state that a later step passed shares those backings with the later state, whose rows sit in the spare rows of the earlier one. So a step of the earlier state takes a copy of it first, which starts a lineage of its own. The later state and every Result read out of it keep their rows. The factor prior and the variance estimators fold in place, as every fold does, so the copy holds their state after the later step. To fold one state into two streams, copy it with [`partial_fit`](@ref) before the first of them.

# Arguments

  - `st`: The state before the step.

# Returns

  - `st::CrossSectionalCarryState`: `st` when its `tip` counts its own observations, and a copy of it otherwise.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`cross_sectional_fold_append`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`partial_fit!`](@ref)
"""
function cross_sectional_carry_own(st::CrossSectionalCarryState)
    return st.tip[] == st.buf.n ? st : copy(st)
end
