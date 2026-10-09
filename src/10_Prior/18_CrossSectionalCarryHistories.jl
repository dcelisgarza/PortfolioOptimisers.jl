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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuilds a [`CrossSectionalCarryState`](@ref) with some of its fields replaced.

# Arguments

  - `st`: The state.
  - `kw`: The fields to replace, by name.

# Returns

  - `st::CrossSectionalCarryState`: The new state.

# Related

  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_carry_with(st::CrossSectionalCarryState, kw::NamedTuple)
    fns = fieldnames(CrossSectionalCarryState)
    return CrossSectionalCarryState(;
                                    merge(NamedTuple{fns}(map(f -> getfield(st, f), fns)),
                                          kw)...)
end
"""
    cross_sectional_move_basis(unseen::Union{ZeroUnseenMember, SolvedUnseenMember},
                               cre::CrossSectionalLinearRegression, st, families)
        -> Option{<:FactorFamilyBasis}
    cross_sectional_move_basis(unseen::Union{ZeroUnseenMember, SolvedUnseenMember},
                               cre::CrossSectionalTargetRegression, st, families)
        -> Option{<:FactorFamilyBasis}
    cross_sectional_move_basis(unseen::AbstractUnseenMemberRule,
                               cre::AbstractCrossSectionalRegressionEstimator, st, families)
        -> nothing

Returns the basis of the fitted observations under the dropped members that moved, when the move folds on the carry fold of a Cross-Sectional Factor Prior, and `nothing` otherwise.

The zero-sum condition of a family is the same set of factor returns for every dropped member. So a regression that minimises the weighted squared residuals gives the same raw factor returns in every basis, on each observation whose design has full rank. [`ZeroUnseenMember`](@ref) makes every observation with an Unseen Member identified, so its answer does not depend on the dropped member. [`SolvedUnseenMember`](@ref) keeps such an observation rank-deficient, and a dependent factor set makes an observation rank-deficient under every rule. The answer of a rank-deficient observation depends on the dropped member under every [`AbstractCrossSectionalSolveAlgorithm`](@ref) that answers it: the answer of least norm reads the coordinates of the basis, and so does the pivot of [`DependentColumnDrop`](@ref). So [`cross_sectional_move_solve`](@ref) solves it again in the new basis. A [`CrossSectionalTargetRegression`](@ref) fits a target, and a target can penalise its coefficients, so its answer can depend on the basis. [`is_basis_invariant`](@ref) answers for the target. A target that answers `true`, such as [`LinearModel`](@ref), gives the same raw factor returns in every basis on an observation of full rank, as a least-squares fit does. [`GeneralisedLinearModel`](@ref) answers `true`: its iterative fit takes the same iterates in every basis, so it gives the same raw factor returns to rounding. On a rank-deficient observation the answer of a target depends on the basis too, under each solve algorithm of the target, and [`cross_sectional_move_solve`](@ref) solves such an observation again through the target. So a move folds under [`CrossSectionalLinearRegression`](@ref), and under a target that answers `true`, with the two Unseen Member rules of the library. Every other pair fits every observation again.

# Algorithm

The method that Julia selects is the algorithm.

 1. Under [`ZeroUnseenMember`](@ref) or [`SolvedUnseenMember`](@ref), with [`CrossSectionalLinearRegression`](@ref), rewrite the basis with [`cross_sectional_rebase`](@ref).
 2. Under the same rules, with [`CrossSectionalTargetRegression`](@ref), rewrite the basis as step 1 does when [`is_basis_invariant`](@ref) answers `true` for `cre.tgt`, and answer `nothing` otherwise.
 3. Under any other pair, answer `nothing`.

# Arguments

  - `unseen`: The Unseen Member rule of the prior.
  - `cre`: The cross-sectional regression estimator of the prior.
  - `st`: The state, whose `fcb` is the basis of the last fit.
  - `families`: The families with their dropped members named.

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The basis under `families`, or `nothing` when the move does not fold.

# Related

  - [`cross_sectional_fold_move`](@ref)
  - [`cross_sectional_rebase`](@ref)
  - [`cross_sectional_move_solve`](@ref)
  - [`AbstractUnseenMemberRule`](@ref)
  - [`is_basis_invariant`](@ref)
"""
function cross_sectional_move_basis(::Union{ZeroUnseenMember, SolvedUnseenMember},
                                    ::CrossSectionalLinearRegression,
                                    st::CrossSectionalCarryState,
                                    families::AbstractVector{<:Pair})
    return cross_sectional_rebase(st.fcb, families, st.nf)
end
function cross_sectional_move_basis(::Union{ZeroUnseenMember, SolvedUnseenMember},
                                    cre::CrossSectionalTargetRegression,
                                    st::CrossSectionalCarryState,
                                    families::AbstractVector{<:Pair})
    if !is_basis_invariant(cre.tgt)
        return nothing
    end
    return cross_sectional_rebase(st.fcb, families, st.nf)
end
function cross_sectional_move_basis(::AbstractUnseenMemberRule,
                                    ::AbstractCrossSectionalRegressionEstimator, ::Any,
                                    ::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rewrites a Factor Family Basis with new dropped members, from its own ratios.

The ratio of a member is its benchmark-weighted exposure over the exposure of the dropped member. So the ratios under a new dropped member ``k^{\\prime}`` are the old ratios over the old ratio of ``k^{\\prime}``, and the ratio of the old dropped member ``k`` is the inverse of that ratio. The function reads no exposure, and its ratios equal the ratios of [`factor_family_basis`](@ref) to rounding.

# Mathematical definition

```math
\\begin{align}
r^{\\prime}_{t}(j) &= \\frac{r_{t}(j)}{r_{t}(k^{\\prime})}\\,, \\quad j \\in \\mathcal{F} \\setminus \\{k, k^{\\prime}\\}\\,, \\\\
r^{\\prime}_{t}(k) &= \\frac{1}{r_{t}(k^{\\prime})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:F_fam_att])
  - $(math_dict[:r_tj_fcb])
  - ``r^{\\prime}_{t}(j)``: Ratio of member ``j`` at observation ``t`` under the new dropped member ``k^{\\prime}``.

# Arguments

  - `fcb`: The Factor Family Basis.
  - `families`: Pairs of `family label => dropped member`, in the order of `fcb.fnm`, each member named.
  - `nf`: Names of the raw factor axis.

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The basis with the new dropped members, or `nothing` when a new dropped member has a zero benchmark-weighted exposure at an observation, where [`factor_family_basis`](@ref) refuses it.

# Related

  - [`cross_sectional_rebase_family`](@ref)
  - [`cross_sectional_fold_move`](@ref)
  - [`FactorFamilyBasis`](@ref)
"""
function cross_sectional_rebase(fcb::FactorFamilyBasis, families::AbstractVector{<:Pair},
                                nf::VecStr)
    di = map(j -> findfirst(isequal(last(families[j])), view(nf, fcb.fi[j])),
             eachindex(fcb.fi))
    blocks = map(j -> cross_sectional_rebase_family(fcb, j, di[j]), eachindex(fcb.fi))
    if any(isnothing, blocks)
        return nothing
    end
    return FactorFamilyBasis(; fnm = fcb.fnm, fi = fcb.fi, di = di,
                             ratios = reduce(hcat, blocks), K = fcb.K)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the block of the ratios of one Factor Family under a new dropped member, as [`cross_sectional_rebase`](@ref) states it.

# Algorithm

 1. Write the ratios of the family into a block with one column per member, and a column of ones at the old dropped member, whose ratio to itself is one.
 2. Answer `nothing` when the column of the new dropped member holds a zero.
 3. Divide every column but that one by it. A family whose member did not move divides by a column of ones, which gives its ratios unchanged.

# Arguments

  - `fcb`: The Factor Family Basis.
  - `j`: Position of the family in `fcb.fnm`.
  - `d`: Position of the new dropped member in `fcb.fi[j]`.

# Returns

  - `R::Option{<:Matrix}`: The block, one column per retained member in the order of `fcb.fi[j]`, or `nothing` when the old ratio of the new dropped member is zero at an observation.

# Related

  - [`cross_sectional_rebase`](@ref)
"""
function cross_sectional_rebase_family(fcb::FactorFamilyBasis, j::Integer, d::Integer)
    col = family_retained_indices(fcb, j)[3]
    R = ones(eltype(fcb.ratios), size(fcb.ratios, 1), length(fcb.fi[j]))
    R[:, axes(R, 2) .!= fcb.di[j]] = view(fcb.ratios, :, col)
    rho = R[:, d]
    if any(iszero, rho)
        return nothing
    end
    return R[:, axes(R, 2) .!= d] ./ rho
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Marks the factors that are not empty in each regression pass of the carry fold of a Cross-Sectional Factor Prior, under a basis whose dropped members moved.

A factor outside a Factor Family that moved keeps its column of the design, so it keeps its marks. The design of a family that moved changes, so the function builds the design of each fitted observation in the new basis, from the first one, until each member of such a family is marked in both passes. The first observations mark each member that the sample sees, so the function reads every observation only when a member stays empty.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, whose `fcb`, `lv1` and `lv` are those of the old basis.
  - `fcb`: The new basis, over the rows of `st.fcb`.

# Returns

  - `marks::NamedTuple`: `lv1` and `lv`, the marks of the two passes on the new reduced axis.

# Related

  - [`cross_sectional_fold_move`](@ref)
  - [`cross_sectional_move_row_marks`](@ref)
  - [`cross_sectional_live_factors`](@ref)
"""
function cross_sectional_move_marks(pe::CrossSectionalFactorPrior,
                                    st::CrossSectionalCarryState, fcb::FactorFamilyBasis)
    old = retained_factor_indices(st.fcb)
    new = retained_factor_indices(fcb)
    raw1 = falses(fcb.K)
    raw = falses(fcb.K)
    raw1[old] = st.lv1
    raw[old] = st.lv
    lv1 = raw1[new]
    lv = raw[new]
    mv = reduce(vcat,
                [family_retained_indices(fcb, j)[2]
                 for j in eachindex(fcb.fi) if fcb.di[j] != st.fcb.di[j]])
    lv1[mv] .= false
    lv[mv] .= false
    s = 0
    while s < size(st.W, 1) && !all(i -> lv1[i] & lv[i], mv)
        s += 1
        r = cross_sectional_move_row_marks(pe, st, fcb, s)
        lv1[mv] .|= r.lv1[mv]
        lv[mv] .|= r.lv[mv]
    end
    return (; lv1 = lv1, lv = lv)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Marks the factors that are not empty at one fitted observation of the carry fold of a Cross-Sectional Factor Prior, in a basis, in each regression pass.

The function runs the steps of [`cross_sectional_fold_regression`](@ref) that the marks read, on the one observation: the lagged exposures reduced in the basis, the eligibility mask, the first-pass weights, and the design of [`unseen_member_design`](@ref). The last pass reads the weights that the state carries, because they read the residuals, which do not depend on the basis.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state.
  - `fcb`: The basis, over the rows of `st.fcb`.
  - `s`: The fitted observation, as a row of `st.W`.

# Returns

  - `marks::NamedTuple`: `lv1` and `lv`, the marks of the observation in the two passes.

# Related

  - [`cross_sectional_move_marks`](@ref)
  - [`cross_sectional_live_factors`](@ref)
"""
function cross_sectional_move_row_marks(pe::CrossSectionalFactorPrior,
                                        st::CrossSectionalCarryState,
                                        fcb::FactorFamilyBasis, s::Integer)
    t = s + pe.lag
    sl = factor_basis_slice(fcb, s:s)
    B = st.Ms[s:s, :, :]
    Zl = reduce_exposures(sl, B)
    Xr = something(st.Xl, st.X)[t:t, :]
    mcl = cross_sectional_rows(st.mcap, s:s)
    msk = cross_sectional_eligible(Xr, Zl, view(st.emsk, t:t, :))
    cross_sectional_cap_finite!(msk, mcl)
    W1 = cs_weights_initial(pe.wa, mcl, msk)
    W = st.W[s:s, :]
    Z1 = unseen_member_design(pe.unseen, sl, B, Zl, W1).Z
    Z = unseen_member_design(pe.unseen, sl, B, Zl, W).Z
    return (; lv1 = cross_sectional_live_factors(Z1, Xr, W1),
            lv = cross_sectional_live_factors(Z, Xr, W))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a move of the dropped member of a Factor Family into the state of the carry fold of a Cross-Sectional Factor Prior, before the step fits its new observations.

A move runs no regression again over the observations of full rank. The raw factor returns of such an observation do not depend on the dropped member, as [`cross_sectional_move_basis`](@ref) states, so the factor returns under the new members are the columns of the raw factor returns that the new basis keeps. [`cross_sectional_move_solve`](@ref) solves each rank-deficient observation again in the new basis: an observation with an Unseen Member under [`SolvedUnseenMember`](@ref), or one with a dependent factor set. The residuals, the regression weights, the idiosyncratic variance, the standardised idiosyncratic returns and the Return Forecast rows do not depend on the dropped member either, so the state keeps them. The factor prior folds again over the new factor returns, because the default factor covariance is not separable by column: its regime multiplier reads every column. That fold reads every fitted observation, so its cost grows with the stream, and so does the test of the rank of [`cross_sectional_move_solve`](@ref). [`cross_sectional_move_marks`](@ref) gives the marks of the Empty Factors.

The state equals the state of [`cross_sectional_fold_refit`](@ref) under the new members to rounding, on each fitted observation whose design has full rank on its columns that are not zero, and on each observation that the fold solves again. Under a [`CrossSectionalTargetRegression`](@ref) of a [`GeneralisedLinearModel`](@ref), the fit is iterative, and the two states agree to rounding when the two bases stop at the same iterate. `GLM` stops when the change in the deviance falls below `max(rtol * deviance, atol)`, with `rtol = 1e-6` and `atol = 1e-6` by default, and the `kwargs` of the target set them. A tolerance near the rounding of the deviance lets rounding stop the two bases one iteration apart at an observation, and the states then differ by that iteration there.

# Algorithm

 1. Answer `nothing` when `pe.ve` does not fold, as [`supports_partial_fit`](@ref) answers, because the step then fits every observation again. Answer `st` when no dropped member moved.
 2. Rewrite the basis of the observations before the step with [`cross_sectional_move_basis`](@ref). Answer `nothing` when the move does not fold, or when a new dropped member has a zero benchmark-weighted exposure, so the fit of every observation refuses it as the batch fit does. [`cross_sectional_rebase_state`](@ref) takes steps 3 to 7.
 3. Expand the factor returns of the fitted observations onto the raw axis with [`cross_sectional_expand`](@ref), and keep the columns of the new basis.
 4. Mark the Empty Factors with [`cross_sectional_move_marks`](@ref).
 5. Solve the observations whose answer depends on the basis again, with [`cross_sectional_move_solve`](@ref).
 6. Fold the factor prior over the factor returns of [`cross_sectional_fold_factor_returns`](@ref) with [`cross_sectional_refold_factors`](@ref).
 7. Reduce the exposures of the last `lag` observations in the new basis, which the regression of the new observations reads.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, whose histories hold the new observations as their last `m` rows.
  - `families`: The families with their dropped members named, or `nothing`.
  - `m`: Number of new observations.

# Returns

  - `st::Option{CrossSectionalCarryState}`: The state in the basis of `families`, or `nothing` when the step fits every observation again.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_fold_choice`](@ref)
  - [`cross_sectional_fold_step`](@ref)
  - [`BatchChoice`](@ref)
"""
function cross_sectional_fold_move(pe::CrossSectionalFactorPrior,
                                   st::CrossSectionalCarryState,
                                   families::Option{<:AbstractVector{<:Pair}}, m::Integer)
    if !supports_partial_fit(pe.ve)
        return nothing
    end
    if families == st.families
        return st
    end
    return cross_sectional_rebase_state(pe, st,
                                        cross_sectional_move_basis(pe.unseen, pe.cre, st,
                                                                   families), m)
end
"""
    cross_sectional_rebase_state(pe::CrossSectionalFactorPrior, st::CrossSectionalCarryState,
                                 fcb::Nothing, m::Integer) -> nothing
    cross_sectional_rebase_state(pe::CrossSectionalFactorPrior, st::CrossSectionalCarryState,
                                 fcb::FactorFamilyBasis, m::Integer) -> CrossSectionalCarryState

Rewrites the state of the carry fold of a Cross-Sectional Factor Prior in a new Factor Family Basis, as steps 3 to 7 of [`cross_sectional_fold_move`](@ref) state. The state records the dropped members of the basis with [`cross_sectional_dropped_names`](@ref), as the fit of every observation records them.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, whose histories hold the new observations as their last `m` rows.
  - `fcb`: The basis of the observations before the step under the new dropped members, or `nothing` when the move does not fold.
  - `m`: Number of new observations.

# Returns

  - `st::Option{CrossSectionalCarryState}`: The state in the basis `fcb`, or `nothing` without a basis.

# Related

  - [`cross_sectional_fold_move`](@ref)
  - [`cross_sectional_move_basis`](@ref)
  - [`cross_sectional_move_solve`](@ref)
"""
function cross_sectional_rebase_state(::CrossSectionalFactorPrior,
                                      ::CrossSectionalCarryState, ::Nothing, ::Integer)
    return nothing
end
function cross_sectional_rebase_state(pe::CrossSectionalFactorPrior,
                                      st::CrossSectionalCarryState, fcb::FactorFamilyBasis,
                                      m::Integer)
    Tf = size(st.Ms, 1) - m
    r = (pe.lag + 1):Tf
    z = (Tf - pe.lag + 1):Tf
    fr = cross_sectional_expand(st.fcb, r, pe.lag, st.csr.f)
    marks = cross_sectional_move_marks(pe, st, fcb)
    f = cross_sectional_move_solve(pe, st,
                                   (; fcb = fcb, f = fr[:, retained_factor_indices(fcb)],
                                    lv = marks.lv))
    cb = cross_sectional_observed_block(st.obs, 1:Tf, r, st.buf.n - size(st.Ms, 1))
    fo = cross_sectional_fold_factor_returns(f, marks.lv, cb)
    (; eps, n, b, h1) = st.csr
    return cross_sectional_carry_with(st,
                                      (;
                                       families = cross_sectional_dropped_names(pe.families,
                                                                                fcb, st.nf),
                                       fcb = fcb,
                                       Z = reduce_exposures(factor_basis_slice(fcb, z),
                                                            st.Ms[z, :, :]),
                                       csr = CrossSectionalRegression(; f = f, eps = eps,
                                                                      n = n, b = b,
                                                                      h1 = h1),
                                       lv1 = marks.lv1, lv = marks.lv,
                                       pe = cross_sectional_refold_factors(pe.pe, fo,
                                                                           st.seed)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Solves again, in a new Factor Family Basis, each fitted observation whose factor returns depend on the dropped members, on the carry fold of a Cross-Sectional Factor Prior.

A change of the dropped member is an invertible change of the coefficients of the family. So the old and the new reduced design of an observation span the same columns, and the fitted values, the residuals and the regression weights do not depend on the dropped member. On an observation of full rank, the factor returns do not depend on it either.

On a rank-deficient observation [`PseudoInverseFallback`](@ref) takes the answer of least norm, and the norm reads the coordinates of the basis. The pivot of [`DependentColumnDrop`](@ref) reads them too. So the answer in the new basis is not a selection of the answer in the old basis, and the function solves such an observation again with the last-pass weights `st.W`, which the stored regression read, on the factors that `mv.lv` marks, as the batch fit solves it. Two cases make such an observation. Under [`SolvedUnseenMember`](@ref) an observation with an Unseen Member is rank-deficient. A dependent factor set makes it rank-deficient under every rule, for example a beta that shrinks to the mean of its industry: where every industry shrinks fully, the style column is a function of the industry columns.

A column that is zero at an observation is an Unseen Member under [`ZeroUnseenMember`](@ref), or a factor that no eligible asset loads on there. The solve gives it a return of zero in every basis, so the test leaves it out, and a design whose other columns have full rank keeps its factor returns. The test reads the exposures of every fitted observation, so its cost grows with the stream.

# Algorithm

 1. Build the design of each fitted observation `s` in `mv.fcb` with [`unseen_member_design`](@ref), on the factors that `mv.lv` marks, and scale its rows of positive weight in `st.W[s, :]` by the square root of the weight.
 2. Keep the factor returns of the observation when [`cross_sectional_rank`](@ref) of the columns of that design that are not zero equals their number.
 3. Otherwise, regress the returns of the observation with [`cross_sectional_live_regression`](@ref), under the policy of `pe.cre`, and map the answer with [`unseen_member_returns`](@ref). Write the factor returns into the row `s` of `mv.f`.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, whose `fcb` is the basis of the last fit.
  - `mv`: The move, `(; fcb, f, lv)`: the new basis over the rows of `st.fcb`; the factor returns of the fitted observations in `fcb`, the columns of the raw factor returns that `fcb` keeps, which the function modifies in place; and the mark of the factors that are not empty in the last pass, in `fcb`.

# Returns

  - `f::MatNum`: The factor returns in `mv.fcb`.

# Related

  - [`cross_sectional_rebase_state`](@ref)
  - [`cross_sectional_move_basis`](@ref)
  - [`cross_sectional_move_row_marks`](@ref)
  - [`PseudoInverseFallback`](@ref)
"""
function cross_sectional_move_solve(pe::CrossSectionalFactorPrior,
                                    st::CrossSectionalCarryState, mv::NamedTuple)
    (; fcb, f, lv) = mv
    Xs = something(st.Xl, st.X)
    for s in axes(f, 1)
        sl = factor_basis_slice(fcb, s:s)
        B = view(st.Ms, s:s, :, :)
        W = view(st.W, s:s, :)
        ud = unseen_member_design(pe.unseen, sl, B, reduce_exposures(sl, B), W)
        idx = findall(>(zero(eltype(W))), view(W, 1, :))
        A = view(ud.Z, 1, idx, lv) .* sqrt.(view(W, 1, idx))
        nz = map(c -> any(!iszero, c), eachcol(A))
        if cross_sectional_rank(view(A, :, nz)) < count(nz)
            t = s + pe.lag
            csr = cross_sectional_live_regression(pe.cre, ud.Z, view(Xs, t:t, :), W, lv).csr
            f[s, :] = view(unseen_member_returns(csr, ud.P).f, 1, :)
        end
    end
    return f
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the factor prior of the carry fold of a Cross-Sectional Factor Prior over the factor returns of every fitted observation: the first `seed` observations as one block, and every later observation alone, as the first fit folds them.

# Arguments

  - `pe`: The factor prior with no fold.
  - `f`: The factor returns of every fitted observation, on the factors that are not empty.
  - `seed`: Number of fitted observations to fold as one block.

# Returns

  - `pe`: The factor prior after every fitted observation.

# Related

  - [`cross_sectional_fold_factors`](@ref)
  - [`cross_sectional_fold_refit`](@ref)
  - [`cross_sectional_fold_move`](@ref)
"""
function cross_sectional_refold_factors(pe::AbstractPriorEstimator, f::MatNum,
                                        seed::Integer)
    pf = cross_sectional_fold_factors(pe, view(f, 1:seed, :))
    if seed < size(f, 1)
        pf = cross_sectional_fold_factors(pf, view(f, (seed + 1):size(f, 1), :))
    end
    return pf
end
"""
    cross_sectional_alive_folds(cre::AbstractCrossSectionalRegressionEstimator) -> Bool
    cross_sectional_alive_folds(cre::Union{CrossSectionalLinearRegression,
                                           CrossSectionalTargetRegression}) -> Bool
    cross_sectional_alive_folds(alg::AbstractCrossSectionalSolveAlgorithm) -> Bool
    cross_sectional_alive_folds(alg::Union{PseudoInverseFallback, MinimumNormSolve,
                                           DependentColumnDrop}) -> Bool

Answers whether a step of the carry fold of a Cross-Sectional Factor Prior folds a factor that comes alive, under the regression estimator `cre`, or under its solve algorithm `alg`.

A factor that comes alive at a step was empty at every fitted observation: its exposure was zero at every pair of positive weight. So the batch fit regresses each fitted observation over a design with a zero column, where the step fitted it without that column. The fold keeps the answer of the step when the zero column does not change it.

  - Under [`PseudoInverseFallback`](@ref) the design with the zero column is rank-deficient, and the solve takes its answer of least norm. A zero column adds nothing to the norm or to the fit, so that answer equals the answer without the column to rounding, and gives the column a return of zero. [`MinimumNormSolve`](@ref) takes the same answer. A [`CrossSectionalTargetRegression`](@ref) under [`PseudoInverseFallback`](@ref) projects its answer onto the row space of the weighted design, which holds a zero at the zero column, so it gives the same answer.
  - [`DependentColumnDrop`](@ref) drops the zero column by the pivot of its rank test, and gives it a return of zero. It fits the columns that remain, in their order, which are the columns of the step.
  - [`RankDeficiencyRefusal`](@ref) refuses the design with the zero column. [`UncheckedSolve`](@ref) throws on such a design when it is square, and hands it to the target of a [`CrossSectionalTargetRegression`](@ref) unchecked. So the batch fit can refuse, or answer otherwise, where the fold would keep the answer. Every other estimator and algorithm is not known to keep the answer. The function answers `false` for them, and the step fits every carried observation again, as the batch fit does.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`CrossSectionalLinearRegression`](@ref) or a [`CrossSectionalTargetRegression`](@ref) answers for its solve algorithm `cre.alg`: `true` under [`PseudoInverseFallback`](@ref), [`MinimumNormSolve`](@ref) and [`DependentColumnDrop`](@ref), and `false` under every other algorithm.
 2. Every other estimator answers `false`.

# Arguments

  - `cre`: The cross-sectional regression estimator of the prior.
  - `alg`: Its solve algorithm.

# Returns

  - `folds::Bool`: `true` when the step keeps the fitted observations.

# Related

  - [`cross_sectional_fold_mark`](@ref)
  - [`cross_sectional_step_factors`](@ref)
  - [`cross_sectional_solve`](@ref)
"""
function cross_sectional_alive_folds(::AbstractCrossSectionalRegressionEstimator)::Bool
    return false
end
function cross_sectional_alive_folds(cre::Union{CrossSectionalLinearRegression,
                                                CrossSectionalTargetRegression})::Bool
    return cross_sectional_alive_folds(cre.alg)
end
function cross_sectional_alive_folds(::AbstractCrossSectionalSolveAlgorithm)::Bool
    return false
end
function cross_sectional_alive_folds(::Union{PseudoInverseFallback, MinimumNormSolve,
                                             DependentColumnDrop})::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the factor returns of a step into the factor prior of the carry fold of a Cross-Sectional Factor Prior.

The factor prior folds the factor returns of the factors that are not empty. A step that marks no other factor folds the factor returns of its new observations into the folded prior. A step where a factor comes alive adds a column to the factor returns. The default factor covariance is not separable by column, so the function folds the factor prior again over every fitted factor return, with a return of zero for the new factor at each observation before the step. That fold reads every fitted observation, so its cost grows with the stream, and it happens at most once for each factor.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state before the step, whose histories hold the new observations as their last rows.
  - `csr`: The regression of every fitted observation, the new observations last.
  - `lv`: The mark of the factors that are not empty in the last pass, after the step.

# Returns

  - `pe`: The factor prior after the step.

# Related

  - [`cross_sectional_fold_step`](@ref)
  - [`cross_sectional_fold_factors`](@ref)
  - [`cross_sectional_refold_factors`](@ref)
  - [`cross_sectional_alive_folds`](@ref)
"""
function cross_sectional_step_factors(pe::CrossSectionalFactorPrior,
                                      st::CrossSectionalCarryState,
                                      csr::CrossSectionalRegression,
                                      lv::AbstractVector{Bool})
    Tf = size(st.Ms, 1)
    if lv != st.lv
        cb = cross_sectional_observed_block(st.obs, 1:Tf, (pe.lag + 1):Tf, st.buf.n - Tf)
        return cross_sectional_refold_factors(pe.pe,
                                              cross_sectional_fold_factor_returns(csr.f, lv,
                                                                                  cb),
                                              st.seed)
    end
    s = size(st.csr.f, 1)
    cb = cross_sectional_observed_block(st.obs, 1:Tf, (pe.lag + s + 1):Tf, st.buf.n - Tf)
    return cross_sectional_fold_factors(st.pe,
                                        cross_sectional_fold_factor_returns(view(csr.f,
                                                                                 (s + 1):size(csr.f,
                                                                                              1),
                                                                                 :), lv,
                                                                            cb))
end
