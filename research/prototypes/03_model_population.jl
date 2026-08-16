# =============================================================================
# Prototype 3 — A population of models, and the disagreement between them.
#
# Purpose
#   `PopulationPredictionResult` exists in
#   `src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl`, and it
#   already carries `sort_by_measure` and `quantile_by_measure`. Its members
#   are **cross-validation paths**: one data split, many folds. What is absent
#   is a population whose members are **models**: one data set, many defensible
#   estimators.
#
#   The distinction matters because the two answer different questions. A path
#   population answers "would this model have worked?". A model population
#   answers "does the answer depend on a choice I could not justify?".
#
#   The meta-optimisers `Stacking`, `SubsetResampling` and `NestedClustered`
#   already run many subproblems, but every one of them *combines* the answers
#   into a single weight vector. This file reports the *spread* instead.
#
# Status
#   Standalone. Depends on `LinearAlgebra` and `Statistics` only.
#
# Notation used throughout this file
#   N    Number of assets.
#   M    Number of members in the population.
#   W    Weight matrix, `N x M`. Column `m` is member `m`'s portfolio.
#   pi   Member probabilities, length `M`, non-negative and summing to one.
#        Uniform unless the caller has a reason otherwise.
#   wbar Consensus portfolio, length `N`, equal to `W * pi`.
#   sig  Covariance matrix, `N x N`.
#
# Sources
#   Krogh, A. and Vedelsby, J. (1995). Neural network ensembles, cross
#     validation, and active learning. Advances in Neural Information
#     Processing Systems 7, 231-238. The ambiguity decomposition reproduced
#     exactly in `ambiguity_decomposition` below.
#   Cremers, K. J. M. and Petajisto, A. (2009). How active is your fund
#     manager? A new measure that predicts performance. Review of Financial
#     Studies 22(9), 3329-3365. Active share.
#   Michaud, R. O. (1998). Efficient Asset Management. Harvard Business School
#     Press. Portfolio resampling, and the argument that a single optimised
#     portfolio overstates its own precision.
#   Jorion, P. (1992). Portfolio optimization in practice. Financial Analysts
#     Journal 48(1), 68-74. The resampling reading of estimation error.
# =============================================================================
module ModelPopulation

using LinearAlgebra, Statistics

export PortfolioPopulation, consensus_weights, weight_dispersion, active_share,
       disagreement_matrix, mean_disagreement, ambiguity_decomposition,
       effective_number_of_models, support_stability, population_report

"""
    PortfolioPopulation{T}

A set of portfolios produced by different models on the same data.

# Fields

  - `W::Matrix{T}`: Weight matrix, `N x M`. Column `m` holds member `m`.
  - `names::Vector{String}`: Member labels, length `M`. A label must say what
    the member *is*, because the output of every function here is only as
    readable as these strings.
  - `probs::Vector{T}`: Member probabilities, length `M`, non-negative and
    summing to one. Uniform by default.

# Notes

  - **The members must be comparable.** Every column indexes the same assets in
    the same order and answers the same question. A population that mixes a
    minimum-variance portfolio with a maximum-return one measures the
    difference between two objectives, not disagreement about one.
"""
struct PortfolioPopulation{T <: Real}
    W::Matrix{T}
    names::Vector{String}
    probs::Vector{T}
    function PortfolioPopulation(W::Matrix{T}, names::Vector{String},
                                 probs::Vector{T}) where {T <: Real}
        M = size(W, 2)
        if length(names) != M
            throw(DimensionMismatch("W has $(M) columns but $(length(names)) names were given"))
        end
        if length(probs) != M
            throw(DimensionMismatch("W has $(M) columns but $(length(probs)) probabilities were given"))
        end
        if any(<(0), probs)
            throw(DomainError(probs, "member probabilities must be non-negative"))
        end
        if !isapprox(sum(probs), one(T); atol = sqrt(eps(T)))
            throw(DomainError(sum(probs), "member probabilities must sum to one"))
        end
        return new{T}(W, names, probs)
    end
end
function PortfolioPopulation(W::AbstractMatrix{<:Real};
                             names::AbstractVector{<:AbstractString} = ["member $(m)"
                                                                        for m in
                                                                            1:size(W, 2)],
                             probs::Union{Nothing, AbstractVector{<:Real}} = nothing)
    Wm = Matrix(float.(W))
    T = eltype(Wm)
    p = isnothing(probs) ? fill(one(T) / size(Wm, 2), size(Wm, 2)) : Vector{T}(probs)
    return PortfolioPopulation(Wm, String.(collect(names)), p)
end

"""
    consensus_weights(pop::PortfolioPopulation) -> Vector

Return the probability-weighted average portfolio `wbar = W * pi`.

# Arguments

  - `pop`: The population.

# Returns

  - `wbar::Vector`, length `N`.

# Notes

  - The consensus is a portfolio only if every member satisfies the same linear
    constraints, because an average of points in a convex set stays in that
    set. Budget, box and linear group constraints all survive. **Cardinality
    and any other integer constraint do not**, so a consensus of ten
    twenty-name portfolios is typically a two-hundred-name portfolio. Report
    the consensus, but do not trade it without re-solving.
"""
consensus_weights(pop::PortfolioPopulation) = pop.W * pop.probs

"""
    weight_dispersion(pop::PortfolioPopulation; probs::AbstractVector = [0.05, 0.5, 0.95])
        -> NamedTuple

Return per-asset summary statistics of the weight across members.

# Arguments

  - `pop`: The population.
  - `probs`: Quantile levels to report, each in `(0, 1)`.

# Returns

A `NamedTuple` with these entries, each of length `N` except `quantiles`:

  - `mean`: The consensus weight of each asset.
  - `sd`: The standard deviation of each asset's weight across members. This is
    the number to put in front of a committee. An asset whose weight is
    `0.04 +/- 0.06` across defensible models is not a four per cent position,
    it is an unresolved question.
  - `range`: `maximum - minimum` per asset.
  - `quantiles`: A matrix `N x length(probs)`.
"""
function weight_dispersion(pop::PortfolioPopulation;
                           probs::AbstractVector{<:Real} = [0.05, 0.5, 0.95])
    W = pop.W
    N, M = size(W)
    wbar = consensus_weights(pop)
    sd = [sqrt(sum(pop.probs[m] * (W[i, m] - wbar[i])^2 for m in 1:M)) for i in 1:N]
    rng = [maximum(view(W, i, :)) - minimum(view(W, i, :)) for i in 1:N]
    qs = Matrix{eltype(W)}(undef, N, length(probs))
    for i in 1:N, (k, p) in enumerate(probs)
        qs[i, k] = quantile(view(W, i, :), p)
    end
    return (; mean = wbar, sd = sd, range = rng, quantiles = qs)
end

"""
    active_share(w1::AbstractVector, w2::AbstractVector) -> Real

Return the active share between two portfolios.

# Arguments

  - `w1`, `w2`: Weight vectors, both of length `N`.

# Returns

  - The scalar `0.5 * sum(abs.(w1 .- w2))`.

# Mathematical definition

    AS(w1, w2) = (1/2) * || w1 - w2 ||_1

For two long-only fully invested portfolios the value lies in `[0, 1]`. Zero
means the two are identical. One means they hold nothing in common. The half
is what makes the measure read as "the fraction of the portfolio that
differs", because every unit moved out of one asset must move into another and
would otherwise be counted twice.
"""
function active_share(w1::AbstractVector{<:Real}, w2::AbstractVector{<:Real})
    if length(w1) != length(w2)
        throw(DimensionMismatch("w1 has length $(length(w1)), w2 has length $(length(w2))"))
    end
    s = zero(float(eltype(w1)))
    @inbounds for i in eachindex(w1)
        s += abs(w1[i] - w2[i])
    end
    return s / 2
end

"""
    disagreement_matrix(pop::PortfolioPopulation) -> Matrix

Return the `M x M` matrix of pairwise active shares.

# Arguments

  - `pop`: The population.

# Returns

  - `D::Matrix`, `M x M`, symmetric with a zero diagonal. `D[i, j]` is the
    active share between members `i` and `j`.

# Notes

  - `D` is a distance matrix, so the library's clustering machinery reads it
    directly. Clustering the *models* rather than the *assets* answers a
    question nothing in the library asks today: which of my twelve estimators
    are really the same estimator?
"""
function disagreement_matrix(pop::PortfolioPopulation)
    W = pop.W
    M = size(W, 2)
    D = zeros(eltype(W), M, M)
    for i in 1:M, j in (i + 1):M
        d = active_share(view(W, :, i), view(W, :, j))
        D[i, j] = d
        D[j, i] = d
    end
    return D
end

"""
    mean_disagreement(pop::PortfolioPopulation) -> Real

Return the probability-weighted mean pairwise active share.

# Returns

  - A scalar in `[0, 1]`. Read it as the fraction of the portfolio that the
    average pair of models disputes. Below about `0.10` the choice of model is
    a detail. Above about `0.40` the portfolio is a statement about the model,
    not about the market.
"""
function mean_disagreement(pop::PortfolioPopulation)
    D = disagreement_matrix(pop)
    p = pop.probs
    M = length(p)
    num = zero(eltype(D))
    den = zero(eltype(D))
    for i in 1:M, j in 1:M
        if i != j
            num += p[i] * p[j] * D[i, j]
            den += p[i] * p[j]
        end
    end
    return iszero(den) ? zero(num) : num / den
end

"""
    ambiguity_decomposition(pop::PortfolioPopulation, sig::AbstractMatrix)
        -> NamedTuple

Split the average member variance into the consensus variance and the
disagreement.

# Arguments

  - `pop`: The population.
  - `sig`: Covariance matrix, `N x N`.

# Returns

A `NamedTuple`:

  - `mean_member`: `sum_m pi_m * w_m' sig w_m`, the variance of the average
    member.
  - `consensus`: `wbar' sig wbar`, the variance of the consensus portfolio.
  - `disagreement`: `sum_m pi_m * (w_m - wbar)' sig (w_m - wbar)`.
  - `residual`: The identity check, which must be zero to machine precision.

# Mathematical definition

The identity is exact for any quadratic form and any set of probabilities:

    sum_m pi_m * w_m' sig w_m
        =  wbar' sig wbar  +  sum_m pi_m * (w_m - wbar)' sig (w_m - wbar)

It is the ambiguity decomposition of Krogh and Vedelsby (1995), written for a
covariance form instead of a squared error. The proof is one line: expand
`w_m = wbar + (w_m - wbar)` and note that the cross term carries
`sum_m pi_m (w_m - wbar) = 0`.

# Notes

  - **This is a proof, not a heuristic, that the consensus is never riskier
    than the average member.** The disagreement term is a quadratic form in a
    positive semi-definite matrix, so it cannot be negative. The gap it
    measures is the risk a caller removes by refusing to pick one model.
  - The same term is the risk a caller *accepts* by picking one. Reported
    beside the portfolio, it is the honest error bar that a single optimisation
    cannot produce.
"""
function ambiguity_decomposition(pop::PortfolioPopulation, sig::AbstractMatrix{<:Real})
    W = pop.W
    N, M = size(W)
    if size(sig) != (N, N)
        throw(DimensionMismatch("sig must be $(N) x $(N), got $(size(sig))"))
    end
    wbar = consensus_weights(pop)
    mean_member = zero(float(eltype(W)))
    disagreement = zero(float(eltype(W)))
    for m in 1:M
        wm = view(W, :, m)
        mean_member += pop.probs[m] * dot(wm, sig, wm)
        d = wm .- wbar
        disagreement += pop.probs[m] * dot(d, sig, d)
    end
    consensus = dot(wbar, sig, wbar)
    return (; mean_member = mean_member, consensus = consensus, disagreement = disagreement,
            residual = mean_member - consensus - disagreement)
end

"""
    effective_number_of_models(pop::PortfolioPopulation) -> Real

Return the effective number of distinct portfolios in the population.

# Arguments

  - `pop`: The population.

# Returns

  - A scalar in `[1, M - 1]` for `M >= 2`, and `1` when every member is
    identical.

# Mathematical definition

Centre the weight matrix, form the `M x M` Gram matrix of the centred columns,
and take the participation ratio of its eigenvalues `lambda`:

    ENM  =  ( sum_k lambda_k )^2  /  sum_k lambda_k^2

The measure equals one when every member differs from the consensus along a
single common direction, whatever the number of members.

# Notes

  - **The upper bound is `M - 1`, not `M`.** Centring on the consensus makes
    the `M` deviation vectors sum to zero, so they span at most `M - 1`
    dimensions and the Gram matrix has at most `M - 1` non-zero eigenvalues.
    A population of four members that disagree in four orthogonal directions
    of equal size scores `3`, not `4`. Read the score as the number of
    **independent directions of disagreement**, which is one less than the
    number of distinct portfolios it takes to produce them.
  - Run ten estimators and get `ENM = 2.1`, and eight of the ten added almost
    nothing. That is a **budget** result: it says where to stop adding models.
  - The measure is the same participation ratio the library already uses for
    the effective number of assets. See `set_weight_norm_2_constraints!` in
    `src/20_Optimisation/09_JuMPConstraints/13_WeightNormConstraints.jl`.
"""
function effective_number_of_models(pop::PortfolioPopulation)
    W = pop.W
    wbar = consensus_weights(pop)
    C = W .- wbar
    G = Symmetric(transpose(C) * C)
    lam = eigvals(G)
    lam = max.(lam, zero(eltype(lam)))
    s1 = sum(lam)
    s2 = sum(abs2, lam)
    return iszero(s2) ? one(s1) : s1^2 / s2
end

"""
    support_stability(pop::PortfolioPopulation; tol::Real = 1e-8) -> NamedTuple

Report how often each asset is held at all, and how much the *sets* of held
assets agree.

# Arguments

  - `pop`: The population.
  - `tol`: An asset counts as held when `abs(weight) > tol`.

# Returns

A `NamedTuple`:

  - `frequency`: Length `N`. The probability-weighted fraction of members that
    hold each asset.
  - `always`: Indices held by every member.
  - `never`: Indices held by no member.
  - `contested`: Indices held by some but not all. **This is the list that
    matters.** It is the set of decisions the data does not settle.
  - `mean_jaccard`: The mean pairwise Jaccard index of the support sets, in
    `[0, 1]`. One means every member picks the same names.

# Notes

  - Weight dispersion and support stability answer different questions. Two
    members may agree perfectly on which assets to hold and disagree entirely
    on how much. A cardinality-constrained problem needs this function, because
    there the choice of names *is* the decision.
"""
function support_stability(pop::PortfolioPopulation; tol::Real = 1e-8)
    W = pop.W
    N, M = size(W)
    held = abs.(W) .> tol
    freq = [sum(pop.probs[m] * held[i, m] for m in 1:M) for i in 1:N]
    always = findall(i -> all(view(held, i, :)), 1:N)
    never = findall(i -> !any(view(held, i, :)), 1:N)
    contested = setdiff(1:N, union(always, never))
    num = 0.0
    cnt = 0
    for a in 1:M, b in (a + 1):M
        sa = view(held, :, a)
        sb = view(held, :, b)
        inter = count(sa .& sb)
        uni = count(sa .| sb)
        num += iszero(uni) ? 1.0 : inter / uni
        cnt += 1
    end
    return (; frequency = freq, always = always, never = never, contested = contested,
            mean_jaccard = iszero(cnt) ? 1.0 : num / cnt)
end

"""
    population_report(pop::PortfolioPopulation, sig::AbstractMatrix;
                      tol::Real = 1e-8) -> NamedTuple

Bundle every statistic in this file into one Result.

# Arguments

  - `pop`: The population.
  - `sig`: Covariance matrix, `N x N`, used for the ambiguity decomposition.
  - `tol`: Support threshold, forwarded to [`support_stability`](@ref).

# Returns

A `NamedTuple` with the fields `consensus`, `dispersion`, `disagreement`,
`mean_disagreement`, `ambiguity`, `effective_models` and `support`.

# Notes

  - In the library this becomes a `Result` struct with a `pretty_show` method,
    not a `NamedTuple`. The prototype uses a `NamedTuple` so that it stays
    free of the library's macros.
"""
function population_report(pop::PortfolioPopulation, sig::AbstractMatrix{<:Real};
                           tol::Real = 1e-8)
    return (; consensus = consensus_weights(pop), dispersion = weight_dispersion(pop),
            disagreement = disagreement_matrix(pop),
            mean_disagreement = mean_disagreement(pop),
            ambiguity = ambiguity_decomposition(pop, sig),
            effective_models = effective_number_of_models(pop),
            support = support_stability(pop; tol = tol))
end

end # module ModelPopulation
