#=
The WPGMA, centroid and median linkages, and the cut in the cophenetic order (#2175, the decision
of #2170, map #2162).

Every stored case under `test/assets/Parity_<Member>_<Case>_<Output>.csv.gz` is an oracle output,
made by `graph2162/lnk2175/oracle_gen.py` and `ties_gen.py` in the parity folder, with SciPy
1.18.1 under the oracle at v1.13.0:

  - `Corr`: the returns of the recipe of #2164 (a 3-factor model, `T = 5n`), n = 7, 10, 20 with 40
    seeds (`Parity_Linkage_Corr_X`), and the oracle's distance `sqrt((1 - ρ) / 2)` at n = 50 with
    10 seeds (`Parity_Linkage_Corr50_D`). For each member: the linkage matrix `Z`, the labels of
    `fcluster(Z, k, "maxclust")` and of `cut_tree(Z)` for each `k` (`Fcluster`, `CutTree`), the
    oracle's number of clusters (`NClusters`), HRP at its defaults with its leaf order (`Hrp`),
    Schur on the same seriation (`Schur`), and HERC at `max_clusters = 2, 3, 5` (`Herc`).
  - `Ties`: SciPy on 80 distance matrices with many equal entries, integer distances in `1:3` and
    the distances of points on a 3 × 3 lattice (`Parity_Linkage_Ties_D`), and `Z` of each member.

Each output packs its cases into one matrix whose first columns are the keys `n` and `seed`,
padded with `NaN`.

The verdicts:

  - Parity: the tree of each member, its order of merges and its heights to 1.4e-15 relative
    (check 1); the ties (check 8); the partition of each cut where `fcluster` gives `k` clusters
    (check 3); HRP on the oracle's order to 2.5e-16, Schur on it, and HERC on WPGMA (check 5).
  - Better: the cut. `fcluster` maxclust gives fewer than `k` clusters on an inverted tree, in 512
    of 1980 centroid cuts and 594 of 1980 median cuts, and the library always gives `k`. The
    oracle's `cut_tree` orders the nodes by raw height, so it garbles 1180 centroid and 1241
    median cuts, and its number of clusters reads it (check 4). Each is `@test_broken`.
  - Better: `Clustering.cutree` throws a `BoundsError` on an inverted tree, and the library's
    cut does not (check 2).
  - Not compared: the number of clusters of the library, whose rule differs from the oracle's
    (#2239), and NCO (#2238). The cut that NCO reads is check 3.
=#
using PortfolioOptimisers, Test, Clustering, LinearAlgebra, StableRNGs, Statistics
include(joinpath(@__DIR__, "parity_harness.jl"))

const LNK_MEMBERS = ("Weighted" => WeightedLinkage(), "Centroid" => CentroidLinkage(),
                     "Median" => MedianLinkage())
const LNK_CASES = vcat([(n, s) for n in (7, 10, 20) for s in 0:39], [(50, s) for s in 0:9])

# The rows of the stored output `A` whose keys are `n` and `s`, without the keys.
lnk_rows(A, n, s, nkey = 2) = A[(A[:, 1] .== n) .& (A[:, 2] .== s), (nkey + 1):end]

# The distance of the case: ours from the stored returns at n <= 20, the oracle's at n = 50.
function lnk_distance(n, s, Xs, D50)
    if n == 50
        y = D50[D50[:, 1] .== s, 2:end][1, :]
        D = zeros(n, n)
        k = 0
        for j in 1:n, i in (j + 1):n
            k += 1
            D[i, j] = D[j, i] = y[k]
        end
        return D
    end
    X = lnk_rows(Xs, n, s)[:, 2:(1 + n)]
    return cor_and_dist(Distance(; alg = CanonicalDistance()),
                        PortfolioOptimisersCovariance(), X)[2]
end

# The assets below each merge: ours in the convention of `Clustering.Hclust`, the oracle's in the
# convention of SciPy, with the leaves `0:(n - 1)` and the merges after them.
function lnk_sets(merges::AbstractMatrix{<:Integer})
    S = Set{Int}[]
    for (a, b) in eachrow(merges)
        push!(S, union(a < 0 ? Set(-a) : S[a], b < 0 ? Set(-b) : S[b]))
    end
    return S
end
function lnk_sets(Z::AbstractMatrix{<:Real}, n::Integer)
    S = Set{Int}[]
    side(v) = v < n ? Set(Int(v) + 1) : S[Int(v) - n + 1]
    for r in eachrow(Z)
        push!(S, union(side(r[1]), side(r[2])))
    end
    return S
end

# The same partition, whatever the labels.
function lnk_same_partition(a, b)
    part(x) = Set(Set(findall(==(i), x)) for i in unique(x))
    return length(a) == length(b) && part(a) == part(b)
end

# The partition after the first `n - k` merges in their order, the definition of the cut.
function lnk_sequence_cut(res, k)
    n = length(res.order)
    mem = Vector{Vector{Int}}(undef, n - 1)
    lab = collect(1:n)
    for v in 1:(n - k)
        mem[v] = vcat([x < 0 ? [-x] : mem[x] for x in res.merges[v, :]]...)
        lab[mem[v]] .= minimum(mem[v])
    end
    return lab
end

# The rule of the oracle's `compute_optimal_n_clusters`, a reference that reads any cut. #2239
# states how it differs from the library's own rule.
function lnk_oracle_nclusters(D, cut)
    n = size(D, 1)
    if n == 2
        return 2
    end
    mc = min(n, max(8, round(Int, sqrt(n))))
    W = map(1:mc) do k
        lv = cut(k)
        s = 0.0
        for i in unique(lv)
            idx = findall(==(i), lv)
            if length(idx) < 2
                continue
            end
            s += mean(D[a, b] for b in idx for a in idx if a > b)
        end
        return s
    end
    gaps = W[3:end] .+ W[1:(end - 2)] .- 2 .* W[2:(end - 1)]
    return argmax(gaps) + 1
end

# The dendrogram of the library for a member on a distance, with the exact leaf order.
lnk_tree(l, D) = PortfolioOptimisers.branch_ordered_hclust(D, l, OptimalLeafOrder())

@testset "1. The tree of each member equals the oracle's, merge by merge" begin
    Xs = parity_load("Linkage", "Corr", "X")
    D50 = parity_load("Linkage", "Corr50", "D")
    for (m, l) in LNK_MEMBERS
        Zs = parity_load("$(m)Linkage", "Corr", "Z")
        same = 0
        worst = 0.0
        for (n, s) in LNK_CASES
            Z = lnk_rows(Zs, n, s)
            res = lnk_tree(l, lnk_distance(n, s, Xs, D50))
            # The branch order flips merges and moves none, so the rows keep the order of SciPy.
            same += lnk_sets(res.merges) == lnk_sets(Z, n)
            worst = max(worst, maximum(abs.(res.heights .- Z[:, 3]) ./ Z[:, 3]))
            @test res.linkage === PortfolioOptimisers.linkage_symbol(l)
        end
        @test same == length(LNK_CASES)
        # 1.32e-15 on 2026-11-23.
        @test worst <= 1e-12
    end
end

@testset "2. An inverted tree: raw heights, cophenetic heights, the cut and the branch orders" begin
    # n = 7, seed 0 of `Corr`. Merge 4 joins merge 3 at a lower height than merge 3.
    merges = [-2 -4; -6 -7; -3 2; -5 3; -1 1; 4 5]
    trees = ("centroid" => [0.46905348031694905, 0.5097267615157287, 0.5695317964040941,
                            0.5489078594022037, 0.5971776475685694, 0.647799597162296],
             "median" => [0.46905348031694905, 0.5097267615157287, 0.5695317964040941,
                          0.5499174323543102, 0.5971776475685694, 0.6511799496852014])
    for (name, h) in trees
        res = Clustering.Hclust(merges, h, [1, 2, 3, 4, 5, 6, 7], Symbol(name))
        @test !issorted(res.heights)
        c = cophenetic_heights(res)
        @test c == [h[1], h[2], h[3], h[3], h[5], h[6]]
        @test issorted(c)
        @test PortfolioOptimisers.cophenetic_order(res) == 1:6
        for k in 1:7
            idx = cut_dendrogram(res, k)
            @test maximum(idx) == k
            @test lnk_same_partition(idx, lnk_sequence_cut(res, k))
        end
        # `Clustering.cutree` replays merge 4 before merge 3, its child.
        @test_throws BoundsError Clustering.cutree(res; k = 2)
        @test_throws DomainError cut_dendrogram(res, 0)
        @test_throws DomainError cut_dendrogram(res, 8)
        root, nodes = to_tree(res)
        @test root.level == 7
        @test sort(pre_order(root)) == 1:7
        # Every branch order keeps the merges and their heights, so every cut keeps its
        # partition, and the leaf order is a permutation.
        D = [abs(i - j) + 0.0 for i in 1:7, j in 1:7]
        for bo in (OptimalLeafOrder(), BarJosephOrder(), ROrder(), nothing)
            hmer = PortfolioOptimisers.hclust_merges(merges[:, 1], merges[:, 2], h)
            PortfolioOptimisers.order_branches!(hmer, D, bo)
            rb = Clustering.Hclust(hmer, Symbol(name))
            @test sort(rb.order) == 1:7
            @test sort(rb.heights) == sort(h)
            @test all(k -> lnk_same_partition(cut_dendrogram(rb, k), cut_dendrogram(res, k)),
                      1:7)
            @test sort(to_tree(rb)[1] |> pre_order) == 1:7
        end
        # An inversion alone rejects no cut.
        nodes = PortfolioOptimisers.tall_first_nodes(res)
        @test all(k -> PortfolioOptimisers.validate_k_value(res, nodes, k), 1:7)
    end
end

@testset "3. The cut against the oracle's `fcluster`" begin
    Xs = parity_load("Linkage", "Corr", "X")
    D50 = parity_load("Linkage", "Corr50", "D")
    # The cuts at which `fcluster` gives fewer than `k` clusters, on 2026-11-23.
    fewer_expected = Dict("Weighted" => 0, "Centroid" => 512, "Median" => 594)
    for (m, l) in LNK_MEMBERS
        F = parity_load("$(m)Linkage", "Corr", "Fcluster")
        equal = 0
        fewer = 0
        cuts = 0
        for (n, s) in LNK_CASES
            res = lnk_tree(l, lnk_distance(n, s, Xs, D50))
            Fr = lnk_rows(F, n, s)
            for k in 1:n
                ours = cut_dendrogram(res, k)
                fc = Int.(Fr[k, 2:(1 + n)])
                cuts += 1
                @test maximum(ours) == k
                if maximum(fc) < k
                    fewer += 1
                else
                    equal += lnk_same_partition(ours, fc)
                end
            end
        end
        @test fewer == fewer_expected[m]
        @test equal == cuts - fewer
        # The oracle's cut gives fewer than `k` clusters on an inverted tree.
        if iszero(fewer)
            @test iszero(fewer)
        else
            @test_broken iszero(fewer)
        end
    end
end

@testset "4. The cut against the oracle's `cut_tree` and its number of clusters" begin
    Xs = parity_load("Linkage", "Corr", "X")
    D50 = parity_load("Linkage", "Corr50", "D")
    garbled_expected = Dict("Weighted" => 0, "Centroid" => 1180, "Median" => 1241)
    for (m, l) in LNK_MEMBERS
        C = parity_load("$(m)Linkage", "Corr", "CutTree")
        NC = parity_load("$(m)Linkage", "Corr", "NClusters")
        garbled = 0
        on_clean = 0
        miss_on_garbled = 0
        for (n, s) in LNK_CASES
            D = lnk_distance(n, s, Xs, D50)
            res = lnk_tree(l, D)
            Cr = lnk_rows(C, n, s)
            theirs(k) = Int.(Cr[k, 2:(1 + n)])
            bad = [!lnk_same_partition(cut_dendrogram(res, k), theirs(k)) for k in 1:n]
            garbled += count(bad)
            ko = Int(lnk_rows(NC, n, s)[1, 1])
            # The reference reproduces the oracle's count from the oracle's own cut.
            @test lnk_oracle_nclusters(D, theirs) == ko
            kr = lnk_oracle_nclusters(D, k -> cut_dendrogram(res, k))
            if any(bad)
                miss_on_garbled += kr != ko
            else
                on_clean += 1
                @test kr == ko
            end
        end
        @test garbled == garbled_expected[m]
        if iszero(garbled)
            @test iszero(miss_on_garbled)
        else
            # The oracle orders the nodes of `cut_tree` by raw height.
            @test_broken iszero(garbled)
            @test_broken iszero(miss_on_garbled)
        end
    end
end

@testset "5. HRP and Schur on the oracle's order, and HERC through the cut" begin
    Xs = parity_load("Linkage", "Corr", "X")
    for (m, l) in LNK_MEMBERS
        H = parity_load("$(m)Linkage", "Corr", "Hrp")
        SC = parity_load("$(m)Linkage", "Corr", "Schur")
        HE = parity_load("$(m)Linkage", "Corr", "Herc")
        F = parity_load("$(m)Linkage", "Corr", "Fcluster")
        hrp = 0.0
        herc = 0.0
        herc_fewer = 0
        for (n, s) in LNK_CASES
            n > 20 && continue
            X = lnk_rows(Xs, n, s)[:, 2:(1 + n)]
            rd = ReturnsResult(; nx = ["a$i" for i in 1:n], X = X)
            cle = ClustersEstimator(; alg = HClustAlgorithm(; linkage = l))
            clr = clusterise(cle, X)
            Hr = lnk_rows(H, n, s)[1, :]
            order = Int.(Hr[1:n])
            pin = Clusters(;
                           res = Clustering.Hclust(clr.res.merges, clr.res.heights, order,
                                                   clr.res.linkage), S = clr.S, D = clr.D,
                           k = clr.k)
            opt = HierarchicalOptimiser(; cle = pin)
            w = optimise(HierarchicalRiskParity(; opt = opt), rd).w
            hrp = max(hrp, maximum(abs, w - Hr[(n + 1):(2n)]))
            ws = optimise(SchurComplementHierarchicalRiskParity(; opt = opt), rd).w
            wos = lnk_rows(SC, n, s)[1, 1:n]
            # The oracle's gamma search misses the first turning point here (#2176).
            if (n, s) == (10, 19) && m != "Weighted"
                @test_broken isapprox(ws, wos; atol = 2e-5)
            else
                # The gamma searches stop within 1.1e-5 of each other.
                @test isapprox(ws, wos; atol = 2e-5)
            end
            HEr = lnk_rows(HE, n, s)
            for (i, k) in pairs((2, 3, 5))
                clk = Clusters(; res = clr.res, S = clr.S, D = clr.D, k = k)
                wk = optimise(HierarchicalEqualRiskContribution(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            cle = clk)),
                              rd).w
                if maximum(lnk_rows(F, n, s)[k, 2:(1 + n)]) < k
                    # The oracle's `fcluster` gives fewer than `k` clusters.
                    herc_fewer += 1
                else
                    herc = max(herc, maximum(abs, wk - HEr[i, 2:(1 + n)]))
                end
            end
        end
        # 2.50e-16 on 2026-11-23.
        @test hrp <= 1e-14
        @test herc <= 1e-14
        if m == "Weighted"
            @test iszero(herc_fewer)
        else
            @test_broken iszero(herc_fewer)
        end
    end
end

@testset "6. Each Clustering.jl member gives the tree that its Symbol gives Clustering.jl" begin
    rng = StableRNG(2175)
    X = randn(rng, 80, 12)
    _, D = cor_and_dist(Distance(; alg = CanonicalDistance()),
                        PortfolioOptimisersCovariance(), X)
    pairs_ = (SingleLinkage() => :single, CompleteLinkage() => :complete,
              AverageLinkage() => :average, WardLinkage() => :ward,
              WardLinkage(; presquared = true) => :ward_presquared)
    for (l, sym) in pairs_
        b = Clustering.hclust(D; linkage = sym, branchorder = :barjoseph)
        r = Clustering.hclust(D; linkage = sym, branchorder = :r)
        for bo in (OptimalLeafOrder(), BarJosephOrder(), ROrder(), nothing)
            a = PortfolioOptimisers.branch_ordered_hclust(D, l, bo)
            @test a.linkage === sym
            # A branch order flips merges, and the R order sorts them: the same tree.
            @test Set(lnk_sets(a.merges)) == Set(lnk_sets(b.merges))
            @test sort(a.heights) == sort(b.heights)
            # No inversion: the cut is the cut of Clustering.jl.
            @test all(k -> cut_dendrogram(a, k) == Clustering.cutree(a; k = k), 1:12)
        end
        @test PortfolioOptimisers.branch_ordered_hclust(D, l, BarJosephOrder()).merges ==
              b.merges
        @test PortfolioOptimisers.branch_ordered_hclust(D, l, ROrder()).order == r.order
    end
    # The R order sorts by the cophenetic height, which is the raw height on these trees, so it
    # is the order of Clustering.jl.
    for sym in (:single, :average, :ward)
        r = Clustering.hclust(D; linkage = sym, branchorder = :barjoseph)
        a = PortfolioOptimisers.hclust_merges(r.merges[:, 1], r.merges[:, 2], r.heights)
        b = PortfolioOptimisers.hclust_merges(r.merges[:, 1], r.merges[:, 2], r.heights)
        PortfolioOptimisers.order_branches!(a, D, ROrder())
        Clustering.orderbranches_r!(b)
        @test (a.mleft, a.mright, a.heights) == (b.mleft, b.mright, b.heights)
    end
    @test HClustAlgorithm().linkage === WardLinkage()
    @test_throws TypeError HClustAlgorithm(; linkage = :ward)
end

# A primitive Lance-Williams clustering, the O(N^3) reference of check 7: merge the closest pair,
# lower index first in a tie, and update by the rule of the member.
function lnk_primitive(l, D)
    n = size(D, 1)
    d = copy(D)
    live = trues(n)
    sz = ones(Int, n)
    mem = [Set([i]) for i in 1:n]
    out = Tuple{Set{Int}, Float64}[]
    for _ in 1:(n - 1)
        best = (Inf, 0, 0)
        for j in 1:n, i in 1:(j - 1)
            if live[i] && live[j] && d[i, j] < best[1]
                best = (d[i, j], i, j)
            end
        end
        h, x, y = best
        for z in 1:n
            if live[z] && z != x && z != y
                d[z, y] = d[y, z] = PortfolioOptimisers.lance_williams_update(l, d[z, x],
                                                                              d[z, y], h,
                                                                              sz[x], sz[y],
                                                                              sz[z])
            end
        end
        live[x] = false
        sz[y] += sz[x]
        mem[y] = union(mem[x], mem[y])
        push!(out, (copy(mem[y]), h))
    end
    return out
end

@testset "7. The kernel without the oracle" begin
    for seed in 1:30, kind in (:euclidean, :random), (_, l) in LNK_MEMBERS
        rng = StableRNG(seed)
        n = 6 + seed % 9
        D = if kind === :euclidean
            P = randn(rng, n, 3)
            [norm(P[i, :] - P[j, :]) for i in 1:n, j in 1:n]
        else
            A = rand(rng, n, n) .+ 0.1
            A = (A + A') / 2
            A[diagind(A)] .= 0
            A
        end
        hmer = PortfolioOptimisers.linkage_merges(l, D)
        ref = lnk_primitive(l, D)
        sets = lnk_sets(hcat(hmer.mleft, hmer.mright))
        @test Set(zip(sets, round.(hmer.heights; sigdigits = 12))) ==
              Set((s, round(h; sigdigits = 12)) for (s, h) in ref)
        if l isa WeightedLinkage
            g = PortfolioOptimisers.generic_linkage_merges(l, D)
            @test lnk_sets(hcat(g.mleft, g.mright)) == sets
            @test g.heights ≈ hmer.heights
        end
        # A merge comes after its children, and each row puts its older child first.
        for v in eachindex(hmer.mleft)
            a, b = hmer.mleft[v], hmer.mright[v]
            @test a < v && b < v
            @test PortfolioOptimisers.older_first(a, b) == (a, b)
        end
    end
    # A distance that is not Euclidean can make a negative square, which becomes zero.
    @test PortfolioOptimisers.lance_williams_update(MedianLinkage(), 0.1, 0.1, 1.0, 1, 1,
                                                    1) == 0.0
    @test PortfolioOptimisers.lance_williams_update(CentroidLinkage(), 0.1, 0.1, 1.0, 1, 1,
                                                    1) == 0.0
    # One leaf has no merge.
    @test isempty(PortfolioOptimisers.linkage_merges(CentroidLinkage(), zeros(1, 1)).heights)
    @test isempty(PortfolioOptimisers.linkage_merges(WeightedLinkage(), zeros(1, 1)).heights)
end

@testset "8. Ties: the tree of SciPy" begin
    Ds = parity_load("Linkage", "Ties", "D")
    for (m, l) in LNK_MEMBERS
        Zs = parity_load("$(m)Linkage", "Ties", "Z")
        for n in (5, 7, 10, 20), c in 0:19
            y = lnk_rows(Ds, n, c)[1, :]
            D = zeros(n, n)
            k = 0
            for j in 1:n, i in (j + 1):n
                k += 1
                D[i, j] = D[j, i] = y[k]
            end
            Z = lnk_rows(Zs, n, c)
            hmer = PortfolioOptimisers.linkage_merges(l, D)
            # The same rows in the same order, with the older child first, as SciPy writes them.
            ids(a) = a < 0 ? -a - 1 : n + a - 1
            @test [ids.(hmer.mleft) ids.(hmer.mright)] == Int.(Z[:, 1:2])
            @test isapprox(hmer.heights, Z[:, 3]; rtol = 1e-12)
        end
    end
end
