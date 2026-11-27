#=
Parity of the seriation slot of HRP and Schur (#2174, map #2162), with the oracle at v1.14.4,
whose seriation package is the one of v1.13.0. The generator of the stored cases is in the
parity folder `graph2162/ser2174`, with a README.

Every stored case is `test/assets/Parity_Seriation_<Case>_<Output>.csv.gz`:

  - `TreeN<n>`, n = 7, 10, 20: five return matrices of the 3-factor model of #2164 (T = 5n),
    stacked in `X`. One column per (seed, linkage), seed-major, over single, complete, average
    and ward: the oracle's optimal leaves `Leaves`, its tree leaves `TreeLeaves`
    (`optimal_ordering = False`), its HRP weights `W` and `TreeW`, its Schur weights `SchurW`,
    and the first three columns of its linkage matrix `Linkage`.
  - `SpectralN<n>`: the same returns, one column per seed: the order, the coordinates, and the
    HRP and Schur weights with `SpectralSeriation()`.
  - `Kernel<case>`: the distance cases of the oracle's seriation tests, one fit each.
  - `Chain`: seven expanding windows, `fit` then `partial_fit`, with asset 4 absent at fold 3
    and assets 4 and 6 absent at fold 4.
  - `Schur2176<n>x<seed><linkage>`: the three cases of #2176.

The verdicts:

  - Parity: the tree, HRP and Schur on a pinned order, the tree order, the spectral kernel and
    its fold memory, and HRP and Schur on the spectral order.
  - Better: the library's leaf order is the exact optimum of its tree, and the oracle's misses it
    in some cases (#2164). Each such case is a `@test_broken`.
  - Better: the oracle's Schur gamma search misses a turning point in the three cases of #2176.
    Each is a `@test_broken`.
=#
using Test, PortfolioOptimisers, LinearAlgebra, Clustering, StableRNGs
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "Seriation at parity with the oracle (#2174)" begin
    PO = PortfolioOptimisers
    load(c, o) = parity_load("Seriation", c, o)
    LINKS = (SingleLinkage(), CompleteLinkage(), AverageLinkage(), WardLinkage())
    adj(D, o) = sum(D[o[i], o[i + 1]] for i in 1:(length(o) - 1))
    # The leaf sets of the merges of Clustering.jl (a negative entry is a leaf) and of a linkage
    # matrix of the oracle (0-based, a merge is `n + row`).
    function our_sets(M)
        S = Set{Int}[]
        for v in axes(M, 1)
            push!(S, union(map(m -> m < 0 ? Set([-m]) : S[m], (M[v, 1], M[v, 2]))...))
        end
        return Set(S)
    end
    function oracle_sets(Z, n)
        S = Set{Int}[]
        for v in axes(Z, 1)
            push!(S,
                  union(map(m -> m < n ? Set([Int(m) + 1]) : S[Int(m) - n + 1],
                            (Z[v, 1], Z[v, 2]))...))
        end
        return Set(S)
    end
    # Every leaf order that the tree permits, by the flips of its merges.
    function tree_orders(M)
        orders(m) =
            if m < 0
                [[-m]]
            else
                [o for (a, b) in ((M[m, 1], M[m, 2]),) for x in orders(a) for y in
                                                                              orders(b)
                 for o in (vcat(x, y), vcat(y, x))]
            end
        return orders(size(M, 1))
    end
    hser(l, bo = OptimalLeafOrder()) = HierarchicalSeriation(;
                                                             cle = ClustersEstimator(;
                                                                                     alg = HClustAlgorithm(;
                                                                                                           linkage = l,
                                                                                                           branchorder = bo)))

    @testset "Checks 1 to 5: the tree, the pinned order, the improvement and the tree order" begin
        for n in (7, 10, 20)
            c = "TreeN$n"
            Xa = load(c, "X")
            L = Int.(load(c, "Leaves"))
            TL = Int.(load(c, "TreeLeaves"))
            W, TW, SW = load(c, "W"), load(c, "TreeW"), load(c, "SchurW")
            Z = load(c, "Linkage")
            T = 5n
            for s in 0:4
                rd = ReturnsResult(; nx = ["a$i" for i in 1:n], X = Xa[(s * T) .+ (1:T), :])
                for (k, l) in enumerate(LINKS)
                    j = 4s + k
                    res = optimise(HierarchicalRiskParity(; ser = hser(l)), rd)
                    clr = res.ser.clr
                    D = clr.D
                    # 1. The same tree.
                    @test our_sets(clr.res.merges) == oracle_sets(Z[:, (3j - 2):(3j)], n)
                    # 2. Parity on the oracle's order.
                    pin = FixedSeriationResult(; order = L[:, j])
                    @test parity_compare(optimise(HierarchicalRiskParity(; ser = pin), rd).w,
                                         W[:, j]; rtol = 1e-12, name = "hrp $c $s $l").ok
                    @test maximum(abs,
                                  optimise(SchurComplementHierarchicalRiskParity(;
                                                                                 ser = pin),
                                           rd).w .- SW[:, j]) <= 5e-5
                    # 3. Our order costs no more than the oracle's, and is the optimum.
                    ours, theirs = adj(D, res.ser.order), adj(D, L[:, j])
                    @test ours <= theirs + 1e-12
                    if n <= 10
                        best = minimum(o -> adj(D, o), tree_orders(clr.res.merges))
                        @test ours <= best + 1e-12
                        # 4. The shortfall of the oracle (#2164).
                        if theirs <= best + 1e-12
                            @test theirs <= best + 1e-12
                        else
                            @test_broken theirs <= best + 1e-12
                        end
                    end
                    # 5. The tree order is the oracle's `optimal_ordering = False`.
                    rt = optimise(HierarchicalRiskParity(; ser = hser(l, nothing)), rd)
                    @test rt.ser.order == TL[:, j]
                    @test parity_compare(rt.w, TW[:, j]; rtol = 1e-12,
                                         name = "tree $c $s $l").ok
                end
            end
        end
    end

    @testset "The three Schur cases of #2176" begin
        for (n, s, l, lk) in
            ((10, 19, :average, AverageLinkage()), (20, 25, :complete, CompleteLinkage()),
             (20, 25, :ward, WardLinkage()))
            c = "Schur2176$(n)x$(s)$(l)"
            rd = ReturnsResult(; nx = ["a$i" for i in 1:n], X = load(c, "X"))
            # On the oracle's order the two searches agree.
            pin = FixedSeriationResult(; order = Int.(vec(load(c, "Leaves"))))
            w = optimise(SchurComplementHierarchicalRiskParity(; ser = pin), rd).w
            @test maximum(abs, w .- vec(load(c, "W"))) <= 5e-5
            # On our order, the oracle's step 0.1 misses the first turning point of the search,
            # and ours finds it. Measured 2.0e-2 to 2.5e-2.
            ser = HierarchicalSeriation(;
                                        cle = ClustersEstimator(;
                                                                alg = HClustAlgorithm(;
                                                                                      linkage = lk)))
            res = optimise(SchurComplementHierarchicalRiskParity(; ser = ser), rd)
            @test res.ser.order == Int.(vec(load(c, "OurLeaves")))
            @test_broken maximum(abs, res.w .- vec(load(c, "OracleOnOurs"))) <= 5e-5
        end
    end

    @testset "Check 6: the spectral kernel, one fit" begin
        cases = ("Random2", "Random3", "Random6", "Random7", "Random12", "Zero2", "Zero3",
                 "Zero6", "Equal2", "Equal3", "Equal6", "Equal6b", "Disconnected", "Gap",
                 "Repeated")
        for c in cases
            k = "Kernel$c"
            D = load(k, "D")
            S = PO.distance_to_similarity(MaximumDistanceSimilarity(); D = D)
            r = PO.spectral_seriation(S)
            st = vec(load(k, "Stats"))
            @test r.order == Int.(vec(load(k, "Order")))
            @test isapprox(r.coordinates, vec(load(k, "Coordinates")); atol = 1e-10)
            @test r.multiplicity == st[1]
            @test isnan(st[2]) ? isnan(r.gap) : isapprox(r.gap, st[2]; atol = 1e-10)
        end
        # A rotated basis of a repeated eigenspace gives the same coordinates: the anchor reads
        # the projector, which does not depend on the basis.
        D = fill(0.7, 6, 6)
        D[diagind(D)] .= 0
        S = PO.distance_to_similarity(MaximumDistanceSimilarity(); D = D)
        U = PO.helmert_basis(Float64, 6)
        Q = Matrix(qr(randn(StableRNG(4), 5, 5)).Q)
        @test isapprox(PO.spectral_coordinate(U * Q, nothing, 1e-12),
                       PO.spectral_seriation(S).coordinates; atol = 1e-12)
        # HRP and Schur on the spectral order.
        for n in (7, 10, 20)
            Xa = load("TreeN$n", "X")
            L = Int.(load("SpectralN$n", "Leaves"))
            C = load("SpectralN$n", "Coordinates")
            W, SW = load("SpectralN$n", "W"), load("SpectralN$n", "SchurW")
            T = 5n
            for s in 0:4
                rd = ReturnsResult(; nx = ["a$i" for i in 1:n], X = Xa[(s * T) .+ (1:T), :])
                h = optimise(HierarchicalRiskParity(; ser = SpectralSeriation()), rd)
                @test h.ser.order == L[:, s + 1]
                @test isapprox(h.ser.coordinates, C[:, s + 1]; atol = 1e-10)
                @test parity_compare(h.w, W[:, s + 1]; rtol = 1e-12, name = "spec $n $s").ok
                w = optimise(SchurComplementHierarchicalRiskParity(;
                                                                   ser = SpectralSeriation()),
                             rd).w
                @test maximum(abs, w .- SW[:, s + 1]) <= 5e-5
            end
        end
    end

    @testset "Check 7: the fold memory" begin
        # The kernel against the oracle's `partial_fit` chain, with an asset that leaves and
        # comes back. The state is over the universe and is viewed onto each fold's mask, as the
        # optimiser does.
        n = 7
        Dall, O, C = load("Chain", "D"), Int.(load("Chain", "Order")),
                     load("Chain", "Coordinates")
        state = nothing
        for f in 1:7
            D = Dall[((f - 1) * n) .+ (1:n), :]
            imsk = BitVector([!isnan(D[i, i]) for i in 1:n])
            idx = findall(imsk)
            S = PO.distance_to_similarity(MaximumDistanceSimilarity(); D = D[idx, idx])
            r = PO.spectral_seriation(S, if isnothing(state)
                                          nothing
                                      else
                                          PO.port_opt_view(state, idx)
                                      end)
            state = PO.seriation_state(r, all(imsk) ? nothing : imsk)
            o = O[:, f]
            @test state.order == o[o .> 0]
            @test isequal(isnan.(state.coordinates), isnan.(C[:, f]))
            @test isapprox(filter(!isnan, state.coordinates), filter(!isnan, C[:, f]);
                           atol = 1e-10)
        end
        # The online arm writes the memory after each fold, and the next fold reads it: each
        # fold equals the step of the memory over its training window. A batch walk-forward
        # keeps no memory: each fold equals a fresh fit.
        X = load("Chain", "X")
        rd = ReturnsResult(; nx = ["a$i" for i in 1:n], X = X)
        ser = SpectralSeriation()
        hrp(pe) = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe),
                                         ser = ser)
        on = cross_val_predict(hrp(Online(EmpiricalPrior())), rd,
                               OnlineIndexWalkForward(60, 30))
        ba = cross_val_predict(hrp(EmpiricalPrior()), rd,
                               IndexWalkForward(60, 30; expand_train = true))
        mem = ser
        for (k, p) in enumerate(on.pred)
            Xk = X[1:(60 + 30 * (k - 1)), :]
            @test p.res.ser.order == seriate(mem, Xk).order
            @test p.res.ser.coordinates ≈ seriate(mem, Xk).coordinates
            @test ba.pred[k].res.ser.order == seriate(ser, Xk).order
            mem = partial_fit!(mem, Xk)
        end
        @test_throws ArgumentError PO.merge_states(mem.cache, mem.cache)
        @test copy(mem.cache).coordinates == mem.cache.coordinates
    end

    @testset "Check 8: the other kernels" begin
        # No oracle. The order is the order of the coordinate of the leading eigenvector of the
        # Laplacian of the scaled dissimilarity, which the test forms directly.
        X = load("TreeN10", "X")[1:50, :]
        for sim in (ExponentialSimilarity(), ComplementSimilarity())
            ser = SpectralSeriation(; sim = sim)
            S, D = cor_and_dist(ser.de, ser.ce, X)
            Ss = PO.distance_to_similarity(sim; S = S, D = D)
            B = (maximum(Ss) .- Ss) ./ (maximum(Ss) - minimum(Ss))
            B[diagind(B)] .= 0
            F = eigen(Symmetric(Diagonal(vec(sum(B; dims = 2))) - B))
            @test F.values[end] - F.values[end - 1] > 1e-8
            v = F.vectors[:, end]
            v *= sign(v[argmax(abs.(v))])
            r = seriate(ser, X)
            @test r.multiplicity == 1
            @test isapprox(r.coordinates, v; atol = 1e-10)
            @test r.order == sortperm(v)
        end
    end
end
