using PortfolioOptimisers, Test, LinearAlgebra, Statistics, StatsBase, StableRNGs
using CovarianceEstimation: CovarianceEstimation

const PO = PortfolioOptimisers

# The geodesic from its first form, S^(1/2) (S^(-1/2) T S^(-1/2))^a S^(1/2). The library
# computes the second form, from the target end, so this derivation shares no step with it.
function geodesic_first_form(S, T, a)
    Sh = sqrt(Symmetric(S))
    Si = inv(Sh)
    vals, vecs = eigen(Symmetric(Si * T * Si))
    R = Sh * (vecs * Diagonal(vals .^ a) * transpose(vecs)) * Sh
    return (R + transpose(R)) / 2
end
# The affine-invariant distance, from the generalised eigenvalues of the pair.
airm(A, B) = sqrt(sum(abs2, log.(eigvals(Symmetric(A), Symmetric(B)))))

# A fixed start and a fixed target. The answers below were produced by the reference
# implementation's own interpolation on these two matrices, printed to 17 digits.
const S3 = [4.0 1.2 -0.6; 1.2 2.0 0.3; -0.6 0.3 1.0]
const T3 = [2.0 0.5 0.0; 0.5 1.5 0.2; 0.0 0.2 1.0]
const REF3 = Dict((0.25, :scaled) =>
                      [3.438923625902696 0.8730560672708096 -0.4794913233793692;
                       0.8730560672708096 2.0119785426973906 0.2960423861817764;
                       -0.4794913233793692 0.2960423861817764 1.1970237477339962],
                  (0.25, :diagonal) =>
                      [3.878865540882063 0.9374303288196849 -0.47799293139738863;
                       0.9374303288196849 1.948710537428578 0.24827423268624055;
                       -0.47799293139738863 0.24827423268624055 0.9836330357018354],
                  (0.25, :custom) =>
                      [3.3394422526679675 0.9788009981501989 -0.41637694497517863;
                       0.9788009981501989 1.8513979548948831 0.2852875162156939;
                       -0.41637694497517863 0.2852875162156939 0.9819440525365716],
                  (0.75, :scaled) =>
                      [2.61467620399093 0.2854499498476929 -0.19280380315354267;
                       0.2854499498476929 2.171736554349288 0.16202243511913234;
                       -0.19280380315354267 0.16202243511913234 1.8319770835376343],
                  (0.75, :diagonal) =>
                      [3.865400271957605 0.3488455531264325 -0.18441448454452036;
                       0.3488455531264325 1.9426918439601069 0.10219895025356447;
                       -0.18441448454452036 0.10219895025356447 0.9813376299613578],
                  (0.75, :custom) =>
                      [2.3583159785076746 0.634617864619692 -0.12123085972978617;
                       0.634617864619692 1.603348184331043 0.23427292498098157;
                       -0.12123085972978617 0.23427292498098157 0.9826669317233228])
const TARGETS3 = Dict(:identity => IdentityTarget(), :scaled => ScaledIdentityTarget(),
                      :common => CommonCovarianceTarget(),
                      :constcor => ConstantCorrelationTarget(),
                      :diagonal => DiagonalTarget(), :custom => T3)
# A small returns matrix, and the reference estimator's answer on it at intensity 0.4, with
# its default sample covariance and its default positive definite repair.
const X10 = [1.2e-05 0.002992 -0.001246; -0.008906 -0.008109 -0.01308;
             0.000601 0.013643 0.001839; -0.006205 0.002417 0.005398;
             0.001054 -0.008883 -0.004839; 0.006953 -0.010661 -0.010602;
             -0.019012 -0.0205 -0.026766; -0.002351 -0.013615 -0.00386;
             0.001568 -0.001242 -0.025946; -0.005387 -0.00264 0.000352]
const REFX10 = Dict(:scaled =>
                        [6.219377158902727e-05 1.8438924377515432e-05 1.379562879829589e-05;
                         1.8438924377515432e-05 8.750185832227732e-05 3.529617755463757e-05;
                         1.379562879829589e-05 3.529617755463757e-05 0.0001054049007256008],
                    :diagonal =>
                        [5.0243224895258666e-05 1.7159579139855345e-05 1.3565025437153587e-05;
                         1.7159579139855345e-05 8.929446129947326e-05 3.76565581356144e-05;
                         1.3565025437153587e-05 3.76565581356144e-05 0.0001187745858675302])

@testset "Geodesic shrinkage covariance" begin
    @testset "construction" begin
        ce = GeodesicShrinkageCovariance()
        @test ce.ce isa Covariance
        @test ce.pdm isa Posdef
        @test ce.tgt isa ScaledIdentityTarget
        @test ce.alpha == 0.1
        @test GeodesicShrinkageCovariance(; alpha = 1, tgt = T3).tgt === T3
        @test_throws DomainError GeodesicShrinkageCovariance(; alpha = -0.1)
        @test_throws DomainError GeodesicShrinkageCovariance(; alpha = 1.1)
        @test_throws DimensionMismatch GeodesicShrinkageCovariance(; tgt = ones(2, 3))
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = [1.0 NaN; NaN 1.0])
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = [1.0 0.5; 0.0 1.0])
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = [1.0 2.0; 2.0 1.0])
    end
    @testset "the targets" begin
        @test PO.shrinkage_target(IdentityTarget(), S3) == I(3)
        @test PO.shrinkage_target(ScaledIdentityTarget(), S3) == 7 / 3 * I(3)
        @test PO.shrinkage_target(DiagonalTarget(), S3) == Diagonal(diag(S3))
        cc = PO.shrinkage_target(CommonCovarianceTarget(), S3)
        @test diag(cc) ≈ fill(7 / 3, 3)
        @test cc[1, 2] ≈ cc[2, 3] ≈ (1.2 - 0.6 + 0.3) / 3
        d = sqrt.(diag(S3))
        r = (1.2 / (d[1] * d[2]) - 0.6 / (d[1] * d[3]) + 0.3 / (d[2] * d[3])) / 3
        kc = PO.shrinkage_target(ConstantCorrelationTarget(), S3)
        @test diag(kc) == diag(S3)
        @test kc ≈ [4.0 r*d[1]*d[2] r*d[1]*d[3]; r*d[1]*d[2] 2.0 r*d[2]*d[3];
                    r*d[1]*d[3] r*d[2]*d[3] 1.0]
        @test PO.shrinkage_target(CommonCovarianceTarget(), fill(2.0, 1, 1)) ==
              fill(2.0, 1, 1)
        @test PO.shrinkage_target(ConstantCorrelationTarget(), fill(2.0, 1, 1)) ==
              fill(2.0, 1, 1)
    end
    @testset "parity with the reference interpolation" begin
        for a in (0.25, 0.75), key in (:scaled, :diagonal, :custom)
            got = PO.geodesic_point(TARGETS3[key], S3, a)
            @test isapprox(got, REF3[(a, key)]; rtol = 1e-12)
        end
        for a in (0.25, 0.75), (key, tgt) in TARGETS3
            T = PO.shrinkage_target(tgt, S3)
            @test isapprox(PO.geodesic_point(tgt, S3, a), geodesic_first_form(S3, T, a);
                           rtol = 1e-12)
        end
    end
    @testset "parity with the reference estimator" begin
        for key in (:scaled, :diagonal)
            ce = GeodesicShrinkageCovariance(; tgt = TARGETS3[key], alpha = 0.4)
            @test isapprox(cov(ce, X10), REFX10[key]; rtol = 1e-12)
            @test isapprox(cov(ce, permutedims(X10); dims = 2), REFX10[key]; rtol = 1e-12)
        end
    end
    @testset "end points" begin
        S = cov(Covariance(), X10)
        for (key, tgt) in TARGETS3
            key == :custom && continue
            @test cov(GeodesicShrinkageCovariance(; tgt = tgt, alpha = 0), X10) ≈ S
            @test cov(GeodesicShrinkageCovariance(; tgt = tgt, alpha = 1), X10) ==
                  PO.shrinkage_target(tgt, S)
        end
    end
    @testset "geometry of the geodesic" begin
        v = tr(S3) / 3
        for a in (0.2, 0.5, 0.9)
            got = PO.geodesic_point(ScaledIdentityTarget(), S3, a)
            @test isapprox(cond(got), cond(S3)^(1 - a); rtol = 1e-10)
            @test isapprox(eigvals(Symmetric(got)),
                           eigvals(Symmetric(S3)) .^ (1 - a) .* v^a; rtol = 1e-12)
            for (key, tgt) in TARGETS3
                T = PO.shrinkage_target(tgt, S3)
                got = PO.geodesic_point(tgt, S3, a)
                @test isposdef(got)
                @test isapprox(airm(S3, got), a * airm(S3, T); rtol = 1e-10)
                @test isapprox(airm(got, T), (1 - a) * airm(S3, T); rtol = 1e-10)
            end
        end
        # Towards the diagonal, the closed form is D^(1/2) R^(1 - a) D^(1/2).
        d = sqrt.(diag(S3))
        R = S3 ./ (d * transpose(d))
        vals, vecs = eigen(Symmetric(R))
        Ra = vecs * Diagonal(vals .^ 0.3) * transpose(vecs)
        @test isapprox(PO.geodesic_point(DiagonalTarget(), S3, 0.7),
                       d .* Ra .* transpose(d); rtol = 1e-12)
    end
    @testset "a singular start is answered, an indefinite one is refused" begin
        rng = StableRNG(123456789)
        Xs = randn(rng, 4, 6)
        for tgt in (ScaledIdentityTarget(), DiagonalTarget(), ConstantCorrelationTarget())
            ce = GeodesicShrinkageCovariance(; pdm = nothing, tgt = tgt, alpha = 0.5)
            got = cov(ce, Xs)
            vals = eigvals(Symmetric(got))
            # The rank of the start survives: its three zero eigenvalues stay at round-off.
            @test count(<(1e-12 * maximum(vals)), vals) == 3
            @test minimum(vals) > -1e-12 * maximum(vals)
            # It is the limit of the answers on the positive definite starts nearby.
            near = PO.geodesic_point(tgt, cov(Xs) + 1e-12 * I, 0.5)
            @test isapprox(got, near; atol = 1e-5 * maximum(vals))
            # The default repair makes the start positive definite, so the answer is too.
            @test isposdef(cov(GeodesicShrinkageCovariance(; tgt = tgt, alpha = 0.5), Xs))
        end
        bad = copy(S3)
        bad[1, 2] = bad[2, 1] = 4.0
        @test_throws DomainError PO.geodesic_point(ScaledIdentityTarget(), bad, 0.5)
        @test_throws DomainError PO.geodesic_point(DiagonalTarget(), [1.0 0.0; 0.0 0.0],
                                                   0.5)
        @test_throws DimensionMismatch PO.geodesic_point(T3, [1.0 0.0; 0.0 1.0], 0.5)
    end
    @testset "element types and the correlation" begin
        S32 = Float32.(S3)
        @test eltype(PO.geodesic_point(ScaledIdentityTarget(), S32, 0.5f0)) == Float32
        @test isapprox(PO.geodesic_point(ScaledIdentityTarget(), S32, 0.5f0),
                       PO.geodesic_point(ScaledIdentityTarget(), S3, 0.5); rtol = 1e-5)
        @test eltype(cov(GeodesicShrinkageCovariance(; alpha = 0.5f0), Float32.(X10))) ==
              Float32
        ce = GeodesicShrinkageCovariance(; alpha = 0.3)
        sigma = cov(ce, X10)
        @test cor(ce, X10) ≈ StatsBase.cov2cor(sigma, sqrt.(diag(sigma)))
        @test all(isone, diag(cor(ce, X10)))
    end
    @testset "views, weights and gaps" begin
        rng = StableRNG(24680)
        X = randn(rng, 100, 4)
        tgt = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]) + fill(0.1, 4, 4))
        ce = GeodesicShrinkageCovariance(; tgt = tgt, alpha = 0.3)
        cev = PO.port_opt_view(ce, [1, 3])
        @test cev.tgt == tgt[[1, 3], [1, 3]]
        @test size(cov(cev, X[:, [1, 3]])) == (2, 2)
        w = StatsBase.pweights(fill(1.0, 100))
        cew = PO.factory(ce, w)
        @test cew.ce.w === w
        @test cov(cew, X) ≈ cov(ce, X)
        # A prior over a gapped sample: an asset with no observation keeps its `NaN` row
        # and column, and the rest is shrunk to a positive definite block.
        Xn = hcat(X[:, 1:2], fill(NaN, 100), X[:, 3:4])
        Xn[1:20, 5] .= NaN
        cvg = GeodesicShrinkageCovariance(; ce = Covariance(; cvg = CoveragePolicy()),
                                          alpha = 0.3)
        @test PO.gap_fill_value(cvg) == PO.gap_fill_value(cvg.ce)
        pr = prior(EmpiricalPrior(; ce = cvg), Xn)
        @test isnan.(diag(pr.sigma)) == [false, false, true, false, false]
        @test isposdef(pr.sigma[[1, 2, 4, 5], [1, 2, 4, 5]])
        rho = cor(cvg, Xn, nothing)
        @test isnan.(diag(rho)) == [false, false, true, false, false]
        @test diag(rho)[[1, 2, 4, 5]] ≈ ones(4)
        # A frame of `NaN` around the finite block: the block is shrunk, the frame is kept.
        sigma = cov(Covariance(), X)
        framed = fill(NaN, 5, 5)
        framed[[1, 2, 4, 5], [1, 2, 4, 5]] = sigma
        PO.geodesic_shrinkage!(ce, framed)
        @test framed[[1, 2, 4, 5], [1, 2, 4, 5]] ≈ PO.geodesic_point(tgt, sigma, 0.3)
        @test all(isnan, framed[3, :]) && all(isnan, framed[:, 3])
        @test all(isnan, PO.geodesic_shrinkage!(ce, fill(NaN, 2, 2)))
    end
    @testset "an incremental fit shrinks the folded matrix" begin
        rng = StableRNG(13579)
        X = randn(rng, 60, 4)
        ce = GeodesicShrinkageCovariance(; tgt = ConstantCorrelationTarget(), alpha = 0.3)
        @test PO.supports_partial_fit(ce)
        @test !PO.supports_partial_fit(GeodesicShrinkageCovariance(;
                                                                   ce = StatsBase.SimpleCovariance()))
        @test isapprox(cov(partial_fit!(ce, X)), cov(ce, X); rtol = 1e-12)
        @test isapprox(cor(partial_fit!(ce, X)), cor(ce, X); rtol = 1e-12)
        folded = foldl(partial_fit!, eachrow(X); init = ce)
        @test isapprox(cov(folded), cov(ce, X); rtol = 1e-10)
    end
    # Each property below is stated in a paper the docstrings cite, and each is checked against
    # the implementation. `phi(T, S, a)` is the geodesic from `S` to `T` at `a`, computed from
    # the target end, and `geodesic_first_form` is the geodesic from the start end.
    phi(T, S, a) = PO.geodesic_point(T, S, a)
    @testset "parity with Musolas, Smith and Marzouk (2021), section 2" begin
        for a in (0.1, 0.35, 0.8)
            got = phi(T3, S3, a)
            # Equation 2.3: A1^(1/2) U Λ^t Uᵀ A1^(1/2), with U Λ Uᵀ the eigendecomposition of
            # the whitened A1^(-1/2) A2 A1^(-1/2).
            @test isapprox(got, geodesic_first_form(S3, T3, a); rtol = 1e-12)
            # The two further forms after equation 2.3: A1 (A1⁻¹ A2)^t = A2 (A2⁻¹ A1)^(1 - t).
            @test isapprox(got, S3 * real(exp(a * log(S3 \ T3))); rtol = 1e-10)
            @test isapprox(got, T3 * real(exp((1 - a) * log(T3 \ S3))); rtol = 1e-10)
            # Equations 2.4 and 2.5: d(A1, φ(t)) = |t| d(A1, A2), with d from the generalised
            # eigenvalues of the pencil.
            @test isapprox(airm(S3, got), a * airm(S3, T3); rtol = 1e-10)
            # Equation 2.1: the distance is invariant under inversion.
            @test isapprox(airm(inv(S3), inv(got)), airm(S3, got); rtol = 1e-10)
            # Equation 2.2: the distance is invariant under congruence, and so is the geodesic.
            Z = [1.0 0.3 -0.2; 0.0 2.0 0.5; 0.4 0.0 1.5]
            @test isapprox(airm(Z * S3 * Z', Z * got * Z'), airm(S3, got); rtol = 1e-10)
            @test isapprox(phi(Z * T3 * Z', Z * S3 * Z', a), Z * got * Z'; rtol = 1e-10)
            # Section 2.2, property 1: φ⁻¹_{A1→A2}(t) = φ_{A1⁻¹→A2⁻¹}(t).
            @test isapprox(inv(got), phi(inv(T3), inv(S3), a); rtol = 1e-10)
            # Section 2.2, property 2: φ_{A1→A2}(t) = φ_{A2→A1}(1 - t).
            @test isapprox(got, phi(S3, T3, 1 - a); rtol = 1e-10)
            # Remark 2.2: φ_{A1→c A1}(t) = c^t A1.
            @test isapprox(phi(2.5 * S3, S3, a), 2.5^a * S3; rtol = 1e-12)
        end
        # The two anchors are the two ends of the curve.
        @test phi(T3, S3, 1) == T3
        @test PO.geodesic_shrinkage!(GeodesicShrinkageCovariance(; pdm = nothing, tgt = T3,
                                                                 alpha = 0), copy(S3)) == S3
    end
    @testset "parity with Bhatia (2007), the geometric mean of chapters 4 and 6" begin
        for a in (0.2, 0.5, 0.7)
            got = phi(T3, S3, a)
            # The weighted geometric mean has det(A #_t B) = det(A)^(1 - t) det(B)^t.
            @test isapprox(det(got), det(S3)^(1 - a) * det(T3)^a; rtol = 1e-12)
            # A pair that commutes: A #_t B = A^(1 - t) B^t. Here B = A², so the answer is A^(1 + t).
            vals, vecs = eigen(Symmetric(S3))
            @test isapprox(phi(S3^2, S3, a), vecs * Diagonal(vals .^ (1 + a)) * vecs';
                           rtol = 1e-10)
        end
        # The midpoint A # B is the unique positive definite solution of X A⁻¹ X = B.
        mid = phi(T3, S3, 0.5)
        @test isposdef(mid)
        @test isapprox(mid * (S3 \ mid), T3; rtol = 1e-12)
    end
    @testset "parity with Schäfer and Strimmer (2005), table 2" begin
        # CovarianceEstimation.jl builds the targets of the same table. A linear shrinkage at
        # intensity one returns its target, so it gives the target built from the same
        # corrected sample covariance.
        CE = CovarianceEstimation
        rng = StableRNG(97531)
        X = randn(rng, 80, 5) * [1.0 0.4 0.2 0.0 0.1; 0 1 0.3 0.2 0; 0 0 1 0.5 0.1;
                                 0 0 0 1 0.3; 0 0 0 0 1]
        S = cov(X)
        for (tgt, cet) in ((IdentityTarget(), CE.DiagonalUnitVariance()),
                           (ScaledIdentityTarget(), CE.DiagonalCommonVariance()),
                           (CommonCovarianceTarget(), CE.CommonCovariance()),
                           (DiagonalTarget(), CE.DiagonalUnequalVariance()),
                           (ConstantCorrelationTarget(), CE.ConstantCorrelation()))
            ref = cov(CE.LinearShrinkage(cet, 1.0; corrected = true), X)
            @test isapprox(PO.shrinkage_target(tgt, S), ref; rtol = 1e-12)
        end
        # The positive definite conditions the docstrings state. Target C has the eigenvalue
        # v - c with multiplicity N - 1 and v + (N - 1) c once; target F has the correlation
        # eigenvalues 1 - r with multiplicity N - 1 and 1 + (N - 1) r once.
        n = 5
        C = PO.shrinkage_target(CommonCovarianceTarget(), S)
        v, c = C[1, 1], C[1, 2]
        @test isapprox(eigvals(Symmetric(C)), sort([fill(v - c, n - 1); v + (n - 1) * c]);
                       rtol = 1e-12)
        F = PO.shrinkage_target(ConstantCorrelationTarget(), S)
        d = sqrt.(diag(F))
        r = F[1, 2] / (d[1] * d[2])
        @test isapprox(eigvals(Symmetric(F ./ (d * d'))),
                       sort([fill(1 - r, n - 1); 1 + (n - 1) * r]); rtol = 1e-12)
        # Target E, perfect positive correlation, has rank one, so no geodesic reaches it.
        E = sqrt.(diag(S)) * sqrt.(diag(S))'
        @test rank(E) == 1
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = E)
    end
end
