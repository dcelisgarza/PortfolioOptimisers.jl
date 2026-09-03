#=
Check `src/03_InputData/04_CrossSectionalTransforms.jl` against the rules its docstrings
state, and against the reference implementation the map of issue #643 ports. Issue #716.

THREE FACTS SHAPE THE PROBES.

1. THE STORED MATRICES COME FROM THE REFERENCE IMPLEMENTATION. Every `REFERENCE_*` matrix
   below was produced by the reference implementation's own five transformers, driven on the
   inputs written beside them, and they are the oracle of the port. `The five members
   reproduce the reference implementation` compares each one cell by cell, including the
   position of every `NaN`.

2. A SECOND, INDEPENDENT DERIVATION STANDS BESIDE THEM. `naive_winsorise`,
   `naive_tanh_shrink`, `naive_standardise`, `naive_percentile_rank` and
   `naive_gaussian_rank` re-derive the same quantities in plain Julia, one observation at a
   time, sharing no code with the implementation: the rank counting is a double loop rather
   than a sorted search, and the inverse normal comes from `Distributions` rather than from
   `erfinv`. So no testset compares the file against itself.

3. THE ESTIMATION SET IS THE ONE RULE EVERY MEMBER SHARES. It is the finite cells carrying a
   positive weight when `w` is given, and the finite cells otherwise. Three testsets pin its
   three consequences separately: an observation with an empty set returns all `NaN`, a cell
   outside the set is still transformed against it, and a group too small to estimate from
   takes the whole observation's statistics.
=#

using Statistics, Distributions

# --- the second derivation ---------------------------------------------------------------
#
# Written one observation at a time from the definitions in the docstrings. It shares no code
# with `src/03_InputData/04_CrossSectionalTransforms.jl`.

function naive_masks(x, wrow)
    fin = isfinite.(x)
    est = isnothing(wrow) ? fin : fin .& (wrow .> 0)
    return fin, est
end
function naive_centre(x, wrow, est)
    idx = findall(est)
    return if isnothing(wrow)
        sum(x[idx]) / length(idx)
    else
        sum(wrow[i] * x[i] for i in idx) / sum(wrow[i] for i in idx)
    end
end
function naive_scale(x, est, mu)
    idx = findall(est)
    return length(idx) < 2 ? 0.0 : sqrt(sum((x[i] - mu)^2 for i in idx) / (length(idx) - 1))
end
function naive_winsorise(X, w, low, high)
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        _, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        v = X[t, findall(est)]
        qlo, qhi = quantile(v, low), quantile(v, high)
        Y[t, :] .= clamp.(view(X, t, :), qlo, qhi)
    end
    return Y
end
function naive_tanh_shrink(X, w, knee)
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        _, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        v = X[t, findall(est)]
        m = median(v)
        s = 1.4826022185056018 * median(abs.(v .- m))
        h = knee * s
        Y[t, :] .= s > 1e-12 ? m .+ h .* tanh.((view(X, t, :) .- m) ./ h) : view(X, t, :)
    end
    return Y
end
function naive_standardise(X, w)
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        fin, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        mu = naive_centre(view(X, t, :), wrow, est)
        s = naive_scale(view(X, t, :), est, mu)
        for i in axes(X, 2)
            Y[t, i] = !fin[i] ? NaN : (s > 1e-12 ? (X[t, i] - mu) / s : 0.0)
        end
    end
    return Y
end
function naive_percentile_rank(X, w)
    P = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        fin, est = naive_masks(view(X, t, :), wrow)
        n = count(est)
        if iszero(n)
            continue
        end
        for i in axes(X, 2)
            if !fin[i]
                continue
            end
            nlt = count(j -> est[j] && X[t, j] < X[t, i], axes(X, 2))
            nle = count(j -> est[j] && X[t, j] <= X[t, i], axes(X, 2))
            P[t, i] = clamp((nlt + nle) / (2n), 0.5 / n, 1 - 0.5 / n)
        end
    end
    return P
end
function naive_gaussian_rank(X, w, scale)
    P = naive_percentile_rank(X, w)
    Q = [isnan(p) ? NaN : quantile(Normal(), p) for p in P]
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        fin, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        mu = naive_centre(view(Q, t, :), wrow, est)
        C = [fin[i] ? Q[t, i] - mu : NaN for i in axes(X, 2)]
        if !scale
            Y[t, :] .= C
            continue
        end
        s = naive_scale(C, est, 0.0)
        for i in axes(X, 2)
            Y[t, i] = !fin[i] ? NaN : (s > 1e-12 ? C[i] / s : 0.0)
        end
    end
    return Y
end

# Cell-by-cell agreement, with every `NaN` in the same place.
function same_matrix(A, B; atol = 1e-12)
    if !(size(A) == size(B))
        return false
    end
    if !(all(isnan.(A) .== isnan.(B)))
        return false
    end
    return all(isnan(a) || isapprox(a, b; atol = atol) for (a, b) in zip(A, B))
end

# --- the inputs --------------------------------------------------------------------------

# The four-asset panel the reference implementation documents its own transformers on. The
# `NaN`s are the blanks, and the two weight matrices are the same estimation universe read
# two ways: `DOC_MASK_W` selects, and `DOC_W` selects and weights.
const DOC_X = [1.0 NaN 3.0 4.0; 4.0 3.0 2.0 1.0; 10.0 20.0 NaN 40.0]
const DOC_MASK_W = [1.0 0.0 1.0 1.0; 1.0 0.0 1.0 1.0; 1.0 1.0 0.0 1.0]
const DOC_W = [3.0 0.0 1.0 2.0; 4.0 0.0 2.0 3.0; 2.0 3.0 0.0 5.0]
const DOC_G = [0 0 1 1; 0 0 1 1; 0 0 1 1]

# A nine-observation panel that carries every awkward observation at once: row 3 has an empty
# estimation set, row 4 is constant, row 5 holds a three-way tie, row 6 holds a single
# estimation asset, row 7 holds finite cells outside the estimation set, row 8 holds two
# assets with no group, and row 9 puts every asset in one group.
const RAND_X = [0.9167957292444923 -0.2346415855536521 NaN -1.8344670329243553 1.7512914063517457 -1.915160074432138 -2.83204471683961;
                NaN 0.8278383772321528 3.0584612838608565 -3.7948411456558597 0.9802467543908795 NaN -1.891796716385608;
                3.9521462606205207 -1.7992302110956562 0.28066323584990494 2.0201470488940534 1.4558208123822998 3.3299888687167467 2.5909454750713525;
                2.5 2.5 2.5 2.5 2.5 2.5 2.5;
                0.4833674175087648 0.4833674175087648 0.4833674175087648 0.830940810334497 6.5726769365446875 4.984498338092746 -0.9726551899575635;
                5.3468383172369585 -4.029556022202094 0.2317386451977349 -0.15643354451165647 1.9531666627800435 1.3469968901429301 0.26289521569212204;
                -0.3065184759687538 -0.7164225092092993 0.0910216470675862 0.4321792560280313 -2.10594472643264 3.637122847535857 1.8817284105300176;
                6.428499233323535 3.2375237670223345 1.5083789238399274 2.6099672161637946 3.4229991281360674 -1.291445788814194 2.3210754150873134;
                1.8056965631636879 -1.8203448982661925 5.891948972017129 0.7467212510705343 2.0126134625870966 4.904709938139171 0.6976501658294585]
const RAND_W = [0.5293353138499124 0.8254881420992161 0.193431752815708 0.5175582505024171 0.607300723965417 0.32042470656143174 0.4682345584677836;
                0.6035528976738073 0.8573179041511593 0.6076849715570043 0.6068553966090401 0.35383753035283527 0.33009970445355086 0.04293537002890879;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                0.12291250798791231 0.34824939185828174 0.8683775294494073 0.13242557607534244 0.606589419258639 0.24523033475580702 0.4302263422780156;
                0.1765644206371929 0.18539480983631107 0.0340164908541416 0.3181686270804682 0.8953686587304909 0.5562418460520835 0.8154129546588738;
                0.0 0.0 0.0 1.0 0.0 0.0 0.0;
                0.0 0.0 0.9177651090153105 0.33878329894824777 0.45698619943304253 0.19143230229673658 0.9975335267830137;
                0.5023892905062024 0.0784366938959522 0.12574614208271728 0.9674584440088674 0.9785237030989982 0.014975039536811674 0.32338743459952624;
                0.9644706890855015 0.00914155886312662 0.949220954680857 0.9973446825549453 0.06995279607591554 0.055587314029686974 0.5168602940358353]
const RAND_G = [1 0 1 2 1 2 0;
                0 0 1 2 0 2 2;
                2 1 2 2 1 0 2;
                2 2 2 0 1 0 1;
                0 1 0 1 0 0 2;
                2 2 1 0 2 0 0;
                2 1 0 2 2 2 1;
                -1 -1 2 0 1 0 1;
                0 0 0 0 0 0 0]

# --- the oracle --------------------------------------------------------------------------
#
# Produced by the reference implementation on the inputs above.

const REFERENCE_STD_VECTOR = [-0.806225774829855 -0.558156305651438 -0.062017367294604234 1.4263994477758974]
const REFERENCE_STANDARDISE = [-1.0910894511799618 NaN 0.2182178902359925 0.8728715609439697;
                               1.161895003862225 0.3872983346207417 -0.3872983346207417 -1.161895003862225;
                               -0.8728715609439694 -0.2182178902359923 NaN 1.091089451179962]
const REFERENCE_STANDARDISE_WG = [-0.5545432546903087 NaN -0.6218206328680191 1.1427251984694726;
                                  0.6225458561187999 -0.1532420568907816 0.5035012041966656 -1.165728610956177;
                                  -1.3373607497456683 0.20821245125832238 NaN 0.4100168291432739]
const REFERENCE_WINSORISE = [1.4 NaN 3.0 3.8; 3.7 3.0 2.0 1.3; 12.0 20.0 NaN 36.0]
const REFERENCE_WINSORISE_W = [1.4 NaN 3.0 3.8; 3.6 3.0 2.0 1.2; 12.0 20.0 NaN 36.0]
const REFERENCE_TANH = [1.1247186586948457 NaN 3.0 3.983484358943085;
                        3.945606186343971 2.9979044051347525 2.0020955948652475 1.0543938136560287;
                        10.165156410569153 20.0 NaN 38.75281341305154]
const REFERENCE_TANH_W = [1.1247186586948457 NaN 3.0 3.983484358943085;
                          3.875281341305154 2.983484358943085 2.0 1.0165156410569152;
                          10.165156410569153 20.0 NaN 38.75281341305154]
const REFERENCE_GAUSSIAN = [-1.0 NaN 0.0 1.0;
                            1.1803020040387493 0.32693604766393136 -0.32693604766393136 -1.1803020040387493;
                            -1.0 0.0 NaN 1.0]
const REFERENCE_GAUSSIAN_WG = [-0.6791367019533091 NaN -0.3454139090430476 1.1914120074514876;
                               0.6985779156934256 0.08635889733764598 0.36442411553058796 -1.1743866312782931;
                               -1.3330544887858795 0.13413753156684322 NaN 0.4527392765742459]
const REFERENCE_PERCENTILE = [0.16666666666666666 NaN 0.5 0.8333333333333334;
                              0.875 0.625 0.375 0.125;
                              0.16666666666666666 0.5 NaN 0.8333333333333334]
const REFERENCE_PERCENTILE_WG = [0.16666666666666666 NaN 0.25 0.75;
                                 0.8333333333333334 0.6666666666666666 0.75 0.25;
                                 0.25 0.75 NaN 0.8333333333333334]
const REFERENCE_GAUSSIAN_N2 = [-0.7071067811865475 0.7071067811865475]
const REFERENCE_PERCENTILE_N2 = [0.25 0.75]
const REFERENCE_RAND_STANDARDISE_WG = [-0.9732008167705817 0.636947972680258 NaN 0.6795691885714986 0.8482610664787932 -1.0976576809784284 -1.1229265099574306;
                                       NaN -0.8756374006965427 1.080447936381937 -0.4495311256884143 0.9139345921394546 NaN 1.0141679977191214;
                                       NaN NaN NaN NaN NaN NaN NaN;
                                       0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                       -1.0712573956327813 -0.6485309453126881 -1.0712573956327813 1.0636569223729115 0.8428459233594862 0.34362053221355515 -1.1508225008148423;
                                       0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                       -0.24327457408267078 -1.0076442814090247 -0.5015503879445279 0.08454059373357573 -1.0418139396690067 1.5068126216329571 0.6208377684732352;
                                       1.1738887597746162 -0.2805141217053441 -1.0686346412704182 -0.17219720648544495 0.1654182691247276 -1.3213157849182775 -1.2642889406776139;
                                       -0.25239203459984605 -1.60893090238727 1.2763164137624106 -0.6485654336528176 -0.17498232501554672 0.9069802835781051 -0.6669234240735282]
const REFERENCE_RAND_GAUSSIAN_WG = [-1.0696547877373648 0.7228452820190501 NaN 0.7228452820190501 0.7228452820190501 -1.0696547877373648 -1.0696547877373648;
                                    NaN -0.757003048462599 1.3634123286129871 -0.757003048462599 0.7053366541669216 NaN 0.7053366541669216;
                                    NaN NaN NaN NaN NaN NaN NaN;
                                    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                    -0.699638117711332 -0.699638117711332 -0.699638117711332 0.7280556914298366 1.2316822089792752 0.3514410777490456 -1.5365228692952078;
                                    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                    -0.4557525008713562 -0.98085346447882 -0.5754619697240673 0.09469446620253628 -1.1416197117323017 1.331008644137374 0.7648509021291396;
                                    0.887209798643473 -0.2205969096129938 -1.3874840205140937 0.09022163677638026 0.09022163677638026 -1.2694100956729926 -1.2694100956729926;
                                    -0.18782517131387136 -1.6451994006771897 1.2695490580494466 -0.5519677213087968 0.17631737868105415 0.5995670603242557 -0.9752174029519983]
const REFERENCE_RAND_PERCENTILE_G = [0.25 0.75 NaN 0.75 0.75 0.25 0.25;
                                     NaN 0.25 0.9 0.25 0.75 NaN 0.75;
                                     0.875 0.25 0.125 0.375 0.75 0.7857142857142857 0.625;
                                     0.5 0.5 0.5 0.5 0.5 0.5 0.5;
                                     0.25 0.25 0.25 0.75 0.875 0.625 0.07142857142857142;
                                     0.8333333333333334 0.16666666666666666 0.35714285714285715 0.16666666666666666 0.5 0.8333333333333334 0.5;
                                     0.375 0.25 0.5 0.625 0.125 0.875 0.75;
                                     0.9285714285714286 0.6428571428571429 0.21428571428571427 0.75 0.75 0.25 0.25;
                                     0.5 0.07142857142857142 0.9285714285714286 0.35714285714285715 0.6428571428571429 0.7857142857142857 0.21428571428571427]
const REFERENCE_RAND_TANH_W = [0.8498457315787247 -0.2394168817853104 NaN -1.829691736692697 1.5644530718248375 -1.908798591751048 -2.7793873412298233;
                               NaN 0.8278383772321528 3.021621116601644 -3.487027965653941 0.9802347671245752 NaN -1.8256649614232;
                               NaN NaN NaN NaN NaN NaN NaN;
                               2.5 2.5 2.5 2.5 2.5 2.5 2.5;
                               0.4833674175087648 0.4833674175087648 0.4833674175087648 0.825200412937653 2.028135099480493 2.0201871796463102 -0.6545444604955515;
                               5.3468383172369585 -4.029556022202094 0.2317386451977349 -0.15643354451165647 1.9531666627800435 1.3469968901429301 0.26289521569212204;
                               -0.30330298712025716 -0.7044233300567011 0.09133970051814871 0.4321792560280313 -1.9824758962839528 3.396856856400813 1.8577881769999078;
                               5.445467122458685 3.2312988916503693 1.5412344260974788 2.6099672161637946 3.409571384910116 -0.25688731337487525 2.32168842877521;
                               1.8056965631636879 -1.2822718982051673 5.1570281236189555 0.7627235664124943 2.0124919691460788 4.55194756701089 0.7159502624022387]
const REFERENCE_RAND_WINSORISE_W = [0.9167957292444923 -0.2346415855536521 NaN -1.8344670329243553 1.7095666224963832 -1.915160074432138 -2.7862004847192363;
                                    NaN 0.8278383772321528 2.9753327026820573 -3.7187193684850497 0.9802467543908795 NaN -1.891796716385608;
                                    NaN NaN NaN NaN NaN NaN NaN;
                                    2.5 2.5 2.5 2.5 2.5 2.5 2.5;
                                    0.4833674175087648 0.4833674175087648 0.4833674175087648 0.830940810334497 6.4773862206375705 4.984498338092746 -0.8852938335095838;
                                    -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647;
                                    -0.3065184759687538 -0.7164225092092993 0.0910216470675862 0.4321792560280313 -2.0180660714926306 3.566907070055623 1.8817284105300176;
                                    6.248169227012285 3.2375237670223345 1.5083789238399274 2.6099672161637946 3.4229991281360674 -1.1234563060549467 2.3210754150873134;
                                    1.8056965631636879 -1.6692651944204535 5.8327146299844514 0.7467212510705343 2.0126134625870966 4.904709938139171 0.6976501658294585]

@testset "The five members reproduce the reference implementation" begin
    # The three-observation panel, read with no weights, with a mask, and with a weight and a
    # classification.
    @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1,
                                                                         high = 0.9),
                                                DOC_X), REFERENCE_WINSORISE)
    @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1,
                                                                         high = 0.9), DOC_X;
                                                w = DOC_MASK_W), REFERENCE_WINSORISE_W)
    @test same_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), DOC_X),
                      REFERENCE_TANH)
    @test same_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), DOC_X;
                                                w = DOC_MASK_W), REFERENCE_TANH_W)
    @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), DOC_X),
                      REFERENCE_STANDARDISE)
    @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(;
                                                                           min_group_size = 2),
                                                DOC_X; w = DOC_W, groups = DOC_G),
                      REFERENCE_STANDARDISE_WG)
    @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(), DOC_X),
                      REFERENCE_GAUSSIAN)
    @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(;
                                                                           min_group_size = 2),
                                                DOC_X; w = DOC_W, groups = DOC_G),
                      REFERENCE_GAUSSIAN_WG)
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), DOC_X),
                      REFERENCE_PERCENTILE)
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 2),
                                                DOC_X; w = DOC_MASK_W, groups = DOC_G),
                      REFERENCE_PERCENTILE_WG)
    # The census's own four-value cross-section, and the two-asset cross-section that pins
    # the smallest rank a member can carry.
    @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(),
                                                [1.0 2.0 4.0 10.0]), REFERENCE_STD_VECTOR)
    @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(), [1.0 2.0]),
                      REFERENCE_GAUSSIAN_N2)
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), [1.0 2.0]),
                      REFERENCE_PERCENTILE_N2)
    # The nine-observation panel, which carries every awkward observation at once.
    @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X;
                                                w = RAND_W), REFERENCE_RAND_WINSORISE_W)
    @test same_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), RAND_X;
                                                w = RAND_W), REFERENCE_RAND_TANH_W)
    @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(;
                                                                           min_group_size = 2),
                                                RAND_X; w = RAND_W, groups = RAND_G),
                      REFERENCE_RAND_STANDARDISE_WG)
    @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(;
                                                                           min_group_size = 2),
                                                RAND_X; w = RAND_W, groups = RAND_G),
                      REFERENCE_RAND_GAUSSIAN_WG)
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 2),
                                                RAND_X; groups = RAND_G),
                      REFERENCE_RAND_PERCENTILE_G)
end

@testset "An independent derivation agrees with the port" begin
    for w in (nothing, RAND_W)
        @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X;
                                                    w = w),
                          naive_winsorise(RAND_X, w, 0.01, 0.99))
        @test same_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), RAND_X;
                                                    w = w),
                          naive_tanh_shrink(RAND_X, w, 3.0))
        @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), RAND_X;
                                                    w = w), naive_standardise(RAND_X, w))
        @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X;
                                                    w = w),
                          naive_percentile_rank(RAND_X, w))
        @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(), RAND_X;
                                                    w = w),
                          naive_gaussian_rank(RAND_X, w, true))
        @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(;
                                                                               scale = false),
                                                    RAND_X; w = w),
                          naive_gaussian_rank(RAND_X, w, false))
    end
    # A group large enough to hold the whole observation reproduces the ungrouped answer,
    # because the fallback then never fires and the re-centring is idempotent.
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 1),
                                                RAND_X; groups = fill(4, size(RAND_X))),
                      naive_percentile_rank(RAND_X, nothing))
end

@testset "The estimation set is the finite cells that carry a positive weight" begin
    # A weight of zero removes an asset from every statistic. Winsorising with the mask must
    # therefore differ from winsorising without it exactly where the excluded asset would
    # have moved a percentile.
    Y0 = cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1, high = 0.9), DOC_X)
    Y1 = cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1, high = 0.9), DOC_X;
                                   w = DOC_MASK_W)
    @test Y0[2, :] != Y1[2, :]
    @test Y0[1, [1, 3, 4]] == Y1[1, [1, 3, 4]]
    # A weight that is positive everywhere selects every finite cell, so it recovers the
    # unweighted answer for a member that reads the weights as a mask alone.
    @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X;
                                                w = ones(size(RAND_X))),
                      cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X))
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X;
                                                w = ones(size(RAND_X))),
                      cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X))
    # The standardiser reads them as a weight as well, so a non-constant weight moves the
    # centre and the unweighted answer is not recovered.
    @test !same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), RAND_X;
                                                 w = RAND_W),
                       cross_sectional_transform(CrossSectionalStandardiser(), RAND_X))
    @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), RAND_X;
                                                w = fill(2.5, size(RAND_X))),
                      cross_sectional_transform(CrossSectionalStandardiser(), RAND_X))
end

@testset "An observation with an empty estimation set returns all NaN" begin
    # Row 3 of the nine-observation panel carries a zero weight at every asset, so no
    # member has anything to transform it against, and no cell of it survives.
    for ct in (CrossSectionalWinsoriser(), CrossSectionalTanhShrinker(),
               CrossSectionalStandardiser(; min_group_size = 2),
               CrossSectionalGaussianRank(; min_group_size = 2),
               CrossSectionalPercentileRank(; min_group_size = 2))
        Y = cross_sectional_transform(ct, RAND_X; w = RAND_W, groups = RAND_G)
        @test all(isnan, Y[3, :])
        @test !all(isnan, Y[4, :])
    end
    # A whole matrix of blanks is the same rule read at every observation.
    @test all(isnan,
              cross_sectional_transform(CrossSectionalStandardiser(), fill(NaN, 2, 3)))
end

@testset "A cell outside the estimation set is still transformed against it" begin
    # Row 7 of the nine-observation panel holds two finite assets of zero weight. They carry
    # a value in the output, and that value is scored against the other five.
    Y = cross_sectional_transform(CrossSectionalStandardiser(), RAND_X; w = RAND_W)
    @test all(isfinite, Y[7, [1, 2]])
    idx = [3, 4, 5, 6, 7]
    mu = sum(RAND_W[7, i] * RAND_X[7, i] for i in idx) / sum(RAND_W[7, i] for i in idx)
    s = sqrt(sum((RAND_X[7, i] - mu)^2 for i in idx) / (length(idx) - 1))
    @test Y[7, 1] ≈ (RAND_X[7, 1] - mu) / s
    # The excluded assets are scored, and they do not move the score of the included ones.
    Y2 = cross_sectional_transform(CrossSectionalStandardiser(), RAND_X[:, idx];
                                   w = RAND_W[:, idx])
    @test Y[7, idx] ≈ Y2[7, :]
end

@testset "A small group and a missing group take the whole observation's statistics" begin
    # Row 8 labels two assets `CS_MISSING_GROUP`. Raising the threshold above every group
    # size sends every asset to the fallback, which is the ungrouped answer up to the final
    # re-centring, and the ranks are then exactly the ungrouped ranks.
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 8),
                                                RAND_X; groups = RAND_G),
                      cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X))
    # A matrix labelled entirely missing is the same fallback read at every cell.
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 1),
                                                RAND_X;
                                                groups = fill(PortfolioOptimisers.CS_MISSING_GROUP,
                                                              size(RAND_X))),
                      cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X))
    # The two assets of row 8 that carry no group are ranked against the whole observation,
    # while their neighbours in a standing group are not.
    P = cross_sectional_transform(CrossSectionalPercentileRank(; min_group_size = 2),
                                  RAND_X; groups = RAND_G)
    Pg = cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X)
    @test P[8, 1] == Pg[8, 1]
    @test P[8, 2] == Pg[8, 2]
    @test P[8, 5] != Pg[8, 5]
end

@testset "The two outlier members accept groups and ignore them" begin
    for ct in (CrossSectionalWinsoriser(), CrossSectionalTanhShrinker())
        @test same_matrix(cross_sectional_transform(ct, RAND_X; groups = RAND_G),
                          cross_sectional_transform(ct, RAND_X))
        @test same_matrix(cross_sectional_transform(ct, RAND_X; w = RAND_W,
                                                    groups = RAND_G),
                          cross_sectional_transform(ct, RAND_X; w = RAND_W))
    end
end

@testset "The three scoring members re-centre and re-scale the whole observation" begin
    # After the grouped pass every observation that carries dispersion has a weighted centre
    # of zero and a unit equal-weighted scale over its own estimation set.
    for ct in (CrossSectionalStandardiser(; min_group_size = 2),
               CrossSectionalGaussianRank(; min_group_size = 2))
        Y = cross_sectional_transform(ct, RAND_X; w = RAND_W, groups = RAND_G)
        for t in (1, 2, 5, 7, 8, 9)
            idx = [i for i in axes(RAND_X, 2) if isfinite(RAND_X[t, i]) && RAND_W[t, i] > 0]
            mu = sum(RAND_W[t, i] * Y[t, i] for i in idx) / sum(RAND_W[t, i] for i in idx)
            @test isapprox(mu, 0.0; atol = 1e-12)
            s = sqrt(sum(Y[t, i]^2 for i in idx) / (length(idx) - 1))
            @test isapprox(s, 1.0; atol = 1e-12)
        end
    end
    # `scale = false` recentres and stops, so the scale is whatever the ranks give.
    Y = cross_sectional_transform(CrossSectionalGaussianRank(; scale = false), RAND_X)
    idx = [i for i in axes(RAND_X, 2) if isfinite(RAND_X[1, i])]
    @test isapprox(sum(Y[1, i] for i in idx) / length(idx), 0.0; atol = 1e-12)
    @test !isapprox(sqrt(sum(Y[1, i]^2 for i in idx) / (length(idx) - 1)), 1.0; atol = 1e-6)
end

@testset "An observation with no dispersion scores zero, and one asset carries no scale" begin
    # Row 4 of the nine-observation panel is constant, so its scale is zero and every finite
    # cell scores zero rather than dividing by it.
    @test all(iszero, cross_sectional_transform(CrossSectionalStandardiser(), RAND_X)[4, :])
    # The tanh shrinker has no scale either, so it returns that observation unchanged.
    @test cross_sectional_transform(CrossSectionalTanhShrinker(), RAND_X)[4, :] ==
          RAND_X[4, :]
    # Row 6 carries a single estimation asset, so the divisor of the scale is zero and the
    # same convention applies.
    @test all(iszero,
              cross_sectional_transform(CrossSectionalStandardiser(), RAND_X; w = RAND_W)[6,
                                                                                          :])
    # A percentile rank still exists on both, because it needs an order and not a scale.
    @test all(==(0.5),
              cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X)[4, :])
end

@testset "The output element type follows the inputs" begin
    @test eltype(cross_sectional_transform(CrossSectionalPercentileRank(),
                                           Float32[1.0 2.0 3.0])) === Float32
    @test eltype(cross_sectional_transform(CrossSectionalStandardiser(), [1 2 3; 4 5 6])) ===
          Float64
    @test eltype(cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1f0,
                                                                    high = 0.9f0),
                                           Float32[1.0 2.0 3.0])) === Float32
    # A weight matrix widens the answer when it is wider than the data.
    @test eltype(cross_sectional_transform(CrossSectionalStandardiser(),
                                           Float32[1.0 2.0 3.0];
                                           w = [1.0, 1.0, 1.0]' .* 1.0)) === Float64
end

@testset "Constructors refuse what their validation states" begin
    @test_throws DomainError CrossSectionalWinsoriser(; low = 0.9, high = 0.1)
    @test_throws DomainError CrossSectionalWinsoriser(; low = -0.1, high = 0.9)
    @test_throws DomainError CrossSectionalWinsoriser(; low = 0.1, high = 1.1)
    @test_throws DomainError CrossSectionalTanhShrinker(; knee = 0.0)
    @test_throws DomainError CrossSectionalTanhShrinker(; knee = Inf)
    @test_throws DomainError CrossSectionalTanhShrinker(; atol = -1.0)
    @test_throws DomainError CrossSectionalTanhShrinker(; atol = NaN)
    @test_throws DomainError CrossSectionalStandardiser(; min_group_size = 0)
    @test_throws DomainError CrossSectionalStandardiser(; atol = -1.0)
    @test_throws DomainError CrossSectionalStandardiser(; atol = Inf)
    @test_throws DomainError CrossSectionalGaussianRank(; min_group_size = 0)
    @test_throws DomainError CrossSectionalGaussianRank(; atol = -1.0)
    @test_throws DomainError CrossSectionalGaussianRank(; atol = Inf)
    @test_throws DomainError CrossSectionalPercentileRank(; min_group_size = 0)
    # The keyword constructors reach the same objects the positional ones do.
    @test CrossSectionalWinsoriser().low == 0.01
    @test CrossSectionalTanhShrinker().knee == 3.0
    @test CrossSectionalStandardiser().min_group_size == 8
    @test CrossSectionalGaussianRank().scale
    @test CrossSectionalPercentileRank().min_group_size == 8
end

@testset "The verb refuses a malformed X, w or groups" begin
    ct = CrossSectionalStandardiser()
    @test_throws PortfolioOptimisers.IsEmptyError cross_sectional_transform(ct,
                                                                            Matrix{Float64}(undef,
                                                                                            0,
                                                                                            0))
    @test_throws DomainError cross_sectional_transform(ct, [1.0 Inf; 2.0 3.0])
    @test_throws DomainError cross_sectional_transform(ct, [1.0 -Inf; 2.0 3.0])
    @test_throws DimensionMismatch cross_sectional_transform(ct, [1.0 2.0];
                                                             w = [1.0 2.0 3.0])
    @test_throws DomainError cross_sectional_transform(ct, [1.0 2.0]; w = [1.0 -1.0])
    @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_transform(ct,
                                                                                [1.0 2.0];
                                                                                w = [1.0 NaN])
    @test_throws DimensionMismatch cross_sectional_transform(ct, [1.0 2.0];
                                                             groups = [1 2 3])
    @test_throws DomainError cross_sectional_transform(ct, [1.0 2.0]; groups = [0 -2])
    # `CS_MISSING_GROUP` itself is admitted: it is the label that says an asset has no group.
    @test size(cross_sectional_transform(ct, [1.0 2.0];
                                         groups = [0 PortfolioOptimisers.CS_MISSING_GROUP])) ==
          (1, 2)
end

@testset "cross_sectional_groups reads a one-hot block" begin
    B = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1, 2, 3)
    @test cross_sectional_groups(B) == [1 PortfolioOptimisers.CS_MISSING_GROUP]
    B2 = zeros(2, 3, 2)
    B2[1, 1, 1] = 1.0
    B2[1, 2, 2] = 1.0
    B2[2, 3, 1] = 1.0
    @test cross_sectional_groups(B2) == [1 2 PortfolioOptimisers.CS_MISSING_GROUP;
                                         PortfolioOptimisers.CS_MISSING_GROUP PortfolioOptimisers.CS_MISSING_GROUP 1]
    # The derived labels drive the verb, which is the route the exposure build takes.
    G = cross_sectional_groups(B2)
    @test size(cross_sectional_transform(CrossSectionalPercentileRank(; min_group_size = 1),
                                         [1.0 2.0 3.0; 4.0 5.0 6.0]; groups = G)) == (2, 3)
end

@testset "cross_sectional_groups reads an Asset Panel's categorical field" begin
    pf = [PanelField(; name = "sector", kind = CategoricalPanelField(; levels = ["a", "b"]),
                     cols = [1, 2]),
          PanelField(; name = "size", kind = NumericPanelField(), cols = [3])]
    pnl = AssetPanel(; pf = pf, amsk = trues(2, 3), emsk = trues(2, 3))
    Z = zeros(2, 3, 3)
    Z[1, 1, 1] = 1.0
    Z[1, 2, 2] = 1.0
    Z[2, 1, 2] = 1.0
    Z[2, 3, 1] = 1.0
    @test cross_sectional_groups(pnl, Z, "sector") ==
          [1 2 PortfolioOptimisers.CS_MISSING_GROUP;
           2 PortfolioOptimisers.CS_MISSING_GROUP 1]
    @test_throws ArgumentError cross_sectional_groups(pnl, Z, "size")
    @test_throws KeyError cross_sectional_groups(pnl, Z, "country")
end

@testset "The internal helpers answer on their own" begin
    # An estimation set whose weights are all zero is excluded by the estimation set's own
    # definition, so the guard is reached only by a direct call. It answers zero rather than
    # dividing by nothing.
    @test PortfolioOptimisers.cross_sectional_weighted_mean([1.0 2.0], [0.0 0.0], 1,
                                                            [1, 2]) == 0.0
    @test PortfolioOptimisers.cross_sectional_weighted_mean([1.0 3.0], [1.0 3.0], 1,
                                                            [1, 2]) == 2.5
    @test PortfolioOptimisers.cross_sectional_weighted_mean([1.0 3.0], nothing, 1,
                                                            [1, 2]) == 2.0
    @test PortfolioOptimisers.cross_sectional_equal_std([1.0 3.0], 1, [1], 1.0) == 0.0
    @test PortfolioOptimisers.cross_sectional_equal_std([1.0 3.0], 1, [1, 2], 2.0) ==
          sqrt(2.0)
    @test PortfolioOptimisers.cross_sectional_indices([true false true], 1) == [1, 3]
    @test PortfolioOptimisers.cross_sectional_stat([1.0, 2.0], 2) == 2.0
    @test PortfolioOptimisers.cross_sectional_stat(3.0, 2) == 3.0
    @test PortfolioOptimisers.cross_sectional_weight_type(nothing) === Bool
    @test PortfolioOptimisers.cross_sectional_weight_type([1.0f0 2.0f0]) === Float32
    @test PortfolioOptimisers.cross_sectional_estimation_mask([true true], nothing) ==
          [true true]
    @test PortfolioOptimisers.cross_sectional_estimation_mask([true true], [1.0 0.0]) ==
          [true false]
    @test PortfolioOptimisers.cross_sectional_row_groups([true true true], [1 1 2], 1) ==
          Dict(1 => [1, 2], 2 => [3])
    @test PortfolioOptimisers.cross_sectional_row_groups([true true true],
                                                         [1 PortfolioOptimisers.CS_MISSING_GROUP 2],
                                                         1) == Dict(1 => [1], 2 => [3])
end
