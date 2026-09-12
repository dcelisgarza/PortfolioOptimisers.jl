#=
The Span Rule and the estimation-mask rule (issue #961, map #955).

The ingestion layer seams at a Listing Span: `listing_span` reads the raw price gaps and states
the interval each asset is listed over, on the price clock; `universe_masks` projects that
interval onto the returns clock and intersects it with finiteness, giving an Asset Panel's two
universe masks. ADR 0129 fixes the pieces, ADR 0131 fixes the projection as `[first + 1, last]`,
and ADR 0132 fixes that the estimation mask is never the caller's to state.

The fixture below carries one column per row of the Span Rule's table, plus the four corner cases
the ticket names: a column that is a gap throughout, an interior gap of length one, a leading gap
that runs to the last observation but one, and a panel that is active throughout.

Two expectations of the ticket were refined after it was written, and the tests below follow the
ADRs rather than the ticket text:

  - Under padding the leading observation is active for **nobody**, because a return consumes the
    earlier price of its pair (ADR 0131). So a panel active throughout is all-`true` on the
    unpadded clock, and all-`true` below the leading row on the padded one.
  - There is no both-supplied override form. A caller's declaration replaces the active mask
    outright, and the estimation mask is always re-derived (ADR 0132), which is what makes
    `emsk ⊆ amsk` hold by construction.
=#
function span_961_prices()
    # observations × assets, one column per case:
    #  1 active throughout      2 leading gap (not yet listed)   3 trailing gap (delisted)
    #  4 interior gap, k = 1    5 gap throughout                 6 priced on the last row alone
    #  7 interior gap, k = 2    8 leading and trailing gap
    return [10.0 NaN 30.0 40.0 NaN NaN 70.0 NaN
            11.0 NaN 31.0 41.0 NaN NaN 71.0 NaN
            12.0 20.0 32.0 NaN NaN NaN NaN 82.0
            13.0 21.0 33.0 43.0 NaN NaN NaN 83.0
            14.0 22.0 NaN 44.0 NaN NaN 74.0 84.0
            15.0 23.0 NaN 45.0 NaN 60.0 75.0 NaN]
end
function span_961_returns(X::AbstractMatrix, padding::Bool)
    T = size(X, 1)
    R = X[2:T, :] ./ X[1:(T - 1), :] .- 1
    return padding ? vcat(fill(NaN, 1, size(X, 2)), R) : R
end
function span_961_amsk_padded()
    return Bool[0 0 0 0 0 0 0 0
                1 0 1 1 0 0 1 0
                1 0 1 1 0 0 1 0
                1 1 1 1 0 0 1 1
                1 1 0 1 0 0 1 1
                1 1 0 1 0 0 1 0]
end
function span_961_emsk_padded()
    return Bool[0 0 0 0 0 0 0 0
                1 0 1 1 0 0 1 0
                1 0 1 0 0 0 0 0
                1 1 1 0 0 0 0 1
                1 1 0 1 0 0 0 1
                1 1 0 1 0 0 1 0]
end
@testset "The Span Rule derives a Listing Span from a price panel" begin
    X = span_961_prices()
    span = listing_span(X)
    @test isa(span, PortfolioOptimisers.ListingSpan)
    @test size(span) == size(X)
    # One row of the Span Rule's table per column, and the corner cases beside them.
    @test span.first == [1, 3, 1, 1, 1, 6, 1, 3]
    @test span.last == [6, 6, 4, 6, 0, 6, 6, 5]
    # A leading run of gaps is an asset not yet listed; a trailing run is a delisting; an interior
    # gap leaves the asset listed, whatever its length.
    @test Matrix(span) == Bool[1 0 1 1 0 0 1 0
                               1 0 1 1 0 0 1 0
                               1 1 1 1 0 0 1 1
                               1 1 1 1 0 0 1 1
                               1 1 0 1 0 0 1 1
                               1 1 0 1 0 1 1 0]
    # A column that is a gap throughout is the empty interval, and a column priced on one
    # observation alone is a leading run that ends on the last row but one.
    @test span.first[5] > span.last[5]
    @test span.first[6] == span.last[6] == 6
    # An absent price is spelled `missing` or non-finite, and the rule unifies both.
    Xm = Matrix{Union{Missing, Float64}}(X)
    Xm[.!isfinite.(X)] .= missing
    spanm = listing_span(Xm)
    @test spanm.first == span.first
    @test spanm.last == span.last
    # A panel with no gap at all is active throughout, on the price clock.
    @test Matrix(listing_span(ones(4, 3))) == trues(4, 3)
    @test_throws PortfolioOptimisers.IsEmptyError listing_span(zeros(0, 3))
end
@testset "A Listing Span is a lazy matrix of booleans" begin
    span = listing_span(span_961_prices())
    @test isa(span, AbstractMatrix{Bool})
    @test Base.IndexStyle(typeof(span)) == IndexCartesian()
    @test span[1, 1]
    @test !span[1, 2]
    @test_throws BoundsError span[7, 1]
    @test_throws BoundsError span[1, 9]
    @test sprint(show, span) == "ListingSpan(6 × 8)"
    @test sprint(show, MIME"text/plain"(), span) == "ListingSpan(6 × 8)"
    @test_throws DimensionMismatch PortfolioOptimisers.ListingSpan([1, 2], [3], 3)
    @test_throws DomainError PortfolioOptimisers.ListingSpan([1], [2], -1)
end
@testset "universe_masks projects the span and intersects it with finiteness" begin
    X = span_961_prices()
    span = listing_span(X)
    amsk_exp = span_961_amsk_padded()
    emsk_exp = span_961_emsk_padded()
    # The padded convention: the clocks align row for row, and the leading observation carries a
    # non-finite return for everybody, so it is active for nobody.
    Rp = span_961_returns(X, true)
    amsk, emsk = universe_masks(span, Rp)
    @test size(Rp, 1) == size(X, 1)
    @test size(amsk) == size(emsk) == size(Rp)
    @test isa(amsk, PortfolioOptimisers.ListingSpan)
    @test isa(emsk, BitMatrix)
    @test Matrix(amsk) == amsk_exp
    @test Matrix(emsk) == emsk_exp
    # The compression survives the crossing: `[first + 1, last]`.
    @test amsk.first == span.first .+ 1
    @test amsk.last == span.last
    # Without padding the returns clock is one observation shorter, and the masks are the padded
    # ones with the leading row dropped.
    Ru = span_961_returns(X, false)
    amsku, emsku = universe_masks(span, Ru)
    @test size(Ru, 1) == size(X, 1) - 1
    @test size(amsku) == size(emsku) == size(Ru)
    @test Matrix(amsku) == amsk_exp[2:end, :]
    @test Matrix(emsku) == emsk_exp[2:end, :]
    @test amsku.first == span.first
    @test amsku.last == span.last .- 1
    # The estimation mask is the active mask intersected with the finiteness of the returns, and
    # it is a subset of it by construction.
    @test Matrix(emsk) == Matrix(amsk) .& isfinite.(Rp)
    @test all(Matrix(emsk) .<= Matrix(amsk))
    # An interior gap is the one thing the two masks disagree about: a Held Gap of `k + 1`
    # observations for a run of `k` gapped prices, and never at an inception or a delisting.
    @test findall(Matrix(amsk) .& .!Matrix(emsk)) ==
          CartesianIndex.([(3, 4), (4, 4), (3, 7), (4, 7), (5, 7)])
end
@testset "universe_masks refuses a statement that does not fit the returns clock" begin
    X = span_961_prices()
    span = listing_span(X)
    R = span_961_returns(X, true)
    @test_throws DimensionMismatch universe_masks(span, R[:, 1:3])
    @test_throws DimensionMismatch universe_masks(span, R[1:3, :])
    @test_throws DimensionMismatch universe_masks(span, vcat(R, R))
end
@testset "A panel active throughout gives all-true masks" begin
    X = 100.0 .+ collect(1.0:6.0) * ones(1, 3)
    span = listing_span(X)
    # Without padding both masks are all-true. Under padding the leading row is active for
    # nobody, because a return consumes the earlier price of its pair (ADR 0131).
    amsku, emsku = universe_masks(span, span_961_returns(X, false))
    @test Matrix(amsku) == trues(5, 3)
    @test Matrix(emsku) == trues(5, 3)
    amskp, emskp = universe_masks(span, span_961_returns(X, true))
    @test Matrix(amskp) == vcat(falses(1, 3), trues(5, 3))
    @test Matrix(emskp) == vcat(falses(1, 3), trues(5, 3))
end
@testset "A caller's declaration replaces the active mask outright" begin
    X = span_961_prices()
    span = listing_span(X)
    R = span_961_returns(X, true)
    # A declaration enters at the same point as the derived span, under the same
    # `AbstractMatrix{Bool}` bound. Here it says the delisted column is listed throughout, which
    # the Span Rule would deny.
    decl = Matrix(span)
    decl[:, 3] .= true
    amsk, emsk = universe_masks(decl, R)
    @test isa(amsk, BitMatrix)
    @test amsk[:, 3] == Bool[0, 1, 1, 1, 1, 1]
    # The derived answer is not merged in: it is replaced.
    @test amsk[:, 3] != span_961_amsk_padded()[:, 3]
    # The estimation mask is never the caller's to state, and is re-derived from the values, so
    # the subset invariant holds by construction rather than by refusal.
    @test Matrix(emsk) == Matrix(amsk) .& isfinite.(R)
    @test all(Matrix(emsk) .<= Matrix(amsk))
    @test emsk[:, 3] == Bool[0, 1, 1, 1, 0, 0]
    # A declaration that leaves and rejoins books no return across its absence.
    gappy = trues(6, 1)
    gappy[3:4] .= false
    amskg, _ = universe_masks(gappy, R[:, 1:1])
    @test vec(amskg) == Bool[0, 1, 0, 0, 0, 1]
    # Applied to a Listing Span's own cells, the cell-by-cell rule reproduces the compressed one,
    # under both conventions.
    @test PortfolioOptimisers.project_span(Matrix(span), 6) == span_961_amsk_padded()
    @test PortfolioOptimisers.project_span(Matrix(span), 5) ==
          span_961_amsk_padded()[2:end, :]
end
@testset "The two masks round trip through an Asset Panel" begin
    X = span_961_prices()
    span = listing_span(X)
    amsk, emsk = universe_masks(span, span_961_returns(X, true))
    ax = size(amsk)
    @test isnothing(PortfolioOptimisers.assert_panel_masks(ax, amsk, emsk))
    pnl = AssetPanel(;
                     pf = [NumericPanelField(; name = "mcap",
                                             vals = reshape(collect(1.0:48.0), 6, 8))],
                     amsk = amsk, emsk = emsk)
    @test !PortfolioOptimisers.panel_is_static(pnl)
    # The compressed active mask rides into the panel as it stands, under the public
    # `AbstractMatrix{Bool}` bound.
    @test isa(pnl.amsk, PortfolioOptimisers.ListingSpan)
    @test pnl.amsk == amsk
    @test pnl.emsk == emsk
    # The invariant is checked rather than coerced, so a pair that breaks it is refused.
    bad = trues(size(emsk)...)
    @test_throws ArgumentError PortfolioOptimisers.assert_panel_masks(ax, amsk, bad)
end
