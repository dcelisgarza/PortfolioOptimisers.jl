#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_61_interfaces_section_census.jl` loads it the
same way.
=#
module TrivialUnitCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Trivial unit census: a pure forward with one caller is inlined, a forward method of any shape is a wrapper, and a kept one is an allow-list entry (ADR 0168, #2071)" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1208, decided by ADR 0168 and built by #1212. A unit that offers nothing but
    indirection is a dead-end: a name between a caller and the one expression it forwards
    to. `diagonal_geometry` was the shape, one method, one caller, and the body
    `DiagonalProjection(; h = h)`. The rule flags exactly that shape and nothing wider:

    1. **one method**, because two methods are dispatch, and a `Nothing`/value pair or a
       two-family pair saves its caller an `isa` branch and keeps the caller's inferred
       type concrete;
    2. a body that is **one depth-one forward**: a bare argument, a literal, a field of an
       argument, or one call (or constructor) whose callee is a name and whose arguments
       are each of those, a global name, a splat of one or a keyword forward. A decision,
       an operator, an index, a nested call, a comprehension or a string template is logic
       the unit owns, and a unit that reduces its caller's complexity earns its keep;
    3. **at most one reference** in `src/` and `ext/` outside its own definition. A
       reference in value position counts as a call does. A call from `test/`, `docs/` or
       `examples/` is not a use, and a name with no reference at all is dead.

    The type-level case is an abstract type with one concrete subtype that nothing codes
    against: no dispatch on it, no field bound, no union alias, so its only reference is the
    subtype's own declaration. An abstract type with a realistic future application stays,
    and the allow-list below is where that application is named.

    Two shapes are not candidates: a constructor method named after its type, the
    library's keyword-constructor idiom, and a method that extends another module's
    function, such as `Base.show`, which is an interface by definition.

    The census is the record. The source is parsed with the parser `code_health/CodeHealth.jl`
    holds (`Meta.parseall`, which is JuliaSyntax since Julia 1.10), docstrings, comments,
    `export`/`public` lines and symbols are not references, and a name is resolved against
    the LOADED module to tell a public binding from a private one and an owned function
    from an extended one. A flagged name is kept only by a hand-written entry below, with a
    one-line reason, and an entry whose name is no longer flagged is stale and fails, so
    the list can only shrink on its own.
    =#
    CH = TrivialUnitCensusHealth.CodeHealth
    PO = PortfolioOptimisers
    root = normpath(joinpath(@__DIR__, ".."))

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)
    trees = Dict(f => CH.parse_file(f; root) for f in files)

    # ------------------------------------------------------------- definitions

    strip_lnn(args) = filter(a -> !(a isa LineNumberNode), args)

    # A leaf, `CH.forward_leaf`, is what a forward may pass on: a name, a literal, `nothing`, a
    # field chain on a name, a splat of one, a keyword forward or a tuple of leaves. One
    # depth-one forward, as rule 2 states it:
    function is_forward(e)
        CH.forward_leaf(e) && return true
        e isa Expr && e.head === :call || return false
        f = e.args[1]
        callee_ok = (f isa Symbol && !CH.is_operator_name(f)) || (f isa Expr &&
                                                                  f.head === :. &&
                                                                  f.args[end] isa QuoteNode &&
                                                                  CH.forward_leaf(f))
        return callee_ok && all(CH.forward_leaf, e.args[2:end])
    end
    function body_trivial(body)
        stmts = body isa Expr && body.head === :block ? strip_lnn(body.args) : Any[body]
        length(stmts) == 1 || return false
        s = stmts[1]
        if s isa Expr && s.head === :return
            isempty(s.args) && return true
            s = s.args[1]
        end
        return is_forward(s)
    end

    # name => [(file, line, trivial, qualified)], and the names of every declared struct.
    defs = Dict{Symbol, Vector{Tuple{String, Int, Bool, Bool}}}()
    structs = Set{Symbol}()
    function record_def!(e, file, line)
        if CH.isdocstring(e)
            e.args[2] isa LineNumberNode && (line = e.args[2].line)
            e = CH.unwrap(e)
        end
        e isa Expr || return nothing
        if e.head === :function || (e.head === :(=) && CH.is_signature(e.args[1]))
            length(e.args) >= 2 || return nothing
            name, qual = CH.signature_name(e.args[1])
            name === nothing && return nothing
            push!(get!(defs, name, Tuple{String, Int, Bool, Bool}[]),
                  (file, line, body_trivial(e.args[2]), qual))
        elseif e.head === :struct
            n = CH.defname(e.args[2])
            isempty(n) || push!(structs, Symbol(n))
        elseif e.head === :macrocall
            l = e.args[2] isa LineNumberNode ? e.args[2].line : line
            record_def!(e.args[end], file, l)
        elseif e.head === :module || e.head === :block || e.head === :toplevel
            for a in (e.head === :module ? e.args[end].args : e.args)
                a isa LineNumberNode && (line = a.line; continue)
                record_def!(a, file, line)
            end
        end
        return nothing
    end
    for f in files
        record_def!(trees[f], f, 0)
    end

    # ------------------------------------------------------------- references

    # `CH.reference_counts` states which positions are a reference. The dead abstract type
    # census, `test_86_dead_abstract_type_census.jl`, reads the same counts.
    refs = CH.reference_counts(trees[f] for f in files)

    # ------------------------------------------------------------- the function census

    function extends_other(name)
        isdefined(PO, name) || return false
        v = getfield(PO, name)
        return v isa Function && parentmodule(v) !== PO
    end
    flagged = Symbol[]
    for (name, ds) in defs
        name in structs && continue
        any(d -> d[4], ds) && continue
        extends_other(name) && continue
        length(ds) == 1 && ds[1][3] && get(refs, name, 0) <= 1 && push!(flagged, name)
    end
    sort!(flagged)

    #=
    The allow-list. A public name is public API, and a user's call is a use the census
    cannot see, so every public forward is listed by name and the list shows every one the
    library carries. A private name is listed with the reason it earns its keep.
    =#
    kept = Dict{Symbol, String}(#
                                # Public descriptor factories: one keyword call each, the
                                # user's door to a named exposure (ADR 0135).
                                :ReturnOnEquity => "public descriptor factory",
                                :SalesToEnterpriseValue => "public descriptor factory",
                                :LogMarketCap => "public descriptor factory",
                                :EbitdaToEnterpriseValue => "public descriptor factory",
                                :AssetsGrowthRate => "public descriptor factory",
                                :SalesGrowthRate => "public descriptor factory",
                                :IssuanceGrowthRate => "public descriptor factory",
                                :EarningsChangeToPrice => "public descriptor factory",
                                :CapexToAssetsChangeInIntensity => "public descriptor factory",
                                :EWMomentum => "public descriptor factory",
                                :EWShareTurnover => "public descriptor factory",
                                :EWAmihudIlliquidity => "public descriptor factory",
                                :EWMarketBeta => "public descriptor factory",
                                :RollingMomentum => "public descriptor factory",
                                :Reversal => "public descriptor factory",
                                :MaxReturn => "public descriptor factory",
                                # Public aliases of the ordered-weights risk measures.
                                :OWA_GMD => "public alias constructor",
                                :OWA_WR => "public alias constructor",
                                :OWA_RG => "public alias constructor",
                                # The value form of a public seam, `nothing` arm; the
                                # `# Interfaces` section of `matrix_processing_algorithm!`
                                # names it for a subtype to answer.
                                :matrix_processing_algorithm => "public seam, the out-of-place arm of an Open Family",
                                # Private names that are simple and convenient.
                                :cv_sequential_info => "the one message of the sequential fold loop, named beside the rule that sends a run there",
                                :_expr_to_lens => "the base case of the lens-building recursion of `expr_to_lens_chain`",
                                :attribution_no_errors => "the empty answer of an attribution, named where the Result reads three fields from it",
                                :no_head_block => "the resolver of the block of a bare head, passed as a function value to `resolve_block_entry`, which takes `f` and its arguments and not a closure, so that no variable of the entry is boxed")

    # ------------------------------------------------------------- the forward census

    #=
    Issue #2071. The function census above flags one shape alone: one method, one reference.
    A wrapper has more shapes: a function whose methods all forward to one owner, a per-form
    forward that sends each form to a different function, a forward with two references, a
    public verb that renames a private helper, a keyword constructor under a name that is not
    its type, and a tuple pack inside a function whose other methods hold logic. So this
    census reads each METHOD, whatever the count of its methods and of its references.

    A method is a forward when its body is one call, at depth one, to a function or a type
    that the package defines, with arguments that are leaves (a name, a literal, a field
    chain, a splat, a keyword forward or a tuple of them), and at least one argument is an
    argument of the method or a field of one. A call that passes none of them returns a
    constant, a trait such as `WeightsInput()`, and sends nothing on. These are exempt:

    1. a constructor method named after its type, the keyword-constructor idiom, and a
       method with keywords alone that calls a positional method of its own name, as
       `DenoiseCovariance(; ce, dn, pdm) = DenoiseCovariance(ce, dn, pdm)` does;
    2. a method of a function of another module, an interface by definition;
    3. a method with a positional argument of type `Nothing`, the `Nothing` arm of a
       `Nothing`/value pair. Its value arm is still a method of its own, so a pair whose value
       arm only forwards is flagged on that arm;
    4. a method with positional arguments that are all untyped, the `Any` fallback of a
       family. A method with keywords alone is not a fallback, so it is read;
    5. a self-call that passes a field of an argument, a recursion on a field.

    The census keys a forward by the pair `(name, callee)`, because one name can hold a kept
    method and a wrapper: `expected_risk`, `factory` and `observed_weights` do. A flagged pair
    is kept only by a name in `kept` or in `interface_functions`, or by a pair in
    `kept_forwards` or in `pending_forwards`. `kept_forwards` names reason (a) or (b) of
    § *A wrapper is for convenience or for a dispatch problem*. `interface_functions` names
    the functions whose methods are interface methods by that section, a method whose callee
    two or more methods share, or the constructor of a type. `pending_forwards` is the debt
    that #2072 removes. An entry whose pair or name is no longer flagged is stale and fails,
    so the debt can only shrink, and a new wrapper of any shape fails the build.

    `CH.forward_methods` is the one reader of this rule. `code_health/forward_survey.jl` reads
    the source with it too and gives each forward a verdict, so the census and the survey agree
    on which method only forwards (#2086). The census passes the check of the loaded module
    for exemption 2.
    =#
    owned = Set(n for (n, ds) in defs if !all(d -> d[4], ds))
    methods = reduce(vcat, (CH.method_defs(relpath(f, root), trees[f]) for f in files))
    # (name, callee) => [(file, line)] of every forward method.
    forwards = CH.forward_methods(methods; owned, structs, foreign = extends_other)
    flagged_forwards = sort!(collect(keys(forwards)))

    #=
    The forwards kept after a check, with the reason. `kept` above holds the public presets,
    and the forward census reads it too.
    =#
    kept_forwards = Dict{Tuple{Symbol, Symbol}, String}(#
                                                        (:performance_summary, :summarise_input) => "public: (a) the keyword form of an exported workflow verb; the positional form is internal and does work (#2071)",
                                                        (:performance_summary, :candidate_summary) => "public: (a) the keyword form of an exported workflow verb; the positional form is internal and does work (#2071)",
                                                        (:set_compact_show!, :set_default!) => "public: (a) a configuration verb that a user calls (#2071)",
                                                        (:with_compact_show, :with_config) => "public: (a) a configuration verb that a user calls (#2071)",
                                                        (:mip_indicators!, :declare_mip_indicators!) => "(b) its kind methods on `mip_indicators!` would be ambiguous with the `AssetMIPSpace` arm, so the generic arm calls `declare_mip_indicators!` (#2071)",
                                                        (:cs_weights_initial, :cross_sectional_cap_weights) => "public: the verb that a weight policy implements; the two capitalisation policies share `cross_sectional_cap_weights`, which the benchmark weights also read with another power (#2071)",
                                                        (:port_opt_view, :port_opt_view) => "public: (a) the one-index form `port_opt_view(pnl, i)` of the public verb selects the assets, and converts the call to the canonical form `(observations, assets, nx)`; the filled slot `:` comes before `i`, so no trailing default holds it (rule 5, #2090)",
                                                        (:panel_info, :panel_info) => "public: (a) the IO-first idiom of `show`: `panel_info(pnl)` writes to `stdout`, so the filled slot `io` comes first and no trailing default holds it (rule 5, #2090)",
                                                        (:panel_mask_lift, :RepeatedObservations) => "(b) a constructor owns no default, so the `Nothing` arm stays in front of `RepeatedObservations` (#2109, #2111)",
                                                        (:panel_field_placeholder, :rebuild_struct) => "(b) the `Nothing` arm holds a policy: `nothing` means that the fill wrote no placeholder, so the arm keeps the Panel Field as it was built, and `rebuild_struct(f, (; pmsk = nothing))` would replace its mask (#2090, #2111)",
                                                        (:obs_weights_view, :obs_weights_fields) => "(default) a walk owns its leaf answer, the identity: `obs_weights_fields` is a callee of the walk, not its owner (#2109, #2089)",
                                                        # src/05_Moments/32_CrossSectionalFactorModel (#2093).
                                                        (:ew_volatility_state, :EWVolatilityState) => "(b) a constructor owns no default, so the `Nothing` arm stays in front of `EWVolatilityState` (#2109, #2111)",
                                                        (:cross_sectional_local_returns, :descriptor_field_values) => "(b) the `Nothing` arm derives the net returns from `cc`, `X`, `lag`, `kept` and `g0`, which `descriptor_field_values` does not take, and its own `Nothing` method gives `abs.(rd.X)`, another answer (#2111, #2093)",
                                                        (:cross_sectional_shared_qr, :cross_sectional_qr) => "(b) the `Nothing` arm answers that no reader needs a factorisation, from the two masks and the solve algorithm, which `cross_sectional_qr` does not take (#2111, #2093)",
                                                        (:idio_vol_ic, :idio_vol_dependence) => "public: (a) the keyword form of an exported diagnostic verb passes its keywords as slots of the positional form (#2093)",
                                                        (:idio_vol_residual_dependence, :idio_vol_dependence) => "public: (a) the keyword form of an exported diagnostic verb passes its keywords as slots of the positional form (#2093)",
                                                        # src/05_Moments (#2092).
                                                        (:windowed_inner, :factory) => "(b) the `Nothing` arm holds a policy: the weights come from three sources, and `factory(est, nothing)` rebuilds the estimator (#2111, #2092)",
                                                        (:coverage_mask_source, :dims_oriented) => "(b) the `Nothing` arm holds a policy: its answer is the panel, and `dims_oriented(dims, nothing)` is `nothing` (#2111, #2092)",
                                                        (:factory, :rebuild_struct) => "(interface) the weights methods of `GeneralisedLinearModel` and `ElasticNet` rebuild the estimator, and 70 functions share the callee `rebuild_struct` (rule 3, rule 6, #2092). The pair, not the name, is kept, because other pairs of `factory` are still pending",
                                                        (:variance_series, :variance_series) => "(default) a walk owns its leaf: the windowed methods recurse on the inner estimator, so the 2-argument fallback converts its call to the expanding window, which no caller can pass (#2116, #2092)",
                                                        (:effective_observation_count, :observation_count) => "(interface) `observation_count` walks the composites by their fields, so its composites call it: it is a function of its own, not a worker of `effective_observation_count` (#2116, #2092)",
                                                        (:partial_fit, :partial_fit!) => "(b) a measured loss (rule 4): the overrides of `Coskewness` and `Cokurtosis` fold one row in 0.29 ms and 2.53 MiB, and 38.19 ms and 151.3 MiB, where the driver `partial_fit!(copy_states(est), x)` takes 0.34 ms and 2.75 MiB, and 38.20 ms and 157.7 MiB, at 30 assets (#2092)")
    # The functions whose forward methods are interface methods, with the check that shows
    # it. #2072 adds a name after a check of each one.
    interface_functions = Dict{Symbol, String}(#
                                               # src/02_Tools (#2089).
                                               :port_opt_view => "the leaf of the view traversal: 12 functions share its callee `nothing_scalar_array_view` (rule 3, #2089)",
                                               :obs_weights_index => "its mask method shares the callee `observed_weights` with `observed_sample` (rule 3, #2089)",
                                               :observed_long_run_form => "one method for each long-run form: 3 functions share its callee `long_run_quadratic_form` (rule 3, #2089)",
                                               :mean_test_rng => "the method of the bootstrap test of an interface verb that `# Interfaces` names: `resolve_rng` is a public verb that many functions share, and the default `nothing` drives no hook (rule 3, #2089)",
                                               # src/03_InputData (#2090).
                                               :gap_fill_seed => "rule 3: its callee `vec_to_real_measure` is a shared public verb",
                                               :apply_preprocessing => "rule 3: its callee `prices_to_returns` is a shared public verb",
                                               :fit_preprocessing => "rule 3: its callees `train_test_split`, a shared public verb, and `rebuild_struct`, which 59 functions share",
                                               :merge_column => "rule 3: its callee `merge_states` is a shared public verb",
                                               :buffer_asset_panel => "rule 3: four functions share its callee `sample_buffer_panel`",
                                               :attach_universe_masks => "rule 3: the callee of its method for a panel with no Panel Field is the constructor `AssetPanel`",
                                               # src/01_Base (#2088).
                                               :observed_weights => "a per-form method sends a `DynamicAbstractWeights` to `get_observation_weights`, a shared public verb (rule 3, #2088)",
                                               :online_state_seed => "each form seeds a buffer through the constructor of its state type, `SampleBufferState`, `ReturnsBufferState` or `PipelineBufferState` (rule 3, #2088)",
                                               :seed_window_path => "its vector method walks the elements through `element_path`, which four walks share (rule 3, #2088)",
                                               :online_entry_state => "its vector method walks the elements through `element_path`, which four walks share (rule 3, #2088)",
                                               :count_or_fraction => "its `Integer` method calls the constructor of `CountOf` (rule 3, #2088)",
                                               # src/04_MatrixProcessing (#2091).
                                               :sample_matrix_processing! => "its method on every estimator other than `MatrixProcessing` calls `matrix_processing!`, an exported verb (rule 3, #2091)",
                                               :matrix_processing_step! => "one method for each step of `mp.order`: each step calls the declared verb of its field, `posdef!`, `denoise!`, `detone!` or `matrix_processing_algorithm!` (rule 3, #2091)",
                                               # src/06_Distance (#2094).
                                               :collapse_rows => "a public verb with one method for each collapse rule: its `LastActiveRow` method calls `last_active_rows`, which `readable_window_rows` shares (rule 3, #2094)",
                                               :readable_window_rows => "one method for each form of the mask of readable cells: the matrix form calls `last_active_rows`, which two functions share, and every other form calls `collapse_rows`, a public verb (rule 3, #2094)",
                                               # src/08_Phylogeny (#2095).
                                               :reader_frames => "the case that rule 3 names: one method for each reader, and each calls `source_frames`, a shared public verb, or `field_frames`, which two functions share (rule 3, #2095)",
                                               :calc_distance_weighted_graph => "one method for each branch: the tree branch already weights its edges with the distance, so it calls `calc_weighted_adjacency_graph`, which 5 functions share (rule 3, #2095)",
                                               :separation_graph => "one method for each separation: the `PathLength` form calls `calc_distance_weighted_graph`, which two functions share (rule 3, #2095)",
                                               :centrality_graph => "one method for each polarity: the distance form calls `calc_distance_weighted_graph`, which two functions share, and the similarity form calls `calc_weighted_adjacency_graph`, which 5 functions share (rule 3, #2095)",
                                               # src/05_Moments/32_CrossSectionalFactorModel (#2093).
                                               :cross_sectional_regression => "its method on the two estimators shares the callee `cross_sectional_block_regression` with the carry fold (rule 3, #2093)",
                                               :cross_sectional_block_regression => "its callee `cross_sectional_regression` is a shared public verb (rule 3, #2093)",
                                               :factor_axis_positions => "two functions share its callee `label_positions` (rule 3, #2093)",
                                               :cs_diagnostic_factor_names => "one method for each form of the factor axis: two functions share `cs_design_labels`, and `reduce_factor_names` is a shared public verb (rule 3, #2093)",
                                               :descriptor_field_values => "its method on a Descriptor calls `descriptor`, a shared public verb (rule 3, #2093)",
                                               :lookback => "two functions share its callee `lookback_max` (rule 3, #2093)",
                                               :carry_lookback => "two functions share its callee `lookback_max` (rule 3, #2093)",
                                               :ew_state_buffer => "five functions share its callee `rolling_state_buffer` (rule 3, #2093)",
                                               :exposure_active_fill! => "eight functions share its callee `descriptor_active_fill!` (rule 3, #2093)",
                                               :target_forecast_warmup => "two functions share its callee `target_forecast_insample` (rule 3, #2093)",
                                               :target_forecast_variances => "two functions share its callee `forecast_idiosyncratic_variances` (rule 3, #2093)",
                                               :forecast_target_history => "four functions share its callee `forecast_idiosyncratic_returns` (rule 3, #2093)",
                                               :folds_forecast_rows => "the query trait of a member: its method on `TargetReturnForecast` reads `target_forecast_folds` of its fields, which two functions share (rule 3, #2093)",
                                               :exposure_coverage_weights => "its callee `cs_diagnostic_weights` is a shared public verb (rule 3, #2093)",
                                               :descriptor_carry => "the generic method, for an estimator that carries no state, calls the constructor of `CarriedEstimators`, and the other methods walk the Descriptors (rule 3, #2093)",
                                               :orthogonal_forecast_rescale => "an interface verb: its default, for a rule that keeps the orthogonal part, calls the constructor of `ForecastSplitResult`, the public type of its contract, and the method of `OrthogonalPartCalibration` scales the part (rule 3, #2093)",
                                               :unseen_member_design => "an interface verb: its methods that change no observation call the constructor of `UnseenMemberDesignResult`, the public type of its contract, and the method of `ZeroUnseenMember` builds each change (rule 3, #2093)",
                                               # src/05_Moments (#2092).
                                               :weighted_comoment_guard => "4 functions share its callee `weighted_moment_guard` (rule 3, #2092)",
                                               :library_covariance_estimator => "its callee is the constructor of `GeneralCovariance` (rule 3, #2092)",
                                               :comovement_finalise => "2 functions share its callee `comovement_ratio` (rule 3, #2092)",
                                               :adopt_pinned_choice => "70 functions share its callee `rebuild_struct` (rule 3, #2092)",
                                               :windowed_panel => "its callee `port_opt_view` is a shared public verb (rule 3, #2092)",
                                               :coskewness => "its Coverage Universe method of a panel shares the callee `coverage_panel_moment` with 7 other functions (rule 3, #2092)",
                                               :coverage_inner_corrected => "3 functions share its callee `partial_fit_corrected` (rule 3, #2092)",
                                               :supports_partial_fit => "its callee `folds` is a shared public verb, and a driver of the hook must call it (rule 3, #2089, #2092)")
    # The debt that #2072 removes. Do not add a pair: fix the wrapper.
    pending_forwards = Set{Tuple{Symbol, Symbol}}([(:apply_fitted_step,
                                                    :apply_fitted_steps),
                                                   (:apply_fitted_step,
                                                    :apply_preprocessing),
                                                   (:assert_online_owner,
                                                    :assert_online_entry),
                                                   (:assert_scenario_refit,
                                                    :assert_scenario_model),
                                                   (:assert_stateless_schedule,
                                                    :assert_stateless_prior),
                                                   (:attribution_active_mask,
                                                    :panel_active_cells),
                                                   (:attribution_expand_errors,
                                                    :expand_factor_covariance),
                                                   (:attribution_finite_rows,
                                                    :attribution_finite),
                                                   (:attribution_fold_weights,
                                                    :weight_path),
                                                   (:attribution_idiosyncratic_covariance,
                                                    :assert_attribution_field),
                                                   (:attribution_lag, :cs_regression_lag),
                                                   (:attribution_standalone,
                                                    :attribution_finite),
                                                   (:attribution_unknown_block,
                                                    :attribution_unknown_rows),
                                                   (:attribution_window_basis,
                                                    :factor_basis_slice),
                                                   (:blend_projection, :project),
                                                   (:bl_posteriors, :vanilla_posteriors),
                                                   (:buffer_column_view, :sample_buffer),
                                                   (:calibration_series_matrix,
                                                    :absolute_drawdown_arr),
                                                   (:calibration_series_matrix,
                                                    :relative_drawdown_arr),
                                                   (:calibration_series_vec,
                                                    :absolute_drawdown_vec),
                                                   (:calibration_series_vec,
                                                    :relative_drawdown_vec),
                                                   (:copy_step_states, :copy_states),
                                                   (:cross_sectional_basis_now,
                                                    :factor_basis_slice),
                                                   (:cross_sectional_move_basis,
                                                    :cross_sectional_rebase),
                                                   (:cross_sectional_neutralise!,
                                                    :neutralise_exposures!),
                                                   (:cross_sectional_prior_option,
                                                    :cross_sectional_prior_pairs),
                                                   (:cross_sectional_reduced_history,
                                                    :reduce_exposures),
                                                   (:cross_val_predict, :fit_and_predict),
                                                   (:cross_val_predict,
                                                    :pipeline_cross_val_predict),
                                                   (:cs_diagnostic_weights,
                                                    :cs_diagnostic_weight_history),
                                                   (:difference_linearised_risk,
                                                    :difference_linearised_risk),
                                                   (:difference_risk, :difference_risk),
                                                   (:difference_risk,
                                                    :resolved_expected_return),
                                                   (:dup_elim_sum_view,
                                                    :dup_elim_sum_matrices),
                                                   (:expected_risk, :expected_ratio),
                                                   (:expected_risk, :expected_return),
                                                   (:expected_risk, :expected_risk),
                                                   (:expected_risk,
                                                    :expected_risk_from_returns),
                                                   (:expected_risk_from_returns,
                                                    :expected_risk_from_returns),
                                                   (:expert_allocation_set,
                                                    :resolve_allocation_set),
                                                   (:factor_attribution,
                                                    :attribution_realised_entry),
                                                   (:factor_attribution,
                                                    :attribution_returns_entry),
                                                   (:factor_attribution,
                                                    :attribution_rolling_entry),
                                                   (:factor_attribution,
                                                    :factor_attribution),
                                                   (:factory, :BudgetedHindsightPathResult),
                                                   (:factory, :factory),
                                                   (:factory, :resolve_deferred_quantities),
                                                   (:factory, :SubsetResamplingResult),
                                                   (:fitted_step_keeps_observations,
                                                    :keeps_observations),
                                                   (:fold_context, :TimeDependentContext),
                                                   (:fold_factor_returns,
                                                    :investable_returns_view),
                                                   (:fold_optimiser, :partial_fit!),
                                                   (:forecast_coverage_policy,
                                                    :coverage_policy),
                                                   (:forecast_location,
                                                    :forecast_state_location),
                                                   (:forget_row_positions, :set_row_key),
                                                   (:gram_geometry, :GramProjection),
                                                   (:group_frames, :pipeline_frames),
                                                   (:group_returns_view,
                                                    :shared_series_fit),
                                                   (:held_timestamps, :context_timestamps),
                                                   (:held_timestamps,
                                                    :pipeline_held_timestamps),
                                                   (:herc_unitary_risks,
                                                    :unitary_expected_risks),
                                                   (:input_entries, :population_entries),
                                                   (:input_entries, :prediction_entries),
                                                   (:is_time_dependent,
                                                    :pipeline_step_is_time_dependent),
                                                   (:leverage_one_residuals,
                                                    :leverage_one_nan),
                                                   (:lift_fee_rate,
                                                    :expand_investable_weights),
                                                   (:linearised_risk, :linearised_risk),
                                                   (:maybe_inject_step, :inject_context),
                                                   (:member_regret, :series_regret),
                                                   (:merge_held_data, :partial_fit!),
                                                   (:mip_bit_space, :AssetMIPSpace),
                                                   (:mip_key, :state_key),
                                                   (:mip_wx!, :get_w),
                                                   (:mip_wx!, :shared_get),
                                                   (:no_bounds_returns_settings,
                                                    :JuMPReturnsSettings),
                                                   (:n_splits, :n_splits),
                                                   (:online_step_fold, :partial_fit!),
                                                   (:online_update!, :online_update!),
                                                   (:opinion_pooling_inner, :forward_prior),
                                                   (:opinion_pooling_kept, :forward_prior),
                                                   (:optimise, :_optimise),
                                                   (:orthogonality_weights,
                                                    :latest_orthogonality_weights),
                                                   (:override_fee_amortisation, :Fees),
                                                   (:owa_l_moment_crm, :ncrra_weights),
                                                   (:partial_fit!, :fold_pipeline),
                                                   (:pipeline_lens, :parse_lens),
                                                   (:pipeline_step_factory, :factory),
                                                   (:pipeline_step_is_time_dependent,
                                                    :is_time_dependent),
                                                   (:predictor_state_seed,
                                                    :forecaster_seed),
                                                   (:predictor_state_view, :port_opt_view),
                                                   (:prior_investable_mask,
                                                    :investable_mask),
                                                   (:prior_probability_factory, :factory),
                                                   (:project, :projection_programme),
                                                   (:rebase_linear_constraints,
                                                    :project_linear_constraint),
                                                   (:regression_factor_names,
                                                    :cs_diagnostic_factor_names),
                                                   (:report_risk_contribution,
                                                    :risk_contribution_figure),
                                                   (:reprojection, :project),
                                                   (:reset_time_dependent_estimator,
                                                    :reset_time_dependent_fields),
                                                   (:reset_time_dependent_step,
                                                    :reset_time_dependent_estimator),
                                                   (:resolve_calibration_slot,
                                                    :calibrated_part),
                                                   (:resolved_constraint_block,
                                                    :ResolvedConstraintBlock),
                                                   (:resolve_deferred_child,
                                                    :resolve_deferred_quantities),
                                                   (:rewrap_step, :PipelineStep),
                                                   (:risk_measure_nothing_scalar_array_view,
                                                    :nothing_scalar_array_view),
                                                   (:rule_state_seed, :forecaster_seed),
                                                   (:rule_state_seed, :rule_state_seed),
                                                   (:rule_state_view, :port_opt_view),
                                                   (:run_step, :run_constraint_step),
                                                   (:run_step, :run_step),
                                                   (:run_step, :run_uncertainty_step),
                                                   (:scalarised_degree,
                                                    :homogeneity_degree),
                                                   (:scale_rows, :window_rows),
                                                   (:scenario_factory, :factory),
                                                   (:sel, :solver_selector),
                                                   (:sel, :ucs_selector),
                                                   (:set_allocation_set_constraints!,
                                                    :set_allocation_set_constraints!),
                                                   (:set_retcode, :JuMPOptimisationResult),
                                                   (:set_retcode, :NestedClusteredResult),
                                                   (:set_retcode, :StackingResult),
                                                   (:set_retcode, :SubsetResamplingResult),
                                                   (:set_risk_constraints!,
                                                    :set_range_risk_constraints!),
                                                   (:set_variance_risk!,
                                                    :set_sdp_variance_risk!),
                                                   (:set_variance_risk!,
                                                    :set_variance_risk!),
                                                   (:sparse_portfolio_iterate,
                                                    :largest_forecast_split),
                                                   (:step_fallback, :partial_fit!),
                                                   (:step_online_member,
                                                    :pipeline_online_member),
                                                   (:summarise_entries, :summarise_returns),
                                                   (:supports_precomputed_returns,
                                                    :weight_independent_target),
                                                   (:systematic_processing!,
                                                    :sample_matrix_processing!),
                                                   (:tail_worst_case_risk,
                                                    :worst_case_risk),
                                                   (:tracking_benchmark, :calc_net_returns),
                                                   (:ucs_prior, :prior),
                                                   (:universe_axis_visit!,
                                                    :universe_axis_children!),
                                                   (:update_online_member,
                                                    :update_online_estimator),
                                                   (:update_online_step,
                                                    :update_online_estimator),
                                                   (:update_time_dependent_estimator,
                                                    :update_time_dependent_fields),
                                                   (:update_time_dependent_step,
                                                    :update_time_dependent_estimator),
                                                   (:variance_risk_bounds_val, :Frontier),
                                                   (:weight_mass_grid,
                                                    :pipeline_lens_val_grid),
                                                   (:weight_path, :rebuild_weight_path),
                                                   (:window_bigger_is_better,
                                                    :bigger_is_better),
                                                   (:window_values, :asset_scores),
                                                   (:with_position_id,
                                                    :MultiPeriodPredictionResult),
                                                   (:worst_case_distribution,
                                                    :ERM_worst_case),
                                                   (:worst_case_distribution,
                                                    :PRM_worst_case),
                                                   (:worst_case_distribution,
                                                    :RRM_worst_case)])

    # ------------------------------------------------------------- the type census

    # An abstract type the package declares with exactly one concrete subtype and no
    # reference outside the declarations. `subtypes` is not used: the suite runs every
    # test file in one process, so it would also see the subtypes the tests declare.
    declared = CH.declared_types(PO)
    children = Dict{Symbol, Vector{Symbol}}()
    for (n, T) in declared
        S = supertype(T)
        S === Any && continue
        S0 = Base.unwrap_unionall(S)
        parentmodule(S0) === PO &&
            haskey(declared, nameof(S0)) &&
            push!(get!(children, nameof(S0), Symbol[]), n)
    end
    flagged_types = Symbol[]
    for (n, T) in declared
        isabstracttype(T) || continue
        kids = get(children, n, Symbol[])
        length(kids) == 1 &&
            !isabstracttype(declared[kids[1]]) &&
            get(refs, n, 0) == 0 &&
            push!(flagged_types, n)
    end
    sort!(flagged_types)

    #=
    The allow-list of abstract types. Each names the realistic future application the type
    is the seam for, or the `# Interfaces` section that already states its contract. A type
    that gains a second subtype, or a method that dispatches on it, is no longer flagged and
    its entry is stale.
    =#
    kept_types = Dict{Symbol, String}(#
                                      :AbstractCrossSectionalRegressionResult => "a second cross-sectional regression result, a robust or a constrained fit",
                                      :AbstractJuMPResult => "a second JuMP result carrier, one that keeps the model",
                                      :AbstractPenalisedRegressionTarget => "carries an `# Interfaces` section: an Open Family",
                                      :AbstractSearchCrossValidationResult => "carries an `# Interfaces` section: an Open Family",
                                      :AbstractShrunkExpectedReturnsEstimator => "a second shrinkage estimator beside `ShrunkExpectedReturns`, kept by default by #1859 (ADR 0222)",
                                      :BaseGerberIQCovariance => "carries an `# Interfaces` section: an Open Family",
                                      :BaseHierarchicalOptimisationResult => "carries an `# Interfaces` section: an Open Family",
                                      :BaseJuMPOptimisationResult => "a second JuMP optimisation result beside `JuMPOptimisationResult`",
                                      :BaseSmythBrobyCovariance => "carries an `# Interfaces` section: an Open Family",
                                      :BootstrapUncertaintySetEstimator => "carries an `# Interfaces` section: an Open Family",
                                      :HierarchicalScalariser => "the max and sum scalarisers beside `MinScalariser`",
                                      :InverseMatrixSparsificationAlgorithm => "a second sparsification beside `LoGo`, a graphical lasso",
                                      :OptimisationModelResult => "carries an `# Interfaces` section: an Open Family")

    # ------------------------------------------------------------- the assertions

    # Floors on the census itself: a parser that quietly stopped matching would satisfy
    # every assertion below with an empty list. Measured 2026-09-21: 2765 names, of which
    # 34 flagged, and 19 flagged abstract types.
    @test length(defs) > 2000
    @test sum(values(refs)) > 50_000
    @test length(flagged) >= 20
    @test length(flagged_types) >= 10

    # A flagged name outside the allow-list is a pure forward with at most one caller, and
    # is inlined at that caller; a dead one is deleted. Both directions are named, so a
    # failure says which name moved and which way.
    unkept = [n for n in flagged if !haskey(kept, n)]
    stale = sort!([n for n in keys(kept) if !(n in flagged)])
    @test ("pure forwards with at most one caller, outside the allow-list", unkept) ==
          ("pure forwards with at most one caller, outside the allow-list", Symbol[])
    @test ("allow-list entries no longer flagged", stale) ==
          ("allow-list entries no longer flagged", Symbol[])
    unkept_types = [n for n in flagged_types if !haskey(kept_types, n)]
    stale_types = sort!([n for n in keys(kept_types) if !(n in flagged_types)])
    @test ("single-subtype abstract types nothing codes against, outside the allow-list",
           unkept_types) ==
          ("single-subtype abstract types nothing codes against, outside the allow-list",
           Symbol[])
    @test ("allow-list types no longer flagged", stale_types) ==
          ("allow-list types no longer flagged", Symbol[])

    # A public entry is public, and a private entry carries a reason. An entry that says
    # "public" for a name that is no longer declared so has lost its justification.
    for (n, why) in kept
        @test !isempty(why)
        if startswith(why, "public")
            @test (n, Base.ispublic(PO, n)) == (n, true)
        end
    end

    # The forward census (#2071). The floor guards the parser: measured 2026-11-13, 375
    # forward methods of 229 names.
    @test sum(length, values(forwards)) > 200
    forward_kept(pair) = haskey(kept, pair[1]) ||
                         haskey(interface_functions, pair[1]) ||
                         haskey(kept_forwards, pair) ||
                         pair in pending_forwards
    unkept_forwards = [(pair, forwards[pair])
                       for pair in flagged_forwards if !forward_kept(pair)]
    @test ("forwards outside the allow-lists: give each form to the owner",
           unkept_forwards) ==
          ("forwards outside the allow-lists: give each form to the owner",
           Tuple{Tuple{Symbol, Symbol}, Vector{Tuple{String, Int}}}[])
    stale_forwards = sort!([pair
                            for pair in union(keys(kept_forwards), pending_forwards)
                            if !haskey(forwards, pair)])
    @test ("forward entries no longer flagged: delete the entry", stale_forwards) ==
          ("forward entries no longer flagged: delete the entry", Tuple{Symbol, Symbol}[])
    flagged_names = Set(first.(flagged_forwards))
    stale_interfaces = sort!([n for n in keys(interface_functions) if !(n in flagged_names)])
    @test ("interface functions no longer flagged: delete the entry", stale_interfaces) ==
          ("interface functions no longer flagged: delete the entry", Symbol[])
    @test isempty(intersect(keys(kept_forwards), pending_forwards))
    for (pair, why) in kept_forwards
        @test !isempty(why)
        if startswith(why, "public")
            @test (pair, Base.ispublic(PO, pair[1])) == (pair, true)
        end
    end
    for (n, why) in interface_functions
        @test !isempty(why)
    end
    # Pinned on the wrappers that #2071 removed.
    for n in
        (:pipeline_data_view, :pipeline_asset_view, :windowed_estimator_suggest, :_Frontier,
         :declare_held_indicators!, :declare_long_short_indicators!,
         :declare_sign_indicators!, :_no_bounds_risk_measure,
         :_no_bounds_no_risk_expr_risk_measure, :exposure_transform,
         :expand_idiosyncratic_covariance, :expand_vector, :simple_variance_kernel,
         :expand_held_member, :expand_investable_columns, :reduce_member_scores,
         :weighted_coverage_covariance, :coverage_keyword_mask, :expand_factor_mu,
         :time_dependent_entry_needs_previous_weights)
        @test (n, isdefined(PO, n)) == (n, false)
    end

    # Pinned on the three units ADR 0168 named and the one dead-end the build removed.
    @test !isdefined(PO, :diagonal_geometry)
    @test !isdefined(PO, :assert_diagonal_bound)
    @test !isdefined(PO, :resolve_turnover)
    @test !isdefined(PO, :set_budget_costs!)
    @test !hasfield(AdaptiveSubgradient, :proj)
    @test !Base.ispublic(PO, :DiagonalProjection)
end
