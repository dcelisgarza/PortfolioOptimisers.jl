# Domain glossary of PortfolioOptimisers.jl

A Julia portfolio-optimisation library. The library organises the domain as a workflow:

> data → moments → prior → optimisation → post-processing

A small set of cross-cutting abstractions (Estimator / Algorithm / Result / Factory) is the base of everything else. This file is a glossary and nothing else. It names the concepts and fixes the words for them. The ADRs in `docs/adr/` record the decisions and their reasoning.

## 1. Core Abstractions

**Estimator**
A configuration object that encodes a statistical or mathematical method and its hyperparameters. It is the main input to the user-facing functions of the library. An Estimator says *how to compute for whatever input it is given*, so it never holds a Result.

**Algorithm**
A type inside an Estimator that selects or changes how the Estimator computes. A caller uses it through an Estimator, never on its own.

**Selector tag**
An Algorithm that has no field. Its only job is to name the branch that a caller takes. Dispatch on the type is all that it does. For example, `SpectralDenoise` selects the spectral branch of `denoise!`. Most subtypes of `AbstractAlgorithm` are selector tags.

**Open Family**
An abstract type whose docstring has an `# Interfaces` section. A subtype written anywhere, in a private path too, dispatches on the methods that the section names. An abstract type with one concrete subtype that no code is written against needs a reason to stay. An Open Family is one such reason, and a realistic future application is the other. Such a type stays only as a hand-written entry in the allow-list of the census that gates dead-end units, and the entry names the section or the application. The section is a precondition, not an exemption, because every abstract type that is the parent of a concrete type has one. ADR 0168.
*Avoid*: an exported or `public` type, which is a narrower thing. The section is the route to a `public` declaration (ADR 0154), not the other way round.

**Result**
A plain data struct that holds the output of a function applied to an Estimator. It is *the answer for the input it was computed on*. A Result is never callable.

**Partial Fit State**
The running quantities that an incremental fit keeps between calls: the observation count, the mean and the second-moment accumulator. With them, the fit folds one more observation into an estimate and does not read the sample again. A state is a Result that no consumer reads. The batch verb called with no data, such as `prior(pe)`, `optimise(opt)` or `fit(pipe)`, first turns a state into the ordinary Result. It is the one kind of Result that an Estimator holds, in a field bound to `Option{<:AbstractPartialFitState}`. ADR 0106 records the exception.
Two verbs fold observations into a state, and ADR 0107 records what each promises. `partial_fit!` is the method that each family writes, and it is the cheapest exact fold of that family. It writes into the arrays of the state where it can, and it promises nothing about an Estimator that the caller kept from before the call. `partial_fit` is one generic method with value semantics. It copies the state and folds the copy, so the Estimator that the caller passed reads what it read before. Every state has a method of `merge_states` and of `copy`.
*Avoid*: Cache. Code can drop a cache with no change to an answer, and it must not drop a state.

**State Merge**
The combination of two Partial Fit States, fitted on disjoint blocks of observations, into the state of the concatenated block. The verb is `merge_states`. The merge is a sum, not an overwrite in which the right operand wins. That is why the verb is not `Base.merge`. The merge makes an incremental fit parallel and associative. A sample split into any set of disjoint blocks gives the state of the whole sample, in whatever order the fit folds the blocks. A family whose state is not a sufficient statistic for its own block refuses the merge and names the reason. Its route is a sequential fit instead.

**Sample Buffer**
The observations that a member keeps verbatim in the Partial Fit State slot, non-finite entries included. A buffer has one of two roles. The two roles take two state types, because the type of the state is the route. A **refit** has no recursion at all. Its call with no data runs the batch verb over the rows of the buffer, so the cap of the buffer *is* a window. A **carry** folds its estimate exactly. It keeps the rows only because a consumer downstream reads them. For example, a Prior Result carries `X` for the scenario risk measures. So the cap of a carry bounds the carried rows alone, and the estimate stays fitted over every observation. The `Online` wrapper is the declaration that seeds a refit buffer. A carrying member seeds its own buffer. It has no choice, because its Result must carry the rows.
A refit buffer records factor returns beside its rows when the fold carries them. The estimator tree of a member decides whether the member reads them: required, never read, or *take what is given*. The type of the estimator that holds the member does not decide it. ADR 0106, ADR 0136.
A buffer that holds the returns also records the implied volatilities beside its rows, cell for cell. The first append pins the premium adjustment `ivpa`. So the volatility surface goes wherever the returns go, and the call with no data gives it to the batch verb. ADR 0137.
When its member reads the panel, a refit buffer also records a time-varying Asset Panel beside its rows, both masks and every Panel Field. A time-varying Panel Field is sample, not fold context. So the buffer of the prior owns it once, and the Fold Context reads it back, as it reads the factor returns. The online step then carries an estimation mask to every prior whose route honours it. ADR 0136 amended, ADR 0193.
The Exogenous Series follows the same rule. When a member of the tree of the prior reads one series, the state of the prior records every series and its names, on the refit and on the carry. The Fold Context reads them back. ADR 0193.
A Pipeline under `Online` buffers its **input data**, because its refit is the batch fit of the whole workflow, and that fit starts from the input. No member below it folds, so one buffer holds the rows. ADR 0142.

**Carry Fold**
The online route of an estimator that folds what can fold and refits the rest over the rows that it carries. It is the rule of ADR 0136. The Cross-Sectional Factor Prior folds as a carry when no `Online` wraps it. A new observation changes no past exposure, regression or idiosyncratic variance. So each step computes them for the new observations alone, and folds the factor prior and the idiosyncratic variance. The call with no data refits the Return Forecast and the idiosyncratic correlation. The prior carries the panel rows that its Descriptors read. It carries every row when one of the Descriptors, or a Return Forecast that reads the panel, has no finite look-back. ADR 0193, #1471.
A factor that comes alive had a return of zero at every fitted observation. So the step keeps the regressions of those observations, and it folds the factor prior again over every fitted factor return (#1627). A regression estimator that refuses the zero column of such an observation fits every carried observation again. When a Batch Choice of a dropped member moves, it changes only the basis of the factor returns. So the step fits the factor prior again, and it fits no past regression again (#1601). A Pinned Choice and a Seed Window keep a value of the first fit. So under either one, the call with no data equals no batch fit.
*Avoid*: reading it as a refit. It fits no past observation again unless a member does not fold.
A scheme under `Online` buffers nothing. There, the same word declares an Online Scheme, and the loop steps and does not refit. ADR 0167.

**Carry Rule**
What a Carry Fold does with a member whose step costs more as the stream grows. Such a member is a Descriptor with no finite look-back and no state of its own, a variance or a factor prior that does not fold, or a part that refits over every carried row. **Fold or Refit**, the default, is the rule of ADR 0136. The fold fits the member again over every row that it needs, so the step stays exact and its cost grows with the stream. **Fold Only** refuses such a member when the caller builds the estimator, so the cost of each step does not grow. A member that reads a bounded window passes both rules. The rule is a field of the estimator that folds as a carry. ADR 0193, #1590.
Fold Only refuses a Batch Choice that can move, because each move fits the factor prior again over the whole stream. A Pinned Choice or a named member passes. #1601.
*Avoid*: reading Fold Only as a faster route. It changes no value and no cost. It only refuses a configuration whose step grows.

**Scenario Cap**
The number of observations that a fold-and-carry Prior Estimator keeps in the `X` of its Result. Its name is `max_scenarios`, and it applies in batch and online alike. It bounds memory and the zero-filled share that a Scenario Fill reports. It does **not** move `mu` or `sigma`, which stay fitted over every observation. So a capped fit equals no batch fit, and the documentation states that divergence and no test checks it. ADR 0136. A capped Result states in `ens` the count of observations that its moments were fitted over. A consumer that prices a sample size, such as an Uncertainty Set (§3.9) or a calibration rule, reads that count and not the rows carried. A bootstrap reads the cap, because it can resample nothing but the rows carried. ADR 0138.
A calibration rule that prices a property of the scenarios of the measure reads the rows carried too. `ScenarioCount` fixes the count of scenarios in the tail, and `EntropyBudget` fixes the coefficient that the model pays over its own rows. ADR 0138, amendment of 2026-10-26.
A distributionally robust measure centres its ambiguity ball on the empirical measure of the rows carried, so it states the count of those rows to its radius rule in the Calibration Context (§3.9). A regularisation penalty prices the error of `mu`, so its radius reads `ens`. ADR 0138, #1833.
Under an entropy pooling posterior, the moments are those of the scenarios. The posterior gives no probability to an observation that the cap removed, so `mu == pr.X' * w`. ADR 0209, #1667.
*Avoid*: reading `size(X, 1)` of a capped Result as its sample size. That is the number of scenarios that it carries.
*Avoid*: reading it as `max_history`, the cap of the `Online` wrapper, which windows the whole fit and does have a batch equal. A caller can set both on one estimator, and they nest.

**Choice Rule**
How a choice that a fit makes over its whole sample behaves on the online step. Such a choice is the dropped member of a Factor Family, the factor set of a stepwise regression, or the projection of a dimension-reduction regression. A **Batch Choice** is the default. It chooses again over every observation at each fit, so the online call with no data equals the batch fit over the same rows. A **Pinned Choice** keeps the choice of the first fit, so the basis of a fold stays fixed and each step stays cheap. The two rules agree in a batch fit, an entropy pooling prior included. That prior refits under the posterior weights, and every refit holds the regression of the fit that its views read, so it reads that choice (#1641, #1649). The rule is a field of the estimator that makes the choice. ADR 0193.
On a Carry Fold, a Batch Choice that moves changes only the basis of the factor returns. So the step fits the factor prior again, and it fits no past regression of full rank again. A rank-deficient observation does not fix its factor returns, so the step solves it again in the new basis. Such an observation has an Unseen Member under the Solved Unseen Member rule, or a dependent factor set. Under a regression estimator that fits a target, the step folds the same way when the target gives the same coefficients in every basis. Otherwise it fits every observation again. An iterative target, such as a generalised linear model, folds too, because it takes the same iterates in every basis. #1601, #1605, #1613, #1614, #1615, #1616.
*Avoid*: reading a Pinned Choice as a state. After the first fit, a refit honours it with the name of the chosen member in its configuration.

**Fold Context**
What an optimiser keeps beside its Prior Estimator when it takes the online step. It has two parts. The first part is every column of the `ReturnsResult` that the prior does not own, each as a buffer of its own. These are the benchmark and timestamp columns, and the factor column and the Exogenous Series only when the tree of the prior never reads them. The second part is the context that the first step pins and every later step checks: the asset, factor and benchmark names and a static Asset Panel. A `ReturnsBufferState` in the Partial Fit State slot of the optimiser holds the Fold Context. From it, `optimise(opt)` with no data rebuilds the `ReturnsResult` that a batch fit over the same observations would have read. The prior owns the returns, and nothing else holds a second copy. A head that holds no prior is the bottom of its own chain and keeps the returns in its Fold Context.
A Pipeline holds a Fold Context too, but only when its prior step owns the rows. The Pipeline is the one outer estimator whose own estimator is not the row owner, because `inject_context` overrides the prior of the optimiser with the prior of the step. When the optimiser step owns the rows, that optimiser already holds the context. ADR 0142. No step pins a time-varying Asset Panel. Its active mask goes into the returns buffer beside the rows, and the call with no data rebuilds the panel from it. ADR 0137.
An Online Portfolio Selection head (§4.1) is the one optimiser that takes a step of its own above the prior, and it steps its fallback beside it. Its call with no data returns a Recursion Result. Its state pins the same names and static panel, and it records the timestamps. It holds no column, because the call rebuilds nothing. ADR 0137 amended, ADR 0155.
*Avoid*: reading it as a second Sample Buffer. It holds no returns where a prior does, and it never holds an estimate. And reading `optimise(opt)` with no data as an online solve. It rebuilds the `ReturnsResult` and runs the ordinary batch path, so nothing above the prior takes a step of its own.

**Choice Surface**
The set of things that a caller picks to specify a problem. It is every concrete type that the package declares and that is a leaf Estimator, a leaf Algorithm, a leaf `AbstractCovarianceEstimator`, or an export under its own name. A caller gets Results and errors back and never chooses them, so they are not in the set. The set names `AbstractCovarianceEstimator` on its own, because it descends from `StatsBase.CovarianceEstimator` and not from Estimator. The export rule catches any family whose root this list does not name.

**Factory**
The mechanism that propagates runtime-computed values down a composed struct tree. It returns a new, fully-configured struct of the same type.

**Deferred Quantity**
A moment Estimator in the place of a value derived from the prior. So the slot takes either the value or the method that computes it. Factory resolves it against the Prior of the optimisation.

**View**
An index selection propagated down a composed struct tree. It restricts an Estimator, Algorithm or Result to a subset of assets or observations.
The View of a Result restricts what the Result states and never fits it again. For example, the View of an Uncertainty Set is the projection of the set, which is the worst case of a portfolio that holds nothing outside the selection. An Estimator that crosses the selection fits the selected data instead.
*Avoid*: using it for a **Tail View** (§3.6), which is a statement about the distribution rather than a selection.

**Propagation Channel**
One generated method of `@propagatable`, and the field tag that opts a field into it. Three exist: factory, view, and prior.

**Pipeline**
An Estimator that represents an end-to-end workflow as an ordered list of steps, each with an optional name. It runs the steps left to right over a Pipeline Context. The Pipeline takes the online step for its steps, and the new rows go through the steps in order. A step whose replay of an earlier row never changes folds the rows into its own Partial Fit State. A step whose fitted state is a column set folds nothing, and `fit(pipe)` with no data applies it as a view on the state of the row owner. The row owner folds the rows that it gets. The row owner is the prior step, else the optimiser step, else a head with no prior. `fit(pipe)` with no data is the batch fit over what the steps hold. So a universe that moves between two steps is a view, never a re-sliced state and never a frozen selection.
A step that writes a Data Slot before the row owner and has no online form refuses the online step, with an error that names the step. `Online(pipe)` is the declared refit of the whole workflow from a buffer of its input, and with a cap it is a rolling Pipeline. ADR 0142.
A cap on the row owner alone, `Online(pe; max_history = w)`, is a window counted in the rows of the owner. A row-local step before it folds a state across the front edge of that window. So the fold route refuses the pair, with an error that names both steps. The rolling window through a Pipeline is `Online(pipe; max_history = w)`. ADR 0142.
The online form of a row-local step is one verb, `partial_fit_transform`. It folds a block of observations and returns the block that its transform gives. The two actions cannot be separated, because the first return of a block reads the last price row that the step kept. The call with no data of such a step is `fit_preprocessing` with no data. `partial_fit!` on such a step is the first element of the return value of the verb. ADR 0142.
*Avoid*: Workflow, Workbench. And using "pipeline" for the informal order of stages in the library. And reading `fit(pipe)` with no data as a step of its own. It rebuilds the context and runs the ordinary steps, as `optimise(opt)` does one layer down.

**Pipeline Context**
The blackboard that the steps of a Pipeline pass along and add to. It is a set of coarse typed slots that each step reads from and writes to.

**Routing Target**
The field inside the optimisation step that receives a computed Pipeline Context slot. A target is finer than a slot. A slot names a stage of the workflow, and a target names a destination field.

**Data Slot**
A Pipeline Context slot whose write changes the asset universe: prices and returns. The opposite is a *derived slot* (prior, phylogeny, uncertainty, constraints, weights), which a step computes from the data.

**Preprocessing Estimator**
The family of Estimators that transform price or returns data inside a Pipeline. The fit of one gives a Result that carries its fitted state. A later call applies that state to unseen data.

**Holdout Split**
The evaluation protocol that reserves the tail of the time-ordered observations as a test window and trains on the head.
*Avoid*: Validation Split. No step tunes anything on the held-out window.

**Embargo**
The deliberate gap between the training and test windows of a Holdout Split. It belongs to neither window.

**Asset Selector**
The Preprocessing Estimator subfamily that restricts the asset universe from returns data. Its fitted state is the set of surviving asset columns.
*Avoid*: Filter, Screen. These words belong to `MissingDataFilter`, which drops and never selects.

**Selection Rule**
The Algorithm that a `ScoreSelector` uses to turn per-asset scores into a keep-mask. *Literal* rules compare raw scores against absolute bounds. *Ordinal* rules read the orientation and take counts or fractions from each end.

**Redundancy Algorithm**
The Algorithm that a `RedundancySelector` uses to decide which assets duplicate information that other assets already carry. It returns a keep-mask, not a partition.

**Trust-neither tie policy**
The convention that when two assets are indistinguishable under the applied criterion, the library keeps neither.

**Vector-to-Scalar Reducers**
Small reusable Algorithms that collapse a vector of reals to a scalar: `MinValue`, `MaxValue`, `MeanValue`, `MedianValue`, `ModeValue`, `StdValue`, `VarValue`, `SumValue`, `ProdValue`, `StandardisedValue`.

**LxNorm error family**
The norm forms used as constraints, as risk-measure targets, and in entropy pooling: `L1Norm`, `L2Norm`, `SquaredL2Norm`, `LpNorm`, `LInfNorm`.

**FullMoment vs SemiMoment**
The Algorithm distinction that runs through all of moment estimation. `FullMoment` includes all deviations, and `SemiMoment` includes only those below a target. It is the basis of the split between symmetric and downside measures.

**Moment Target**
The value from which a moment risk measure measures the deviations of the net portfolio returns. `calc_moment_target` selects it from the `mu` slot of the measure: the mean of the returns, weighted by the observation weights when the measure holds them, when `mu` is `nothing`. It is the expected portfolio return net of the mean fee per period when `mu` holds expected asset returns, and the number itself when `mu` is a number. A number is a minimum acceptable return, so it takes no fee.

## 2. Data

**ReturnsResult**
The central data structure that carries all return series through the library: asset, factor and benchmark returns, the Exogenous Series, their names, timestamps, implied volatility, and the Asset Panel. `prices_to_returns` makes it.
Its asset returns are in the base currency of the investor, with or without Currency Factors (§3). A model can need returns net of its Observed Factors. Under Currency Factors, these are the local-currency returns. Such a model derives them, or reads them from a Panel Field that the caller names. It never reads the asset returns as local. See ADR 0184.
Code that builds a `ReturnsResult` from another one carries every field, and drops one only when the field has no meaning on the new axis. See ADR 0204.

**Excess Frame**
The asset returns that a reader fits on when it states `x_src = ExcessSource()`. They are `X - B`, and `benchmark_relative_returns` gives them as a matrix at the point of use. The readers are a prior, a factor regression, the prior of an uncertainty set, a clustering, phylogeny or centrality estimator, and the step of an online portfolio selection head. The `ReturnsResult` stays raw. It keeps `X` and `B`, each reader subtracts once, and no head, inner optimiser or fallback forwards a frame. Returns with no benchmark have no Excess Frame, and the readers refuse them. See ADR 0211 and ADR 0215.
A Prior Result records the source of the prior that fitted it. So three kinds of Result keep their frame where an optimiser receives the Result in the place of its prior: the Result of the call with no data of an online fold, a prior step of a `Pipeline`, and a precomputed Result. The outer problem of a meta-optimiser adds `B` back to the returns of an excess Result. See ADR 0215.
A covariance forecast evaluation with `x_src = ExcessSource()` fits, steps and scores its forecast in the Excess Frame, and its Result records the frame. The frame is `DataSource()` for `x_src = nothing`. The verb states the source, because its estimator receives a matrix.
An online portfolio selection head with `x_src = ExcessSource()` steps on the price relative `1 + r - b` of the Excess Frame, and every reader of the rows inside its rule reads `X - B`. Its rows stay raw, so a leader and the prior of a Risk Loss or of a programme set state their own source, and `predict` reports the raw wealth of the fund.
*Avoid*: relative returns, which can also mean a ratio to the benchmark.

**Returns Source**
The returns that a reader of the returns states that it fits on: the returns data as they arrived, the returns data in the Excess Frame, or the prior. The prior gives its returns in the frame that it fitted in, and gives a regression its loadings. A reader that states no source reads the returns data, and a regression first takes the loadings of the prior. See ADR 0211 and ADR 0212.
The readers of one optimiser can state different sources. Take a reader that states no source and reads the raw returns, beside a reader in the Excess Frame. It gets a warning when the caller builds the optimiser, and a stated raw source stops the warning. See ADR 0214.
A graph reader that reads the returns data meets each gap that the prior filled, and its own moment estimator decides what the gap is. A mask-aware estimator reads the observed cells, so no **Scenario Fill** (§3.6) reaches the clusters. Inside a fit, a plain estimator leaves the gapped asset out of the **Investable Mask** (§3.6) and names it, and outside a fit it refuses the gapped sample. See ADR 0211.
*Avoid*: matrix source, which also names the source of a feature matrix.

**Data Benchmark**
The benchmark that the returns data carries, the `B` of a `ReturnsResult`, used as a tracking benchmark. A caller states it as `DataBenchmarkTracking()`. It holds no series of its own. Each reader that holds the returns data binds it to the `B` of its own rows with `bind_data_benchmark`: an optimiser after its view at the Investable Mask, the online step on its buffer, a fold on its training rows, an online portfolio selection head on its rows window, and the value-level `expected_risk` on a `ReturnsResult`. So it follows the rows where a `ReturnsTracking` series, fixed at construction, cannot.
A vector `B` is the benchmark series. A matrix `B` holds one benchmark per asset, and the series is `B * w`, so the tracking error is that of `(X - B) * w`. A prior in the Excess Frame refuses it, because the tracking error would subtract `B` twice. Only the prior whose returns the measure reads counts, so a clustering in the Excess Frame does not. A plain risk measure on a prior in the Excess Frame gives the active risk instead.
*Avoid*: a returns benchmark, which names the fixed series of `ReturnsTracking`.

**PricesResult**
The container of aligned, time-indexed price-level series. It is the prices-level mirror of `ReturnsResult`, and the input to price Preprocessing Estimators. It also carries the Exogenous Series as level series, which `prices_to_returns` converts with the return rule of the assets.

**Exogenous Series**
A named series over the observation axis that belongs to no asset, such as a Currency Excess Index (§3), a macro index or an observed market index. A `PricesResult` holds its levels and a `ReturnsResult` its returns, converted with the return rule of the assets. A consumer reads a series only by its name, so no model takes the block whole as its factors. Every observation slice of the data that the series goes with also slices the series. The series never adds or drops an observation of the assets. The ingestion aligns it by timestamp, and a gap stays a gap. The consumer that names the series refuses an infinite value only on the rows that it fits, and it reads a `NaN` by its own gap rule. See ADR 0184.
*Avoid*: factor returns, which a time-series factor model reads whole as its factors. And a Panel Field, which is per asset.

**Feature Matrix**
The assets × features matrix a Feature Distance measures, or its time-varying form, observations × assets × features. The library builds it at the point of use from the Panel Fields that a Feature Selector names. A numeric field gives one column, a categorical field one 0/1 column per level, a tensor field one column per label, and an observed mask one 0/1 column. Each column carries a label, which is the Feature Selector entry that selects exactly that column. A Placeholder in a value column becomes `placeholder`, `NaN` by default, in the matrix. A cell that a fill policy wrote holds data, and the matrix keeps its value. Every other cell is finite.
The time-varying form holds the observation rows that its consumer reads, not every row of the panel. A Feature Distance stacks the rows that its collapse algorithm names. So under the default `LastRow`, the matrix of the kernel is the last observation alone, 1 × assets × features, and the kernel reads a lifted static field once. The active mask of the stacked rows goes with the matrix. A matrix stacked by hand from the panel holds every row unless `rows` says otherwise.
*Avoid*: reading it as a stored quantity. No `PricesResult` or `ReturnsResult` holds a Feature Matrix. An Asset Panel holds Panel Fields, and the library derives the matrix from them. Also Characteristic (see **Characteristic Vector**, §3.9). And reading "feature" as *factor*, which is a return series.

**Feature Selector**
The statement on a Feature Distance that says which Panel Fields the Feature Matrix stacks. An entry names one field, or one field with the levels or labels that it keeps, or one tensor field with one group of its labels (`LabelGroup`, a Factor Family on an exposure tensor), or the observed mask of one field. A bare field name means the values of the field alone. An absent selector means the values of every field. The library drops an entry that names nothing that the panel holds, with a warning. Under `strict`, it refuses the entry.
*Avoid*: reading an entry as a column position. Every column has a name. And a taxonomy key, which is the name of a categorical Panel Field, not a second namespace.

**Asset Panel Estimator**
A producer that turns something the library already computes into a static Asset Panel with one Panel Field, at the point of use. `RegressionPanel` builds a loadings tensor from the regression of the prior, and `PhylogenyPanel` builds a proximity tensor from the returns. It is configuration on a Feature Distance, so it refits on the universe of the subproblem that runs it, and it holds no data.
*Avoid*: reading it as a source of a Feature Matrix. It makes a panel, and the matrix is stacked from that panel as from any other panel. And a classification that the caller supplies, which enters `asset_panel` as a categorical Panel Field, not through a producer.

**Asset Panel**
The data of the universe. Its **two masks are what it states**, and its named Panel Fields are optional per-asset data over a shared asset axis. Its static shape carries fields alone, indexed by asset and with no masks: a taxonomy, a loadings matrix, an adjacency. Its point-in-time shape is the active mask and the estimation mask over an observation axis, and it is complete with no field at all. The ordinary case of the ingestion layer is two masks and no feature data. So when there are no fields, the panel reads the shared axis from the masks. See ADR 0129.
`asset_panel` resolves every blank cell before the Asset Panel exists. A Universe Sets key enters as one Panel Field through `panel_input`. A string key gives a categorical field, and a numeric key gives a numeric field. The field takes the name of the key without its asset-axis prefix. A static input that meets a time-varying input, or the two masks, is lifted to the observation axis as a field that is constant in time. See ADR 0102.
*Avoid*: reading masks of `nothing` as *every asset is active*. That is what the sentinel means for a **Coverage Universe** (§3.6) and an **Investable Mask** (§3.6). On a panel it means that the panel is **static**. A panel that the ingestion layer emits always carries both masks. This holds for a price table with no gaps too, and its masks are then all true. So a `ReturnsResult` that holds no panel at all was built outside the layer. See ADR 0132. And reading `pnl` as profit-and-loss. The field holds the panel. And reading the panel as one matrix. It is a set of fields, and the Feature Matrix is stacked from them. And reading a graded or weighted membership matrix as anything but a tensor Panel Field that the caller writes as data.

**Panel Field**
One named quantity of an Asset Panel. It owns its values and its observed mask. Examples are a market capitalisation (numeric), a sector classification (categorical, integer codes over declared levels), and a factor exposure tensor (tensor, with third-axis labels and optional groups). When a fill filled the blanks of a Panel Field, the field carries an observed mask that shows which cells the fill touched. It also carries a placeholder mask that shows which of those cells hold a Placeholder. A field whose policy refuses blanks carries neither mask. Under an asset view, a Panel Field follows the universe, and a tensor field whose labels are the asset names is sliced on its label axis too. Under a meta-optimiser collapse, a Panel Field becomes its convex combination on the Sub-Portfolios. A categorical field becomes a tensor field of membership fractions over its levels, and a mask holds where any weighted member holds.
A cell outside the universe keeps a finite value too (ADR 0102). So every consumer reads the active mask to leave the cell out, and a Descriptor reads it as `NaN`. A caller who wants a marker in the values chooses it at read time through `panel_field_values`. The choices are `inactive = NaN`, `inactive = 0`, or `nothing` for the stored value, and `unobserved` takes the same choices for a filled cell. The panel stores no policy. `test/test_06j_inactive_cell_census.jl` holds every consumer to it (#1411).

**Placeholder**
The value that a Panel Field stores in a blank that no fill reached: the `val` of a `ForwardPanelFill` before the first observation of a stretch, or past the run of its `lim`, and the zero row of a `RegressionPanel` outside the Investable Mask. A Panel Field holds no blank, so the cell needs a finite value, but the value is not data. The placeholder mask `pmsk` marks it, and no reader reads it. A Feature Distance and a meta-optimiser collapse skip it as they skip an inactive cell (ADR 0102, #1631). Any other value that a fill policy writes is data. Such values are the constant of a `ConstantPanelFill`, which the absence means, and a value that a directional fill carries.
A Feature Distance, the Feature Matrix and a meta-optimiser collapse read a filled cell. The descriptors, the exposures and `panel_field_values` read the observed mask, so they read a filled cell as unobserved.
*Avoid*: calling every unobserved cell a placeholder. A cell is unobserved when the raw input held a blank there, and a filled cell is unobserved and holds data.

**Panel Manifest**
The small table that describes an Asset Panel beside the table of its values, so that a reader can read the panel back. It holds the shape of the panel, its asset names and observation labels, the kind of each Panel Field and whether it carries an observed mask and a placeholder mask, the levels of a categorical field in code order and whether its column holds the levels or the codes, and the axis name, labels and groups of a tensor field. `panel_manifest` writes it, `panel_dataframe` writes the values, and `asset_panel(df, mf)` reads the two back after any table format, CSV included, which keeps no type and no metadata.
*Avoid*: reading the values table alone as the panel. A column name cannot hold a level that no cell uses, the order of the levels, or the groups of a tensor field, and the long layout drops every asset and observation outside the universe. And storing the manifest as table metadata, which CSV drops.

**Listing Span**
The interval of the price clock over which an asset is listed, per asset column. It runs from the first priced observation of the asset to its last. The **Span Rule** (below) derives it. It is the compressed form of the active mask of an Asset Panel, and it is exact, not approximate. An interior gap leaves an asset listed, so its active set *is* that interval, and the interval costs two integers instead of a boolean per cell. A column that is a gap throughout is the empty interval. The span is input data, not fitted state, because a listing calendar is a fact about the instruments and not an estimate from returns. That is why the library can derive it once over the whole panel and slice it per fold. Its type is a matrix of booleans. So a declaration of the caller, such as a listing calendar or a constituency that leaves and rejoins, enters at the same point, and the library does not second-guess it. See ADR 0129.
*Avoid*: reading it as the active mask itself, which is observations × assets on the *returns* clock. The span is on the price clock, and it projects to `[first + 1, last]`. A return uses the earlier price of its pair. So the first return of an asset is one observation after its first price, and its last return is on its last price. And a per-window derivation, which reads a delisting that straddles the end of a window as dead, not held.

**Span Rule**
The reading that turns the *position* of a gap in a price column into the universe that it implies. A leading run of gaps is an asset not yet listed, and a trailing run is a delisting. An interior gap is a suspension or a holiday on an asset that is still listed and still held. With this rule, a caller who holds only prices can state a universe, because the position already carries the distinction that a listing calendar would give. Its output is a **Listing Span** (above). The estimation mask is then the active mask intersected with the finiteness of the returns. See ADR 0129.
*Avoid*: reading it as a policy. It drops nothing and fills nothing. It only says which assets are in the universe at each observation. A **Universe Policy** (below) that drops runs after both masks are fixed. A caller states a **Held Price** (below) before the masks, so the estimation mask records it.

**Universe Policy**
A judgement about the data that changes the asset universe, such as a coverage floor, a redundancy rule or a score threshold. It is always fitted on a training window and replayed by name on every later window. It is never derived over the whole panel. The *number* can be the caller's, but the *set it selects* would come from observations that postdate every fold, and that is survivorship bias in its exact definition.
It always runs **after the span**. At the returns level it is an `AbstractAssetSelector`, and `port_opt_view` slices the masks and the Panel Fields out with the column. At the price level it is `MissingDataFilter`, and `listing_span_view` slices the span and does not derive a new one. The conversion holds no policy. It computes a return, so a deletion is always a step that a caller composed. The **Coverage Universe** or **Coverage Policy** (§3.6) answers "not enough history", and so does the warm-up of a mask-aware estimator when it reads a partial column. See ADR 0130 and ADR 0133.
*Avoid*: the **Span Rule** (above), which reads a fact about the instruments, so the library derives it panel-wide. That license does not extend to a judgement. *Avoid*: computing one again per window without a fit, which moves the universe between train and test and makes the fold refuse.

**Held Price**
The price that a fill carries across a **Held Gap** (§4.6). It is the last priced observation, held forward. So the returns inside the gap are flat, and the return of the first priced observation after the gap carries the whole move across it. A caller states it to mean *the price did not move*, as in a suspension or a holiday. It conserves wealth, where a zero in place of the gapped returns discards the whole move across the gap. The **Listing Span** (above) bounds the fill that states it. So the fill touches Held Gaps alone, and it never fabricates a price where the asset was not yet listed or was delisted. The fill is off unless the caller asks for it. It runs at the price level, so a filled cell is finite in the returns and the estimation mask says that the asset traded. The fill also admits a per-asset constant, and that convention is the weaker one. A fill of a suspension with a reduced price makes two moves that the market never printed. See ADR 0130.
*Avoid*: `gap_fill_value`, a trait of a covariance estimator that names the value that the estimator puts in place of a gap that it gets. That is a private repair of an estimator. A Held Price is a statement of the caller about the data.

**Gap Return**
The return on the observation that ends a **Held Gap** (§4.6). A return is the change between two consecutive observations, so by default there is none. A run of `k` gapped prices leaves `k + 1` non-finite returns, and no cell records the move across the gap. The inception of an asset reads the same way, so a re-pricing after a suspension reads as a re-listing. A caller who wants the move booked states an algorithm. `CatchUpGapReturn` puts `P[t+k] / P[t-1] - 1` on the re-pricing observation, and so shortens the Held Gap to `k`. An algorithm can write only a cell that reads a gapped price, inside the **Listing Span** (above) of the asset and after its first price. So a gap can never spread, and no algorithm invents a return before a first price. The estimation mask reads the values that it gets, and it does not know which algorithm made them. So a caller who books a catch-up owns a `k + 1`-period observation in one-period moments. See ADR 0131.
A cell computed from two observed prices is frozen even when it is non-finite: a zero price is not a gap, and its `-1` or `-Inf` return and the `Inf` after it keep the values of the conversion.
*Avoid*: a forward fill, which is not one. It rewrites the price on the gapped observations, so it is a **Held Price** (above) stated before the masks, where a Gap Return is stated after the conversion and touches no price.

**Implied Volatility**
A forward-looking estimate of an asset's expected price fluctuation, derived from current option prices. It is not a historical measurement.

**Implied Volatility Premium (VRP)**
The difference between Implied Volatility and realised volatility. It is usually positive, because investors pay to hedge downside risk.

## 3. Statistics

**Windowed Estimator**
A moment estimator that restricts an inner moment estimator to a sub-window of the observations, then passes the sub-window to it. The window decides *which observations the inner estimator reads*, never how it computes the moment from them.
Its window follows one of two rules, and the two agree in a batch fit. A **Rolling Window**, the default, keeps the last observations of every fit, so on the online step it refits over a window that moves. A **Seed Window** cuts only the observations of the first fit, and the inner estimator then folds every later observation. A Seed Window needs an inner estimator that folds exactly, and an estimator that holds it and folds. A refit has no first fit to remember, so it refuses a Seed Window. ADR 0193.
*Avoid*: reading a Seed Window as a state of the Windowed Estimator. Its first fold uses up the window. The fold returns the estimator with `window = nothing`, and the inner estimator holds the state of the fold. #1469.

### 3.1 Expected Returns (Moments)

**Expected Returns Estimator**
Computes a per-asset mean-return vector. Variants:

- **SimpleExpectedReturns**: sample mean, with optional observation weights.
- **ShrunkExpectedReturns**: shrinks the sample mean toward a target. Targets: `GrandMean`, `VolatilityWeighted`, `MeanSquaredError`. Algorithms: `JamesStein`, `BayesStein`, `BodnarOkhrinParolya`. The target and the algorithm are independent choices, so the estimator has nine combinations. Under `BodnarOkhrinParolya` the three targets return the same estimate, because that algorithm reads only the direction of its target and all three point along the vector of ones.
  `JamesStein` and `BayesStein` blend the sample mean and the target by an intensity, and their field `intensity` holds the rule that clamps it. `UnitIntervalIntensity`, the default, clamps it into `[0, 1]`, so the blend stays between the two. `NonNegativeIntensity` clamps it at zero, so the sample mean replaces a negative intensity. `UnclampedIntensity` keeps the raw formula. `SteinShrinkage` ignores the field and clips into `[0, 1]`.
- **EquilibriumExpectedReturns**: implied (equilibrium) returns by reverse optimisation, Π = λ·Σ·w_eq. It is the equilibrium anchor of Black-Litterman.
- **ExcessExpectedReturns**: returns net of a reference rate or a risk-free rate.
- **MedianExpectedReturns**: the per-asset median, weighted or not.
- **StandardDeviationExpectedReturns** / **VarianceExpectedReturns**: the standard deviations or the variances of the assets, for a "return" slot that carries dispersion.
- **CustomValueExpectedReturns**: user-supplied per-asset values.
- **WindowedExpectedReturns**: the Windowed Estimator for expected returns.
- **PriceLevelExpectedReturns**: the return that takes the last price to a statistic of a window of price levels. The statistic is a moving average, an exponential moving average, a spatial median, a window peak, a lagged price or a reweighted relative, on a typed `alg` slot. It is the Price Relative Forecast of the Online Portfolio Selection family, which the estimator reconstructs from returns with the last level at one. A windowed statistic truncates its window over the first rows. A folding statistic (`ExponentialMovingAverage`, `ReweightedPriceRelative`) is an exact recursion over the relatives, and a `PriceLevelForecastState` holds it. ADR 0158.
A folding statistic is mask-aware, so its batch arm and its online arm read a gap the same way. It folds the active assets of a row, and it resets an asset that the active mask turns off to the cold seed of that asset. It reads a gap at an active asset as a level that did not move. It answers `NaN` for an asset that has folded no level, and the head holds the leg of that asset flat, as it holds any forecast that is not defined. Its batch arm over an `AssetPanel` is that fold, so the fold and the batch agree by construction. A windowed statistic takes the Coverage Universe of its window instead, and it refuses a mask. ADR 0172.
A folding statistic can carry a memory beside its vector, the last `memory_rows` relatives on the same state. A recursion that also reads a window (`KernelTrendPattern`) then folds like any other, and the head holds no rows for it. Its batch form is the recursion from the first row. #1191.
- **Composite price-level statistics**: `TruncatedExponentialMovingAverage`, `GaussianWeightedDoubleEstimate`, `TrendSwitch` over a trend test (`PairwiseSlopeSum`, `RegressionSlope`) and `CompositeTrend`. The later papers of the forecast-reading arm read these statistics. Each one truncates every window it holds over the first rows, and one that holds a folding member reads every row. ADR 0158, ADR 0165.
- **PriorExpectedReturns**: the adapter that reads the mean of a Prior as an expected-returns vector, so any `me` slot can hold a Prior. It refuses a Prior that needs factor returns, because the expected-returns interface carries none. ADR 0158.

**Return Forecast**
A per-asset prediction of the idiosyncratic return of the next period. The caller supplies it, or an estimator fits it from Descriptor Scores. A Prior splits it against its latest Factor Exposures into two parts. The spanned part blends into the factor mean, and the orthogonal part enters the mean vector of the Prior under a scale.
*Avoid*: `alpha` (§5), which is the Significance Level of a tail. Also Expected Returns (above), which is a moment estimated from the sample.

**Prequential Calibration**
The rule by which a fitted Return Forecast reads the predictions that put it into return units. The model fitted on the targets that had matured at an observation predicts that observation, and this prediction is the forecast that the member published there. The rule reads no later observation. So a new observation leaves every earlier prediction as it is, and the batch fit and a step of the carry fold run one rule. It is the default `cv` of `TargetReturnForecast` (`PrequentialCalibration()`, ADR 0200). A plain `LinearModel` folds its fit through its normal equations (`NormalEquationsFit`). Another regression target fits again at each observation.
*Avoid*: an out-of-fold calibration, which a cross-validation estimator in the same field gives, and whose early folds train on later observations. Also Forecast Calibration (below), which measures the scale of a forecast after the fact, and walk-forward, which refits on a schedule of the caller.

**Spanned Shrinkage**
The weight in `[0, 1]` that a Cross-Sectional Factor Prior gives its fitted factor mean against the spanned part of its Return Forecast. A value of one keeps the fitted factor mean, and a value of zero takes the spanned part alone. It is a Calibration Slot (§3.9), so a Calibration Rule can compute it from the fit.
Its default is the precision blend, a rule that weighs the fitted factor mean and the spanned part by the error of each, so with no Return Forecast it shrinks the factor mean towards zero.
*Avoid*: Orthogonal Forecast Scale (below), which scales the other part of the forecast.

**Orthogonal Forecast Scale**
The scale in `[0, ∞)` that multiplies the part of the Return Forecast that the latest Factor Exposures do not span, before that part enters the mean vector. A value of zero drops the part, and a value above one scales it up. It is a Calibration Slot (§3.9), so a Calibration Rule can compute it from the fit.
Its default is one. The calibration slope of the orthogonal part on the next idiosyncratic return is the opt-in rule, because it reads the Return Forecast history.
*Avoid*: Spanned Shrinkage (above), which blends the other part of the forecast. Also `orthogonal_scaling` of the orthogonal uncertainty sets, which scales a penalty and not a forecast.

**Orthogonal Forecast Fit**
The rule by which a Cross-Sectional Factor Prior makes the orthogonal part of a fitted Return Forecast carry the scale of its fit. A fitted member regresses the forward idiosyncratic return, which the regression makes orthogonal to the Factor Exposures. So the part of its forecast that the exposures do not span has too small a scale. The default neutralises the Descriptor Scores of such a member against every estimated factor before the fit. Another rule fits the calibration slope on the orthogonal part, and a third reads the member as it is.
*Avoid*: Orthogonal Forecast Scale (above), which multiplies the part after this rule has acted, and which a caller states or a Calibration Rule computes.

**Systematic Repair**
The rule by which a Cross-Sectional Factor Prior decides which steps of its matrix processing run on the systematic block of the asset covariance, the factor covariance projected through the latest Factor Exposures of the investable assets, before it adds the idiosyncratic block. The block has rank at most the number of factors, so a positive-definite repair of it moves it by round-off alone. The prior repairs the sum of the two blocks under every rule. The default runs every step on the block except the positive-definite repair, and the other rule runs every step. Each rule is named for what the prior does to the systematic block before it adds the idiosyncratic block. ADR 0199.
*Avoid*: the repair of the factor covariance, which the factor matrix processing does on the factor axis before the projection.

**Return Forecast Estimator**
A producer of a Return Forecast. It makes the forecast from Descriptor Scores and the factor-model block of the Prior it serves, or from a stated vector. It scores its Descriptors over the whole history of the `ReturnsResult`. It reads the block on the rows of the block, which are the last rows of the `ReturnsResult`, so its Result is on the axis of the block.

**Descriptor Scores**
The cross-sectional scores of a set of Descriptors. Each score is winsorised and standardised per observation. It is then regressed out against named Factor Exposures under the estimation mask or the regression weights of the factor-model block, and standardised again under the same weights. Every fitted Return Forecast Estimator starts from these scores.
*Avoid*: Factor Exposure (§3.4), which is scored under the benchmark weights and names a factor of the model.

**Forecast Unit**
The unit in which the descriptors of a Return Forecast Estimator forecast. It is the idiosyncratic return, or the idiosyncratic Sharpe ratio. The estimator converts a Sharpe ratio back with the idiosyncratic volatility, so a Return Forecast is always in return units.

**Forecast Evaluation**
The out-of-sample reading of a Return Forecast before an optimiser sees it. The evaluation pairs the history of the forecast once with the Forward Target that the forecast answers for. It pairs them at the evaluation dates, which bound the sample and stride by a step. The pair goes into a lean Result (`ForecastEvaluationResult`) that carries the pair, the dates and the parameters, and computes no statistic of its own.
Every statistic is a verb over that Result. The statistics are the two information coefficients and their coverage, the two alpha portfolios and the quantile spread, the Forecast Calibration, the holding-period and decay tables, and the contemporaneous factor correlations. One columnar summary (`ForecastSummaryResult`) reads the Results and holds one entry per forecast, so its case of length 2 is the comparison of two forecasts.
The Result also carries the universe that the coverage divides by (`umsk`). This is the estimation mask of the Asset Panel, cut to the rows of the block. So on a point-in-time panel, an asset that has not listed yet is outside the universe, and the coverage does not count it as missed. A coverage of one means that the evaluation scored every admitted asset, however few. See #1070.
The evaluation writes the forecast onto that universe once, at the pairing. So every statistic reads the estimation universe, and no statistic carries a mask of its own (#1074). The contemporaneous factor correlations read every observation, and the evaluation dates are opt-in for them (#1071).
Two evaluations can answer one question on two grids, because a member that is refit warms up and one that publishes its history does not. `forecast_evaluation_align` puts the two on the grid they share. It rebuilds each Result on `max(lo):step:min(hi)` and leaves its pair as it is. The summary takes `align = true` to call it (#1073).
A **silenced date** is an evaluation date whose universe is not empty and at which the evaluation scores no asset. Its coverage is zero, so the mean coverage counts it. The summary reports the count of such dates, and beside it the coverage over the scored dates (#1512).
The summary is a comparison, and a comparison is paired: one estimand, one sample, one unit. `paired = false` gives a **listing** instead. A listing is the summary of each evaluation on its own with no check, and its rows are not a comparison of skill (#1512).
The evaluation computes in the element types of the forecast and the target, and each type must hold a fraction and `NaN`. The evaluation does not convert the pair to a floating-point type, and it does not refuse the pair by its type. So an integer pair raises an `InexactError` at the first statistic that writes one, and the caller converts it. #1331 ruled this not a defect, under the rule that numeric types come from the data.
*Avoid*: Covariance Forecast Evaluation, which calibrates a covariance forecast through a walk-forward. Also Factor Model Diagnostic, which is measured in sample on the fitted block, and alpha evaluation and backtest.

**Mean Test**
The test of the null hypothesis that a serially dependent series, such as an information coefficient series, has a mean of zero. It is an `AbstractMeanTestEstimator`, and `mean_test` answers a t-statistic, a two-sided p-value and a confidence interval of the mean.
The default `WildBootstrapMeanTest` reads a long-run variance at a fixed bandwidth, a fraction `b` of the span of the series. It takes its p-value and its interval from a Gaussian wild bootstrap. So it reads the dependence past the overlap of the forward windows, and a change of the volatility over time. `OverlapMeanTest` reads the overlap alone, and it reads the Student law. Its `law` sets the divisor and the degrees of freedom from the null law of overlapping windows. `NullMomentLaw`, the default, uses the first two moments, and `BandCountLaw` uses `n - 2L - 1`.
The kernel (`BartlettKernel`, `QuadraticSpectralKernel`), the bandwidth fraction, the span rule (`FiniteEntrySpan`, `FullGridSpan`), the divisor `n - ddof` of the long-run variance, the number of draws, the level `q`, the minimum-data rules and the generator are fields, and the generator follows the `rng` and `seed` convention of the library. The summary gives the test the window of the grid, `horizon / step`, as an exact ratio, so the history rule counts holding periods exactly.
The interval excludes zero exactly when the p-value is at most `q`. The t-statistic of the default test is not on the scale of the normal law, so the p-value or the interval is the figure to read.
The two comparisons, `covariance_forecast_compare` and `log_wealth_regret`, test their loss difference with a Mean Test at a window of one step, through `loss_difference_test`. There, `OverlapMeanTest()` is the rule of zero lags of Diebold and Mariano (#1787).
The field `alg` of `WildBootstrapMeanTest` is the algorithm that draws the bootstrap, an `AbstractBootstrapDraws`, and every algorithm draws the same law. `ObservationDraws` draws one multiplier per finite entry, and it is the default of every kernel but one. `QuadratureSumDraws` is the default of `QuadraticSpectralKernel`. It writes the kernel as a product of Gauss-Legendre features, and it draws the few Gaussian sums that the statistic of a draw reads. So the cost of a draw does not grow with the length of the series. Its node count follows from a bound at a tolerance, or from a count that the caller gives. `EigenSumDraws` draws the same sums through an exact eigen factor of the matrix of the weights, for every kernel.
The field `form` of `QuadratureSumDraws` is the rule for the weighted sum of the observed series, an `AbstractLongRunForm`. The default `FeatureLongRunForm` reads the sum through the same features as the draws, so the whole test is linear in the length of the series. `ExactLongRunForm` sums every lag at the exact weights, which is quadratic in the length.
*Avoid*: t-test, which names the independent case. Also HAC test, which names the variance estimate and not the test.

**Forward Target**
What a Return Forecast is scored against. It is the mean of a named history over the window `[t + lag, t + lag + horizon)` ahead of each observation, on the observation axis of the factor-model block. The member of the typed family (`AbstractForecastTarget`) names the history. By default the history is the idiosyncratic return, which is the component that a fitted Return Forecast forecasts. The other histories are the asset return and a numeric Panel Field. The horizon and the lag are parameters of the evaluation, not of the target.
*Avoid*: Realised Target, which is what a covariance forecast is compared with. Also the training target of a Return Forecast inside `TargetReturnForecast`, which is the same forward mean read in sample at fit time, and label and dependent variable.

**Forecast Calibration**
The out-of-sample scale of a Return Forecast. It is the slope of a weighted zero-intercept regression of the realised forward target on the forecast, pooled over every scorable pair of an evaluation. It comes with the curve of that relation over quantile bins of the forecast. A slope of `1` says that the forecast is already in the units of the target. A slope above `1` says that its magnitude is too small, and one below `1` says that it is too large. It is the one reading of a forecast that a rescaling of the forecast changes. So an optimiser that reads the forecast as a mean depends on it.
*Avoid*: Calibration Rule and Calibration Slot (§3.9), which compute the number that a slot of an uncertainty set would otherwise state. Also the calibration of a fitted factor model, which is the standard deviation of its standardised idiosyncratic returns over the fitted history, each return against the variance estimated before it. Also the calibration coefficient of a Return Forecast Estimator, which puts the predictions of a member into return units at fit time, and does not measure whether they landed there.
The calibration of a covariance forecast is a Covariance Forecast Evaluation. It asks whether the realised return has unit scale under the forecast, not whether the scale of a forecast is right.

**Covariance Forecast Evaluation**
The out-of-sample calibration of a covariance forecast. At each step of a walk-forward, the evaluation compares the forecast that the estimator makes before the step with the Realised Target of the observations in the step. The target is centred on the location that the forecast is about. The evaluation reads the comparison as three ratios whose target is one, and as two losses that only compare two forecasts. The three ratios are over every direction, per asset, and along a test portfolio. Its online form puts an Online Scheme in the place of the walk-forward. The scheme steps the estimator instead of a refit, and the evaluation reads the same numbers.
*Avoid*: a forecast evaluation, which scores a Return Forecast against a forward return. Also Forecast Calibration, which is the scale of that evaluation, and the calibration of a fitted factor model, which is measured in sample.
The mean of a ratio over the steps weights each step by a step weighting. By default the weight is the degrees of freedom of the step, which gives the mean of least variance. The other weighting gives the same weight to every step, which gives the plain mean. A scored step is a step with at least one active asset, and only a scored step enters a summary.

**Realised Target**
What the evaluation turns the observations of a step into before it compares them with a covariance forecast. The default is the realised covariance, which sums the outer products of the centred rows. The other is the horizon return, which sums the rows first and takes one outer product. The two are equal over a step of one observation. Under either, an observation that a step lacks counts in neither the target nor its count. So the evaluation compares the forecast cell by cell against the count of rows that both assets share.
*Avoid*: the forward target of a Return Forecast evaluation (`AbstractForecastTarget`), which is a return over a horizon ahead of the forecast. Also the `PortfolioTarget` of the regime-adjusted covariance, which is a direction along which a regime is measured, and Routing Target, which names where a Pipeline Context slot is delivered inside the optimisation step.

### 3.2 Covariance & Variance (Moments)

**Covariance Estimator**
Computes an asset covariance matrix and its correlation matrix. The core wrappers are `Covariance`, `GeneralCovariance`, `SimpleVariance` and `CorrelationCovariance`. The robust and specialised families are:

- **Gerber** (`Gerber0/1/2`): counts co-movements, and ignores small moves below a threshold as noise.
- **GerberIQ**: Gerber with a temporal look-back and decay, and with thresholds scaled by the volatility of each asset (`BasicGerberIQ`, `PartialGerberIQ`, `FullGerberIQ`).
- **Smyth-Broby** (`SmythBroby0/1/2`, `…Gerber*`, `…Count*`): extensions of the Gerber family that weight co-movements by magnitude, by vote counts, or by both.
- **DistanceCovariance**: covariance derived from a distance metric between assets.
- **LowerTailDependenceCovariance**: dependence in the joint lower tail, the co-movement in a crash.
- **RankCovariance**: based on rank correlation, `KendallCovariance` (τ) and `SpearmanCovariance` (ρ).
- **MutualInfoCovariance**: dependence by mutual information, over a histogram binning Algorithm.
- **RegimeAdjustedExpWeightedVariance/Covariance**: online exponentially weighted estimators, rescaled by a detected market regime state. The regime statistic divides a realised square by an estimated variance. So by default (`debias = ExactDebias()`) it divides by the bias that the inverse of the estimate puts in the moment that its method reads (ADR 0190). A HAC estimate (`hac_lags`) is a quadratic form in the returns with a banded weight matrix, so that bias reads the spectrum of the matrix. By default the recursion puts no floor on a HAC square (`hac_floor = nothing`), and the scalar estimator puts a floor only on the variance it returns. A HAC square can be negative. So on the separate correlation path, each HAC row divides by the volatility before its update (`hac_vol_before = VolatilityBeforeUpdate()`), and a plain row divides by the volatility after it.
  Under an estimated location, the deviation shares the returns of the estimate. So the bias also holds the dependence of the deviation on the estimate. `ExactDebias()` reads both, and `LawDebias()` reads the law of the estimate alone.
  On the separate correlation path, each product divides by the variance state over its weight (`cor_vol = BiasCorrectedVolatility()`). So the first products of the warm-up weigh no more than the bias tables assume. `RawVolatility()` reads the raw state, which is the rule of the oracle. The statistic scores only an asset whose block is built, and `PortfolioTarget` refuses a row whose direction has a variance at or below `min_val` (ADR 0218).
  A built block need not be positive definite in the HAC warm-up, or where the assets have different histories. `PortfolioTarget(; indefinite)` names what its statistic does with such a block. `RefuseIndefiniteWarmUp()`, the default, refuses a block with a negative eigenvalue while the gate of `MahalanobisTarget` for the same assets is closed. `ClipIndefiniteBlock()` clips the eigenvalues of its correlation at zero. `ScoreIndefiniteBlock()` scores it as it is (#1736).
  The bias factor of `MahalanobisTarget` comes from a level recursion over the directions of the estimate, and `MahalanobisTarget(; recursion)` names its rule. `MeanLevelRecursion()`, the default, takes the mean log-determinant of each level inside the exponential, which reads 0.2 % to 1.3 % low near the gate. `GaussianLevelRecursion()` also carries the covariance of the log-determinant. It is a second-order closure that reads within the noise of a Monte Carlo, at a cost of seconds for each count of assets (#1727).
- **Centring** (of an exponentially weighted moment): the location from which each deviation is taken, a field of the four exponentially weighted moment estimators. `EstimatedCentring()`, the default, takes the normalised exponentially weighted mean of the returns before each return. It divides each squared deviation by its exact factor `1 + 1 / n_eff`, and each product of a pair by `1 + c_ij`. So the moment is unbiased from the second valid return, whatever the mean. `PreCentred()` takes the returns as deviations from zero. `ZeroStartCentring()` takes the location of a recursion from zero, not divided by its weight (ADR 0190). A centring names the law of its deviations with `centring_law`, and the bias tables of a regime statistic read the tables of that law. So `ZeroStartCentring()` reads the law of `PreCentred()`.
- **ImpliedVolatility** (covariance): scales the covariance with implied volatility.
- **DenoiseCovariance / DetoneCovariance / ProcessedCovariance**: wrap another covariance estimator and apply matrix processing.
- **GeodesicShrinkageCovariance**: wraps another covariance estimator and applies Geodesic Shrinkage (§3.5) to the matrix it computes.
- **PortfolioOptimisersCovariance**: a composite estimator that bundles covariance estimation with post-processing.
- **WindowedCovariance** / **WindowedVariance**: the Windowed Estimators for covariance and variance.

### 3.3 Higher-Order Moments

**Coskewness** / **Cokurtosis**
Third- and fourth-order co-moment tensors, with `FullMoment`/`SemiMoment` and Windowed variants. High-order priors and higher-moment risk measures read them.

### 3.4 Regression (factor modelling)

**Regression Estimator**
Builds a factor model that maps factor returns to asset returns. Factor priors use it. The families are:

- **StepwiseRegression**: greedy feature selection, `ForwardSelection` or `BackwardElimination`, under a criterion (`PValue`, or one of the symbols `:aic`, `:aicc`, `:bic`, `:r2`, `:adjr2`).
- **DimensionReductionRegression**: regression on reduced factors, with the targets `PCA` and `PPCA`.
- **Regression target** models: `LinearModel`, `GeneralisedLinearModel` (GLM).

**Cross-Sectional Regression**
A regression of one observation's asset returns on the lagged Factor Exposures across the assets, one fit per observation.
Its **solve algorithm** decides what an observation with a rank-deficient design gets. It means the same thing whether the fit is closed-form or runs an external target. The answer is the answer of least norm by default, a return of zero for each dependent factor, a refusal, or the answer of the solver unchecked. A target reaches the answer of least norm by a fit on the factors that the rank test keeps, projected onto the row space of the design (#1625).
The **rank test** of an observation reads its weighted design, never the Gram matrix of the design, because the Gram matrix squares the condition number. The fit, a basis-invariant target, the diagnostics and the attribution standard errors all keep the columns that the fit keeps (#1719). A basis-invariant target is fitted on the design whose weighted Gram matrix is the identity.
A **Dropped Column** is a factor whose column the solve algorithm `DependentColumnDrop` left out of the fit of one observation. So its return is zero by the rule, not from the data. The fit marks it in `dc`, and a diagnostic or a standard error gives it no estimation error, as it gives an Unseen Member none (#1719).
A **Leverage-One Pair** is an asset whose exposures give it a direction of the design of its own at one observation, such as the only member of a level of a one-hot factor. The fit reproduces its return, so its idiosyncratic return is zero by construction and states nothing about its specific risk. The fit marks it in `h1`, a least-squares fit writes an exact zero residual there, and the idiosyncratic diagnostics leave it out (#1423).
A Factor Attribution reads its total for such a pair as exact, and the split between the systematic and the idiosyncratic part as unknown. This is the opposite of an asset in the warm-up of its variance. Under the default Unknown Entry Rule, every variance number that reads the split of a held pair is `NaN`, and the means keep their values (ADR 0113, amendment of #1579).
The variance of a Leverage-One Pair is not identified either. So under the default, a standard error of the attribution is `NaN` where its sandwich gives a marked pair a coefficient that is not zero. This holds for the systematic error of a holder, the level of the pair, and the factors that the zero-sum constraint of its family ties to that level (#1580).
The **Leverage-One Mask** `l1mask` of a Cross-Sectional Factor Prior states how the prior reads such a pair when it standardises the idiosyncratic returns into its scenarios and its idiosyncratic correlation. `MaskedLeverageOne`, the default, reads the pair as a gap. So the scenarios give the pair the average standardised return of its observation, whatever its variance, and the correlation leaves it out. `UnmaskedLeverageOne` standardises the zero residual as it is, as the oracle does. The result is zero over a positive variance, and the average over a zero variance (#1720).
*Avoid*: Regression Estimator (above), whose families fit one asset at a time over the observations.

**Descriptor**
A per-asset value computed from one or more Panel Fields at one observation: a momentum, a book-to-price ratio, an exponentially weighted volatility.
Its **look-back** is the number of trailing observations that its value reads. It is a count for a rolling or lagged Descriptor, and it is unbounded for a recursion from the first row. A fold of the prior that holds it keeps only the panel rows that the look-backs read. ADR 0193.
*Avoid*: Score (§1), which is the per-asset number a `ScoreSelector` ranks on.

**Descriptor Estimator**
A producer of one Descriptor.

**Cross-Sectional Transform**
A rescaling of a Descriptor within one observation, across the assets. Examples are a winsorisation, a standardisation against a benchmark-weighted centre, a Gaussian or percentile rank, and a tanh shrink. It reads the benchmark weights and an optional group label at the call. It keeps no fitted state, and it returns a value of the same shape.
*Avoid*: Preprocessing Estimator (§2), which transforms a series along time and fits a Result.

**Factor Exposure**
The loading of one asset on one factor at one observation, built from Descriptors or from a one-hot Panel Field. The matrix of these loadings is the loadings matrix that a Regression carries, and a Factor Exposure Constraint (§4.4) reads it there.
*Avoid*: net and gross exposure (§4.4), which are statements about the weights rather than about a factor model.

**Exposure Estimator**
A producer of a Factor Exposure matrix.
*Avoid*: Asset Panel Estimator (§2), which produces the panel that the distance and clustering stack reads.

**Factor Family**
A set of factors carrying one benchmark-weighted zero-sum constraint, so exactly one member of the set is redundant.
*Avoid*: the bare word "family", which names a group of Estimator types everywhere else in this glossary.

**Neutralisation**
The replacement of one Factor Exposure, or one Descriptor score, by its residual after a weighted regression across the assets on the exposures of other factors. The residual is standardised again, so it carries none of the tilt of the targets. Its owner states the weights. These are the benchmark weights for a Factor Exposure, and the estimation mask or the regression weights of the block for a Descriptor score.
*Avoid*: orthogonalisation, which is the same operation under another name. Also Detoning (§3.5), which removes principal components from a correlation matrix.

**Factor Family Basis**
The time-varying change of basis that imposes the zero-sum constraint of each Factor Family. It drops one member and rewrites the others. It is stored as the per-observation ratios of the kept members to the dropped one.
*Avoid*: the reduced loadings, which are what the basis produces rather than the basis itself.

**Empty Factor**
An estimated factor of a Cross-Sectional Factor Prior whose exposure is zero at every pair of positive regression weight in one fit, so the data state nothing about its return. A sub-universe with no asset in one level of a one-hot factor leaves that level empty, and a meta-optimiser gives its sub-problems such sub-universes. The fit keeps the factor on every axis, with a return of zero at every observation, a mean of zero, and a zero row and column of the factor covariance.
*Avoid*: a dropped factor, which is the member of a Factor Family that the Factor Family Basis rewrites through the others.

**Unseen Member**
A member of a constrained Factor Family that no asset of positive regression weight loads on at one observation of a Cross-Sectional Factor Prior. The exposures lag the returns, so the zero-sum condition of the observation reads earlier benchmark weights, which can still weight the member after its last asset delists. The data state nothing about its return there. The Unseen Member rule of the prior states the return. By default it is a return of zero at that observation, with the zero-sum condition held over the other members. So the observation is identified, and its factor returns do not depend on the dropped member (#1606).
The Cross-Sectional Factor Model block stores the rule, so the regression diagnostics and the standard errors of a factor attribution read the design that the fit regressed on, and they do not depend on the dropped member either (#1609).
*Avoid*: an Empty Factor, which no asset loads on at any observation of the fit.

**Currency Excess Index**
The value in the base currency of a deposit in one currency, funded in the base currency. It is the exchange rate times the cash total-return index of the currency, divided by the cash total-return index of the base currency. It is the level series whose return is the Currency Excess Return, and `currency_excess_index` builds it. See ADR 0184.
*Avoid*: an exchange rate, which leaves out the two cash returns.

**Currency Excess Return**
The return, in the base currency of the investor, of the cash of one currency. It is the return of its Currency Excess Index, in simple returns (1 + exchange rate return)(1 + cash rate) / (1 + base cash rate) - 1. In log returns, the base-currency excess return of an asset is exactly its local excess return plus the Currency Excess Return of its currency. In simple returns, the sum leaves out the cross term, the local return times the exchange rate return. That term depends on the asset, so no per-currency series can hold it. See ADR 0184.
*Avoid*: a currency conversion. The library converts no asset return, and the Currency Excess Return is data that the caller supplies or builds from exchange rates and cash rates.

**Observed Factor**
A factor of a Cross-Sectional Factor Prior whose return the caller observes and the fit does not estimate. Its member, an `AbstractObservedExposureEstimator`, gives the exposures and names the column of the Exogenous Series that holds each return. The regression runs on the returns net of the Observed Factors. The Observed Factors then follow the estimated factors on every axis of the factor model. No Factor Family Basis or Neutralisation touches them, and attribution reports them as direct returns with no standard error. A Factor Family holds estimated factors or Observed Factors, never both. See ADR 0184.
*Avoid*: a time-series factor, which a `FactorPrior` reads whole from the factor returns and regresses each asset on.

**Currency Factor**
The Observed Factor of one currency. Its return is the Currency Excess Return of the currency, read from the Exogenous Series by the name of the currency. Its exposure is one-hot on the assets that hold that currency. `CurrencyExposure` gives one per currency level. See ADR 0184.
*Avoid*: a reserved family label. The family of the Currency Factors is a property of their member, `currency` by default.

### 3.5 Matrix Processing

**Denoising**
Removal of statistical noise from a covariance or correlation matrix by spectral thresholding (Random Matrix Theory). The algorithms are `SpectralDenoise`, `FixedDenoise` and `ShrunkDenoise`.

**Detoning**
Removal of the largest principal components (market modes) from a correlation matrix. What remains is the asset-specific correlation.

**Geodesic Shrinkage**
Movement of a covariance matrix towards a target along the shortest path between two positive definite matrices under the affine-invariant metric. The targets are `IdentityTarget`, `ScaledIdentityTarget`, `CommonCovarianceTarget`, `ConstantCorrelationTarget`, `DiagonalTarget`, or a fixed positive definite matrix. It reads the matrix alone, so an incremental fit runs it.
*Avoid*: linear shrinkage, which averages the matrix and the target entry by entry.

**Posdef**
Projection of a matrix to the nearest positive definite matrix.
Its algorithm owns the repair, and the test that leaves a matrix unchanged. `NearestCorrelationMatrix.Newton`, the default, finds the nearest correlation matrix. `ClippedNearestCorrelation` clips the correlation eigenvalues at a floor, tries once more at ten times the floor, and keeps the variances.

**Matrix Processing**
The composing estimator that applies a sequence of post-processing steps to a covariance or correlation matrix. The steps are Posdef, Denoising, Detoning, and a custom algorithm.

**Matrix Square Root**
A matrix `L` with `L L' = Σ`. A factor prior states its covariance through it as `chol`, and a second-order cone reads it. `nothing` takes the plain Cholesky factor and refuses a matrix that is not positive definite. `RidgeCholeskySquareRoot` adds a ridge to the diagonal that grows tenfold until the factor exists, and `EigenFallbackSquareRoot` takes the square root of the eigendecomposition of a positive semidefinite matrix. Every estimator that takes a square root of a covariance holds the algorithm in `mtx_sqrt`: the two factor priors, `FactorBlackLittermanPrior`, `Variance`, `StandardDeviation`, `DistributionValueatRisk`, `UncertaintySetVariance`, `ArithmeticReturn`, `RelaxedRiskBudgeting`, `Kurtosis`, `NegativeSkewness`, `NormBallUncertaintySetAlgorithm` and `OrthogonalUncertaintySet`, and each defaults to `EigenFallbackSquareRoot`.
*Avoid*: Cholesky factor, for a square root that is not triangular or that carries a ridge.

### 3.6 Prior

**Prior**
The full statistical summary of asset returns that feeds an optimiser. A Prior Estimator bundles Moment Estimators and can adjust them with the assumptions of the user. The name comes from "prior statistics", not from Bayesian inference.

The priors differ on two structural axes. A **low-order** prior holds the mean and the covariance, and a **high-order** prior adds the coskewness and the cokurtosis. The **data dependency** suffixes `_A` / `_F` / `_AF` say whether the prior needs factor returns. The concrete estimators are:

- **EmpiricalPrior**: moments computed directly from returns.
- **FactorPrior** / **HighOrderFactorPriorEstimator**: moments reconstructed through a factor model.
- **CrossSectionalFactorPrior**: moments reconstructed through a factor model whose Factor Exposures are built from an Asset Panel and fitted by a Cross-Sectional Regression.
- **HighOrderPriorEstimator**: empirical high-order prior.
- **Black-Litterman family**: blends market-equilibrium priors with investor views. The members are `BlackLittermanPrior`, `BayesianBlackLittermanPrior`, `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior`, and `BlackLittermanViews` is the container of the views.
- **EntropyPoolingPrior**: re-weights scenario probabilities to satisfy views with minimal relative entropy.
- **MeucciEntropyPoolingPrior**: the earlier entropy pooling estimator, whose CVaR route root-finds the Value at Risk level rather than writing a formulation.
- **OpinionPoolingPrior**: consensus of several priors, by `LinearOpinionPooling` or `LogarithmicOpinionPooling`.
- **SyntheticDataPrior**: moments of a panel drawn from a fitted Synthetic-Data Generator, with or without a Stress Statement. The vine copula is the first generator.

**Synthetic-Data Generator**
A model of the joint distribution of asset returns. A fit makes it from a panel, and a draw takes scenarios from it. Its estimator holds every parameter of the fit, including any conditioning that the model can honour. Its Result is the fitted model. A draw of `n` scenarios from it under a stated `rng` returns the observations × assets matrix and a **Simulation Result** (below). The vine copula is the first generator. A bootstrap or a path-simulating model joins the family with its own estimator, and the prior that wraps it does not change. A conditioning that the fitted model can honour is a field of the Result of the fit. So to condition a fitted generator again is to change a field of the Result, never to fit it again. ADR 0153.
*Avoid*: reading the generator as a Prior. It makes scenarios, and a **Synthetic-Data Prior** (below) turns them into moments.

**Synthetic-Data Prior**
The Prior Estimator that wraps a Synthetic-Data Generator. It fits the generator when it receives the estimator, or it takes a fitted Result as it is. It then draws `n_sim` scenarios under its own `rng`/`seed`, repairs any constant column, and fits an inner low-order Prior Estimator on the synthetic panel. It returns that Result whole, with the Simulation Result of the draw. Its `X` is the synthetic panel and its Original Returns Matrix is `nothing`, because the panel of the caller is not over the same observations. **Fit once, stress many** is the workflow that fits the generator once, changes the conditioning on its Result, and wraps it again. ADR 0153.
*Avoid*: reading `ens` as the count of observations that the generator was fitted over. It is the count that the inner estimator stated over the synthetic rows. The Result of the generator holds the fitted count.

**Simulation Result**
The record of how a draw ran. It holds the route that the sampler took (exact, rejection or importance), the acceptance rate, the draws that it took, and the mask of the columns that the constant-column repair changed. The draw returns it with the matrix, and the Prior Result holds it under `sim`. A reweighting wrapper dispatches on it to fit the inner moments again on the drawn panel, and it does not draw again. ADR 0153.
*Avoid*: reading it as importance weights. A weighted draw states its weights in `w` of the Prior Result and their Kish count in `ens`. The default importance route resamples, so both stay `nothing`.

**Group Pair View**
An entropy pooling correlation or covariance view whose two sides are groups and not single assets, `"(gA, gB) == 0.35"`. The two groups must have equal length. The view **spans** one asset pair per position. It emits one constraint row for each pair, and a `prior(gA, gB)` reference inside it resolves to the prior value of that pair. See ADR 0079.
*Avoid*: reading it as a statement about one summary of the correlation block. The library defines no such aggregate.

**Departed View**
A view row in which one of the written-out names has left the **Investable Mask**. The fit drops the whole row, and it does not fit the row without that leg. The drop never refuses, under either setting of `strict`. The name was correct over the universe that the caller received, and no caller can know in advance which asset a prior will fail to estimate. The drop is not silent either. The site writes what it dropped into the **Departure Ledger** (below), and the entry of the fit reports it once. The dropped row joins `excl`, so a confidence for each view leaves with it.
A **group** named in a row is a different case. The group sheds its departed members before its coefficient is spread, so the Black-Litterman mean divides by the count of the members that remain, and its row stays. A group is a description that the data resolves, but a written-out name is the caller pointing at one asset. A departure that empties the whole view set fits with no views, and the one report at the entry of the fit is then a warning. See ADR 0125.
*Avoid*: reading it as a refusal. A name on neither the asset universe nor the **Non-Investable Axis** (§3.6) is a typo, and a typo still raises an error under `strict`. The departed name is the one case that never raises.

**Departure Ledger**
The list of what the departures of a fit cost, which the entry of the fit keeps. The sites that drop fill it, and the entry that derived the mask reads it once. The entry of a fit is the step that derives the Investable Mask and reduces the problem to it, `port_opt_view` for an optimisation. A **Departed View** (above) records the row that it lost, and a shed group records how many members left. Each records it in the words of the entry, not of the site, so the report reads as one sentence about one event. It is `nothing` wherever no entry collects, for example when a caller assembles constraints outside a fit. That sentinel keeps the ordinary path free of the ledger. The entry names the *process* that it reports for, an optimisation or an entropy pooling fit. More than one kind of entry makes a **Non-Investable Axis**, and each kind has its own consequence to state. See ADR 0125.
*Avoid*: reading it as a diagnostic channel. A departure is not an error. The ledger is the record behind one `@info`, and `strict_diagnostic` still reports a typo where it happens.
*Avoid*: expecting one message for each fit overall. A prior fitted inside an optimisation reports its view rows, and the entry of the optimisation then reports its constraints, because neither layer can report the events of the other.

**Tail View**
A view on a quantile risk measure of the posterior, CVaR, EVaR or RLVaR, and not a view on a moment. It is the one view family that is not a linear function of the posterior probabilities, and the one family that `ep_tail_views!` lowers. A VaR view is not a tail view. It constrains the tail mass, which is linear in the posterior probabilities, and `ep_var_views!` lowers it separately.
A tail view names one or more assets, each with a coefficient. Each of the three measures is concave in the posterior probabilities. So a view whose coefficients have one sign is a **positive combination**, and its lower bound is a convex set. The dual formulations write it exactly, with one block for each asset. A view whose coefficients have both signs is a **relative view**, and its feasible set is not convex. It takes the integer formulation or a **Surrogate Row**. See ADR 0069.
*Avoid*: confusing it with a **View** (§1), which is the index-selection mechanism.
*Avoid*: calling a group view relative. A group of two or more members is a positive combination, and it needs no integer variable.

**Surrogate Row**
The linear upper bound that a sequential formulation of a tail view writes for each measure on the wrong side of the inequality. The formulation reads it from the primal representation, with the multipliers fixed at the last posterior. It is the value at risk for CVaR, the primal pair for RLVaR, and the tangent of the primal with a fixed dual variable for EVaR. It is sufficient for the view and tight at the point where it was read. `entropy_pooling` reads it again and solves again until it is tight at the answer. See ADR 0069.
*Avoid*: reading its slack as a violated view. The view holds on every posterior of the sequence, and the slack says only that the divergence can still fall.

**Search Bracket**
The span that a scalar search of a **Tail View** runs over. It is stated in the units that the search works in, and the name says which units. `zlo_frac` is a fraction of a proven upper end, and `log_zlo` and `log_zhi` are additive offsets on the logarithm of the loss range. A parameter whose default follows from the data is a field of the view estimator with the default `nothing`. A group of parameters whose defaults are plain numbers is an algorithm type of its own. See ADR 0069.
*Avoid*: reading one family's number under another's rule. A fraction written where an offset is read passes every guard and searches the wrong span.

**Risk-Free Shift**
The one place where a Black-Litterman prior reads its `rf` field. The update blends the prior mean with the view returns, so it runs on the total-return scale that the view returns use. A prior mean from a wrapped estimator is already on that scale, and the shift does not change it. An equilibrium mean from `EquilibriumExpectedReturns` is a bare risk premium, and the rate converts it *before* the update. A member with no equilibrium branch has nothing to convert, so it adds the rate to the posterior asset mean instead.
*Avoid*: calling it a round trip. Nothing subtracts the rate, and the update is affine and not a translation, so a conversion and its inverse around the update would not cancel.

**Factor Lift**
The step from the factor axis to the asset axis. It fits the loadings, rebuilds the returns through them, projects the factor moments, and can add a residual block.
*Avoid*: calling it a **Forward**, which passes a wrapped Result along its own axis and does not cross axes.

**Original Returns Matrix**
The returns matrix that the caller supplied, as distinct from the one that a Prior Result holds. The two differ only where a factor prior overwrites the returns with its reconstruction.

**Coverage Universe**
The assets that a Prior Estimator gives to its plain moment estimators for one fit. An asset is in it when its return is finite and its Asset Panel active mask is `true` at every row of the window. The prior reduces its returns matrix to it before the fit. It then expands every block of its result back to the full universe, with `NaN` outside the Coverage Universe. The prior does not read the estimation mask. With no panel or no masks, the rule is finiteness alone. It is `nothing` when every asset is in it, and that sentinel skips the slice and the expansion.
A mask-aware moment estimator reads the masks of the panel itself, so it takes the whole window instead and makes its own frame. An estimator with a **Coverage Policy** (below) does the same. Three more entry points reduce to it. They are an Optimisation Estimator that fits no prior, the pre-selection step, so that a selector ranks only live assets, and the random asset subset of a cross-validation scheme, which the scheme draws from the Coverage Universe of its own window. See ADR 0117 and ADR 0120.
A plain moment estimator reads a cell that the active mask turns off as a non-finite return. Each of its three forms does this. The Asset Panel method leaves the asset out. The matrix method with an `active_mask` keyword refuses the sample with `IsNonFiniteError`. The fold writes `NaN` in that cell, so the asset gives `NaN` from then on. A matrix call of a prior gives the masks as keywords, and the Asset Panel method reads an `active_mask` keyword in place of the mask of the panel. So the two calls reach the same Coverage Universe. See ADR 0117, amended for #1648.
A per-asset series beside the returns acts as `X` at the same cell. A gap in `iv` narrows the Coverage Universe of every fit whose estimator tree holds an `ImpliedVolatility`. The narrowing goes through the root of the moment verb, not only through the panel method of that estimator. In the Excess Frame, a gap in a per-asset benchmark is a `NaN` excess return. A gap in a **Shared Series** (below) does not narrow it. The fit drops the observation instead. See ADR 0117, amended for #1672.
*Avoid*: the **Investable Mask** (below). The library derives it from the result after the fit, and it can be narrower, because a mask-aware estimator gives `NaN` during its warm-up. Also avoid the active mask itself, which is per observation, but the Coverage Universe is per fit.
*Avoid*: reading the narrowing of one estimator as a different Coverage Universe. An estimator that reads a second input narrows further inside its own fit. For example, `ImpliedVolatility` drops a column whose implied volatilities are absent over the window. Reduce-and-expand then writes the `NaN` frame of that column, as it does for an absent return. See ADR 0135.

**Shared Series**
A series that every asset reads at the same observation: a benchmark of one column in the Excess Frame, and each factor return of a fit that reads the factor returns. A gap in it at one observation leaves no asset with a value there. So the fit drops that observation from every field of the returns data, and it keeps every asset. A gap in the series of one asset removes that asset from the **Coverage Universe** (above) instead. The drop happens once, at the outermost entry that holds the data, so every reader that pairs the observations of a prior with the returns data sees the same observations. An online step keeps the row and marks it, because the buffer of a prior and the Fold Context each hold one row for every observation. Every fit through the buffer then drops the marked row. `MissingDataFilter` counts such a gap as an observation that misses every asset. See ADR 0117, amended for #1672.
A Prior Result records the observations that the drop removed in `ss_gaps`, with one mark for each observation of the returns data that it read. So a reader that holds that data pairs the rows of `X` with the correct observations. A reader that holds the data after the same drop has fewer observations than the marks, and it pairs the last rows. An optimiser maps the prior that it fits after its own drop back to the observations of its returns data. #1744, ADR 0207.
The group of a head decides the drop of a benchmark gap. The head drops it when any reader of the group states `ExcessSource()`, and a raw reader of the group then loses the observation too. A `Pipeline` applies the same rule to each step whose value its optimiser receives, in batch and online. The rule decides which observations a reader reads. It never decides the frame of the reader, which its own `x_src` states. #1774, ADR 0215.
*Avoid*: a per-asset benchmark, which holds one series for each asset and acts as `X`. Also avoid dropping the observation from a fit that does not read the series. That drop loses data that the fit had.

**Coverage Policy**
The opt-in that a moment estimator holds in its `cvg` field. It replaces the **Coverage Universe** (above) by available-case estimation. It is `nothing` by default, and dispatch reads that arm as the reduce-and-expand path, so an estimator that holds no policy costs nothing. With a policy set, the estimator fits every cell of a moment on the observations at which every asset of that cell is finite and active. Each cell has its own denominator. An asset reaches the answer where the `min_coverage` floor of the policy and its coverage algorithm admit it.
The algorithm is a dispatchable family, with one member for each rule for a delisted asset. One member keeps the history of the asset and drops it from the frame. One resets it, so that a relisting starts cold. One holds it for a stated staleness. The policy lets a plain estimator hold an asset that lists after the first row of the window. It reaches every order. The third and fourth orders run their spectral step on the admitted block alone, and they refuse an uncovered cell of the block by name. See ADR 0117.
*Avoid*: reading it as a rolling window. It never forgets an observation. The only thing that it changes is which observations each cell had. Also avoid reading `min_coverage` as a count. It is a share of the folded observations. Also avoid reading it as an admission rule alone. It also states what a **Scenario Fill** (below) passes in silence, because a filled share is a coverage share read from the other end.

**Undetermined Pair**
Two assets that a **Coverage Policy** (above) admits, each with a variance, but that share too few observations for a covariance. Their cell would be `NaN` inside the finite block. The pairs make a graph, and a set of assets removes every pair exactly when it is a vertex cover of that graph. The `peel` rule of the policy chooses the cover at admission. `MinimalPeel`, the default, removes the fewest assets. `GreedyPeel` removes the asset with the most pairs first. `NoPeel` removes none, so the matrix repair refuses the `NaN`. A peeled asset is `NaN` across its row and column, and it leaves the **Investable Mask** (below). The peel warns and names the assets, and it refuses under `strict`. See ADR 0117.
*Avoid*: peeling in the matrix repair, which would drop the asset from the covariance and keep it in the universe that every other consumer reads. Also avoid reading the peel as silent. It is a reported loss of data.

**Investable Mask**
The assets that an optimisation can trade. The library derives it from a Prior Result and stores it nowhere. It is `true` where `mu` and the diagonal of `sigma` are finite, and, where the Prior Result holds higher co-moments, where their per-asset diagonals are finite too. A Prior Estimator fits on the **Coverage Universe** (above) and returns a result on the *full* asset universe, in which an asset that it could not estimate is `NaN`.
Every Optimisation Estimator that fits a prior derives the mask once at its entry, before it clusters, samples or solves. It reduces the prior and every constraint that the caller stated to the mask, runs whole on the investable universe, and expands the solved weights into a zero vector of the full length. Its result holds the mask with the reduced objects that it ran on, and the mask tells a zero weight from a refusal. A head that fits no prior derives the mask from the **Coverage Universe**, so every optimisation result holds one. It is `nothing` when every asset is investable, which skips both halves. See ADR 0115 and ADR 0120.
A reduced `Fees` also records in `imsk` the mask that it was reduced on. Its two liquidation charges, `lq` and `flq`, apply on the complement of the mask, but a caller states them over the full universe, and `investable_fees_view` cannot tell the two apart by width. So `investable_fees_view` reduces an unmarked fee and returns a marked one unchanged. See ADR 0121 and #1067.
A result therefore holds three things on two universes. `w` is on the universe of the caller, and `pr` and `fees` are on the universe that the fit solved. A consumer that takes a result pairs them on the investable universe through one verb, `result_investable_view`. It takes the two objects of the result as they are. It views at the mask a value or a fee that the caller states over the full universe. A drawn figure of a result draws only that universe, with each bar under its own name. See ADR 0118 and #884.
*Avoid*: confusing it with an **Asset Panel**'s active mask, which is stored and is a point-in-time fact. The active mask says that an asset is listed and tradable. The Investable Mask says that the estimator made a finite moment for it. The two differ where an asset is active and not yet warmed up, and where its regression fails.

**Scenario Fill**
The zero that a prior writes into the missing rows of an investable column, once, in the pass of the estimator itself, so that every consumer of a Prior Result reads a finite investable column. It exists because a **Coverage Policy** (above) admits an asset from the observations that it has, but the returns column of that asset keeps a gap at every row that it missed. It is the counterpart in the fit of a **Held Gap** (§4.6), and it writes the same value for the same reason. The asset contributed nothing on that observation, so its slice was in cash.
A non-investable asset keeps its whole `NaN` column, so the **Investable Mask** (above) does not change, and `mu` and `sigma` do not change. The cost is exact. A measure at level `alpha` over an admitted column of coverage `c` reads the `alpha / c` level of the observed sample, because the zeros do not enter the tail but increase the denominator. The fill is silent within the coverage floor that the caller already stated. Where the caller stated no floor, the library names every fill. See ADR 0118.
An estimator that folds and carries names each asset **once**, not once for each call with no data, because its **Sample Buffer** (§1) holds the set of assets that it already named. A batch fit remembers nothing and names what it finds. `strict` reads no memory. It refuses any fill at any call with no data. See ADR 0136.
*Avoid*: reading it as imputation. The glossary keeps that word for a **Held Price** (§2) that a caller states. The fill invents no return. It records an absence as no contribution.
*Avoid*: reading its notice as a notice about the tail alone. The same invented cells reach the correlation that a dendrogram computes again from the matrix when its graph reader reads the prior. They also reach the column that an entropy-pooling view is calibrated against, and the outer returns matrix that a meta-optimiser builds.
*Avoid*: expecting the Prior Result to record which cells it wrote. They are the non-finite entries of the matrix of the caller, intersected with the **Investable Mask**, and every entry point that could act on them holds that matrix.
*Avoid*: reading its share as a fraction of the matrix. It is the filled count of the worst investable column over the observations. A denominator over the whole matrix scales with the universe and hides the one column that the notice is there to catch.

**Non-Investable Axis**
The names that the **Investable Mask** (above) left out. A **Universe Sets** declares them under `nikey`, in the order of the complement. The axis exists so that a name-keyed constraint still resolves after a reduction. A bound, rate or threshold stated for an asset that has since left resolves on this axis, and the fit drops it and does not refuse it. The entry of the fit reports the departure once, with its cost. It is *made by the entry of a fit, never written inside an optimisation*. The entry of a fit writes it after it views the estimator, and the view drops it. So a Universe Sets that holds one was reduced by exactly one entry for exactly one problem, and a cluster of a nested optimisation cannot inherit the departures of its parent. A caller can declare one by hand outside a fit, which is how a forced-liquidation rate resolves with no optimisation around it.
It is bare. It has no prefixed partition and no unique-entry twin, because its entries are unique by construction and a plain group is already axis-blind. See ADR 0124.
*Avoid*: reading it as one rule for every constraint shape. The fit skips a **value** keyed by a departed name, but it drops whole a **row** that names one. The **Departed View** (above) and ADR 0125 own that half.
*Avoid*: reading it as a record of events. It is a universe, not an event log. It names the assets that are absent from this problem, and it says nothing about when or why they left. The library derives it from the **Investable Mask**, and a fee charges the **Forced Liquidation** over it.

### 3.7 Distance

**Distance Matrix**
A symmetric, zero-diagonal matrix in which larger values mean less relatedness. The fundamental input to Phylogeny.

**Distance Estimator**
Converts correlation or returns into a distance: `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, `CorrelationDistance`, `VariationInfoDistance`, `CanonicalDistance`. `Distance` is the configurable container and `DistanceDistance` computes a distance-of-distances.

**Feature Distance**
The one Distance Estimator that measures something other than returns. It applies a metric to the rows of a Feature Matrix, which it stacks from the Panel Fields that its Feature Selector names, so the hierarchy that results shows an exogenous structure. A clustering or phylogeny result records nothing about that matrix. `feature_labels(de, pr, rd, X)` derives what it measured from the estimator, the prior result and the returns data, and the kernel derives the matrix the same way.
It reads each asset, or each pair of assets, only at its own readable rows of the Asset Panel, as a Coverage Policy fits a covariance cell. A readable row is a row where the asset is active and no value column of the selector holds a Placeholder. The `LastObservation` family reads the last row under `LastRow`, and the last readable row of each asset under `LastActiveRow`. The two aggregates restrict their weights to the readable rows. `StackObservations` stacks the rows that a pair shares and rescales the distance to the whole window. See ADR 0191.
An asset that it cannot read departs at the entry of a fit as a non-investable asset, and a direct call refuses it. A pair with no shared active row takes the `pair` rule of its collapse. `RefusePair` refuses. `DropFewerRows` drops the asset with fewer active rows at the entry of a fit. `FeatureFallback` measures the pair by the features of each asset, collapsed over its own active rows.
Its checks read the same cells as its collapse. A cell must be finite and in the domain of the metric only where the asset is readable. So a `NaN` or a negative value at an unreadable cell refuses nothing and changes no distance (#1734).
*Avoid*: reading an inactive cell or a Placeholder as a value that the distance measures. An inactive cell holds a fill value, a zero or a stale value, and a Placeholder holds no data (ADR 0102), so no collapse reads either. Also avoid skipping a cell that a fill policy wrote. It holds data (#1631).

**Similarity Matrix Algorithm**
The transform that turns a Distance Matrix into a similarity matrix: `MaximumDistanceSimilarity`, `ExponentialSimilarity`, `GeneralExponentialSimilarity`, `ComplementSimilarity`, `AngularSimilarity`.

**Non-Negative Similarity Matrix Algorithm**
The narrower similarity family that the PMFG path admits. Its members cannot be negative over their own admissible distances.

### 3.8 Phylogeny

**Phylogeny**
The characterisation of asset relationships derived from a Distance Matrix. The biological metaphor is intentional. Related assets cluster like species on an evolutionary tree. It has three sub-concepts:

- **Clustering**: groups related assets into a dendrogram. The estimator is `ClustersEstimator` with `HClustAlgorithm`, `DBHT` (PMFG-based, with the `LoGo` sparse inverse) or `KMeansAlgorithm`, and `OptimalNumberClusters` finds the cluster count.
- **Network**: assets as graph nodes and edges. `NetworkEstimator` builds a minimum spanning tree with `KruskalTree`, `BoruvkaTree` or `PrimTree`.
- **Centrality**: measures which assets are the most influential. The algorithms are `BetweennessCentrality`, `ClosenessCentrality`, `DegreeCentrality`, `EigenvectorCentrality`, `KatzCentrality`, `Pagerank`, `RadialityCentrality` and `StressCentrality`.

`PhylogenyResult` holds the resulting matrix or vector.

**Phylogeny Features**
The reverse direction. A phylogeny is reused as a Panel Field, not consumed as one, because an assets × assets proximity matrix is a tensor field whose labels are the assets. `PhylogenyPanel` is the Asset Panel Estimator that builds it.

**Separation**
How far apart two assets are in a Network, and how far is too far. It has two members. `HopCount` is the number of edges on the shortest path, and `PathLength` is the summed distance along it.

**Separation Budget Rule**
A callable in the slot of the budget number of a Separation. It computes the number from the data, and does not state it in advance. The members are `HopCountQuantile` and `PathLengthQuantile`.

**Separation Decay**
The rule that turns a Separation into a score: `LinearDecay`, `ExponentialDecay`, `ReciprocalDecay`, `NoDecay`.
*Avoid*: confusing it with the `decay` of the exponentially-weighted moment estimators, which is a smoothing constant over observations.

**Polarity**
Which of two opposite quantities the edge weights of a Centrality algorithm must be: `DistancePolarity` (small is close) or `SimilarityPolarity` (large is close). It is a fact about the algorithm's mathematics, not about the graph.

### 3.9 Uncertainty Sets

**Uncertainty Set**
A neighbourhood of one quantity, a mean vector or a covariance matrix, inside which a robust optimiser protects against the worst case. The shapes are **Box** (per-parameter bounds), **Ellipsoidal** (a joint confidence region), the covariance-only **Compact** (below), **Norm Ball** (below, a geometry map of any rank under a norm order, on either axis), and the mean-only **ℓ1** and **Signed ℓ1** cross-polytopes. The constructors are `DeltaUncertaintySet`, `NormalUncertaintySet`, `ARCHUncertaintySet` and `CharacteristicUncertaintySet`.
The chi-squared radius of a joint region at level `1 − q` reads the dimension of the set, the number of free coordinates that its error spans: `N` on the mean axis, `N(N + 1) / 2` on a full covariance shape, because a symmetric matrix has that many free entries, and `N²` on a diagonal covariance shape (ADR 0188).

**Compact Covariance Uncertainty Set**
The covariance shape stated as a radius, a diagonal metric square root and a basis of the directions the worst case spares, rather than as a shape matrix on the vectorised covariance. Its worst-case variance is a quadratic penalty on the weights, so the consumer adds one cone and one free coefficient vector, and not the lifted semidefinite block that the Ellipsoidal shape needs. `CompactCovarianceUncertaintySet` builds it.
Its radius is a size that the caller states, or a size that a rule of the compact-radius family computes from the sample and the span. The rules are `ResidualInflation`, a chi-squared bound on the idiosyncratic variances, and `VarianceFraction`, a stated fraction of the nominal variance at a reference portfolio. The radius has a family of its own because its units change with the Orthogonality Metric (below) of the same estimator, which no Calibration Rule can read (ADR 0127).
`ResidualInflation` reads the sampling law of each idiosyncratic variance from the loadings block. The fit that measured the variance records its degrees of freedom and its divisor with it, for each asset. The rule refuses a block that records none, unless the rule states its own count (ADR 0127, amended).
*Avoid*: Ellipsoidal (above), whose radius is a quantile of a dimension. The radius of this shape is a penalty multiplier, dimensionless under the default metric and in variance units under the identity.

**Norm-Ball Uncertainty Set**
A shape stated as a radius, a geometry map and a norm order. The set is the image of the normalised ball of that order under the map, centred on the quantity that it bounds. Its worst case is the radius times the dual norm of the transpose of the map, applied to the exposure. The map can have fewer columns than the quantity has entries, so the shape can be flat, and a set confined to an Orthogonal Subspace needs that. It applies to both axes. The Ellipsoidal, Box and ℓ1 shapes are its order-2, order-∞ and order-1 cases with a square or diagonal map.
`NormBallUncertaintySet` builds it, and `NormBallUncertaintySetAlgorithm` selects it on the Normal and bootstrap estimators. The Normal estimator factors the asymptotic covariance that it already builds, so its set is the Ellipsoidal set in another form. The bootstrap estimator stores its own deviations as the map, so its set holds the sample second moment exactly, and it needs neither a square matrix of side N² nor a positive definite repair.
*Avoid*: Ellipsoidal (above) for a flat set, because an Ellipsoidal shape matrix must be full rank.

**Orthogonal Subspace**
The orthogonal complement of the column space of the loading matrix, taken under an Orthogonality Metric (below). An Uncertainty Set confined to it prices no error in a direction the factor model already explains.

**Orthogonality Metric**
The cross-sectional weighting under which the library takes the span of the loading matrix and its Orthogonal Subspace (above). It is the benchmark weights, the regression weights, the inverse idiosyncratic variances, or the identity, each read from the factor block of the Prior Result at its latest observation. A caller states it as a marker, and the default is the inverse idiosyncratic variance.
*Avoid*: the weights of a Cross-Sectional Regression (§3.4), which the fit of the prior computes for each observation. A metric only selects a quantity that the fit already stored.

**Orthogonal Scaling**
The shape that a mean Uncertainty Set takes inside the Orthogonal Subspace (above). It is the identity, which gives every orthogonal direction the same uncertainty, or the idiosyncratic covariance projected onto the subspace, which gives a noisy direction more uncertainty than a quiet one. A caller states it as a marker, and the default is the identity. It changes the shape of the set and not the subspace that holds the set, so the radius reads the same rank either way.
*Avoid*: Orthogonality Metric (above), which fixes the subspace that holds the set. A scaling fixes only the shape of the set inside that subspace.

**ucs Triple**
The three ways to ask an Uncertainty Set estimator for its sets. `ucs` gives the mean and covariance sets as a pair, `mu_ucs` gives the mean half, and `sigma_ucs` gives the covariance half. Most estimators fit from returns data through a Prior Estimator of their own. An estimator can instead read the Prior Result of the optimisation itself, and then it answers the same three verbs from that result. An estimator confined to an Orthogonal Subspace must do so. Any of the four returns-data estimators does so when it holds no Prior Estimator of its own. The optimiser gives it the prior where it builds the constraint. Such a set is centred on the quantity that the optimiser solves on, and it folds with that quantity under the online step, with no state of its own. A set with its own Prior Estimator is fitted again over the observations seen so far at every call with no data. ADR 0138.
*Avoid*: asking a set with no Prior Estimator of its own for a fit from returns data. It has no prior to fit, and it refuses by name.

**Characteristic Vector**
The per-asset quantity an ℓ1 uncertainty set is built around, usually the expected return, entering the objective as `mu'w`.
*Avoid*: Feature (§2), which is a coordinate of an asset in a matrix whose rows become a distance.

**Radius Calibration**
The conversion from "how many assets should I hold?" to the radius that produces it. It is a calibration, not a constraint, so a further constraint can change the realised count.

**Activation Ladder**
The non-decreasing sequence of radii at which each successive asset of a ranking joins the ℓ1 solution, one rung per asset. A Radius Calibration reads it as a bracket. A radius strictly between the `q`-th rung and the next activates exactly `q` assets. The long-only ladder ranks the characteristic alone. The paired ladder adds the ladder of the reversed, negated ranking, so one rung activates a long/short pair.

**Ambiguity Set**
A neighbourhood of a whole distribution, not of one of its moments: Wasserstein (the data moves), Gelbrich (the moments are wrong), or divergence (the probabilities are wrong). It is a **reading of machinery that already exists**, not an object. No estimator constructs one.
*Avoid*: Uncertainty Set (above), which is an object that a caller builds and passes.

**Ambiguity Radius**
The size of an Ambiguity Set, in the units of the scenarios that the model prices. For a Wasserstein or a Gelbrich ball, it enters the model as the coefficient of a dual-norm penalty on the weights. So the same number is named `val` on a Regularisation Estimator and `r` on a distributionally robust risk measure. Those scenarios are the returns on every owner but one. The distributionally robust drawdown measure measures its transport cost against the per-asset drawdowns, so its radius has drawdown units, and its Calibration Series states them.
*Avoid*: Norm Ceiling (below), which bounds the same norm instead of multiplying it.

**Worst-Case Distribution**
The distribution in an Ambiguity Set at which a risk measure reaches its value: one probability for each observation, whose expected loss is the risk. The entropic, the relativistic and the power norm measures solve a conic programme for their value, and the dual of that programme gives the distribution. Held fixed, it gives the gradient of the measure, so `risk_contribution` and `risk_gradient` take one solve for the distribution and differentiate its expected loss, not the solve. `worst_case_distribution` reads it, and it returns `nothing` for a measure whose value is not a solve.
*Avoid*: Uncertainty Set (above), which bounds a moment and not the probabilities.

**Norm Ceiling**
The upper bound on a norm of the weight vector, in a constraint and not in the objective: `l2c`, `lpc` and `linfc` on `JuMPOptimiser`. Its reciprocal is a floor on the Effective Number of Assets, so it is a statement about diversification and has no units. A ceiling and an Ambiguity Radius read the same norm, but they are different quantities. So each takes its own family of Calibration Rules, and the slot of one admits no rule of the other family. `LpRegularisation` takes both readings. So its `val` is the one slot whose reading the field that holds the term decides, and not the bound of the field itself.
*Avoid*: Ambiguity Radius (above), which is the coefficient of the norm and not a bound on it.

**Calibration Rule**
A value in a slot that computes its own number from the Prior instead of stating one. Factory resolves it on the clustering route, and the risk-constraint route resolves it inside a JuMP build, so the constructor of the containing type validates the result. Both routes give the rule the same effective solver. A stated number keeps the quantity fixed across a refit. A rule keeps fixed the things that the rule is defined in terms of, and lets the quantity change. A rule takes the name of the method that it runs. It takes the name of the quantity as a suffix only where another rule or a mathematical object already has the bare method word. A rule states a default for every keyword that it can, so a bare call constructs. The two rules whose keyword is the whole content of the rule state no default, and each refuses a bare call with a message that names the quantity. ADR 0095 owns both lists.
A Cross-Sectional Factor Prior resolves its own two slots inside its fit, after the split of its Return Forecast, so each fit, refit and online step moves them.
*Avoid*: Radius Calibration (above), which is one specific conversion rather than the mechanism.

**Calibration Slot**
The field that holds a Calibration Rule, and the thing that names the quantity: the Significance Level of one end of the distribution, the Deformation Parameter (§5) of one end, the Ambiguity Radius, the Esfahani-Kuhn tail weight, the Norm Ceiling, the Spanned Shrinkage (§3.1), or the Orthogonal Forecast Scale (§3.1). The type bound of the slot names the one rule family that computes its quantity, so construction refuses a rule of another family. The bound also admits a plain function of the five arguments of the rule, which is the case that has no type. A function names no family, so there the name of the slot states the quantity. `LpRegularisation.val` is the one slot that names two quantities, and it is the one slot that admits no plain function. A caller therefore writes the rule alone, and the slot stores what the caller wrote. ADR 0095 owns the mechanism.
*Avoid*: Calibration Rule (above), which is what a slot holds rather than the field that holds it.

**Travelling Pair**
Two slots whose Calibration Rules must resolve in a stated order, because one rule reads the number that the other resolved to. The library has two pair shapes. A Deformation Parameter (§5) is next to the Significance Level of its own end of the distribution, and three rules read that sibling: an entropy budget and two tail-index readings. An Esfahani-Kuhn tail weight is next to that same Significance Level, and one rule reads it. The rule reads the sibling `alpha` from the Calibration Context (below). So the owning type resolves `alpha` first, and states the number in the context of the slot that reads it. No derivation can find that order. So a type that owns a Travelling Pair writes the resolution next to its declaration. A type that owns none writes no resolution. Its declaration is the whole statement, and the library derives the resolution from it.

**Calibration Series**
The series of the sample that a Calibration Rule reads: the returns, the absolute drawdowns, or the relative drawdowns. A marker is an Estimator that a caller states. It is a property of the slot owner and not of the rule, because one slot key is for a measure of the return distribution and for a drawdown measure alike. The owner states it as a trait and puts it in the Calibration Context (below). No rule holds a marker of its own. A rule forms no portfolio. So the library forms a drawdown series for each column of the sample, with one entry for each observation, and every count that the rule forms stays the same. A rule needs the marker when its answer has the units of the quantity or reads its shape: the two tail-index readings, the three Ambiguity Radius rules and the Esfahani-Kuhn tail weight. A Significance Level is a probability and a Norm Ceiling bounds the weight vector, so neither family reads one.
*Avoid*: Travelling Pair (above), which is a pair of slots rather than a property of the owner.

**Calibration Context**
The record of what a calibration site knows that the slot's key does not: the Significance Level of a sibling slot, the Calibration Series (above) the slot owner prices, and the norm order of the constraint or of the penalty that holds the quantity. A Cross-Sectional Factor Prior also states the two parts of its Return Forecast, its factor-model block and, when a rule asks for it, the forecast history. It is the sixth argument of the resolver and the fifth of every Calibration Rule. So a rule reads the fields that it needs, and a rule that reads none names the type and ignores it. No rule holds a field for any of them. Each belongs to the site, and a rule cannot know which site it reached, so the rule has nothing to overwrite. A caller who runs a rule outside a measure builds the context the site would have built.
A site that prices the empirical measure of its own rows, which is a distributionally robust measure, also states the count of those rows. The key cannot say if a radius prices those rows or the observations behind the moments, so the site says it. A site that states no count prices the moments. ADR 0138, #1833.
*Avoid*: Travelling Pair (above), which is the ordering between two slots rather than the record the order produces.

## 4. Optimisation

**Optimisation Estimator**
Makes portfolio weights from a Prior Result and zero or more constraints and risk measures. The primary split is **JuMP-based** (requires a solver), **Non-JuMP-based** (naive and clustering), and **Meta-optimisers** (wrap others).

### 4.1 Naive (Non-JuMP)

- **EqualWeighted**: `1/N` across assets.
- **InverseVolatility**: weights inversely proportional to asset volatility.
- **RandomWeighted**: random feasible weights, as a baseline.
- **BestConstantRebalancedPortfolio**: the constant portfolio that maximises the sample's log wealth on the simplex. It uses Cover's (1984) fixed point and no solver. It is the solver-free Hindsight Comparator of the online family, and the solver-free optimiser that a `FollowTheLeader` rule solves again on the default set. Under a bounded set it answers the repaired fixed point, and it refuses a programme set. A bounded or constrained one is `MeanRisk` under `LogarithmicReturn`. ADR 0161, ADR 0164.
- **OnlinePortfolioSelection**: updates the allocation from each realised price relative through an Online Update held on `alg`. It fits no moment and solves no programme of its own.

**Online Portfolio Selection**
The naive head that updates its allocation from each realised price relative through an Online Update. It fits no moment and solves no programme of its own, and it carries the regret guarantee that the paper of the rule states against every price sequence. It is one head, `OnlinePortfolioSelection`, generic over the rule it holds. Every shared verb belongs to the head: the batch Causal Pass, the Recursion Result, the block step, the projection and the refusals. A member is a rule, its own state, and one update. A meta-learning aggregator over experts is a rule over rules, not a second head. ADR 0155.
*Avoid*: the `Online` wrapper, which declares a Sample Buffer on an estimator and an Online Scheme on a scheme. Either can carry this head without being it.

**Online Update**
The one-step rule of an Online Portfolio Selection head, `(state, w_t, x_t, rows) → (state', w_{t+1})`. Its inputs are the rule's own state `st`, the allocation held during period `t`, the price relative that period `t` revealed, and the rows that the head holds through that period. Its outputs are the state and the allocation for period `t+1`, on the Allocation Set of the head. The head holds it on its `alg` slot, and it is the only thing that varies between members of the family. The rule writes the state in place where it can, and the state can be nothing. Examples of a state are a Gram matrix, a belief covariance and the Rule States of the experts. The allocation is always a new vector, and the rule never holds the rows. ADR 0155, ADR 0157.
The rows reach the rule as a Returns Result of the buffer. It holds the rows verbatim under the pinned names, and the active masks of the rows as a time-varying Asset Panel. So a statistic over the rows reduces to its Coverage Universe, as the batch verb does. The price relative is one at a gap, because the leg sat in cash, and a kernel reads the rows the same way. ADR 0170.
*Avoid*: `partial_fit!`, which is the verb that applies the rule to a block of rows. And the step of an Online Scheme, which is the cadence of the Fold Loop, not the rule.

**Next-Period Allocation**
The convention under which an Online Portfolio Selection head answers, in batch and online alike. Read after rows `1:t`, the answer is the portfolio for period `t+1`, and row `t` of any weight path reads rows `1:t-1` and nothing later. It is the protocol of the papers, where the update to `b_{t+1}` closes every iteration. It takes one more update than a backtest driver takes, because the answer of an optimiser is always for the period after its data. ADR 0155.
*Avoid*: the portfolio held during the last row, which the wealth of a backtest uses and a live deployment never trades.

**Causal Pass**
The batch verb of an Online Portfolio Selection head. It starts from the start weights, applies the Online Update over every row of the window in order, and answers the Next-Period Allocation. A batch call on a head that already carries a state runs from the start weights, and it neither reads nor writes the state. To *continue from here*, call `partial_fit!` and then `optimise(opt)` with no data. ADR 0155.
*Avoid*: a refit, which no member of the family does. And the hindsight benchmark over the same window, which is `MeanRisk` under a logarithmic return.

**Recursion Result**
The second kind of answer that an optimiser can declare for `optimise(opt)` with no data (ADR 0137 amended). The allocation of the state is the answer, wrapped in a Result, and no batch path runs. Its batch verb is the same Causal Pass that the online step takes. So a batch fit over the folded rows and the call with no data are exactly identical, because both take the same sequence of single-row updates. The state behind it is a Rule State beside the rows, which it holds once. The state pins the Fold Context, records every timestamp folded, and holds no shared column buffer. It refuses a State Merge, because an online update is not a sufficient statistic for its block. The result is a `NaiveOptimisationResult` with one vector. A weight sequence is the product of a walk-forward, never an optimisation result. ADR 0155.
A Recursion Result that fails walks the fallback chain. The call rebuilds no rows, so the head also folds every step into its fallback, and the fallback reads from its own state the rows that a batch fit would give it. A fallback under a Reconstitution takes no step. It gets the rebuilt `ReturnsResult` and fits its own prior from it, never the carried prior of the optimiser it replaces. ADR 0137 amended, ADR 0155 amended.
*Avoid*: a Reconstitution, the first kind, under which the call with no data rebuilds the `ReturnsResult` and runs the batch path. And Recursion Read-out, the retired name. And reading the answer of the head off the state by hand, which skips the Result, the mask and the finaliser.

**Online Selection Rule**
A member of the Online Portfolio Selection family, a struct that the head holds on its `alg` slot. The struct takes the name that the paper gives the algorithm, in full words. A leading "Online" is dropped because the family says it. A trailing "Optimisation" or "System" is dropped because the head says it. "Portfolio" stays where the paper names the portfolio. The acronym of the literature goes in the docstring and in the roster below, never in the type name. A numbered variant of a paper that changes one formula is a typed slot on the rule (`slack`, `formulation`). A variant that changes what the rule reads is the same rule under a different Price Relative Forecast on its `me` slot. Every rule is one of three mechanisms: a closed form followed by the projection, an Expert Mixture, or a follow-the-leader over a Sample Selector. The five families of the survey are a grouping of the docs, not a type layer. ADR 0156.
When two papers share one update and neither names the mechanism, the struct takes the name of the mechanism, and the name of each paper is a constructor of it. An example is `ForecastReversion(; me, eps)` with `MovingAverageReversion` and `RobustMedianReversion`. ADR 0158.
The first-order rules are one such mechanism, Mirror Descent, with `ExponentiatedGradient` and `GradientProjection` as constructors. The tracking step is another, `ForecastTracking(; me, eps)`, with `PeakPriceTracking` as constructor. A fourth set carries the online convex optimisation reading of the family. The steps it adds are state and slots on the rules above, not new closed forms. ADR 0165.

**Expert Mixture**
The rule over rules. It holds a vector of expert rules and a weighting. The weighting is itself an Online Selection Rule, and it reads the expert-return vector `r_t = (⟨h_k(t), x_t⟩)_k` in place of the price relative. The state of the mixture is the state of every expert plus the weight vector over the experts. Its answer is the weighted sum of the next-period allocations of the experts. Under the buy-and-hold weighting it is the wealth-weighted mixture, a buy and hold over the experts. Over sampled constant rebalanced portfolios it is the universal portfolio. Under exponentiated gradient or the Newton step it is the online gradient update or the online Newton update. ADR 0156.
Its Online Update has one more step than the update of an expert. First every expert projects onto the Allocation Set of the head, and the weighting moves the weight vector on the Expert Set of the mixture. Then the mixture projects the blend onto the Allocation Set of the head once more, in the Euclidean geometry, from its own Price-Adjusted Allocation. This projection is the identity wherever the blend is already feasible (every convex kind under buy-and-hold). It is the repair where a turnover ceiling under another weighting, or a cardinality kind, makes the blend infeasible. The mixture skips it by type on a closed-form set. On a programme set, a mixture of `K` experts pays `K + 1` programmes per period. ADR 0163.
By default each expert reads its gradient at its own iterate. Under the `BlendPoint` Gradient Point, each expert reads it at the played blend of the mixture, which is the shared gradient on which the papers state the dynamic-regret mixtures. The weight vector starts at the Start Allocation of the mixture over the experts, `p0`. It is uniform by default, and the mixture projects it once onto the Expert Set in the geometry of the weighting at the seed, as the head projects `w0` onto its own set. The constructors `Ader` and `Sword` give the mixture over first-order experts at a geometric grid of rates, under the exponentiated-gradient weighting and the blend point. The mixture over single-asset constant rebalanced portfolios under the Switching Weighting is `SwitchingPortfolio`. ADR 0165.
*Avoid*: a meta-optimiser, a second head, or a walk-forward. The mixture is one rule, and its weight vector is state, not a scored product.

**Expert Set**
The Allocation Set over the `K` experts of an Expert Mixture. The weighting projects its step onto this set, in the Projection Geometry of the weighting rule. The mixture holds it on its `eset` slot. When the slot is `nothing`, the default, the set is the bare `K`-simplex. Otherwise it is weight bounds on the share of each expert: a scalar bound broadcasts over the experts, and a vector bound has one entry per expert. It admits the closed-form kind only, so it never needs a solver. A mixture whose expert set changes each period admits scalar bounds alone. ADR 0163.
*Avoid*: the Allocation Set of the head, which is over the assets and is the set that the blend meets. And the `proj` of a weighting, which is the geometry in which the mixture reaches the Expert Set, not the set.

**Allocation Set Constraint**
The adapter through which the re-solve of a follow-the-leader rule honours the Allocation Set of the head. It wraps the resolved set, the Price-Adjusted Allocation of the step and the rows of the head as a custom JuMP constraint. It appends that constraint to the `ccnt` of the held optimisation estimator for that solve. So the feasible region of the programme is the set intersected with whatever the held estimator carries itself. Both homes hold, and a caller who wants one home leaves the held estimator bare.
The adapter calls the builders of the set on the model of the leader. So the turnover ceiling measures from the Price-Adjusted Allocation, and the variance ceiling reads the `pe` of the set on the rows of the head, as they do in a projection. The solver-free `BestConstantRebalancedPortfolio` has no model to add it to. A bounded set goes into its `wb` and answers the repaired fixed point. A programme set is refused where the rule and the set meet, with an error that names it. ADR 0164.
The rows of the set enter the model of the leader without the names that the builders of the head registered. The bounds and the budget are anonymous, the linear constraints take a prefix of their own, and the turnover and tracking error take a free index. So both homes hold on one model. A MIP kind registers the indicators of the model, and one model holds them once, so the caller states it in one home. It needs a MIP-capable solver on the held estimator, whose model it enters.
*Avoid*: a translation of the set into the constraint fields of the held estimator, which needs a rebuild per head and a precedence rule. And a projection of the optimum, which is not the leader.

**Sample Selector**
The part of a follow-the-leader rule that names the past rows on which the held optimisation estimator solves again. It names every row so far, the last `W`, or the rows whose preceding window resembles the latest one by histogram cell, kernel radius, nearest neighbours or correlation. An empty selection answers the uniform portfolio, projected onto the Allocation Set. The selector reads the rows it names from the rows that the head holds once, unbounded for a prefix. The re-solve runs on the Allocation Set through an Allocation Set Constraint. ADR 0156, ADR 0157, ADR 0164.
*Avoid*: a Prior. A matched sample is a conditional distribution, but the selector feeds a re-solve inside a rule, and it is not on the prior slot of the head.

**Rule State**
The pair `(st, w)` of an Online Selection Rule's own state `st` and the allocation that it holds during the current period. The family recurses over this unit. The Partial Fit State of the head holds one Rule State. Beside it, the state holds the rows that any rule of the tree reads, the pinned asset names and static Asset Panel, and every timestamp folded. The head holds the rows once, as a Sample Buffer of returns verbatim, `NaN` included, with the active mask of each row beside it. The need of the rule tree, `rows_needed`, caps the buffer. From these rows the head forms the price relative `1 .+ r` that the rule reads, which is one at a gap.
An Expert Mixture's own state is a vector of Rule States, one per expert, beside the Rule State of its weighting. The allocation of that weighting is the weight vector over the experts. A view by asset slices every per-asset axis, renormalises every allocation, and forwards the state `st` to the rule. The rule refuses when its state cannot slice. A copy is deep. The head refuses the merge once, on its own state. ADR 0157.
The state `st` is what the rule keeps from one Online Update to the next: a Gram matrix, a belief covariance, the Rule States of its experts, or `nothing` for a rule that keeps nothing. A docstring calls it the state of the rule.
*Avoid*: a rows buffer inside a rule's own state, because a mixture would then hold the rows once per expert. And a forced zero from the head. A step reads a non-finite return as cash, silently at an inactive asset and as a Held Gap at an active one. So the allocation stays over the full pinned universe, and a relisting asset enters again at the weight of the recursion. The Investable Mask acts when `optimise(opt)` runs with no data. Inside the recursion it acts only where the prior of a programme set cannot price a leg, and the set then writes that leg as zero, as a batch head does. And a filled zero in the buffer, because a statistic over a filled row reads a return that never happened. ADR 0162, ADR 0170.

**Block Step**
What a fold of several rows means to an Online Portfolio Selection head. The head applies that many Online Updates, one per row in order, and then one call of `optimise(opt)` with no data. So a walk-forward with `test_size = k` holds the Next-Period Allocation as of the end of the block for `k` periods. The head computes the `k − 1` allocations formed inside the block and never holds them. The state never learns the cadence of the loop. This is why the batch Causal Pass and the online step agree exactly at every `test_size`. A rule that must rebalance weekly runs on weekly returns at `test_size = 1`. ADR 0157.
*Avoid*: one update on the compounded price relative of the block, which is a different algorithm. And a refusal of `test_size > 1`, which would make it impossible to evaluate the head at a rebalance coarser than its data.

**Price Relative Forecast**
The one vector that a forecast-reading Online Selection Rule consumes, `x̂ = 1 .+ mu`. The rule reads it from the Expected Returns Estimator on its `me` slot, and the vector stands for the price relative of the next period. It is the input of the rule, not of the head. The head holds no Prior. A rule that needs a forecast refuses a missing one at construction, and it ships the statistic of its paper as the default. A Prior enters only through `PriorExpectedReturns`.
A forecaster folds where its fold is exact, and its state is then a `ForecasterState` on the Rule State. Otherwise the rule fits it again from the rows that the head holds. Its window is `rows_needed`: one when it folds, and unbounded when it is batch-only. A forecast is flat, one in every asset, where the forecaster has fewer rows than its second moment needs or answers a non-finite entry. The step of every rule holds on a flat forecast. A covariance never enters through this vector. A constraint that reads a covariance holds its own estimator. ADR 0158.
*Avoid*: Return Forecast, the idiosyncratic prediction that a factor Prior splits against its exposures. And the `Online` wrapper on the forecaster, which the head refuses because its own rows buffer is the buffer. And a Reversion Target, which is a wrong name for a momentum forecaster.

**Allocation Set**
The set in which every allocation of an Online Portfolio Selection head lies. The head holds it on its `set` slot. It is the simplex by default. Otherwise it is every constraint kind that a `JuMPOptimiser` takes, under the field names and type bounds of the optimiser, always under a budget of one. It comes in two kinds. Every Projection Geometry reaches the first kind in closed form, so it carries no solver. The second kind is a programme, so it needs a solver. So the type of a configuration states whether it needs a solver. Cash is an asset of the set with price relative one, never a budget below one. ADR 0159.
The risk ceilings of a programme set are Risk Measures whose `settings.ub` is a number. The JuMP builder of each measure writes the ceiling, with the set as the Risk Constraint Owner, on the prior result fitted on the rows of the head. A variance or standard deviation that holds its matrix reads no rows, and it is the set's own cone. A kind keyed by name resolves once per fold over `sets`. A kind that reads the rows resolves at every step: a ceiling, a tracking error, a centrality or phylogeny row, an exposure row, a return floor. The programme set also carries the direct objective penalties of the optimiser. The Objective Penalty folds them into the divergence of the geometry, so a penalised step is a proximal mirror step with the same regret bound. Its builders run in the order of the builders of a JuMP head. Fees and a budget below one have no field. A fee is a deduction from a return that the projection does not have, and cash is a column. ADR 0159, issue #1206.
*Avoid*: Constraint Space, which is the basis in which the names of a constraint resolve, not the set. And a Weight Finaliser, which repairs a solved vector after the fact, where the Allocation Set is the set onto which the step projects.

**Risk Constraint Owner**
The object that a risk-measure builder takes in its `opt` slot. It is the JuMP optimiser whose programme the constraint joins, or the programme Allocation Set whose projection it joins. A builder asks an owner three things and nothing else: the solver against which a Deferred Quantity resolves (`risk_constraint_solver`), the risk-contribution rows of a `Variance` (`risk_contribution_constraints`), and whether the bound is honoured (`set_risk_upper_bound!`). The returns and the moments come from the prior result. `RiskConstraintOwner` is the union of the two, and `RiskBoundOwner` is the union of the owners whose bound is honoured. ADR 0159.
*Avoid*: an optimisation estimator, which the programme set is not. And a protocol of `sets`, `strict` and the scales, which the builders never read off the owner.

**Projection Geometry**
The divergence in which an Online Selection Rule projects its raw step back onto the Allocation Set. The rule holds it on its `proj` slot where the rule admits a choice. Where the geometry is the rule's own gradient mass, the rule builds it at the step (ADR 0168). The geometries are squared Euclidean distance, relative entropy, the Tsallis and log-barrier (Burg) potentials, the diagonal norm of a rule's own gradient mass, and the norm of the rule's own Gram matrix. Every one but the Gram norm is a scalar root on the default set (ADR 0165).
The slot of each rule is bounded to the geometries that its theorem covers, and it defaults to the geometry of its paper. So the geometry is a statement that the rule makes, not a parameter that the caller sets. A bound outside the domain of the geometry, such as a negative weight under relative entropy, is refused when the rule and the set meet. Sparsity belongs to the geometry. A Euclidean projection can give a zero, and an entropic one cannot. ADR 0159.
*Avoid*: a `JuMPWeightFinaliserFormulation`, which is an error norm on a repair, not a divergence on a step.

**Constrained Update**
The two halves of every Online Update. First the rule takes its unconstrained step to a raw vector. Then it projects that vector onto the Allocation Set in its Projection Geometry. The projection is closed form where the set admits it. Otherwise it is a programme in the divergence of the geometry. `set_allocation_set_constraints!` assembles that programme from the same constraint builders that a JuMP optimiser uses. Its sibling `add_allocation_set_constraints!` is what the Allocation Set Constraint of a JuMP head calls. The rule never sees the constraints, and the set never sees the rule. The turnover ceiling of a programme set is the per-asset ceiling of the library, `|w_i − ŵ_i| ≤ tn_i`, against the Price-Adjusted Allocation. ADR 0159.
A follow-the-leader rule is the one member whose step is itself a programme, and it does not project its optimum. The Allocation Set enters its re-solve as an Allocation Set Constraint, and the optimum is the leader on the set. The rule projects only the damped mix `(1 − γ) w⋆ + γ w_t`, in the Euclidean geometry. ADR 0164.
*Avoid*: a solve of the rule's own objective in a closed-form member, which no member does. And a projection of a solved optimum, which is not the leader. And a Weight Finaliser applied after the step.

**Held Step**
An Online Update with a projection programme that did not solve, or that the step could not form on the rows it holds. The projection answers the Price-Adjusted Allocation that it received, so the fund trades nothing that period. The rule's own state still absorbs the row. The head warns once with the timestamp of the row and continues. The Result of the head reports the retcode of the step that produced the Next-Period Allocation. ADR 0159.
The hold is the decision of the family itself. So `optimise(opt)` with no data reports an `OptimisationSuccess` whose `res` is the `HeldStep` record. The record holds the timestamp of the row, the reason, and the solver trials of a programme that failed. The fallback chain never runs on a hold, and the fold threads and drifts the held allocation. ADR 0160.
A programme set whose prior is fitted on the rows of the head holds the step while the head has fewer than two rows, because a covariance of one observation does not exist. A caller who wants the ceiling from row one gives the risk measure its matrix.
A solver-free projection onto a `BoundedAllocationSet` also holds the step when its scalar root misses the budget in floating point. The root of a barrier geometry does this when a base of `1e20` cancels the multiplier that brings it to a share of the budget. The projection never returns an allocation off the budget. A raw step on which no projection is defined is refused with a `DomainError`, not held. Such a step has a non-finite entry, or it is an entropic step whose positive entries cannot reach the budget at their caps. Issue #1258.
*Avoid*: a Held Gap, which is a missing return at a held weight, per observation and asset. And a fallback to a weaker set, which would drop a constraint on the day it binds. And a re-trade to the previous target, which is a trade that the programme refused.

**Projection Step**
The task-scoped context inside which the projections of an Online Update run. It holds the rows that the head holds through the period, the timestamp of the row, and the log of Held Steps. The signature of the projection is `project(proj, set, q, w)` and no more. So the head opens one Projection Step around each Online Update. The covariance cone and tracking error of a programme set read the rows from it, and a failed programme writes its record to it. Outside a step, a projection reads no rows and reports a hold as a warning.
*Avoid*: a rows argument on `project`, which every rule would have to thread. And a mutable slot on the Allocation Set, which is configuration.

**Price-Adjusted Allocation**
The book that an Online Portfolio Selection head holds at the end of period `t` before it trades, `ŵ_t = w_t .* x_t / ⟨w_t, x_t⟩`. It is one row of the self-financing Weight Drift at budget one. It is a closed form of the update's own inputs, and the step computes it on that row. It is always the reference of the turnover ceiling of the Allocation Set, so the ceiling bounds the trade that the step executes. That trade is zero for `BuyAndHold`, whose update is this vector, and the full rebalance for `ConstantRebalancedPortfolio`. The cost-aware rules update from this vector. At `test_size = 1` it equals the Held Weights of the fold, so a turnover fee charged under a Previous-Weights Source prices the same trade that the ceiling bounds. ADR 0160.
*Avoid*: the distance between two consecutive targets, which is a change of decision and not a trade, and which every rule's own regulariser already measures. And the threaded previous weights of the loop, which an Online Update never reads.

**Start Allocation**
The allocation that an Online Portfolio Selection head holds during the first period of its recursion. The head holds it on its `w0` slot. It is a given vector over the pinned asset names. The head pins and views it with the names, and projects it once onto the Allocation Set in the Projection Geometry of the rule at the first step. When no vector is given, it is `1/N` over the whole pinned universe, unlisted names included, and the head projects it onto the set as it projects a given one. The legs of the unlisted names earn a zero return until they list. The Start Allocation reaches the first Online Update as its `w`, and that is the whole of its meaning. A rule that steps from `w` continues from it. A rule whose next allocation is not a step from `w` replaces it at that update, so the fund holds it for one period. Such rules are a constant rebalanced portfolio, an Expert Mixture and the Newton step. ADR 0162.
Such a rule can still carry the start forward. The gradient of the Newton step reads the held `w`, and an expert that steps from `w` starts at the Start Allocation.
A set that reads the rows of the head has nothing to project onto before the first row. So it holds the start as given until the first Online Update, which holds every step until the head has two rows.
An Expert Mixture or a follow-the-leading-history rule holds a start for each expert. That start is the Start Allocation for a rule that reads `w`, and a constant rebalanced portfolio's own `w`. It meets the set in the same way, in the expert's own geometry, so no expert earns a period-one return outside the set.
*Avoid*: the previous weights of the loop, which reach the head through `factory` into a fee and a fallback, and never into the recursion. And a uniform start over the active mask of the first row, which is a forced zero that a multiplicative rule holds forever. A caller who wants that start gives `w0`.

**Log-Wealth Regret**
The gap in log terminal wealth between a comparator and a strategy over one and the same sequence of rows, `Σ_t log(1 + r_{b,t}) − Σ_t log(1 + r_{a,t})`. It is positive when the comparator wins. The per-period form divides by the number of rows. The per-row difference series carries a Mean Test of equal expected log growth, which is exact only for a comparator that did not read the rows. A verb reads it off two prediction results, never a scorer. A hyperparameter search ranks on log wealth per period, which orders candidates as regret against any fixed comparator would. Negative regret is expected on many sequences and is not a defect. ADR 0161.
Beside the regret, the Result reports the running regret, which is the cumulative sum of the per-row difference. It also reports the path length of the comparator, `P_T = Σ_t ‖u_t − u_{t−1}‖₂`, over its per-fold targets. The targets are embedded by asset name on the union of the universes of the folds, and the length is `NaN` for a comparator of one fold. Dynamic regret is regret read against the path that a Hindsight Split comparator was allowed to walk, and that path length is zero for a static comparator. ADR 0165.
*Avoid*: a summary column, because the performance summary reads one series and holds no comparator. And regret between two runs over different rows, which the verb refuses.

**Hindsight Comparator**
An estimator fit on the same rows on which it is then scored, and predicted in sample over those rows. So its log wealth is the ceiling against which a causal strategy is measured. The best constant rebalanced portfolio is `MeanRisk` under `LogarithmicReturn` and a maximum-return objective (or `BestConstantRebalancedPortfolio` with no solver). Any JuMP optimiser with a returns slot under the same return is its own log-return oracle. The best stock is the top-1 Asset Selector under `MeanReturn(; flag = true)` composed with `EqualWeighted`. The same estimator run through a walk-forward on the `cv` of the strategy is the causal comparator over the same rows. ADR 0161.
*Avoid*: a member of the online family that reads the whole buffer, which would break the Causal Pass. And a walk-forward fit that reads its test rows by accident. A Hindsight Split reads them by declaration, per row.

**Mirror Descent**
The first-order Online Selection Rule of the family, `MirrorDescent(; eta, proj, alpha, obj)`. It takes one step `argmin_w η⟨g, w⟩ + D(w, w_t)` in the divergence that its Projection Geometry holds. So online gradient descent is its Euclidean case and exponentiated gradient is its entropic case, and each is a constructor. Its `alpha` is the uniform mix that makes the entropic step universal without a lower bound on the price relatives. The update reads mixed relatives and the head plays a mixed allocation, while the Rule State carries the unmixed iterate. Its `eta` is a number or a Learning-Rate Schedule. Its `obj` is log wealth or a Risk Loss. The optimistic step wraps it and carries a secondary iterate, which the next step reads in place of the played one. ADR 0165.
*Avoid*: a first-order step written twice under two names. And `ExpectationMaximisation`, which is a different update and not a mirror map. It is the Soft-Bayes convex combination of holding and posterior.

**Learning-Rate Schedule**
A rule for the step size of a first-order Online Selection Rule at period `t`. The rule holds it on its `eta` slot in place of a number. It is a function of the period count and of the rule's own state. So it can follow the count alone (the inverse square root). It can read a running statistic of the run that the state keeps for it (the self-confident rate). Or it can name a stage boundary at which the rule restarts its allocation at the Start Allocation and its state at its seed (the doubling trick). At a constant rate every rule is the fixed-rate step of its paper. The expectation-maximisation step is written in its online form, whose fixed-share pull toward the Start Allocation vanishes at a constant rate. ADR 0165.
A schedule writes its statistic once per row, after the step by default, so the rate of a period reads the past alone. A schedule that chooses the rate from the period's own row declares it (`reads_period_row`), and the rule then writes the statistic before it reads the rate. The windowed best rate is the schedule that does this. It replays the exponentiated gradient at every rate of a set as an expert that the state keeps. It takes the rate of the expert whose wealth over the last `window` periods is the largest (`WindowedBestRate`, MAEG, and AEG over the whole history). The expectation-maximisation step refuses it, because its online form also reads the rate of the next period.
*Avoid*: `TimeDependent`, which schedules a field per fold from outside the recursion. And a schedule that owns state, which would put a state beside the state of the rule.

**Gradient Transform**
A rescaling of the gradient of the period that a Mirror Descent rule applies before its mirror step. The rule holds it on its `grad` slot. It is the identity by default. Otherwise it is one of the momentum transforms of the exponentiated-gradient-with-momentum paper: an exponential moving average of the gradient, a root-mean-square normalisation, or both. Each transform keeps its averages on the Rule State and does not change the update itself. So it stacks with any Projection Geometry and any Learning-Rate Schedule. `EGE`, `EGR` and `EGA` are constructors of the entropic rule under each transform. ADR 0165.
*Avoid*: a transform that changes the geometry, which is a Projection Geometry (the diagonal norm of AdaGrad), not a rescaling of the exponent. And a bias correction that the paper does not apply.

**Gradient Predictor**
The hint along which an optimistic step plays a half-step before it sees the next price relative. The wrapper holds it on its `predictor` slot. It is the last gradient (the default, under which the regret is the gradient path length), the mean gradient, or a gradient formed from a Price Relative Forecast. The extra-gradient step is the last gradient taken at the played point. It is a predictor and not a rule. It is the default, because the optimistic step reads its own gradient at the played allocation. The last gradient evaluated again at the new secondary iterate is the Mirror-Prox form, which is a flag on the same predictor. ADR 0165.
The wrapper keeps the last hint and the running sums of the squared hint residuals on its state. So the adaptive rate that reads them is a Learning-Rate Schedule on the `eta` of the wrapped rule, and no new verb. On the state of any other rule, it is refused with an error that names it.
*Avoid*: a Price Relative Forecast on the `me` slot of the rule, toward which a reversion or tracking rule steps. A predictor is the gradient that a first-order rule anticipates.

**Gradient Point**
The point at which the first-order experts of an Expert Mixture evaluate their gradient. The mixture holds it on its `grad` slot. Each expert evaluates at its own iterate (the default). Or every expert evaluates at the played blend of the mixture while it steps from its own iterate, which is the shared gradient on which the papers state the dynamic-regret bounds. ADR 0165.
*Avoid*: a shared state. Every expert keeps its own Rule State under either point.

**Risk Loss**
The objective on which a first-order Online Selection Rule can step in place of log wealth. The rule holds it on its `obj` slot. It is a risk measure, or a scalarised vector of them, evaluated over the last `window` rows that the head holds. Its gradient is read through `risk_gradient`, which is a closed form where the library states one and the finite-difference kernel otherwise. The rule makes one step per period toward the minimiser of the measure, estimated again each period. The regret theorems hold for the convex measures alone. ADR 0165.
At every step the measure resolves against the prior that its `pe` fits on the window, as a Programme Allocation Set resolves its ceiling. So `Variance()` reads the covariance of the window, and a measure that holds its own matrix keeps it. A prior on one row has no covariance, so the loss is read from the second row, and `window` is at least two. `RiskLoss` and `LogWealth` share the unexported supertype `AbstractOnlineObjective`, whose verbs are `loss_gradient` and `rows_needed`.
When its prior states `x_src = ExcessSource()`, the loss fits the prior on `X - B`, so `Variance()` is the tracking variance against the benchmark of the rows. The source of the head does not reach it, as a Programme Allocation Set also states its own source. ADR 0211.
*Avoid*: a solved objective, which is a follow-the-leader over a Sample Selector. And a differentiation package, which the library does not take on for per-type work.

**Hindsight Split**
A cross-validator whose every fold tests on one row and also trains on that row. The training rows are the prefix through the row, or the row alone. So an estimator run through it under the Hindsight Comparator rule gives the per-row comparators against which dynamic regret is stated. These are be-the-leader for the prefix, and the per-period minimiser for the row alone. It reads its test rows by declaration, and its docstring says so first. The path length of the comparator, the summed distance between consecutive targets, is reported beside the regret. ADR 0165.
*Avoid*: a walk-forward with a look-ahead by accident. And a comparator argument on the regret verb, which reads two prediction results and fits nothing.

**Budgeted Hindsight Path**
The best comparator sequence under a path-length budget, `max Σ_t log⟨u_t, x_t⟩` over allocations on the resolved weight bounds with `Σ_t ‖u_t − u_{t−1}‖_p ≤ L`. It is the comparator against which dynamic regret at a given budget is stated. It is one concave programme over the whole panel, with an exponential cone per row and a norm cone per step. The estimator `BudgetedHindsightPath(; L, p, wb, sets, slv)` fits it under the Hindsight Comparator rule. `optimise` answers the path as one Result with the Investable Masks of the rows. `predict` over the same rows answers one fold per row, so the regret verb reads it unchanged. At `L = 0` it is the static best constant rebalanced portfolio. At a slack budget it is the per-period minimiser. In between it is neither per-row comparator of a Hindsight Split. A missing return fixes the allocation of the asset at zero on that row, and the path is charged for the forced step. ADR 0165.
*Avoid*: a verb that hides the fit. And a Hindsight Split, whose folds fit again. And a programme Allocation Set, which the estimator does not honour, because it takes weight bounds alone.

**The roster** (name · acronym · family · set), ADR 0156. Set 1: `BuyAndHold` · UBAH · benchmark; `ConstantRebalancedPortfolio` · CRP, UCRP · benchmark; `MirrorDescent` · EG, GP, OGD · follow-the-winner, with the constructors `ExponentiatedGradient` and `GradientProjection` (ADR 0165); `NewtonStep` · ONS · follow-the-winner; `PassiveAggressiveMeanReversion` · PAMR · follow-the-loser; `ForecastReversion` · OLMAR, RMR, OLMAR-2, RPRT, GWR, LOAD · follow-the-loser, with the constructors `MovingAverageReversion`, `RobustMedianReversion` (ADR 0158), `ExponentialMovingAverageReversion`, `ReweightedPriceRelativeTracking`, `GaussianWeightingReversion` and `LocalAdaptiveLearning` (ADR 0165); `ExpertMixture` · BAH_W · meta-learning; `UniversalPortfolio` · UP · follow-the-winner, a constructor of the mixture.
Set 2: `ConfidenceWeightedMeanReversion` · CWMR; `AntiCorrelation` · Anticor; `TransactionCostOptimisation` · TCO; `ForecastTracking` · PPT, AICTR, TPPT · with the constructors `PeakPriceTracking`, `AdaptiveInputCompositeTrend` and `TrendPromotePriceTracking` (ADR 0165); `KernelTrendTracking` · KTPT · with the constructor `KernelTrendPatternTracking` over `KernelTrendPattern` (#1191); `ShortTermSparsePortfolio` · SSPO · with the algorithms `L1Optimum`, `HuberOptimum` and `AlternatingDirectionMethod` (#1259), all follow-the-loser; `ExpectationMaximisation` · EM, Soft-Bayes · follow-the-winner; the weightings `AggregatingAlgorithm` · AA, `TopK` · CORN-K, `WeakAggregatingAlgorithm` · WAA, with the constructor `AggregatingExponentialGradient` · WAEG, CAEG of the mixture, and `SwitchingWeighting` · SP, meta-learning, the last with the constructor `SwitchingPortfolio` of the mixture (ADR 0165).
Set 3: `FollowTheLeader` · FTL, SCRP, VRP, WSCRP, follow-the-winner, with the Sample Selectors `Prefix`, `LastRows`, `HistogramMatch` · BH, `KernelMatch` · BK, `NearestNeighbourMatch` · BNN, `CorrelationMatch` · CORN, `ClusterMatch` · KMNLOG (ADR 0165), pattern-matching, and the optimiser variants BS, BM, BGV, and the constructors `ShortTermLossControlPortfolio` · SPOLC over `RankOneCovariance` (ADR 0165) and `LowDimensionEnsemblePortfolio` · OLDEM over `LowDimensionEnsemblePrior`, a Prior whose forecast and covariance come off the same lagged regressions; `FollowTheLeadingHistory` · FLH, meta-learning.
Set 4, from the parity decision (ADR 0165): `AdaptiveSubgradient` · AdaGrad · follow-the-winner; `OptimisticStep` · OMD, extra-gradient · follow-the-winner, a wrapper over `MirrorDescent`; the constructors `Ader` and `Sword` of the mixture · meta-learning; the geometries `TsallisProjection`, `LogBarrierProjection`, `DiagonalProjection`; the Learning-Rate Schedules `InverseSquareRootRate`, `DoublingTrickRate`, `SelfConfidentRate`, `HintResidualRate`, `WindowedBestRate` · MAEG, AEG, with the constructors `MAEG` and `AEG` of the entropic rule; the Gradient Predictors `LastGradient`, `MeanGradient`, `ForecastGradient`; the Gradient Transforms `GradientMomentum`, `RootMeanSquareGradient`, `AdaptiveMomentGradient` · EGE, EGR, EGA, with the constructors `EGE`, `EGR`, `EGA` of the entropic rule; `RiskLoss`. The risk-aversion correlation-driven rule · RACORN-K and the combination-weights gradient descent · CW-OGD are configurations.
Beside the family, not in it: `BestConstantRebalancedPortfolio` · BCRP · benchmark (hindsight), a naive head of its own. The best stock in hindsight is an Asset Selector composed with `EqualWeighted`, not a type. ADR 0161.

### 4.2 Clustering (Non-JuMP)

- **HierarchicalRiskParity (HRP)**: recursive bisection of the cluster tree, allocating by risk.
- **HierarchicalEqualRiskContribution (HERC)**: splits weight down the cluster tree between whole clusters in inverse proportion to their risk, then inside each cluster in inverse proportion to each asset's own risk. Despite the name, the clusters do not contribute equally to risk.
- **SchurComplementHierarchicalRiskParity (SCHRP)**: HRP augmented with a Schur-complement step.

`HierarchicalOptimiser` is the shared configuration for these.

**Branch order**
The dendrogram's leaf permutation. It never changes the merge tree or the cluster memberships.
`:optimal`, the default, is the exact optimal leaf ordering. Of the orders that the tree permits by flips of its merges, it is the one with the least sum of distances between adjacent leaves. `:barjoseph` is the greedy heuristic of Clustering.jl, which decides each merge once from its four outermost leaves. `:r` is the order of the R function `hclust`. Clustering.jl itself maps `:optimal` to the heuristic, so the library dispatches the name on `Val` before it reaches `hclust`. See ADR 0055.
*Avoid*: reading `:optimal` in Clustering.jl's own documentation as this order.

### 4.3 JuMP-based

- **MeanRisk**: the efficient-frontier trade-off between expected return and risk. The classic objective optimiser.
- **RiskBudgeting (RB)**: targets prescribed risk *contributions* per asset or factor rather than a return/risk trade-off.
- **RelaxedRiskBudgeting (RRB)**: a convex relaxation of Risk Budgeting.
- **FactorRiskContribution (FRC)**: optimises or targets risk contributions attributed to factors.
- **Off-factor weights**: under `flag = true` of a factor optimiser, the weights `w2` on the directions the loadings do not span, so the asset weights are `b1 w1 + b2 w2`. Their Euler contribution takes no factor budget. `FactorRiskBudgeting(; hedge = true)` fixes them at the minimum-variance hedge of the factor exposures, `b2' Σ w = 0`, so their contribution to the variance is zero.
- **NearOptimalCentering (NOC)**: finds a robust interior point near the efficient frontier.

**JuMP Returns Estimator**
One term of the objective's return expression: `ArithmeticReturn` (dot-product) or `LogarithmicReturn` (geometric). The model-global return expression is their weighted sum.

**Objective Functions**
`MinimumRisk`, `MaximumUtility`, `MaximumRatio` (Sharpe-type), `MaximumReturn`.
`MaximumRatio` divides the excess return by the risk raised to `1/d`. Here `d` is the degree of the risk expression in the homogenised weights and `k`. So the SOC and quadratic `Variance` give the Sharpe ratio. A variance on a lifted weight matrix reads the corner of the lift. These variances are the semidefinite `Variance`, and the box, ellipsoid and norm-ball `UncertaintySetVariance`. Under `SquareRootRatio`, the default, the objective term of such a variance reads a second lift with a constant corner. The term then has degree two and gives the Sharpe ratio. Under `LinearRatio` the term reads the lift with the corner `k`. It then has degree one and gives the excess return per unit of variance. A bound reads the corner-`k` lift under both rules. Issues #1321, #1676, ADR 0217.

**Solver**
A wrapper around one external backend, such as Clarabel or HiGHS. It holds the module, the settings and the name of the backend, and the checks that validate a returned solution.

**JuMPOptimiser**
The shared JuMP model configuration: one or more `Solver`s as a fallback chain, and the JuMP-level settings.

**Model State**
The state of the JuMP model while the builders build it: the shared variables, expressions and scales that the constraint and risk builders read and write in turn.

**Per-Build Risk State**
The part of Model State that belongs to one risk build, not to the whole model. A build nested inside that build does not share it.
The lifted matrix `W` and the variance marks belong to the weights, not to the build. A nested build on the same weights shares them, such as a `DependentVariableTracking` build or a programme Allocation Set in a leader's model.

**Model Assembly**
The fixed sequence in which the constraint and risk builders of an Optimisation Estimator run. They turn an empty model into a fully constrained one. It is not Model State, which is the data that those builders read and write.

**Objective Penalty**
The accumulator through which every *soft* contribution reaches the objective: regularisation, soft Turnover and Tracking, and Custom Objective Terms. A contribution always worsens the objective, so a reward is a negative contribution.
A programme Allocation Set's penalties and its semidefinite phylogeny's `p · tr(W)` reach a leader's objective through the same accumulator, from inside the Allocation Set Constraint. ADR 0164.
Under MaximumRatio every regularisation penalty has degree one in the homogenisation variable `k`. The model divides a squared norm by `k`. So a penalty keeps its weight against a degree-one risk when the normalisation `ohf` changes. Issue #1318.

**Custom Term**
The user-facing extension point for a preference that the library does not name. A Custom Objective Term prices such a preference, and a Custom Constraint enforces it.

### 4.4 Constraints

**Universe Sets**
A user-defined mapping of names to named groups (sectors, countries), groups, or unique-member groups. It declares every axis that it carries, the assets and the two factor axes. It is the base input of nearly all Constraint Generation.

**Universe Prefix Grammar**
The six key prefixes that a Universe Sets carries, and the rule that each one declares. `xkey` names the asset universe and is the one mandatory axis. A key with the `xkey` prefix is a partition of the asset universe and has the length of the asset universe. `uxkey` prefixes a unique-entry variant, which names the `xkey`-prefixed partition that it draws from. `tfkey` and `utfkey` mean the same on the Time-Series Factor Axis, and `cfkey` and `ucfkey` mean the same on the Cross-Sectional Factor Axis. Both factor axes are optional, and the library demands one only at the point of need. No prefix may be a prefix of another, so a key resolves to exactly one axis. A key that matches no prefix is a plain group. The library expands it by name, and the expansion reads no axis.

**Time-Series Factor Axis**
The factor universe that a time-series regression is written on. A Universe Sets declares it under `tfkey`. Its names are the columns of `F` of a returns result. So a caller copies `rd.nf` into the dict, and the axis always agrees with the loadings.
*Avoid*: "the factor axis" alone, which no longer names one list.

**Cross-Sectional Factor Axis**
The factor universe that a cross-sectional regression is written on. A Universe Sets declares it under `cfkey`. Its names are the Factor Exposures that the fit read. These names exist only inside the fitted block, so no returns result carries them. A consumer never picks between the two axes by hand. `factor_axis_key` reads the key from the loadings result that the consumer already holds.
*Avoid*: "the factor axis" alone, which no longer names one list.

**Constraint Space**
The basis in which the names of a constraint resolve. Assets are the absence of a re-basis. `FactorSpace` is the only member, and the `ExposureConstraintEstimator` wrapper declares it.

**Factor Exposure Constraint**
A linear constraint on a portfolio's factor weights, written in factor names and projected through the loadings.

**Constraint Generation**
The user-facing utilities that convert high-level specifications into the numeric form that optimisers need. They build on Universe Sets. The utilities are:

- **LinearConstraint / LinearConstraintEstimator**: parses equations into `A·w ≤/= B`.
- **ExposureConstraintEstimator**: re-bases a linear constraint into another Constraint Space.
- **WeightBounds**: per-asset lower and upper bounds. Group-level sum bounds are linear constraints, not Weight Bounds.
- **RiskBudget**: per-name or per-group risk-contribution targets.
- **Phylogeny constraints**: `SemiDefinitePhylogeny` (SDP coupling) and `IntegerPhylogeny` (cardinality and grouping). `CentralityConstraint` constrains by centrality. Both phylogeny kinds also apply to the factor weights of a `FactorRiskContribution` through its `frc_ple` field, where an `IntegerPhylogeny` takes the bounds of its gate from the asset weight bounds.
- **Threshold**: buy-in thresholds, the minimum non-zero position size.
- **AssetSetsMatrixEstimator**: builds binary group-membership matrices.

**JuMP Constraints**
The layer that adds numeric constraint data to a JuMP model: the budget constraints, `LpRegularisation`, and the Turnover, Tracking and Fees constraints.

**Net vs Gross Budget**
`bgt` is the net exposure `1ᵀw`, `sbgt` the short side, and `gbgt` the gross exposure or leverage `‖w‖₁`. `gbgt` exists for the one combination that the other two cannot reach, a pinned gross exposure with a free net exposure.

**Exact vs Bounded Budgets**
The distinction between a budget that *bounds* the realised exposure and one that *pins* it. A pin needs a sign bit for each asset, which makes the problem mixed-integer. So a pin is opt-in.

**Time-Dependent Input (Schedule)**
An optimiser input whose value changes across the folds of a cross-validation scheme, in place of one value for the whole horizon. Only **problem definition** may vary this way, never **execution control**.

**Schedule of Optimisers**
A Time-Dependent Input whose per-fold values are whole optimisers, so the strategy varies over the backtest.

**Time-Dependent Context**
The per-fold information that the Fold Loop hands to a function-form Time-Dependent Input: the position and count of the fold, its data, and the weights of the previous fold when the loop threads them.

**Time-Dependent Callable**
A struct, not a bare function, that computes the per-fold value of a Time-Dependent Input from the Time-Dependent Context. Its parameters are data, and it declares as a trait whether it needs the previous weights. A member declares what it returns: a constraint value, or an optimiser.

**Weight Finaliser**
A post-solve adjustment that forces weights into the feasible region at the budget that they carry. There are four. `IterativeWeightFinaliser` is the default. It redistributes the weights and keeps the ratios of the free weights, and it falls back to the Euclidean projection when it cannot reach the bounds. `EuclideanWeightFinaliser` is the exact Euclidean projection. `EntropicWeightFinaliser` is the exact entropic projection for long-only weights, and it keeps the ratios of the free weights. `JuMPWeightFinaliser` solves again.

**Turnover**
The change in the portfolio relative to a reference, either the previous weights or a static target. It can be a soft penalty or a hard constraint.

**Tracking Error**
The closeness of the portfolio to a benchmark or target: `WeightsTracking` against a reference weight vector, `ReturnsTracking` against a benchmark return series, and `DataBenchmarkTracking` against the benchmark that the returns data carries.

### 4.5 Meta-optimisers

Meta-optimisers wrap and compose other optimisers. They use Cross-Validation to estimate the inner out-of-sample returns. The members are:

- **NestedClustered (NCO)**: optimise within clusters, then across clusters.
- **Stacking**: combine several base optimisers' weights through a meta-optimiser.
- **SubsetResampling**: average optimisations over many asset or observation subsets.

**Sub-Portfolio**
The unit for which a meta-optimiser solves one inner problem. It is also one asset of the synthetic universe over which the outer optimiser allocates. The sub-portfolios of NCO are cluster index sets. The sub-portfolios of Stacking are its inner optimisers.

**Panel Collapse**
The Asset Panel of the outer problem of a meta-optimiser. It holds each Panel Field of the inner panel, and `iv` and `ivpa`, collapsed onto the sub-portfolios with the normalised inner weights. These weights divide out the gross exposure of each sub-portfolio. `collapse_asset_panel` and `prepare_outer_rd` do the collapse.
At each observation, and for each Panel Field, the collapse weighs only the members that are active and hold no Placeholder there. The rule in the `pcol` field of `NestedClustered` and `Stacking` states what the weight of any other member becomes. `RenormaliseActive()`, the default, divides by the weight of the active observed members, so each value stays a convex combination. `InactiveAsCash()` reads the missing weight as cash with a zero feature, as the outer returns do (ADR 0192).
A sub-portfolio is active where one member with weight is active. This holds on the fold-less path and on the cross-validated path.
*Avoid*: reading the value of an inactive member, or a Placeholder, as part of the feature of a sub-portfolio. An inactive cell holds a fill value, a zero or a stale value, and a Placeholder holds no data (ADR 0102). And skipping a cell that a fill policy wrote, because it holds data (#1631).

### 4.6 Cross-Validation

Cross-validation tunes hyperparameters, evaluates out of sample, and gives meta-optimisers their inner out-of-sample returns. The two axes are **optimisation vs non-optimisation** and **sequential (time-aware) vs non-sequential**. The schemes are:

- **KFold**: k-fold with optional purging and embargoing.
- **CombinatorialCrossValidation**: all train/test combinations, which give several paths.
- **WalkForward**: `IndexWalkForward` / `DateWalkForward` for time series.
- **MultipleRandomised**: random asset subsets crossed with rolling windows.
- **Search**: `GridSearchCrossValidation`, `RandomisedSearchCrossValidation`.

**Fold Loop**
The one loop every cross-validation entry point runs. Per fold it resolves the estimator, views the data, swaps the Time-Dependent Inputs, and hands the callback one Fold.

**Fold**
The record the Fold Loop hands its callback: the fold's index and count, its already-resolved estimator, its already-viewed data, and its own training and test index vectors.

**Online Scheme**
A walk-forward wrapped in `Online`. Under it, the Fold Loop fits each fold by the online step in place of a refit. The loop warms up once on the first training window. It then folds the new observations of each fold into one estimator that it threads from fold to fold, and it reads that estimator out where a refit would have run. Only its function constructor builds it: `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit`. The constructor takes the keywords of the scheme minus its window parameter, and it sets that parameter to `true`, because a fold cannot un-fold an observation. So every nested-prefix scheme steps, and a rolling scheme cannot be written online. The estimator declares a window, through the cap of the `Online` wrapper on the prior. The scheme never declares it. ADR 0140, ADR 0167.
The wrapper is the same word that the estimator tree already uses. On a scheme it is not transient. The loop reads it at every fold, and every function that takes a scheme dispatches on its type. The loop starts cold, and a Time-Dependent Input may not reach a field that carries a Partial Fit State. ADR 0140, ADR 0167.
The only other entry is a Resume, which starts nothing cold. The loop skips the folds that the Result holds and continues from the next fold. ADR 0144.
*Avoid*: a Fold Fit, the retired name for the switch that a walk-forward once carried in a field. And `Online(cv)` written by hand, which raises an error that names it. And the `max_history` of the wrapper, which is a window on an estimator and not a keyword of a scheme.

**Resume**
The declaration that hands the Result of an Online Scheme back to the Fold Loop, so the evaluation continues where it stopped. The Result carries the threaded estimator, folded through the last training end, and the weights and held weights of the last fold. The loop takes the full history, extended. It skips every fold up to the one whose training window ends where the state stopped. It then takes up the next fold exactly as the one-shot run over the longer history would. A Resume is transient, in the manner of the `Online` wrapper on an estimator and of a Time-Dependent Input. At the entry of the loop it resolves to a copy of what the Result carries, and then it is gone. So the loop never writes the Result, and the Result resumes any number of times.
A resume needs timestamps. They are the only thing that ties the prefix of the history to what the state folded. A resume refuses a Result whose last fold was a partial window. Such a fold is the first half of a full fold of the longer run, and it is terminal. The live view of the leftover rows is a second, terminal resume of the same Result. ADR 0144.
The loop finds the fold to continue from by the last held timestamp of the state, not by the number of folds that the Result holds. So a resumed Result, which holds only the new folds, resumes again. `vcat` stacks a run and its resumes for scoring.
*Avoid*: reading it as a warm start. Under a resume the loop folds only the rows that the one-shot run would fold before the next fold, and it warms up on nothing. And `refit_last`, an exit that folds the purge and the test rows into the state. That state equals no fold of any run. A deployment is one manual step from the resumable state instead, in the value form `partial_fit(res.opt, rows)`. The bang form writes the held timestamps of the Result in place, and the Result then no longer resumes.

**Weight Drift**
The movement of the held weights of a fold away from its target weights, because each position grows at its own return under the self-financing recursion. When the drift is unset, a fold reports `X * w` on every observation, which is the quantity that the optimiser maximises.
An Online Portfolio Selection head never reads the drift of a fold. Its Online Update starts from the allocation that it chose. A block of rows drifts the target at the start of the block, while the recursion walks through allocations that the head never held. The only drift that the head reads is its own one-row form, the Price-Adjusted Allocation. ADR 0160.
*Avoid*: using it for the distance that a `TrackingError` or a `TurnoverRiskMeasure` bounds. That distance is a divergence between two portfolios. Weight Drift is the movement of the weights of one portfolio.

**Previous-Weights Source**
Which weights of the previous fold the Fold Loop threads to the next: its target weights, or its drifted weights after its last observation. It changes what `Turnover`, `TurnoverEstimator`, `WeightsTracking`, `TurnoverRiskMeasure` and the turnover fee measure, and it changes nothing else. It is a field of the two walk-forwards and `HindsightSplit` alone, because a scheme whose folds carry no history has no previous fold to read.
An unset source follows the Weight Drift. It is the target weights when the scheme drifts nothing, and the drifted weights when the scheme drifts. A fund that drifts holds the drifted weights at the end of the fold, and its next trade starts there. `TargetWeights` and `DriftedWeights` state the source explicitly, and an explicit source overrides the rule for the two mixed pairs. ADR 0110.
The previous fold is the last fold whose weights the source can thread, and it is not always the fold before. A fold whose solve failed has `NaN` target weights. So the target read skips it and reads the last solved fold. A source that reads the Held Weights reads those of the failed fold, which are finite, because a failed fold holds. ADR 0145.
For an Online Portfolio Selection head, the source reaches only the turnover fee and a `PreviousWeights` fallback. The update and the turnover ceiling of the Allocation Set read the Price-Adjusted Allocation and no threaded weight. A turnover fee on that head is measured against the held book, or against nothing that the head does. So the Fold Loop refuses it by name at its entry when no source is set. ADR 0160.
A frontier sweep gives no previous weights to thread. Its fold holds one portfolio for each sweep point, and each fold resolves its own span, so no one portfolio is the previous one. The Fold Loop refuses an estimator that sweeps a frontier and reads the previous weights, whichever source is set. ADR 0174.
*Avoid*: Turnover, which measures the trade a source implies and does not choose the source.

**Held Weights**
The record that a fold keeps of what the portfolio held, as opposed to its target: the asset returns that the fold scored, the weights that its drift started from, the weights after its last observation, and the Weight Drift form that produced them. The library rebuilds the weight path from that record on demand. It stores the path only when `store_weight_path` of the scheme asks for it.
On a solved fold the start weights are the target of the fold. On a failed fold they are the previous weights that the fold received. A fold that could not rebalance holds what it held, so its record is finite while its target and its series stay `NaN`. A failed fold with nothing to hold records `NaN` and drifts nothing. ADR 0145.
*Avoid*: using it for the target weights. Those are the decision, and the optimisation result holds them.

**Held Gap**
An observation and asset pair at which the weight of the portfolio is non-zero and the return of the asset is missing. A fold first views its test window at the Investable Mask. It then sets every missing return of the investable columns to zero, once, before it forms its Net Returns and before the Weight Drift compounds. So the series is always finite, and the missing weight is cash on that observation. A zero weight at a missing return gives no message. The strictness policy names a Held Gap, with a warning by default and a refusal under `strict`. See ADR 0118.
*Avoid*: a non-investable asset. That is a per-fit fact that the Investable Mask states, and the optimiser removes the asset when it reduces the problem. A Held Gap is per observation, and it happens where the universe changes after the fit.

**Drawn Plot**
A figure that draws a block of a prior result as it stands: a heatmap of `sigma`, a bar chart of `mu`, a heatmap of the loadings or of the coskewness. It keeps the **full** universe. The backend leaves a blank cell or a missing bar where an asset is not investable, so the blank records the gap. Its ranking and its colour limits read the finite entries alone, because `NaN` sorts first and turns every reduction into `NaN`. See ADR 0118.
*Avoid*: a Computed Plot (below), which cannot keep the frame. `plot_cokurtosis` is both, with one arity for each.

**Computed Plot**
A figure that computes a new quantity from a block of a prior result before it draws: an eigenspectrum, a phylogeny network, a centrality bar chart. `eigvals` refuses a `NaN`, and a plain moment estimator refuses a gapped sample, so such a figure reduces to the Investable Mask at its entry and draws the investable universe alone. Its docstring says that a non-investable asset is not drawn. See ADR 0118.
*Avoid*: a Drawn Plot (above). The two differ by whether the figure computes, not by whether it is a heatmap or a bar chart.

### 4.7 Finite Allocation (post-processing)

Finite Allocation turns continuous weights into whole shares for a fixed cash budget, because real markets have no fractional shares.

- **DiscreteAllocation**: MIP-based exact allocation.
- **GreedyAllocation**: heuristic greedy rounding.

**FiniteAllocationInput**
The problem data fed to a Finite Allocation optimiser: target weights, asset prices, cash budget, the cash held before the trade, and optional horizon and fees.
*Avoid*: FiniteAllocation (that is the family), AllocationProblem, AllocationInput.

**Allocation Fee**
A fee charged inside a Finite Allocation, on the money in each position rather than on a weight. The allocator holds the share counts and the prices, so `shares ⊙ prices` is that money exactly. The charge enters the budget constraint of each sub-problem and never its objective. So every unit of fee competes with a unit of position. The result carries the charge that it paid. The cash held before the trade is a field of the FiniteAllocationInput, because the turnover term prices the money traded. See ADR 0123.
*Avoid*: a fee taken out of the cash before the allocation runs, and any fee priced as a rate times a weight times a price, which names no quantity.

**Collateral Algorithm**
The rule that gives each side of a Finite Allocation its cash. The short side trades first. So the rule gives the short side its cash from the target weights and the cash. It gives the long side its cash from the money that the short side sold and the fee that the short side paid. It is a field of the FiniteAllocationInput, not of the allocator, so every allocator of a fallback chain reads one rule. **ProceedsCollateral** (the default) spends the short proceeds on the long side and bounds the net money of the book. **CashCollateral** gives the short proceeds to no side and bounds the gross money of the book by a collateral amount, the cash when none is stated. See ADR 0180.
*Avoid*: margin (a broker's requirement, which a Collateral Algorithm can model but does not name), budget (the sum of the target weights), and a rule whose sign changes with the budget.

## 5. Risk Measures

**Risk Measure**
A measure of portfolio risk. Legal usage splits the measures three ways. An **Optimisation** measure has a JuMP formulation, so it can be an objective or a constraint. A **Non-Optimisation** measure is for analysis only. A **Hierarchical** measure has no JuMP formulation, but it is valid as a clustering risk proxy.

**Risk Input Kind**
A classification independent of legal usage: what a measure reads when the library evaluates its expected risk. The three kinds are **net-returns**, **weights-returns-fees** and **weights-only**.

**Precomputed-returns contract**
The rule for the evaluation of a risk measure on a reduced net-return series that the caller holds, with no weights to apply. It is well defined only where the result is a function of the series alone, and only on a finite series. The library does not check the series, because every internal caller gives it a finite one. A tail measure on a gapped series gives a finite, wrong number. A caller who holds a gapped series compacts it first, which gives the drop-per-column answer. See ADR 0118.

**XatRisk**
The naming convention in which "X" stands for "Value" or "Drawdown". It is the same family, applied to returns or to drawdowns. *Relative* variants are the hierarchical drawdown forms. *Range* variants penalise the gap between the two tails. *Distributionally Robust (DR)* variants optimise against worst-case scenario distributions.

**Range Tails**
The two point measures whose sum is a Range variant: the **loss** tail on the net portfolio returns, and the **gain** tail on their negation at the second level of the range.

**Negated Upper Tail**
The sign convention that lets a Range be a sum. The gain tail is the base measure applied to the *negated* returns, so it is reported on the same sign convention as the loss tail and the two add rather than subtract.

**Significance Level**
The probability mass in one end of the return distribution that a tail measure prices: `alpha` in the loss end, `beta` in the gain end, both in `(0, 1)`. At sample length `T` a tail at `alpha` holds `ceil(alpha * T)` of the sample's scenarios. The two ends are the **tail**, which is the lower one, and the **head**, which is the upper one. A Calibration Slot (§3.9) names its own end, so one rule serves both.
Six tail measures also take `alpha = 0`: `ConditionalValueatRisk`, `EntropicValueatRisk`, `ConditionalDrawdownatRisk`, `EntropicDrawdownatRisk` and the two relative drawdown variants. Their tail then holds only the largest loss among the observations with positive weight, which is the weighted form of `WorstRealisation` and `MaximumDrawdown` (ADR 0202).
*Avoid*: Deformation Parameter (below), which shares the range `(0, 1)` and nothing else.

**Deformation Parameter**
The Kaniadakis `kappa` of a relativistic measure, in `(0, 1)`. It sets the shape of the deformed logarithm through which the measure prices its dual variable. It is not a probability, and it names no end of the distribution on its own. A `kappa` slot addresses the end that its sibling Significance Level addresses. That is why the Range form carries `kappa_a` beside `alpha` and `kappa_b` beside `beta`.
*Avoid*: Significance Level (above).

**Risk Series**
The per-observation series a conic tail measure reduces. Two exist: the net portfolio returns, and the negated drawdown path. Both are signed as returns, so a loss is a negative entry.

### Settings & combination

- **Risk Measure Settings**: the per-measure configuration of scaling, bounds and direction, in `RiskMeasureSettings`, `HierarchicalRiskMeasureSettings` and `MaxRiskMeasureSettings`.
- **Scalariser**: combines several risk measures into one scalar. The members are `SumScalariser`, `MaxScalariser`, `MinScalariser` and `LogSumExpScalariser`. There is no return-side counterpart, because several return terms are always a weighted sum.
- **Degeneracy Guard**: the refusal that an optimiser or objective makes when the expression that it reads is identically zero.
- **Combination Weight**: the weight that an element carries inside a combination of *several* elements. It has no effect on a lone element. The elements combined are risk measures, return terms, sub-portfolios, or whole portfolios.
- **Frontier / FrontierBoundEstimator**: the efficient-frontier point count and the bound transforms `LinearBound`, `SquareRootBound`, `SquaredBound`.
- **Frontier Sweep**: the sequence of solves a bound-swept optimiser runs. It is a product, so every swept entry on both axes joins the same enumeration.

### Dispersion / moment measures

- **Variance**, **StandardDeviation**, **UncertaintySetVariance** (robust).
- **LowOrderMoment** (`FirstLowerMoment`, `MeanAbsoluteDeviation` (MAD), `SecondMoment`, `EvenMoment`) and **HighOrderMoment** (`ThirdLowerMoment`, `FourthMoment`). With a variance estimator in `ve`, `HighOrderMoment` divides the moment by a power of the second moment of the same deviations, so it gives the standardised moment.
- **Lower Moment**: an odd moment is a lower moment only, so `FirstLowerMoment` and `ThirdLowerMoment` put "Lower" in the name and hold no field. The full odd moment can be negative, and a negative risk does not work in every optimisation. The lower moment reads only the deviations below the target, so it penalises the downside and stays non-negative. An even moment is never negative, so `SecondMoment`, `EvenMoment` and `FourthMoment` hold `FullMoment` or `SemiMoment` in the field `alg`. The name form and the field form are two concepts, not two spellings of one concept (#1879).
- **Kurtosis**, **Skewness**, **NegativeSkewness**, **VarianceSkewKurtosis** (composite), **ThirdCentralMoment**. `Kurtosis` and `Skewness` read a comoment matrix, `kt` and `sk`, as `Variance` reads `sigma`, so their functors give the value that their models bound. `ThirdCentralMoment` reads the returns, and with a variance estimator in `ve` it gives the standardised skewness.
- **MedianAbsoluteDeviation**: centred by `MedianCentering` or `MeanCentering`. It has no acronym. MAD and the factory alias `MAD` name the mean absolute deviation, a `LowOrderMoment` with `MeanAbsoluteDeviation`.

### Quantile / tail measures (XatRisk)

- **Value-at-Risk (VaR)**: `ValueatRisk` and `ValueatRiskRange`, and the drawdown forms `DrawdownatRisk` and `RelativeDrawdownatRisk`.
- **Conditional (CVaR / Expected Shortfall)**: `ConditionalValueatRisk`, its Range and DR forms, and the drawdown `ConditionalDrawdownatRisk` (CDaR).
- **Entropic (EVaR)**: `EntropicValueatRisk`, its Range form, and `EntropicDrawdownatRisk` (EDaR).
- **Relativistic (RLVaR)**: `RelativisticValueatRisk`, its Range form, and `RelativisticDrawdownatRisk` (RLDaR).
- **Power Norm**: `PowerNormValueatRisk` (PNVaR), its Range form, and `PowerNormDrawdownatRisk`.

**OWA (Ordered Weights Array)**
A weighted sum of *sorted* return realisations. An Algorithm that composes L-moments makes the weights. The sum can approximate CVaR and higher-moment criteria.
Three formulations optimise it. `ApproxOrderedWeightsArray`, the default, is a p-norm upper bound and the fastest. `ExactOrderedWeightsArray` is exact at `T²` rows, for a short sample. `SortingNetworkOrderedWeightsArray` is exact at `O(T log² T)` rows through a relaxed sorting network, for a long sample, and needs a nondecreasing weight vector. Its `merge` picks the network, and its `ordering` picks the order of the network variables.

### Drawdown measures

- **AverageDrawdown**, **UlcerIndex** (RMS drawdown), **MaximumDrawdown**. Each has a `Relative…` hierarchical form.

### Other measures

- **BrownianDistanceVariance (BDVar)**: distance-covariance-based dispersion.
- **WorstRealisation**: the worst single-period loss.
- **NoRisk**: contributes no risk term, so an optimiser that takes a risk measure can state a problem that has no risk.
- **Range**: the spread between the best and worst realisations.
- **TurnoverRiskMeasure**: turnover expressed as a risk quantity.
- **TrackingRiskMeasure**: benchmark deviation measured as a *norm* of the portfolio-versus-benchmark difference.
- **RiskTrackingRiskMeasure**: benchmark deviation measured through a configurable *risk measure* applied to that difference.
- **EqualRisk**: enforces equal risk contributions (hierarchical).
- **RiskRatio** / **NonOptimisationRiskRatio**: ratio-form measures for hierarchical use.

### Non-optimisation (analysis) measures

- **ExpectedReturn**: the scalar expected return of a portfolio for given weights. It is not the per-asset mean vector in the Prior.
- **MeanReturn**, **ThirdCentralMoment**: reporting statistics.
- **ExpectedReturnRiskRatio** / **MeanReturnRiskRatio**: a configurable return-to-risk ratio. The choice of risk measure gives the Sharpe, Calmar and Sortino ratios.

**Probabilistic Sharpe Ratio**
The probability that the true Sharpe ratio of a return series is greater than a benchmark Sharpe ratio, given the length of the series and the skewness and kurtosis of its returns. Every quantity is per period. `probabilistic_sharpe_ratio` computes it.

**Minimum Track Record Length**
The least number of observations at which the Probabilistic Sharpe Ratio of a series reaches a confidence level, for the sample moments of the series. It is a number of observations, and it is infinite when the Sharpe ratio does not exceed the benchmark. `minimum_track_record_length` computes it.

**Deflated Sharpe Ratio**
The Probabilistic Sharpe Ratio of the best of several trials, against the expected maximum Sharpe ratio of the search as its benchmark. It corrects the selected Sharpe ratio for the number and the dispersion of the trials. The count is of *independent* trials. Correlated trials count as fewer, and `implied_independent_trials` estimates their number. `deflated_sharpe_ratio` computes it, and `expected_max_sharpe_ratio` computes the benchmark.

## 6. Post-Processing

**Net Returns**
Portfolio returns adjusted for fees and turnover costs. The library computes them before drawdowns.

**Drawdowns**
Peak-to-trough declines computed from Net Returns, and the input series to drawdown-based risk measures.

**Fees**
The composite of holding and trading costs: **long** (management), **short** (borrowing, or a credit at a negative rate), **turnover** (commission) and **fixed** (a constant charge on any non-zero weight). The long, short and turnover rates are charged **every period**, at the frequency of the returns. The two fixed amounts are charged **one time** for the whole holding period. See issue #898. A return series reads the two fixed amounts as a fraction of capital, and an Allocation Fee reads them as a currency amount.
The short rate `s` may be negative down to `-l`, which keeps the fee convex. A negative rate credits a short position, and `s = -l` is the linear fee `l w` (#1518).

**Fee Amortisation**
The algorithm that spreads the one-off terms of a Fees object evenly over a holding period. The one-off terms are the two fixed charges and the fixed Forced Liquidation charge. It names a clock and carries no count. `nothing` charges those terms one time, on the first observation of the series. A stated algorithm spreads them over the observations that the charging site knows. Every site that charges a fee knows that count and passes it in, so no fee stores a count and no count goes stale. It reaches no other term, because the proportional and turnover rates are charged every period.
*Avoid*: fee smoothing, fee spreading, fee timing.

**Forced Liquidation**
The sale of a position in an asset that has left the Investable Mask. The optimiser does not choose the trade, so no turnover bound limits it. The trade is not free. A Fees object holds two liquidation charges. The proportional charge, `lq`, charges the rate times the absolute previous weight. The fixed charge, `flq`, charges its amount when the absolute previous weight is not zero. The proportional charge is a rate per period, and it is charged on every observation beside the turnover term. The fixed charge is a currency amount charged one time, on the clock that the Fee Amortisation names. The two charges hold the previous weights and the rates on the full universe until the optimiser reduces the fees at its entry. After the reduction they hold them on the complement of the Investable Mask, so a reduced Fees object carries two axes.
A fit charges it at its outer level only, never inside a cluster or an inner head. The realised series of the fold and the finite allocation charge it once, and the allocation charges it as an Allocation Fee. See ADR 0121 and 0123.
*Avoid*: Turnover, which is the trade that the optimiser chooses among investable assets. And a Held Gap, which is a missing return on an asset that the fold still holds.

**Finite Allocation**
The discretisation of weights into whole shares within a cash budget. See §4.7.

**Factor Attribution**
The decomposition of the volatility and the return of a portfolio into a systematic part, an idiosyncratic part and an Unattributed Remainder. It reads the factor-model block of a Prior Result, and it reports on three axes: by factor, by Factor Family and by asset. The **predicted** attribution reads the block's loadings, factor moments and idiosyncratic covariance. The **realised** attribution reads the factor-return, idiosyncratic-return and exposure histories against a realised net return series, and may roll over windows.
The realised attribution of a cross-validation matches each fold to the block row by row. The block records the rows of the data that its fit covers, and each fold records its test rows. The match is by timestamp when both carry timestamps, and by position otherwise. The attribution applies the exposure lag to the whole block before it cuts the block to the fold rows. So the first test row keeps the exposures of the row before it. ADR 0195.
Both factor model blocks record the rows. A `CrossSectionalFactorModel` records the rows of its fit, and a `Regression` records the observations of the Prior Result that holds it. A Pipeline fold records its test rows only when every fitted preprocessing step keeps the observations of the window, which the step states with `keeps_observations`. A refit over a capped buffer records no positions, because the buffer does not count the rows it dropped. ADR 0195.
*Avoid*: factor risk contribution (the Euler decomposition of any risk measure through a pseudo-inverse, which moves idiosyncratic risk into the factors), risk attribution, performance attribution (Brinson).

**Unattributed Remainder**
The part of a Factor Attribution that the factor model does not explain: what is left of the total after the systematic and idiosyncratic parts. On the realised side it holds fees, cash, weight drift inside a period, the exposure lag and the cross-sectional intercept share. On the predicted side it holds the gap between `mu` and `sigma` of the Prior Result and those of the block, which a wrapping Prior opens. So it is at rounding level on a plain fit. Its share of the variance is the check by which the reader finds whether the attribution means something. The library states its causes and never guards it.
*Avoid*: residual (taken by the idiosyncratic part), error term, alpha.

**Unknown Entry Rule**
What a Factor Attribution does with an entry of the factor model that the Prior Result does not state, such as the idiosyncratic variance of a held asset in the warm-up of its variance. `EntrywiseUnknown`, the default, keeps every entry that the prior states. It gives `NaN` for every number that reads an entry that the prior does not state. So the exposures and the systematic numbers of such a holding stay exact, while its total and every share of the portfolio volatility are unknown. `ZeroUnknown` reads as zero every unknown entry, and every absent return in the standalone moments of the asset axis. This understates the variance. An entry of an asset with a weight of zero adds nothing under either rule. ADR 0113, amendment of #1515.
The variance split of a held Leverage-One Pair is a second kind of unknown entry. `KindwiseUnknown(; unstated, leverage)` holds one rule for each kind, and each preset applies to both kinds. `EntrywiseUnknown` gives `NaN` for the split. `ZeroUnknown` reads the split as fitted and wholly systematic, as the oracle does. ADR 0113, amendment of #1579.
A held pair with no return at an active cell is a holiday, whose return is exactly zero, so the realised attribution fills it with no message. A held pair outside the active span is not a holiday. The attribution warns about it, or refuses it under `strict`.
*Avoid*: reading `ZeroUnknown` as the conservative choice. A zero variance is a lower bound, not an estimate. *Avoid*: an Investable Mask, which says which assets a prior estimated. This rule says what an attribution does with a holding that the mask left out.

**Factor Model Diagnostic**
A statistic that measures how well a fitted cross-sectional factor model describes the data of its fit. It reads the factor-model block of a Prior Result, and it never adds to the block. Each diagnostic is one verb that answers with a series or a table. The same verb answers over the bare histories that the block holds, so a caller with arrays and no block reads the same number. The diagnostics fall into three groups: those of the cross-sectional regression fit (the significance of each factor return, the collinearity and the conditioning of the exposures, the goodness of fit), those of the exposure history (the correlation between factors, the information coefficient, the stability, the dispersion and the coverage of each exposure), and those of the standardised idiosyncratic returns (the calibration, the tails, the shape, and the dependence of the residual on its forecast volatility). A plot of a diagnostic draws the answer of the verb and computes nothing of its own.
On a block, an answer with a factor axis is a `FactorDiagnosticResult` that carries the names and the Factor Family labels of its own axis, raw or reduced. `port_opt_view` selects its factors by position, by name, or by family with a `LabelGroup`. So a caller computes a diagnostic over every factor once and then selects the factors it reads, and the Factor Model Summary is selected the same way (#1403).
*Avoid*: factor model statistic, regression diagnostic (the group name, not the term), model check, plot-only diagnostic.

**Factor Model Summary**
The per-factor table of a fitted cross-sectional factor model. For every raw factor it gathers the annualised return, the annualised volatility, the Sharpe ratio and the first-order autocorrelation of the factor return. It also gathers the means over the history of the Factor Model Diagnostics of the factor for significance, collinearity, stability and coverage. It is one Result. The library builds it from the Factor Model Diagnostics, never from a computation of its own. The table keeps a factor that the Factor Family Basis drops from the regression. The four return statistics of such a factor read its return in `fr` of the block, which the constraint of its family fixes. They are `NaN` only on a block with no `fr`. Its three collinearity columns are `NaN`.
*Avoid*: factor summary table, diagnostics table, factor report, model summary.

## 7. Errors & Status

**PortfolioOptimisersError**
The base type of the library exceptions that argument validation raises.

**OptimisationReturnCode**
The status of an optimisation: `OptimisationSuccess` or `OptimisationFailure`.
