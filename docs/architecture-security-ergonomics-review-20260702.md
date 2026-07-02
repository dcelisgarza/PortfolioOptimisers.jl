# Architecture, Security & Ergonomics Review — 2026-07-02

Scope: `src/` (91k LOC, 177 files), `ext/PortfolioOptimisersPlotsExt.jl`, with usage evidence
from `examples/*.jl`, `user_guide/*.jl`, and `test/`. Priority ordering, as requested:
**maintainability > robustness & correctness > testability > usability > performance.**

Vocabulary follows the review skills: a *module* is anything with an interface and an
implementation; *shallow* = interface nearly as complex as the implementation; a *seam* is
where an interface lives; *locality* = change and bugs concentrated in one place; the
*deletion test* asks whether deleting a module makes complexity vanish (pass-through) or
reappear in N callers (earning its keep). Ergonomics terms: *friction*, *signal*,
*convention test*, *default test*, *blast radius*.

## 0. State of play — what previous reviews already fixed

This codebase has absorbed an unusual amount of prior review output. Verified during this
pass (not re-suggested):

- **ADR 0008 is implemented.** `assemble_jump_model!`
  ([10_JuMPOptimiser.jl:1286](../src/20_Optimisation/10_JuMPOptimiser.jl#L1286)) is a single
  27-line assembly pipeline called by all five JuMP optimisers; the feared five copy-pasted
  `_optimise` sequences are gone. Only a ~5-line model-construction *head* is still repeated
  per optimiser.
- ADRs 0001–0015 (factory/view macros `@fprop`/`@vprop`/`@pprop`/`@cprop`/`@wprop`, typed
  model state, prefix-namespaced risk state, `risk_input_kind` trait, precomputed-returns
  contract, deep `JuMPOptimisationResult`, matrix-processing order symbols, forwarding
  macro, docs-by-pipeline-stage, suffix naming) are in force, with dedicated tests
  (`test_03b_jump_model_assembly.jl`, `test_09c_risk_input_kind.jl`, `test_27_prefix_registration.jl`,
  `test_28_seam_lock.jl`).
- The June-23 ergonomics candidates are shipped: literature acronym aliases
  (`src/24_Aliases.jl`), validator argument names, error-text fixes.

**Clean bills of health** (checked, no action — record so future reviews don't re-open):

- **Black-Litterman family** is genuinely deep: `bl_preroll`/`calc_omega`/`vanilla_posteriors`
  are shared by Factor/Augmented/Bayesian variants; validation centralised in `assert_bl`.
- **EntropyPooling** view handling reuses the shared equation parser; H0/H1/H2 are markers
  selecting staging, not duplicated paths; error messages carry remediation advice.
- **No type piracy**: all `Statistics.*`/`Base` overloads dispatch on package-owned types.
- **Threading** is `FLoops.@floop` over a configurable executor with disjoint writes; no
  shared-mutation hazards found.
- **RNG discipline**: every stochastic component takes `rng::AbstractRNG` + `seed`.
- **`[compat]` bounds** exist for every dependency.
- **Error quality is a caller-facing strength**: 800+ argcheck-family guards, typed
  `PortfolioOptimisersError` hierarchy, actionable messages. Preserve this bar in any
  refactor below.

**Known-open candidates from the 2026-06-17 architecture review** (still valid, folded into
the roadmap in §6, not re-derived): drawdown constraint trio behind one builder; one generic
`Windowed` wrapper instead of five; unified Coskewness/Cokurtosis container; tracking-norm
builders behind a cone seam; XatRisk variant-vs-formulation grid (`build_tail!`); plus the
memory-tracked future items `@risk_measure` declaration macro and Gerber marker fold-in.

---

## 1. Security

### S1 · Constraint equation parser can invoke any `Base` function — **Strong**

- **Files**: [12_ConstraintGeneration/02_LinearConstraintGeneration.jl:658-677](../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl#L658)
- **Problem**: `eval_numeric_functions` reduces numeric subexpressions of user-supplied
  constraint strings via `Base.invokelatest(getfield(Base, fname), args...)` — *any*
  function name in the `Base` namespace, with numeric arguments. A constraint string such
  as `"3 * exit(1) * AAPL <= 1"` terminates the host process. Constraint equations are
  exactly the kind of input a downstream app reads from config files or UI fields, so this
  is an unnecessary attack/robustness surface, and an unknown name produces a cryptic
  `UndefVarError`-shaped failure instead of a parse error.
- **Direction**: replace `getfield(Base, fname)` with a small allowlist
  (`+ - * / ^ sqrt exp log log2 log10 abs min max`); reject anything else with a
  `ParsingError` naming the offending call. One-line blast radius; no legitimate caller
  loses anything.

### S2 · `PythonCall` is a hard dependency for two leaf features — **Strong**

- **Files**: [08_Moments/10_Histogram.jl:198-252](../src/08_Moments/10_Histogram.jl#L198),
  [14_UncertaintySets/04_BootstrapUncertaintySets.jl:93-99](../src/14_UncertaintySets/04_BootstrapUncertaintySets.jl#L93),
  `Project.toml`, `CondaPkg.toml`
- **Problem**: every installation of the library drags in Python + CondaPkg + astropy +
  arch, to serve exactly two features: three histogram bin-width rules (`Knuth`,
  `FreedmanDiaconis`, `Scott` for `MutualInfoCovariance`) and the three block bootstraps of
  `ARCHUncertaintySet`. This is a supply-chain surface (a whole second package ecosystem
  inside the trust boundary), an install-failure mode on locked-down/air-gapped machines,
  and a latent runtime failure (`pyimport` at call time). Note `pyimport` also runs inside
  the hot function on every call.
- **Direction** (either is a win; both are breaking only for `Project.toml`):
  1. *Native reimplementation* — Freedman-Diaconis and Scott are one-line formulas;
     stationary/circular/moving block bootstraps are ~30 lines each; Knuth's rule is a 1-D
     optimisation over `Roots`/`Optim` (already dependencies). Preferred: kills the
     dependency entirely and makes the features testable.
  2. *Package extension* — move both behind a `PythonCallExt` weakdep with a clear
     "load PythonCall to enable X" error, mirroring the Plots extension pattern.

### S3 · Verified non-issues

The search-CV lens parser is literal-only (no runtime `eval`;
[09_Base_SearchCrossValidation.jl:271-309](../src/20_Optimisation/02_CrossValidation/09_Base_SearchCrossValidation.jl#L271));
no `Downloads`/`Serialization`/subprocess/`unsafe_*` surface exists in `src`/`ext`; the
top-level `eval(quote…)` blocks run at package load on package-owned types only (see M8).

---

## 2. Robustness & correctness

### R1 · Solver fallback chain breaks on missing primal; trial diagnostics overwritten — **Strong**

- **Files**: [08_Base_JuMPOptimisation.jl:699-739](../src/20_Optimisation/08_Base_JuMPOptimisation.jl#L699)
- **Problem**: after `JuMP.optimize!` succeeds without throwing, lines 717-719 call
  `JuMP.value.(model[:w])` *outside any `try`*. A solver that terminates cleanly but
  without a primal result (proven infeasible, unbounded, iteration limit) makes
  `JuMP.value` throw — so instead of recording the trial and moving to the next solver in
  the fallback chain, the whole `optimise` call aborts with an unhandled exception. The
  fallback chain is the documented contract of `Solver`/`JuMPOptimiser`. Separately, line
  730 unconditionally overwrites `trials[solver.name]`, destroying the
  `:assert_is_solved_and_feasible` error captured at line 727 — the diagnostic the user is
  told to inspect (`retcode.res`) loses the actual exception.
- **Direction**: move the `value` reads inside the guarded region (or behind
  `JuMP.has_values(model)`); merge trial records instead of replacing them. Small, pure
  correctness fix; add a test with a deliberately infeasible model and a two-solver chain.

### R2 · `robust_cov`/`robust_cor` swallow all exceptions, silently changing semantics — **Strong**

- **Files**: [08_Moments/01_Base_Moments.jl:586-633](../src/08_Moments/01_Base_Moments.jl#L586) (and `robust_cor` below)
- **Problem**: 16 bare `catch` clauses implement "try with kwargs → on *any* error retry
  without kwargs → on *any* error densify `X` and retry twice". A genuine numerical error
  in `cov` is masked and the call silently retried with the user's kwargs *dropped* —
  a different estimate than requested, with no signal. It is also untestable which branch
  ran. The intended clean solution — `hasmethod`-based dispatch — is already sketched in a
  comment block directly beneath (lines 602-608).
- **Direction**: promote the commented `hasmethod`/`applicable` version; if a fallback must
  remain, catch `MethodError` only.

### R3 · `set_risk_upper_bound!` catch-all silently drops user bounds — **Strong**

- **Files**: [20_RiskMeasureConstraints/01_BaseRiskConstraints.jl:174-176](../src/20_Optimisation/20_RiskMeasureConstraints/01_BaseRiskConstraints.jl#L174)
- **Problem**: `set_risk_upper_bound!(args...) = nothing` means a user who sets
  `settings.ub` on a risk measure used with any optimiser outside the `NonFRCJuMPOpt`
  union (e.g. `FactorRiskContribution`) gets the bound *silently ignored* — a
  wrong-portfolio-no-error path, the exact class ADR 0006 §2 was written to eliminate.
- **Direction**: make the fall-through throw (or at minimum `@warn`) when `ub !== nothing`;
  keep a typed no-op only for the cases where ignoring is semantically correct.

### R4 · Non-strict constraint parsing turns an asset-name typo into a zero row — **Strong**

- **Files**: [12_ConstraintGeneration/02_LinearConstraintGeneration.jl:399-413](../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl#L399) (`group_to_val!`), `collect_terms!`
- **Problem**: under the default non-strict path an unknown asset/group name in an equation
  only `@warn`s and yields a zero coefficient row — the constraint the user wrote quietly
  vanishes from the optimisation. Nonlinear terms (`w_A*w_B`) are not explicitly rejected;
  they fall through as an unrecognised variable string. Warnings scroll away; portfolios
  ship.
- **Direction**: default `strict = true` for unknown names (breaking, justified: silence
  here is a wrong portfolio); add an explicit "nonlinear term" `ParsingError` in
  `collect_terms!`.

### R5 · SCHRP mutates the shared covariance in place during recursion — **Worth exploring**

- **Files**: [06_SchurComplementHierarchicalRiskParity.jl:659-681](../src/20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl#L659)
- **Problem**: `schur_complement_weights` writes `sigma[lc,lc]`/`sigma[rc,rc]` into the
  matrix it recurses on, and scales `w` in place; correctness depends on exact recursion
  order. Any refactor that reorders traversal or shares `sigma` across branches corrupts
  results silently. The nearby `catch` also rethrows as a fresh `ArgumentError`, dropping
  the original exception as cause.
- **Direction**: pass sub-block copies down (allocation is trivial next to the solve);
  chain the original exception in the rethrow.

### R6 · Constructor validation is applied in roughly half the surface — **Worth exploring**

- **Files**: validator family at [01_Base.jl:2014-2205](../src/01_Base.jl#L2014); gaps across `08_Moments` and others
- **Problem**: the `assert_nonempty/finite/nonneg/gt0` helpers and typed errors are
  well-designed, but only ~57 of 69 statistics-layer files use any of them, against 124
  keyword constructors; many inner constructors are bare `new(...)`. A bad field surfaces
  as an opaque downstream error instead of a typed construction-time error — inconsistent
  guarantees across an otherwise uniform Estimator surface (convention-test failure for
  extension authors, too).
- **Direction**: fold "which asserts run" into the declaration macro of M5 where possible;
  otherwise adopt the checklist per constructor. Two micro-fixes found on the way:
  `@assert(a >= 0)` inside the public `GaussianDecay` doctest
  ([35_GerberIQCovariance.jl:301](../src/08_Moments/35_GerberIQCovariance.jl#L301)) teaches
  the anti-pattern (`@assert` is elidable; the library standard is `@argcheck`), and
  `val_dict` ([01_Base.jl:735](../src/01_Base.jl#L735)) is the only non-`const` of the four
  doc-interpolation globals.

### R7 · NOC is missing the `MinScalariser` reduction — **Strong** (symptom of M1)

- **Files**: [13_NearOptimalCentering.jl:366-470](../src/20_Optimisation/13_NearOptimalCentering.jl#L366)
- **Problem**: `near_optimal_centering_risks` implements Sum/LogSumExp/Max but not Min,
  while HRP/HERC implement all four — a divergence that exists *because* the reduction
  family is copy-pasted per optimiser (M1). A `MinScalariser` + multi-measure NOC call
  fails at dispatch rather than by design.
- **Direction**: fixing M1 removes the gap structurally; until then add the missing method
  or a typed "unsupported" error.

---

## 3. Maintainability (structural deepening)

### M1 · Extract the Scalariser reduction combinator (~23 re-encodings) — **Strong**

- **Files**: [07_HierarchicalEqualRiskContribution.jl:300-536](../src/20_Optimisation/07_HierarchicalEqualRiskContribution.jl#L300) (24 methods),
  [05_HierarchicalRiskParity.jl:329-424](../src/20_Optimisation/05_HierarchicalRiskParity.jl#L329) (4),
  [13_NearOptimalCentering.jl:366-470](../src/20_Optimisation/13_NearOptimalCentering.jl#L366) (3),
  plus dispatch sites in 10_JuMPOptimiser.jl and 11_MeanRisk.jl
- **Problem**: the four reduction bodies (Sum → accumulate; Max → `typemin`+compare; Min →
  `typemax`+compare; LogSumExp → collect + `logsumexp/gamma`) are byte-identical across
  every family; within HERC the 8-method `_o!`/`_i!` pairs differ by one `view` expression.
  ~23 dispatch methods encode a 3-line combinator. Deletion test: extract it and all of
  this collapses to one-liners; a LogSumExp normalisation bug currently needs ~6 fixes.
- **Direction**: one `scalarise(sca::Scalariser, itr; f)` (or do-block) reducer in the
  scalariser file; each optimiser supplies only its per-element risk closure. Also closes
  R7 and makes the reducers unit-testable without any optimiser (T2).

### M2 · Extract the Gerber-family co-movement kernel (~15 copies) — **Strong**

- **Files**: [06_SmythBrobyCovariance.jl](../src/08_Moments/06_SmythBrobyCovariance.jl#L497) (9 methods:
  lines 497, 581, 671, 754, 846, 945, 1032, 1118, 1208),
  [05_GerberCovariance.jl:292,378,463](../src/08_Moments/05_GerberCovariance.jl#L292),
  [35_GerberIQCovariance.jl:1649,1692,1738](../src/08_Moments/35_GerberIQCovariance.jl#L1649)
- **Problem**: the highest raw duplication in the repo. One `@floop for j / for i / for k`
  kernel — threshold guards, co-movement classification, symmetric
  `rho[j,i]=rho[i,j]=(pos-neg)/den` tail — reproduced ~15 times; variants differ only in a
  per-region weighting, a denominator policy, and Gerber2's diagonal standardisation.
  Any fix to the significance-zone logic is a 15-site change.
- **Direction**: one kernel function taking a per-observation *classification/weight
  policy* and a *denominator policy*; the existing marker types select policies. This is
  the enabling step for (and is more valuable than) the speculative marker fold-in already
  on file; it also gives the kernel a single test surface.

### M3 · Merge the `mip`/`smip` half-file mirror — **Strong**

- **Files**: [09_JuMPConstraints/05_MIPConstraints.jl](../src/20_Optimisation/09_JuMPConstraints/05_MIPConstraints.jl) (973 LOC)
- **Problem**: lines 22-482 (asset-space MIP: `mip_wb`, `short_mip_threshold_constraints`,
  `mip_constraints`, `set_mip_constraints!`) are mirrored at 484-973 with an `s`/`sg`
  prefix (subset-space). The two longest bodies (130/129 LOC) are near-line-identical.
  Every cardinality/threshold fix must land twice; locality is split by a prefix.
- **Direction**: parameterise the builders over key-prefix + bounds source (the
  prefix-registration machinery from ADR 0005 already exists for exactly this shape).

### M4 · One `assert_unit_interval` for the 25+ hand-copied `alpha`/`kappa` checks — **Strong**

- **Files**: `19_RiskMeasures/06,07,08,09,19_*.jl` (25 sites), drifted variant at
  [10_OWARiskMeasures.jl:967](../src/19_RiskMeasures/10_OWARiskMeasures.jl#L967)
- **Problem**: `zero(x) < x < one(x)` + `DomainError` message is copy-pasted 25+ times and
  has *already drifted once* (OWA's message shape differs). The matching docstring line
  `- \`0 < alpha < 1\`.` is duplicated in parallel. This is the same disease the
  `assert_nonempty_*` family already cures elsewhere.
- **Direction**: `assert_unit_interval(x, name)` next to its siblings in 01_Base.jl; a
  shared doc fragment via the existing `field_dict` mechanism. Mechanical, zero blast
  radius, and a prerequisite for the `@risk_measure` macro (M5).

### M5 · `@risk_measure` declaration macro (extends the tracked future item) — **Worth exploring**

- **Files**: [19_RiskMeasures/28_RiskMeasureTools.jl](../src/19_RiskMeasures/28_RiskMeasureTools.jl),
  [01_Base_RiskMeasures.jl](../src/19_RiskMeasures/01_Base_RiskMeasures.jl), all measure files
- **Problem**: adding a risk measure today touches: struct + inner validation (M4), keyword
  constructor whose kwarg names must equal field names, functor(s), `risk_input_kind`,
  `supports_precomputed_returns` behaviour, `@fprop`/`@vprop`/`@pprop` tags, a
  `set_risk_constraints!` builder, docs-table row, and export. Two of these are *implicit
  reflection contracts*: the `28_RiskMeasureTools.jl` codegen assumes a field literally
  named `settings` and field↔kwarg symmetry — violations fail at precompile (loud, but
  cryptic and far from the definition site).
- **Direction**: the already-tracked `@risk_measure` macro should declare, in one place:
  settings field, validated hyperparameter fields (emitting M4 asserts), input-kind trait,
  and the Value→Range→Drawdown expansion of the XatRisk grid (the range/drawdown variants
  are mechanical over the tail builder — see known-open grid candidate). The macro then
  *guarantees* the reflection contract instead of assuming it.

### M6 · Collapse the five shallow `*Result` wrappers — **Worth exploring**

- **Files**: [11_MeanRisk.jl:24](../src/20_Optimisation/11_MeanRisk.jl#L24),
  [12_FactorRiskContribution.jl:27](../src/20_Optimisation/12_FactorRiskContribution.jl#L27),
  [13_NearOptimalCentering.jl:55](../src/20_Optimisation/13_NearOptimalCentering.jl#L55),
  [14_RiskBudgeting.jl:103](../src/20_Optimisation/14_RiskBudgeting.jl#L103), RelaxedRiskBudgeting
- **Problem**: post-ADR-0011, the concrete JuMP result structs are down to the identical
  two fields `(jr, fb)` plus ~100-180 lines of constructor/docstring each — pure
  pass-throughs whose only job is a type name. Deletion test: replacing them with one
  parametric `RiskJuMPOptimisationResult{OE}` makes the complexity vanish; nothing
  dispatches on the concrete names (ADR 0011 confirmed this).
- **Problem to check first**: user-visible `show`/docs may want the friendly name — keep
  printable aliases if so. Completes ADR 0011 rather than contradicting it.

### M7 · Unify the two `optimise_JuMP_model!` solver loops — **Worth exploring**

- **Files**: [08_Base_JuMPOptimisation.jl:699](../src/20_Optimisation/08_Base_JuMPOptimisation.jl#L699) vs
  [10_JuMPModelOptimisation.jl:337](../src/10_JuMPModelOptimisation.jl#L337), consumer at
  [22_DiscreteFiniteAllocation.jl:325-330](../src/20_Optimisation/22_DiscreteFiniteAllocation.jl#L325)
- **Problem**: the solver-fallback loop (set optimizer → optimize → record trial → assert
  feasible) is encoded twice with incompatible return contracts (`(retcode, sol)` vs
  `.success`/`.trials`), and the allocation path hand-rewraps one into the other. R1's fix
  would otherwise land in only one of the two.
- **Direction**: one loop, one trial-record contract; do it together with R1.

### M8 · Replace load-time `eval(quote…)` codegen with generic methods — **Worth exploring**

- **Files**: [19_RiskMeasures/28_RiskMeasureTools.jl:1-75](../src/19_RiskMeasures/28_RiskMeasureTools.jl#L1),
  [09_JuMPConstraints/01_Returns_and_ObjectiveFunctions.jl:355](../src/20_Optimisation/09_JuMPConstraints/01_Returns_and_ObjectiveFunctions.jl#L355)
- **Problem**: four `eval` loops stamp out per-concrete-type methods of
  `no_bounds_risk_measure`/`bounds_risk_measure`/… whose bodies are *already fully generic*
  (they reflect `fieldnames(typeof(r))` at runtime); the returns-estimator loop generates
  exactly 2 methods. The generated methods are invisible to grep/`@edit`, docstrings attach
  awkwardly, and the mechanism is what makes the M5 reflection contract implicit. The
  package already depends on Accessors: each function is essentially
  `@set r.settings = RiskMeasureSettings(...)`.
- **Direction**: four ordinary generic methods on `::RiskMeasure` (+ the
  `HierarchicalRiskMeasure` no-op), via Accessors or `Setfield`-style reconstruction.
  Deletes ~70 lines of codegen and hundreds of compiled methods; behaviour identical.

### M9 · `posdef_estimator` accessor for the 21 four-level reaches — **Strong**

- **Files**: [14_UncertaintySets/03_NormalUncertaintySets.jl](../src/14_UncertaintySets/03_NormalUncertaintySets.jl)
  (21 × `posdef!(ue.pe.ce.mp.pdm, …)`), ~24 similar sites noted in ADR 0009 across priors
- **Problem**: NormalUncertaintySets is hard-coupled to the exact nesting `pe→ce→mp→pdm`;
  renaming any intermediate field breaks 21 sites. ADR 0009 explicitly preserved `mp.pdm`
  *because* of these reads — an accessor is the deferred half of that decision. The same
  file also has 12 copy-shaped `ucs`/`mu_ucs`/`sigma_ucs` methods recomputing the same
  sigma-assembly + posdef tail.
- **Direction**: `posdef_estimator(x)` (or `@forward_properties`) as the single path;
  factor the sigma-assembly tail so the 12 methods keep only their geometry-specific step.

### M10 · Shared CV `fit_and_predict` driver — **Worth exploring**

- **Files**: `20_Optimisation/02_CrossValidation/` — KFold(2), WalkForward(8),
  Combinatorial(5), MultipleRandomised(4), base(4) reimplementations; `Base.split` in
  [01_Base_CrossValidation.jl:35](../src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl#L35)
  spans ~223 lines
- **Problem**: each CV scheme re-encodes the train/test-index → fit → predict → score loop;
  the prediction/scoring invariant lives in N places, and a new scheme copies the largest
  one. The schemes' genuine content is index generation only.
- **Direction**: schemes expose a split iterator; one shared driver owns fit/predict/score.

### M11 · Meta-optimiser legality checks behind one seam — **Worth exploring**

- **Files**: [17_NestedClustered.jl:104-261,420-449](../src/20_Optimisation/17_NestedClustered.jl#L104)
  (14 `assert_*_optimiser` overloads, one spanning 114 lines),
  [18_Stacking.jl:249-276](../src/20_Optimisation/18_Stacking.jl#L249),
  [19_SubsetResampling.jl:281-294](../src/20_Optimisation/19_SubsetResampling.jl#L281)
- **Problem**: "is this optimiser legal as inner/outer" is re-encoded across three
  meta-optimiser files; adding an optimiser type means finding all three.
- **Direction**: one `assert_meta_optimiser(role, opt)` trait check in
  16_Base_MetaOptimisation.jl. Pairs naturally with U5 (introspection).

### M12 · Split NearOptimalCentering's private mini-pipeline — **Worth exploring**

- **Files**: [13_NearOptimalCentering.jl](../src/20_Optimisation/13_NearOptimalCentering.jl) (1157 LOC)
- **Problem**: two `_optimise` methods where the frontier path (line ~1042) builds and
  solves inline, *bypassing* `assemble_jump_model!` while the other path (line ~1093) uses
  it — the two heads diverge silently; six `solve_noc!` overloads and frontier machinery
  share the file with ordinary plumbing.
- **Direction**: route both heads through the shared assembly; extract the NOC solve/
  frontier machinery to a sub-file.

---

## 4. Testability

### T1 · Risk-constraint builders are untestable below a full solve — **Strong**

- **Files**: all of `20_RiskMeasureConstraints/` (54 `set_risk_constraints!` methods; zero
  direct tests — every test goes through `optimise` + a live solver)
- **Problem**: each builder is ~70% shared scaffolding (`get_constraint_scale` 57×,
  `set_risk_bounds_and_expression!` 46×, `preg!` 44×) around a ~10-line measure-specific
  cone — the only part worth testing, and the only part you *can't* reach without an
  optimiser + prior + solver. The interface is the test surface, and the current interface
  is "run the whole pipeline".
- **Direction**: a builder combinator that takes the cone closure and owns key-naming +
  bounds + return (this is the same seam as the known-open drawdown-trio and tracking-cone
  candidates — do them as one change). Cone closures then get unit tests against a bare
  `JuMP.Model` with MOI shape assertions, no solver. Roughly this converts the 43-minute
  solver-bound suite's marginal cost of a new measure into milliseconds.

### T2 · Expose the pure kernels as an internal testing surface — **Worth exploring**

- **Files**: scalariser reducers (M1); `schur_augmentation`/`symmetric_step_up_matrix`/
  `naive_portfolio_risk` ([06_…HRP.jl:522-603](../src/20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl#L522));
  `roundmult`; `combination_by_index`/`sample_unique_assets`; `expr_to_lens_chain`
- **Problem**: the deep numeric kernels are pure but unexported and undocumented as seams,
  so tests exercise them only through solver-requiring wrappers.
- **Direction**: mark them `public` (not exported), one docstring each, direct unit tests.
  Cheap; also improves AI-navigability.

### T3 · Long single-statement lines defeat review and diffs — **Speculative**

- **Files**: e.g. [08_EntropicXatRiskConstraints.jl:70](../src/20_Optimisation/20_RiskMeasureConstraints/08_EntropicXatRiskConstraints.jl#L70)
  (~600-char single statements assigning 3-6 model keys + a `@variables` block)
- **Direction**: falls out of T1's combinator; otherwise just formatting discipline.

### T4 · DBHT stage documentation — **Speculative**

- **Files**: [11_Phylogeny/04_DBHT.jl](../src/11_Phylogeny/04_DBHT.jl) (2076 LOC, well-decomposed,
  MATLAB-ported names: `clique3`, `BubbleCluster8s`, `HierarchyConstruct4s`…)
- **Direction**: no refactor; add stage-level docstrings + targeted tests for
  `PMFG_T2s`/`BubbleCluster8s` only if this area churns.

---

## 5. Usability / ergonomics

### U1 · Default solver + defaulted `check_sol`/`settings` — **Strong** (defaults)

- **Evidence**: `check_sol = (; allow_local = true, allow_almost = true)` appears **98×**
  across examples/user_guide; `settings = Dict("verbose" => false)` on nearly every
  `Solver`; the identical Clarabel block opens ~52 files
  (e.g. [examples/1_foundations/01_Getting_Started.jl:73-75](../examples/1_foundations/01_Getting_Started.jl#L73)).
- **Problem**: pure default-test failure — the 90th-percentile caller always passes these
  values, on the very first object a beginner must construct.
- **Direction**: make that `check_sol` the `Solver` default and default `name`; keep
  solver-module choice explicit (no hard Clarabel dep). The examples' preamble shrinks by
  ~3 lines everywhere.

### U2 · A high-level façade for the blessed path — **Strong** (discoverability)

- **Evidence**: `JuMPOptimiser(` 131×, `Solver(` 109×, `prices_to_returns`+`TimeArray` 102×
  across the corpus; every example runs data → `ReturnsResult` → `Solver` → `JuMPOptimiser`
  → optimiser → `optimise` before any content.
- **Problem**: ~8 lines of wiring precede the first result; the pipeline is the library's
  power, but the entry fee is paid by everyone including the person evaluating it for five
  minutes.
- **Direction**: an `optimise(MeanRisk(), rd; slv = …)`-shaped convenience (auto-wrapping
  the `JuMPOptimiser` with defaults), or a documented `quickstart()` recipe constructor.
  Keep it a thin sugar layer over the existing API — no new semantics.

### U3 · Naming: `opti`/`opto` and the `r` vs `rk` split — **Strong** (naming)

- **Evidence**: `Stacking(; opti = […], opto = MeanRisk(…))`
  ([examples/5_validation_tuning/02_Hyperparameter_Tuning.jl:69-72](../examples/5_validation_tuning/02_Hyperparameter_Tuning.jl#L69));
  `MeanRisk(; r = …)` 108× vs `MeanReturnRiskRatio(; rk = …)` 14×.
- **Problem**: `opt`/`opti`/`opto` are three collisions one letter apart in a single
  expression (convention-test failure; a swapped inner/outer is silent). `rk` breaks the
  learned `r =` pattern on a sibling type.
- **Direction**: rename to `inner`/`outer` (blast radius: Stacking/NCO call sites) and
  `rk` → `r`, with deprecation aliases for one release. Complements ADR 0015.

### U4 · `DiscreteAllocation` takes three loose positionals — **Worth exploring** (signal)

- **Evidence**: `optimise(da, res.w, vec(values(X[end])), 4206.9)`
  ([examples/1_foundations/01_Getting_Started.jl:142](../examples/1_foundations/01_Getting_Started.jl#L142))
- **Problem**: weights/prices/cash are order-remembered untyped positionals on the one
  function beginners run last (and the plot functions mix positional `rd` with keyword
  `pr =` in the same Getting-Started block — lines 114-119).
- **Direction**: keyword form `optimise(da; w, prices, cash)` (or a small input struct);
  unify plot signatures on positional `rd`.

### U5 · Programmatic measure↔optimiser compatibility — **Worth exploring** (discoverability)

- **Problem**: which risk measures a given optimiser accepts lives only in the docs table;
  composing an unusual pair means reading prose. (The *errors* on wrong pairs are already
  excellent — keep them.)
- **Direction**: `supported_risk_measures(::Type{<:Optimiser})` driven by the existing
  traits; generate the docs table from it so table and dispatch cannot drift. Pairs with
  M5/M11.

### U6 · Document the two conventions callers must infer — **Worth exploring** (docs)

- **Evidence**: ~15 two-letter field abbreviations (`pe` 341×, `slv` 383×, `ce`, `me`,
  `ve`, `mp`, `wb`, `cle`/`clr`…) whose `-e`-estimator/`-r`-result scheme is only
  discoverable by pattern-spotting; risk measures are functors while priors/optimisers use
  verbs (`prior`, `optimise`) — the "callable" signal breaks across families.
- **Direction**: a short "reading the API" legend in the user guide (one table for the
  abbreviation scheme, three sentences for functor-vs-verb). No code change.

---

## 6. Sequenced roadmap

Phases ordered by the requested priority weighting; items within a phase are independent.

**Phase 1 — correctness & security patches (small, high value, non-breaking except R4):**
S1 allowlist · R1+M7 solver loop fix/unification · R2 `robust_cov` · R3 `ub` fall-through ·
R4 strict parsing (breaking) · R7 missing MinScalariser (interim method) · R6 micro-fixes
(`val_dict` const, doctest `@assert`).

**Phase 2 — the three big deepenings (maintainability + testability):**

1. **T1 builder combinator** absorbing the known-open drawdown trio, tracking-norm cone,
   and XatRisk grid/`build_tail!` items — one seam, three prior candidates, plus unit-testable
   cones and the tail-sign convention centralised (currently hand-flipped in 6 files).
2. **M1 scalariser combinator** (closes R7 structurally).
3. **M2 Gerber kernel extraction**.

**Phase 3 — dependency & declaration hygiene:**
S2 PythonCall removal/extension · M4 `assert_unit_interval` · M8 de-codegen · M5
`@risk_measure` macro (after M4/M8 clear its path; subsumes the tracked future item) ·
M9 posdef accessor.

**Phase 4 — consolidations (breaking, schedule with a version bump):**
M3 mip/smip merge · M6 result-wrapper collapse · M10 CV driver · M11 meta-optimiser
legality seam · M12 NOC split · the remaining June-17 known-open items (Windowed wrapper,
Coskewness/Cokurtosis container).

**Phase 5 — ergonomics release:**
U1 solver defaults · U2 façade · U3 renames (with deprecations) · U4 allocation keywords ·
U5 `supported_risk_measures` · U6 legend · T2 public pure kernels.

## Top recommendation

**Phase 1 first — specifically R1 + R3 + R4 + S1 in one sitting.** They are the only
silent-wrong-answer / process-kill paths found, each is a sub-hour fix with an obvious
regression test, and none disturbs the architecture. Of the structural work, **the Phase-2
builder combinator (T1)** is the highest-leverage single change: it retires three
already-agreed open candidates, converts the least-testable 5,800 lines in the repo into
unit-testable cone closures, and centralises the duplicated tail-sign convention — the
classic deep-module win: more behaviour, smaller interface.
