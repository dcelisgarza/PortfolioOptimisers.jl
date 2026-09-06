# 863 — What JuMP, MathOptInterface and the solvers give for reusing a model across steps

Research ticket #863 of wayfinder map #861. Written 2026-09-06.

Sources are the JuMP manual (`jump.dev/JuMP.jl/stable`), the MathOptInterface manual and
reference, the ParametricOptInterface documentation and source, the MOI wrapper source of
Clarabel, HiGHS, SCS and Pajarito, the solvers' own documentation, and the library at the tip
of `dev`, `a003ca910b`. No Julia was run. A claim about the library cites a file and a line; a
claim about a document cites its page and section. A statement I could not source is marked as
an assumption, with what a later ticket must measure.

---

## Summary

- **A `Parameter` is a fixed variable unless the solver says otherwise, and none of the
  library's solvers says otherwise.** MOI lets a solver treat a `Parameter` variable as a
  constant, but Clarabel, HiGHS, SCS and Pajarito declare no support for it, so JuMP's bridge
  turns each parameter into a variable fixed by `EqualTo`. A parameter times a variable is a
  *quadratic* term to JuMP. ParametricOptInterface (POI) substitutes the value out and pushes a
  change to the inner solver with `MOI.modify`. The library already uses `Parameter` for the
  frontier sweep, without POI (section 1).
- **In-place modification reaches the solver only for HiGHS.** HiGHS declares the incremental
  interface and maps every `modify`, `delete` and row-bound change to one C call. Clarabel and
  SCS declare neither the incremental interface nor `modify`, so every modification lands in
  JuMP's cache and the next `optimize!` is a full `copy_to`: Clarabel constructs a new
  `Solver`, SCS a new workspace with a fresh KKT factorisation. The library sets the optimizer
  and solves afresh on every call of `optimise_JuMP_model!` (section 2).
- **A primal start is read by HiGHS, SCS and Pajarito, and dropped by Clarabel.** MOI's copy
  skips an unsupported start silently, so `set_start_value` costs nothing and buys nothing
  under Clarabel. For an interior-point method a start is not a basis: the literature finds a
  naive warm start from a previous optimum can cost more iterations than a cold start, and
  only a centred, shifted start pays (section 3).
- **A covariance that changes every entry rewrites `N(N+1)/2` coefficients in either form,
  and the conic form has no per-entry change object.** The quadratic form takes one
  `ScalarQuadraticCoefficientChange` per distinct pair; the cone row `[t; G w]` takes one
  `MultirowChange` per variable or a whole-function `set`. A parameter can stand in for an
  entry of `G` only through POI, and only in the objective for an entry of `Σ` (section 4).
- **A scenario row cannot be appended to a vector constraint, and every scenario coefficient
  moves when `T` moves.** The library writes each measure's scenario rows as one
  `VectorAffineFunction`-in-`Nonnegatives(T)`; MOI has no dimension-growing modification, so a
  new observation is a new variable, a new constraint, and `T+1` coefficient rewrites in the
  risk expression. Under HiGHS that is three C calls; under Clarabel and SCS it is a rebuild
  (section 5).
- **Fourteen of the library's risk families read the sample, six read a moment only, and two
  read neither.** The variance, kurtosis, skewness, distribution-VaR and the SDP moment stack
  read `sigma`, `kt`, `V`, `sk` or `mu` only; every quantile, drawdown, OWA, tracking and
  moment-deviation measure reads `X` one row per observation; `NoRisk` and the turnover measure
  read neither. The full table is section 6.
- **A kept model is already a Result — `JuMPOptimisationResult.model` — and a kept model that
  an Estimator holds between steps falls under ADR 0106.** It would have to be an
  `AbstractPartialFitState`, non-consumable, with a refusing `merge_states`, a `Base.copy`
  that `JuMP.copy_model` only half supplies, and a recorded shape it refuses to outgrow
  (section 7).

---

## 1. Parameters

### What `Parameter` and `set_parameter_value` do

- JuMP manual, *Variables › Parameters*: "Parameters are implemented as decision variables
  constrained to a `Parameter` set", read with `parameter_value` and written with
  `set_parameter_value`. "If the solver supports the `MOI.Parameter` set, it may decide to
  replace all instances of the parameter variable by the associated constant. If the solver
  does not support parameters, it will add the parameter as a decision variable with fixed
  bounds."
- MOI reference, *Standard form › `Parameter`*: "a variable constrained to the `Parameter` set
  cannot have other constraints added to it, and the `Parameter` set can never be deleted.
  Thus, solvers are free to treat the variable as a constant, and they need not add it as a
  decision variable to the model." The variable is added through `add_constrained_variable`,
  and a solver declares support with `supports_add_constrained_variable`, not
  `supports_constraint`.
- POI documentation, *How and why ParametricOptInterface is needed*: without POI, MOI's bridge
  rewrites `p in Parameter(1)` as `p in MOI.EqualTo(1.0)`, "a new decision variable with fixed
  bounds for every parameter in the problem".

### How a parameter times a variable is represented

- JuMP manual, *Variables › Parameters*: "JuMP treats a parameter multiplied by a decision
  variable as a quadratic expression, even though it is equivalent to a linear expression."
  A linear solver then fails on a linear programme that carries a multiplicative parameter,
  and the manual names POI as the workaround.
- POI, *background* page: the `EqualTo` bridge "cannot handle multiplicative
  parameter-variable terms", because "the resulting problem is a quadratic constraint".

### Whether a solver sees a constant or a fixed variable

None of the four MOI solvers in the library's environments declares
`supports_add_constrained_variable` for `Parameter`:

- Clarabel's `MOI_wrapper.jl` declares support for `VectorAffineFunction` and
  `VectorOfVariables` in its cone list and for an affine or (behind `use_quad_obj`) quadratic
  objective, and nothing else (its two `supports_constraint` and two `supports` methods, near
  lines 231–270).
- HiGHS's `MOI_wrapper.jl` declares `VariableIndex` bounds and integrality, and
  `ScalarAffineFunction` in the four scalar sets; no `Parameter`.
- SCS's and Pajarito's wrappers declare no `Parameter` support either (searched for the
  token; none).

So through every one of them a parameter is a fixed variable in the solver's model, and a
`set_parameter_value` is a bound change on that variable.

### What POI adds, and what it costs

- POI *background*: POI "substitutes out the parameters with their value before passing the
  constraint or objective to the inner optimizer", and on a change "efficiently modifies the
  inner optimizer to reflect the new parameter values".
- POI source `MOI_wrapper.jl`: `supports_add_constrained_variable(::Optimizer, Parameter)` is
  `true` (lines 694–698). `supports_constraint` for a `ScalarQuadraticFunction` or a
  `VectorQuadraticFunction` is answered by whether the inner solver supports the *affine*
  function in that set (lines 1065–1101), which is how a parameter-times-variable term in a
  cone row reaches a conic solver. `MOI.optimize!` calls `update_parameters!` when any
  parameter changed, then the inner `optimize!` (lines 2065–2075). A user `MOI.modify` on a
  parametric constraint is refused: "Parametric constraint cannot be modified in
  ParametricOptInterface, because it would conflict with parameter updates" (lines 1373–1387).
- POI source `update_parameters.jl`: the push uses `ScalarConstantChange` and
  `VectorConstantChange` for additive parameters, `ScalarCoefficientChange` for a parameter
  times a variable in a scalar function, and `MultirowChange` for one in a vector function
  (lines 139, 157, 188–198). Nothing replaces a whole function.
- POI *reference*, `Optimizer`: the wrapper takes `evaluate_duals` ("Set it to `false` to
  increase performance when the duals of parameters are not necessary") and
  `save_original_objective_and_constraints`, which "greatly increases the memory footprint";
  the constructor wraps an inner solver that does not support the incremental interface in
  a cache (`ParametricOptInterface.jl`, constructor).
- POI index page: parameters multiply quadratic variable terms "in objectives only", as
  `c * p * x * y`, so a parameter may scale a quadratic objective term but not a quadratic
  constraint term.

The cost is therefore one `MOI.modify` per changed coefficient or constant, issued to the
inner solver. What that buys depends entirely on the inner solver: under HiGHS the modify is
one C call; under Clarabel or SCS, which define no `modify`, the change lands in the cache and
the next solve is a full `copy_to` (section 2). Under those two, POI saves the Julia-side
rebuilding of expressions and nothing solver-side.

### What the library does today

- `set_ret_frontier_parameters!` and `set_risk_frontier_parameters!` create one
  `JuMP.Parameter` per swept bound and one bound constraint `sc * (expr - p * k) >= 0`
  (`src/20_Optimisation/08_Base_JuMPOptimisation.jl:1210-1219` and `1249-1259`).
- `set_frontier_point!` writes each point with `set_parameter_value`
  (`08_Base_JuMPOptimisation.jl:1320-1327`), and `frontier_sweep!` solves once per point
  through `optimise_JuMP_model!` (`08_Base_JuMPOptimisation.jl:1363-1376`). ADR 0062 records
  the seam. No POI is loaded; a commented-out `Solver` naming POI sits at
  `src/20_Optimisation/20_RiskMeasureConstraints/01_BaseRiskConstraints.jl:363-369`.
- `k` is the literal `1` under every head but `MaximumRatio`, where it is a variable
  (`src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl:1098-1105`).
  Under that head the bound `p * k` is a parameter times a variable, which JuMP represents as
  quadratic. Whether the bridged model is accepted by Clarabel in that case is not stated by
  any source I read; a later ticket must measure it.
- The scalar (non-swept) bounds are constants in the constraint, not parameters
  (`01_BaseRiskConstraints.jl:389-396`, `02_Returns_and_ObjectiveFunctions.jl:1178-1183`), so a
  step that moves a bound rewrites a constant, not a parameter.

## 2. In-place modification

### What JuMP and MOI offer

- JuMP manual, *Constraints › Modify a constant term* and *Modify a variable coefficient*:
  `set_normalized_rhs(con, v)`, `add_to_function_constant(con, v)`,
  `set_normalized_coefficient(con, x, v)`, the quadratic form
  `set_normalized_coefficient(con, x, x, v)`, and the vector form
  `set_normalized_coefficient(con, x, [(i, v), ...])`. *Delete a constraint*: `delete(model,
  con)` and `delete(model, vector_of_cons)`, then `unregister` to reuse a name.
- JuMP manual, *Objectives*: `set_objective_coefficient(model, x, v)` and
  `set_objective_coefficient(model, x, y, v)`, both with vectorised forms;
  `set_objective_function` replaces the whole objective.
- MOI manual, *Problem modification*: two categories, replacing a set or function with
  `set(ConstraintSet)` / `set(ConstraintFunction)` (same type only) or `transform` (new
  index), and changing a component in place with `modify` and one of `ScalarConstantChange`,
  `ScalarCoefficientChange`, `ScalarQuadraticCoefficientChange`, `VectorConstantChange`,
  `MultirowChange`. The vector method `modify(model, cis, changes)` applies many at once.
  The page warns: "Some `ModelLike` objects do not support problem modification."
- MOI reference, *Models*: `supports_incremental_interface(::ModelLike)` defaults to `false`
  (MOI source, `MathOptInterface.jl`). `copy_to` "empties" the destination and invalidates
  every prior index. The two-argument `optimize!(dest, src)` is the one-shot copy-then-solve
  a non-incremental solver implements.
- MOI *Utilities › CachingOptimizer*: three states, `NO_OPTIMIZER`, `EMPTY_OPTIMIZER`,
  `ATTACHED_OPTIMIZER`. In `AUTOMATIC` mode, "when an operation isn't supported by the current
  optimizer, it drops to `EMPTY_OPTIMIZER` and retries against the cache"; `optimize!` then
  re-attaches, which is a `copy_to`. JuMP manual, *Models*: `Model(Solver)` wraps the solver
  in this cache, and `direct_model` does not.

### Whether a modification keeps the solver's model

| Solver | Incremental interface | `modify` | `delete` | What a change costs |
|---|---|---|---|---|
| Clarabel | not declared (default `false`) | not defined | not defined | cache, then full `copy_to`: `dest.solver = Solver(P, q, A, b, cone_spec, settings)` |
| HiGHS | `true` (`MOI_wrapper.jl:3969`) | defined | defined | one C call per change |
| SCS | not declared | not defined | not defined | `optimize!(dest, src)` calls `MOI.empty!(dest)` and rebuilds; `scs_init` refactorises the KKT matrix |
| Pajarito | not declared in `MOI_wrapper.jl` | not defined | not seen | not determined (see below) |

Per solver:

- **Clarabel.** The wrapper's `copy_to` reads the whole source model, assembles `P, q, A, b`
  and the cone list, and constructs a new `Solver` object every time; `optimize!` calls
  `Clarabel.solve!` on it (`MOI_wrapper.jl`, lines 144–170 and 271–290). The native Julia
  solver *does* have in-place updates, `update_P!`, `update_q!`, `update_A!`, `update_b!` and
  `update_data!` (`src/data_updating.jl`), which "overwrite internal problem data structures
  in a solver object with new data, avoiding new memory allocations"; a matrix update must
  keep the sparsity pattern, and the Clarabel user guide, *Problem Data Updates*, refuses
  updates when `presolve_enable = true` or `chordal_decomposition_enable = true`. None of it
  is reachable through MOI, because the wrapper rebuilds the `Solver` on every `copy_to`.
- **HiGHS.** `modify` on the objective maps `ScalarConstantChange` to
  `Highs_changeObjectiveOffset`, `ScalarCoefficientChange` to `Highs_changeColCost`, and
  `ScalarQuadraticCoefficientChange` to `Highs_passHessian`, which re-passes the *whole*
  Hessian (lines 2042–2067); on a constraint, `ScalarCoefficientChange` maps to
  `Highs_changeCoeff` (line 3575); `set(ConstraintSet)` maps to `Highs_changeRowBounds`;
  `delete` maps to `Highs_deleteColsByRange` and `Highs_deleteRowsByRange` (lines 1768,
  3496); `add_constraint` for an affine row is `Highs_addRow`. The HiGHS guide, *Further
  features › Hot start*, adds that "if LP modifications are made via HiGHS methods, any basis
  stored internally will be modified to allow the best possible hot start." HiGHS accepts
  scalar affine rows and variable bounds only, no vector or cone function, and solves QPs by
  active set (HiGHS docs, front page).
- **SCS.** The wrapper implements the two-argument `optimize!(dest, src)`, which begins with
  `MOI.empty!(dest)`; every solve is a fresh problem. The SCS C API has `scs_update`, which
  "can reuse the SCS workspace in another solve if the only problem data that has changed are
  the `b` and `c` vectors", and `scs_init`, where "KKT matrix factorization is performed" (SCS
  docs, *C/C++ API*). The Julia wrapper does not expose `scs_update`.
- **Pajarito.** Its `MOI_wrapper.jl` contains no `supports_incremental_interface`, no
  `copy_to`, and no `modify` (searched by token); `MOI.empty!` and `MOI.optimize!` forward to
  `empty_optimize` and `optimize`. How the model reaches its OA and conic sub-solvers is in a
  file I could not fetch; a later ticket must read it. Pajarito is a mixed-integer conic
  outer-approximation solver over an `oa_solver` and a `conic_solver` (README), so at best it
  inherits each sub-solver's behaviour above.
- **Optim.** In `test/Project.toml` but not an MOI solver. The library uses it for the entropy
  pooling prior (`src/13_Prior/11_MeucciEntropyPoolingPrior.jl:469`), outside the JuMP model.

### What the library does today

`optimise_JuMP_model!` calls `JuMP.set_optimizer(model, solver.solver; add_bridges)` and then
`JuMP.optimize!(model)` for each solver in turn (`08_Base_JuMPOptimisation.jl:1752-1768`);
`set_optimizer` on a cached model resets it to `EMPTY_OPTIMIZER`, so every solve — including
every frontier point — copies the whole cache into a fresh solver instance. Every head starts
from `JuMP.Model()` (`src/20_Optimisation/11_MeanRisk.jl:766`), never `direct_model`.
`add_bridges` defaults to `true` (`src/10_JuMPModelOptimisation.jl:76`).

## 3. Warm starts

- JuMP manual, *Variables › Start values*: `set_start_value(x, v)` and `start_value(x)`;
  "Some solvers do not support start values. If a solver does not support start values, an
  `MathOptInterface.UnsupportedAttribute{MathOptInterface.VariablePrimalStart}` error will be
  thrown." *Constraints › Constraint start values*: `set_start_value(con, v)` for the
  constraint primal and `set_dual_start_value(con, v)` for the dual.
- MOI reference: `VariablePrimalStart` is "the initial assignment to the variable's primal
  value that the optimizer may use to warm-start the solve"; `ConstraintPrimalStart` and
  `ConstraintDualStart` are the same for the constraint's primal and dual.
- MOI source, `Utilities/copy.jl`, `pass_attributes`: when the destination does not support
  an attribute, `VariableName` and `VariablePrimalStart` are skipped, and so are
  `ConstraintName`, `ConstraintPrimalStart` and `ConstraintDualStart`, with the comment
  "Skipping names and start values is okay." So through the cache a start the solver cannot
  read is dropped silently; the JuMP error above applies in direct mode.

Which solvers read a start:

| Solver | `VariablePrimalStart` | `ConstraintPrimalStart` | `ConstraintDualStart` | Mechanism |
|---|---|---|---|---|
| Clarabel | no | no | no | none declared in the wrapper |
| HiGHS | yes (`MOI_wrapper.jl:1938-1945`) | no | no | `Highs_setSparseSolution` at `optimize!` (line 3368) |
| SCS | yes | yes | yes | `sol.primal`, `sol.slack`, `sol.dual` passed with `warm_start = true` |
| Pajarito | yes (`MOI_wrapper.jl:120`) | not seen | not seen | not determined |

What a start buys, per method:

- **Simplex (HiGHS, LP).** A basis hot-starts the simplex solver, and "presolve is not
  performed because it is not currently possible to solve a presolved LP using a basis for the
  original problem" (HiGHS guide, *Hot start*). A primal solution alone is what the MOI
  wrapper passes; the guide states that HiGHS will complete a partial basis. For a MIP a
  feasible assignment "will be used to provide the MIP solver with an initial primal bound".
- **ADMM (SCS).** The `warm_start` setting: "Set to True if you initialize the solver with a
  guess of the solution" (SCS docs, *Settings*); `scs_solve` "uses the entries of `sol` as
  warm-start for the solve" (*C/C++ API*). The iterates start from `(x, y, s)` and the
  factorisation is unchanged; a good guess cuts iterations, a poor one does not hurt the
  factorisation.
- **Interior point (Clarabel).** The wrapper declares no start, and the Clarabel pages I read
  state nothing about warm starting. The reason is in the method: an interior-point iterate
  must sit near the central path, and a previous optimum sits on the boundary. Yildirim and
  Wright (2002, *SIAM Journal on Optimization* 12, "Warm-start strategies in interior-point
  methods for linear programming") show that a start taken from the neighbouring solution
  needs to be shifted back toward the interior to save iterations at all, and John and
  Yildirim (2008, *Computational Optimization and Applications* 41) measure the gain at tens
  of percent of iterations on small perturbations and nothing on large ones. Skajaa, Andersen
  and Ye (2013, *Mathematical Programming Computation* 5) give the analogue for the
  homogeneous self-dual embedding Clarabel uses. So even if a start were passed, it would buy
  a fraction of the iterations and none of the setup: the KKT system is assembled and
  factorised per iteration regardless.

The library sets a primal start on `w` when `wi` is given (`08_Base_JuMPOptimisation.jl:1675-1679`
and `1703-1707`), and under Clarabel that start is dropped at `copy_to`.

## 4. The quadratic form

### How the library writes a covariance

- Cone form. `StandardDeviation` writes `[sc * sd; sc * G * w] in SecondOrderCone`
  (`20_RiskMeasureConstraints/02_VarianceConstraints.jl:177-180`), with `G` the upper Cholesky
  factor `cholesky(sigma).U` or the prior's `chol`, cached once as the shared `:G`
  expression (`02_VarianceConstraints.jl:22-28`, `51-59`).
- Quadratic form. `Variance` with `QuadRiskExpr` writes the *same* cone row **and**
  `dot(w, sigma, w)` (`02_VarianceConstraints.jl:365-376`); with `SquaredSOCRiskExpr` it writes
  the cone row and `dev^2` (`354-364`).
- SDP form. `tr(sigma * W)` over the lifted `W` (`346-353`); the factor-risk-contribution
  head writes `tr(b1' sigma b1 W)` (`590-603`).

### What changes when every entry of `Σ` changes

`Σ` is symmetric `N × N`, so it has `N(N+1)/2` distinct entries.

- **Quadratic form.** MOI stores each off-diagonal pair once: "the coefficient `b` in front
  of off-diagonal elements in `Q` should be left as `b`, because the mirrored index will be
  implicitly added", and a diagonal coefficient is doubled (MOI reference, *Standard form ›
  `ScalarQuadraticFunction`*). A full change is therefore `N(N+1)/2`
  `ScalarQuadraticCoefficientChange` objects, issued to the objective when the risk is
  scalarised into it or to the bound constraint when it is bounded (JuMP:
  `set_objective_coefficient(model, x, y, v)` or `set_normalized_coefficient(con, x, y, v)`).
  Under HiGHS each objective change re-passes the whole Hessian
  (`Highs_passHessian`), so the vectorised call is the one to use. Under Clarabel the
  quadratic objective is only accepted behind `use_quad_obj`, and a quadratic *constraint* is
  bridged to a cone before it reaches the solver.
- **Cone form.** `G` is upper triangular, so the row `G * w` carries `N(N+1)/2` nonzero
  coefficients in one `VectorAffineFunction` with `N + 1` outputs. MOI has no per-entry
  change for a vector function; the only in-place object is `MultirowChange`, "a change in
  the linear coefficients of a single variable in a vector-valued function", so a full change
  is `N` `MultirowChange` objects (JuMP: `set_normalized_coefficient(con, x, [(i, v), ...])`
  per variable) or one `set(ConstraintFunction)` with the whole new function. The count of
  coefficients rewritten is the same `N(N+1)/2`; the count of calls is `N` or `1`.
- **Both at once.** Every `Variance` builder writes the cone row, so a `QuadRiskExpr` change
  rewrites `N(N+1)/2` quadratic coefficients *and* `N(N+1)/2` cone coefficients. Note that
  `G` is a factor: a change to one entry of `Σ` moves, in general, every entry of `G` at and
  below the affected row, so there is no partial update of the cone row for a partial update
  of `Σ`. This is a property of the Cholesky factorisation, not of any source read here.
- **SDP form.** `tr(sigma * W)` has `N(N+1)/2` coefficients over the entries of `W`, one
  `ScalarCoefficientChange` each in the objective or bound.

Under Clarabel and SCS every one of those calls lands in the cache and the solve is a full
`copy_to` (section 2), so the coefficient count above is the cost of the JuMP-side edit, and
the solver-side cost is a rebuild regardless.

### Whether a parameter can stand in for an entry of `G` or `Σ`

- An entry of `G` multiplies a variable, `g_ij * w_j`. JuMP represents that as quadratic
  (*Variables › Parameters*), so the cone row becomes a `VectorQuadraticFunction`-in-
  `SecondOrderCone`. Clarabel accepts `VectorAffineFunction` and `VectorOfVariables` only, and
  MOI's `Parameter` bridge yields a fixed variable, not a constant, so the product stays
  bilinear. Without POI the model is refused or bridged into a form Clarabel does not accept;
  no source I read names a bridge that removes a bilinear parameter term.
- With POI it is accepted: `supports_constraint(VectorQuadraticFunction, S)` is answered by the
  inner solver's `VectorAffineFunction`-in-`S` support (POI `MOI_wrapper.jl:1084-1092`), and a
  change is pushed as `MultirowChange` per variable (`update_parameters.jl:157`). POI then
  holds `N(N+1)/2` parameter variables and a parametric copy of the function, with the
  memory note on `save_original_objective_and_constraints` above.
- An entry of `Σ` in the quadratic form multiplies two variables, `p_ij * w_i * w_j`, a cubic
  term. POI accepts it "in objectives only" (index page), so a parameter can stand in for an
  entry of `Σ` in a minimum-variance objective and not in a variance bound.
- A parameter cannot stand in for an entry of `Σ` in the SDP form `tr(Σ W)` other than as
  `p_ij * W_ij`, which is again parameter times variable and accepted through POI as a scalar
  `ScalarCoefficientChange` per entry.

## 5. The scenario measures

### What the library builds per observation

Every scenario measure reads `X` through `set_portfolio_returns!`, which registers the `T`
affine expressions `X * w` once per prefix (`08_Base_JuMPOptimisation.jl:1815-1821`), and
`set_net_portfolio_returns!`, which subtracts fees (`1843-1856`). On top of that, one build
per family:

| Family | Per-observation variables | Per-observation rows | Site |
|---|---|---|---|
| CVaR / CDaR | `z[1:T] >= 0` | one vector row `(z + series) .+ var >= 0`, `Nonnegatives(T)` | `07_ConditionalXatRiskConstraints.jl:113-131` |
| DRCVaR / DRCDaR | `s, tu, tv [1:T]`, `u, v [1:T, 1:N]` | two vector rows of `T`, `2T` `NormInfinityCone(1+N)`, two `T`-rows | `07_ConditionalXatRiskConstraints.jl:290-323` |
| EVaR / EDaR | `u[1:T]` | `T` `ExponentialCone`s, one budget row | `08_EntropicXatRiskConstraints.jl:107-129` |
| RLVaR / RLDaR | `omega, psi, theta, epsilon [1:T]` | `2T` `PowerCone`s, one `T`-row | `09_RelativisticXatRiskConstraints.jl:119-158` |
| PowerNorm VaR / DaR | `slack, v [1:T]` | `T` `PowerCone`s, one `T`-row, one budget row | `19_PowerNormXatRiskConstraints.jl:113-140` |
| MIP VaR / DaR | `z[1:T]` binary | one `T`-row, one cardinality row | `06_XatRiskConstraints.jl:132-152` |
| FLM / MAD | `flm` or `mad [1:T] >= 0` | one `T`-row | `03_MomentRiskMeasureConstraints.jl:118-128`, `172-182` |
| Second moment | `[1:T]` semi-slack when `SemiMoment` | one SOC over `T`, one `T`-row | `03_MomentRiskMeasureConstraints.jl:339-369` |
| Even moment | `u, t [1:T]` | `2T` `PowerCone`s | `03_MomentRiskMeasureConstraints.jl:443-485` |
| Drawdowns (ADD, UCI, MDD, and the DaR twins) | `dd[1:T+1]` | two `T`-rows and one scalar row | `01_BaseRiskConstraints.jl:534-549` |
| OWA exact / range | `owa, owa_a, owa_b [1:T]` | one `T`-row, one `T × T` `Nonpositives` block | `10_OWARiskMeasuresConstraints.jl:27-35`, `111-123` |
| OWA approx | `nu, eta [1:T]`, `epsilon, psi [1:T, 1:M]` | `T·M` `PowerCone`s, one `T`-row | `10_OWARiskMeasuresConstraints.jl:314-373` |
| Brownian distance variance | `Dt [1:T, 1:T]` symmetric | `T(T+1)/2` `NormOneCone(2)`s or two `T × T` blocks | `14_BrownianDistanceVarianceConstraints.jl:43-64`, `146-149` |
| Worst realisation / Range | one scalar | one `T`-row | `15_WorstRealisationConstraints.jl:38-52`, `16_RangeConstraints.jl:49-54` |
| Tracking (L1, L2, Lp, Linf) | `r_tr[1:T]` for Lp | one cone of dimension `1 + T`, or `T` `PowerCone`s | `18_TrackingRiskMeasureConstraints.jl:52-74`, `152-175`, `202-242`, `269-292` |

The risk expression's coefficients depend on `T` in every family: `var + sum(z) / (alpha T)`
for CVaR (`07:123-124`), `mean(flm)` for FLM (`03:122`), `t - z log(alpha T)` for EVaR
(`08:130`), `uci / sqrt(T)` (`12:54`), `t / T` and `t / sqrt(T - ddof)` for tracking
(`18:63`, `165`), and the observation-weight branches renormalise every coefficient when
`wi` changes (for example `07:126-127`).

### Whether a row can be appended, and at what cost

- JuMP builds `@constraint(model, A * x >= b)` without a dot as "a single constraint that is a
  `MOI.VectorAffineFunction` in `MOI.Nonnegatives`" (JuMP manual, *Constraints › Vectorized
  constraints*), which is the form every scenario builder above uses.
- MOI's set has a fixed dimension. The modification page offers `VectorConstantChange` and
  `MultirowChange` for a vector function, both over its existing outputs, and `transform`
  replaces the set with one of the same family; nothing grows a constraint. So a row is not
  appended to a vector constraint. A new observation is: one `add_variable` per per-observation
  variable (`z_{T+1}`), one `add_constraint` for its row (a scalar row, or a new one-output
  vector constraint), and `T + 1` `ScalarCoefficientChange`s on the risk expression wherever it
  was placed, the objective or a bound. A drawdown adds `dd_{T+2}` and two rows.
- Dropping the oldest observation is `delete(model, z_1)`: MOI states that "constraints
  containing the variable are either deleted or modified according to set support for
  dimension updates" (MOI reference, *Variables › `delete`*). Which sets declare that support
  I did not read; a later ticket must check `MOI.supports_dimension_update` for
  `Nonnegatives`, `Nonpositives` and `Zeros`, because the answer decides whether a rolling
  window can shrink a vector row or must rebuild it.
- The cost per solver follows section 2. HiGHS: `Highs_addRow`, `Highs_deleteRowsByRange`,
  `Highs_changeColCost` per coefficient, with the basis kept for a hot start; but HiGHS
  accepts scalar rows only, so JuMP's bridge layer must scalarise the vector row first, and
  whether `delete` and `modify` pass through that bridge without a rebuild of the bridged
  constraint is not stated by any source I read. Clarabel and SCS: a rebuild, whose cost is
  the symbolic and numeric setup of a problem with `T` more rows, so appending one row costs
  the same as rebuilding the model. The families that run on HiGHS are the LP-representable
  ones — CVaR, CDaR, MIP VaR, FLM, MAD, the drawdowns, OWA exact, worst realisation, range,
  L1 and Linf tracking; every cone family needs Clarabel or SCS.

## 6. What the library builds today

The head assembles one model per call: `JuMP.Model()`, `set_model_scales!`, the ratio
variable `k`, `set_w!` over `size(X, 2)` weights, the weight bounds, then
`assemble_jump_model!` (`11_MeanRisk.jl:766-775`; `10_JuMPOptimiser.jl:1653-1704` lists the
order: linear weight constraints, MIP and SMIP, turnover, tracking error, the three norm
ceilings, the four regularisations, fees, the risk measures, the return terms, integer and SDP
phylogeny, custom constraints). The scales `sc` and `so` are registered as constant
expressions (`08_Base_JuMPOptimisation.jl:526-532`).

One row per risk-measure family. "Moment" means the builder reads `mu`, `sigma`, `chol`,
`kt`, `sk`, `V` or `S2` of the prior and nothing per observation. "Sample" means it reads
`pr.X` (or the drawdown of `pr.X`) one row per observation, and `pr.w` when observation
weights are set. "Neither" means it reads only the measure's own fields.

| Family | Reads | Builder (file under `src/20_Optimisation/20_RiskMeasureConstraints/`) |
|---|---|---|
| `Variance` (SOC, squared SOC, quadratic, SDP) | moment: `sigma`, `chol`; `eltype(pr.X)` only | `02_VarianceConstraints.jl:549` (`set_risk_constraints!`), `354`, `365`, `346`; `G` at `22-28` |
| `Variance` under `FactorRiskContribution` | moment: `sigma`, and the loadings `b1` | `02_VarianceConstraints.jl:590` |
| `StandardDeviation` | moment: `sigma`, `chol` | `02_VarianceConstraints.jl:171`, `213` |
| `UncertaintySetVariance` | moment: `sigma`, plus a fitted set (`sigma_ucs`, which may read `rd`) | `02_VarianceConstraints.jl:810`; the four set forms `660`, `683`, `716`, `750` |
| `LowOrderMoment{FirstLowerMoment}` | sample: `pr.X`; moment: `pr.mu` as target | `03_MomentRiskMeasureConstraints.jl:108` |
| `LowOrderMoment{MeanAbsoluteDeviation}` | sample; `pr.mu` as target | `03_MomentRiskMeasureConstraints.jl:161` |
| `LowOrderMoment{SecondMoment}` | sample; `pr.mu` as target | `03_MomentRiskMeasureConstraints.jl:329` |
| `LowOrderMoment{EvenMoment}` | sample; `pr.mu` as target | `03_MomentRiskMeasureConstraints.jl:431` |
| `Kurtosis` (approximate, `N::Integer`) | moment: `kt`, `length(pr.mu)` | `04_KurtosisConstraints.jl:211`; eigen-split at `62-74` |
| `Kurtosis` (exact) | moment: `kt`, `S2`, `L2`, `chol_kt` | `04_KurtosisConstraints.jl:293`; factor at `23-39` |
| `NegativeSkewness` | moment: `V`, `sk` | `05_NegativeSkewnessConstraints.jl:151`; factor at `23-37` |
| `ValueatRisk{MIPValueatRisk}` | sample | `06_XatRiskConstraints.jl:67`, body `125` |
| `ValueatRisk{DistributionValueatRisk}` | moment: `mu`, `sigma`/`chol` | `06_XatRiskConstraints.jl:225` |
| `ValueatRiskRange` (MIP; distribution) | sample; moment | `06_XatRiskConstraints.jl:187`; `281` |
| `DrawdownatRisk` (MIP) | sample (drawdown) | `06_XatRiskConstraints.jl:340` |
| `ConditionalValueatRisk` | sample | `07_ConditionalXatRiskConstraints.jl:65`, body `109` |
| `ConditionalValueatRiskRange` | sample | `07_ConditionalXatRiskConstraints.jl:163` |
| `DistributionallyRobustConditionalValueatRisk` | sample: `pr.X` and `pr.X .+ 1`; `size(pr.X, 2)` | `07_ConditionalXatRiskConstraints.jl:210`, body `277` |
| `DistributionallyRobustConditionalValueatRiskRange` | sample | `07_ConditionalXatRiskConstraints.jl:365` |
| `ConditionalDrawdownatRisk` | sample (drawdown) | `07_ConditionalXatRiskConstraints.jl:406` |
| `DistributionallyRobustConditionalDrawdownatRisk` | sample (drawdown, `pr.X`) | `07_ConditionalXatRiskConstraints.jl:450` |
| `EntropicValueatRisk`, `…Range`, `EntropicDrawdownatRisk` | sample | `08_EntropicXatRiskConstraints.jl:59`, `162`, `194`; body `102` |
| `RelativisticValueatRisk`, `…Range`, `RelativisticDrawdownatRisk` | sample | `09_RelativisticXatRiskConstraints.jl:55`, `194`, `226`; body `106` |
| `OrderedWeightsArray` exact; range | sample | `10_OWARiskMeasuresConstraints.jl:101`; `187`; shared `owa` at `27` |
| `OrderedWeightsArray` approximate; range | sample | `10_OWARiskMeasuresConstraints.jl:299`; `411` |
| `AverageDrawdown` | sample (drawdown) | `11_AverageDrawdownConstraints.jl:42` |
| `UlcerIndex` | sample (drawdown) | `12_UlcerIndexConstraints.jl:43` |
| `MaximumDrawdown` | sample (drawdown) | `13_MaximumDrawdownConstraints.jl:42` |
| `BrownianDistanceVariance` | sample (`T × T`) | `14_BrownianDistanceVarianceConstraints.jl:137`; forms `43`, `54`, `90`, `97` |
| `WorstRealisation` | sample | `15_WorstRealisationConstraints.jl:77`; body `38` |
| `Range` | sample | `16_RangeConstraints.jl:44` |
| `NoRisk` | neither | `16_NoRiskConstraints.jl:27` |
| `TurnoverRiskMeasure` | neither: the measure's own `w` | `17_TurnoverRiskMeasureConstraints.jl:43` |
| `TrackingRiskMeasure{L1Norm}` | sample: `pr.X` and `tracking_benchmark(tr, X)` | `18_TrackingRiskMeasureConstraints.jl:52` |
| `TrackingRiskMeasure{L2Norm, SquaredL2Norm}` | sample | `18_TrackingRiskMeasureConstraints.jl:152`; forms `102`, `111` |
| `TrackingRiskMeasure{LpNorm}` | sample | `18_TrackingRiskMeasureConstraints.jl:202` |
| `TrackingRiskMeasure{LInfNorm}` | sample | `18_TrackingRiskMeasureConstraints.jl:269` |
| `RiskTrackingRiskMeasure{IndependentVariableTracking}` | what the nested measure reads | `18_TrackingRiskMeasureConstraints.jl:411` |
| `RiskTrackingRiskMeasure{DependentVariableTracking}` | sample: `expected_risk(…, pr.X, fees)` at `477`, plus the nested measure | `18_TrackingRiskMeasureConstraints.jl:465` |
| `PowerNormValueatRisk`, `…Range`, `PowerNormDrawdownatRisk` | sample | `19_PowerNormXatRiskConstraints.jl:61`, `176`, `207`; body `107` |
| `VarianceSkewKurtosis` | moment: `sigma`, `sk`, `kt`, `D2`, `L2`, `S2`; `size(pr.X, 2)` | `20_VarianceSkewKurtosisConstraints.jl:31` |
| `GenericValueatRiskRange` | what its two tails read | `21_GenericValueatRiskRangeConstraints.jl:29` |

The shared pieces the risk rows sit on:

| Piece | Reads | Site |
|---|---|---|
| `w` | `size(X, 2)` only; a primal start from `wi` | `08_Base_JuMPOptimisation.jl:1703`, `1675` |
| `X * w`, net returns | sample, `T × N` coefficients | `08_Base_JuMPOptimisation.jl:1815`, `1843` |
| drawdown path `dd` | sample | `01_BaseRiskConstraints.jl:534` |
| `ArithmeticReturn` | moment: `mu` (or the term's own) | `09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl:1504` |
| `ArithmeticReturn` with an uncertainty set | moment: `mu`, plus a fitted set | `02_Returns_and_ObjectiveFunctions.jl:1815` |
| `LogarithmicReturn` | sample: `T` exponential cones over `X * w` | `02_Returns_and_ObjectiveFunctions.jl:1831` |
| `NoReturn` | neither | `02_Returns_and_ObjectiveFunctions.jl:1860` |
| weight bounds, budgets, cardinality, thresholds, norms, regularisation, fixed fees | neither: `wb`, `opt.*` | `09_JuMPConstraints/04_WeightConstraints.jl:116`, `03_BudgetConstraints.jl:559`, `07_CardinalityConstraints.jl:31`, `08_ThresholdConstraints.jl:62`, `13_WeightNormConstraints.jl:63`, `12_RegularisationConstraints.jl:187` |
| turnover constraint and turnover fees | neither of the prior; the *previous weights* `tn.w`, a per-step datum | `09_TurnoverConstraints.jl:65-84`, `10_FeesConstraints.jl:79-84` |
| tracking error constraint | sample: `pr.X` | `11_TrackingErrorConstraints.jl:64-80`, `119`, `144`, `179`; risk form `225-235` |
| frontier bounds | a `Parameter` per swept bound | `08_Base_JuMPOptimisation.jl:1210`, `1249` |

Three facts the table implies for a step:

1. A moment-only model (variance, kurtosis, skewness, distribution VaR, the SDP stack, with
   an arithmetic return) has a fixed shape across steps: the row count depends on `N` only.
   Its step is a coefficient rewrite of the sizes in section 4.
2. A sample model's shape grows with `T`, and its risk expression's coefficients move with
   `T` (section 5). A rolling window of fixed `T` keeps the shape and rewrites every
   scenario coefficient: the `T × N` entries of `X * w`, plus the family's own rows.
3. The turnover pieces read the previous weights, which change every step whatever the
   measure, so no model with turnover is reusable without at least an `N`-coefficient
   rewrite of `tn.w * k`.

## 7. What a kept model would have to satisfy under ADR 0106

`JuMPOptimisationResult` already carries the model as data: its `model` field is
`Option{<:JuMP.Model}` (`src/20_Optimisation/10_JuMPOptimiser.jl:313`, `324-339`), filled when
`save = true` (`11_MeanRisk.jl:780-783`). A model kept *between* steps is a Result an
Estimator would hold, and `CLAUDE.md` refuses that except through ADR 0106, so the decision
that reads this report has to place it there. ADR 0106 admits one shape: a field bound to
`Union{Nothing, <:AbstractPartialFitState}`, defaulting to `nothing`, whose value is not
consumable — "no consumer of a moment, a prior or a risk measure reads a state", and a
read-out verb turns it into the ordinary Result first — and which answers `merge_states`
(the fold, or a refusal naming the reason) and `Base.copy` with no aliasing. A kept model
would therefore have to be a struct under that root; be read by nothing but the step verb
that turns it into weights; refuse `merge_states`, since two JuMP models are not a sum; and
supply a `Base.copy`. `JuMP.copy_model` copies "the MOI model in the cache" and the object
dictionary (so the Model State registry of ADR 0004 and 0037 survives through the returned
reference map), but "the new model will have no optimizer", so the copy must re-attach one,
and it refuses a `direct_model` ("Cannot copy a model in `DIRECT` mode"), which rules out the
one JuMP mode in which HiGHS's incremental interface avoids the cache. It would also have to
record the shape it was built for — `N`, `T`, the measure set, and whether turnover is
present — and refuse a step whose shape differs, because every in-place path above keeps the
shape (`Parameter` values, `modify`, and Clarabel's own `update_data!`, which demands an
unchanged sparsity pattern), and a shape change is a rebuild by every route.

## 8. What a later ticket must measure

- Whether the frontier bound `p * k` under `MaximumRatio` is accepted by Clarabel through
  JuMP's bridges, since JuMP represents it as quadratic (section 1).
- The route by which Pajarito loads and reloads its sub-solvers (section 2).
- `MOI.supports_dimension_update` for the sets the scenario rows use, and whether `modify`
  and `delete` pass through the scalarising bridge HiGHS needs for a vector row without a
  rebuild of the bridged constraint (section 5).
- The wall-clock split, for one representative moment model and one representative sample
  model under Clarabel, between the JuMP-side assembly (`assemble_jump_model!`) and the
  `copy_to` plus solve, since under Clarabel that split is the whole of what any reuse can
  save.
