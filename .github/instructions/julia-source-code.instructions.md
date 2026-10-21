---
applyTo: "src/**/*.jl"
---

# Julia Source Code Guidelines for PortfolioOptimisers.jl

## Estimator, Algorithm, and Result Roles

These three abstract hierarchies form the backbone of the library. Understanding their distinct roles is critical for correct design.

- **Estimators** (`<: AbstractEstimator`):
  - User-facing. Compose algorithms and/or other estimators as fields.
  - Are the entry points for computation — all high-level API functions accept estimators.
  - May consume data (e.g., a returns matrix) and produce results or transformed data.
  - Example: `Covariance`, `EmpiricalPrior`, `Denoise`.

- **Algorithms** (`<: AbstractAlgorithm`):
  - Internal dispatch mechanism. Never called directly from user-facing APIs.
  - Modify or specialise the behaviour of an estimator they are stored in.
  - Must not contain data — only parameters that tune an algorithm's behaviour.
  - Example: `FullMoment`, `SemiMoment`, `SpectralDenoise`, `Newton`.

- **Results** (`<: AbstractResult`):
  - Returned by functions that consume estimators when the output is complex enough to warrant its own type (e.g., contains multiple arrays, metadata).
  - Can themselves be passed as inputs to further computations — functions must dispatch on both estimator and result types where this makes sense.
  - Example: `LowOrderPrior`, `Clusters`, `OptimisationResult`.

## Type Definitions

- **Abstract types**:

  - Always prefix with `Abstract` (e.g., `AbstractCovarianceEstimator`).
  - Include comprehensive docstrings explaining their role in the type hierarchy.
  - List related types in the `# Related` section.
  - When subtypes must implement specific methods, document this in an `# Interfaces` section (see the docstring guide).

- **Struct types**:

  - Use `@concrete` from `ConcreteStructs.jl` — it auto-generates type parameters so `struct MyType{T1, T2}` boilerplate is not needed.
  - Do not bound a type parameter in the header, neither as `struct MyType{T1 <: Real}` nor as a `@concrete` field `x <: Real`. Bound the argument of the inner constructor instead (see *Constructor Pattern*). A bound in the header is one that every signature naming the type must meet, and inference can fail or take very long on it. A struct that `@concrete` cannot write, because its parameters reach a parametric supertype or two fields share one, declares its parameters by hand with no bound. `test/test_75_concrete_struct_census.jl` gates the rule and holds each exception with its reason.
  - **The common instance of a struct wastes no padding (#1754).** Julia lays out the fields in the order of their declaration and aligns each field to the alignment of its type, so a narrow field before a wider one gets padding. A `Bool` before an `Int` or a pointer costs seven bytes.
    The common instance is the one that the keyword constructor builds from its defaults. Put its 8-byte fields first, and put its narrow fields together after them: `Bool`, `Int32`, `Float32`, a small isbits struct. Julia rounds the size of a struct up to its largest alignment, so a struct with one narrow field loses the same bytes in every order, and its narrow field can stand anywhere. A field that holds a singleton, such as `nothing`, takes no bytes and never breaks the alignment, so it can stand anywhere too.
    In the order that wastes no byte, keep a field next to the fields it relates to, then next to the fields of its category. The positional constructor follows the field order, and a new order of the fields is not a breaking change. `test/test_80_struct_layout_census.jl` gates the rule on every struct that has a constructor with no arguments.
  - Use `DocStringExtensions.TYPEDEF` in the docstring header for struct types.
  - All fields must be documented using inline `"$(field_dict[:key])"` strings and reflected in the `# Fields` section via `$(DocStringExtensions.FIELDS)`.
  - Call `@define_pretty_show(TypeName)` immediately after any new struct that should display nicely in the REPL (all estimators, algorithms, and results).
  - Every source file must end with an `export` statement listing all public symbols defined in that file.

## Capability Catalogue (required for every new public-facing addition)

Every concrete type on the Choice Surface, and every exported function, is listed in the **Capability Catalogue** — the user-facing inventory of what the package can do (see ADR 0040).

**When you add a new type or exported function, you must also place it in `docs/capability_catalogue.jl`.** This is not optional bookkeeping: it is enforced by `test/test_26_docs.jl`, which fails if any name on the Choice Surface is absent, and CI runs that test on every PR touching `src/`. A concrete type the package declares is on the surface when it is a leaf subtype of `AbstractEstimator`, of `AbstractAlgorithm` or of `AbstractCovarianceEstimator`, or when it is an export under its own name; a Result and an error are subtracted. `choice_surface_names` in `docs/generate_capability_catalogue.jl` states that rule once.

- **New type on the choice surface** — add a `Cap(:YourType)` to the group it belongs to. Pick the group by the *job it does*, not the file it lives in.
- **New type the library constructs for itself** — a marker or other internal type no caller ever writes is not a choice, so list it in `NOT_A_CHOICE` with the reason `:internal` instead of cataloguing it. It keeps its docstring and its API page.
- **New exported function** — either add a `Cap`, mention it in a section's `Prose` (a prose `@ref` counts as catalogued), or, if it is genuinely not a user-facing capability, list it in `NOT_A_FEATURE` with a reason: `:alias`, `:base_overload`, `:trait`, or `:internal`.
- **Do not write a description.** Each entry's one-line description is taken from the first sentence of its docstring at build time, so there is exactly one description of every type in the repo. Pass `label` only where the docstring genuinely reads worse in a bullet (for example when a group's children would all repeat the same prefix).
- **Removing an export or a type?** Also remove its `NOT_A_FEATURE` or `NOT_A_CHOICE` entry — both checks run in both directions and a stale exemption fails too.

Because descriptions come from docstrings, the docstring summary convention below is load-bearing for this page, not just for the API reference.

## Constructor Pattern

The library uses a strict **inner/outer constructor split** for all structs:

- **Inner constructor** (positional arguments, type assertions, validation):
  - Accepts concrete positional arguments (no keyword args).
  - Performs all validation using `@argcheck` or shared `assert_*` helpers.
  - Calls `new{typeof(arg1), typeof(arg2), ...}(arg1, arg2, ...)` explicitly.

- **Outer constructor** (keyword arguments, default values):
  - Accepts keyword arguments with default values.
  - Performs any additional validation that requires comparing multiple fields (and cannot be done in the inner constructor because the struct does not yet exist).
  - Delegates to the inner constructor.

```julia
@concrete struct MyEstimator <: AbstractEstimator
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:oow])"
    w
    function MyEstimator(ce::StatsBase.CovarianceEstimator, w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)      # shared validation helper
        return new{typeof(ce), typeof(w)}(ce, w)
    end
end
function MyEstimator(;
                     ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(),
                     w::Option{<:ObsWeights} = nothing)
    return MyEstimator(ce, w)
end
```

- **Prefer shared validation helpers** over inline `@argcheck` where they exist:
  - `assert_nonempty_nonneg_finite_val(x, sym)` — non-empty, non-negative, finite.
  - `assert_nonempty_gt0_finite_val(x, sym)` — non-empty, positive, finite.
  - `assert_nonempty_finite_val(x, sym)` — non-empty, finite.
  - `assert_matrix_issquare(A, sym)` — square matrix.

## Type Aliases and `Option{T}`

- **`Option{T}`**: Alias for `Union{Nothing, T}`. Use it for any field or argument that can be absent.

  ```julia
  function foo(x::Option{<:VecNum} = nothing) ...
  ```

- **Common type aliases** defined in `src/01_Base/08_TypeAliases.jl` — always prefer them over writing out the full union/abstract type:

  | Alias | Meaning |
  | --- | --- |
  | `VecNum` | `AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}` |
  | `MatNum` | `AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}` |
  | `ArrNum` | `AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}` |
  | `VecInt` | `AbstractVector{<:Integer}` |
  | `Num_VecNum` | `Union{<:Number, <:VecNum}` |
  | `VecNum_MatNum` | `Union{<:VecNum, <:MatNum}` |
  | `ObsWeights` | `StatsBase.AbstractWeights` |
  | `Option{T}` | `Union{Nothing, T}` |

## Numeric types come from the data

A numeric type is **derived**, never **coerced**. Read it off the arguments with `eltype`,
`typeof`, `real` and `promote_type`, and let the arithmetic widen it. The library must stay open to
number types it has never seen — an automatic-differentiation dual, a unit-carrying quantity, an
arbitrary-precision float — because each one arrives through a caller's array and never through a
change here. Every forced type closes the library to one of them.

```julia
Tf = eltype(X)                                  # the element type of one argument
Tf = promote_type(real(eltype(X)), typeof(p))   # the element type of several
Tf = float_if_integer(eltype(X))                # the same, when an integer must be repaired
```

**Repair an integer type with `float_if_integer`, and with nothing else.** An integer element
type cannot hold a mean, a variance, a weight or a `NaN`, so a site whose data can be an integer
must widen that one type. Take the type of the data, and pass it to `float_if_integer` in
[`src/02_Tools/03_TypeUtilities.jl`](../../src/02_Tools/03_TypeUtilities.jl). It returns the
floating-point type of `T` when `T <: Integer`, and `T` itself for every other type. So an
integer panel computes in `Float64`, a `Float32` panel stays in `Float32`, and a `Rational`
panel stays exact. That function holds the one call of `float` in `src/` and `ext/`.

**Do not repair an integer type with `float`.** `float(Int)` is `Float64`, so one integer
argument inside a `promote_type` widens the whole computation to `Float64` and a `Float32` caller
silently loses its precision. The wrapper also decides for every type that needed no repair: a
`Rational` argument becomes inexact, and a number type another package defines is converted to
whatever `float` says. `nextfloat` and `prevfloat` are different verbs. They step a value and
coerce nothing, so this rule does not reach them.

**Do not repair an integer type with the type of a division.** `typeof(one(T) / one(T))`,
`typeof(zero(T) / one(Int))` and `typeof(one(T) / 1)` are the same coercion in a different
spelling. A number type is free to define its division in a different type, so the spelling
decides for every type, and not for integers alone. Sessions wrote it to repair integer
returns after the census refused `float(`, and each such repair was reverted.

**Derive from the operation only when the operation leaves every type.** A square root of a
`Rational` is not a `Rational`, so a site that stores a square root takes
`typeof(sqrt(one(float_if_integer(T))))`. A site that answers `NaN` for the other branch of a
real quotient `a / b` takes `oftype(one(a) / one(b), NaN)`, because that value is a quotient.
Neither is a repair of an integer input.

**A concrete type is admitted only when the code really needs that type**, and then only when no
method converts internally. An index is the usual case.

- An index from a floating-point value takes the in-function conversion: `ceil(Int, x)`,
  `floor(Int, x)`, `round(Int, x)` and their siblings. Never `Int(ceil(x))`, which rounds in one
  type and then converts in another.
- An index or a count from integer arithmetic keeps `eltype`, and divides with `÷` rather than
  with `/` followed by a conversion.

**Keep a signature as wide as the body allows.** Prefer `VecNum`, `MatNum`, `ArrNum` and
`<:Number` over a narrower bound, so a caller may bring a number type from another package. Narrow
to `<:Real` only where the body needs an order, and to a concrete type only where the body needs
that exact representation.

## A divisor's correction is named by who makes it

The divisor of a spread is the number of observations, less a correction. Two keywords name that
correction, and each one names who makes it.

- **`ddof::Integer` where the library makes the correction itself.** The code subtracts the integer
  from the number of observations, `T - ddof`, and refuses a negative one. The members of
  `NormError` and `EvenMoment` take it.
- **`corrected::Bool` where `Statistics` or `StatsBase` makes it.** The code passes the flag on, to
  `Statistics.std`, `Statistics.var`, `StatsBase.SimpleCovariance` or `StatsBase.varcorrection`.
  `SimpleVariance`, `StdValue` and `VarValue` take it. Under observation weights the flag selects a
  divisor for each weight type, and no integer states that divisor.

One definition never names both, and a definition that names `ddof` never calls an upstream
spread. Such a site states an integer and applies a flag, so `ddof = 2` either raises or silently
means `ddof = 1`. To offer an integer, subtract it yourself. To delegate, take `corrected` and pass
it on.

A partial fit that reproduces a correction `StatsBase` makes on the batch path subtracts the flag
itself, as `n - corrected`. It keeps the name `corrected`, because it reads the field of an
estimator whose batch path delegates. One field cannot carry two names.

The default of each keyword is written at its site and is the site's own decision.
[ADR 0087](../../docs/adr/0087-one-written-bias-correction-default-not-the-upstream-one.md) fixes
`corrected = true`.

## A Returns Result drops no field that the mathematics lets it carry

Code that builds a `ReturnsResult` from another one, or from a buffer, a state or a window of one,
carries every field: `nx`, `X`, `nf`, `F`, `nb`, `B`, `ne`, `E`, `ts`, `iv`, `ivpa` and `pnl`. A
field that a consumer does not read today is carried too, because the next consumer cannot see
that it was dropped.

- **Drop a field only when it has no meaning on the new axis.** An asset-indexed field after a
  collapse to synthetic assets with no collapse rule is an example. State the reason in a comment
  or in the docstring at the site.
- **Subtract at the point of use rather than rebuild.** A flag that changes the returns a reader
  fits on, such as `brt`, never rebuilds the `ReturnsResult`. Each reader takes the flag as an
  argument and reads `benchmark_relative_returns(rd, brt)`, a matrix, so `rd` keeps its raw `X`
  and `B`, and no reader subtracts twice. A reader that a change adds takes `brt` too.
- **Refuse by name where a field cannot travel.** A fold that holds numbers alone refuses `iv` with
  an error that names it. A silent drop is never correct.
- **Refuse a flag that needs a field the data does not carry.** `brt = true` on returns with no `B`
  is an `ArgumentError`, not a pass-through.

[ADR 0204](../../docs/adr/0204-a-returns-result-drops-no-field-that-the-mathematics-lets-it-carry.md)
owns the rule.

## Union Type Aliases and Dispatch Groups

When multiple abstract subtypes share a common interface, define a `const` union alias and dispatch on it:

```julia
const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                  <:AbstractLowOrderPriorEstimator_AF}
```

This avoids duplicating method definitions. A union alias is a **dispatch alias**, and its docstring states what the alias groups and why the group exists. Which sections that docstring carries is stated by [`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) § *Section Structure for Aliases*, which is the Authority for it.

## Docstrings

[`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) is the Authority for every docstring rule. It states which sections each kind of unit carries, what each section holds, the dictionaries in `src/01_Base/01_DocstringDictionaries/` that a description interpolates from, the mathematical notation, and the `jldoctest` blocks. Read it before you write a docstring, and change a docstring rule there and nowhere else.

An **alias** is the case to check first. Its docstring carries a different set of sections from the unit it names, the set differs by kind of alias, and `test/test_26_docs.jl` reds the file over a section outside that set. See § *Section Structure for Aliases*.

## Immutability and `Accessors.jl`

- The library **never uses mutable structs**. All types are immutable.
- When a field of an existing struct instance must be changed, use `Accessors.jl`:

  ```julia
  using Accessors: @set
  new_obj = @set obj.field = new_value
  ```

- Functions must be as pure as possible — avoid side effects and global state.

## Input Validation

- **Always use `@argcheck` for validation**:

    ```julia
    @argcheck !isempty(v) IsEmptyError("v cannot be empty")
    @argcheck all(isfinite, v) DomainError(v, "`v` holds a non-finite entry, and the estimator reads every entry, so state a finite vector.")
    @argcheck size(A, 1) == size(B, 1) DimensionMismatch("A and B must have same rows")
    ```

- **Common validation patterns**:

  - Empty checks: `@argcheck !isempty(x) IsEmptyError(...)`
  - Nothing checks: `@argcheck !isnothing(x) IsNothingError(...)`
  - Finite checks: `@argcheck all(isfinite, x) DomainError(x, ...)`
  - Dimension checks: `@argcheck size(A) == size(B) DimensionMismatch(...)`

- **Prefer shared helpers over inline `@argcheck`** — see constructor section above.

### A `DomainError` carries the value and a message

A `DomainError` is raised in one shape: `DomainError(value, message)`. The `value` is the
quantity that is out of range, so a caller who catches the error reads it off `.val`; a relation
between two quantities carries the pair as a tuple. The `message` names the field the caller
wrote, the value it holds, what the code does with it, and the range it must lie in, so the
caller can act on the message alone.

```julia
@argcheck(fraction <= one(fraction),
          DomainError(fraction,
                      "`EffectiveAssetFloor.fraction` is $fraction, and a fraction of the universe is at most one. The rule multiplies it by the asset count to get the floor, so a fraction above one names a floor above the universe. State a value in `(0, 1]`."))
```

Two other spellings raise the same type and are refused:

- `DomainError("sentence")` puts the sentence in `.val` and leaves `.msg` empty, so `showerror`
  prints `DomainError with <sentence>:` and then nothing, and a caller who reads `.val` gets a
  string where a number was promised.
- A bare `DomainError` type makes `ArgCheck` write the message from the source expression, so
  the caller reads `one(fraction)` and `fraction <= one(fraction) must hold` where a field name
  and a range were owed.

A guard that fits one of the `assert_*` verbs of `src/01_Base/10_Assertions.jl` — `assert_gt0`,
`assert_nonneg`, `assert_finite`, `assert_unit_interval`, `assert_closed_unit_interval` and
their composites — is routed through that verb rather than written inline, because the verb
already carries the shape and names the symbol the caller passed.
`test/test_63_domain_error_shape_census.jl` gates the rule over `src/` and `ext/`.

## Multiple Dispatch

- **Prefer multiple dispatch over conditionals** for algorithm/method selection.
- Define specific methods for different type combinations rather than using if/else on types.
- Use abstract types and union aliases in method signatures for flexibility.
- Functions that accept either an estimator or a result must dispatch on both:

  ```julia
  function prior(pe::AbstractPriorEstimator, X::MatNum, ...) ... end
  function prior(pr::AbstractPriorResult, args...; kwargs...) = pr  # passthrough
  ```

### A branch that a type decides is a method

An early exit is the usual place where an `if` on a type hides: `isnothing(x) && return y`, or
`if !(r.sigma isa DeferredQuantity) return r end`, which more statements follow. When the
condition only tests the type of a positional argument, or of a field of one, the definition is
two cases, and each case is a method. Split it by dispatch on that type:

- a `::Nothing` method beside the method for a value. It types every other argument like the
  method for a value. An `::Any` there takes calls that the method for a value refuses, and an
  `args...` can make a call ambiguous;
- one method for each type that the `isa` tests;
- for a field, a method that forwards the field as an argument, and one method for each type
  of the field:

  ```julia
  set_sdp_variance_risk!(model, i, r, pr) = set_sdp_variance_risk!(model, i, r, pr, r.ratio_alg)
  function set_sdp_variance_risk!(model, i, r, pr, ::LinearRatio) ... end
  function set_sdp_variance_risk!(model, i, r, pr, ratio_alg::RatioFormulation) ... end
  ```

Prefer the third form, a method that forwards the field as an argument, to a method that bounds a
type parameter of the struct by its position, such as `Variance{<:Any, <:DeferredQuantity}`. The
position of a parameter moves when a field is added, and two such bounds can intersect at
`Union{}`, which JET reports (#1759). Run the JET gate, `code_health/jet.jl`, scoped to the branch
after a split.

When the type that the condition tests is an extension point, the method for the general case
takes the abstract type, and the method for the known case takes the concrete type. A user's
own subtype then reaches the general case, as it did through the branch. Above, a user's own
`RatioFormulation` implements only `sdp_objective_lift_flag`, which the general method reads.

The split changes the source, not the compiled code. The compiler knows the type of each
argument of a method, and it removes the dead arm of a branch on that type: under `LinearRatio`
the typed body of the early-return form of `set_sdp_variance_risk!` held no call to the lift of
the objective (#1752). The gain is that a method states one case, its signature names the case,
and the reader finds the case by dispatch, not inside the body of another case.

Inside a method, the type of each argument is known, and so is the type of a field that the
type of the argument carries as a parameter or as an entry of a `NamedTuple`. That holds also
when the caller holds a `Union`, because the call splits the union before it reaches the
method. Keep the branch when no dispatch can take it:

- a field that declares a `Union`, such as `SampleBufferState.G`, because one struct type holds
  a value of either type;
- a keyword argument, because a keyword does not dispatch;
- a local, or an argument that the body assigns again.

A condition that also tests a value, such as `isnothing(x) || isempty(x)`, is a run-time branch
and stays. An early exit on a value is not a breach of this rule. A mix of type tests that two
methods cannot state also stays, such as `isnothing(a) && !isa(b, T)` or
`isnothing(a) || isnothing(b)`: it needs a method for each combination, and more methods to
settle their ambiguities, which is harder to read than the branch.

`test/test_79_type_decided_early_return_census.jl` gates the rule over `src/` and `ext/`. Its
`RUN_TIME` list names each site of a field that declares a `Union`, with its reason, and every
other early exit on a type fails it.

## A unit that offers nothing but indirection is inlined

A unit is a dead-end when it has **one method**, a body that is **one depth-one forward**, and
**at most one reference** in `src/` and `ext/` outside its own definition (ADR 0168). A depth-one
forward is a bare argument, a literal, a field of an argument, or one call whose callee is a name
and whose arguments are each of those, a global name, a splat or a keyword forward. Such a unit
is a name between a caller and the expression it forwards to, and it is inlined at that caller. A
unit with no reference at all is dead and is deleted. The shape is one method, one caller, and a
body that only constructs the geometry token the caller could construct itself.

Everything wider earns its keep, because it reduces its caller's complexity, keeps the caller's
inferred type concrete, or names a rule the caller would otherwise restate:

- **Two methods are dispatch.** A `Nothing`/value pair and a two-family pair save the caller an
  `isa` branch. They are never candidates, whatever their bodies hold.
- **A decision, an operator, an index, a nested call, a comprehension or a string template is
  logic the unit owns.** A ternary that picks a formulation, a message template, an accessor
  that slices, and a helper that names one rule all stay.
- **A public forward is public API.** A user's call is a use the census cannot see, so it is
  flagged and kept by an allow-list entry, and the list shows every public forward the library
  carries.

The type-level case is an abstract type with one concrete subtype that nothing codes against: no
dispatch on it, no field bound, no union alias. It stays when it has a realistic future
application, and the allow-list is where that application is named.

`test/test_70_trivial_unit_census.jl` is the census and the allow-list. A flagged name outside
the list fails the build, and an entry whose name is no longer flagged is stale and fails too, so
the list can only shrink on its own.

## Branches that share an outcome are one branch

When two branches do the same thing, write one branch whose condition joins theirs. Each branch
is a path that the reader must follow and that a test must reach, so a chain of branches with one
outcome costs more to read and to test than one branch. Merge these four shapes:

- **Two exits in a row with the same body**, which ends in `return`, `continue`, `break`, `throw`,
  `error` or `rethrow`. Join the conditions with `||`. The form `c || return x` exits when `c` is false,
  so its condition is `!c`.

  ```julia
  if n < 2 ||
     !isfinite(m) ||
     (!isnothing(mt.min_obs) && n < mt.min_obs) ||
     (!isnothing(mt.min_periods) && T / window < mt.min_periods)
      return false
  end
  ```

- **Two adjacent arms of an `if` chain with the same body.** Join the conditions with `||`, as in
  `elseif (i == k && j == l) || (i == l && j == k)`.
- **An `if` with no `else` whose body is only an `if` with no `else`.** Join the conditions with
  `&&`.
- **An `else` whose body is only an `if`**, a ternary included. Write it as an `elseif`.

The merge costs nothing at run time. `||` and `&&` evaluate from left to right and stop at the
first operand that decides the result, so the merged condition tests the same conditions in the
same order.

Keep the branches apart in two cases, because the merge makes the text harder to read:

- A later exit whose condition calls a function whose name ends in `!`. In a merged
  `posdef_accepts(pdm.alg, X) || zero_variance_block!(f, X)`, the change of `X` happens only
  when the first test fails, and the reader of the condition does not see that.
- An `else` whose `if` tests the same condition as the last `if` of the first arm. The two arms
  are a table of two rows and two columns, and the nesting shows the table.

`test/test_81_one_outcome_branch_census.jl` gates the rule over `src/` and `ext/`. A site whose
merge makes the text harder to read for a different reason goes on its `KEEP` list with that
reason.

## Return Type Annotations

- See `.github/instructions/julia-return-types.instructions.md` for full guidelines.
- Key rule: annotate functions whose return type is always the same concrete type (e.g., `-> Nothing` for assertion helpers, `-> ConcreteEstimatorType` for factory methods).

## A problem is built in a fixed order

Build a solver problem, a stacked matrix, or a sum whose rounding the order can move from a sorted
key list, `sort!(collect(keys(d)))`, or from an ordered container. Do not take the iteration order
of a `Dict` or a `Set`: it follows the insertion history and the rehashes of the container, so one
view set can stack its blocks in two orders, and two orders give two problems whose solutions
differ in the last digits. A test that compares such a solution then passes alone and fails beside
other test sets. The entropy pooling solve stacks its view blocks by sorted key since #1751.

## Code Organization

- **File naming**: Source files are prefixed numerically to indicate load order (e.g., `src/01_Base/02_TypeRoots.jl`). A prefix is unique within its directory, and the `include` list of `src/PortfolioOptimisers.jl` is the listing of `src/` sorted by prefix, so the number on a file is the order it loads in. A new file takes the next free number of its directory, or renumbers the files that load after it. `docs/src/public_api/` and `docs/src/private_api/` are each numbered the same way, to read beside `src/`. `test/test_47_alias_and_module_census.jl` gates both claims (ADR 0147).
- **Letter parts**: A file over the size threshold of ADR 0101 that holds one subject in a logical order is cut into letter parts, which add the suffixes `_a`, `_b`, and so on to its name: `src/05_Moments/05_Gerber/03_GerberIQCovariance_a.jl` and `src/05_Moments/05_Gerber/03_GerberIQCovariance_b.jl` are the two parts of one file. The parts keep the number of the file, load in letter order, and count as one entry of the directory, so a run starts at `_a` and leaves no letter out. A cut falls before a documented definition, and each part takes as many definitions as fit under the threshold with room for its own `export` line. A file that mixes subjects is not cut this way: it becomes a directory of files, one subject to a file (ADR 0179). `test/test_47_alias_and_module_census.jl` gates the run (ADR 0178).
- **Directories**: A family of files under one subject is a directory, numbered as one entry of its parent, and its files are numbered inside it: `src/05_Moments/05_Gerber/` holds the three Gerber-family covariances, `src/17_Optimisation/05_JuMP/` everything that builds the JuMP model. A file whose subject already has a directory goes into it. A family of two or more files at the top of a directory that reads as one subject takes a directory of its own (ADR 0150). `docs/src/public_api/` and `docs/src/private_api/` each carry one page per source file at the same path, so a directory in `src/` is a directory of pages on each mirror (ADR 0128).
- **Module structure**: Each submodule focuses on a specific domain (moments, risk, priors, etc.).
- **Type hierarchy**: Subtype the appropriate abstract type (`AbstractEstimator`, `AbstractAlgorithm`, `AbstractResult`).
- **Exports**: Every source file ends with an `export` line listing all public symbols it defines. Do not export internal helpers.

## Composability

- Design estimators and algorithms to be composable.
- Accept other estimators/algorithms as parameters when appropriate.
- Return result types that encapsulate outcomes for easy chaining.
- Implement `factory(estimator, w::ObsWeights)` to propagate observation weights through composed estimators.
- Implement `port_opt_view(estimator, i)` and `obs_weights_view(estimator, i)` to support windowed or cross-validated slicing.

## Error Handling

- Use custom exception types (subtype `PortfolioOptimisersError`):

  - `IsEmptyError` for empty collections.
  - `IsNothingError` for unexpected `nothing` values.
  - `IsNonFiniteError` for non-finite numbers.
  - `DimensionMismatch` for size mismatches.
  - `DomainError` for out-of-range values.
