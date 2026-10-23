---
applyTo: "src/**/*.jl"
---

# Julia Source Code Guidelines for PortfolioOptimisers.jl

## Estimator, Algorithm, and Result Roles

These three abstract hierarchies are the base of the library. A correct design keeps their roles apart.

- **Estimators** (`<: AbstractEstimator`):
  - An estimator faces the user. It holds algorithms, other estimators, or both, as fields.
  - An estimator is the entry point of a computation. Every high-level API function accepts an estimator.
  - An estimator can read data, such as a returns matrix, and give results or transformed data.
  - Example: `Covariance`, `EmpiricalPrior`, `Denoise`.

- **Algorithms** (`<: AbstractAlgorithm`):
  - An algorithm is an internal dispatch mechanism. A user-facing API never calls it directly.
  - It changes or specialises the behaviour of the estimator that holds it.
  - It must not hold data. It holds only the parameters that tune its behaviour.
  - Example: `FullMoment`, `SemiMoment`, `SpectralDenoise`, `Newton`.

- **Results** (`<: AbstractResult`):
  - A function that reads an estimator returns a result when the output needs a type of its own, for example when it holds several arrays or metadata.
  - A result can be the input of a later computation. Where this makes sense, a function must dispatch on both the estimator type and the result type.
  - Example: `LowOrderPrior`, `Clusters`, `OptimisationResult`.

## Type Definitions

- **Abstract types.**

  - Always prefix the name with `Abstract`, as in `AbstractCovarianceEstimator`.
  - Give each one a docstring that explains its role in the type hierarchy.
  - List related types in the `# Related` section.
  - When a subtype must implement specific methods, state them in an `# Interfaces` section, as the docstring guide says.

- **Struct types.**

  - Use `@concrete` from `ConcreteStructs.jl`. It writes the type parameters, so you do not write the `struct MyType{T1, T2}` boilerplate.
  - Do not bound a type parameter in the header, neither as `struct MyType{T1 <: Real}` nor as a `@concrete` field `x <: Real`. Bound the argument of the inner constructor instead (see *Constructor Pattern*). A bound in the header is a bound that every signature that names the type must meet. Inference can fail on it, or take a long time. A struct that `@concrete` cannot write, because its parameters reach a parametric supertype or two fields share one, declares its parameters by hand with no bound. `test/test_75_concrete_struct_census.jl` gates the rule and holds each exception with its reason.
  - **The common instance of a struct wastes no padding (#1754).** Julia lays out the fields in the order of their declaration. It aligns each field to the alignment of its type, so a narrow field before a wider one gets padding. A `Bool` before an `Int` or a pointer costs seven bytes.
    The common instance is the one that the keyword constructor builds from its defaults. Put its 8-byte fields first, and put its narrow fields together after them: `Bool`, `Int32`, `Float32`, a small isbits struct. Julia rounds the size of a struct up to its largest alignment, so a struct with one narrow field loses the same bytes in every order, and its narrow field can go anywhere. A field that holds a singleton, such as `nothing`, takes no bytes and never breaks the alignment, so it can go anywhere too.
    In the order that wastes no byte, keep a field next to the fields it relates to, then next to the fields of its category. The positional constructor follows the field order, and a new order of the fields is not a breaking change. `test/test_80_struct_layout_census.jl` gates the rule on every struct that has a constructor with no arguments.
  - Use `DocStringExtensions.TYPEDEF` in the docstring header for struct types.
  - Document every field with an inline `"$(field_dict[:key])"` string. The `# Fields` section shows the fields through `$(DocStringExtensions.FIELDS)`.
  - Call `@define_pretty_show(TypeName)` directly after any new struct that the REPL must show in a readable form. Every estimator, algorithm and result is such a struct.
  - Every source file must end with an `export` statement that lists every public symbol that the file defines.

## Capability Catalogue (required for every new public-facing addition)

The **Capability Catalogue** lists every concrete type on the Choice Surface and every exported function. It is the user-facing inventory of what the package can do (see ADR 0040).

**When you add a new type or exported function, you must also place it in `docs/capability_catalogue.jl`.** This step is not optional. `test/test_26_docs.jl` fails if a name on the Choice Surface is absent, and CI runs that test on every PR that touches `src/`. A concrete type that the package declares is on the surface when it is a leaf subtype of `AbstractEstimator`, of `AbstractAlgorithm` or of `AbstractCovarianceEstimator`, or when it is an export under its own name. A Result and an error are not on the surface. `choice_surface_names` in `docs/generate_capability_catalogue.jl` states that rule once.

- **New type on the choice surface.** Add a `Cap(:YourType)` to its group. Pick the group by the *job it does*, not by the file that holds it.
- **New type that the library constructs for itself.** A marker or other internal type that no caller writes is not a choice. List it in `NOT_A_CHOICE` with the reason `:internal`, and do not catalogue it. It keeps its docstring and its API page.
- **New exported function.** Add a `Cap`, or name it in the `Prose` of a section, where a prose `@ref` counts as catalogued. If it is not a user-facing capability, list it in `NOT_A_FEATURE` with one of the reasons `:alias`, `:base_overload`, `:trait` or `:internal`.
- **Do not write a description.** At build time, the catalogue takes the one-line description of each entry from the first sentence of its docstring, so the repo holds exactly one description of each type. Pass `label` only where the docstring reads worse in a bullet, for example when all the children of a group repeat the same prefix.
- **An export or a type that you remove.** Remove its `NOT_A_FEATURE` or `NOT_A_CHOICE` entry too. Both checks run in both directions, so a stale exemption fails as well.

The descriptions come from the docstrings. So the convention for the summary line of a docstring decides the text of this page, as well as the text of the API reference.

## Constructor Pattern

The library uses a strict **inner/outer constructor split** for all structs:

- **Inner constructor** (positional arguments, type assertions, validation):
  - It takes concrete positional arguments and no keyword arguments.
  - It does all the validation, with `@argcheck` or the shared `assert_*` helpers.
  - It calls `new{typeof(arg1), typeof(arg2), ...}(arg1, arg2, ...)` explicitly.

- **Outer constructor** (keyword arguments, default values):
  - It takes keyword arguments with default values.
  - It does any more validation that compares several fields and that the inner constructor cannot do, because the struct does not exist yet.
  - It calls the inner constructor.

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
  - `assert_nonempty_nonneg_finite_val(x, sym)` checks for a non-empty, non-negative and finite value.
  - `assert_nonempty_gt0_finite_val(x, sym)` checks for a non-empty, positive and finite value.
  - `assert_nonempty_finite_val(x, sym)` checks for a non-empty and finite value.
  - `assert_matrix_issquare(A, sym)` checks for a square matrix.

## Type Aliases and `Option{T}`

- **`Option{T}`.** The alias for `Union{Nothing, T}`. Use it for any field or argument that can be absent.

  ```julia
  function foo(x::Option{<:VecNum} = nothing) ...
  ```

- **Common type aliases.** `src/01_Base/08_TypeAliases.jl` defines them. Always prefer an alias to the full union or abstract type that it names:

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
number types that it has never seen, such as an automatic-differentiation dual, a quantity with a
unit, or an arbitrary-precision float. Each one arrives through the array of a caller, and never
through a change here. Every forced type closes the library to one of them.

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
argument inside a `promote_type` widens the whole computation to `Float64`, and a `Float32` caller
loses its precision with no warning. The wrapper also decides for every type that needed no repair.
A `Rational` argument becomes inexact, and a number type that another package defines becomes
whatever `float` returns. `nextfloat` and `prevfloat` are different verbs. They step a value and
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

**Admit a concrete type only when the code needs that type**, and then only when no method
converts internally. An index is the usual case.

- An index from a floating-point value takes the in-function conversion: `ceil(Int, x)`,
  `floor(Int, x)`, `round(Int, x)` and their siblings. Never write `Int(ceil(x))`, which rounds in
  one type and then converts in another.
- An index or a count from integer arithmetic keeps `eltype`, and divides with `÷` rather than
  with `/` followed by a conversion.

**Keep a signature as wide as the body allows.** Prefer `VecNum`, `MatNum`, `ArrNum` and
`<:Number` over a narrower bound, so that a caller can bring a number type from another package. Narrow
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
spread. Such a site states an integer but applies a flag, so `ddof = 2` either raises an error or
means `ddof = 1` with no warning. To offer an integer, subtract it yourself. To delegate, take
`corrected` and pass it on.

A partial fit that reproduces a correction that `StatsBase` makes on the batch path subtracts the
flag itself, as `n - corrected`. It keeps the name `corrected`, because it reads the field of an
estimator whose batch path delegates. One field cannot carry two names.

Each site writes the default of its keyword, and that default is the decision of the site.
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
- **Subtract at the point of use rather than rebuild.** A field that changes the returns a reader
  fits on, such as `x_src`, never rebuilds the `ReturnsResult`. Each reader holds its own `x_src`
  field and reads `benchmark_relative_returns(rd, x_src)`, a matrix, so `rd` keeps its raw `X`
  and `B`, and no reader subtracts twice. A reader that a change adds holds an `x_src` field too
  ([ADR 0211](../../docs/adr/0211-each-reader-of-the-returns-states-its-own-source.md)).
- **Refuse by name where a field cannot travel.** A fold that holds numbers alone refuses `iv` with
  an error that names it. A silent drop is never correct.
- **Refuse a source that needs a field the data does not carry.** `x_src = ExcessSource()` on
  returns with no `B` is an `ArgumentError`, not a pass-through.

[ADR 0204](../../docs/adr/0204-a-returns-result-drops-no-field-that-the-mathematics-lets-it-carry.md)
owns the rule.

## Union Type Aliases and Dispatch Groups

When several abstract subtypes share an interface, define a `const` union alias and dispatch on it:

```julia
const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                  <:AbstractLowOrderPriorEstimator_AF}
```

The alias saves a copy of each method definition. A union alias is a **dispatch alias**, and its docstring states what the alias groups and why the group exists. [`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) § *Section Structure for Aliases* states which sections that docstring carries, and it is the Authority for that rule.

## Docstrings

[`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) is the Authority for every docstring rule. It states which sections each kind of unit carries, what each section holds, the dictionaries in `src/01_Base/01_DocstringDictionaries/` that a description interpolates from, the mathematical notation, and the `jldoctest` blocks. Read it before you write a docstring, and change a docstring rule there and nowhere else.

An **alias** is the case to check first. Its docstring carries a different set of sections from the unit that it names, and the set differs by kind of alias. `test/test_26_docs.jl` fails the file on a section outside that set. See § *Section Structure for Aliases*.

## Immutability and `Accessors.jl`

- The library **never uses mutable structs**. All types are immutable.
- To replace a field that the code names, call the keyword constructor, so every rule of
  construction runs again. `rebuild_estimator` calls it for any type. Guard the call with `===`, so
  that the function returns the object itself when nothing changed:

  ```julia
  new_obj = new_value === obj.field ? obj : rebuild_estimator(obj, (; field = new_value))
  ```

  A small type can call its own keyword constructor instead. The rule covers every form of
  `Accessors.jl` on a named field: `Accessors.@set`, `Accessors.@reset`, `Accessors.setproperties`,
  and `Accessors.set` with a literal lens such as `PropertyLens{:ucs}()`.
- Use `Accessors.jl` only when the code cannot know the field in advance. Tuning is the case: a key
  of a search names the field at run time, `parse_lens` turns the key into a lens, and
  `Accessors.set(opt, lens, val)` applies it. A lens whose field a type parameter names,
  `PropertyLens{target}()` in the pipe routing, is the same case.
- The sites in `src/` that predate this rule (#1806) are listed on #1807, which converts them. A new
  site follows the rule.

- Functions must be as pure as possible. Avoid side effects and global state.

## Input Validation

- **Always use `@argcheck` for validation**, as in these examples:

    ```julia
    @argcheck !isempty(v) IsEmptyError("v cannot be empty")
    @argcheck all(isfinite, v) DomainError(v, "`v` holds a non-finite entry, and the estimator reads every entry, so state a finite vector.")
    @argcheck size(A, 1) == size(B, 1) DimensionMismatch("A and B must have same rows")
    ```

- **Common validation patterns.**

  - Empty checks: `@argcheck !isempty(x) IsEmptyError(...)`
  - Nothing checks: `@argcheck !isnothing(x) IsNothingError(...)`
  - Finite checks: `@argcheck all(isfinite, x) DomainError(x, ...)`
  - Dimension checks: `@argcheck size(A) == size(B) DimensionMismatch(...)`

- **Prefer shared helpers over inline `@argcheck`.** See § *Constructor Pattern* above.

### A `DomainError` carries the value and a message

The library raises a `DomainError` in one shape, `DomainError(value, message)`. The `value` is the
quantity that is out of range, so a caller who catches the error reads it from `.val`. A relation
between two quantities carries the pair as a tuple. The `message` names the field that the caller
wrote, the value it holds, what the code does with it, and the range it must lie in, so the caller
can act on the message alone.

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
  the caller reads `one(fraction)` and `fraction <= one(fraction) must hold` where it needed a
  field name and a range.

Some guards fit one of the `assert_*` verbs of `src/01_Base/10_Assertions.jl`: `assert_gt0`,
`assert_nonneg`, `assert_finite`, `assert_unit_interval`, `assert_closed_unit_interval` and
their composites. Route such a guard through that verb, and do not write it inline. The verb
already carries the shape, and it names the symbol that the caller passed.
`test/test_63_domain_error_shape_census.jl` gates the rule over `src/` and `ext/`.

## Multiple Dispatch

- **Prefer multiple dispatch over conditionals** to select an algorithm or a method.
- Define one method for each combination of types, and do not branch with if/else on a type.
- Use abstract types and union aliases in method signatures, so that a method takes more types.
- Functions that accept either an estimator or a result must dispatch on both:

  ```julia
  function prior(pe::AbstractPriorEstimator, X::MatNum, ...) ... end
  function prior(pr::AbstractPriorResult, args...; kwargs...) = pr  # passthrough
  ```

### A branch that a type decides is a method

An `if` on a type is usually an early exit, such as `isnothing(x) && return y`, or
`if !(r.sigma isa DeferredQuantity) return r end` with more statements after it. When the
condition only tests the type of a positional argument, or of a field of one, the definition is
two cases, and each case is a method. Split it by dispatch on that type:

- A `::Nothing` method beside the method for a value. It types every other argument like the
  method for a value. An `::Any` there takes calls that the method for a value refuses, and an
  `args...` can make a call ambiguous.
- One method for each type that the `isa` tests.
- For a field, a method that forwards the field as an argument, and one method for each type
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
argument of a method, and it removes the dead arm of a branch on that type. Under `LinearRatio`,
the typed body of the early-return form of `set_sdp_variance_risk!` held no call to the lift of
the objective (#1752). The gain is that a method states one case, its signature names the case,
and the reader finds the case by dispatch, not inside the body of another case.

Inside a method, the type of each argument is known, and so is the type of a field that the
type of the argument carries as a parameter or as an entry of a `NamedTuple`. That holds also
when the caller holds a `Union`, because the call splits the union before it reaches the
method. Keep the branch when no dispatch can take it:

- A field that declares a `Union`, such as `SampleBufferState.G`, because one struct type holds
  a value of either type.
- A keyword argument, because a keyword does not dispatch.
- A local, or an argument that the body assigns again.

A condition that also tests a value, such as `isnothing(x) || isempty(x)`, is a run-time branch
and stays. An early exit on a value is not a breach of this rule. A mix of type tests that two
methods cannot state also stays, such as `isnothing(a) && !isa(b, T)` or
`isnothing(a) || isnothing(b)`. Such a mix needs a method for each combination, and more methods
to settle their ambiguities, which is harder to read than the branch.

`test/test_79_type_decided_early_return_census.jl` gates the rule over `src/` and `ext/`. Its
`RUN_TIME` list names each site of a field that declares a `Union`, with its reason, and every
other early exit on a type fails it.

## A unit that offers nothing but indirection is inlined

A unit is a dead-end when it has **one method**, a body that is **one depth-one forward**, and
**at most one reference** in `src/` and `ext/` outside its own definition (ADR 0168). A depth-one
forward is a bare argument, a literal, a field of an argument, or one call whose callee is a name
and whose arguments are each of those, a global name, a splat or a keyword forward. Such a unit
is only a name between a caller and the expression that it forwards to. Inline it at that caller.
A unit with no reference at all is dead, so delete it. The usual shape is one method, one caller,
and a body that only constructs the geometry token that the caller could construct itself.

A unit wider than that stays, because it reduces the complexity of its caller, keeps the inferred
type of the caller concrete, or names a rule that the caller would otherwise restate:

- **Two methods are dispatch.** A `Nothing`/value pair and a two-family pair save the caller an
  `isa` branch. They are never candidates, whatever their bodies hold.
- **A decision, an operator, an index, a nested call, a comprehension or a string template is
  logic the unit owns.** A ternary that picks a formulation, a message template, an accessor
  that slices, and a helper that names one rule all stay.
- **A public forward is public API.** The call of a user is a use that the census cannot see. So
  the census flags a public forward, an allow-list entry keeps it, and the list shows every public
  forward of the library.

The type-level case is an abstract type with one concrete subtype that nothing codes against: no
dispatch on it, no field bound, no union alias. It stays when it has a realistic future
application, and the allow-list is where that application is named.

`test/test_70_trivial_unit_census.jl` is the census and the allow-list. A flagged name outside
the list fails the build. An entry whose name is no longer flagged is stale and fails too, so the
list can only shrink.

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

- `.github/instructions/julia-return-types.instructions.md` holds the full guidelines.
- The key rule is to annotate a function whose return type is always the same concrete type, for example `-> Nothing` for an assertion helper and `-> ConcreteEstimatorType` for a factory method.

## A problem is built in a fixed order

Build a solver problem, a stacked matrix, or a sum whose rounding the order can move from a sorted
key list, `sort!(collect(keys(d)))`, or from an ordered container. Do not take the iteration order
of a `Dict` or a `Set`. That order follows the insertion history and the rehashes of the container.
So one view set can stack its blocks in two orders, and two orders give two problems whose
solutions differ in the last digits. A test that compares such a solution then passes alone and fails beside
other test sets. The entropy pooling solve stacks its view blocks by sorted key since #1751.

## Code Organization

- **File naming.** A numeric prefix on each source file states its load order, as in `src/01_Base/02_TypeRoots.jl`. A prefix is unique within its directory, and the `include` list of `src/PortfolioOptimisers.jl` is the listing of `src/` sorted by prefix, so the number on a file is the order it loads in. A new file takes the next free number of its directory, or renumbers the files that load after it. `docs/src/public_api/` and `docs/src/private_api/` are each numbered the same way, to read beside `src/`. `test/test_47_alias_and_module_census.jl` gates both claims (ADR 0147).
- **Letter parts.** Cut a file into letter parts when it is over the size threshold of ADR 0101 and holds one subject in a logical order. The parts add the suffixes `_a`, `_b`, and so on to its name. For example, `src/05_Moments/05_Gerber/03_GerberIQCovariance_a.jl` and `src/05_Moments/05_Gerber/03_GerberIQCovariance_b.jl` are the two parts of one file. The parts keep the number of the file, load in letter order, and count as one entry of the directory, so a run starts at `_a` and leaves no letter out. A cut falls before a documented definition, and each part takes as many definitions as fit under the threshold with room for its own `export` line. Do not cut a file that mixes subjects this way. It becomes a directory of files, one subject to a file (ADR 0179). `test/test_47_alias_and_module_census.jl` gates the run (ADR 0178).
- **Directories.** A family of files under one subject is a directory, numbered as one entry of its parent, and its files are numbered inside it. For example, `src/05_Moments/05_Gerber/` holds the three Gerber-family covariances, and `src/17_Optimisation/05_JuMP/` holds everything that builds the JuMP model. A file whose subject already has a directory goes into it. A family of two or more files at the top of a directory that reads as one subject takes a directory of its own (ADR 0150). `docs/src/public_api/` and `docs/src/private_api/` each carry one page per source file at the same path, so a directory in `src/` is a directory of pages on each mirror (ADR 0128).
- **Module structure.** Each submodule covers one domain, such as moments, risk or priors.
- **Type hierarchy.** Subtype the right abstract type: `AbstractEstimator`, `AbstractAlgorithm` or `AbstractResult`.
- **Exports.** Every source file ends with an `export` line that lists every public symbol that the file defines. Do not export internal helpers.

## Composability

- Design estimators and algorithms so that they compose.
- Accept other estimators and algorithms as parameters where it fits.
- Return a result type that holds the outcome, so that a later call can take it as input.
- Implement `factory(estimator, w::ObsWeights)` to propagate observation weights through composed estimators.
- Implement `port_opt_view(estimator, i)` and `obs_weights_view(estimator, i)` to support windowed or cross-validated slicing.

## Error Handling

- Use the custom exception types, each a subtype of `PortfolioOptimisersError`:

  - `IsEmptyError` for empty collections.
  - `IsNothingError` for unexpected `nothing` values.
  - `IsNonFiniteError` for non-finite numbers.

- Use the `Base` exception types for the other two cases. They are not subtypes of
  `PortfolioOptimisersError`:

  - `DimensionMismatch` for size mismatches.
  - `DomainError` for out-of-range values.
