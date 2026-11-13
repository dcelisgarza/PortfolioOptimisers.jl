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
  - Do not bound a type parameter in the header, neither as `struct MyType{T1 <: Real}` nor as a `@concrete` field `x <: Real`. Bound the argument of the inner constructor instead (see *Constructor Pattern*). A bound in the header is a bound that every signature that names the type must meet. Inference can fail on it, or take a long time. A struct that `@concrete` cannot write, because its parameters reach a parametric supertype or two fields share one, declares its parameters by hand with no bound. A recursive struct is such a case: a field that holds a node of the same type takes the parameters of the node itself, as `left::Union{Nothing, ClusterNode{T, H}}` does. With `@concrete`, the type of a node would hold the shape of its subtree, so each new tree would be a new type, and each method that reads a tree would compile again. `test/test_75_concrete_struct_census.jl` gates the rule and holds each exception with its reason.
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
  | `VecNum` | `AbstractVector{<:Union{Number, JuMP.AbstractJuMPScalar}}` |
  | `MatNum` | `AbstractMatrix{<:Union{Number, JuMP.AbstractJuMPScalar}}` |
  | `ArrNum` | `AbstractArray{<:Union{Number, JuMP.AbstractJuMPScalar}}` |
  | `VecInt` | `AbstractVector{<:Integer}` |
  | `Num_VecNum` | `Union{Number, VecNum}` |
  | `VecNum_MatNum` | `Union{VecNum, MatNum}` |
  | `ObsWeights` | `StatsBase.AbstractWeights` |
  | `Option{T}` | `Union{Nothing, T}` |

## A `Union` or a `Tuple` names its members without a bound

Write no `<:` before a member of a `Union{…}` or a parameter of a `Tuple{…}`. The rule holds for
every annotation in `src/` and `ext/`: a keyword argument, a positional argument, a `where`
bound, an `isa` test, a field, a type alias, and the `# Constructors` signature of a docstring.

| Spelling with a bound | Plain spelling, the same type |
| --- | --- |
| `x::Union{<:Real, <:AbstractLearningRateSchedule}` | `x::Union{Real, AbstractLearningRateSchedule}` |
| `x::Option{<:Tuple{<:Number, <:Number}}` | `x::Option{<:Tuple{Number, Number}}` |
| `isa(m, Union{<:A, <:B, <:C})` | `isa(m, Union{A, B, C})` |
| `const Num_VecNum = Union{<:Number, <:VecNum}` | `const Num_VecNum = Union{Number, VecNum}` |

Julia builds a type with a bounded member again on each call that evaluates it: a keyword
argument that the caller gives, and an `isa` test. `Union{<:Real, <:A}` costs 160 bytes on each
such call, and `Option{<:Tuple{<:Number, <:Number}}` costs 352. The plain spelling costs nothing,
and it is the same type (`==`), in an invariant position too: Julia normalises
`Vector{Union{<:Real, <:A}}` to `Vector{Union{Real, A}}`. A member that is a `UnionAll`, such as
`Online` or `VecNum`, covers all its parameters already, so a `<:` before it adds nothing.

A bound in any other type keeps its meaning and costs nothing: `Option{<:Integer}`,
`AbstractVector{<:Real}`, `Pair{<:Symbol, <:Real}`. Keep it. Do not rewrite
`NTuple{N, <:Number} where {N}` as `Tuple{Vararg{Number}}`. The diagonal rule makes the element
type of the first one concrete, so the two types differ.

`test/test_85_keyword_rebuild_allocation_census.jl` reads the source text of `src/` and `ext/`
for a bounded member, and it measures the keyword rebuild of each estimator against a positional
call (#1824).

## Numeric types come from the data

A numeric type is **derived**, never **coerced**. Read it off the arguments with `eltype`,
`typeof`, `real` and `promote_type`, and let the arithmetic widen it. The library must stay open to
number types that it has never seen, such as an automatic-differentiation dual, a quantity with a
unit, or an arbitrary-precision float. Each one arrives through the array of a caller, and never
through a change here. Every forced type closes the library to one of them.

```julia
Tf = eltype(X)                                  # the element type of one argument
Tf = promote_type(real(eltype(X)), typeof(p))   # the element type of several
```

**Integer data are supported only where an integer is the natural type of the data** (#1972).
An integer stays where it is the natural type: share counts and lot sizes of the finite
allocators, cardinalities, integer programming, `ddof`, counts and indices, the codes of a
`CategoricalPanelField`, `Bool` masks, the `Int` result of `phylogeny_matrix`, and the integer
count of a metric such as `Hamming`. An integer goes from the continuous kernels: returns, prices,
exposures, weights, moments, regressions, distances and their diagnostics. An integer there holds
no mean, variance, weight or `NaN`, so integer data make no sense in those kernels.

**Integer data enter at one place.** The carriers that a user builds convert an integer element
type once, at construction: `ReturnsResult`, `PricesResult`, `NumericPanelField`,
`TensorPanelField`, and the price ingestion layer. They call `float_if_integer` in
[`src/02_Tools/03_TypeUtilities.jl`](../../src/02_Tools/03_TypeUtilities.jl). It returns the
floating-point type of `T` when `T <: Integer`, and `T` itself for every other type, and its array
method returns a float copy of an integer array and every other array itself. So a `Float32` panel
stays in `Float32`, and a `Rational` panel stays exact. That function holds the one call of
`float` in `src/` and `ext/`.

**A continuous kernel does not repair an integer.** It takes the type of its data with `eltype`
and `promote_type`, and it does not call `float_if_integer`. A verb called with a bare integer
matrix is outside the supported domain, and it needs no guard. A verb that gives a wrong answer
silently for an integer input, rather than raise, takes a refusal in its constructor or at its
entry. A site whose integer is a natural type, such as the codes of a categorical field filled
with a fraction, still calls `float_if_integer` on that type.
`test/test_55_numeric_coercion_census.jl` holds the list of the files that call it, so a new call
is a decision that the reviewer sees.

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
`typeof(sqrt(one(T)))`. A site that answers `NaN` for the other branch of a
real quotient `a / b` takes `oftype(one(a) / one(b), NaN)`, because that value is a quotient.
Neither is a repair of an integer input.

**A threshold comes from the type too.** A test of whether a value is negligible against another
takes its tolerance from the element type, never from a literal: the default tolerance of
`isapprox`, or a multiple of `eps(T)`. A literal tolerance has a type of its own. It widens a
narrower comparison, and it makes an exact one inexact.

**Admit a concrete type only when the code needs that type**, and then only when no method
converts internally. An index is the usual case.

- An index from a floating-point value takes the in-function conversion: `ceil(Int, x)`,
  `floor(Int, x)`, `round(Int, x)` and their siblings. Never write `Int(ceil(x))`, which rounds in
  one type and then converts in another.
- An index or a count from integer arithmetic keeps `eltype`, and divides with `÷` rather than
  with `/` followed by a conversion.

**Support a type narrower than `Float64` on a best-effort basis (#1827).** Make a narrower type
work when the fix costs the `Float64` default nothing. Do not make the default worse, and do not
make a compromise that gives nothing back, to hold a narrower type. When the narrower type cannot
hold the answer for that reason, promote it with `promote_type(typeof(x), Float64)` at the one
site. `Float64`, `BigFloat` and a dual on `Float64` keep their own type. Convert the result back
to the type of the input with `oftype` where the result holds in it. State the promotion in a
`!!! warning` admonition in the docstring of the type or function that the user calls, and add the
site to `RULED` in `test/test_55_numeric_coercion_census.jl`, which otherwise refuses that
spelling. A site that widens a computation that its narrower input already holds is not this
case. It is the breach of #1808.

**Keep a signature as wide as the body allows.** Prefer `VecNum`, `MatNum`, `ArrNum` and
`<:Number` over a narrower bound, so that a caller can bring a number type from another package. Narrow
to `<:Real` only where the body needs an order, and to a concrete type only where the body needs
that exact representation.

## A container takes the element type of its values

The element type of a container is **derived** from the values that it holds, and `Any` is the
last resort. The rule holds for every container: `Vector`, `Matrix`, `Array`, `Dict`, `Set`,
`Ref`, `Pair`, `Tuple`, and a type parameter of a struct that holds values. A container of `Any`
makes each read a dynamic dispatch, and it boxes each value. Inference then gives `Any` to every
value that leaves the container, so the instability spreads far from the container, and JET
cannot read past it.

- **Let the construction take the type.** A comprehension, `map` and a broadcast take the element
  type of their values. `identity.(v)` narrows a vector whose element type is wider than its
  values, such as the join of a part that was empty and took an inferred type.
- **A copy that takes values of two kinds promotes.** Write
  `Vector{promote_type(eltype(a), eltype(b))}(a)`, never `collect(Any, a)`. Base promotes a `Pair`
  and a `NamedTuple` field by field, so the result names the family of the values. When a
  conversion changes a value that must stay as it is, take `typejoin` or `Base.promote_typejoin`
  in its place: `promote_type(Vector{Symbol}, Vector{String})` is `Vector{Any}`, and the copy
  converts each list to it (#1979).
- **A container filled out of order is built in parts.** Build each part with a comprehension,
  and join the parts in order, for example `vcat(m1, m2)[invperm(vcat(i1, i2))]`. Do not
  preallocate `Vector{Any}(undef, n)` to fill it by index.
- **An abstract element type that names a family is correct** when the values vary across the
  family, such as `Vector{AbstractPriorEstimator}`. It is narrower than `Any`, and it tells the
  reader what the container holds.

Keep `Any` only where no narrower type exists, and write a comment beside it that states why. The
argument list of an `Expr` is the known case, because Julia defines `Expr.args` as a
`Vector{Any}`, so code generation that builds one keeps `Any[]`. A value of any type is the
other case, such as a value that `throw` takes or the value that a function step of a `Pipeline`
returns. `test/test_87_any_container_census.jl` gates the rule. It reads the parse tree of
`src/` and `ext/`, and its `ALLOWED` list names each definition that keeps `Any`, with the
reason.

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
const AbstractLowOrderPriorEstimator_A_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                  AbstractLowOrderPriorEstimator_AF}
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
- The fold of a Partial Fit State is the one exception. It builds the new state through the
  positional constructor of the state, called by name, as `merge_states` and the seed functions
  do. The fold runs on every step, and the keyword constructor costs it three things (#1807):
  - It checks values that a fold writes by design. A fold under no Coverage Policy writes `NaN`
    into the cell of an inactive asset, and `assert_partial_fit_state` refuses it.
  - It reads every entry of the state. For a covariance state of 100 assets, it added 64 % to the
    cost of one step.
  - A field with a declared `Union` type, such as the gap marks of a `SampleBufferState`, puts a
    `Union` value into the `NamedTuple` of `rebuild_estimator`. The type of that `NamedTuple` is
    then abstract, and the step dispatches at run time. A positional call splits the `Union` at
    the call site, so it stays concrete.

  The estimator that holds the state still takes `rebuild_estimator`.

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
@argcheck(m <= N,
          DomainError(m,
                      "`$key` keeps a floor of $m effective assets, and the universe of the prior result holds $N. No portfolio has more effective assets than the universe, so the ceiling admits none. State a count of at most $N, or a `FractionOf` of the universe."))
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

- **Two methods are dispatch when each arm does work.** A `Nothing`/value pair and a two-family
  pair save the caller an `isa` branch. A pair whose value arm only forwards to another function
  is a wrapper: the `Nothing` method belongs on that function, and the callers call it. A set of
  methods that only sends each form of an input on to another function is a wrapper all the
  same. § *A wrapper is for convenience or for a dispatch problem* below governs both.
- **A decision, an operator, an index, a nested call, a comprehension or a string template is
  logic the unit owns.** A ternary that picks a formulation, a message template, an accessor
  that slices, and a helper that names one rule all stay.
- **A public forward is public API.** The call of a user is a use that the census cannot see. So
  the census flags a public forward, an allow-list entry keeps it, and the list shows every public
  forward of the library. The entry names one of the two reasons of § *A wrapper is for
  convenience or for a dispatch problem*. That the name is public is not a reason by itself.

The type-level case is an abstract type with one concrete subtype that nothing codes against: no
dispatch on it, no field bound, no union alias. It stays when it has a realistic future
application, and the allow-list is where that application is named.

`test/test_70_trivial_unit_census.jl` is the census and the allow-list. A flagged name outside
the list fails the build. An entry whose name is no longer flagged is stale and fails too, so the
list can only shrink.

### A wrapper is for convenience or for a dispatch problem

Keep indirection to a minimum. Give each form of an input to the function that owns the job, as a
method of that function, and call that function. A wrapper is a function whose methods only send
their arguments on to another function, or hold a form that the other function could hold. To
pack an argument into a tuple, to move a keyword into a positional slot, or to reorder the
arguments is still only to send them on, so the method is a wrapper. #1998 is the case: a
one-method form of a score on a return series only packed the series into a tuple for a helper
of `oriented_score`. It went, and #2071 gave the series to `oriented_score` as a method. A
wrapper earns its place for two reasons only:

- **A convenience for the user.** A keyword constructor, a preset such as `BookToPrice`, or a
  short name for a call that users write often.
- **A dispatch problem that a method on the owning function cannot solve.** The owning function
  belongs to another module, so a new method is type piracy. Or a new method makes a method
  ambiguity that no narrower signature removes.

A new form of an input is a new method of its reader and of its check, not a new function in
front of them. #2005 is the case. A reader of the inputs of a scorer sent two of its four forms
to `descriptor_field_values` and held the other two, and its check did the same for
`assert_panel_terms`. Both wrappers went, and the generic functions took their methods.
`test/test_70_trivial_unit_census.jl` reads each method: a method whose body only forwards fails
the build unless its name is on an allow-list with reason (a) or (b). A helper that holds a part
of a job for several forms, and a `Nothing` arm, still hold by review.

The two reasons are the whole list. Each of these is a description of a wrapper, not a reason to
keep it:

- **The name is public, or a hub that callers reach through one name.** Give the hub the
  methods: each form of the input is a method of the public function. When the wrapper itself
  is the public name, remove it all the same. The break is the cost of the better form, and
  `docs/src/migration.md` names it (`src/CLAUDE.md`, the first rule). A released spelling is
  never a reason by itself.
- **The target takes the slot as a keyword, by the precedent of its family.** Give the target a
  method for the form, and keep the keyword in that method.
- **Another arm of the function holds logic, or owns a fallback.** The arm that only forwards
  still moves. A `Nothing` arm moves with it, to the function that takes the value.
- **The function only inserts a field to dispatch on it, or picks an arity by type.** The
  callers pass the field, or the owner takes the method for the type.
- **Several methods share one body.** Write one method on the abstract type, or on the `Union`
  of a mixed family.

**One owner, one name, even at the cost of a second copy.** A helper that holds part of the job
of a function for several forms of input is a second name for that job. Give each form to the
owning function as a method, even when a short rule is then written once per form, such as the
reduction of a vector by the type of a scalariser. The `# Algorithm` section of the owner states
that the copies agree, and a test holds each copy. #2071 is the case: `reduce_member_scores`
held the reduction of `oriented_score` for a series and for a fitted result. It went, and
`oriented_score` holds the reduction in its methods, one set for each sample.

**The rulings of #2072.** The maintainer ruled these on 2026-11-13. They hold for every change
and every addition:

1. **The standing test.** Pick the form that improves the code most by architecture,
   maintenance, ergonomics and Julia idiom, with no loss of performance. When the effect is not
   clear, a measurement decides. A break is not a cost.
2. **A verb owns a default only when the default is a true answer for every caller.** A query
   trait owns `false`, as `needs_previous_weights` does. A structural traversal owns the
   identity on a leaf, as `copy_states(x) = x` does, which is the idiom of `fmap` in Functors.jl
   and of Adapt.jl. A contract verb such as `factory` owns no default, so a missing method
   stays a `MethodError`. A choice that belongs to a container, such as which steps of a
   pipeline take the weights, stays in a function of the container, and that function is kept
   by reason (b).
3. **An interface method is not a wrapper** when two or more methods share its callee, or when
   its callee is the constructor of a type. `reader_frames(x::T, path, ::Bool) =
   source_frames(x.x_src, path, true)` is the case. When the method is the only caller of its
   callee, the body of the callee moves into the method.
4. **A template hook is not a second name.** A hook that the `# Interfaces` section names, and
   that does one step of a driver, is part of the driver: `_optimise` under `optimise`, as
   `__solve` under `solve` in SciML. A method of the verb that goes around the driver is a
   wrapper.
5. **A self-forward that drops an ignored argument** becomes one method with an ignored or
   defaulted positional slot. A conversion to the canonical form is reason (a) on a public
   verb. On a private function, the conversion goes.
6. **A rebuild by hand that copies every field but one** goes to
   `rebuild_estimator(x, (; f = v))`. A site stays by hand only on a measured loss of
   performance.

**A user verb is declared.** A keyword verb that is kept as a convenience for the user is
exported or `public`. A verb that is neither is internal, and an internal keyword wrapper goes:
its callers call the positional form.

A wrapper that one of the two reasons keeps names that reason in the allow-list of
`test/test_70_trivial_unit_census.jl`.

## The simple form wins

Write the simplest code that does the work. A layer, a closure, a type, a macro or a generated
method is a cost: the reader must follow it, and a test must reach it. Add one only when it buys
at least one of these gains:

- a useful abstraction for a known future development,
- performance,
- architecture,
- ergonomics,
- maintainability,
- usability.

A form that buys none of them is needless complexity. Replace it with the simple form. When you
review a complex form, name the gain that it buys. If you cannot name one, the simple form wins.

### A function argument is passed only when the callee controls the call

A `do` block or a lambda passes a function, not a value. Pass a function only when the callee must
decide when the function runs, how many times it runs, or inside what it runs:

- **Zero times or once, by a decision of the callee.** A memoise verb such as `state_build!` runs
  the builder only when its key is not registered. `get!` builds a default only when the key is
  missing, and a lazy message is built only when the check fails.
- **Many times.** `map`, `sum`, `findall`, `stencil_table` and `finite_difference_gradient` call
  the function once for each item.
- **Between steps of the callee.** A hook such as the one of `frontier_sweep!` or
  `exp_weighted_pass!` runs inside a loop that the callee owns.
- **Inside a scope.** `with_projection_step` and `with_config` set a `ScopedValue` for the time
  that the function runs.

If the callee calls the function once, with no condition and with nothing around it, pass the
value. A scope that a caller opens only to give settings to one call is the same case. Give the
call a keyword instead. For example, `suggest_declared_key` passes its configuration to
`did_you_mean` in the keyword `sd`. It does not open a `with_string_distance` scope for that one
call (#1839).

A `do` block that a memoise verb takes is not a candidate. In
`state_build!(model, prefix, :W1_vr_sk_kt) do … end`, the value form builds the variables and the
`PSDCone` constraint before `state_build!` looks at the key. Each call that finds the key
registered then adds an unused copy to the model. The `do` block returns the registered entry and
builds nothing (ADR 0037).

The rule holds by review.

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

## A hot path is measured, and its results stay bit-identical

A **hot path** is code whose time grows with the observations, the assets or the folds: a loop
over observations, a fit that a fold repeats, a kernel that each step of an online fold calls.
The static gate of ADR 0175 reads a trap from the text of one definition. These rules cover what
only a measurement shows. [`.github/prompts/performance-pass.prompt.md`](../prompts/performance-pass.prompt.md)
is the workflow that applies them.

- **The slice that a kernel reads is contiguous.** An array of rank three holds its observations
  on its last axis (ADR 0230), so `view(A, :, :, t)` is one block in memory and BLAS takes it
  directly. A matrix stays `observations × assets`, as `X` is. When a hot loop reads a strided
  slice, change the layout of the array that it reads, not the loop.
- **A stage reads a view of the history that the stage before it wrote.** A copy of a history
  costs its full size at each fit, and it was the largest saving of #1973.
- **A loop allocates its buffers once for each task.** A threaded loop over observations takes its
  buffers from `FLoops.@init`, sized for the largest observation, and writes each observation into
  them. A buffer takes the element type of the arrays that it holds, as the numeric rule above
  states.
- **Arrays that grow together share one backing.** Two histories that a fold appends to at the
  same rows are views of one array. An append writes into spare capacity that the backing holds,
  and the backing doubles when the capacity runs out. A `vcat` at each step copies every row that
  the history holds.
- **Each block of a result is computed once.** Do not form a product to read one block of it, and
  do not compute again a block that an earlier step holds.
- **A branch that a speed-up adds is a method.** The fast case and the general case dispatch on a
  type or a `Val`, as § *A branch that a type decides is a method* states, so the complexity gate
  stays green.
- **A speed-up keeps every output bit-identical.** A probe of the configurations that the change
  reaches, mixed element types among them, compares every output with the base bit for bit. One
  cause of a difference is accepted: a change of the order of a sum, or a BLAS call that now
  takes a contiguous slice, which moves the last few bits. The commit, the ADR of the change and
  `docs/src/migration.md` then state the largest relative change and its cause. Any other
  difference is a defect of the change.
- **A claim of speed is a measurement of one run.** The base and the change run in the same
  process order, single-threaded, as the median of five fits after a warm-up. A comparison with
  another library times that library in that same run.

## A JuMP model is built in the measured forms

Issue #2062 measured each choice that a builder of a JuMP model makes. The probes ran in a bare
process that loads JuMP, Clarabel and HiGHS only, on N = 100 assets and T = 1000 observations, as
the minimum of many builds after a warm-up. Each pair of forms gave the same row to MOI, unless the
rule says otherwise. These rules bind every builder under `src/` and `ext/`.

A change to an existing site passes a stop rule. It saves at least 1 % of the build time or of the
bytes of the typical model that reaches the site, or it moves the solve time by at least 5 %. The
typical model is a `MeanRisk` on the same N and T. Its build costs 1.8 MiB for a variance, 44 MiB
for a conditional value at risk, and 309 MiB for an approximate ordered weights array before #2062.

**Every expression and every row stays registered.** A builder registers each entry with
`state_set!` under its prefix, or with a named macro. A user introspects the model, and a
report reads it, so no speed gain trades a registration away. A model that the library does not
return can hold anonymous entries: the model of a risk value, of a weight finaliser, of a
projection, of an entropy pooling posterior or of a path score. A named macro and an anonymous
macro that `state_set!` registers cost the same.

| Choice | Rule | Measurement |
| --- | --- | --- |
| Where the arithmetic is done | Do the arithmetic on JuMP expressions inside the macro of the row, or inside a registered `JuMP.@expression` | A vector on T built in a local before the macro: 35.1 MiB against 23.7 MiB in the macro. A scalar on N: 108 KiB against 92.5 KiB, below the stop rule, so both forms are correct for a scalar |
| A negated series | Write the minus inside the macro of the row, `-net_X - …` or `wr .- net_X`. When many builders read the negated series, build it once in a registered `JuMP.@expression` that each of them reads, as `risk_series` does for the gain tail (`gain_net_X`) and for the drawdown path (`neg_dd`) | A negated copy before the macro: 31.1 MiB against 24.7 MiB. A sign variable, `sg * net_X`, gains nothing, because the macro copies the product again |
| A sum over one axis | Write one `sum(view(A, :, j))`, or one `LinearAlgebra.dot(view(c, t, :), view(A, t, :))`, for each row or column. Never `sum(A; dims = d)` or `mean(A; dims = d)` on a JuMP array | Base reduces over an axis with a plain `+`, which copies each partial sum. The robust conditional value at risk: 781 MiB against 107 MiB for each row, so its model built in 664 ms and 861 MiB, against 1954 ms and 2210 MiB. The column sums of the ordered weights array: 144 MiB against 1 MiB, so its approximate model built in 70 ms and 70 MiB, against 129 ms and 308 MiB |
| An expression built in a loop | `JuMP.add_to_expression!`, or a generator inside the macro. Never `+=` | 26.8 KiB and 28.4 KiB against 718 KiB on N terms |
| A dot product | `LinearAlgebra.dot(c, w)` or `transpose(c) * w`. Never `sum(c .* w)` | 13.2 KiB against 62.5 KiB |
| A shared expression | Register it once, and let each row read the entry | Five rows that read the entry: 116 KiB. The same rows with the expression inlined: 149 KiB |
| A vector of rows | One vector-valued row, with the comparison unbroadcast, for every model and every solver. Keep a container `[i = 1:N]` for a cone of each index | The form does not depend on the solver: JuMP converts each form to the other when a solver needs it. The probe added the JuMP build and the copy to Clarabel, SCS and HiGHS over a thousand rows. The vector row was cheaper at every size from 3 to 102 terms in each entry (31.8 ms against 35.5 ms at 3 terms, 347 ms against 361 ms at 102 terms). The JuMP build alone favours the vector row up to about 12 terms, and costs up to 25 % more for a dense row (47 ms against 35 ms at 102 terms). Clarabel and SCS copy the vector row 25 to 30 % faster, and HiGHS copies scalar rows faster. The rows of the discrete allocator hold 2 or 3 terms, and their vector form builds 13 % faster and copies faster or equal on all three solvers |
| A bound on one variable | A variable bound, `lower_bound = 0`. A bound that scales with `k`, such as `w ⊖ k * lb`, stays a row | A thousand bounds: 179 KiB, against 767 KiB for one vector row and 963 KiB for a thousand rows. The solve is the same |
| The sense of a row | `expr >= 0` and `expr in MOI.GreaterThan(0)` are equal. Use the set when the caller passes it | The same row and the same bytes |
| The circled operators | Keep `⊖`, `⊕` and `⊙` on a rate that can be a scalar or a vector, because they keep the number types open | The same bytes as the plain operators, except `scalar ⊙ w`, which costs 49 KiB more on N terms, below the stop rule |
| String names | Keep `JuMP.set_string_names_on_creation(model, str_names)`, with `str_names = false` by default | A named `@variable` of a thousand entries with names is 2.6 times slower |
| The model | `JuMP.Model()`, with `JuMP.set_optimizer` at the solve | The only form that serves a list of solvers. A direct model of JuMP binds the model to one solver, so the next solver of the list cannot take it, and a solver that takes no change after the first copy refuses it (Clarabel refused the first variable). `add_bridges = false` refuses each row that the solver does not take in its own form (Clarabel refused the scalar equality rows). The build costs the same in `Model()` and in `Model(optimizer)` |
| Plural macros | One singular `@variable`, `@constraint` or `@expression` for each entry. Never `@variables`, `@constraints` or `@expressions` | A plural block builds the same model, with the same bytes and time, and the formatter aligns its lines far to the right |

`test/test_88_jump_build_form_census.jl` gates the plural macros, the sums over an axis, the `+=`
and the `sum(c .* w)`. The other rules have no static form that a census can read, so a review
reads them.

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

## The caller owns the settings of a solver

The library does not write a tolerance, an iteration limit or any other setting of a solver. Each
setting comes from the `Solver` that the caller gives. A problem that needs a tighter solve to
hold its answer states that need, with the measured value, in the docstring that the caller
reads. The caller then sets it.

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
