"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `pairs` to the documentation dictionary `dict`, and throw if a key already holds a
different description.

A `Dict` is last-wins, so a key written twice drops the earlier description with no warning
and makes its prose unreachable. The files of `src/01_Base/01_DocstringDictionaries/` fill
each table in parts, so a key can be written twice in two files as well as in one. This
function is the guard against both: it fails at load time and names both descriptions, so
the duplicate is visible instead of silent.

A key that returns with the description it already holds changes nothing. That is how a
second evaluation of the same file reads, for example when Revise evaluates an edited file
again.

# Algorithm

 1. For each pair, in the order the caller wrote it, raise an `ArgumentError` when `dict` already holds the key with a different description. The message names `name`, the key, and both descriptions.
 2. Otherwise store the pair in `dict`.

# Arguments

  - `dict::Dict{Symbol, String}`: The table to fill. It is changed in place.
  - `name::Symbol`: Name of the table, used in the error message.
  - `pairs`: The key-description pairs, in the order they are written.

# Validation

  - A key that `dict` already holds, from an earlier call or from an earlier pair of this one, keeps its description. A different description raises an `ArgumentError` naming `name`, the key, and both descriptions.

# Returns

  - `dict::Dict{Symbol, String}`: The table, filled.

# Related

  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`math_dict`](@ref)
  - [`err_name_dict`](@ref)
  - [`ref_dict`](@ref)
"""
function unique_key_dict!(dict::Dict{Symbol, String}, name::Symbol,
                          pairs::Pair{Symbol, <:AbstractString}...)
    for (key, val) in pairs
        if haskey(dict, key) && dict[key] != val
            throw(ArgumentError("`$(name)` has a repeated key, `:$(key)`. Each key must appear exactly once.\n  first: $(dict[key])\n  later: $(val)"))
        end
        dict[key] = val
    end
    return dict
end
"""
    arg_dict

Maps a parameter key to the docstring description of the corresponding argument or
field, so that a single description is written once here and interpolated into every
docstring that mentions that parameter (via `\$(arg_dict[key])` for `# Arguments`
entries, or through the derived [`field_dict`](@ref) for `# Fields` entries).

Each value has the form ``"`name`: description."``, where `name` is the display name
the caller sees and everything after the first `:` is the prose; `field_dict`
strips the ``"`name`: "`` prefix. A few illustrative entries:

    :ce   => "`ce`: Covariance estimator."
    :oow  => "`w`: Optional observation weights vector `observations × 1`, ..."
    :per  => "`pr`: Prior estimator or result."
    :pler => "`pl`: Network estimator, phylogeny result, clustering estimator, or clustering result."
    :plsrc => "`pl`: Network estimator or clustering estimator -- a source that refits, never a precomputed result."

The five `*_Arg*.jl` files of `src/01_Base/01_DocstringDictionaries/` fill the table, one
subject to a file, and together they are the full table of keys and descriptions. A key
must appear once: [`unique_key_dict!`](@ref) fills the table and refuses a key that another
entry holds with a different description, because a `Dict` drops the earlier entry in
silence.

# Related

  - [`unique_key_dict!`](@ref)
  - [`field_dict`](@ref)
  - [`val_dict`](@ref)
"""
const arg_dict = Dict{Symbol, String}()
"""
    field_dict

Derived dictionary mapping argument keys to field description strings, used for `\$(FIELDS)`-style docstring interpolation.

Each entry is derived from [`arg_dict`](@ref) by stripping the leading parameter name prefix (everything up to and including the first `:`). `07_FieldDict.jl` fills it after the last file of `arg_dict`, so it holds one entry for each key of `arg_dict`.

# Related

  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`math_dict`](@ref)
"""
const field_dict = Dict{Symbol, String}()
"""
    err_name_dict

Maps high-order-moment argument keys to the domain noun used in error messages, so a
message names what the caller supplied (e.g. `cokurtosis`) rather than the bare field
symbol. The symbol itself is appended at the call site, giving messages like
``cokurtosis (`kt`) cannot be empty``.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
"""
const err_name_dict = Dict{Symbol, String}()
"""
    val_dict

Validation rules for certain arg_dict terms used in the documentation of `PortfolioOptimisers.jl`.

`:relax` is the exception: it is the fixed opening sentence of a `## Relaxation` subsection under `# JuMP formulation`, held here so that the wording cannot drift between docstrings.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`field_dict`](@ref)
  - [`ret_dict`](@ref)
"""
const val_dict = Dict{Symbol, String}()
"""
    ret_dict

Dictionary containing return value descriptions for common parameters used in `PortfolioOptimisers.jl`.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`field_dict`](@ref)
  - [`val_dict`](@ref)
"""
const ret_dict = Dict{Symbol, String}()
"""
    math_dict

Dictionary of mathematical notation descriptions used for docstring interpolation throughout `PortfolioOptimisers.jl`.

Keys are symbols that identify mathematical variables or subscripts; values are LaTeX-formatted strings suitable for embedding in docstrings.

A key owns a definition, not a glyph: one glyph carries different quantities in different families, so a second quantity on the same glyph takes its own key under its own symbol.

The four `*_Math*.jl` files of `src/01_Base/01_DocstringDictionaries/` fill the table, one subject to a file. A key must appear once: [`unique_key_dict!`](@ref) refuses a key that another entry holds with a different description.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`ref_dict`](@ref)
"""
const math_dict = Dict{Symbol, String}()
"""
    ref_dict

Maps a key of `docs/src/References.bib` to the formatted `# References` bullet for that
work, so the reference text is written once here and interpolated wherever a docstring
cites it.

Each value is a complete bullet body: the citation marker for the key, followed by the
reference in the style `DocumenterCitations` renders in the bibliography. A citing docstring
writes the whole bullet as one interpolation of this table and never pastes the reference
prose inline. A pasted copy drifts from the entry in `References.bib` and from the other
copies of itself: before this table existed, `gerber2025squeezing` was pasted 31 times and
`mlp1` 13 times.

`13_References.jl` fills the table, and it is the single source of truth. A key must appear
once: [`unique_key_dict!`](@ref) fills the table and refuses a repeat.

# Related

  - [`unique_key_dict!`](@ref)
  - [`math_dict`](@ref)
"""
const ref_dict = Dict{Symbol, String}()
