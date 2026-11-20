"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `pairs` to the documentation dictionary `dict`, and throw if a key already holds a
different description.

A `Dict` keeps the last value written to a key. A key written twice drops the earlier
description with no warning, and no docstring can read that description again. The files of
`src/01_Base/01_DocstringDictionaries/` fill each table in parts, so a key can occur twice in
one file or in two files. This function fails at load time in both cases, and its message names
both descriptions.

A key that comes back with the description that it already holds changes nothing. A second
evaluation of the same file writes each key in this way, for example when Revise evaluates an
edited file again.

# Algorithm

 1. For each pair, in the order the caller wrote it, raise an `ArgumentError` when `dict` already holds the key with a different description. The message names `name`, the key, and both descriptions.
 2. Otherwise store the pair in `dict`.

# Arguments

  - `dict::Dict{Symbol, String}`: The table to fill. The function changes it in place.
  - `name::Symbol`: The name of the table, which the error message shows.
  - `pairs`: The key-description pairs, in the order that the caller writes them.

# Validation

  - A key that `dict` already holds, from an earlier call or from an earlier pair of this call, keeps its description. A different description raises an `ArgumentError` that names `name`, the key, and both descriptions.

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

Maps a parameter key to the description of the argument or field that has that key. Each
description is written once, in this table. A `# Arguments` entry interpolates
`\$(arg_dict[key])`, and a `# Fields` entry reads the same text from [`field_dict`](@ref).

Each value has the form ``"`name`: description."``. The `name` is the name that the caller
sees, and the prose starts after the first `:`. `field_dict` removes the ``"`name`: "``
prefix. Four entries of the table are:

    :ce   => "`ce`: Covariance estimator."
    :oow  => "`w`: Optional observation weights vector `observations × 1`, ..."
    :per  => "`pr`: Prior estimator or result."
    :pler => "`pl`: Network estimator, phylogeny result, clustering estimator, or clustering result."

The five `*_Arg*.jl` files of `src/01_Base/01_DocstringDictionaries/` fill the table, one
subject to a file. A key must occur once. [`unique_key_dict!`](@ref) fills the table, and it
refuses a key that another entry holds with a different description, because a `Dict` drops
the earlier entry with no warning.

# Related

  - [`unique_key_dict!`](@ref)
  - [`field_dict`](@ref)
  - [`val_dict`](@ref)
"""
const arg_dict = Dict{Symbol, String}()
"""
    field_dict

Maps an argument key to the description of a field. An inline field docstring interpolates this table, and `\$(DocStringExtensions.FIELDS)` shows the result under `# Fields`.

Each entry is the value of [`arg_dict`](@ref) for the same key, with the name prefix removed. The prefix is all the text up to the first `:`, and the `:` itself. `07_FieldDict.jl` fills the table after the last file of `arg_dict`, so the table holds one entry for each key of `arg_dict`.

# Related

  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`math_dict`](@ref)
"""
const field_dict = Dict{Symbol, String}()
"""
    err_name_dict

Maps the key of a high-order moment argument to the noun that an error message uses. The
message names what the caller gave, for example `cokurtosis`, and not only the name of the
field. The call site adds the field name after the noun, and the message reads
``cokurtosis (`kt`) cannot be empty``.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
"""
const err_name_dict = Dict{Symbol, String}()
"""
    val_dict

Maps a key of [`arg_dict`](@ref) to the validation rule that a `## Validation` subsection states for that argument.

The key `:relax` is the one exception. Its value is the first sentence of each `## Relaxation` subsection under `# JuMP formulation`. The table holds it so that all docstrings use the same words.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`field_dict`](@ref)
  - [`ret_dict`](@ref)
"""
const val_dict = Dict{Symbol, String}()
"""
    ret_dict

Maps a key to the description of a return value that many `# Returns` sections share.

# Related

  - [`unique_key_dict!`](@ref)
  - [`arg_dict`](@ref)
  - [`field_dict`](@ref)
  - [`val_dict`](@ref)
"""
const ret_dict = Dict{Symbol, String}()
"""
    math_dict

Maps a key to the definition of a mathematical symbol. A `Where:` bullet of a docstring interpolates the definition.

A key names a mathematical variable or a subscript. Its value is a string in LaTeX that a docstring can contain as it is.

A key holds one definition, not one glyph. One glyph can stand for different quantities in different families, so each quantity gets its own key.

The four `*_Math*.jl` files of `src/01_Base/01_DocstringDictionaries/` fill the table, one subject to a file. A key must occur once. [`unique_key_dict!`](@ref) refuses a key that another entry holds with a different description.

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
work. The table holds the text of each reference once, and each docstring that cites the work
interpolates it.

Each value is the complete text of one bullet. It starts with the citation marker for the key,
and the reference follows in the style that `DocumenterCitations` uses in the bibliography. A
docstring writes the whole bullet as one interpolation of this table. It never contains a
copy of the reference text. A copy goes out of step with the entry in `References.bib` and
with the other copies. Before this table existed, the docstrings held 31 copies of
`gerber2025squeezing` and 13 copies of `mlp1`.

`13_References.jl` fills the table, and no other file holds the text of a reference. A key
must occur once. [`unique_key_dict!`](@ref) fills the table and refuses a repeated key.

# Related

  - [`unique_key_dict!`](@ref)
  - [`math_dict`](@ref)
"""
const ref_dict = Dict{Symbol, String}()
