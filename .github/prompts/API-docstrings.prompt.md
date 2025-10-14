---
mode: ask
description: Generate Julia docstrings.
---

Create Julia docstrings for the selected code.

  - Place the type (without constructor), function (with type information), or macro (with type information).
  - Give a general description.

## Abstract types

  - Add a heading "# Related Types" at the end of the docstring.
    
      + List the subtypes.

## Concrete Types

  - Add a heading "# Fields".
    
      + List the fields without their types.
      + Every field should have a description of what it does.

  - If applicable, add a heading "# Constructors".
    
      + List the keyword constructors.
      + Include their types and default arguments.
      + Write how the arguments correspond to the fields.
  - If applicable, add a heading "## Validation".
    
      + List the validation rules.
  - If applicable, add a heading "# Examples".
    
      + Provide a short example as a jldoctest.
  - Add a heading "# Related" at the end of the docstring.
    
      + List related types, functions, macros, and packages.

## Functions and Macros

  - Add a heading "# Arguments".
    
      + List the arguments and keyword arguments without their types.
      + Every argument should have a description of what it does.

  - Add a heading "# Returns".
    
      + Describe what the function or macro returns.
      + In the return variable include its type.
  - If applicable, add a heading "# Validation".
    
      + List the validation rules.
  - Add a heading "# Details".
    
      + Provide a bullet point summary of the function or macro's behavior.
  - Add a heading "# Related" at the end of the docstring.
    
      + List related types, functions, macros, and packages.
