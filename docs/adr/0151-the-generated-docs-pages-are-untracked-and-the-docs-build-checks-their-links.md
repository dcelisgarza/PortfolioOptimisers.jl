---
status: accepted
---

# The generated docs pages are untracked, and the Docs build checks their links

## Context

`docs/src/user_guide/**`, `docs/src/examples/**`, `docs/src/capability_catalogue.md` and
`docs/src/api/*_TypeHierarchy.md` were committed to the tree, and `CLAUDE.md` § Editing and
`.markdownlintignore` described them as generated and overwritten by CI. No workflow overwrote
them. `Docs.yml` regenerates them inside the runner's checkout — `generate_files`,
`generate_type_hierarchy` and `generate_capability_catalogue` in `docs/make.jl` — and builds from
that copy, which the runner then discards. The committed copies were whatever the last person to
run a docs build locally happened to commit. `docs/src/user_guide/03_Risk_Measures.md` was last
written at `b81c10766c`, two releases before the `.jl` source it claimed to render.

Two consumers read the committed copies and were misled.

`LinkChecker.yml` runs lychee over the checkout. At `6fbaf7f97b` (#1058) it went red on
`docs/src/user_guide/03_Risk_Measures.md:106`, a link to a page #1058 had moved. The `.jl` source
carried the same stale link, but fixing the source alone left the gate red, because the gate read
the generated file. `4b4f018daa` had to edit both, against the rule that only the `.jl` is edited.
The Docs build of the same commit had already caught the link on its own: `makedocs` runs without
`warnonly`, so an unresolved local link is a `:cross_references` failure and the build is red.

A reader of the tree — a reviewer, an agent, `grep` — saw prose and links the current `.jl` no
longer produced. Round two of the PR 625 review found two commits regenerating
`docs/src/examples/**` by hand at +2820 and +2882 lines, and asked each piece to diff its examples
against the committed markdown; the committed markdown carried no rendered output, so the diff
could not be taken from the tree.

Issue #1077 put the choice: stop committing the pages, or regenerate and commit them from CI with a
bot commit on every docs change.

## Decision

**The four generated paths are untracked. The docs build writes them, and the Docs build's
cross-reference check is the gate on the links they carry.**

1. `docs/src/examples/`, `docs/src/user_guide/`, `docs/src/capability_catalogue.md` and
   `docs/src/api/*_TypeHierarchy.md` are in `.gitignore`. The committed copies are deleted at
   `fd0f97a9e4`. `examples/**/*.ipynb` was already ignored.
2. `docs/make.jl` changes nothing. `generate_files` already regenerates a page whose output is
   missing, so a checkout that carries none of them is the full-rebuild case it handled before.
3. `LinkChecker.yml` changes nothing. Lychee reads the checkout, the checkout no longer carries
   the pages, and the Docs build checks their local links on every push and pull request. The
   external links of the generated pages — fourteen distinct URLs at the time of writing — are no
   longer checked by lychee. A generation step before lychee would have loaded the package and
   every docs dependency into a second runner to check fourteen links; the cost is not paid.
4. `CLAUDE.md` § Editing says the pages are untracked and that no gate reads one from the tree,
   so a broken link in a generated page is fixed at the `.jl` source alone. `STANDARDS.md` routes
   *a generated docs file* to that section and to this ADR. `.markdownlintignore` keeps the four
   paths, because a local docs build leaves them in the tree and the lint hook would otherwise
   read them.

**Rejected: regenerate and commit from CI.** A workflow step on `dev` that runs the three
generators and commits the result keeps the tree browsable and keeps lychee honest, at the cost of
a bot commit on every docs change. Every such commit lands between two sessions' rebases, the
worktree rule of `CLAUDE.md` asks each session to rebase onto `dev` before it merges, and the
generated pages are the largest files in the tree — `git log` and every diff would carry them.
The tree is browsable at the deployed site, which is built from the same generators.

## Consequences

- A broken link in a generated page is red on the Docs build, and it is fixed at the `.jl` source
  once. There is no second copy to edit.
- A reviewer reading the tree sees the `.jl` source and nothing that claims to be its rendering.
  The rendering is the deployed site, or a local `julia --project=docs docs/make.jl`.
- `README.md` linked the sample dataset at `docs/src/examples` on `main`; it links `examples/`,
  where the file is tracked.
- The external links of the generated pages are unchecked until a gate reads the `.jl` sources.
  ADR 0083 is amended, because its Context section describes the pages as committed.

## Verification

`git ls-files docs/src/user_guide docs/src/examples docs/src/capability_catalogue.md
'docs/src/api/*_TypeHierarchy.md'` lists nothing. `git check-ignore` answers each of the four
paths. `test/test_46_standards_citation_census.jl` and `test/test_50_docs_sitemap.jl` pass, and
the ADR index carries the row.

## Amendment (2026-09-17)

[ADR 0128 § Amendment](0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md)
moves the type hierarchy out of `docs/src/api/` to `docs/src/TypeHierarchy.md`, a fixed name
with no numeric prefix. The fourth of the four untracked paths this ADR names changes with it:
`docs/src/api/*_TypeHierarchy.md` becomes `docs/src/TypeHierarchy.md` in `.gitignore`,
`.markdownlintignore`, `CLAUDE.md` § Editing and the citation allow-list of
`test/test_46_standards_citation_census.jl`. The other three paths, the reasoning in Decision
and Consequences, and the rejection of a CI-committed alternative are unchanged.

Verification is re-run against the new path: `git ls-files docs/src/user_guide
docs/src/examples docs/src/capability_catalogue.md docs/src/TypeHierarchy.md` lists nothing,
and `git check-ignore` answers all four.

## Amendment (2026-09-26)

**A hand-written page links another page by the `@id` label of its H1, never by a relative
`.md` path.** The hand-written pages are `docs/src/*.md`, `docs/src/contribute/` and the two
mirror trees. They stay in the checkout, so lychee reads them, and the Docs build resolves their
links too. A relative path breaks when either page moves. The file split `0ed224263c` moved
`public_api/17_Optimisation/01_Base_Optimisation/01_OptimisationTypes.md` one directory deeper,
its link to the aliases page broke, and the Link checker was red on `dev` until `ee854441e3`.
At that commit the hand-written pages held 117 such links in 116 files (#1353).

1. The target page's H1 is `# [Title](@id label)`. A public mirror page takes
   `api-<title>`, and a private mirror page takes `private-api-<title>` with the `: private API`
   suffix dropped. The title is lower case, and a run of other characters is one hyphen. A page
   that already carries a label keeps it. `page_h1` in `docs/page_metadata.jl` unwraps the
   label, so the derived `Description` of a mirror page does not change.
2. The link is `[text](@ref label)`. `.lychee.toml` excludes `@ref`, and the Docs build fails on
   a label that does not resolve.
3. The label is explicit. A bare `[Title](@ref)` needs the heading text to be unique across the
   site, and "Threshold Constraints: private API" is the H1 of two pages. A backticked
   `` [`Title`](@ref) `` is a docstring reference and fails the Docs build.
4. `test/test_64_docs_page_metadata_census.jl` refuses a relative `.md` link on a hand-written
   page. A link whose target holds a colon is a URL, and the census does not read it.

The Literate sources under `user_guide/` and `examples/` are generated pages. This amendment does
not govern them, and the Docs build stays the gate on their links as the Decision states.
