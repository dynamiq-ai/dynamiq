<!--
Title: Conventional Commits, e.g. `fix(retrievers): keep the filter on a Qdrant search`. CI checks it, and it becomes the
commit on main and the line in the release notes. Mark a breaking change with `!`.
Fill in each section and delete these comments.
-->

<!-- What changes and why. Link the issue it fixes, e.g. "Fixes #1234". -->

## Compatibility

<!--
Do workflows saved as YAML still load and run? New fields are optional with a default, and a moved or renamed class or
field keeps its old name working. For a breaking change, say how to migrate. For a new dependency, say which extra it
goes in and why it is needed. Write "No change" if none of this applies.
-->

## Verified

<!--
What you ran and what you saw: the tests you added and whether they fail without the change, `make prepare`, the pytest
command CI runs (see CONTRIBUTING.md), and any example or provider you ran end to end.
-->
