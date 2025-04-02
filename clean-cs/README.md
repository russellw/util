Reduce entropy of C# code.

Assumes e.g. `clang-format` run before and after, so it doesn't need to worry about exact spaces.

Applies the following transformations:

- Capitalize the first word of the first line comment in a block thereof.
- Remove redundant braces.
- Sort case labels in a section.
- Sort case sections in a switch statement.
- Sort comparison operands:
  - `b == a` &rarr; `a == b`
  - `a > b` &rarr; `b < a`
- Sort class members by visibility, category, name.

In general, each user or project will only want a subset of these transformations. The simplest way to achieve this is to fork the project, edit the section of `Program.cs` marked `// Apply transformations`, and delete the unwanted ones.

To suppress sorting the members of a particular class or enum, mark it with a comment like:

```
// NO-SORT
class MyClass ...
```
