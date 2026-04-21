# Artifact Trust Boundary Design

**Problem**

ForeSight artifact loading is pickle-backed, but parts of the public documentation
currently describe `load_forecaster_artifact(...)` and artifact inspection flows as
metadata-only operations. That wording understates the trust boundary: these flows
unpickle artifact contents and therefore must be treated as trusted-source-only.

**Approved Approach**

Use the smallest credible fix:

1. Correct the Python API and guide documentation so they describe the real runtime
   behavior and explicitly warn about pickle-backed loading.
2. Surface the same trusted-source warning on the top-level `foresight artifact`
   CLI help entrypoint, not only on subcommands.
3. Tighten regression tests so they lock the specific public surfaces that were
   drifting, rather than only asserting that the warning appears somewhere in a
   large document.
4. Generalize the plan-doc absolute-link guard so it protects the underlying
   constraint instead of one hard-coded workspace prefix.

**Non-Goals**

- Redesign artifact serialization
- Introduce a metadata-only safe loader
- Expand release gating beyond this focused trust-boundary patch

**Validation**

Focused tests should prove:

- API docs no longer describe `load_forecaster_artifact(...)` as metadata-only
- User-facing guide pages explicitly warn that artifact inspection commands still
  load pickle-backed artifacts from trusted sources only
- `foresight artifact --help` shows the trusted-source warning
- The absolute-link regression test catches generic workspace-style absolute links
