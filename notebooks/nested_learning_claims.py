import marimo

__generated_with__ = "0.7.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo


@app.cell
def __(mo):
    mo.md(r"""
    # Nested Learning / HOPE — Experiments (stub)

    Goals:
    - Implement fast-weight layer with learned optimizer (MLX).
    - Compare to Transformer/RNN on continual tasks.
    - Frequency ablations (single vs multi).
    - Capture intuition and reviewer Qs with visuals.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## TODOs
    - [ ] Fast-weight layer + optimizer-as-memory.
    - [ ] Tasks: copy/induction, synthetic continual learning.
    - [ ] Ablations: learned vs fixed optimizer; frequencies.
    - [ ] Visuals: adaptation/forgetting curves; heatmaps.
    - [ ] Reviewer prompts and notes inline.
    """)
    return


if __name__ == "__main__":
    app.run()
