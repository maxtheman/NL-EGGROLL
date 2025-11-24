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
    # Titans — Experiments (stub)

    Goals:
    - Long-context recall benchmarks vs Transformer/RetNet.
    - Ablate contextual vs persistent memory components.
    - Throughput/memory profiling.
    - Capture intuition and reviewer Qs with visuals.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## TODOs
    - [ ] Implement Titans-style memory module in MLX.
    - [ ] Tasks: far-pointer, long retrieval (8k–64k tokens).
    - [ ] Ablations: contextual vs persistent memory.
    - [ ] Visuals: recall vs distance, throughput vs length.
    - [ ] Reviewer prompts and notes inline.
    """)
    return


if __name__ == "__main__":
    app.run()
