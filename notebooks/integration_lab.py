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
    # Integration Lab — HOPE/Titans-lite + EGGROLL (stub)

    Goals:
    - Train a fast-weight (HOPE/Titans-lite) model with EGGROLL ES.
    - Curriculum: continual adaptation + long-context retrieval.
    - Compare to backprop-trained baselines.
    - Capture intuition, visuals, and reviewer Qs.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## TODOs
    - [ ] Define model architecture (fast weights + memory).
    - [ ] Implement ES training loop (reuse EGGROLL step).
    - [ ] Baselines: backprop HOPE/Titans-lite, Transformer.
    - [ ] Metrics: adaptation speed, long-horizon recall, throughput.
    - [ ] Visuals: learning curves, recall vs distance, cost profiles.
    - [ ] Reviewer prompts and notes inline.
    """)
    return


if __name__ == "__main__":
    app.run()
