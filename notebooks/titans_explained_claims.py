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
    # Titans Explained — Math & Metrics (stub)

    Goals:
    - Extract equations (surprise, memory dynamics) into code.
    - Generate synthetic tasks matching the formalism.
    - Validate that empirical Titans runs align with defined metrics.
    - Capture intuition and reviewer Qs with visuals.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## TODOs
    - [ ] Port key equations into MLX functions.
    - [ ] Build synthetic surprise-driven task.
    - [ ] Compare measured signals to theoretical predictions.
    - [ ] Visuals: surprise over time, memory trajectories.
    - [ ] Reviewer prompts and notes inline.
    """)
    return


if __name__ == "__main__":
    app.run()
