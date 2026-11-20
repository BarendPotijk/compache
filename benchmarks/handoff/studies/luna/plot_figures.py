"""Render the README cost table and cost/retention plot from the frozen comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "figures"
SUITES = {"goodai": "GoodAI LTM", "memoryarena": "MemoryArena"}
# Policy order and marker identity are consistent across the two figures.
POLICIES = {
    "generic": ("Compache", "o", "#bd5f42"),
    "opencode": ("OpenCode", "s", "#373a37"),
    "headroom": ("Headroom*", "^", "#69705e"),
}
WINDOWS = (5, 30)
PAPER = "#faf9f5"
INK = "#181816"
CORAL = "#d87959"
MUTED = "#575650"
HEADROOM_NOTE = "* Headroom 0.44.0 in cache mode with compression disabled. The shared host handled overflow."
MOBILE_NOTE = "* Headroom 0.44.0 in cache mode with compression\ndisabled. The shared host handled overflow."


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def cost_table(data):
    fig = plt.figure(figsize=(10.6, 3.67), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1000), ylim=(346, 0))
    ax.set_axis_off()
    ax.text(18, 33, "Total cost per run", fontsize=23, weight="bold", color=INK)
    ax.text(982, 22, "Exploratory · fitted on synthetic RAG", ha="right",
            fontsize=11.5, weight="bold", color=INK)
    ax.text(982, 38, "GPT-6 Luna · low effort", ha="right", fontsize=10.5, color=MUTED)

    edges = (18, 285, 427, 629, 805, 982)
    top, body, height = 55, 108, 42
    bottom = body + 4 * height
    for column, title in enumerate(("Benchmark", "Cache", *[p[0] for p in POLICIES.values()])):
        y = (top + body) / 2
        ax.text(edges[column] + 12, y - 7 if title == "Compache" else y,
                title, va="center", fontsize=13, weight="bold")
        if title == "Compache":
            ax.text(edges[column] + 12, y + 11, "(fitted)", va="center", fontsize=10.5, color=MUTED)

    for group, (suite, title) in enumerate(SUITES.items()):
        start = body + group * 2 * height
        ax.text(30, start + 34, title, fontsize=15, color=INK, weight="bold", va="center")
        subtitle = "Long-term memory" if suite == "goodai" else "Progressive Search"
        ax.text(30, start + 57, subtitle, fontsize=12.5, color="#605f58", va="center")
        for index, ttl in enumerate(WINDOWS):
            y = start + index * height
            costs = [data[suite][arm, ttl]["usd_per_turn"] for arm in POLICIES]
            ax.text(edges[1] + 12, y + height / 2, f"{ttl} min", va="center", fontsize=13)
            for col, cost in enumerate(costs, start=2):
                best = cost == min(costs)
                if best:
                    ax.add_patch(Rectangle((edges[col], y), edges[col + 1] - edges[col],
                                           height, color="#efd0bd" if col == 2 else "#ddded6", lw=0))
                ax.text(edges[col] + 12, y + height / 2, f"${cost:.5f}", va="center",
                        fontsize=14, weight="bold" if best else "normal")

    for x in edges:
        ax.vlines(x, top, bottom, color=INK, lw=.9)
    for y in (top, body, body + 2 * height, bottom):
        ax.hlines(y, edges[0], edges[-1], color=INK, lw=.9)
    for y in (body + height, body + 3 * height):
        ax.hlines(y, edges[1], edges[-1], color=INK, lw=.85)
    ax.add_patch(Rectangle((edges[2], top), edges[3] - edges[2], bottom - top,
                           fill=False, ec=CORAL, lw=2.3, zorder=4))
    ax.text(18, 304, "USD per user run, including summaries, recovery and cache maintenance.",
            fontsize=11.5, color="#4f4f49")
    ax.text(18, 328, HEADROOM_NOTE, fontsize=10.4, color=MUTED)
    save(fig, "benchmark-cost-per-run.png")


def point(row):
    counts = row["retention"]
    return row["usd_per_turn"], 100 * counts["kept"] / counts["assessed"]


def mobile_cost_table(data):
    fig = plt.figure(figsize=(4.5, 5.45), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(100, 0))
    ax.set_axis_off()
    ax.text(4, 7, "Total cost per run", fontsize=20, weight="bold")
    ax.text(4, 11.9, "Exploratory · fitted on synthetic RAG", fontsize=10.5, weight="bold", color=INK)
    ax.text(4, 15.9, "USD per user run · GPT-6 Luna · low effort", fontsize=10.2, color=MUTED)
    for group, suite in enumerate(SUITES):
        y = 22.5 + group * 34
        name = "GoodAI LTM" if suite == "goodai" else "MemoryArena Progressive Search"
        ax.text(4, y, name, weight="bold", fontsize=14)
        top = y + 2.5
        row_edges = (top, top + 5.4, top + 14.0, top + 19.75, top + 25.5)
        for x, label in zip((6, 51, 75), ("Policy", "5 min", "30 min")):
            ax.text(x, top + 2.8, label, va="center", fontsize=11, weight="bold")
        for j, (arm, (label, _, _)) in enumerate(POLICIES.items(), start=1):
            row, end = row_edges[j:j + 2]
            yc = (row + end) / 2
            if arm == "generic":
                ax.text(6, yc + 2.0, "(fitted)", va="center", fontsize=9.5, color=MUTED)
            ax.text(6, yc - 1.3 if arm == "generic" else yc, label, va="center", fontsize=11.5)
            for x, right, ttl in zip((49, 73), (73, 96), WINDOWS):
                cost = data[suite][arm, ttl]["usd_per_turn"]
                best = cost == min(data[suite][other, ttl]["usd_per_turn"] for other in POLICIES)
                if best:
                    ax.add_patch(Rectangle((x, row), right - x, end - row,
                                           facecolor="#efd0bd" if arm == "generic" else "#ddded6", lw=0))
                ax.text(x + 2, yc, f"${cost:.5f}", va="center", fontsize=11.5,
                        weight="bold" if best else "normal")
        for y_edge in row_edges:
            ax.hlines(y_edge, 4, 96, color=INK, lw=.7)
        for x in (4, 49, 73, 96):
            ax.vlines(x, row_edges[0], row_edges[-1], color=INK, lw=.7)
    ax.text(4, 94, MOBILE_NOTE, fontsize=10.1, color=MUTED, va="center", linespacing=1.4)
    save(fig, "benchmark-cost-per-run-mobile.png")


def note(ax, row, offset, text=None):
    counts = row["retention"]
    label = text or f"{counts['kept']}/{counts['assessed']}"
    ax.annotate(label, point(row), xytext=offset, textcoords="offset points",
                ha="left" if offset[0] >= 0 else "right", va="center",
                fontsize=11, color=INK,
                arrowprops={"arrowstyle": "-", "color": "#8b897d", "lw": .75,
                            "shrinkA": 4, "shrinkB": 7})


def retention_plot(data, *, mobile=False):
    fig = plt.figure(figsize=(4.5, 8.4) if mobile else (10.6, 5.85), facecolor=PAPER)
    left = .043 if mobile else .018
    fig.text(left, .973 if mobile else .923, "Cross-run dependencies kept" +
             ("\nvs. cost per run" if mobile else " vs. cost per run"),
             va="top" if mobile else "baseline", fontsize=18 if mobile else 23, weight="bold", color=INK)
    if not mobile:
        fig.text(.982, .927, "GPT-6 Luna · low effort", ha="right", fontsize=11, color=MUTED)

    fig.text(left, .877 if mobile else .873, "Exploratory · fitted on synthetic RAG",
             fontsize=10.5 if mobile else 11.5, color=INK, weight="bold")
    legend_y = .832 if mobile else .816
    label_starts = (.047, .370, .664) if mobile else (.025, .191, .335)
    pad = .040 if mobile else .020
    for (arm, (name, marker, color)), x in zip(POLICIES.items(), label_starts):
        fig.add_artist(Line2D([x + .009], [legend_y], marker=marker, linestyle="none", markersize=7,
                              color=color, transform=fig.transFigure))
        fig.text(x + pad, legend_y, name, va="center", fontsize=12.5)
        if arm == "generic":
            fig.text(x + pad, legend_y - .027 if mobile else legend_y - .036, "(fitted)",
                     va="center", fontsize=10.2, color=MUTED)
    fill_keys = [Line2D([], [], ls="none", marker="o", mec=INK, ms=6.5, mfc=color, label=label)
                 for color, label in (("none", "5-minute cache"), (INK, "30-minute cache"))]
    fig.legend(handles=fill_keys, loc="center right" if not mobile else "center left",
               bbox_to_anchor=(left if mobile else .982, .768 if mobile else legend_y),
               ncol=2, frameon=False, labelcolor=MUTED, fontsize=11, handlelength=1,
               columnspacing=1.4, borderpad=0, handletextpad=.45)
    rule_y = .738 if mobile else .74
    fig.add_artist(Line2D([left, 1 - left], [rule_y, rule_y], color=INK, lw=.8,
                          transform=fig.transFigure))

    for i, (suite, title) in enumerate(SUITES.items()):
        rect = [.16, .495 - i * .329, .785, .187] if mobile else [.084 + i * .485, .226, .391, .410]
        ax = fig.add_axes(rect, facecolor=PAPER)
        title = title if suite == "goodai" else title + " Progressive Search"
        ax.set_title(title, loc="left", color=INK, fontsize=14, weight="bold", pad=13)
        # GoodAI costs differ by less than one tenth of a cent. Expose those
        # differences with separate, explicitly priced x scales and a shared y scale.
        if suite == "goodai":
            ax.set_xlim(.08618, .08750)
            ax.set_xticks([.0862, .0866, .0870, .0874])
            ax.xaxis.set_major_formatter(StrMethodFormatter("${x:.4f}"))
        else:
            ax.set_xlim(.123, .143)
            ax.set_xticks([.125, .130, .135, .140])
            ax.xaxis.set_major_formatter(StrMethodFormatter("${x:.3f}"))
        ax.set_ylim(-4, 109)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(colors=INK, length=3.5, labelsize=11, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK)
        ax.grid(axis="y", color="#deddd5", linewidth=.62)
        ax.set_axisbelow(True)
        if i == 0 and not mobile:
            ax.set_ylabel("Cross-run dependencies kept", color=INK, fontsize=11.5, labelpad=8)

        rows = data[suite]
        # A larger hollow symbol remains visible behind its filled counterpart at exact ties.
        for ttl in WINDOWS:
            for arm, (_, shape, color) in POLICIES.items():
                x, y = point(rows[arm, ttl])
                ax.plot(x, y, marker=shape, ls="none", ms=11 if ttl == 5 else 6,
                        mfc="none" if ttl == 5 else color, mec=color, mew=1.4, zorder=3)
        if suite == "goodai":
            counts = rows["generic", 30]["retention"]
            note(ax, rows["generic", 30], (14, 34),
                 f"{counts['kept']}/{counts['assessed']} · all three policies")
        else:
            note(ax, rows["headroom", 30], (6, 34))
            note(ax, rows["generic", 30], (6, -34))

    fig.text(.525, .107 if mobile else .111, "Total cost per user run (USD)", ha="center",
             color=INK, fontsize=12.5)
    fig.text(left, .024 if mobile else .030, MOBILE_NOTE if mobile else HEADROOM_NOTE,
             fontsize=10.2, color=MUTED, va="bottom", linespacing=1.35)
    save(fig, "benchmark-cost-vs-retention-mobile.png" if mobile else "benchmark-cost-vs-retention.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "text.color": INK, "axes.unicode_minus": False})
    results = json.loads((HERE / "comparison.json").read_text())
    rows = {suite: {(r["arm"], r["ttl"]): r for r in results[suite]["metrics"]} for suite in SUITES}
    cost_table(rows)
    mobile_cost_table(rows)
    retention_plot(rows)
    retention_plot(rows, mobile=True)
