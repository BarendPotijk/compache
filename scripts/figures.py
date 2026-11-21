#!/usr/bin/env python3
"""Draw the figures under docs/assets from the data files in docs/assets/data.

    python scripts/figures.py              # every figure
    python scripts/figures.py map thread   # some of: map, thread, openclaw, workflow

Every figure paints its own surface and carries a light and a dark palette, so it
reads on either GitHub theme. The palette, typeface and marks are the calculator's.

Where the data comes from:
    policy-grid.json   python benchmarks/docchat/run.py grid --data <fast> <mixed> <slow> --output ...
    thread.json        python benchmarks/docchat/run.py thread --data <build> --thread <id> --recipe ... --output ...
    openclaw.json      node scripts/openclaw_comparison.mjs --json docs/assets/data/openclaw.json
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DATA = ASSETS / "data"

STYLE = """
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px;fill:#0b0b0b}
.surface{fill:#fcfcfb;stroke:rgba(11,11,11,.1)}
.t2{fill:#52514e}.tm{fill:#898781}.b{font-weight:600}.h{font-size:13px;font-weight:600}.big{font-size:15px;font-weight:600}
.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}.box{fill:none;stroke:#c3c2b7}
.c{fill:#2a78d6}.cs{stroke:#2a78d6}.n{fill:#898781}.ns{stroke:#898781}.o{fill:#eb6834}.os{stroke:#eb6834}
.cell{fill:#e1e0d9}.ring{stroke:#fcfcfb;stroke-width:2}.line{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.arc{fill:none;stroke-width:1.75;stroke-linecap:round}
@media (prefers-color-scheme:dark){
text{fill:#f0efec}.surface{fill:#1a1a19;stroke:rgba(255,255,255,.1)}.t2{fill:#c3c2b7}
.grid{stroke:#2c2c2a}.axis{stroke:#383835}.box{stroke:#52514e}
.c{fill:#3987e5}.cs{stroke:#3987e5}.o{fill:#d95926}.os{stroke:#d95926}
.cell{fill:#383835}.ring{stroke:#1a1a19}
}
"""


def svg(width, height, title, description, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">\n'
        f'<title id="t">{escape(title)}</title>\n<desc id="d">{escape(description)}</desc>\n'
        f"<style>{STYLE}</style>\n"
        f'<rect class="surface" x=".5" y=".5" width="{width - 1}" height="{height - 1}" rx="10"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def text(x, y, content, cls="", anchor="start"):
    cls = f' class="{cls}"' if cls else ""
    anchor = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{cls}{anchor}>{escape(str(content))}</text>'


def dollars(value):
    return f"${value:,.2f}" if value < 10 else f"${value:,.0f}"


def away(seconds):
    return f"{seconds} seconds" if seconds < 120 else f"{seconds // 60} minutes"


# -- the policy map ---------------------------------------------------------


def policy_map(data):
    """Three panels, one per return pattern: cost against how much history stays in view."""
    width, height = 880, 424
    left, right, top, bottom, gap = 52, 20, 140, 48, 28
    clocks = list(data["clocks"].items())
    panel = (width - left - right - gap * (len(clocks) - 1)) / len(clocks)
    plot = height - top - bottom
    highest = max(
        c["dollars_per_1000_turns"] for _, clock in clocks for c in clock["candidates"]
    )
    step = 100 if highest > 150 else 50 if highest > 60 else 10
    ceiling = step * -(-highest // step)
    most = max(c["max_pings"] for _, clock in clocks for c in clock["candidates"])

    def y(value):
        return top + plot - value / ceiling * plot

    body = [
        text(24, 30, "Cost per 1,000 turns of every policy the fit priced", "big"),
        text(
            24,
            49,
            f"DocChat, {clocks[0][1]['test_turns']:,} held-out turns, {data['model_name']} at list price. A point is one policy.",
            "t2",
        ),
    ]
    legend = [
        ("line ns", "No keep-warm pings"),
        ("line cs", f"Up to {most} keep-warm pings"),
        ("dot c", "The policy compache fitted"),
        ("dot n", "Keep everything, no pings"),
    ]
    at = 24
    for kind, label in legend:
        if kind.startswith("line"):
            body.append(
                f'<line class="{kind}" x1="{at}" x2="{at + 18}" y1="70" y2="70"/>'
            )
            at += 24
        else:
            body.append(
                f'<circle class="{kind[4:]} ring" cx="{at + 5}" cy="70" r="5.5"/>'
            )
            at += 16
        body.append(text(at, 74, label, "t2"))
        at += 6.4 * len(label) + 22
    summary = []
    for i, (name, clock) in enumerate(clocks):
        x0 = left + i * (panel + gap)

        def x(share, x0=x0):
            return x0 + share * panel

        rows = clock["candidates"]
        fitted = next(r for r in rows if r["fitted"])
        default = next(r for r in rows if r["recipe"] is None and r["max_pings"] == 0)
        ratio = default["dollars_per_1000_turns"] / fitted["dollars_per_1000_turns"]
        summary.append(
            f"Users back after {away(clock['median_idle_seconds'])}: fitted {dollars(fitted['dollars_per_1000_turns'])}, "
            f"keep everything without pings {dollars(default['dollars_per_1000_turns'])}."
        )
        body.append(
            text(
                x0,
                top - 30,
                f"Users back after {away(clock['median_idle_seconds'])}",
                "h",
            )
        )
        body.append(
            text(x0, top - 14, f"keeping everything costs {ratio:.1f}× as much", "tm")
        )
        for tick in range(0, int(ceiling) + 1, step):
            body.append(
                f'<line class="{"axis" if tick == 0 else "grid"}" x1="{x0:.1f}" x2="{x0 + panel:.1f}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
            )
            if i == 0:
                body.append(text(x0 - 8, y(tick) + 4, f"${tick}", "tm", "end"))
        for share in (0, 0.5, 1):
            anchor = "start" if share == 0 else "end" if share == 1 else "middle"
            body.append(text(x(share), top + plot + 17, f"{share:.0%}", "tm", anchor))
        for pings, cls in ((0, "ns"), (most, "cs")):
            line = sorted(
                (r for r in rows if r["max_pings"] == pings),
                key=lambda r: r["tokens_kept"],
            )
            points = [
                (x(r["tokens_kept"]), y(r["dollars_per_1000_turns"])) for r in line
            ]
            body.append(
                f'<path class="line {cls}" d="M'
                + " L".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + '"/>'
            )
            fill = "n" if cls == "ns" else "c"
            body += [
                f'<circle class="{fill}" cx="{px:.1f}" cy="{py:.1f}" r="2.5"/>'
                for px, py in points
            ]
        dx, dy = x(default["tokens_kept"]), y(default["dollars_per_1000_turns"])
        body.append(f'<circle class="n ring" cx="{dx:.1f}" cy="{dy:.1f}" r="5.5"/>')
        body.append(
            text(
                dx - 9, dy - 8, dollars(default["dollars_per_1000_turns"]), "t2", "end"
            )
        )
        fx, fy = x(fitted["tokens_kept"]), y(fitted["dollars_per_1000_turns"])
        body.append(f'<circle class="c ring" cx="{fx:.1f}" cy="{fy:.1f}" r="6.5"/>')
        body.append(
            text(fx - 11, fy + 4, dollars(fitted["dollars_per_1000_turns"]), "b", "end")
        )
    body.append(
        text(
            left + (width - left - right) / 2,
            height - 12,
            "Share of the history kept in view",
            "t2",
            "middle",
        )
    )
    return svg(
        width,
        height,
        "The cheapest policy depends on how users return",
        "Cost per 1,000 turns against the share of history kept in view, without and with keep-warm pings, "
        "for three return patterns. " + " ".join(summary),
        body,
    )


# -- one thread -------------------------------------------------------------


def thread(data):
    """One conversation as three lanes, with a curve from each turn to the earlier parts it needed."""
    turns = data["turns"]
    count = len(turns)
    width = 880
    left, right, top = 178, 150, 104
    span = (width - left - right) / count
    lanes = {
        "input": (top + 44, 8),
        "retrieval": (top + 76, 30),
        "output": (top + 130, 10),
    }
    height = top + 190
    marker = top + 14
    names = {"input": "Questions", "retrieval": "Retrieved text", "output": "Answers"}
    recipe = data["recipe"]
    kept = {
        "input": recipe["input_runs"],
        "retrieval": recipe["retrieval_runs"],
        "output": recipe["output_runs"],
    }

    def x(turn):
        return left + (turn - 0.5) * span

    def median(kind):
        values = sorted(t["tokens"][kind] for t in turns)
        return values[len(values) // 2]

    needing = [t for t in turns if isinstance(t["needs"], dict)]
    in_view = sum(t["needs"]["in_view"] for t in needing)
    unknown = sum(t["needs"] == "unknown" for t in turns)
    body = [
        text(
            24,
            30,
            "What each turn of one conversation needed from earlier turns",
            "big",
        ),
        text(
            24,
            49,
            f"A {count}-turn DocChat thread about a medicine label. {len(needing)} turns need something earlier, "
            f"{count - len(needing) - unknown} need nothing, {unknown} could not be assessed.",
            "t2",
        ),
        '<path class="arc cs" d="M24,72 q9,-10 18,0"/>',
        text(48, 74, "needed part still in view", "t2"),
        '<path class="arc os" d="M222,72 q9,-10 18,0"/>',
        text(246, 74, "needed part archived, fetched back", "t2"),
        '<circle class="box" cx="478" cy="70" r="3.5"/>',
        text(488, 74, "needs nothing earlier", "t2"),
        text(634, 74, "?", "tm"),
        text(646, 74, "not assessed", "t2"),
        text(left - 14, marker + 4, "Turn", "t2", "end"),
        text(width - right + 14, marker + 4, "Kept in view for", "t2"),
    ]
    for kind, (y0, tall) in lanes.items():
        body.append(text(left - 14, y0 + tall / 2 + 1, names[kind], "b", "end"))
        body.append(
            text(
                left - 14,
                y0 + tall / 2 + 16,
                f"about {median(kind):,} tokens a turn",
                "tm",
                "end",
            )
        )
        window = kept[kind]
        note = (
            "the whole thread"
            if window is None
            else "1 turn"
            if window == 1
            else f"{window} turns"
        )
        body.append(
            text(
                width - right + 14,
                y0 + tall / 2 + 4,
                note,
                "b" if kind == "retrieval" else "",
            )
        )
        for t in turns:
            body.append(
                f'<rect class="cell" x="{x(t["turn"]) - span * 0.36:.1f}" y="{y0}" width="{span * 0.72:.1f}" height="{tall}" rx="2"/>'
            )
    for t in turns:
        number = t["turn"]
        if number == 1 or number % 5 == 0:
            body.append(text(x(number), marker - 14, number, "tm", "middle"))
        if t["needs"] == "unknown":
            body.append(text(x(number), marker + 4, "?", "tm", "middle"))
        elif isinstance(t["needs"], dict):
            body.append(
                f'<circle class="{"c" if t["needs"]["in_view"] else "o"}" cx="{x(number):.1f}" cy="{marker}" r="4.5"/>'
            )
        else:
            body.append(
                f'<circle class="box" cx="{x(number):.1f}" cy="{marker}" r="3.5"/>'
            )
    # Curves last, so they sit above the cells they point at.
    for t in needing:
        cls = "c" if t["needs"]["in_view"] else "o"
        for kind, source in t["needs"]["parts"]:
            y0, tall = lanes[kind]
            x1, x2, y2 = x(t["turn"]), x(source), y0 + tall / 2
            bend = min(60, 18 + 5 * (t["turn"] - source))
            body.append(
                f'<rect class="{cls}" x="{x2 - span * 0.36:.1f}" y="{y0}" width="{span * 0.72:.1f}" height="{tall}" rx="2"/>'
            )
            body.append(
                f'<path class="arc {cls}s" d="M{x1:.1f},{marker + 5} C{x1:.1f},{marker + 5 + bend} {x2 + span * 0.9:.1f},{y2:.1f} {x2 + span * 0.36:.1f},{y2:.1f}"/>'
            )
    return svg(
        width,
        height,
        "What each turn of one conversation needed from earlier turns",
        f"A {count}-turn conversation drawn as three lanes: questions, retrieved text and answers. "
        f"{len(needing)} turns need an earlier part. For {in_view} of them that part is still in view under the fitted "
        f"policy, which keeps questions and answers for a long time and retrieved text for one turn. "
        f"The other {len(needing) - in_view} fetch it back.",
        body,
    )


# -- against OpenClaw -------------------------------------------------------


def openclaw(data):
    """Two bars per thread length: OpenClaw's defaults in grey, compache in blue."""
    rows = data["rows"]
    width, height = 880, 96 + 62 * len(rows) + 30
    x0, full, bar, top, step = 190, 520, 18, 100, 62
    longest = max(r["openclaw"] for r in rows)
    body = [
        text(24, 30, f"Cost per 1,000 turns · {data['model']}", "big"),
        text(
            24,
            49,
            "The calculator's default workload at three thread lengths, against OpenClaw's defaults.",
            "t2",
        ),
        '<rect class="n" x="24" y="64" width="10" height="10" rx="2"/>',
        text(40, 73.5, "OpenClaw", "t2"),
        '<rect class="c" x="116" y="64" width="10" height="10" rx="2"/>',
        text(132, 73.5, "compache", "t2"),
        f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 10}" y2="{top + step * len(rows) - 14}"/>',
    ]
    described = []
    for i, row in enumerate(rows):
        y0 = top + i * step
        body.append(
            text(x0 - 14, y0 + bar + 5, f"{row['turns']}-turn threads", "b", "end")
        )
        for j, (key, cls) in enumerate((("openclaw", "n"), ("compache", "c"))):
            length = max(2, row[key] / longest * full)
            yy = y0 + j * (bar + 2)
            body.append(
                f'<path class="{cls}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x0 + length + 8,
                    yy + bar - 4.5,
                    dollars(row[key]),
                    "b" if key == "compache" else "t2",
                )
            )
        saving = 1 - row["compache"] / row["openclaw"]
        body.append(text(width - 24, y0 + bar + 5, f"{saving:.0%} less", "h", "end"))
        described.append(
            f"{row['turns']}-turn threads: OpenClaw {dollars(row['openclaw'])}, compache {dollars(row['compache'])}"
        )
    return svg(
        width,
        height,
        f"Cost per 1,000 turns on {data['model']}: OpenClaw against compache",
        "Modeled cost per 1,000 turns at three thread lengths. "
        + "; ".join(described)
        + ".",
        body,
    )


# -- how it works -----------------------------------------------------------


def workflow():
    """Three steps: record threads, fit a policy, apply it on every request."""
    width, height = 880, 208
    steps = [
        (
            "1  Record",
            "Your threads",
            ["each message with its time", "labels for a sample of turns"],
        ),
        (
            "2  Fit",
            "compache fit",
            ["replays every candidate policy", "against the cache at your timestamps"],
        ),
        (
            "3  Apply",
            "policy.json",
            [
                "what stays in view, per kind of part",
                "how many pings keep the cache warm",
            ],
        ),
    ]
    box, gap, top = 252, 38, 58
    body = [text(24, 32, "Fit once on recorded threads, apply on every request", "big")]
    for i, (step, name, lines) in enumerate(steps):
        x0 = 24 + i * (box + gap)
        body.append(
            f'<rect class="box" x="{x0}" y="{top}" width="{box}" height="118" rx="8"/>'
        )
        body.append(text(x0 + 18, top + 28, step, "tm b"))
        body.append(text(x0 + 18, top + 54, name, "big"))
        body += [
            text(x0 + 18, top + 78 + 17 * k, line, "t2") for k, line in enumerate(lines)
        ]
        if i < len(steps) - 1:
            ax = x0 + box + 8
            body.append(
                f'<path class="line cs" d="M{ax},{top + 59} h{gap - 16} m-6,-6 l6,6 l-6,6"/>'
            )
    body.append(
        text(
            24,
            height - 12,
            "compache decides what each request holds and when a ping is due. Your application sends the requests and the pings.",
            "tm",
        )
    )
    return svg(
        width,
        height,
        "How compache works: record, fit, apply",
        "Recorded threads with timestamps and labels for a sample of turns go into compache fit, which replays every "
        "candidate policy against the cache. The result is a policy file that says what stays in view and how many "
        "keep-warm pings to send, applied on every request without a model call.",
        body,
    )


FIGURES = {
    "map": ("policy-map.svg", "policy-grid.json", policy_map),
    "thread": ("thread-dependencies.svg", "thread.json", thread),
    "openclaw": ("openclaw-cost.svg", "openclaw.json", openclaw),
    "workflow": ("workflow.svg", None, workflow),
}


def main(argv=None):
    names = (sys.argv[1:] if argv is None else argv) or list(FIGURES)
    for name in names:
        if name not in FIGURES:
            raise SystemExit(
                f"unknown figure {name!r}; choose from {', '.join(FIGURES)}"
            )
        target, source, draw = FIGURES[name]
        args = [json.loads((DATA / source).read_text("utf-8"))] if source else []
        (ASSETS / target).write_text(draw(*args), "utf-8")
        print(f"wrote {(ASSETS / target).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
