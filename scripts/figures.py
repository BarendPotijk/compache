#!/usr/bin/env python3
"""Draw the figures under docs/assets from the data files in docs/assets/data.

    python scripts/figures.py              # every figure
    python scripts/figures.py breaks thread   # some of: headline, requests, breaks, agents, sessions, thread, openclaw, preview, live, compare, workflow

Every figure paints its own surface and carries a light and a dark palette, so it
reads on either GitHub theme. The palette, typeface and marks are the calculator's.

Where the data comes from:
    policy-grid.json   python benchmarks/docchat/run.py grid --data <fast> <mixed> <slow> --output ...
    thread.json        python benchmarks/docchat/run.py thread --data <build> --thread <id> --recipe ... --output ...
    openclaw.json      node scripts/openclaw_comparison.mjs --json docs/assets/data/openclaw.json
    live.json          python benchmarks/live/run.py report --work <folder> --figure docs/assets/data/live.json
    answers.json       the graded answers on benchmarks/live/README.md
    requests.json      python scripts/request_sizes.py
    agents.json        python benchmarks/agents/public.py price <runs> --output ...
    agents-live.json   python benchmarks/agents/run.py report --work <folders> --figure ..., with its headings
    compare.json       python benchmarks/compare/run.py price --data <fast> <mixed> <slow> --sizes ... --output ...
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
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


# -- the headline -----------------------------------------------------------


def headline(data):
    """Two panels: what the provider billed and how many answers were right, without and with compache."""
    answers = json.loads((DATA / "answers.json").read_text("utf-8"))
    ways = [w for w in data["ways"] if answers["model"] in w["detail"]]

    def average(arm):
        # In decimals: a float mean lands on either side of half a cent, by Python version.
        costs = [
            Decimal(str(a["cost"])) for w in ways for a in w["arms"] if a["name"] == arm
        ]
        mean = sum(costs) / len(costs)
        return mean, f"${mean.quantize(Decimal('0.01'), ROUND_HALF_UP)}"

    (before, was), (after, now) = average("No policy"), average("Fitted on past traces")
    saving = f"{1 - after / before:.0%}"
    right = (answers["no_policy"], answers["past_traces"])
    width, height, x0, bar, top = 880, 182, 138, 22, 92
    panels = [
        (
            "Bill per 1,000 user messages",
            x0,
            (before, after),
            (was, now),
        ),
        (
            f"Right answers, out of {answers['turns']}",
            x0 + 400,
            right,
            (str(right[0]), str(right[1])),
        ),
    ]
    body = [
        text(24, 32, f"{saving} off the bill, with as many answers right", "big"),
        text(
            24,
            51,
            f"Live calls to {answers['model']}, billed by the provider: {answers['turns'] // len(ways)} messages "
            f"compache had not seen, each through {len(ways)} integrations.",
            "t2",
        ),
        text(x0 - 14, top + 38, "Without", "t2", "end"),
        text(x0 - 14, top + 70, "With compache", "b", "end"),
    ]
    for name, left, values, labels in panels:
        full = 260
        body += [
            text(left, top + 5, name, "h"),
            f'<line class="axis" x1="{left}" x2="{left}" y1="{top + 18}" y2="{top + 80}"/>',
        ]
        for j, (value, label) in enumerate(zip(values, labels, strict=True)):
            yy, length = top + 22 + j * 32, float(value / max(values)) * full
            body.append(
                f'<path class="{"c" if j else "n"}" d="M{left},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(text(left + length + 8, yy + 15.5, label, "h" if j else "t2"))
    return svg(
        width,
        height,
        f"{saving} off the bill, with as many answers right",
        f"Live calls to {answers['model']}. Billed per 1,000 user messages: {was} without compache, "
        f"{now} with. Answers graded right of {answers['turns']}: {right[0]} without, {right[1]} with.",
        body,
    )


# -- what one conversation sends ----------------------------------------------


def requests(data):
    """One column per message: the whole history in grey, what compache sends in blue, the cache underneath."""
    turns = data["turns"]
    width, height = 880, 388
    left, right, top, bottom = 58, 58, 108, 96
    span = (width - left - right) / len(turns)
    plot = height - top - bottom
    step = 20_000
    ceiling = step * -(-max(t["full"] for t in turns) // step)
    thick = min(22, span * 0.7)

    def y(value):
        return top + plot - value / ceiling * plot

    def column(cls, x0, value):
        h = value / ceiling * plot
        r = min(4, h)
        return (
            f'<path class="{cls}" d="M{x0:.1f},{top + plot:.1f} v-{h - r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},-{r:.1f} '
            f'h{thick - 2 * r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} v{h - r:.1f} z"/>'
        )

    total = sum(t["full"] for t in turns)
    sent = sum(min(t["sent"], t["full"]) for t in turns)
    body = [
        text(24, 30, "What one chat sends to the model, message by message", "big"),
        text(
            24,
            49,
            f"A {len(turns)}-message chat with an HR handbook, tokens per request. "
            f"compache sends {1 - sent / total:.0%} fewer in total.",
            "t2",
        ),
        '<rect class="cell" x="24" y="64" width="10" height="10" rx="2"/>',
        text(40, 73.5, "the whole history", "t2"),
        '<rect class="c" x="160" y="64" width="10" height="10" rx="2"/>',
        text(176, 73.5, "what compache sends", "t2"),
        '<circle class="o" cx="327" cy="69" r="4.5"/>',
        text(338, 73.5, "user away too long, cache expired: compache trims", "t2"),
        '<path class="line cs" d="M664,69 l4,4 l7,-8"/>',
        text(682, 73.5, "ping kept the cache warm", "t2"),
    ]
    for tick in range(0, ceiling + 1, step):
        body.append(
            f'<line class="{"axis" if tick == 0 else "grid"}" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
        )
        body.append(
            text(
                left - 8, y(tick) + 4, f"{tick // 1000}k" if tick else "0", "tm", "end"
            )
        )
    base = top + plot
    trims = 0
    archived = 0
    for i, t in enumerate(turns):
        middle = left + (i + 0.5) * span
        x0 = middle - thick / 2
        shown = min(t["sent"], t["full"])
        if shown < t["full"]:
            body.append(column("cell", x0, t["full"]))
        body.append(column("c", x0, shown))
        if t["archived"] > archived:
            trims, archived = trims + 1, t["archived"]
        if t["cache"] == "cold":
            body.append(
                f'<circle class="o" cx="{middle:.1f}" cy="{base + 16}" r="4.5"/>'
            )
        elif t["pings_before"]:
            body.append(
                f'<path class="line cs" d="M{middle - 5:.1f},{base + 16} l4,4 l7,-8"/>'
            )
        if t["turn"] == 1 or t["turn"] % 5 == 0:
            body.append(text(middle, base + 41, t["turn"], "tm", "middle"))
    last = turns[-1]
    end = left + (len(turns) - 0.5) * span + thick / 2
    body += [
        text(end + 8, y(last["full"]) + 9, f"{last['full'] / 1000:.0f}k", "t2"),
        text(end + 8, y(last["sent"]) + 9, f"{last['sent'] / 1000:.0f}k", "h"),
        text(
            left + (width - left - right) / 2, base + 60, "User message", "t2", "middle"
        ),
        text(
            24,
            height - 14,
            "Trimming rewrites the history, which breaks the cache. So compache trims when the cache is already gone.",
            "t2",
        ),
    ]
    cold = sum(t["cache"] == "cold" for t in turns)
    return svg(
        width,
        height,
        "What one chat sends to the model, message by message",
        f"A {len(turns)}-message chat. Sending the whole history grows to {last['full']:,} tokens on the last message, "
        f"compache sends {last['sent']:,} there. The cache had expired before {cold} messages and compache trimmed the history "
        f"{trims} times, each time on one of those. Before {sum(t['pings_before'] > 0 for t in turns)} other messages a ping kept the cache warm.",
        body,
    )


# -- how long users are away ---------------------------------------------------


def grouped(title, subtitle, legend, groups, description, size):
    """A panel of columns per group, the saving beside them and what was picked under them."""
    width, height = 880, 400
    left, right, top, bottom, gap = 24, 24, 112, 52, 20
    panel = (width - left - right - gap * (len(groups) - 1)) / len(groups)
    plot, thick, space = height - top - bottom, 46, 14
    base = top + plot
    highest = max(value for group in groups for _, value in group["columns"])
    body = [
        "<style>.n2{fill:#c3c2b7}@media (prefers-color-scheme:dark){.n2{fill:#5a5954}}</style>",
        text(24, 32, title, "big"),
        text(24, 51, subtitle, "t2"),
    ]
    at = 24
    for cls, label in legend:
        body.append(
            f'<rect class="{cls}" x="{at}" y="66" width="10" height="10" rx="2"/>'
        )
        body.append(text(at + 16, 75.5, label, "t2"))
        at += 6.2 * len(label) + 44
    for i, group in enumerate(groups):
        x0, columns = left + i * (panel + gap), group["columns"]
        wide = len(columns) * thick + (len(columns) - 1) * space
        body += [
            text(x0, top - 4, group["heading"], "h"),
            f'<line class="axis" x1="{x0:.1f}" x2="{x0 + wide:.1f}" y1="{base}" y2="{base}"/>',
        ]
        # The tallest column leaves room for its own number under the heading.
        scale = (plot - 40) / highest
        for k, (cls, value) in enumerate(columns):
            x, tall = x0 + k * (thick + space), value * scale
            body.append(
                f'<path class="{cls}" d="M{x:.1f},{base} v-{tall - 4:.1f} a4,4 0 0 1 4,-4 h{thick - 8} '
                f'a4,4 0 0 1 4,4 v{tall - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x + thick / 2,
                    base - tall - 8,
                    size(value),
                    "h" if cls == "c" else "t2",
                    "middle",
                )
            )
        # The saving stands beside the column it belongs to.
        beside = x0 + wide + space
        for row, (saved, against) in enumerate(group["beside"]):
            body += [
                text(beside, base - 56 + 36 * row, saved, "h"),
                text(beside, base - 41 + 36 * row, against, "tm"),
            ]
        body += [
            text(x0, base + 20, "compache picked", "tm"),
            text(x0, base + 36, group["picked"], "t2"),
        ]
    return svg(width, height, title, description, body)


def breaks(data):
    """Three groups of columns, one per return pattern: sending everything on each cache, and compache."""
    clocks = list(data["clocks"].values())

    def everything(clock, seconds):
        return next(
            r["dollars_per_1000_turns"]
            for r in clock["candidates"]
            if r["recipe"] is None
            and r["max_pings"] == 0
            and r["cache_ttl_seconds"] == seconds
        )

    groups, summary = [], []
    for clock in clocks:
        fitted = next(r for r in clock["candidates"] if r["fitted"])
        ours = fitted["dollars_per_1000_turns"]
        short, long = everything(clock, 300), everything(clock, 3600)
        pings = fitted["max_pings"]
        cache = "1-hour" if fitted["cache_ttl_seconds"] == 3600 else "5-minute"
        chose = f"{cache} cache, " + (f"up to {pings} pings" if pings else "no pings")
        groups.append(
            {
                "heading": f"Users back after {away(clock['median_idle_seconds'])}",
                "columns": [("n", short), ("n2", long), ("c", ours)],
                "beside": [
                    (f"{1 - ours / short:.0%} less", "than 5-minute"),
                    (f"{1 - ours / long:.0%} less", "than 1-hour"),
                ],
                "picked": chose,
            }
        )
        summary.append(
            f"Users back after {away(clock['median_idle_seconds'])}: everything {dollars(short)} on the 5-minute cache, "
            f"{dollars(long)} on the 1-hour cache, compache {dollars(ours)} with {chose}."
        )
    return grouped(
        "The longer users are away, the more compache saves",
        f"Cost per 1,000 messages. Same document chats, {data['model_name']}, "
        f"{clocks[0]['test_turns']:,} messages compache had not seen.",
        (
            ("n", "Send everything, 5-minute cache"),
            ("n2", "Send everything, 1-hour cache"),
            ("c", "compache"),
        ),
        groups,
        "Cost per 1,000 messages of sending everything and of compache, for three return patterns. "
        + " ".join(summary),
        dollars,
    )


def agents(data):
    """Three groups of columns, one per way of working: an agent's calls as they are, and with compache."""
    groups, summary = [], []
    for row in data["figure"]:
        ours, whole, best = row["policy"], row["whole"], row["cached"]
        picked = f"{row['rounds']} rounds in full, "
        pings = row["user_pings"]
        if row["header_ttl"] > row["ttl"]:
            picked += "1-hour instructions"
        elif pings:
            picked += f"{pings} ping{'' if pings == 1 else 's'}"
        else:
            picked += f"{'1-hour' if row['ttl'] == 3600 else '5-minute'} cache"
        groups.append(
            {
                "heading": row["heading"],
                "columns": [("n", whole), ("n2", best), ("c", ours)],
                "beside": [
                    (f"{1 - ours / whole:.0%} less", "than everything"),
                    (f"{1 - ours / best:.0%} less", "than cached"),
                ],
                "picked": picked,
            }
        )
        summary.append(
            f"{row['heading']}: everything {dollars(whole)}, cached at its best {dollars(best)}, "
            f"compache {dollars(ours)} with {picked}."
        )
    return grouped(
        "What an agent saves follows from how it is worked with",
        f"Cost per 1,000 model calls on {data['model_name']}. {data['runs']} public coding-agent runs, "
        "the half compache had not seen.",
        (
            ("n", "Send everything"),
            ("n2", "Send everything, cached at its best"),
            ("c", "compache"),
        ),
        groups,
        "Cost per 1,000 model calls of sending everything and of compache, for three ways of working. "
        + " ".join(summary),
        dollars,
    )


def sessions(data):
    """A panel per long session, one line per run: what the provider had billed after each prompt."""
    called = {
        "none": "alone",
        "headroom": "Headroom",
        "kompressor": "Kompressor",
        "compache": "compache",
    }
    width, height = 880, 372
    left, right, top, bottom, gap = 62, 134, 116, 56, 186
    panel = (width - left - right - gap) / 2
    plot, base = height - top - bottom, height - bottom
    body = [
        "<style>.n2s{stroke:#b5b3a9}.n2{fill:#b5b3a9}.over{paint-order:stroke;stroke-width:6;stroke-linejoin:round}"
        "@media (prefers-color-scheme:dark){.n2s{stroke:#66655f}.n2{fill:#66655f}}</style>",
        text(24, 32, data["title"], "big"),
        text(24, 51, data["subtitle"], "t2"),
    ]
    at = 24
    for cls, label in (
        ("ns", "Claude Code alone"),
        ("n2s", "with Headroom or Kompressor"),
        ("cs", "with compache"),
    ):
        body.append(f'<path class="line {cls}" d="M{at},71 h16"/>')
        body.append(text(at + 23, 75.5, label, "t2"))
        at += 6.2 * len(label) + 48
    said = []
    for i, part in enumerate(data["panels"]):
        x0, runs = left + i * (panel + gap), part["runs"]
        count = max(len(run["curve"]) for run in runs)
        highest = max(run["curve"][-1] for run in runs)
        step = next(s for s in (0.1, 0.2, 0.5, 1.0, 2.0) if highest / s <= 4.5)
        ceiling = step * -(-highest // step)

        def x(k, x0=x0, count=count):
            return x0 + (k - 1) / (count - 1) * panel

        def y(value, ceiling=ceiling):
            return base - value / ceiling * plot

        body.append(text(x0 - 38, top - 14, part["heading"], "h"))
        for j in range(round(ceiling / step) + 1):
            at = y(j * step)
            body.append(
                f'<line class="{"grid" if j else "axis"}" x1="{x0:.1f}" x2="{x0 + panel:.1f}" '
                f'y1="{at:.1f}" y2="{at:.1f}"/>'
            )
            label = f"${j * step:.2f}" if j else "0"
            body.append(text(x0 - 9, at + 4, label, "tm", "end"))
        for k in (1, 5, 10, 15, count):
            if k == count or count - k > 2:
                body.append(text(x(k), base + 20, k, "tm", "middle"))
        body.append(text(x0 + panel / 2, base + 40, "Prompt", "t2", "middle"))
        order, ends = ("headroom", "kompressor", "none", "compache"), []
        for run in sorted(runs, key=lambda r: order.index(r["arm"])):
            points = [(x(k), y(v)) for k, v in enumerate(run["curve"], 1)]
            cls = {"compache": "c", "none": "n"}.get(run["arm"], "n2")
            path = " ".join(f"{a:.1f},{b:.1f}" for a, b in points)
            body.append(f'<polyline class="line {cls}s" points="{path}"/>')
            ends.append((*points[-1], cls, run))
        for a, b, cls, _ in ends:
            body.append(f'<circle class="{cls} ring" cx="{a:.1f}" cy="{b:.1f}" r="5"/>')
        # A label at the end of each line, moved apart where two would touch, with a line to its own.
        free = float("-inf")
        for a, b, _, run in sorted(ends, key=lambda end: end[1]):
            cost, last = run["curve"][-1], len(run["curve"])
            words = f"{called[run['arm']]} ${cost:.2f}"
            style = "h" if run["arm"] == "compache" else "t2"
            if count - last > 2:
                # It ends inside the panel: over the lines there, or just above its own end.
                top_one = all(
                    other["curve"][last - 1] <= cost
                    for other in runs
                    if len(other["curve"]) >= last
                )
                # What it stands over shows no line.
                style += " ring over"
                body.append(
                    text(a - 10, b, words, style, "end")
                    if top_one
                    else text(a + 3, b - 11, words, style)
                )
                continue
            place = max(b + 4, free + 16)
            free = place
            if place - b - 4 > 2:
                body.append(
                    f'<line class="axis" x1="{a + 7:.1f}" y1="{b + 2:.1f}" x2="{a + 13:.1f}" y2="{place - 4:.1f}"/>'
                )
            body.append(text(a + 16, place, words, style))
        said.append(
            f"{part['heading']}: "
            + ", ".join(
                f"{called[r['arm']]} ${r['curve'][-1]:.2f} after {len(r['curve'])} prompts"
                for r in runs
            )
        )
    return svg(
        width,
        height,
        data["title"],
        f"{data['subtitle']} {'. '.join(said)}.",
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


# -- the calculator's link preview ------------------------------------------------


def preview(data):
    """What a shared calculator link shows: the calculator's default workload, priced both ways.

    Link previews need a bitmap. `site/preview.png` is this figure, taken with a browser:

        chrome --headless --window-size=1200,630 --screenshot=site/preview.png docs/assets/preview.svg
    """
    row = data["rows"][0]
    width, height, x0, full, bar = 1200, 630, 80, 820, 74
    longest = max(row["openclaw"], row["compache"])
    style = (
        'text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:#0b0b0b}'
        ".two{fill:#52514e}.c{fill:#2a78d6}.n{fill:#898781}"
    )

    def line(x, y, size, content, cls="", weight=400):
        return (
            f'<text x="{x:.0f}" y="{y}" font-size="{size}" font-weight="{weight}"'
            f"{f' class={chr(34)}{cls}{chr(34)}' if cls else ''}>{escape(content)}</text>"
        )

    body = [
        f'<rect class="c" x="{x0}" y="66" width="30" height="30" rx="8"/>',
        line(x0 + 46, 94, 36, "compache", weight=650),
        line(x0, 186, 58, "What would your long chats cost?", weight=650),
        line(
            x0,
            238,
            28,
            f"{data['model']} · {row['turns']}-turn threads · 25k retrieved tokens per turn",
            "two",
        ),
    ]
    for i, (key, name, cls) in enumerate(
        (("openclaw", "OpenClaw’s defaults", "n"), ("compache", "compache", "c"))
    ):
        top = 306 + i * 118
        length = row[key] / longest * full
        body += [
            line(x0, top - 10, 24, name, "two"),
            f'<rect class="{cls}" x="{x0}" y="{top}" width="{length:.1f}" height="{bar}" rx="8"/>',
            line(x0 + length + 22, top + 52, 44, dollars(row[key]), weight=650),
        ]
    saving = 1 - row["compache"] / row["openclaw"]
    body += [
        line(x0, 568, 32, f"{saving:.0%} less per 1,000 turns, modeled.", weight=650),
        line(x0 + 562, 568, 32, "Put in your own numbers.", "two"),
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t">\n'
        f'<title id="t">compache cost calculator: {escape(data["model"])}, {row["turns"]}-turn threads, '
        f"OpenClaw {dollars(row['openclaw'])}, compache {dollars(row['compache'])} per 1,000 turns</title>\n"
        f"<style>{style}</style>\n"
        f'<rect width="{width}" height="{height}" fill="#f9f9f7"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


# -- live model calls ---------------------------------------------------------


def live(data):
    """One group of bars per way in: what the provider billed under no policy and under each policy."""
    ways = data["ways"]
    rows = sum(len(way["arms"]) for way in ways)
    bar, pitch, gap, top = 14, 21, 26, 104
    width, height = 880, top + rows * pitch + len(ways) * gap + 10
    x0, full = 392, 356
    longest = max(arm["cost"] for way in ways for arm in way["arms"])
    body = [
        text(
            24,
            30,
            "What the provider billed per 1,000 turns, through every way in",
            "big",
        ),
        text(24, 49, data["subtitle"], "t2"),
        '<rect class="n" x="24" y="64" width="10" height="10" rx="2"/>',
        text(
            40, 73.5, "No policy: the application as it is, with its own caching", "t2"
        ),
        '<rect class="c" x="396" y="64" width="10" height="10" rx="2"/>',
        text(412, 73.5, "With a compache policy", "t2"),
    ]
    described, y0 = [], top
    for way in ways:
        arms, base = way["arms"], way["arms"][0]["cost"]
        body.append(text(24, y0 + bar - 2.5, way["name"], "b"))
        body += [
            text(24, y0 + bar + 15.5 + 16 * k, line, "tm")
            for k, line in enumerate(way["detail"])
        ]
        body.append(
            f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y0 - 5}" y2="{y0 + len(arms) * pitch - 2}"/>'
        )
        said = []
        for j, arm in enumerate(arms):
            yy, length = y0 + j * pitch, max(6, arm["cost"] / longest * full)
            body.append(text(x0 - 12, yy + bar - 2.5, arm["name"], "t2", "end"))
            body.append(
                f'<path class="{"c" if j else "n"}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x0 + length + 8,
                    yy + bar - 2.5,
                    dollars(arm["cost"]),
                    "b" if j else "t2",
                )
            )
            if j:
                change = f"{arm['cost'] / base - 1:+.0%}".replace("-", "\u2212")
                body.append(text(x0 + length + 54, yy + bar - 2.5, change, "t2"))
            said.append(f"{arm['name'].lower()} {dollars(arm['cost'])}")
        described.append(
            f"{way['name']}, {', '.join(way['detail'])}: " + ", ".join(said)
        )
        y0 += len(arms) * pitch + gap
    return svg(
        width,
        height,
        "What the provider billed per 1,000 turns through every way in, with and without a compache policy",
        f"{data['subtitle']} {'. '.join(described)}.",
        body,
    )


# -- other tools ----------------------------------------------------------------


def compare(data):
    """One group of bars per return pattern: what each tool's requests cost, and compache's."""
    clocks = data["clocks"]
    rows = sum(len(clock["arms"]) for clock in clocks)
    bar, pitch, gap, top = 14, 21, 26, 92
    width, height = 880, top + rows * pitch + len(clocks) * gap + 4
    x0, full = 388, 372
    longest = max(arm["cost"] for clock in clocks for arm in clock["arms"])
    body = [
        text(24, 30, "Cost per 1,000 turns of document chats, by tool", "big"),
        text(
            24,
            49,
            f"DocChat's held-out chats on {data['model']}, priced by the provider's cache rules. Headroom and Kompressor were",
            "t2",
        ),
        text(
            24,
            65,
            "run on the retrieved text. Cache kept warm is the cheapest of ten settings. compache's price includes fetching back.",
            "t2",
        ),
    ]
    described, y0 = [], top
    for clock in clocks:
        arms, base = clock["arms"], clock["arms"][0]["cost"]
        body.append(
            text(
                24,
                y0 + bar - 2.5,
                f"Users back after {away(clock['away_seconds'])}",
                "b",
            )
        )
        body.append(text(24, y0 + bar + 13.5, "at the median", "tm"))
        body.append(
            f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y0 - 5}" y2="{y0 + len(arms) * pitch - 2}"/>'
        )
        said = []
        for j, arm in enumerate(arms):
            ours = arm["name"] == "compache"
            yy, length = y0 + j * pitch, max(6, arm["cost"] / longest * full)
            body.append(
                text(x0 - 12, yy + bar - 2.5, arm["name"], "b" if ours else "t2", "end")
            )
            body.append(
                f'<path class="{"c" if ours else "n"}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x0 + length + 8,
                    yy + bar - 2.5,
                    dollars(arm["cost"]),
                    "b" if ours else "t2",
                )
            )
            if j:
                change = f"{arm['cost'] / base - 1:+.0%}".replace("-", "\u2212")
                body.append(text(x0 + length + 54, yy + bar - 2.5, change, "t2"))
            said.append(f"{arm['name']} {dollars(arm['cost'])}")
        described.append(
            f"Users back after {away(clock['away_seconds'])}: " + ", ".join(said)
        )
        y0 += len(arms) * pitch + gap
    return svg(
        width,
        height,
        "Cost per 1,000 turns of document chats: other tools against compache",
        f"DocChat's held-out chats on {data['model']}. {'. '.join(described)}.",
        body,
    )


def tools(data):
    """Two panels, one bar per tool in each: what the provider billed, and how many answers were right."""
    arms, base = data["arms"], data["arms"][0]
    bar, pitch, top = 16, 27, 106
    width, height = 880, top + len(arms) * pitch + 12
    panels = [
        (262, 232, "cost", "Bill per 1,000 turns"),
        (656, 150, "correct", f"Right answers, out of {base['graded']}"),
    ]
    title = "Live: the same conversations through each tool"
    body = [
        text(24, 30, title, "big"),
        text(
            24,
            49,
            f"{data['conversations']} held-out DocChat conversations, {base['turns']} turns on {data['model']}, "
            "at the users' own pace. Billed by the provider.",
            "t2",
        ),
    ]
    for x0, full, key, name in panels:
        longest = max(arm[key] for arm in arms)
        body += [
            text(x0, top - 18, name, "h"),
            f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 6}" y2="{top + len(arms) * pitch - 5}"/>',
        ]
        for j, arm in enumerate(arms):
            ours, with_ours = arm["name"] == "compache", "compache" in arm["name"]
            yy, length = top + j * pitch, max(6, arm[key] / longest * full)
            label = f"${arm[key]:.2f}" if key == "cost" else str(arm[key])
            body += [
                f'<path class="{"c" if with_ours else "n"}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>',
                text(x0 + length + 8, yy + bar - 4, label, "b" if ours else "t2"),
            ]
            if key != "cost":
                continue
            body.append(
                text(x0 - 12, yy + bar - 4, arm["name"], "b" if ours else "t2", "end")
            )
            if j:
                change = f"{arm[key] / base[key] - 1:+.0%}".replace("-", "\u2212")
                body.append(text(x0 + length + 54, yy + bar - 4, change, "t2"))
    said = [
        f"{arm['name']} ${arm['cost']:.2f}, {arm['correct']} of {arm['graded']} answers right"
        for arm in arms
    ]
    return svg(width, height, title, ". ".join(said) + ".", body)


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
    "headline": ("headline.svg", "live.json", headline),
    "requests": ("requests.svg", "requests.json", requests),
    "breaks": ("breaks-cost.svg", "policy-grid.json", breaks),
    "agents": ("agents-cost.svg", "agents.json", agents),
    "sessions": ("agents-live.svg", "agents-live.json", sessions),
    "thread": ("thread-dependencies.svg", "thread.json", thread),
    "openclaw": ("openclaw-cost.svg", "openclaw.json", openclaw),
    "preview": ("preview.svg", "openclaw.json", preview),
    "live": ("live-cost.svg", "live.json", live),
    "compare": ("compare-cost.svg", "compare.json", compare),
    "tools": ("compare-live.svg", "compare-live.json", tools),
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
