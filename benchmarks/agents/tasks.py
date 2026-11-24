#!/usr/bin/env python3
"""Long tasks for coding agents: a repository that departs from its own specification.

    python tasks.py <folder> --seed 1

A task is a Python package of fourteen modules, a SPEC.md that says in words what
each one does, and a test suite that shows some of what is wrong. Every module
departs from its section in one or two details. The grade is a hidden suite: how
many modules then do what the specification says. Constants, cases and the choice
of modules follow from the seed, so a task cannot be known by heart.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

PROMPT = (
    "SPEC.md says what each module in rules/ has to do. Every module departs from its section in one or "
    "two details. Work through them one at a time in alphabetical order: read the module, compare it with "
    "its section in SPEC.md, correct the module, and run `python3 -m unittest -v` after each one. The tests "
    "cover only a part of the specification, so go by SPEC.md. Change nothing but the modules in rules/. "
    "When every module does what SPEC.md says, reply with the single word DONE."
)


def rule(spec: str, good: str, bad: str, cases: str, shown: str) -> dict:
    """One module: its section, what is right, what is in the repository, and inputs to check it with."""
    clean = textwrap.dedent
    return {
        "spec": clean(spec).strip(),
        "good": clean(good).strip(),
        "bad": clean(bad).strip(),
        # Python expressions: a list of argument tuples, drawn for the hidden and for the shown tests.
        "cases": clean(cases).strip(),
        "shown": clean(shown).strip(),
    }


def rules(r: random.Random) -> dict[str, dict]:
    """The modules, with constants drawn from `r`."""
    grace, rate, cap = r.randint(3, 9), r.choice([2, 3, 4]), r.choice([15, 20, 25])
    free, step, base, per = (
        r.choice([40, 50, 75]),
        r.choice([2, 5]),
        r.choice([4, 6, 7]),
        r.choice([1, 2]),
    )
    off, fixed, least = r.choice([10, 15]), r.choice([5, 8]), r.choice([60, 80])
    euros, day, double = r.choice([4, 5, 8]), r.randrange(5), r.choice([2, 3])
    first, second, low, mid, high = (
        r.choice([1000, 2000]),
        r.choice([5000, 8000]),
        3,
        5,
        8,
    )
    length, sep, mod = r.choice([8, 10]), r.choice("-_"), r.choice([7, 11])
    attempts, start, most = r.choice([5, 6]), r.choice([2, 3]), r.choice([40, 60])
    width, tail = r.choice([24, 30]), r.choice([8, 10])
    open_, close = r.choice([7, 8]), r.choice([19, 20])
    months, share = r.choice([3, 6]), r.choice([25, 40])
    keep, near = r.choice([5, 7]), r.choice([2, 3])
    return {
        "late_fee": rule(
            f"""
            `late_fee(amount_cents, days_late)` gives the fee in cents for a payment that comes late.
            The first {grace} days are free. After them {rate}% of the amount is due for every week that has
            begun, so day {grace + 1} starts the first week and day {grace + 8} the second. The fee never
            exceeds {cap}% of the amount. Parts of a cent are rounded half up. Nothing is due on an amount
            of zero or less, or on a payment that is not late.
            """,
            f"""
            def late_fee(amount_cents, days_late):
                late = days_late - {grace}
                if late <= 0 or amount_cents <= 0:
                    return 0
                weeks = -(-late // 7)
                percent = min(weeks * {rate}, {cap})
                return (amount_cents * percent * 2 + 100) // 200
            """,
            f"""
            def late_fee(amount_cents, days_late):
                late = days_late - {grace}
                if late <= 0 or amount_cents <= 0:
                    return 0
                weeks = late // 7 + 1
                percent = min(weeks * {rate}, {cap})
                return amount_cents * percent // 100
            """,
            f"[(a, d) for a in (0, -5, 999, 1250, 33333) for d in ({grace}, {grace + 1}, {grace + 7}, {grace + 8}, {grace + 14}, 90)]",
            f"[(1250, {grace + 7}), (33333, {grace + 14}), (1000, {grace})]",
        ),
        "shipping": rule(
            f"""
            `shipping(weight_grams, order_euros, island)` gives the price in euros. An order of {free} euros or
            more ships free, unless it goes to an island. The price is {base} euros for the first {step} kilos and
            {per} more for every further kilo that has begun. To an island 3 euros are added, after everything
            else, and to a free order it is those 3 euros alone. A weight of zero or less raises ValueError.
            """,
            f"""
            def shipping(weight_grams, order_euros, island):
                if weight_grams <= 0:
                    raise ValueError("weight")
                over = max(0, weight_grams - {step * 1000})
                price = {base} + {per} * -(-over // 1000)
                if order_euros >= {free}:
                    price = 0
                return price + 3 if island else price
            """,
            f"""
            def shipping(weight_grams, order_euros, island):
                if weight_grams < 0:
                    raise ValueError("weight")
                over = max(0, weight_grams - {step * 1000})
                price = {base} + {per} * (over // 1000)
                if order_euros > {free}:
                    price = 0
                return price + 3 if island else price
            """,
            f"[(w, o, i) for w in (0, 1, {step * 1000}, {step * 1000 + 1}, {step * 1000 + 2500}) for o in ({free - 1}, {free}, 500) for i in (False, True)]",
            f"[({step * 1000 + 1}, 1, False), (100, {free}, True)]",
        ),
        "voucher": rule(
            f"""
            `voucher(total_cents, percent_code, fixed_code)` gives what is left to pay. The percent code takes
            {off}% off and counts only when the total is {least} euros or more. The fixed code takes {fixed} euros
            off what is left after the percent code, and counts whatever the total. What a code takes off is
            rounded down to a whole cent. The result is never below zero.
            """,
            f"""
            def voucher(total_cents, percent_code, fixed_code):
                left = total_cents
                if percent_code and total_cents >= {least * 100}:
                    left -= total_cents * {off} // 100
                if fixed_code:
                    left -= {fixed * 100}
                return max(0, left)
            """,
            f"""
            def voucher(total_cents, percent_code, fixed_code):
                left = total_cents
                if fixed_code:
                    left -= {fixed * 100}
                if percent_code and left >= {least * 100}:
                    left -= left * {off} // 100
                return left
            """,
            f"[(t, p, f) for t in (0, 100, {fixed * 100 - 1}, {least * 100 - 1}, {least * 100}, {least * 100 + 7}, 99999) for p in (False, True) for f in (False, True)]",
            f"[({least * 100}, True, True), (100, False, True)]",
        ),
        "points": rule(
            f"""
            `points(amount_cents, weekday, discounted)` gives the loyalty points of a purchase: one for every
            full {euros} euros. On weekday {day} (Monday is 0) points count {double} times. A discounted purchase
            earns none, except on that weekday, when it earns the single amount. Amounts below zero are returns
            and take points away: the same number as the purchase would have earned.
            """,
            f"""
            def points(amount_cents, weekday, discounted):
                whole = abs(amount_cents) // {euros * 100}
                special = weekday == {day}
                if discounted:
                    earned = whole if special else 0
                else:
                    earned = whole * {double} if special else whole
                return -earned if amount_cents < 0 else earned
            """,
            f"""
            def points(amount_cents, weekday, discounted):
                whole = amount_cents // {euros * 100}
                special = weekday == {day + 1}
                if discounted:
                    return 0
                return whole * {double} if special else whole
            """,
            f"[(a, w, d) for a in (0, {euros * 100 - 1}, {euros * 100}, 12345, -{euros * 100 + 50}, -9999) for w in ({day}, {(day + 1) % 7}) for d in (False, True)]",
            f"[(12345, {day}, False), (-{euros * 100 + 50}, 6, False)]",
        ),
        "commission": rule(
            f"""
            `commission(sales)` gives a seller's commission in euros, rounded half up to a whole euro. It is
            {low}% of the part of the sales up to {first}, {mid}% of the part from there to {second}, and {high}% of
            the part beyond. Each rate holds for its own part only. Sales are whole euros. Below zero it is zero.
            """,
            f"""
            def commission(sales):
                if sales <= 0:
                    return 0
                parts = (min(sales, {first}) * {low}
                         + max(0, min(sales, {second}) - {first}) * {mid}
                         + max(0, sales - {second}) * {high})
                return (parts * 2 + 100) // 200
            """,
            f"""
            def commission(sales):
                if sales <= 0:
                    return 0
                if sales <= {first}:
                    parts = sales * {low}
                elif sales <= {second}:
                    parts = sales * {mid}
                else:
                    parts = sales * {high}
                return parts // 100
            """,
            f"[(s,) for s in (-3, 0, 17, 50, {first}, {first + 1}, {first + 750}, {second}, {second + 1}, {second + 5555})]",
            f"[({first + 750},), (50,)]",
        ),
        "sku": rule(
            f"""
            `sku(text)` gives an article number in its written form: letters in upper case, every space, dot and
            `{sep}` left out, and zeros put in front until there are {length} characters. Then comes one more
            character: the sum of the places of its characters, counted from 1 at the left, times their value,
            modulo {mod}, written as a digit, or as X for 10. A digit's value is itself, a letter's is its place in
            the alphabet, A being 1. Text that is longer than {length} characters before the padding, or holds
            anything but letters and digits, raises ValueError.
            """,
            f"""
            def sku(text):
                plain = "".join(c for c in text.upper() if c not in " .{sep}")
                if len(plain) > {length} or any(not c.isascii() or not c.isalnum() for c in plain):
                    raise ValueError(text)
                plain = plain.rjust({length}, "0")
                total = sum(place * _value(c) for place, c in enumerate(plain, 1))
                left = total % {mod}
                return plain + ("X" if left == 10 else str(left))


            def _value(c):
                return int(c) if c.isdigit() else ord(c) - ord("A") + 1
            """,
            f"""
            def sku(text):
                plain = "".join(c for c in text.upper() if c not in " {sep}")
                if len(plain) > {length} or any(not c.isascii() or not c.isalnum() for c in plain):
                    raise ValueError(text)
                plain = plain.ljust({length}, "0")
                total = sum(place * _value(c) for place, c in enumerate(plain))
                left = total % {mod}
                return plain + ("X" if left == 10 else str(left))


            def _value(c):
                return int(c) if c.isdigit() else ord(c) - ord("A") + 1
            """,
            f"""[(t,) for t in ("ab 12", "A{sep}7.c", "zz9", "", "{"1" * length}", "{"1" * (length + 1)}", "a*b", "q4{sep}44")]""",
            f"""[("ab 12",), ("zz9",)]""",
        ),
        "backoff": rule(
            f"""
            `backoff(attempt)` gives how many seconds to wait after a failed attempt, the first being 1. It is
            {start} seconds after the first, and twice as long after each one that follows, but never more than
            {most}. After attempt {attempts} and any later one the result is None: no further attempt is made.
            An attempt below 1 raises ValueError.
            """,
            f"""
            def backoff(attempt):
                if attempt < 1:
                    raise ValueError(attempt)
                if attempt >= {attempts}:
                    return None
                return min({most}, {start} * 2 ** (attempt - 1))
            """,
            f"""
            def backoff(attempt):
                if attempt < 1:
                    raise ValueError(attempt)
                if attempt > {attempts}:
                    return None
                return min({most}, {start} * 2 ** attempt)
            """,
            f"[(a,) for a in range(-1, {attempts + 3})]",
            f"[(1,), ({attempts},)]",
        ),
        "shorten": rule(
            f"""
            `shorten(text)` makes a title fit. Runs of white space count as one space and are dropped at the ends.
            A title of at most {width} characters stays. A longer one keeps its beginning and its last {tail}
            characters, with `...` in between, and is then exactly {width} characters long. Where the beginning
            would end on a space, that space is dropped, and the result is one shorter.
            """,
            f"""
            def shorten(text):
                text = " ".join(text.split())
                if len(text) <= {width}:
                    return text
                return text[:{width - tail - 3}].rstrip(" ") + "..." + text[-{tail}:]
            """,
            f"""
            def shorten(text):
                text = text.strip()
                if len(text) < {width}:
                    return text
                return text[:{width - tail}] + "..." + text[-{tail}:]
            """,
            f"""[(t,) for t in ("", "  a   b ", "x" * {width}, "x" * {width + 1}, "{"ab " * 14}", "one two three four five six seven eight", "a" * {width - tail - 4} + " " + "b" * 40)]""",
            f"""[("x" * {width},), ("  a   b ",)]""",
        ),
        "opening": rule(
            f"""
            `opening(weekday, hour)` says whether the shop is open. On weekdays 0 to 4 it opens at {open_} and
            closes at {close}: it is open at {open_} and no longer at {close}. On weekday 5 it opens one hour later
            and closes three hours earlier. On weekday 6 it is closed. An hour outside 0 to 23 or a weekday
            outside 0 to 6 raises ValueError.
            """,
            f"""
            def opening(weekday, hour):
                if not 0 <= weekday <= 6 or not 0 <= hour <= 23:
                    raise ValueError((weekday, hour))
                if weekday == 6:
                    return False
                if weekday == 5:
                    return {open_ + 1} <= hour < {close - 3}
                return {open_} <= hour < {close}
            """,
            f"""
            def opening(weekday, hour):
                if not 0 <= weekday <= 6 or not 0 <= hour <= 24:
                    raise ValueError((weekday, hour))
                if weekday == 6:
                    return False
                if weekday == 5:
                    return {open_ + 1} <= hour < {close - 2}
                return {open_} <= hour <= {close}
            """,
            f"[(w, h) for w in (-1, 0, 4, 5, 6, 7) for h in (-1, 0, {open_ - 1}, {open_}, {open_ + 1}, {close - 4}, {close - 3}, {close - 2}, {close - 1}, {close}, 23, 24)]",
            f"[(0, {close}), (5, {close - 3})]",
        ),
        "refund": rule(
            f"""
            `refund(price_cents, months_used, opened)` gives what is paid back. Unopened goods are paid back in
            full within {months} months, the last of them included. Opened goods lose {share}% of their price at
            once and an equal part of what is left for every month of use, so that nothing remains after
            {months} months. Parts of a cent are rounded down. After {months} months nothing is paid back.
            """,
            f"""
            def refund(price_cents, months_used, opened):
                if months_used > {months}:
                    return 0
                if not opened:
                    return price_cents
                left = price_cents * {100 - share}
                return left * ({months} - months_used) // ({months} * 100)
            """,
            f"""
            def refund(price_cents, months_used, opened):
                if months_used >= {months}:
                    return 0
                if not opened:
                    return price_cents
                left = price_cents * {share} // 100
                return left * ({months} - months_used) // {months}
            """,
            f"[(p, m, o) for p in (0, 999, 10000, 12345) for m in (0, 1, {months - 1}, {months}, {months + 1}) for o in (False, True)]",
            f"[(10000, {months}, False), (12345, 1, True)]",
        ),
        "rotation": rule(
            f"""
            `rotation(names, week)` gives who is on duty in a week, the first being 1: the names take turns in the
            order given. Every {keep}th week nobody is on duty, the result is None, and nobody's turn is used up:
            whoever would have been next is next in the week after. An empty list gives None. A week below 1
            raises ValueError.
            """,
            f"""
            def rotation(names, week):
                if week < 1:
                    raise ValueError(week)
                if not names or week % {keep} == 0:
                    return None
                turn = week - 1 - week // {keep}
                return names[turn % len(names)]
            """,
            f"""
            def rotation(names, week):
                if week < 1:
                    raise ValueError(week)
                if not names or week % {keep} == 0:
                    return None
                return names[week % len(names)]
            """,
            f"""[(n, w) for n in ([], ["a"], ["a", "b", "c"]) for w in (0, 1, 2, {keep - 1}, {keep}, {keep + 1}, {2 * keep}, {2 * keep + 1}, 50)]""",
            f"""[(["a", "b", "c"], 1), (["a", "b", "c"], {keep + 1})]""",
        ),
        "batches": rule(
            f"""
            `batches(stock, wanted)` says what to take from which batch. `stock` is a list of `(name, count, days)`,
            days being how long it still keeps. What keeps shortest goes first, and of two that keep equally long
            the one that comes first by name. A batch that keeps fewer than {near} days is never used. The result
            is a list of `(name, count)` in the order taken, without empty ones. Where there is not enough, the
            result is None.
            """,
            f"""
            def batches(stock, wanted):
                usable = sorted((b for b in stock if b[2] >= {near}), key=lambda b: (b[2], b[0]))
                taken = []
                for name, count, _ in usable:
                    part = min(count, wanted)
                    if part > 0:
                        taken.append((name, part))
                        wanted -= part
                return taken if wanted <= 0 else None
            """,
            f"""
            def batches(stock, wanted):
                usable = sorted((b for b in stock if b[2] > {near}), key=lambda b: (b[2], b[1]))
                taken = []
                for name, count, _ in usable:
                    part = min(count, wanted)
                    taken.append((name, part))
                    wanted -= part
                return taken
            """,
            f"""[(s, w) for s in ([("b", 4, 9), ("a", 3, 9), ("c", 2, {near}), ("d", 9, {near - 1})], [("x", 0, 5), ("y", 5, 5)], []) for w in (0, 1, 2, 5, 9, 10)]""",
            f"""[([("b", 4, 9), ("a", 3, 9), ("c", 2, {near})], 5), ([], 3)]""",
        ),
        "versions": rule(
            """
            `versions(a, b)` compares two versions such as `1.12.0` and `1.9-rc2` and gives -1, 0 or 1. Numbers are
            compared as numbers, from the left, and a missing one counts as 0. A version with a part after a `-`
            comes before the same version without. Two such parts are compared by their letters first, then by
            their number, where a missing number counts as 0.
            """,
            """
            import re


            def versions(a, b):
                left, right = _key(a), _key(b)
                return (left > right) - (left < right)


            def _key(text):
                numbers, _, pre = text.partition("-")
                found = [int(n) for n in numbers.split(".")]
                found += [0] * (6 - len(found))
                if not pre:
                    return found, 1, "", 0
                letters, number = re.fullmatch(r"([a-zA-Z]*)(\\d*)", pre).groups()
                return found, 0, letters, int(number or 0)
            """,
            """
            import re


            def versions(a, b):
                left, right = _key(a), _key(b)
                return (left > right) - (left < right)


            def _key(text):
                numbers, _, pre = text.partition("-")
                found = numbers.split(".")
                if not pre:
                    return found, 0, "", ""
                letters, number = re.fullmatch(r"([a-zA-Z]*)(\\d*)", pre).groups()
                return found, 1, letters, number
            """,
            """[(a, b) for a in ("1.9", "1.12.0", "1.12", "2.0-rc2", "2.0-rc10", "2.0-beta", "2.0", "2.0-rc") for b in ("1.9", "1.12.0", "2.0-rc2", "2.0")]""",
            """[("1.9", "1.12.0"), ("2.0-rc2", "2.0")]""",
        ),
        "quantity": rule(
            """
            `quantity(text)` reads an amount such as `3`, `2x6`, `2 x 6 pcs` or `1.5kg`. Two numbers with an `x`
            between them are multiplied. A unit `kg` gives grams, `l` gives millilitres, `pcs` and no unit give
            pieces. The result is `(amount, unit)`, the unit being `g`, `ml` or `pcs`, and the amount a whole
            number. An amount of pieces that is not whole, a unit that is not known, or anything else raises
            ValueError. Upper and lower case are the same, and spaces do not matter.
            """,
            """
            import re
            from fractions import Fraction


            def quantity(text):
                found = re.fullmatch(r"(\\d+(?:\\.\\d+)?)(?:x(\\d+(?:\\.\\d+)?))?([a-z]*)", re.sub(r"\\s", "", text.lower()))
                if not found:
                    raise ValueError(text)
                amount = Fraction(found[1]) * Fraction(found[2] or 1)
                units = {"kg": ("g", 1000), "l": ("ml", 1000), "pcs": ("pcs", 1), "": ("pcs", 1)}
                if found[3] not in units:
                    raise ValueError(text)
                unit, times = units[found[3]]
                amount *= times
                if amount.denominator != 1:
                    raise ValueError(text)
                return int(amount), unit
            """,
            """
            import re
            from fractions import Fraction


            def quantity(text):
                found = re.fullmatch(r"(\\d+(?:\\.\\d+)?)(?:x(\\d+(?:\\.\\d+)?))?([a-z]*)", re.sub(r"\\s", "", text))
                if not found:
                    raise ValueError(text)
                amount = Fraction(found[1]) + Fraction(found[2] or 0)
                units = {"kg": ("g", 1000), "l": ("ml", 100), "pcs": ("pcs", 1), "": ("pcs", 1)}
                if found[3] not in units:
                    raise ValueError(text)
                unit, times = units[found[3]]
                return int(amount * times), unit
            """,
            """[(t,) for t in ("3", "2x6", "2 X 6 pcs", "1.5kg", "0.75 L", "1.5", "2x1.5", "3 boxes", "", "x", "2x2.5pcs")]""",
            """[("2x6",), ("1.5kg",)]""",
        ),
        "overlap": rule(
            """
            `overlap(slots)` gives the first pair of bookings that clash, as the pair of their places in the list,
            or None. A booking is `(room, start, end)` in minutes, and takes the room from its start up to but not
            including its end. A booking whose end is not after its start takes nothing. Of several clashing pairs
            the first is the one whose first place is smallest, and then whose second is.
            """,
            """
            def overlap(slots):
                for i, (room, start, end) in enumerate(slots):
                    for j in range(i + 1, len(slots)):
                        other, begin, stop = slots[j]
                        if room == other and max(start, begin) < min(end, stop):
                            return i, j
                return None
            """,
            """
            def overlap(slots):
                for i, (room, start, end) in enumerate(slots):
                    for j in range(i + 1, len(slots)):
                        other, begin, stop = slots[j]
                        if max(start, begin) <= min(end, stop):
                            return i, j
                return None
            """,
            """[(s,) for s in ([], [("a", 0, 60), ("a", 60, 90)], [("a", 0, 60), ("b", 30, 90), ("a", 59, 61)], [("a", 10, 10), ("a", 0, 60)], [("a", 0, 5), ("b", 0, 9), ("b", 8, 9), ("a", 4, 6)])]""",
            """[([("a", 0, 60), ("a", 60, 90)],), ([("a", 0, 60), ("b", 30, 90), ("a", 59, 61)],)]""",
        ),
        "split_bill": rule(
            """
            `split_bill(total_cents, shares)` divides an amount by whole shares, such as `[2, 1, 1]`, and gives a
            list of cents in the same order. Each gets the exact part rounded down. The cents that are left go
            one each to those whose rounding lost the most, and of equals to the one that comes first. A share of
            zero gets nothing. Where all shares are zero, or one is below zero, ValueError is raised.
            """,
            """
            def split_bill(total_cents, shares):
                whole = sum(shares)
                if whole <= 0 or min(shares) < 0:
                    raise ValueError(shares)
                parts = [total_cents * s // whole for s in shares]
                lost = [(-(total_cents * s % whole), i) for i, s in enumerate(shares)]
                for _, i in sorted(lost)[: total_cents - sum(parts)]:
                    parts[i] += 1
                return parts
            """,
            """
            def split_bill(total_cents, shares):
                whole = sum(shares)
                if whole <= 0:
                    raise ValueError(shares)
                parts = [total_cents * s // whole for s in shares]
                for i in range(total_cents - sum(parts)):
                    parts[i] += 1
                return parts
            """,
            """[(t, s) for t in (0, 1, 100, 101, 1000) for s in ([1], [1, 1, 1], [2, 1, 1], [0, 3, 4], [5, 0, 7, 11], [0, 0], [3, -1])]""",
            """[(100, [1, 1, 1]), (101, [0, 3, 4])]""",
        ),
        "due_date": rule(
            f"""
            `due_date(year, month, day, workdays, holidays)` gives the date, as `(year, month, day)`, that lies that
            many working days after the given one. Saturdays, Sundays and the `(month, day)` pairs in `holidays`
            are no working days, and the given day itself never counts. With zero workdays it is the given day
            itself if that is a working day, and else the next one. Fewer than zero raises ValueError.
            """,
            """
            from datetime import date, timedelta


            def due_date(year, month, day, workdays, holidays):
                if workdays < 0:
                    raise ValueError(workdays)
                at = date(year, month, day)
                if workdays == 0:
                    while not _works(at, holidays):
                        at += timedelta(1)
                for _ in range(workdays):
                    at += timedelta(1)
                    while not _works(at, holidays):
                        at += timedelta(1)
                return at.year, at.month, at.day


            def _works(at, holidays):
                return at.weekday() < 5 and (at.month, at.day) not in holidays
            """,
            """
            from datetime import date, timedelta


            def due_date(year, month, day, workdays, holidays):
                at = date(year, month, day)
                for _ in range(workdays):
                    at += timedelta(1)
                    if not _works(at, holidays):
                        at += timedelta(1)
                return at.year, at.month, at.day


            def _works(at, holidays):
                return at.weekday() < 5 and (at.month, at.day) not in holidays
            """,
            """[(2026, 12, d, n, [(12, 25), (1, 1)]) for d in (18, 19, 24, 25, 31) for n in (-1, 0, 1, 2, 7)]""",
            """[(2026, 12, 24, 1, [(12, 25), (1, 1)]), (2026, 12, 19, 0, [])]""",
        ),
        "thermostat": rule(
            f"""
            `thermostat(readings, target)` gives for every reading whether the heating is on after it. It starts
            off. It goes on when a reading is more than {near} degrees below the target, and off when one is at or
            above it. In between it stays as it was. A reading of None is a sensor that failed: the heating stays
            as it was, and after {near} failures in a row it goes off until the next reading.
            """,
            f"""
            def thermostat(readings, target):
                on, failed, states = False, 0, []
                for reading in readings:
                    if reading is None:
                        failed += 1
                        if failed >= {near}:
                            on = False
                    else:
                        failed = 0
                        if reading < target - {near}:
                            on = True
                        elif reading >= target:
                            on = False
                    states.append(on)
                return states
            """,
            f"""
            def thermostat(readings, target):
                on, failed, states = False, 0, []
                for reading in readings:
                    if reading is None:
                        failed += 1
                        if failed > {near}:
                            on = False
                    else:
                        if reading <= target - {near}:
                            on = True
                        elif reading > target:
                            on = False
                    states.append(on)
                return states
            """,
            f"[(r, 20) for r in ([], [{20 - near}, {19 - near}, 19, 20, 19], [10, None, None, None, 19, 10, None, 10], [10] + [None] * {near} + [19], [10, 20, 21, {20 - near}])]",
            f"[([{19 - near}, 19, 20], 20), ([10] + [None] * {near}, 20)]",
        ),
    }


FILLER = '''
def describe():
    """What this module is about, for the help page."""
    return {name!r}.replace("_", " ")


def explain(*arguments):
    """The result for some arguments as a line of text, for the log."""
    try:
        found = {name}(*arguments)
    except ValueError as error:
        return f"{name}{{arguments}} is not possible: {{error}}"
    return f"{name}{{arguments}} = {{found!r}}"
'''


def _expected(code: str, name: str, cases: str) -> list:
    """What the right module gives for each case."""
    scope: dict = {}
    exec(code, scope)  # the rules above, never anything else
    found = []
    for arguments in eval(cases, {}):
        try:
            found.append([list(arguments), repr(scope[name](*arguments))])
        except ValueError:
            found.append([list(arguments), "ValueError"])
    return found


TEST = '''"""Checks of {name}."""
import unittest

from rules.{name} import {name}

CASES = {cases}


class Test(unittest.TestCase):
{methods}


if __name__ == "__main__":
    unittest.main()
'''
METHOD = """    def test_{k}(self):
        arguments, expected = CASES[{k}]
        try:
            found = repr({name}(*arguments))
        except ValueError:
            found = "ValueError"
        self.assertEqual(found, expected, f"{name} of {{arguments}}")
"""


def _suite(name: str, checks: list) -> str:
    cases = "[\n" + "".join(f"    ({a!r}, {e!r}),\n" for a, e in checks) + "]"
    methods = "\n".join(METHOD.format(k=k, name=name) for k in range(len(checks)))
    return TEST.format(name=name, cases=cases, methods=methods.rstrip())


def make(folder: Path, seed: int, modules: int = 14) -> dict:
    """Write a task. Returns its hidden checks by module."""
    r = random.Random(seed)
    drawn = rules(r)
    names = sorted(r.sample(sorted(drawn), modules))
    (folder / "rules").mkdir(parents=True)
    (folder / "rules" / "__init__.py").write_text("")
    sections, hidden = ["# What the rules say\n"], {}
    for name in names:
        one = drawn[name]
        sections.append(f"## {name}\n\n{one['spec']}\n")
        about = f'"""{name.replace("_", " ").capitalize()}: see its section in SPEC.md."""\n\n'
        imports, _, rest = one["bad"].rpartition("\n\n\ndef " + name)
        body = f"{imports}\n\n\ndef {name}{rest}" if imports else one["bad"]
        (folder / "rules" / f"{name}.py").write_text(
            about + body + "\n\n" + FILLER.format(name=name)
        )
        (folder / f"test_{name}.py").write_text(
            _suite(name, _expected(one["good"], name, one["shown"]))
        )
        hidden[name] = _expected(one["good"], name, one["cases"])
    (folder / "SPEC.md").write_text("\n".join(sections))
    return hidden


def grade(folder: Path, hidden: dict) -> dict:
    """How many modules do what the specification says, and which."""
    import subprocess
    import sys

    right = {}
    for name, checks in hidden.items():
        code = (
            "import ast, sys\n"
            f"from rules.{name} import {name} as f\n"
            "ok = True\n"
            "for arguments, expected in ast.literal_eval(sys.stdin.read()):\n"
            "    try:\n        found = repr(f(*arguments))\n"
            "    except ValueError:\n        found = 'ValueError'\n"
            "    except Exception:\n        found = 'error'\n"
            "    ok = ok and found == expected\n"
            "print(ok)\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=folder,
            input=repr(checks),
            capture_output=True,
            text=True,
            timeout=60,
        )
        right[name] = done.stdout.strip() == "True"
    return {"right": sum(right.values()), "of": len(right), "modules": right}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(make(args.folder, args.seed), indent=1)[:200])
