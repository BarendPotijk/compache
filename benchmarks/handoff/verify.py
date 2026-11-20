"""Reconcile grouped public-chat benchmark evidence using Python's standard library."""
from collections import Counter, defaultdict
from decimal import Decimal as D
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def price(row):
    fresh=row["input"]-row["read"]-row["write"]
    require(row["band"] in {"ordinary","long"}, "Unknown tariff band")
    require(row["calls"]>=3, "Unexpected sparse billing group")
    require(all(type(c) is int and c>=0 for c in (fresh,row["read"],row["write"],row["output"])), "Unbalanced tokens")
    long=row["band"]=="long"
    cost=(D(".10")*fresh+D(".01")*row["read"]+D(".125")*row["write"])*(2 if long else 1)
    cost+=D(".50")*row["output"]*(D("1.5") if long else 1)
    return cost/1_000_000


def verify(path):
    d=json.loads(path.read_text())
    require(d["schema_version"]==1,"Unsupported schema")
    cells=defaultdict(Counter)
    grades=defaultdict(Counter)
    groups=defaultdict(set)
    traffic=defaultdict(Counter)
    case_tally=defaultdict(Counter)
    physical_count=0
    physical_cost=D(0)
    for kind in ("physical","modeled"):
        seen=set()
        for r in d[kind]:
            coords=r["suite"],r["policy"],r.get("window_minutes"),r["case"],r["band"],r["role_group"]
            require(coords not in seen,"Duplicate billing group")
            seen.add(coords)
            require(kind!="physical" or "window_minutes" not in r,"Physical expense duplicated across replay windows")
            money=price(r)
            require(D(r["usd"])==money, f"Tariff disagreement: {coords}")
            if kind=="physical":
                physical_count+=r["calls"]
                physical_cost+=money
            else:
                g=coords[:3]
                cells[g].update(usd=money, input=r["input"],read=r["read"],write=r["write"])
                traffic[g][r["case"]]+=r["calls"]
    nturns=0
    for a in d["trajectories"]:
        ids=[r["turn"] for r in a["turns"]]
        require(ids==list(range(1,len(ids)+1)), "Unexpected turn order")
        nturns+=len(ids)
        for w in a["windows_minutes"]:
            group=a["suite"],a["policy"],w
            bucket=cells[group]
            case=a["case"]
            require(case not in groups[group], "Duplicate source conversation in window/policy")
            groups[group].add(case)
            require(traffic[group][case]>=len(ids), "Insufficient group coverage")
            bucket["tool_rounds"]+=sum(a["tool_rounds_by_phase"].values())
            for z in a["tool_counts"]:
                require(z["count"]>=1 and z["phase"] in a["tool_rounds_by_phase"],"Invalid tool counts")
                case_tally[group,case][z["tool"],z["status"],z["source"]]+=z["count"]
            for t in a["turns"]:
                require(0<=t["kept"]<=t["assessed"],"Invalid retained count")
                require((t["grade"] is None and t["passed"] is None) or
                        (isinstance(t["grade"],str) and isinstance(t["passed"],bool)), "Invalid outcome")
                bucket.update(turns=1,cuts=int(t["action"] not in {"host","keep"}),host_summaries=int(t["host_summary"]),
                              assessed=t["assessed"],kept=t["kept"],unknown=t["unknown"])
                if t["grade"]:
                    grades[group][t["grade"],"total"]+=1
                    grades[group][t["grade"],"passed"]+=t["passed"]
    reference={(r["suite"],r["policy"],r["window_minutes"]):r for r in d["reference"]}
    require(len(reference)==len(d["reference"])==12 and set(cells)==set(reference),"Missing reference cell")
    for g,c in sorted(cells.items()):
        r=reference[g]
        require(len(groups[g])==2,"Expected two source conversations")
        for col in ("usd","turns","input","read","write","cuts","host_summaries","assessed","kept","unknown","tool_rounds"):
            require(c[col]==D(str(r[col])), f"Reference disagreement: {g}, {col}")
        got={k:[grades[g][k,"passed"],grades[g][k,"total"]] for k,_ in grades[g]}
        require(got==r["grades"],f"Grades disagree: {g}")
        print(f"{g[0]:12} {g[1]:8} {g[2]:2}m   {c['turns']:2} turns   USD/run {c['usd']/c['turns']:.5f}   "
              f"kept {c['kept']}/{c['assessed']}   cuts {c['cuts']}   tool rounds {c['tool_rounds']}")
    require(physical_count==459 and physical_cost==D("21.423110690"),"Physical total differs")
    require(nturns==204 and sum(v["turns"] for v in cells.values())==306,"User turn total differs")
    print(f"Physical expenditure across {physical_count} responses: USD {physical_cost:f}.")
    for case in ("arena-080","arena-090"):
        for arm in ("compache","headroom"):
            a,=[x for x in d["trajectories"] if x["case"]==case and x["policy"]==arm and 30 in x["windows_minutes"]]
            tally=case_tally[("memoryarena",arm,30),case]
            print(f"{case}, 30m, {arm}: {sum(a['tool_rounds_by_phase'].values())} tool rounds; "
                  f"{a['tool_rounds_by_phase'].get('before_first_host_summary',0)} before first summary; "
                  f"{tally['retrieve','invalid_arguments','id_unseen']} retrieve calls returned invalid_arguments for unseen IDs.")


if __name__=="__main__":
    verify(Path(__file__).with_name("aggregate.json"))
