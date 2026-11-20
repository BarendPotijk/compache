"""Build grouped, text-free benchmark evidence from a local public-chat research workbench."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path


POLICIES = {"generic":"compache","opencode":"opencode","headroom":"headroom"}
MIN_CALLS = 3


def read(path):
    return json.loads(path.read_bytes())


def add(groups, key, inp, out, reads, writes, usd):
    groups[key].update(calls=1, input=inp, output=out, read=reads, write=writes, usd=Decimal(usd))


def pool(groups, mode):
    """Retain tariff separation while coalescing sparse role breakdowns."""
    grouped=defaultdict(dict)
    for (*coords,role), values in groups.items():
        grouped[tuple(coords)][role]=values
    rows=[]
    for coords, roles in sorted(grouped.items()):
        sparse=sorted(role for role, v in roles.items() if v["calls"]<MIN_CALLS)
        if sparse:
            anchor=max(roles, key=lambda role:(roles[role]["calls"], role))
            combined=Counter()
            for role in sorted(set(sparse+[anchor])):
                combined.update(roles.pop(role))
            roles["combined"]=combined
        for role, data in sorted(roles.items()):
            if data["calls"]<MIN_CALLS:
                raise ValueError("Tariff stratum too small to export")
            suite,case,policy,w,band=coords
            out=dict(suite=suite,case=case,policy=policy,band=band,role_group=role,**data)
            if mode=="modeled":
                out["window_minutes"]=w
            out["usd"]=format(data["usd"],"f")
            rows.append(out)
    return rows


def tool_source(action, visible):
    if action["name"]!="retrieve":
        return "n/a"
    args=json.loads(action["arguments"])
    doc=args.get("document_id")
    if not doc:
        return "id_missing"
    if doc not in visible:
        return "id_unseen"
    return "introduced" if args.get("version") in visible[doc] else "version_unseen"


def export(root,dest):
    prices=defaultdict(Counter)
    physical=defaultdict(Counter)
    outcomes=[]
    reference=[]
    total_rejections=0
    for suite in ("goodai", "memoryarena"):
        comparison=read(root/suite/"comparison.json")
        corrections=read(root/"cache-correction.json")["suites"][suite]["changed_scenarios"]
        amended={(r["key"],r["clock_minutes"]):r for r in corrections}
        originals=[json.loads(line) for line in (root/("live-"+suite)/"receipts.jsonl").read_bytes().splitlines()]
        observed=set()
        virtual={r["key"]:r for r in originals if r["event"]=="virtual"}
        total_rejections+=sum(r["event"]=="rejected" and r.get("arm") in POLICIES for r in originals)
        for r in originals:
            if r["event"]!="metered" or r["arm"] not in POLICIES:
                continue
            if r["response_id"] in observed:
                raise ValueError("Repeated physical response")
            observed.add(r["response_id"])
            case,policy=r["case_id"],POLICIES[r["arm"]]
            usage=r["usage"]
            inp,out=usage["input_tokens"],usage["output_tokens"]
            role=r["policy"].get("retry_of_role",r["role"])
            band="long" if inp>272_000 else "ordinary"
            add(physical,(suite,case,policy,0,band,role),inp,out,usage["input_tokens_details"]["cached_tokens"],
                usage["input_tokens_details"]["cache_write_tokens"],r["accounting"]["list_price_usd"])
            windows=[r["minutes"]] if r["arm"]=="generic" else [5,30]
            for w in windows:
                scenario=virtual[r["key"]]["scenarios"][str(w)]
                corrected=amended.get((r["key"],w))
                clocks=scenario["modeled_usage"]
                reads,writes,usd=clocks["cached_input_tokens"],clocks["cache_write_input_tokens"],scenario["accounting"]["list_price_usd"]
                if corrected:
                    assert (reads,writes)==(corrected["old_reads"],corrected["old_writes"])
                    reads,writes,usd=corrected["corrected_reads"],corrected["corrected_writes"],corrected["corrected_usd"]
                add(prices,(suite,case,policy,w,band,role),inp,out,reads,writes,usd)

        scored=read(root/suite/"scored-answers.json")
        grades={(g["arm"],g["ttl"],t["case"],t["turn"]):t for g in scored for t in g["answers"]}
        for r in sorted((r for r in originals if r["event"]=="lane_complete" and r["arm"] in POLICIES),key=lambda r:r["lane"]):
            case,policy=r["case_id"],POLICIES[r["arm"]]
            windows=[r["minutes"]] if r["arm"]=="generic" else [5,30]
            diag=r["diagnostics"]
            visible=defaultdict(set)
            turns=[]
            tool_counts=Counter()
            rounds_phase=Counter()
            had_checkpoint=False
            q=read(root/suite/"heldout"/(case+".json"))
            for i,(fixture,answer) in enumerate(zip(q["turns"],diag["answers"],strict=True),1):
                had_checkpoint |= bool(answer["host_checkpoint"])
                phase="from_first_host_summary" if had_checkpoint else "before_first_host_summary"
                for doc in fixture["contexts"]:
                    visible[doc["id"]].add(doc["version"])
                for batch in diag["tool_rounds"]:
                    if batch["turn"]==i:
                        rounds_phase[phase]+=1
                        for action in batch["tools"]:
                            tool_counts[(phase,action["name"],action["status"],tool_source(action,visible))]+=1
                assessed=[x for x in fixture["requirements"] if x["historical"]=="yes"]
                dropped=set(answer["archived_ids"])
                retained=sum(any(dropped.isdisjoint(g) for g in req["alternatives"]) for req in assessed)
                g=grades[r["arm"],windows[0],case,i]
                assert len(assessed)-retained==len(g["missed_labels"])
                turns.append(dict(turn=i,action=answer["planner_action"],host_summary=bool(answer["host_checkpoint"]),
                                  assessed=len(assessed),kept=retained,unknown=sum(x["historical"]=="unknown" for x in fixture["requirements"]),
                                  grade=g["kind"],passed=g["pass_mark"]))
            outcomes.append(dict(suite=suite,case=case,policy=policy,windows_minutes=windows,turns=turns,
                                 tool_rounds_by_phase=dict(sorted(rounds_phase.items())),
                                 tool_counts=[dict(phase=p,tool=t,status=s,source=k,count=n) for (p,t,s,k),n in sorted(tool_counts.items())]))
        for row in comparison["metrics"]:
            if row["arm"] in POLICIES:
                reference.append(dict(suite=suite,policy=POLICIES[row["arm"]],window_minutes=row["ttl"],
                                      usd=str(row["modeled_usd"]),turns=row["turns"],input=row["input_tokens"],
                                      read=row["cache_read_tokens"],write=row["cache_write_tokens"],
                                      tool_rounds=row["tool_rounds"],host_summaries=row["summaries"],
                                      cuts=row["recipe_cuts"],assessed=row["retention"]["assessed"],kept=row["retention"]["kept"],
                                      unknown=row["retention"].get("unknown",0),
                                      grades={k:v for k,v in row["quality"].items() if k in {"native","intermediate","supplemental","terminal"}}))
    dest.mkdir(parents=True,exist_ok=True)
    bundle=dict(schema_version=1, physical=pool(physical,"physical"),modeled=pool(prices,"modeled"),
                trajectories=outcomes,reference=reference,definitive_rejections=total_rejections)
    (dest/"aggregate.json").write_text(json.dumps(bundle,indent=2)+"\n")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("workbench",type=Path,help="Original outputs/tuned-chat workbench")
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    export(args.workbench,args.destination)
