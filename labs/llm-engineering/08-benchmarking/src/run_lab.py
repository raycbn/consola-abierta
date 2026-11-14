#!/usr/bin/env python3
"""Analyse client-observed LLM serving traces. Python standard library only."""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
REQ_FIELDS = {"request_id","scenario","scheduled_ms","sent_ms","first_token_ms","completed_ms","prompt_tokens","output_tokens","status"}
EVENT_FIELDS = {"request_id","chunk_index","received_ms","tokens_in_chunk"}

def num(v, field, row, empty=False):
    if v is None or str(v).strip() == "":
        if empty: return None
        raise ValueError(f"row {row}: {field} is required")
    try: x=float(v)
    except (ValueError, TypeError): raise ValueError(f"row {row}: {field} must be numeric") from None
    if not math.isfinite(x): raise ValueError(f"row {row}: {field} must be finite")
    return x

def read_csv(path, required):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        missing=required-set(reader.fieldnames or [])
        if missing: raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        return list(reader)

def pct(values, p):
    xs=sorted(float(x) for x in values if x is not None)
    if not xs: return None
    if len(xs)==1: return xs[0]
    h=(len(xs)-1)*p/100.0; lo=math.floor(h); hi=math.ceil(h)
    return xs[lo]+(xs[hi]-xs[lo])*(h-lo)

def distribution(values):
    xs=[float(x) for x in values if x is not None]
    if not xs: return {"count":0,"mean":None,"p50":None,"p90":None,"p95":None,"p99":None,"max":None}
    return {"count":len(xs),"mean":statistics.fmean(xs),"p50":pct(xs,50),"p90":pct(xs,90),"p95":pct(xs,95),"p99":pct(xs,99),"max":max(xs)}

def make_fixture(seed=20261114):
    """Deterministic teaching traces. No model/API/GPU/serving runtime is executed."""
    DATA.mkdir(parents=True,exist_ok=True); rng=random.Random(seed); reqs=[]; events=[]
    for scenario in ("steady","bursty"):
        finish=[]
        for i in range(48):
            scheduled=i*520 if scenario=="steady" else (i//8)*6200+(i%8)*18
            active=sum(1 for t in finish if t>scheduled)
            queue=max(0,active-3)*24.0
            sent=scheduled+max(0,active-6)*7.0
            prompt=rng.choice([96,160,256,512,1024,1536,2048])
            out=rng.choice([12,16,24,32,48])
            prefill=72.0+prompt*0.19
            first=sent+max(20.0,prefill+queue+rng.uniform(-8,8))
            chunks=[1]; remaining=out-1
            while remaining:
                if remaining>=3 and rng.random()<0.12: size=3
                elif remaining>=2 and rng.random()<0.18: size=2
                else: size=1
                size=min(size,remaining); chunks.append(size); remaining-=size
            gap=25+max(0,active-2)*3.8
            times=[first]
            for _ in chunks[1:]: times.append(times[-1]+max(5,gap+rng.uniform(-3,3)))
            failed=i in ((31,45) if scenario=="steady" else (14,39))
            rid=f"{scenario}-{i+1:03d}"
            if failed:
                end=sent+2500+queue
                reqs.append(dict(request_id=rid,scenario=scenario,scheduled_ms=round(scheduled,3),sent_ms=round(sent,3),first_token_ms="",completed_ms=round(end,3),prompt_tokens=prompt,output_tokens=0,status="timeout",server_queue_ms="",server_prefill_ms="",server_decode_ms=""))
            else:
                end=times[-1]
                reqs.append(dict(request_id=rid,scenario=scenario,scheduled_ms=round(scheduled,3),sent_ms=round(sent,3),first_token_ms=round(first,3),completed_ms=round(end,3),prompt_tokens=prompt,output_tokens=out,status="success",server_queue_ms=round(queue,3),server_prefill_ms=round(prefill,3),server_decode_ms=round(end-first,3)))
                for j,(t,n) in enumerate(zip(times,chunks),1):
                    events.append(dict(request_id=rid,chunk_index=j,received_ms=round(t,3),tokens_in_chunk=n))
            finish.append(end)
    with (DATA/"requests.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(reqs[0])); w.writeheader(); w.writerows(reqs)
    with (DATA/"token-events.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(events[0])); w.writeheader(); w.writerows(events)
    (DATA/"manifest.json").write_text(json.dumps({"dataset":"synthetic_llm_serving_trace","seed":seed,"requests":len(reqs),"token_events":len(events),"scenarios":["steady","bursty"],"provenance":"Generated locally by this lab. No model, API, GPU or serving runtime was used.","warning":"Do not quote these values as model or hardware performance."},indent=2)+"\n",encoding="utf-8")
    return DATA/"requests.csv",DATA/"token-events.csv"

def load_trace(request_path,event_path):
    raw=read_csv(request_path,REQ_FIELDS); raw_events=read_csv(event_path,EVENT_FIELDS) if event_path and event_path.exists() else []
    reqs=[]; seen=set()
    for i,row in enumerate(raw,2):
        rid=row["request_id"].strip()
        if not rid or rid in seen: raise ValueError(f"{request_path} row {i}: empty or duplicate request_id {rid!r}")
        seen.add(rid); status=row["status"].strip().lower()
        if status not in {"success","error","timeout","rejected","cancelled"}: raise ValueError(f"{request_path} row {i}: unsupported status {status!r}")
        scheduled=num(row["scheduled_ms"],"scheduled_ms",i); sent=num(row["sent_ms"],"sent_ms",i)
        first=num(row.get("first_token_ms"),"first_token_ms",i,True); completed=num(row["completed_ms"],"completed_ms",i)
        prompt=int(num(row["prompt_tokens"],"prompt_tokens",i)); output=int(num(row["output_tokens"],"output_tokens",i))
        if prompt<0 or output<0: raise ValueError(f"{request_path} row {i}: token counts cannot be negative")
        if not scheduled<=sent<=completed: raise ValueError(f"{request_path} row {i}: expected scheduled <= sent <= completed")
        if first is not None and not sent<=first<=completed: raise ValueError(f"{request_path} row {i}: first_token_ms outside send/completion interval")
        if status=="success" and (first is None or output<1): raise ValueError(f"{request_path} row {i}: successful request needs first token and output_tokens >= 1")
        if status!="success" and first is None and output!=0: raise ValueError(f"{request_path} row {i}: failed request without first token must have output_tokens=0")
        r={"request_id":rid,"scenario":row["scenario"].strip() or "unspecified","scheduled_ms":scheduled,"sent_ms":sent,"first_token_ms":first,"completed_ms":completed,"prompt_tokens":prompt,"output_tokens":output,"status":status,"client_queue_ms":sent-scheduled,"ttft_ms":first-sent if first is not None else None,"e2e_ms":completed-sent,"scheduled_e2e_ms":completed-scheduled,"tpot_ms":(completed-first)/(output-1) if first is not None and output>1 else None}
        for key in ("server_queue_ms","server_prefill_ms","server_decode_ms"):
            v=row.get(key,""); r[key]=num(v,key,i,True) if v not in (None,"") else None
        reqs.append(r)
    byid={r["request_id"]:r for r in reqs}; groups=defaultdict(list)
    for i,row in enumerate(raw_events,2):
        rid=row["request_id"].strip()
        if rid not in byid: raise ValueError(f"{event_path} row {i}: unknown request_id {rid!r}")
        idx=int(num(row["chunk_index"],"chunk_index",i)); t=num(row["received_ms"],"received_ms",i); n=int(num(row["tokens_in_chunk"],"tokens_in_chunk",i))
        if idx<1 or n<1: raise ValueError(f"{event_path} row {i}: chunk index and token count must be positive")
        groups[rid].append({"chunk_index":idx,"received_ms":t,"tokens_in_chunk":n})
    itl=[]
    for r in reqs:
        g=sorted(groups.get(r["request_id"],[]),key=lambda x:x["chunk_index"])
        if r["status"]!="success":
            if g: raise ValueError(f"{r['request_id']}: failed request unexpectedly has token events")
            continue
        if not g: continue
        if [x["chunk_index"] for x in g]!=list(range(1,len(g)+1)): raise ValueError(f"{r['request_id']}: chunk indexes must be contiguous from 1")
        if abs(g[0]["received_ms"]-r["first_token_ms"])>0.001: raise ValueError(f"{r['request_id']}: first event differs from first_token_ms")
        if sum(x["tokens_in_chunk"] for x in g)!=r["output_tokens"]: raise ValueError(f"{r['request_id']}: chunk token counts do not reconcile")
        if g[-1]["received_ms"]>r["completed_ms"]+0.001: raise ValueError(f"{r['request_id']}: event occurs after completion")
        for a,b in zip(g,g[1:]):
            gap=b["received_ms"]-a["received_ms"]
            if gap<=0: raise ValueError(f"{r['request_id']}: event timestamps must increase")
            itl.append(gap)
    return reqs,groups,itl

def max_concurrency(reqs):
    points=[]
    for r in reqs: points.extend([(r["sent_ms"],1),(r["completed_ms"],-1)])
    active=peak=0
    for _,delta in sorted(points,key=lambda x:(x[0],x[1])): active+=delta; peak=max(peak,active)
    return peak

def summarize(reqs,itl,slo,cost):
    if not reqs: raise ValueError("No requests found")
    ok=[r for r in reqs if r["status"]=="success"]; bad=[r for r in reqs if r["status"]!="success"]
    start=min(r["scheduled_ms"] for r in reqs); end=max(r["completed_ms"] for r in reqs); duration=max((end-start)/1000,.001)
    inp=sum(r["prompt_tokens"] for r in ok); out=sum(r["output_tokens"] for r in ok); total=inp+out
    good=[r for r in ok if r["ttft_ms"]<=slo["ttft_ms"] and (r["tpot_ms"] is None or r["tpot_ms"]<=slo["tpot_ms"]) and r["e2e_ms"]<=slo["e2e_ms"]]
    result={"request_count":len(reqs),"successful_requests":len(ok),"failed_requests":len(bad),"error_rate":len(bad)/len(reqs),"measurement_window_s":duration,"request_throughput_rps":len(ok)/duration,"goodput_rps":len(good)/duration,"slo_good_requests":len(good),"slo_good_fraction":len(good)/len(ok) if ok else None,"max_in_flight_concurrency":max_concurrency(reqs),"input_tokens":inp,"output_tokens":out,"total_tokens":total,"input_token_throughput_tps":inp/duration,"output_token_throughput_tps":out/duration,"total_token_throughput_tps":total/duration,"ttft_ms":distribution([r["ttft_ms"] for r in ok]),"tpot_ms":distribution([r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]),"itl_ms":distribution(itl),"e2e_ms":distribution([r["e2e_ms"] for r in ok]),"client_queue_ms":distribution([r["client_queue_ms"] for r in reqs]),"server_queue_ms":distribution([r["server_queue_ms"] for r in reqs if r["server_queue_ms"] is not None]),"slo_thresholds_ms":slo,"cost_model":None}
    if cost is not None:
        rate,gpus=cost; amount=rate*gpus*duration/3600
        result["cost_model"]={"user_supplied_gpu_hour_rate":rate,"gpu_count":gpus,"measured_runtime_s":duration,"estimated_compute_cost":amount,"cost_per_1m_output_tokens":amount*1_000_000/out if out else None,"cost_per_1m_total_tokens":amount*1_000_000/total if total else None,"warning":"Excludes idle capacity, amortisation, power, networking, storage, platform and operational costs."}
    return result

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in rows: w.writerow({k:row.get(k) for k in fields})

def markdown_report(overall,scenarios):
    lines=["# LLM benchmarking trace report",""," > Synthetic teaching trace. No model, API, GPU or serving engine was executed.","","## Overall",f"- Requests: {overall['request_count']} ({overall['successful_requests']} successful, {overall['failed_requests']} failed)",f"- Measurement window: {overall['measurement_window_s']:.3f} s",f"- Request throughput: {overall['request_throughput_rps']:.3f} req/s",f"- Output token throughput: {overall['output_token_throughput_tps']:.3f} tok/s",f"- Total token throughput: {overall['total_token_throughput_tps']:.3f} tok/s",f"- Maximum in-flight concurrency: {overall['max_in_flight_concurrency']}",f"- SLO goodput: {overall['goodput_rps']:.3f} req/s ({overall['slo_good_requests']} requests)","","## Latency distributions (ms)","","| Metric | n | Mean | p50 | p90 | p95 | p99 | Max |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ("ttft_ms","tpot_ms","itl_ms","e2e_ms","client_queue_ms","server_queue_ms"):
        d=overall[key]; vals=[d["count"]]+[d[k] for k in ("mean","p50","p90","p95","p99","max")]
        cells=["—" if x is None else (str(x) if j==0 else f"{x:.2f}") for j,x in enumerate(vals)]
        lines.append("| "+key.replace("_ms","").upper()+" | "+" | ".join(cells)+" |")
    lines += ["","## Scenario comparison","","| Scenario | Requests | RPS | Output tok/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Max concurrency |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,s in sorted(scenarios.items()):
        lines.append(f"| {name} | {s['request_count']} | {s['request_throughput_rps']:.3f} | {s['output_token_throughput_tps']:.3f} | {s['ttft_ms']['p95']:.2f} | {s['tpot_ms']['p95']:.2f} | {s['e2e_ms']['p95']:.2f} | {s['max_in_flight_concurrency']} |")
    lines += ["","## Interpretation limits","", "- Percentiles use linear interpolation (R-7) over the supplied samples.","- p95/p99 are descriptive; 48 requests per scenario are not enough to establish a production tail-latency SLO.","- ITL is measured between streamed content events, not between internal model tokens when a chunk bundles several tokens.","- TPOT is calculated once per successful request as (E2E − TTFT) / (output tokens − 1).","- Client queue time is scheduled_ms to sent_ms. Server queue time is independent and must not be added blindly to client latency.","- Throughput uses earliest scheduled arrival to latest completion, includes drain time and excludes warmup (not present in this fixture).","- Synthetic latencies are validation fixtures, not model, GPU, network or engine benchmarks.",""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description="Analyse request-level and streamed-token LLM traces.")
    ap.add_argument("--requests",type=Path); ap.add_argument("--events",type=Path)
    ap.add_argument("--output",type=Path,default=REPORTS)
    ap.add_argument("--slo-ttft-ms",type=float,default=500); ap.add_argument("--slo-tpot-ms",type=float,default=80); ap.add_argument("--slo-e2e-ms",type=float,default=5000)
    ap.add_argument("--gpu-hour-cost",type=float); ap.add_argument("--gpu-count",type=int,default=1)
    args=ap.parse_args()
    if min(args.slo_ttft_ms,args.slo_tpot_ms,args.slo_e2e_ms)<=0: ap.error("SLO thresholds must be positive")
    if args.gpu_hour_cost is not None and args.gpu_hour_cost<0: ap.error("--gpu-hour-cost cannot be negative")
    if args.gpu_count<1: ap.error("--gpu-count must be >= 1")
    if args.requests: rp,ep=args.requests,args.events
    else: rp,default=make_fixture(); ep=args.events or default
    reqs,groups,itl=load_trace(rp,ep); slo={"ttft_ms":args.slo_ttft_ms,"tpot_ms":args.slo_tpot_ms,"e2e_ms":args.slo_e2e_ms}
    cost=(args.gpu_hour_cost,args.gpu_count) if args.gpu_hour_cost is not None else None
    overall=summarize(reqs,itl,slo,cost); names=sorted({r["scenario"] for r in reqs}); scenarios={}
    for name in names:
        sub=[r for r in reqs if r["scenario"]==name]; ids={r["request_id"] for r in sub}
        sub_itl=[b["received_ms"]-a["received_ms"] for rid in ids for g in [sorted(groups.get(rid,[]),key=lambda x:x["chunk_index"])] for a,b in zip(g,g[1:])]
        scenarios[name]=summarize(sub,sub_itl,slo,cost)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in reqs:
        rows.append({**r,"ttft_ms":"" if r["ttft_ms"] is None else round(r["ttft_ms"],6),"tpot_ms":"" if r["tpot_ms"] is None else round(r["tpot_ms"],6),"e2e_ms":round(r["e2e_ms"],6),"client_queue_ms":round(r["client_queue_ms"],6)})
    write_csv(args.output/"request-metrics.csv",rows,["request_id","scenario","status","scheduled_ms","sent_ms","first_token_ms","completed_ms","prompt_tokens","output_tokens","client_queue_ms","ttft_ms","tpot_ms","e2e_ms","server_queue_ms","server_prefill_ms","server_decode_ms"])
    itlrows=[]
    for rid,g in groups.items():
        g=sorted(g,key=lambda x:x["chunk_index"])
        for a,b in zip(g,g[1:]): itlrows.append({"request_id":rid,"from_chunk":a["chunk_index"],"to_chunk":b["chunk_index"],"itl_ms":round(b["received_ms"]-a["received_ms"],6)})
    write_csv(args.output/"itl-samples.csv",itlrows,["request_id","from_chunk","to_chunk","itl_ms"])
    payload={"method":"R-7 linear-interpolation percentiles","source_requests":str(rp),"source_events":str(ep) if ep else None,"overall":overall,"scenarios":scenarios,"provenance":"Synthetic fixture or user-supplied trace; inspect paths and manifest."}
    (args.output/"metrics.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    (args.output/"lab-report.md").write_text(markdown_report(overall,scenarios),encoding="utf-8")
    checks=[]
    def check(name,ok): checks.append((name,bool(ok)))
    ok=[r for r in reqs if r["status"]=="success"]; bad=[r for r in reqs if r["status"]!="success"]
    check("request IDs are unique",len({r["request_id"] for r in reqs})==len(reqs))
    check("schedule/send/completion timestamps are ordered",all(r["scheduled_ms"]<=r["sent_ms"]<=r["completed_ms"] for r in reqs))
    check("successful requests have a first token",all(r["first_token_ms"] is not None for r in ok))
    check("failed requests are retained",len(ok)+len(bad)==len(reqs))
    check("successful output-token counts are positive",all(r["output_tokens"]>0 for r in ok))
    check("token event chunks reconcile",all(sum(x["tokens_in_chunk"] for x in groups.get(r["request_id"],[]))==r["output_tokens"] for r in ok if groups.get(r["request_id"])))
    check("stream event timestamps strictly increase",all(b["received_ms"]>a["received_ms"] for g0 in groups.values() for a,b in zip(sorted(g0,key=lambda x:x["chunk_index"]),sorted(g0,key=lambda x:x["chunk_index"])[1:])))
    check("TTFT is non-negative",all(r["ttft_ms"] is None or r["ttft_ms"]>=0 for r in reqs))
    check("TPOT defined only for outputs longer than one token",all((r["tpot_ms"] is None)==(r["output_tokens"]<=1 or r["first_token_ms"] is None) for r in reqs))
    check("latency percentiles are monotonic",all(d["p50"]<=d["p90"]<=d["p95"]<=d["p99"]<=d["max"] for d in (overall["ttft_ms"],overall["tpot_ms"],overall["itl_ms"],overall["e2e_ms"]) if d["count"]))
    check("input-token totals reconcile",overall["input_tokens"]==sum(r["prompt_tokens"] for r in ok))
    check("output-token totals reconcile",overall["output_tokens"]==sum(r["output_tokens"] for r in ok))
    check("request-throughput formula",abs(overall["request_throughput_rps"]-overall["successful_requests"]/overall["measurement_window_s"])<1e-9)
    check("output-throughput formula",abs(overall["output_token_throughput_tps"]-overall["output_tokens"]/overall["measurement_window_s"])<1e-9)
    check("total-throughput formula",abs(overall["total_token_throughput_tps"]-overall["total_tokens"]/overall["measurement_window_s"])<1e-9)
    check("goodput does not exceed request throughput",overall["goodput_rps"]<=overall["request_throughput_rps"]+1e-12)
    check("maximum concurrency is positive",overall["max_in_flight_concurrency"]>=1)
    check("scenario summaries are complete",len(scenarios)==len(names))
    check("ITL samples equal streamed event gaps",overall["itl_ms"]["count"]==len(itl))
    manifest=DATA/"manifest.json"
    check("synthetic data has explicit provenance",not manifest.exists() or "No model" in manifest.read_text(encoding="utf-8"))
    print("LLM BENCHMARK TRACE ANALYSER")
    print(f"Requests: {rp}"); print(f"Token events: {ep if ep else 'not supplied'}")
    print(f"Requests: {overall['request_count']} | success: {overall['successful_requests']} | failed: {overall['failed_requests']}")
    print(f"Window: {overall['measurement_window_s']:.3f}s | output: {overall['output_token_throughput_tps']:.3f} tok/s | max concurrency: {overall['max_in_flight_concurrency']}")
    print("Checks:")
    for name,passed in checks: print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    n=sum(1 for _,passed in checks if passed); print(f"RESULT: {n}/{len(checks)} PASS"); print(f"Reports: {args.output}")
    if n!=len(checks): sys.exit(1)

if __name__=="__main__": main()
