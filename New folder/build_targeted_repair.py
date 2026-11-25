import json
from pathlib import Path

BASE=Path(r"HM-RAG + MAMMQA 16-question smoke experiment\hm-rag-mammqa-runtime-optimized-v1.ipynb")
TARGET=Path(r"HM-RAG + MAMMQA 16-question smoke experiment\hm-rag-mammqa-targeted-scientific-repair-v2.ipynb")
nb=json.loads(BASE.read_text(encoding="utf-8")); assert len(nb["cells"])==28

def set_cell(i,text):
    c=nb["cells"][i]; c["source"]=text.strip("\n").splitlines(keepends=True)
    if c["cell_type"]=="code":c["outputs"]=[]; c["execution_count"]=None

set_cell(0,r'''
# HM-RAG + MAMMQA: targeted scientific repair v2

**Default phase:** `ENGINEERING_GATE` — exactly two DEBUG16 questions, no gold evaluation.  
**Purpose:** repair candidate-to-context retention, verifier semantics, targeted feedback execution,
and evidence-aware aggregation without changing full-corpus retrieval or the MAMMQA topology.

The original analyzed questions are named `DEBUG16_V1`. A deterministic, disjoint `VALIDATION16_V2`
is frozen before inference for the next unbiased small validation, but is **not** launched automatically.
''')

cell3="".join(nb["cells"][3]["source"])
cell3=cell3.replace('SEED=42; RUN_MODE="smoke"; RUN_RETRIEVAL_ONLY=False; FAST_VALIDATION=True\nNUM_SHARDS=1; SHARD_ID=0\nEXPERIMENT_ID="qa_full_runtime_optimized_v1_seed42"',
'''SEED=42; RUN_MODE="smoke"; RUN_RETRIEVAL_ONLY=False
EXPERIMENT_PHASE="ENGINEERING_GATE"  # ENGINEERING_GATE | DEBUG16_REGRESSION | VALIDATION16_V2
FAST_VALIDATION=EXPERIMENT_PHASE=="ENGINEERING_GATE"
NUM_SHARDS=1; SHARD_ID=0; VALIDATION_SELECTION_SEED=20260924
EXPERIMENT_ID={"ENGINEERING_GATE":"qa_targeted_repair_v2_gate_seed42","DEBUG16_REGRESSION":"qa_targeted_repair_v2_debug16_seed42","VALIDATION16_V2":"qa_targeted_repair_v2_validation16_seed42"}[EXPERIMENT_PHASE]''')
cell3=cell3.replace('TEXT_CONTEXT_K=3; TABLE_CONTEXT_K=2; IMAGE_CONTEXT_K=2\nFALLBACK_TEXT_K=1; FALLBACK_TABLE_K=1; FALLBACK_IMAGE_K=1',
'''TEXT_CONTEXT_K=3; TABLE_CONTEXT_K=2; IMAGE_CONTEXT_K=2
TEXT_CONTEXT_K_MAX=4; TABLE_CONTEXT_K_MAX=3; IMAGE_CONTEXT_K_MAX=2
FALLBACK_TEXT_K=1; FALLBACK_TABLE_K=1; FALLBACK_IMAGE_K=1
FEEDBACK_CANDIDATE_K=20; CONTEXT_DUPLICATE_THRESHOLD=0.72
CONTEXT_SELECTION_WEIGHTS={"relevance":1.0,"original_question":0.35,"uncovered_query":0.25,"modality_need":0.15,"graph":0.10,"feedback":0.30,"redundancy":0.35}''')
old='''input_root=Path("/kaggle/input"); cache_roots=[]
if input_root.exists():
    for p in input_root.iterdir():
        for candidate in [p,p/"persistent_cache_package"]:
            if (candidate/"indexes").exists() or (candidate/"cache").exists():cache_roots.append(candidate)
cache_roots += [BASE_OUTPUT_ROOT]
CACHE_READ_ROOTS=list(dict.fromkeys(p.resolve() for p in cache_roots if p.exists()))'''
new='''input_root=Path("/kaggle/input"); cache_roots=[]
if input_root.exists():
    # Accept direct, nested, renamed, or previously published Kaggle cache datasets.
    for marker in list(input_root.rglob("text_faiss.index"))+list(input_root.rglob("image_manifest.json")):
        parent=marker.parent; root=parent.parent if parent.name in {"indexes","cache"} else parent
        cache_roots.append(root)
    for p in input_root.iterdir():
        cache_roots.extend([p,p/"persistent_cache_package",p/"hm_mammqa"])
cache_roots += [BASE_OUTPUT_ROOT]
CACHE_READ_ROOTS=list(dict.fromkeys(p.resolve() for p in cache_roots if p.exists() and ((p/"indexes").exists() or (p/"cache").exists())))
print("Persistent cache roots discovered:",[str(p) for p in CACHE_READ_ROOTS])'''
assert old in cell3; cell3=cell3.replace(old,new)
cell3=cell3.replace('assert NUM_SHARDS>=1 and 0<=SHARD_ID<NUM_SHARDS and REPAIR_STAGES=={"aggregator"}',
'''assert NUM_SHARDS>=1 and 0<=SHARD_ID<NUM_SHARDS and REPAIR_STAGES=={"aggregator"}
assert EXPERIMENT_PHASE in {"ENGINEERING_GATE","DEBUG16_REGRESSION","VALIDATION16_V2"}''')
set_cell(3,cell3)

cell15="".join(nb["cells"][15]["source"])
cell15=cell15.replace('insights=tag_value(raw,"insights"); answer=tag_value(raw,"answer"); subanswer=tag_value(raw,"subanswer"); ids=tag_value(raw,"evidence_ids")',
'''insights=tag_value(raw,"insight") or tag_value(raw,"insights"); answer=tag_value(raw,"answer"); subanswer=tag_value(raw,"subanswer"); ids=tag_value(raw,"evidence_ids")''')
start=cell15.index('def parse_compact_verifier(raw):')
end=cell15.index('\ndef parse_compact_aggregator(raw):',start)
cell15=cell15[:start]+r'''def parse_compact_verifier(raw):
    def line(label):
        match=re.search(fr"^\s*{label}\s*:\s*([^\n]*)",str(raw),re.I|re.M); return match.group(1).strip() if match else ""
    status=line("STATUS").lower(); best=line("BEST_ANSWER"); ids=line("SUPPORT_IDS")
    mod=line("RETRY_MODALITY").lower(); query=line("RETRY_QUERY")
    if status not in {"supported","insufficient","conflict"}:return None
    if status=="supported" and (not best or best.upper()=="NONE"):return None
    if mod not in {"text","table","image","none"}:mod="none"
    return {"status":status,"best_answer":"" if best.upper()=="NONE" else best,
            "supporting_doc_ids":[x.strip() for x in re.split(r"[,;]",ids) if x.strip() and x.strip().upper()!="NONE"],
            "retry_modality":mod,"retry_query":"" if query.upper()=="NONE" else query}
'''+cell15[end:]
cell15=cell15.replace('"protected_prompt_truncated":False,"seed":seed,"oom_retried":oom_retried}',
'''"protected_prompt_truncated":False,"seed":seed,"oom_retried":oom_retried,"hit_max_new_tokens":int(output_ids.shape[-1])>=max_new_tokens}''')
cell15=cell15.replace('key="stage1" if stage.startswith("stage1_") else ("stage2" if stage.startswith("stage2_") else stage.replace("_format_repair",""))',
'''key="stage1" if stage.startswith("stage1_") else ("stage2" if stage.startswith("stage2_") else ("verifier" if stage.startswith("verifier") else stage.replace("_format_repair","")))''')
set_cell(15,cell15)

cell19="".join(nb["cells"][19]["source"])
prompt_start=cell19.index('SYNTH_PROMPT=')
prompt_end=cell19.index('\n\ndef run_cross_modal_agents',prompt_start)
cell19=cell19[:prompt_start]+'''SYNTH_PROMPT="""Synthesize from the named anchor expert and raw evidence from other modalities only.
Return exactly:
<answer>short candidate answer OR INSUFFICIENT</answer>
<insight>one concise evidence-grounded sentence</insight>
<evidence_ids>comma-separated supplied IDs or NONE</evidence_ids>
No long explanation, chain-of-thought, or outside knowledge."""'''+cell19[prompt_end:]
set_cell(19,cell19)

cell17="".join(nb["cells"][17]["source"])
old_selector='''def select_context(question,candidates,context_budgets):
    context={m:[] for m in ["text","table","image"]}
    for mod,k in context_budgets.items():
        pool=candidates.get(mod,[])
        context[mod]=CLIP.rerank(question,[d for d in pool if d.get("image_path")],k) if mod=="image" and pool else pool[:k]
    return context
'''
new_selector=r'''def candidate_terms(doc):
    text=" ".join(str(doc.get(k,"")) for k in ["title","retrieval_text","reasoning_content","raw_content"])
    return set(re.findall(r"[a-z0-9]+",text.lower()))

def redundancy(a,b):
    x,y=candidate_terms(a),candidate_terms(b)
    return len(x&y)/max(1,len(x|y))

def query_hits(doc):
    return [(str(p.get("query_text","")),str(p.get("query_source",""))) for p in doc.get("query_provenance",[]) if p.get("query_text")]

def select_modality_context(modality,pool,base_k,max_k,plan,feedback_doc_ids=None):
    feedback_doc_ids=set(map(str,feedback_doc_ids or [])); pool=[dict(d) for d in pool]
    for rank,d in enumerate(pool,1):d.setdefault("_candidate_rank",rank)
    query_records=plan.get("retrieval_queries",[]); original=next((q["text"] for q in query_records if q.get("source")=="original_question"),"")
    subqueries=[q["text"] for q in query_records if q.get("source")=="decomposed_subquery"]
    needed=modality in set(plan.get("needed_modalities",[])); confidence=float(plan.get("modality_confidence",{}).get(modality,0))
    top=pool[:max(1,base_k)]; duplicate_top=any(redundancy(a,b)>=CONTEXT_DUPLICATE_THRESHOLD for i,a in enumerate(top) for b in top[i+1:])
    query_space={q for d in pool for q,_ in query_hits(d)}; uncovered_possible=sum(q in query_space for q in subqueries)>max(0,base_k-1)
    target=base_k
    if needed and target<max_k and (uncovered_possible or duplicate_top or (subqueries and confidence>=0.55)):target+=1
    target=min(target,max_k,len(pool)); selected=[]; covered=set(); diagnostics=[]

    def add(doc,reason,score):
        hits=query_hits(doc); source=hits[0][0] if hits else ("GRAPH_EXPANSION" if doc.get("graph_score") else "")
        item=dict(doc); item.update(selected_rank=len(selected)+1,candidate_rank=doc["_candidate_rank"],source_query=source,selection_reason=reason,selection_score=float(score))
        selected.append(item); covered.update(q for q,_ in hits)
        diagnostics.append({"doc_id":str(doc["parent_doc_id"]),"modality":modality,"selected_rank":len(selected),"candidate_rank":doc["_candidate_rank"],
                            "retrieval_score":float(doc.get("final_score",0)),"source_query":source,"selection_reason":reason,"selection_score":float(score)})

    original_candidates=[d for d in pool if any(source=="original_question" or query==original for query,source in query_hits(d))]
    if target and original_candidates:add(original_candidates[0],"original_question_top",2.0)
    while len(selected)<target:
        best=None
        for d in pool:
            if any(str(x["parent_doc_id"])==str(d["parent_doc_id"]) for x in selected):continue
            hits=query_hits(d); new_queries={q for q,_ in hits if q not in covered}; rel=1/max(1,d["_candidate_rank"])
            red=max([redundancy(d,x) for x in selected] or [0.0]); graph=1.0 if d.get("graph_score",0)>0 else 0.0
            is_feedback=str(d["parent_doc_id"]) in feedback_doc_ids
            score=(CONTEXT_SELECTION_WEIGHTS["relevance"]*rel+
                   CONTEXT_SELECTION_WEIGHTS["uncovered_query"]*bool(new_queries)+
                   CONTEXT_SELECTION_WEIGHTS["modality_need"]*needed+
                   CONTEXT_SELECTION_WEIGHTS["graph"]*graph+
                   CONTEXT_SELECTION_WEIGHTS["feedback"]*is_feedback-
                   CONTEXT_SELECTION_WEIGHTS["redundancy"]*red)
            reason=("feedback_added" if is_feedback else "subquery_coverage" if new_queries else "graph_expansion" if graph else
                    "modality_coverage" if needed and not selected else "diversity_fill" if red<CONTEXT_DUPLICATE_THRESHOLD else "high_fused_score")
            candidate=(score,-d["_candidate_rank"],d,reason)
            if best is None or candidate[:2]>best[:2]:best=candidate
        if best is None:break
        add(best[2],best[3],best[0])
    return selected,diagnostics,{"base_k":base_k,"selected_k":len(selected),"max_k":max_k,"adaptive_expansion":len(selected)>base_k,
                                 "covered_queries":sorted(covered),"duplicate_top_detected":duplicate_top}

def select_context(question,candidates,context_budgets,plan,feedback_doc_ids=None):
    context={m:[] for m in ["text","table","image"]}; diagnostics=[]; stats={}
    max_k={"text":TEXT_CONTEXT_K_MAX,"table":TABLE_CONTEXT_K_MAX,"image":IMAGE_CONTEXT_K_MAX}
    for mod,k in context_budgets.items():
        pool=list(candidates.get(mod,[]))
        if mod=="image" and pool:
            ranks={str(d["parent_doc_id"]):i+1 for i,d in enumerate(pool)}
            pool=CLIP.rerank(question,[d for d in pool if d.get("image_path")],len(pool))
            for d in pool:d["_candidate_rank"]=ranks.get(str(d["parent_doc_id"]),len(pool)+1)
        chosen,rows,mod_stats=select_modality_context(mod,pool,k,max_k[mod],plan,feedback_doc_ids)
        context[mod]=chosen; diagnostics.extend(rows); stats[mod]=mod_stats
    return context,diagnostics,stats
'''
assert old_selector in cell17; cell17=cell17.replace(old_selector,new_selector)
cell17=cell17.replace('context=select_context(q.question,candidates,context_budgets)',
'''context,selection_diagnostics,selection_stats=select_context(q.question,candidates,context_budgets,plan)''')
cell17=cell17.replace('"context_counts":{m:len(xs) for m,xs in context.items()}}',
'''"context_counts":{m:len(xs) for m,xs in context.items()},"context_selection":selection_diagnostics,"context_selection_stats":selection_stats}''')
cell17=cell17.replace('return {"retrieval_candidates":candidates,"context_evidence":context,"graph_provenance":graph_provenance,',
'''return {"retrieval_candidates":candidates,"context_evidence":context,"graph_provenance":graph_provenance,"context_selection_diagnostics":selection_diagnostics,"context_selection_stats":selection_stats,''')
set_cell(17,cell17)

set_cell(20,r'''
# Cell 21 - evidence-aware verifier: concrete candidates, neutral INSUFFICIENT votes, deterministic decoding
VERIFY_PROMPT="""You are an evidence verifier. Decide whether available retrieved evidence supports a concrete candidate.
Do NOT require every modality or agent to answer. An agent output of INSUFFICIENT is neutral, not a contradiction.
SUPPORTED: at least one concrete candidate has direct support and no stronger contradictory evidence.
INSUFFICIENT: no concrete candidate has enough support yet.
CONFLICT: two or more different concrete answers have credible conflicting support.
Return compact lines only.
For supported:
STATUS: supported
BEST_ANSWER: short answer
SUPPORT_IDS: comma-separated IDs or NONE
For insufficient/conflict also add:
STATUS: insufficient | conflict
RETRY_MODALITY: text | table | image | none
RETRY_QUERY: short query or NONE"""

def normalize_consensus_answer(value):
    value=str(value or "").strip().strip('"\'').lower(); value="".join(" " if c in string.punctuation else c for c in value)
    return " ".join(value.split())

def synthesis_consensus(synth):
    answers=[]
    for modality,envelope in synth.items():
        parsed=envelope.get("parsed_output") if envelope.get("parse_success") else None
        answer=(parsed or {}).get("candidate_answer","")
        if not answer:answer=tag_value(envelope.get("raw_output",""),"answer")
        normalized=normalize_consensus_answer(answer)
        concrete=bool(normalized and normalized not in {"insufficient","abstain","none","unknown"})
        answers.append({"modality":modality,"answer":answer,"normalized_answer":normalized,"concrete":concrete,
                        "evidence_ids":list(map(str,(parsed or {}).get("supporting_doc_ids",[])))})
    groups=defaultdict(list)
    for row in answers:
        if row["concrete"]:groups[row["normalized_answer"]].append(row)
    best=max(groups.items(),key=lambda kv:(len(kv[1]),kv[0]),default=("",[]))
    return {"stage2_answers":answers,"concrete_answer_count":sum(x["concrete"] for x in answers),
            "normalized_answer_groups":{k:len(v) for k,v in groups.items()},
            "consensus_answer":best[1][0]["answer"] if best[1] else "","consensus_count":len(best[1]),
            "candidate_evidence_ids":sorted({i for x in answers for i in x["evidence_ids"]})}

def verifier_evidence(synth,evidence):
    cited=set(synthesis_consensus(synth)["candidate_evidence_ids"]); selected=[]
    for mod in ["text","table","image"]:
        docs=evidence.get(mod,[]); preferred=[d for d in docs if str(d["parent_doc_id"]) in cited]
        selected.extend(clean_evidence((preferred or docs)[:1],mod,VERIFIER_EVIDENCE_CHARS))
    return selected

def verify_answer(q,synth,evidence,consensus=None,stage_name="verifier"):
    assert_inference_boundary("verify_answer",q); consensus=consensus or synthesis_consensus(synth)
    protected={"qid":q.qid,"question":q.question,"candidate_syntheses":{m:compact_or_raw(x,700) for m,x in synth.items()},"consensus_diagnostics":consensus}
    envelope=stage_call(VERIFY_PROMPT,protected,verifier_evidence(synth,evidence),q.qid,stage_name,VERIFIER_DO_SAMPLE,parse_compact_verifier,False)
    parsed=envelope.get("parsed_output"); envelope["effective_status"]=parsed.get("status") if parsed else "verifier_error"
    envelope["verifier_error"]=not bool(parsed); envelope["verifier_parse_failure"]=not bool(parsed)
    return envelope
''')

set_cell(21,r'''
# Cell 22 - executable one-round feedback, second cheap verifier and evidence-aware aggregation
AGG_PROMPT="""Resolve the synthesis results using the verifier decision.
ABSTAIN only when evidence genuinely remains insufficient or supported answers remain in conflict.
INSUFFICIENT outputs from other agents are neutral and do not invalidate a verifier-supported answer.
If verifier status is supported, normally preserve its BEST_ANSWER unless the supplied syntheses show a genuine contradiction.
Return exactly: <answer>short final answer OR ABSTAIN</answer>"""

def run_reasoning(q,evidence):
    t=time.perf_counter(); experts=run_modality_experts(q,evidence); stage1=time.perf_counter()-t
    t=time.perf_counter(); synth=run_cross_modal_agents(q,experts,evidence); stage2=time.perf_counter()-t
    return experts,synth,{"stage1_seconds":stage1,"stage2_seconds":stage2}

def aggregate_answer(q,synth,verification,consensus,sees_question=True):
    protected={"qid":q.qid,"syntheses":{m:compact_or_raw(x,700) for m,x in synth.items()},
               "verifier":compact_or_raw(verification,700),"consensus_diagnostics":consensus}
    if sees_question:protected["question"]=q.question
    return stage_call(AGG_PROMPT,protected,[],q.qid,"aggregator",MAMMQA_DO_SAMPLE,parse_compact_aggregator,True)

def merge_evidence(old,new):
    out={}
    for mod in ["text","table","image"]:
        vals={str(d["parent_doc_id"]):dict(d) for d in old.get(mod,[])}
        for d in new.get(mod,[]):
            did=str(d["parent_doc_id"])
            if did not in vals:vals[did]=dict(d); continue
            current=vals[did]; combined=current.get("query_provenance",[])+d.get("query_provenance",[]); unique={}
            for p in combined:unique[(str(p.get("query_text","")),str(p.get("query_source","")),int(p.get("rank",0)),str(p.get("method","")))]=p
            current["query_provenance"]=list(unique.values())
            for field in ["final_score","dense_score","bm25_score","hybrid_rrf_score","graph_score"]:
                values=[x for x in [current.get(field),d.get(field)] if x is not None]
                if values:current[field]=max(values)
        out[mod]=sorted(vals.values(),key=lambda d:(-float(d.get("final_score",0)),str(d["parent_doc_id"])))
    return out

def compact_pool(pool):
    keep=["parent_doc_id","doc_id","modality","final_score","dense_score","bm25_score","hybrid_rrf_score","graph_score","query_provenance",
          "candidate_rank","selected_rank","source_query","selection_reason","selection_score"]
    return {m:[{k:d.get(k) for k in keep if d.get(k) is not None} for d in docs] for m,docs in pool.items()}

def compact_round(round_no,query_records,bundle):
    return {"round":round_no,"query_records":query_records,"candidate_ids":bundle["candidate_ids"],"context_ids":bundle["context_ids"],
            "routing":bundle["routing"],"retrieval_latency_seconds":bundle["retrieval_latency_seconds"],
            "context_selection_diagnostics":bundle.get("context_selection_diagnostics",[])}

def deterministic_feedback_request(q,plan,synth,verifier_parsed):
    status=(verifier_parsed or {}).get("status")
    if status not in {"insufficient","conflict"}:return None
    available=["text","table"]+(["image"] if IMAGE_PATHS else [])
    supplied_mod=(verifier_parsed or {}).get("retry_modality"); supplied_query=str((verifier_parsed or {}).get("retry_query","")).strip()
    if supplied_mod in available and supplied_query:return {"modality":supplied_mod,"query":supplied_query,"source":"verifier"}
    consensus=synthesis_consensus(synth); insufficient=[x["modality"] for x in consensus["stage2_answers"] if not x["concrete"]]
    conflict_modalities=[x["modality"] for x in consensus["stage2_answers"] if x["concrete"]] if status=="conflict" else []
    choices=conflict_modalities or insufficient or available; weights=plan.get("modality_confidence",{})
    modality=supplied_mod if supplied_mod in available else max([m for m in choices if m in available] or available,key=lambda m:(float(weights.get(m,0)),m))
    queries=[x for x in plan.get("normalized_subqueries",[]) if str(x).strip()]; query=supplied_query or (queries[0] if queries else q.question)
    return {"modality":modality,"query":query,"source":"deterministic_fallback"}

def apply_aggregator_policy(aggregator,verification,consensus,feedback_used,candidate_count,context_count):
    parsed=aggregator.get("parsed_output") if aggregator.get("parse_success") else None; answer=(parsed or {}).get("answer","")
    verifier=(verification or {}).get("parsed_output") if (verification or {}).get("parse_success") else None; status=(verifier or {}).get("status","verifier_error")
    if status=="supported" and str((verifier or {}).get("best_answer","")).strip():
        answer=verifier["best_answer"]
        return canonical_answer(answer),"answered",None,True
    if status=="conflict":return "ABSTAIN","abstain","unresolved_conflict",True
    if status=="insufficient":return "ABSTAIN","abstain","verifier_insufficient_after_feedback" if feedback_used else "context_missing_support",True
    if answer and str(answer).strip().upper()!="ABSTAIN":return canonical_answer(answer),"answered",None,True
    if candidate_count==0:return "ABSTAIN","abstain","no_candidate_evidence",True
    if consensus.get("concrete_answer_count",0)==0:return "ABSTAIN","abstain","no_concrete_synthesis",True
    if not aggregator.get("parse_success"):return "","generation_error","generation_failure",False
    return "ABSTAIN","abstain","aggregator_uncertain",True

def run_pipeline(q,cfg):
    assert_inference_boundary("run_pipeline",q); started=time.perf_counter(); calls0=QWEN_CALL_COUNTER[q.qid]; repairs0=FORMAT_REPAIR_CALL_COUNTER[q.qid]
    t=time.perf_counter(); plan=plan_query(q); planner_seconds=time.perf_counter()-t
    bundle=retrieve_evidence(q,plan,retrieval_mode=cfg["retrieval_mode"],router_mode=cfg["router_mode"],use_graph=cfg["graph"])
    evidence=bundle["context_evidence"]; experts,synth,reason_timing=run_reasoning(q,evidence); consensus=synthesis_consensus(synth)
    t=time.perf_counter(); initial_verification=verify_answer(q,synth,evidence,consensus,"verifier"); verifier_seconds=time.perf_counter()-t
    final_verification=initial_verification; second_verification=None; rounds=1; added=[]; targeted=None; feedback_seconds=0.0
    retrieval_round_details=[compact_round(1,plan["retrieval_queries"],bundle)]
    reasoning_round_details=[{"round":1,"experts":experts,"cross_modal":synth,"verification":initial_verification}]
    verifier_parsed=initial_verification.get("parsed_output") if initial_verification.get("parse_success") else None
    request=deterministic_feedback_request(q,plan,synth,verifier_parsed)
    if cfg["feedback"] and request:
        feedback_start=time.perf_counter(); mod=request["modality"]; rq=request["query"]; assert_clean_retrieval_query(rq)
        old_context_ids={m:[str(d["parent_doc_id"]) for d in xs] for m,xs in evidence.items()}
        retry_plan={"retrieval_queries":build_query_records(q.question,[rq]),"normalized_subqueries":[rq],"needed_modalities":[mod],
                    "modality_confidence":{m:float(m==mod) for m in ["text","table","image"]},"routing_confidence":1.0}
        retry=retrieve_evidence(q,retry_plan,retrieval_mode=cfg["retrieval_mode"],router_mode="hard",use_graph=cfg["graph"])
        feedback_ids={str(d["parent_doc_id"]) for d in retry["retrieval_candidates"].get(mod,[])}
        merged_candidates=merge_evidence(bundle["retrieval_candidates"],retry["retrieval_candidates"])
        context_budgets=bundle["routing"]["context_budgets"]
        evidence,selection_diag,selection_stats=select_context(q.question,merged_candidates,context_budgets,plan,feedback_ids)
        bundle["retrieval_candidates"]=merged_candidates; bundle["context_evidence"]=evidence
        bundle["candidate_ids"]={m:[d["parent_doc_id"] for d in xs] for m,xs in merged_candidates.items()}; bundle["context_ids"]={m:[d["parent_doc_id"] for d in xs] for m,xs in evidence.items()}
        bundle["context_selection_diagnostics"]=selection_diag; bundle["context_selection_stats"]=selection_stats
        new_context_ids={m:[str(d["parent_doc_id"]) for d in xs] for m,xs in evidence.items()}; old_flat={x for xs in old_context_ids.values() for x in xs}; new_flat={x for xs in new_context_ids.values() for x in xs}; added=sorted(new_flat-old_flat); rounds=2
        targeted={"feedback_triggered":True,"feedback_modality":mod,"feedback_query":rq,"fallback_source":request["source"],
                  "old_context_ids":old_context_ids,"new_candidate_ids":retry["candidate_ids"],"new_context_ids":new_context_ids,
                  "new_unique_evidence_count":len(added),"feedback_new_evidence":bool(added)}
        retrieval_round_details.append(compact_round(2,retry_plan["retrieval_queries"],retry))
        experts,synth,retry_timing=run_reasoning(q,evidence); consensus=synthesis_consensus(synth)
        reason_timing["stage1_seconds"]+=retry_timing["stage1_seconds"]; reason_timing["stage2_seconds"]+=retry_timing["stage2_seconds"]
        t=time.perf_counter(); second_verification=verify_answer(q,synth,evidence,consensus,"verifier_second"); verifier_seconds+=time.perf_counter()-t; final_verification=second_verification
        reasoning_round_details.append({"round":2,"experts":experts,"cross_modal":synth,"verification":second_verification})
        feedback_seconds=time.perf_counter()-feedback_start
    t=time.perf_counter(); agg=aggregate_answer(q,synth,final_verification,consensus,cfg["agg_question"]); aggregator_seconds=time.perf_counter()-t
    candidate_count=sum(len(x) for x in bundle["retrieval_candidates"].values()); context_count=sum(len(x) for x in evidence.values())
    answer,status,abstention_reason,answer_ok=apply_aggregator_policy(agg,final_verification,consensus,bool(targeted),candidate_count,context_count)
    timing={"planner_seconds":planner_seconds,"retrieval_seconds":sum(x["retrieval_latency_seconds"] for x in retrieval_round_details),
            **reason_timing,"verifier_seconds":verifier_seconds,"feedback_seconds":feedback_seconds,"aggregator_seconds":aggregator_seconds,"total_seconds":time.perf_counter()-started}
    verifier_diag={"initial_verifier_status":initial_verification.get("effective_status"),"initial_best_answer":((initial_verification.get("parsed_output") or {}).get("best_answer","")),
                   "retry_modality":request.get("modality") if request else None,"retry_query":request.get("query") if request else None,"feedback_used":bool(targeted),
                   "new_evidence_found":bool(added),"second_verifier_status":second_verification.get("effective_status") if second_verification else None,
                   "second_best_answer":((second_verification.get("parsed_output") or {}).get("best_answer","")) if second_verification else None}
    final={"answer":answer,"status":status,"abstention_reason":abstention_reason,"answer_parse_success":answer_ok,"infrastructure_failure":not answer_ok}
    return {"qid":q.qid,"question":q.question,"inference_input_fields":["qid","question"],"planner":plan,
            "retrieval_candidates":compact_pool(bundle["retrieval_candidates"]),"context_evidence":compact_pool(evidence),
            "candidate_ids":bundle["candidate_ids"],"context_ids":bundle["context_ids"],"routing":bundle["routing"],
            "context_selection_diagnostics":bundle.get("context_selection_diagnostics",[]),"context_selection_stats":bundle.get("context_selection_stats",{}),
            "retrieval_latency_seconds":bundle["retrieval_latency_seconds"],"graph_provenance_count":len(bundle.get("graph_provenance",[])),
            "initial_verification":initial_verification,"verification":final_verification,"aggregator_raw":agg,"raw_final_output":agg.get("raw_output"),
            "parsed_final_answer":agg.get("parsed_output"),"final_decision":final,"retrieval_round_details":retrieval_round_details,
            "reasoning_round_details":reasoning_round_details,"synthesis_consensus":consensus,"verifier_diagnostics":verifier_diag,
            "targeted_feedback":targeted,"retrieval_rounds":rounds,"new_docs_added":added,"final_answer":answer,"abstain":answer=="ABSTAIN",
            "abstention_reason":abstention_reason,"forced_parser_abstain":False,"timing":timing,
            "qwen_call_count":QWEN_CALL_COUNTER[q.qid]-calls0,"format_repair_call_count":FORMAT_REPAIR_CALL_COUNTER[q.qid]-repairs0,"latency_seconds":timing["total_seconds"]}
''')

set_cell(24,r'''
# Cell 25 - named disjoint sets; default engineering gate uses only two DEBUG16 questions
DEBUG16_V1=[
 "f7dd1ac143507bd2ab9a480f18b86706","254afe636a34c3a5d94efa22d95465ad",
 "ebb69554283456251305b4b8ab579f76","4b9a4868455f98a69a8494ef1dfe7205",
 "186146ea1f5a9f7ada7f7fd31c5c75a4","c9fcae0ac6e3aa3f08375db9e4fe9037",
 "5dfd4afc175cdebf43295316a540d70c","6d372c501004c3e448b1991f39160e99",
 "b5d7685e595423638d26dd60e99fcc39","3126664c6ce707c6aca2c475ace8db08",
 "183a1eec0e38eaee513a5ec7996f55f1","2a5fdab4235e28037389777f473c4f9f",
 "2f154508615dc2d34b4d4b7fdced0b0e","22e22f3b558eeae2178ddd9cb6bc2b04",
 "1e3f1a08d865b3676a5ca81d8adc270b","ceb2b6a63d5f3b5eb8075979a55e7471"]
row_by_qid={query_only(r).qid:r for r in dev_raw}; assert all(qid in row_by_qid for qid in DEBUG16_V1)

def validation_stratum(row):
    meta=row.get("metadata",{}) or {}; qtype=str(meta.get("type","unknown")); modalities=meta.get("modalities",[]) or []
    return qtype+"|"+"+".join(sorted(map(str,modalities)))
def freeze_validation16_v2(rows):
    eligible=[r for r in rows if query_only(r).qid not in set(DEBUG16_V1)]; strata=defaultdict(list)
    for r in eligible:strata[validation_stratum(r)].append(r)
    rng=random.Random(VALIDATION_SELECTION_SEED)
    for values in strata.values():rng.shuffle(values)
    selected=[]
    while len(selected)<16:
        progressed=False
        for key in sorted(strata):
            if strata[key] and len(selected)<16:selected.append(strata[key].pop()); progressed=True
        if not progressed:break
    assert len(selected)==16; return [query_only(r).qid for r in selected]

VALIDATION16_V2=freeze_validation16_v2(dev_raw)
assert len(DEBUG16_V1)==len(set(DEBUG16_V1))==16 and len(VALIDATION16_V2)==len(set(VALIDATION16_V2))==16
assert not set(DEBUG16_V1)&set(VALIDATION16_V2)
(OUTPUT_ROOT/"frozen_question_sets.json").write_text(json.dumps({"DEBUG16_V1":DEBUG16_V1,"VALIDATION16_V2":VALIDATION16_V2,"validation_selection_seed":VALIDATION_SELECTION_SEED},indent=2),encoding="utf-8")

if EXPERIMENT_PHASE=="ENGINEERING_GATE":active_qids=DEBUG16_V1[:2]
elif EXPERIMENT_PHASE=="DEBUG16_REGRESSION":active_qids=DEBUG16_V1
else:active_qids=VALIDATION16_V2
SELECTED_BASE=[row_by_qid[qid] for qid in active_qids]; SELECTED=SELECTED_BASE[SHARD_ID::NUM_SHARDS]; SELECTED_QIDS=[query_only(r).qid for r in SELECTED]
assert SELECTED_QIDS==active_qids[SHARD_ID::NUM_SHARDS]
if EXPERIMENT_PHASE=="ENGINEERING_GATE":assert SELECTED_QIDS==DEBUG16_V1[:2] and NUM_SHARDS==1
(OUTPUT_ROOT/"selected_questions.json").write_text(json.dumps([asdict(query_only(r)) for r in SELECTED],indent=2,ensure_ascii=False),encoding="utf-8")
print("EXPERIMENT_PHASE =",EXPERIMENT_PHASE,"| selected =",len(SELECTED_QIDS),"| qids =",SELECTED_QIDS)
print("VALIDATION16_V2 frozen and disjoint; it will not run during the engineering gate.")
''')

set_cell(25,r'''
# Cell 26 - gold-free synthetic unit tests and targeted change table
def synthetic_verifier_rule(agent_answers,supported_answers):
    concrete=[normalize_consensus_answer(x) for x in agent_answers if normalize_consensus_answer(x) not in {"","insufficient","none","abstain"}]
    supported={normalize_consensus_answer(x) for x in supported_answers}
    supported_concrete=sorted(set(concrete)&supported)
    if len(supported_concrete)>=2:return "conflict"
    if len(supported_concrete)==1:return "supported"
    return "insufficient"

VERIFIER_UNIT_TESTS={
 "unanimous_supported":synthetic_verifier_rule(["Andrew Lloyd Webber"]*3,["Andrew Lloyd Webber"])=="supported",
 "single_supported_neutral_insufficient":synthetic_verifier_rule(["Eddy Merckx","INSUFFICIENT","INSUFFICIENT"],["Eddy Merckx"])=="supported",
 "supported_conflict":synthetic_verifier_rule(["X","Y","INSUFFICIENT"],["X","Y"])=="conflict",
 "all_insufficient":synthetic_verifier_rule(["INSUFFICIENT"]*3,[])=="insufficient"}

synthetic_q=QueryInput("synthetic_feedback","Identify the requested visual attribute")
synthetic_plan={"normalized_subqueries":["identify logo color"],"retrieval_queries":build_query_records(synthetic_q.question,["identify logo color"]),
                "needed_modalities":["image"],"modality_confidence":{"text":.1,"table":.1,"image":.8}}
synthetic_synth={"image":{"parse_success":True,"parsed_output":{"candidate_answer":"INSUFFICIENT","supporting_doc_ids":[]},"raw_output":"<answer>INSUFFICIENT</answer>"}}
explicit=deterministic_feedback_request(synthetic_q,synthetic_plan,synthetic_synth,{"status":"insufficient","retry_modality":"image","retry_query":"identify logo color"})
fallback=deterministic_feedback_request(synthetic_q,synthetic_plan,synthetic_synth,{"status":"insufficient","retry_modality":"none","retry_query":""})
mock_calls=Counter()
def mock_image_retrieval():
    mock_calls["image"]+=1
    return [{"parent_doc_id":"new_image","doc_id":"new_image","modality":"image","final_score":1.0,"retrieval_text":"logo color blue",
             "query_provenance":[{"query_text":"identify logo color","query_source":"decomposed_subquery","rank":1,"score":1.0,"method":"hybrid"}]}]
old={"image":[{"parent_doc_id":"old_image","doc_id":"old_image","modality":"image","final_score":.5,"retrieval_text":"unrelated logo",
               "query_provenance":[{"query_text":synthetic_q.question,"query_source":"original_question","rank":1,"score":.5,"method":"hybrid"}]}]}
new_docs=mock_image_retrieval(); merged=merge_evidence(old,{"image":new_docs}); chosen,_,_=select_modality_context("image",merged["image"],1,2,synthetic_plan,{"new_image"})
FEEDBACK_UNIT_TESTS={"explicit_request":explicit=={"modality":"image","query":"identify logo color","source":"verifier"},
 "missing_fields_fallback":fallback and fallback["modality"]=="image" and bool(fallback["query"]),"retrieval_called_once":mock_calls["image"]==1,
 "new_candidates_merged":{d["parent_doc_id"] for d in merged["image"]}=={"old_image","new_image"},"context_reranked":any(d["parent_doc_id"]=="new_image" for d in chosen)}

def mock_envelope(parsed):return {"parse_success":parsed is not None,"parsed_output":parsed,"raw_output":""}
consensus_one={"concrete_answer_count":1}
a1=apply_aggregator_policy(mock_envelope({"answer":"ABSTAIN","status":"abstain"}),mock_envelope({"status":"supported","best_answer":"X"}),consensus_one,False,3,2)
a2=apply_aggregator_policy(mock_envelope({"answer":"ABSTAIN","status":"abstain"}),mock_envelope({"status":"insufficient","best_answer":""}),{"concrete_answer_count":0},True,3,2)
a3=apply_aggregator_policy(mock_envelope({"answer":"X","status":"answered"}),mock_envelope({"status":"conflict","best_answer":""}),consensus_one,True,3,2)
AGGREGATOR_UNIT_TESTS={"preserve_verifier_supported":a1[0]=="X" and a1[1]=="answered","insufficient_after_feedback_abstains":a2[0]=="ABSTAIN","conflict_after_feedback_abstains":a3[0]=="ABSTAIN"}
UNIT_TEST_RESULTS={"verifier":VERIFIER_UNIT_TESTS,"feedback":FEEDBACK_UNIT_TESTS,"aggregator":AGGREGATOR_UNIT_TESTS}
assert all(all(group.values()) for group in UNIT_TEST_RESULTS.values()),UNIT_TEST_RESULTS

assert RETRIEVAL_CANDIDATE_K==20 and MAX_INPUT_TOKENS==4096 and REPAIR_STAGES=={"aggregator"}
assert parse_compact_verifier("STATUS: supported\nBEST_ANSWER: Example\nSUPPORT_IDS: D1")
CHANGE_ROWS=[
 ["Context selector","top-k slice","query/modality-aware deterministic MMR"],["Subquery coverage","none","reserved coverage bonus"],
 ["Redundancy handling","none","token-Jaccard penalty"],["Adaptive K","fixed 3/2/2","bounded 4/3/2"],
 ["Verifier semantics","conservative agreement","one supported candidate can pass; INSUFFICIENT is neutral"],
 ["Verifier fallback","missing fields skipped feedback","deterministic modality/query fallback"],["Feedback trigger","often skipped","always once for valid insufficient/conflict"],
 ["Feedback retrieval","existing narrow context","full target-modality K=20"],["Second verification","skipped","one cheap final verifier"],
 ["Aggregator abstention","verifier insufficiency dominated","evidence-aware explicit reason"],["Consensus diagnostic","none","normalized groups/count/evidence IDs"],
 ["Gold context retention","coverage only","post-freeze lost IDs + retention"],["Cache reuse","direct paths only","recursive Kaggle cache discovery + validation"]]
CHANGE_DF=pd.DataFrame(CHANGE_ROWS,columns=["Component","Old behavior","New behavior"])
print("\nTARGETED SCIENTIFIC REPAIR\n",CHANGE_DF.to_string(index=False)); print("\nUNIT TEST RESULTS\n",json.dumps(UNIT_TEST_RESULTS,indent=2))
(DIRS["logs"]/"targeted_change_summary.json").write_text(json.dumps({"changes":[dict(zip(["component","old","new"],x)) for x in CHANGE_ROWS],"unit_tests":UNIT_TEST_RESULTS},indent=2),encoding="utf-8")
print("PRE-RUN TESTS PASSED. Engineering gate remains limited to",len(SELECTED),"questions.")
''')

set_cell(22,r'''
# Cell 23 - FULL configuration only; baselines and the oracle are intentionally not executed
CONFIGS={
 "FULL":{"planner":True,"router":True,"router_mode":"soft","retrieval_mode":"hybrid",
         "graph":True,"verifier":True,"feedback":True,"agg_question":True}
}
assert list(CONFIGS)==["FULL"]
print("Execution methods:",list(CONFIGS),"(B1, B2, and MAMMQA_ORACLE are not scheduled)")
''')

set_cell(26,r'''
# Cell 27 - phase-aware FULL-only checkpointed inference; no gold enters inference
def append_jsonl(path,row):append_jsonl_log(path,row)
def read_jsonl_checked(path):
    path=Path(path)
    if not path.exists():return []
    rows=[]
    with path.open(encoding="utf-8") as f:
        for no,line in enumerate(f,1):
            if line.strip():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError as e:raise ValueError(f"{path}:{no}: {e}") from e
    return rows

def clean_for_json(x):
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {k:clean_for_json(v) for k,v in x.items() if k!="source_record"}
    if isinstance(x,(list,tuple)):return [clean_for_json(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x

def completed(path):
    rows=read_jsonl_checked(path); qids=[r["qid"] for r in rows]
    assert len(qids)==len(set(qids)); assert qids==SELECTED_QIDS[:len(qids)],f"Stale or out-of-order checkpoint: {qids}"
    return set(qids)

CONFIGS={"FULL":{"planner":True,"router":True,"router_mode":"soft","retrieval_mode":"hybrid","graph":True,"verifier":True,"feedback":True,"agg_question":True}}
METHODS=["FULL"]
prediction_name="FULL.jsonl" if NUM_SHARDS==1 else f"FULL_shard_{SHARD_ID}.jsonl"
IMPLEMENTATION_FINGERPRINT=hashlib.sha256(json.dumps({
 "prompts":[PLANNER_PROMPT,EXPERT_PROMPT,SYNTH_PROMPT,VERIFY_PROMPT,AGG_PROMPT],"stage_tokens":STAGE_MAX_NEW_TOKENS,
 "context_base":[TEXT_CONTEXT_K,TABLE_CONTEXT_K,IMAGE_CONTEXT_K],"context_max":[TEXT_CONTEXT_K_MAX,TABLE_CONTEXT_K_MAX,IMAGE_CONTEXT_K_MAX],
 "candidate_k":RETRIEVAL_CANDIDATE_K,"selection_weights":CONTEXT_SELECTION_WEIGHTS,"feedback_k":FEEDBACK_CANDIDATE_K},sort_keys=True).encode()).hexdigest()
EXPERIMENT_CONFIG={"experiment_id":EXPERIMENT_ID,"experiment_phase":EXPERIMENT_PHASE,"implementation_fingerprint":IMPLEMENTATION_FINGERPRINT,
 "model":MODEL_NAME,"seed":SEED,"validation_selection_seed":VALIDATION_SELECTION_SEED,
 "temperature":MAMMQA_TEMPERATURE,"top_p":MAMMQA_TOP_P,"do_sample":MAMMQA_DO_SAMPLE,"verifier_do_sample":False,
 "stage_token_limits":STAGE_MAX_NEW_TOKENS,"max_input_tokens":MAX_INPUT_TOKENS,
 "context_k_base":{"text":TEXT_CONTEXT_K,"table":TABLE_CONTEXT_K,"image":IMAGE_CONTEXT_K},
 "context_k_max":{"text":TEXT_CONTEXT_K_MAX,"table":TABLE_CONTEXT_K_MAX,"image":IMAGE_CONTEXT_K_MAX},
 "selection_weights":CONTEXT_SELECTION_WEIGHTS,"duplicate_jaccard_threshold":CONTEXT_DUPLICATE_THRESHOLD,
 "fallback_context_k":{"text":FALLBACK_TEXT_K,"table":FALLBACK_TABLE_K,"image":FALLBACK_IMAGE_K},
 "candidate_k":RETRIEVAL_CANDIDATE_K,"feedback_candidate_k":FEEDBACK_CANDIDATE_K,
 "retrieval_mode":RETRIEVAL_MODE,"router_mode":ROUTER_MODE,
 "graph":{"enabled":ENABLE_GRAPH,"seed_k":GRAPH_SEED_K,"max_expanded":GRAPH_MAX_EXPANDED_DOCS,"max_degree":GRAPH_MAX_DEGREE},
 "verifier":True,"feedback":True,"max_retrieval_rounds":MAX_RETRIEVAL_ROUNDS,"image_processor_use_fast":IMAGE_PROCESSOR_USE_FAST,
 "bge_device":BGE_DEVICE,"clip_device_requested":CLIP_DEVICE,"num_shards":NUM_SHARDS,"shard_id":SHARD_ID,
 "debug16_v1":DEBUG16_V1,"validation16_v2":VALIDATION16_V2,"qids":SELECTED_QIDS,"methods":METHODS,"web_enabled":ENABLE_WEB,
 "cache_read_roots":[str(x) for x in CACHE_READ_ROOTS]}
if RUNTIME_CONFIG_PATH.exists():
    prior_config=json.loads(RUNTIME_CONFIG_PATH.read_text())
    mutable_selection_fields={"qids"}
    assert {k:v for k,v in prior_config.items() if k not in mutable_selection_fields}=={k:v for k,v in EXPERIMENT_CONFIG.items() if k not in mutable_selection_fields},"Experiment id has incompatible inference settings"
RUNTIME_CONFIG_PATH.write_text(json.dumps(EXPERIMENT_CONFIG,indent=2),encoding="utf-8")

STARTUP_SECONDS=time.perf_counter()-NOTEBOOK_START_TIME
def run_full(rows):
    path=DIRS["predictions"]/prediction_name; existing=read_jsonl_checked(path); done=completed(path); completed_times=[float(r.get("latency_seconds",0)) for r in existing]
    timed={r.get("qid") for r in read_jsonl_checked(TIMING_LOG_PATH)}
    for row in existing:
        if row["qid"] not in timed:append_jsonl(TIMING_LOG_PATH,{"qid":row["qid"],**row.get("timing",{}),"qwen_call_count":row.get("qwen_call_count"),"format_repair_call_count":row.get("format_repair_call_count")})
    print(f"[FULL] {len(done)}/{len(rows)} already complete | startup_seconds={STARTUP_SECONDS:.1f}",flush=True)
    for position,raw in enumerate(rows,1):
        q=query_only(raw)
        if q.qid in done:print(f"[FULL] {position}/{len(rows)} RESUME-SKIP {q.qid}",flush=True); continue
        started=time.perf_counter(); result=clean_for_json(run_pipeline(q,CONFIGS["FULL"])); append_jsonl(path,result); done.add(q.qid)
        elapsed=time.perf_counter()-started; completed_times.append(elapsed); append_jsonl(TIMING_LOG_PATH,{"qid":q.qid,**result["timing"],"qwen_call_count":result["qwen_call_count"],"format_repair_call_count":result["format_repair_call_count"]})
        complete=len(done); avg=float(np.mean(completed_times)); remaining=max(0,len(rows)-complete)*avg/60
        print(f"[FULL] {complete}/{len(rows)} complete | time_this_question={elapsed:.1f}s | average_question_time={avg:.1f}s | estimated_remaining_minutes={remaining:.1f} | qwen_calls={result['qwen_call_count']} | repair_calls={result['format_repair_call_count']}",flush=True)
        if complete%4==0:gc.collect()
    return path

assert model is not None and processor is not None and METHODS==["FULL"]
INFERENCE_STARTED=time.perf_counter(); RESULT_PATHS={"FULL":run_full(SELECTED)}; INFERENCE_SECONDS=time.perf_counter()-INFERENCE_STARTED
PREDICTION_ROWS={"FULL":read_jsonl_checked(RESULT_PATHS["FULL"])}
assert [r["qid"] for r in PREDICTION_ROWS["FULL"]]==SELECTED_QIDS
assert all(set(r.get("inference_input_fields",[]))=={"qid","question"} for r in PREDICTION_ROWS["FULL"])
print("\nPREDICTIONS FROZEN. Inference boundary remained qid + question only.")
for r in PREDICTION_ROWS["FULL"]:print(r["qid"],"=>",r["final_answer"])
''')

set_cell(27,r'''
# Cell 28 - gold-free engineering gate OR post-freeze evaluation; never auto-launches another phase
def mean_valid(vals):
    vals=[float(v) for v in vals if v is not None and not (isinstance(v,float) and math.isnan(v))]; return float(np.mean(vals)) if vals else math.nan
def envelopes_for(row,kind):
    if kind=="planner":return [row.get("planner",{}).get("planner_generation")]
    if kind=="stage1":return [x for rr in row.get("reasoning_round_details",[]) for x in rr.get("experts",{}).values()]
    if kind=="stage2":return [x for rr in row.get("reasoning_round_details",[]) for x in rr.get("cross_modal",{}).values()]
    if kind=="verifier":return [rr.get("verification") for rr in row.get("reasoning_round_details",[]) if rr.get("verification") is not None]
    if kind=="aggregator":return [row.get("aggregator_raw")]
    return []
def stage_health(rows,kind):
    env=[x for r in rows for x in envelopes_for(r,kind) if x]; parsed=sum(bool(x.get("parse_success")) for x in env)
    return {"parsed":parsed,"total":len(env),"rate":parsed/len(env) if env else 0.0}
def stage1_usable_health(rows):
    env=[x for r in rows for x in envelopes_for(r,"stage1") if x]
    usable=sum(bool(x.get("parse_success") or str(x.get("raw_output","")).strip()) for x in env)
    return {"usable":usable,"total":len(env),"rate":usable/len(env) if env else 0.0}
def generation_flags(rows):
    env=[x for r in rows for k in ["planner","stage1","stage2","verifier","aggregator"] for x in envelopes_for(r,k) if x]
    diagnostics=[d for x in env for d in x.get("generation_diagnostics",[])]
    return {"oom_events":sum(bool(x.get("oom") or x.get("oom_retried")) for x in diagnostics),
            "protected_prompt_truncations":sum(bool(x.get("protected_prompt_truncated")) for x in diagnostics),
            "stage2_hit_max_new_tokens":sum(bool(d.get("hit_max_new_tokens")) for x in env for d in x.get("generation_diagnostics",[]) if str(d.get("stage","")).startswith("stage2_"))}
def status_distribution(rows):
    counts=Counter()
    for r in rows:
        v=r.get("verification") or {}; p=v.get("parsed_output") if v.get("parse_success") else None
        counts[(p or {}).get("status","verifier_error")]+=1
    return {k:counts.get(k,0) for k in ["supported","insufficient","conflict","verifier_error"]}

RAW_ROWS=PREDICTION_ROWS["FULL"]
flags=generation_flags(RAW_ROWS)
health={"experiment_phase":EXPERIMENT_PHASE,"question_count":len(RAW_ROWS),"checkpoint_count":len(read_jsonl_checked(RESULT_PATHS["FULL"])),
 "planner":stage_health(RAW_ROWS,"planner"),"stage1_schema":stage_health(RAW_ROWS,"stage1"),"stage1_usable":stage1_usable_health(RAW_ROWS),
 "stage2":stage_health(RAW_ROWS,"stage2"),"verifier":stage_health(RAW_ROWS,"verifier"),"aggregator":stage_health(RAW_ROWS,"aggregator"),
 "verifier_status_distribution":status_distribution(RAW_ROWS),"qwen_calls_per_question":[r["qwen_call_count"] for r in RAW_ROWS],
 "repair_calls_per_question":[r["format_repair_call_count"] for r in RAW_ROWS],"feedback_calls":sum(bool(r.get("targeted_feedback")) for r in RAW_ROWS),
 "feedback_executable_unit_test":all(FEEDBACK_UNIT_TESTS.values()),"question_seconds":[r["timing"]["total_seconds"] for r in RAW_ROWS],
 "startup_seconds":STARTUP_SECONDS,"inference_seconds":INFERENCE_SECONDS,**flags}
health["average_qwen_calls"]=mean_valid(health["qwen_calls_per_question"]); health["average_repair_calls"]=mean_valid(health["repair_calls_per_question"]); health["average_question_seconds"]=mean_valid(health["question_seconds"])
no_repair_chain=all(int(x)<=1 for x in health["repair_calls_per_question"])
READY_FOR_NEW_VALIDATION=(EXPERIMENT_PHASE=="ENGINEERING_GATE" and len(RAW_ROWS)==2 and health["checkpoint_count"]==2 and
 health["planner"]["rate"]==1.0 and health["stage1_usable"]["rate"]==1.0 and health["stage2"]["rate"]==1.0 and
 health["verifier"]["rate"]==1.0 and health["aggregator"]["rate"]==1.0 and all(all(x.values()) for x in UNIT_TEST_RESULTS.values()) and
 health["oom_events"]==0 and health["protected_prompt_truncations"]==0 and no_repair_chain and health["feedback_executable_unit_test"] and
 not any(r.get("final_decision",{}).get("infrastructure_failure") for r in RAW_ROWS))
health["READY_FOR_NEW_VALIDATION"]=READY_FOR_NEW_VALIDATION
(DIRS["metrics"]/"pipeline_health.json").write_text(json.dumps(health,indent=2,allow_nan=True),encoding="utf-8")

gate_rows=[{"qid":r["qid"],"final_answer":r["final_answer"],"runtime_seconds":r["timing"]["total_seconds"],"qwen_calls":r["qwen_call_count"],
 "feedback_calls":int(bool(r.get("targeted_feedback"))),"planner_parse":r["planner"]["planner_parse_success"],
 "stage1_usable_rate":stage1_usable_health([r])["rate"],"stage2_parse_rate":stage_health([r],"stage2")["rate"],
 "initial_verifier_status":r.get("verifier_diagnostics",{}).get("initial_verifier_status"),"feedback_triggered":bool(r.get("targeted_feedback")),
 "second_verifier_status":r.get("verifier_diagnostics",{}).get("second_verifier_status"),"aggregator_parse":stage_health([r],"aggregator")["rate"],
 "repair_calls":r["format_repair_call_count"]} for r in RAW_ROWS]
pd.DataFrame(gate_rows).to_csv(DIRS["metrics"]/"engineering_runtime.csv",index=False)

if EXPERIMENT_PHASE=="ENGINEERING_GATE":
    cache_status={"read_roots":[str(x) for x in CACHE_READ_ROOTS],"export_root":str(CACHE_EXPORT_ROOT),
                  "valid_read_root_found":bool(CACHE_READ_ROOTS),"note":"Missing/invalid artifacts rebuild once and export for mounting as a Kaggle dataset."}
    report={"scope":"GOLD_FREE_2_QUESTION_ENGINEERING_GATE","root_cause":"Candidate retrieval was deep (K=20), but fixed top slices ignored query coverage and redundancy, causing useful candidates to be discarded before reasoning.",
     "context_selector":"Original-question preservation + subquery coverage + modality/graph bonuses + duplicate-aware MMR + bounded adaptive K.",
     "verifier_rule":"A directly evidenced concrete candidate may be supported; INSUFFICIENT agents are neutral; two supported incompatible candidates are conflict.",
     "feedback_rule":"Missing retry fields deterministically fall back to planner need and unresolved subquery/original question; one full-index target-modality retry maximum.",
     "aggregator_rule":"Preserve verifier-supported best answer; abstain only for final insufficiency, unresolved conflict, or explicit evidence/generation reason.",
     "unit_tests":UNIT_TEST_RESULTS,"runtime_rows":gate_rows,"cache_status":cache_status,"health":health,
     "READY_FOR_NEW_VALIDATION":READY_FOR_NEW_VALIDATION,"automatic_next_run":False}
    (DIRS["metrics"]/"engineering_gate_report.json").write_text(json.dumps(report,indent=2,allow_nan=True),encoding="utf-8")
    print("\n==================================================\nTARGETED REPAIR: 2-QUESTION GOLD-FREE ENGINEERING GATE\n==================================================")
    print(pd.DataFrame(gate_rows).to_string(index=False)); print("\nVERIFIER STATUS DISTRIBUTION",health["verifier_status_distribution"])
    print("OOM =",health["oom_events"],"| protected truncation =",health["protected_prompt_truncations"],"| Stage-II hit cap =",health["stage2_hit_max_new_tokens"])
    print("UNIT TESTS =",json.dumps(UNIT_TEST_RESULTS,indent=2)); print("CACHE STATUS =",json.dumps(cache_status,indent=2))
    print("\nREADY_FOR_NEW_VALIDATION =",READY_FOR_NEW_VALIDATION)
    print("STOP: no gold was materialized and VALIDATION16_V2 was not launched.")
else:
    # Gold is materialized only here, after all selected predictions have been frozen.
    evaluated=[]
    for prediction in RAW_ROWS:
        ev=evaluation_only(row_by_qid[prediction["qid"]]); row=copy.deepcopy(prediction); em,f1=em_f1(row["final_answer"],ev.gold_answers)
        gold=set(map(str,ev.gold_support_ids)); candidates={str(x) for xs in row.get("candidate_ids",{}).values() for x in xs}; selected={str(x) for xs in row.get("context_ids",{}).values() for x in xs}
        candidate_gold=sorted(gold&candidates); selected_gold=sorted(gold&selected); lost_gold=sorted((gold&candidates)-selected)
        retention=len(selected_gold)/len(candidate_gold) if candidate_gold else math.nan
        stage2_values=[]
        for env in envelopes_for(row,"stage2"):
            parsed=(env or {}).get("parsed_output") or {}; value=parsed.get("candidate_answer","")
            if value:stage2_values.append(value)
        gold_present=any(em_f1(v,ev.gold_answers)[0]==1 for v in stage2_values) if row.get("abstain") else False
        row.update(gold_answers=ev.gold_answers,gold_support_ids=ev.gold_support_ids,gold_support_modalities=ev.gold_support_modalities,
                   exact_match=em,token_f1=f1,retrieval_metrics=retrieval_scores(row,ev),candidate_gold_ids=candidate_gold,
                   selected_gold_ids=selected_gold,lost_gold_ids=lost_gold,gold_context_retention=retention,gold_present_in_stage2=gold_present)
        evaluated.append(row)
    FULL_ROWS=evaluated

    def all_parent_ids(row,pool):return {str(d["parent_doc_id"]) for docs in (row.get(pool,{}) or {}).values() for d in docs}
    def classify_failure(row):
        if row["exact_match"]:return "correct"
        if row.get("planner",{}).get("planner_parse_failure"):return "planner_parse_failure"
        if stage1_usable_health([row])["rate"]<1:return "stage1_generation_failure"
        if stage_health([row],"stage2")["rate"]<1:return "stage2_generation_failure"
        if stage_health([row],"verifier")["rate"]<1:return "verifier_error"
        if stage_health([row],"aggregator")["rate"]<1:return "aggregation_failure"
        gold=set(map(str,row.get("gold_support_ids",[]))); candidates=all_parent_ids(row,"retrieval_candidates"); context=all_parent_ids(row,"context_evidence")
        if gold and not gold<=candidates:return "retrieval_failure"
        if gold and not gold<=context:return "context_selection_failure"
        if row.get("gold_present_in_stage2") and row.get("abstain"):return "avoidable_abstention"
        if row.get("abstention_reason"):return row["abstention_reason"]
        return "answer_reasoning_failure"

    for r in FULL_ROWS:r["failure_type"]=classify_failure(r)
    def metric(name):return mean_valid([r.get("retrieval_metrics",{}).get(name) for r in FULL_ROWS])
    answered=[r for r in FULL_ROWS if not r.get("abstain")]; correct_answered=sum(r["exact_match"] for r in answered)
    metrics={"phase":EXPERIMENT_PHASE,"n":len(FULL_ROWS),"EM":mean_valid([r["exact_match"] for r in FULL_ROWS]),"token_F1":mean_valid([r["token_f1"] for r in FULL_ROWS]),
     "coverage":len(answered)/len(FULL_ROWS),"abstention_rate":1-len(answered)/len(FULL_ROWS),"answered_correct":int(correct_answered),
     "answered_wrong":int(len(answered)-correct_answered),"abstained":int(len(FULL_ROWS)-len(answered)),
     "answered_accuracy":correct_answered/len(answered) if answered else math.nan,
     "R@1":metric("recall@1"),"R@3":metric("recall@3"),"R@5":metric("recall@5"),"R@10":metric("recall@10"),"MRR":metric("mrr"),
     "candidate_gold_coverage":metric("gold_candidate_pool_coverage"),"selected_context_gold_coverage":metric("gold_context_coverage"),
     "gold_context_retention":mean_valid([r["gold_context_retention"] for r in FULL_ROWS]),"verifier_status_distribution":status_distribution(FULL_ROWS),
     "feedback_activation_rate":mean_valid([bool(r.get("targeted_feedback")) for r in FULL_ROWS]),
     "feedback_success_rate":mean_valid([bool((r.get("targeted_feedback") or {}).get("feedback_new_evidence")) for r in FULL_ROWS if r.get("targeted_feedback")]),
     "stage2_gold_present_rate_for_abstained":mean_valid([bool(r.get("gold_present_in_stage2")) for r in FULL_ROWS if r.get("abstain")]),
     "failure_taxonomy":dict(Counter(r["failure_type"] for r in FULL_ROWS))}
    diagnostic_rows=[{"qid":r["qid"],"final_answer":r["final_answer"],"exact_match":r["exact_match"],"token_f1":r["token_f1"],
      "candidate_gold_ids":"|".join(r["candidate_gold_ids"]),"selected_gold_ids":"|".join(r["selected_gold_ids"]),"lost_gold_ids":"|".join(r["lost_gold_ids"]),
      "gold_context_retention":r["gold_context_retention"],"gold_present_in_stage2":r["gold_present_in_stage2"],"abstention_reason":r.get("abstention_reason"),
      "initial_verifier_status":r.get("verifier_diagnostics",{}).get("initial_verifier_status"),"feedback_used":bool(r.get("targeted_feedback")),
      "second_verifier_status":r.get("verifier_diagnostics",{}).get("second_verifier_status"),"failure_type":r["failure_type"]} for r in FULL_ROWS]
    pd.DataFrame(diagnostic_rows).to_csv(DIRS["metrics"]/"postfreeze_diagnostics.csv",index=False)
    (DIRS["metrics"]/"validation_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=True),encoding="utf-8")
    if EXPERIMENT_PHASE=="DEBUG16_REGRESSION":
        old={"label":"OLD FULL / DEBUG16 development baseline","candidate_gold_coverage":0.7604,"selected_context_gold_coverage":0.2396,
             "verifier_insufficient_rate":1.0,"feedback_activation_rate":0.0,"abstention_rate":0.75,"EM":0.0,"token_F1":0.0}
        comparison={"warning":"DEBUG16 is development/regression data, not unbiased test performance.","old":old,"new":metrics}
        (DIRS["metrics"]/"debug16_old_vs_new.json").write_text(json.dumps(comparison,indent=2,allow_nan=True),encoding="utf-8")
        print("\nDEBUG16 DEVELOPMENT/REGRESSION COMPARISON (NOT UNBIASED TEST PERFORMANCE)\n",json.dumps(comparison,indent=2,allow_nan=True))
    else:print("\nVALIDATION16_V2 POST-FREEZE METRICS\n",json.dumps(metrics,indent=2,allow_nan=True))
    print("STOP: this phase is complete; no other question set was launched automatically.")
''')

nb['metadata']['targeted_scientific_repair']={
    'version':'v2','default_phase':'ENGINEERING_GATE','gold_free_gate_questions':2,
    'debug_set':'DEBUG16_V1','validation_set':'VALIDATION16_V2','auto_launch_validation':False
}
for cell in nb['cells']:
    if cell.get('cell_type')=='code':
        cell['execution_count']=None; cell['outputs']=[]
TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print(TARGET)
