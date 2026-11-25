import json
from pathlib import Path

SOURCE=Path(r"HM-RAG + MAMMQA 16-question smoke experiment\hm-rag-mammqa-targeted-scientific-repair-v2.ipynb")
TARGET=Path(r"HM-RAG + MAMMQA 16-question smoke experiment\hm-rag-mammqa-targeted-scientific-repair-v2-VALIDATION16.ipynb")
nb=json.loads(SOURCE.read_text(encoding="utf-8"))
assert len(nb["cells"])==28

def source(i):return "".join(nb["cells"][i]["source"])
def set_cell(i,text):
    cell=nb["cells"][i]; cell["source"]=text.strip("\n").splitlines(keepends=True)
    if cell.get("cell_type")=="code":cell["execution_count"]=None; cell["outputs"]=[]

set_cell(0,r'''
# HM-RAG + MAMMQA: frozen VALIDATION16_V2 experiment

This notebook runs the already-repaired pipeline on the deterministic, disjoint 16-question
`VALIDATION16_V2` set. The QA implementation and inference settings are frozen: this is a
measurement run, not a tuning run.

Predictions are checkpointed after every question. Gold answers and support annotations are
materialized only after all 16 predictions have been frozen.
''')

cell3=source(3)
cell3=cell3.replace('EXPERIMENT_PHASE="ENGINEERING_GATE"  # ENGINEERING_GATE | DEBUG16_REGRESSION | VALIDATION16_V2',
                    'EXPERIMENT_PHASE="VALIDATION16_V2"  # frozen measurement run; do not tune mid-run')
cell3=cell3.replace('print("Persistent cache roots discovered:",[str(p) for p in CACHE_READ_ROOTS])',
'''print("Persistent cache roots discovered:",[str(p) for p in CACHE_READ_ROOTS])
CACHE_STATUS={}  # reporting only; never changes retrieval semantics''')
set_cell(3,cell3)

cell7=source(7)
cell7=cell7.replace('print("Loaded persistent image manifest:",manifest_path); return mapping',
'''CACHE_STATUS["image_manifest"]="HIT"; print("CACHE HIT / BUILT image_manifest: HIT",manifest_path); return mapping''')
cell7=cell7.replace('print("Created publishable image manifest:",out)',
'''CACHE_STATUS["image_manifest"]="BUILT"; print("CACHE HIT / BUILT image_manifest: BUILT",out)''')
cell7=cell7.replace('IMAGE_PATHS=discover_image_paths(images_raw)',
'''IMAGE_PATHS=discover_image_paths(images_raw)
if "image_manifest" not in CACHE_STATUS:
    CACHE_STATUS["image_manifest"]="BUILT_IN_MEMORY"; print("CACHE HIT / BUILT image_manifest: BUILT_IN_MEMORY")''')
set_cell(7,cell7)

cell9=source(9)
cell9=cell9.replace('self.index=faiss.read_index(str(paths["index"])); print(f"Loaded persistent {self.modality} index from",root); return',
'''self.index=faiss.read_index(str(paths["index"])); CACHE_STATUS[f"{self.modality}_index"]="HIT"; print(f"CACHE HIT / BUILT {self.modality} index: HIT",root); return''')
cell9=cell9.replace('print(f"Built {self.modality} cache package under",CACHE_EXPORT_ROOT)',
'''CACHE_STATUS[f"{self.modality}_index"]="BUILT"; print(f"CACHE HIT / BUILT {self.modality} index: BUILT",CACHE_EXPORT_ROOT)''')
cell9=cell9.replace('if payload.get("fingerprint")==fp:self.index=payload["index"]; print(f"Loaded persistent {self.modality} BM25 from",root); return',
'''if payload.get("fingerprint")==fp:self.index=payload["index"]; CACHE_STATUS[f"{self.modality}_bm25"]="HIT"; print(f"CACHE HIT / BUILT {self.modality} BM25: HIT",root); return''')
cell9=cell9.replace('self.index=BM25Okapi([bm25_tokens(d["retrieval_text"]) for d in self.records]); paths["bm25"].write_bytes(pickle.dumps({"fingerprint":fp,"index":self.index},pickle.HIGHEST_PROTOCOL))',
'''self.index=BM25Okapi([bm25_tokens(d["retrieval_text"]) for d in self.records]); paths["bm25"].write_bytes(pickle.dumps({"fingerprint":fp,"index":self.index},pickle.HIGHEST_PROTOCOL))
        CACHE_STATUS[f"{self.modality}_bm25"]="BUILT"; print(f"CACHE HIT / BUILT {self.modality} BM25: BUILT",CACHE_EXPORT_ROOT)''')
set_cell(9,cell9)

# Add flat reporting aliases to the prediction row without changing inference or decision logic.
cell21=source(21)
old='''"reasoning_round_details":reasoning_round_details,"synthesis_consensus":consensus,"verifier_diagnostics":verifier_diag,
            "targeted_feedback":targeted,"retrieval_rounds":rounds,"new_docs_added":added,"final_answer":answer,"abstain":answer=="ABSTAIN",
            "abstention_reason":abstention_reason,"forced_parser_abstain":False,"timing":timing,
            "qwen_call_count":QWEN_CALL_COUNTER[q.qid]-calls0,"format_repair_call_count":FORMAT_REPAIR_CALL_COUNTER[q.qid]-repairs0,"latency_seconds":timing["total_seconds"]}'''
new='''"reasoning_round_details":reasoning_round_details,"synthesis_consensus":consensus,"verifier_diagnostics":verifier_diag,
            "stage1_outputs":experts,"stage2_answers":consensus.get("stage2_answers",[]),
            "initial_verifier_status":verifier_diag["initial_verifier_status"],"initial_verifier_best_answer":verifier_diag["initial_best_answer"],
            "feedback_used":bool(targeted),"retry_modality":verifier_diag["retry_modality"],"retry_query":verifier_diag["retry_query"],
            "feedback_candidate_ids":(targeted or {}).get("new_candidate_ids",{}),"feedback_new_evidence_count":(targeted or {}).get("new_unique_evidence_count",0),
            "feedback_context_ids":(targeted or {}).get("new_context_ids",{}),"second_verifier_status":verifier_diag["second_verifier_status"],
            "second_verifier_best_answer":verifier_diag["second_best_answer"],"aggregator_output":agg,
            "targeted_feedback":targeted,"retrieval_rounds":rounds,"new_docs_added":added,"final_answer":answer,"abstain":answer=="ABSTAIN",
            "abstention_reason":abstention_reason,"forced_parser_abstain":False,"timing":timing,"total_seconds":timing["total_seconds"],
            "qwen_call_count":QWEN_CALL_COUNTER[q.qid]-calls0,"format_repair_call_count":FORMAT_REPAIR_CALL_COUNTER[q.qid]-repairs0,"latency_seconds":timing["total_seconds"]}'''
assert old in cell21
cell21=cell21.replace(old,new)
set_cell(21,cell21)

set_cell(24,r'''
# Cell 25 - reuse or deterministically freeze the disjoint VALIDATION16_V2 set before inference
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
def deterministic_validation16(rows):
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
    assert len(selected)==16
    return [query_only(r).qid for r in selected]

# Reuse any previously frozen set that matches the fixed debug set and seed. Conflicting
# valid frozen sets are a hard stop rather than an invitation to choose after seeing results.
frozen_candidates=[OUTPUT_ROOT/"frozen_question_sets.json",
                   BASE_OUTPUT_ROOT/"qa_targeted_repair_v2_gate_seed42"/"frozen_question_sets.json",
                   BASE_OUTPUT_ROOT/"qa_targeted_repair_v2_validation16_seed42"/"frozen_question_sets.json"]
if input_root.exists():frozen_candidates.extend(input_root.rglob("frozen_question_sets.json"))
valid_frozen=[]
for path in dict.fromkeys(Path(p) for p in frozen_candidates):
    if not path.exists():continue
    try:
        payload=json.loads(path.read_text(encoding="utf-8")); candidate=list(map(str,payload.get("VALIDATION16_V2",[])))
        seed=int(payload.get("validation_selection_seed",VALIDATION_SELECTION_SEED)); debug=list(map(str,payload.get("DEBUG16_V1",[])))
        if len(candidate)==16 and len(set(candidate))==16 and seed==VALIDATION_SELECTION_SEED and debug==DEBUG16_V1 and all(q in row_by_qid for q in candidate):
            valid_frozen.append((path,candidate))
    except Exception as exc:print("Ignoring invalid frozen-set file",path,exc)
if valid_frozen:
    unique_sets={tuple(qids) for _,qids in valid_frozen}
    assert len(unique_sets)==1,f"STOP: conflicting pre-existing VALIDATION16_V2 sets: {[str(p) for p,_ in valid_frozen]}"
    VALIDATION16_V2=list(next(iter(unique_sets))); VALIDATION_SET_SOURCE=str(valid_frozen[0][0])
else:
    VALIDATION16_V2=deterministic_validation16(dev_raw); VALIDATION_SET_SOURCE="deterministic_generation"

intersection=set(DEBUG16_V1)&set(VALIDATION16_V2)
print("DEBUG16 count =",len(DEBUG16_V1)); print("VALIDATION16_V2 count =",len(VALIDATION16_V2)); print("intersection count =",len(intersection))
assert len(DEBUG16_V1)==len(set(DEBUG16_V1))==16
assert len(VALIDATION16_V2)==len(set(VALIDATION16_V2))==16
assert not intersection,"STOP: DEBUG16_V1 and VALIDATION16_V2 overlap"
assert EXPERIMENT_PHASE=="VALIDATION16_V2" and FAST_VALIDATION is False

frozen_payload={"experiment_phase":EXPERIMENT_PHASE,"DEBUG16_V1":DEBUG16_V1,"VALIDATION16_V2":VALIDATION16_V2,
                "validation_selection_seed":VALIDATION_SELECTION_SEED,"source":VALIDATION_SET_SOURCE}
(OUTPUT_ROOT/"frozen_question_sets.json").write_text(json.dumps(frozen_payload,indent=2),encoding="utf-8")
active_qids=VALIDATION16_V2
SELECTED_BASE=[row_by_qid[qid] for qid in active_qids]; SELECTED=SELECTED_BASE[SHARD_ID::NUM_SHARDS]; SELECTED_QIDS=[query_only(r).qid for r in SELECTED]
assert NUM_SHARDS==1 and SHARD_ID==0 and SELECTED_QIDS==VALIDATION16_V2 and len(SELECTED_QIDS)==16
selected_public=[{"experiment_phase":EXPERIMENT_PHASE,"selection_seed":VALIDATION_SELECTION_SEED,
                  "qid":query_only(r).qid,"question":query_only(r).question} for r in SELECTED]
assert all(set(x)=={"experiment_phase","selection_seed","qid","question"} for x in selected_public)
(OUTPUT_ROOT/"selected_questions.json").write_text(json.dumps(selected_public,indent=2,ensure_ascii=False),encoding="utf-8")
assert all(set(asdict(query_only(r)))=={"qid","question"} for r in SELECTED)
print("VALIDATION16_V2 source =",VALIDATION_SET_SOURCE); print("selected qids =",SELECTED_QIDS)
print("LEAKAGE CHECK PASSED")
''')

set_cell(26,r'''
# Cell 27 - frozen FULL-only VALIDATION16_V2 inference with per-question checkpointing
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
METHODS=["FULL"]; prediction_name="FULL.jsonl"
FINGERPRINT_PAYLOAD={"model":MODEL_NAME,"seed":SEED,"prompts":[PLANNER_PROMPT,EXPERT_PROMPT,SYNTH_PROMPT,VERIFY_PROMPT,AGG_PROMPT],
 "retrieval":{"mode":RETRIEVAL_MODE,"router":ROUTER_MODE,"candidate_k":RETRIEVAL_CANDIDATE_K,"rrf_k":RRF_K},
 "context":{"base":[TEXT_CONTEXT_K,TABLE_CONTEXT_K,IMAGE_CONTEXT_K],"max":[TEXT_CONTEXT_K_MAX,TABLE_CONTEXT_K_MAX,IMAGE_CONTEXT_K_MAX],
            "fallback":[FALLBACK_TEXT_K,FALLBACK_TABLE_K,FALLBACK_IMAGE_K],"weights":CONTEXT_SELECTION_WEIGHTS,"duplicate_threshold":CONTEXT_DUPLICATE_THRESHOLD},
 "graph":{"enabled":ENABLE_GRAPH,"seed_k":GRAPH_SEED_K,"max_expanded":GRAPH_MAX_EXPANDED_DOCS,"max_degree":GRAPH_MAX_DEGREE},
 "generation":{"do_sample":MAMMQA_DO_SAMPLE,"temperature":MAMMQA_TEMPERATURE,"top_p":MAMMQA_TOP_P,"stage_tokens":STAGE_MAX_NEW_TOKENS,"max_input":MAX_INPUT_TOKENS},
 "verifier":{"do_sample":VERIFIER_DO_SAMPLE,"protocol":VERIFY_PROMPT},"feedback":{"candidate_k":FEEDBACK_CANDIDATE_K,"max_rounds":MAX_RETRIEVAL_ROUNDS},
 "aggregator":{"protocol":AGG_PROMPT,"repair_stages":sorted(REPAIR_STAGES)}}
IMPLEMENTATION_FINGERPRINT=hashlib.sha256(json.dumps(FINGERPRINT_PAYLOAD,sort_keys=True).encode()).hexdigest()
EXPERIMENT_CONFIG={"experiment_id":EXPERIMENT_ID,"experiment_phase":EXPERIMENT_PHASE,"implementation_fingerprint":IMPLEMENTATION_FINGERPRINT,
 "fingerprint_payload":FINGERPRINT_PAYLOAD,"validation_selection_seed":VALIDATION_SELECTION_SEED,"debug16_v1":DEBUG16_V1,
 "validation16_v2":VALIDATION16_V2,"qids":SELECTED_QIDS,"methods":METHODS,"cache_read_roots":[str(x) for x in CACHE_READ_ROOTS]}
if RUNTIME_CONFIG_PATH.exists():
    prior_config=json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    assert prior_config==EXPERIMENT_CONFIG,"STOP: resumed run has a mismatching implementation/config fingerprint or qid set"
RUNTIME_CONFIG_PATH.write_text(json.dumps(EXPERIMENT_CONFIG,indent=2),encoding="utf-8")
print("IMPLEMENTATION_FINGERPRINT =",IMPLEMENTATION_FINGERPRINT)

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
        elapsed=time.perf_counter()-started; completed_times.append(elapsed)
        append_jsonl(TIMING_LOG_PATH,{"qid":q.qid,**result["timing"],"qwen_call_count":result["qwen_call_count"],"format_repair_call_count":result["format_repair_call_count"]})
        complete=len(done); avg=float(np.mean(completed_times)); remaining=max(0,len(rows)-complete)*avg/60
        print(f"[FULL] {complete}/{len(rows)} complete | time_this_question={elapsed:.1f}s | average_question_time={avg:.1f}s | estimated_remaining_minutes={remaining:.1f} | qwen_calls={result['qwen_call_count']} | repair_calls={result['format_repair_call_count']}",flush=True)
        if complete%4==0:gc.collect()
    return path

assert EXPERIMENT_PHASE=="VALIDATION16_V2" and FAST_VALIDATION is False and SELECTED_QIDS==VALIDATION16_V2
assert model is not None and processor is not None and METHODS==["FULL"]
INFERENCE_STARTED=time.perf_counter(); RESULT_PATHS={"FULL":run_full(SELECTED)}; INFERENCE_SECONDS=time.perf_counter()-INFERENCE_STARTED
PREDICTION_ROWS={"FULL":read_jsonl_checked(RESULT_PATHS["FULL"])}
assert [r["qid"] for r in PREDICTION_ROWS["FULL"]]==SELECTED_QIDS and len(PREDICTION_ROWS["FULL"])==16
assert all(set(r.get("inference_input_fields",[]))=={"qid","question"} for r in PREDICTION_ROWS["FULL"])
print("\nPREDICTIONS FROZEN. Gold evaluation begins only now.")
for r in PREDICTION_ROWS["FULL"]:print(r["qid"],"=>",r["final_answer"])
''')

set_cell(27,r'''
# Cell 28 - research-quality post-freeze VALIDATION16_V2 evaluation and reports
assert EXPERIMENT_PHASE=="VALIDATION16_V2" and len(PREDICTION_ROWS["FULL"])==16
RAW_ROWS=PREDICTION_ROWS["FULL"]; PRE_EVAL_QWEN_CALLS=dict(QWEN_CALL_COUNTER); PRE_EVAL_REPAIR_CALLS=dict(FORMAT_REPAIR_CALL_COUNTER)

def mean_valid(values):
    values=[float(v) for v in values if v is not None and not (isinstance(v,float) and math.isnan(v))]
    return float(np.mean(values)) if values else math.nan
def envelope_rounds(row,key):return [x for rr in row.get("reasoning_round_details",[]) for x in rr.get(key,{}).values() if x]
def planner_envelope(row):return row.get("planner",{}).get("planner_generation") or {}
def aggregator_envelope(row):return row.get("aggregator_raw") or {}
def initial_verifier_envelope(row):return row.get("initial_verification") or {}
def second_verifier_envelope(row):
    return row.get("verification") or {} if row.get("targeted_feedback") else {}
def usable(envelope):return bool(envelope and (envelope.get("parse_success") or str(envelope.get("raw_output","")).strip()))
def modality_health(rows,kind,modality):
    env=[]
    for row in rows:
        for rr in row.get("reasoning_round_details",[]):
            group=rr.get("experts" if kind=="stage1" else "cross_modal",{})
            if group.get(modality):env.append(group[modality])
    success=sum(usable(x) if kind=="stage1" else bool(x.get("parse_success")) for x in env)
    return {"success":success,"total":len(env),"rate":success/len(env) if env else math.nan}
def parsed_health(envelopes):
    env=[x for x in envelopes if x]; success=sum(bool(x.get("parse_success")) for x in env)
    return {"success":success,"total":len(env),"rate":success/len(env) if env else math.nan}
def verifier_status(envelope):
    parsed=envelope.get("parsed_output") if envelope and envelope.get("parse_success") else None
    return (parsed or {}).get("status","verifier_error")
def status_summary(envelopes):
    counts=Counter(verifier_status(x) for x in envelopes); total=len(envelopes)
    return {k:{"count":counts.get(k,0),"percentage":counts.get(k,0)/total if total else math.nan} for k in ["supported","insufficient","conflict","verifier_error"]}
def generation_diagnostics(rows):
    diagnostics=[]
    for row in rows:
        envelopes=[planner_envelope(row),aggregator_envelope(row),initial_verifier_envelope(row)]
        if row.get("targeted_feedback"):envelopes.append(row.get("verification") or {})
        envelopes+=envelope_rounds(row,"experts")+envelope_rounds(row,"cross_modal")
        diagnostics.extend(d for env in envelopes if env for d in env.get("generation_diagnostics",[]))
    return diagnostics

# Gold annotations are first accessed below, after the frozen-prediction assertion in Cell 27.
FULL_ROWS=[]
for prediction in RAW_ROWS:
    ev=evaluation_only(row_by_qid[prediction["qid"]]); row=copy.deepcopy(prediction); em,f1=em_f1(row["final_answer"],ev.gold_answers)
    gold=set(map(str,ev.gold_support_ids)); candidates={str(x) for xs in row.get("candidate_ids",{}).values() for x in xs}; selected={str(x) for xs in row.get("context_ids",{}).values() for x in xs}
    candidate_gold=sorted(gold&candidates); selected_gold=sorted(gold&selected); lost_gold=sorted((gold&candidates)-selected)
    retention=len(selected_gold)/len(candidate_gold) if candidate_gold else math.nan
    consensus=row.get("synthesis_consensus",{}); stage2_values=[]
    for envelope in envelope_rounds(row,"cross_modal"):
        parsed=envelope.get("parsed_output") if envelope.get("parse_success") else None
        value=str((parsed or {}).get("candidate_answer","")).strip()
        if value:stage2_values.append(value)
    gold_present=bool(row.get("abstain") and any(em_f1(v,ev.gold_answers)[0]==1 for v in stage2_values))
    row.update(gold_answers=ev.gold_answers,gold_support_ids=ev.gold_support_ids,gold_support_modalities=ev.gold_support_modalities,
               exact_match=em,token_f1=f1,retrieval_metrics=retrieval_scores(row,ev),candidate_gold_ids_found=candidate_gold,
               selected_gold_ids_found=selected_gold,lost_gold_ids=lost_gold,gold_context_retention=retention,
               gold_present_in_stage2=gold_present,answered_wrong=bool(not row.get("abstain") and not em))
    FULL_ROWS.append(row)
assert dict(QWEN_CALL_COUNTER)==PRE_EVAL_QWEN_CALLS and dict(FORMAT_REPAIR_CALL_COUNTER)==PRE_EVAL_REPAIR_CALLS,"Post-gold generation is forbidden"

def classify_failure(row):
    if row["exact_match"]:return "correct"
    if not row.get("final_decision",{}).get("answer_parse_success") or row.get("final_decision",{}).get("infrastructure_failure"):return "answer_format_failure"
    gold=set(map(str,row.get("gold_support_ids",[]))); candidates={str(x) for xs in row.get("candidate_ids",{}).values() for x in xs}; selected={str(x) for xs in row.get("context_ids",{}).values() for x in xs}
    if gold and not gold<=candidates:return "retrieval_failure"
    if gold and not gold<=selected:return "context_selection_failure"
    final_status=verifier_status(row.get("verification") or {}); initial_status=verifier_status(row.get("initial_verification") or {})
    if not row.get("abstain") and final_status=="supported":return "wrong_supported_answer"
    if row.get("gold_present_in_stage2") and final_status!="supported":return "verifier_failure"
    if row.get("targeted_feedback") and initial_status in {"insufficient","conflict"} and final_status!="supported":return "feedback_failure"
    if row.get("abstain") and not gold:return "legitimate_insufficient_abstention"
    consensus=row.get("synthesis_consensus",{})
    if consensus.get("concrete_answer_count",0)==0:return "reasoning_failure"
    if row.get("abstain") and final_status in {"insufficient","conflict"}:return "legitimate_insufficient_abstention"
    if row.get("abstain"):return "aggregation_failure"
    return "reasoning_failure"
for row in FULL_ROWS:row["failure_type"]=classify_failure(row)

def metric(name):return mean_valid([r["retrieval_metrics"].get(name) for r in FULL_ROWS])
answered=[r for r in FULL_ROWS if not r.get("abstain")]; abstained=[r for r in FULL_ROWS if r.get("abstain")]
exact_correct=sum(int(r["exact_match"]) for r in FULL_ROWS); answered_correct=sum(int(r["exact_match"]) for r in answered); answered_wrong=len(answered)-answered_correct
initial_envelopes=[initial_verifier_envelope(r) for r in FULL_ROWS]; second_envelopes=[r.get("verification") or {} for r in FULL_ROWS if r.get("targeted_feedback")]
initial_summary=status_summary(initial_envelopes); second_summary=status_summary(second_envelopes)
feedback_rows=[r for r in FULL_ROWS if r.get("targeted_feedback")]; new_evidence_count=sum(bool((r.get("targeted_feedback") or {}).get("feedback_new_evidence")) for r in feedback_rows)
feedback_success=sum(verifier_status(r.get("initial_verification") or {}) in {"insufficient","conflict"} and verifier_status(r.get("verification") or {})=="supported" for r in feedback_rows)
feedback_summary={"feedback_trigger_count":len(feedback_rows),"feedback_activation_rate":len(feedback_rows)/len(FULL_ROWS),
 "new_evidence_found_count":new_evidence_count,"feedback_success_count":int(feedback_success),
 "feedback_success_definition":"initial insufficient/conflict + feedback ran + second verifier supported",
 "feedback_success_rate":feedback_success/len(feedback_rows) if feedback_rows else math.nan}
context_summary={"gold_support_total":sum(len(r["gold_support_ids"]) for r in FULL_ROWS),
 "candidate_support_found":sum(len(r["candidate_gold_ids_found"]) for r in FULL_ROWS),
 "selected_support_retained":sum(len(r["selected_gold_ids_found"]) for r in FULL_ROWS),
 "support_lost_during_context_selection":sum(len(r["lost_gold_ids"]) for r in FULL_ROWS),
 "candidate_gold_support_coverage":metric("gold_candidate_pool_coverage"),"selected_context_gold_support_coverage":metric("gold_context_coverage"),
 "mean_gold_context_retention":mean_valid([r["gold_context_retention"] for r in FULL_ROWS])}
summary_row={"experiment":EXPERIMENT_ID,"n":len(FULL_ROWS),"exact_correct":exact_correct,"EM":exact_correct/len(FULL_ROWS),
 "mean_token_F1":mean_valid([r["token_f1"] for r in FULL_ROWS]),"answered_count":len(answered),"answer_rate":len(answered)/len(FULL_ROWS),
 "abstention_count":len(abstained),"abstention_rate":len(abstained)/len(FULL_ROWS),"answered_correct":answered_correct,
 "answered_wrong":answered_wrong,"answered_accuracy":answered_correct/len(answered) if answered else math.nan}
retrieval_row={"experiment":EXPERIMENT_ID,"n":len(FULL_ROWS),"R@1":metric("recall@1"),"R@3":metric("recall@3"),"R@5":metric("recall@5"),
 "R@10":metric("recall@10"),"MRR":metric("mrr"),**{k:context_summary[k] for k in ["candidate_gold_support_coverage","selected_context_gold_support_coverage","mean_gold_context_retention"]}}

diagnostics=generation_diagnostics(FULL_ROWS); stage2_cap_hits=sum(bool(x.get("hit_max_new_tokens")) for x in diagnostics if str(x.get("stage","")).startswith("stage2_"))
oom_count=sum(bool(x.get("oom") or x.get("oom_retried")) for x in diagnostics); protected_truncations=sum(bool(x.get("protected_prompt_truncated")) for x in diagnostics)
health={"experiment_phase":EXPERIMENT_PHASE,"questions_completed":len(FULL_ROWS),"questions_expected":16,
 "planner":parsed_health([planner_envelope(r) for r in FULL_ROWS]),
 "stage1":{"text":modality_health(FULL_ROWS,"stage1","text"),"table":modality_health(FULL_ROWS,"stage1","table"),"image":modality_health(FULL_ROWS,"stage1","image")},
 "stage2":{"text":modality_health(FULL_ROWS,"stage2","text"),"table":modality_health(FULL_ROWS,"stage2","table"),"image":modality_health(FULL_ROWS,"stage2","image")},
 "initial_verifier":parsed_health(initial_envelopes),"second_verifier":parsed_health(second_envelopes),
 "aggregator":parsed_health([aggregator_envelope(r) for r in FULL_ROWS]),"oom_count":oom_count,"protected_prompt_truncations":protected_truncations,
 "repair_count":sum(r["format_repair_call_count"] for r in FULL_ROWS),"stage2_max_token_hit_count":stage2_cap_hits,
 "mean_qwen_calls_per_question":mean_valid([r["qwen_call_count"] for r in FULL_ROWS]),"mean_runtime_seconds_per_question":mean_valid([r["timing"]["total_seconds"] for r in FULL_ROWS]),
 "verifier_calls_per_question":[1+int(bool(r.get("targeted_feedback"))) for r in FULL_ROWS],"feedback_calls":len(feedback_rows),"cache_status":CACHE_STATUS}

per_question=[]
for r in FULL_ROWS:
    rm=r["retrieval_metrics"]; consensus=r.get("synthesis_consensus",{}); vd=r.get("verifier_diagnostics",{}); tf=r.get("targeted_feedback") or {}
    per_question.append({"qid":r["qid"],"question":r["question"],"gold":json.dumps(r["gold_answers"],ensure_ascii=False),"prediction":r["final_answer"],
     "EM":r["exact_match"],"token_F1":r["token_f1"],"abstained":r["abstain"],"answered_wrong":r["answered_wrong"],
     "candidate_gold_R1":rm.get("recall@1"),"candidate_gold_R3":rm.get("recall@3"),"candidate_gold_R5":rm.get("recall@5"),"candidate_gold_R10":rm.get("recall@10"),
     "gold_support_ids":"|".join(r["gold_support_ids"]),"candidate_gold_ids_found":"|".join(r["candidate_gold_ids_found"]),
     "selected_gold_ids_found":"|".join(r["selected_gold_ids_found"]),"lost_gold_ids":"|".join(r["lost_gold_ids"]),
     "candidate_gold_support_count":len(r["candidate_gold_ids_found"]),"selected_gold_support_count":len(r["selected_gold_ids_found"]),"gold_context_retention":r["gold_context_retention"],
     "planner_parse":bool(r.get("planner",{}).get("planner_parse_success")),"stage1_usable_rate":mean_valid([modality_health([r],"stage1",m)["rate"] for m in ["text","table","image"]]),
     "stage2_parse_rate":mean_valid([modality_health([r],"stage2",m)["rate"] for m in ["text","table","image"]]),
     "stage2_answers":json.dumps(consensus.get("stage2_answers",[]),ensure_ascii=False),"stage2_consensus_answer":consensus.get("consensus_answer"),"stage2_consensus_count":consensus.get("consensus_count"),
     "initial_verifier_status":vd.get("initial_verifier_status"),"initial_verifier_best_answer":vd.get("initial_best_answer"),"feedback_used":bool(tf),
     "retry_modality":vd.get("retry_modality"),"retry_query":vd.get("retry_query"),"feedback_candidate_ids":json.dumps(tf.get("new_candidate_ids",{})),
     "feedback_new_evidence_count":tf.get("new_unique_evidence_count",0),"feedback_context_ids":json.dumps(tf.get("new_context_ids",{})),
     "second_verifier_status":vd.get("second_verifier_status"),"second_verifier_best_answer":vd.get("second_best_answer"),
     "aggregator_parse":bool(aggregator_envelope(r).get("parse_success")),"abstention_reason":r.get("abstention_reason"),"gold_present_in_stage2":r["gold_present_in_stage2"],
     "failure_type":r["failure_type"],"total_seconds":r["timing"]["total_seconds"],"qwen_calls":r["qwen_call_count"],"repair_calls":r["format_repair_call_count"]})

failure_rows=[{"qid":r["qid"],"prediction":r["final_answer"],"abstained":r["abstain"],"answered_wrong":r["answered_wrong"],
               "abstention_reason":r.get("abstention_reason"),"failure_type":r["failure_type"],"gold_present_in_stage2":r["gold_present_in_stage2"]} for r in FULL_ROWS]
verifier_summary={"initial":initial_summary,"after_feedback":second_summary,"initial_total":len(initial_envelopes),"second_total":len(second_envelopes)}
failure_counts=dict(Counter(r["failure_type"] for r in FULL_ROWS)); abstention_reasons={k:sum(r.get("abstention_reason")==k for r in abstained) for k in ["no_candidate_evidence","context_missing_support","no_concrete_synthesis","verifier_insufficient_after_feedback","unresolved_conflict","aggregator_uncertain","generation_failure"]}

pd.DataFrame([summary_row]).to_csv(DIRS["metrics"]/"summary.csv",index=False)
pd.DataFrame([retrieval_row]).to_csv(DIRS["metrics"]/"retrieval_summary.csv",index=False)
pd.DataFrame(per_question).to_csv(DIRS["metrics"]/"per_question_FULL.csv",index=False)
pd.DataFrame(failure_rows).to_csv(DIRS["metrics"]/"failure_analysis.csv",index=False)
(DIRS["metrics"]/"pipeline_health.json").write_text(json.dumps(health,indent=2,allow_nan=True),encoding="utf-8")
(DIRS["metrics"]/"verifier_summary.json").write_text(json.dumps(verifier_summary,indent=2,allow_nan=True),encoding="utf-8")
(DIRS["metrics"]/"feedback_summary.json").write_text(json.dumps(feedback_summary,indent=2,allow_nan=True),encoding="utf-8")
(DIRS["metrics"]/"context_retention_summary.json").write_text(json.dumps(context_summary,indent=2,allow_nan=True),encoding="utf-8")

VALIDATION16_V2_COMPLETE=len(FULL_ROWS)==16 and [r["qid"] for r in FULL_ROWS]==VALIDATION16_V2
required_files=["summary.csv","retrieval_summary.csv","per_question_FULL.csv","failure_analysis.csv","pipeline_health.json","verifier_summary.json","feedback_summary.json","context_retention_summary.json"]
VALIDATION16_V2_EVALUATED=VALIDATION16_V2_COMPLETE and all((DIRS["metrics"]/x).exists() for x in required_files)
assert not (set(DEBUG16_V1)&set(VALIDATION16_V2))

def pct(x):return "N/A" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{100*x:.2f}%"
print("\n============================================================\nVALIDATION16_V2 RESULTS\n============================================================")
print("Dataset:\nMMQA dev\n\nQuestions:\n16\n\nDisjoint from DEBUG16_V1:\nYES\n\nSeed:\n",VALIDATION_SELECTION_SEED)
print("\n------------------------------------------------------------\nQA\n------------------------------------------------------------")
print("Exact Match:",pct(summary_row["EM"])); print("Token F1:",pct(summary_row["mean_token_F1"])); print("Answered:",len(answered),"/ 16 ("+pct(summary_row["answer_rate"])+")")
print("Abstained:",len(abstained),"/ 16 ("+pct(summary_row["abstention_rate"])+")"); print("Answered correct:",answered_correct); print("Answered wrong:",answered_wrong); print("Answered accuracy:",pct(summary_row["answered_accuracy"]))
print("\n------------------------------------------------------------\nRETRIEVAL\n------------------------------------------------------------")
for key in ["R@1","R@3","R@5","R@10","MRR"]:print(key+":",pct(retrieval_row[key]))
print("Candidate support coverage:",pct(context_summary["candidate_gold_support_coverage"])); print("Selected-context support coverage:",pct(context_summary["selected_context_gold_support_coverage"])); print("Mean gold-context retention:",pct(context_summary["mean_gold_context_retention"]))
print("\n------------------------------------------------------------\nVERIFIER\n------------------------------------------------------------\nInitial:")
for key in ["supported","insufficient","conflict","verifier_error"]:print(key,"=",initial_summary[key]["count"],"("+pct(initial_summary[key]["percentage"])+")")
print("After feedback:")
for key in ["supported","insufficient","conflict","verifier_error"]:print(key,"=",second_summary[key]["count"],"("+pct(second_summary[key]["percentage"])+")")
print("\n------------------------------------------------------------\nFEEDBACK\n------------------------------------------------------------")
print("Triggered:",feedback_summary["feedback_trigger_count"]); print("Activation rate:",pct(feedback_summary["feedback_activation_rate"])); print("New evidence found:",feedback_summary["new_evidence_found_count"])
print("Resolved to supported:",feedback_summary["feedback_success_count"]); print("Feedback success rate:",pct(feedback_summary["feedback_success_rate"]))
print("\n------------------------------------------------------------\nABSTENTION\n------------------------------------------------------------")
print("Total:",len(abstained)); print("Reasons:")
for key,value in abstention_reasons.items():print(key,"=",value)
print("Gold answer present in Stage-II despite abstention:",sum(r["gold_present_in_stage2"] for r in abstained))
print("\nFailure taxonomy:",failure_counts)
print("\n------------------------------------------------------------\nPIPELINE HEALTH\n------------------------------------------------------------")
print("Planner:",health["planner"]); print("Stage-I usable:",health["stage1"]); print("Stage-II:",health["stage2"]); print("Verifier:",health["initial_verifier"],"second:",health["second_verifier"]); print("Aggregator:",health["aggregator"])
print("OOM:",oom_count); print("Protected truncations:",protected_truncations); print("Stage-II token-cap hits:",stage2_cap_hits); print("Average Qwen calls/question:",round(health["mean_qwen_calls_per_question"],3)); print("Average runtime/question:",round(health["mean_runtime_seconds_per_question"],3)); print("Repair calls:",health["repair_count"])
print("\nOLD DEVELOPMENT RUN reference only: metrics came from a different 16-question sample; differences are descriptive, not an apples-to-apples improvement claim.")
print("Old candidate/context coverage: 76.04% / 23.96%; new:",pct(context_summary["candidate_gold_support_coverage"]),"/",pct(context_summary["selected_context_gold_support_coverage"]))
print("\nVALIDATION16_V2_COMPLETE =",VALIDATION16_V2_COMPLETE); print("VALIDATION16_V2_EVALUATED =",VALIDATION16_V2_EVALUATED)
print("============================================================")
''')

nb["metadata"]["validation16_v2"]={"phase":"VALIDATION16_V2","selection_seed":20260924,"question_count":16,
    "source_notebook":SOURCE.name,"pipeline_semantics_changed":False,"gold_access":"post-freeze only"}
for cell in nb["cells"]:
    if cell.get("cell_type")=="code":cell["execution_count"]=None; cell["outputs"]=[]
TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print(TARGET)
