# Cell 7 — worker script (one process per GPU); written to disk, launched by Cell 8
"""MAMMQA hop-trace worker: runs the released 7-agent MAMMQA pipeline on its shard of questions,
one full model copy per GPU. Launched by the notebook as:
    CUDA_VISIBLE_DEVICES=<g> python mammqa_trace_worker.py <output_root> <rank> <world>
Reads config/run_config.json and inputs/items.json (gold-free). Writes checkpoints/<qid>.json."""
import os, sys, json, re, time, math, copy, hashlib, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OUT = Path(sys.argv[1]); RANK = int(sys.argv[2]); WORLD = int(sys.argv[3])
CFG = json.loads((OUT / "config" / "run_config.json").read_text(encoding="utf-8"))
ITEMS = json.loads((OUT / "inputs" / "items.json").read_text(encoding="utf-8"))
MOCK = CFG["MOCK"]
CACHE = OUT / "cache"; CKPT = OUT / "checkpoints"; LOGS = OUT / "logs"
for d in (CACHE, CKPT, LOGS):
    d.mkdir(parents=True, exist_ok=True)
LOGF = LOGS / f"worker{RANK}.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [w{RANK}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{RANK}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- official MAMMQA prompts (prompts.py, verbatim)
Agent_stage1_system_prompt = """
You are an expert agent specialized in analyzing single-modality inputs (such as text, table, or image) and answering questions by extracting insights and systematically breaking down complex questions into simpler subquestions.

### **Task**:
You will be provided with an input and a related question. Your job is to:

Step 1:  
- Identify the modality type of the provided input.  
  Possible types: text(s), table(s), or image(s).

Step 2:  
- Clearly understand the question and carefully analyze the input.  
- Extract insights relevant to the question, example:
  - Key information (numbers, statistics, entities, trends).
  - Temporal insights (time, date, durations, timelines, etc.).

Examples for Step 2:  
- Text example insight:  
  "The text mentions that sales increased by 20 percent from January to March, highlighting quarterly growth."  
- Table example insight:  
  "The table shows the peak attendance (350 people) occurred on Saturday, June 12, 2023, indicating highest weekend engagement."  
- Image example insight:  
  "The image clearly indicates a street sign labeled '5th Avenue' and a clock showing the time as 2:15 PM, suggesting the photo was taken in mid-afternoon."

Step 3:  
- Based on these extracted insights, carefully break down the main question into simpler and more direct subquestions or counter-questions.

Examples for Step 3:  
- Main Question: "What was the monthly growth rate during Q1?"  
  - Subquestions:  
    - "What were the sales figures for January, February, and March individually?"  
    - "By how much did the sales figures change each month?"  

- Main Question: "When was attendance lowest and highest during the event period?"  
  - Subquestions:  
    - "Which date had the lowest attendance according to the provided table?"  
    - "Which date had the highest attendance according to the provided table?"

- Main Question: "At what time was the image captured?"  
  - Subquestion:  
    - "What specific time details are visible in the image?"

    
## **Important Additional Guidelines & Formatting**:

- Always think step-by-step through your analysis.
- Clearly output the identified modality type in:
  <modality> identified modality type here </modality>

- Clearly output your extracted insights in:
  <insights> your extracted insights here </insights>

- If possible, provide the final answer to original question within:
  <answer> your final answer here </answer>

- Provide answers to subquestions, wrap these in:
  <subanswer> your answer to subquestion here </subanswer>

- Only use the provided data.  
  Do not include any external or internal knowledge beyond what's explicitly given.

"""



Agent_stage2_system_prompt = """
You are an expert cross agent specialized in analyzing multiple-modalities (such as text, table, or image), insights from specialised agent(s) (such as text, table, or image) and answering questions by extracting insights and systematically breaking down complex questions into simpler subquestions.

### **Task**:
You will be provided with multiple inputs ( insights from a specialised agent(s) and multimodal input(s) ) and a related question. Your job is to:

Step 1:  
- Clearly understand the question and carefully analyze the input.  
- Extract insights relevant to the question, example:
  - Key information (numbers, statistics, entities, trends).
  - Temporal insights (time, date, durations, timelines, etc.).

Step 2:  
- Based on these extracted insights and agent insights, carefully break down the main question into simpler and more direct subquestions or counter-questions.

    
## **Important Additional Guidelines & Formatting**:

- Always think step-by-step through your analysis.

- Clearly output your extracted insights in:
  <insights> your extracted insights here </insights>

- Provide answers to subquestions, wrap these in:
  <subanswer> your answer to subquestion here </subanswer>

- Provide the final answer to original question within:
  <answer> your final answer here </answer>


- Only use the provided data.  
  Do not include any external or internal knowledge beyond what's explicitly given.

"""




Agent_stage3_system_prompt ="""
You are the final aggregator agent. Your input consists of three responses generated by cross–modal synthesis agents. Each response results from combining one modality's reasoning with the evidence from the other two modalities for the given question. Your task is to generate the most accurate final answer by following these rules:

(A) Consistency Check:

If at least two responses provide the same answer along with clear, robust reasoning, select that answer as final.

(B) Fallback Rule:

If two responses indicate that the available information is insufficient but one response gives a concrete answer with detailed evidence, choose the concrete answer.

(C) Conflict Resolution:

If all three responses differ, examine the quality of their reasoning. Weigh the clarity, depth, and coherence of the explanations, and select the answer with the strongest supporting rationale.

(D) Final Synthesis:

Provide your final answer along with a brief explanation summarizing the key points that influenced your decision.

Ensure that your decision-making is transparent, logically consistent."""

FP = {k: sha256_text(v)[:16] for k, v in {"stage1": Agent_stage1_system_prompt, "stage2": Agent_stage2_system_prompt,
                                          "stage3": Agent_stage3_system_prompt}.items()}
assert FP == CFG["OFFICIAL_PROMPT_FINGERPRINTS"], f"prompts not verbatim: {FP}"

FORMAT_ADDENDUM = (
    "\n\nFinish your response with the final answer on its own line in the form <answer>...</answer>. "
    "Inside the tags write only the short answer span (an entity, number, date, or yes/no); "
    "if the question has several answers, separate them with ' | '. No explanation inside the tags."
)

# ---------------------------------------------------------------- images (Qwen2.5-VL smart_resize)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
PATCH = 28
MIN_T, MAX_T, BUDGET = CFG["MIN_IMAGE_TOKENS"], CFG["MAX_IMAGE_TOKENS"], CFG["VISUAL_TOKEN_BUDGET"]


def smart_resize(h, w, factor, min_pixels, max_pixels):
    if max(h, w) / max(1, min(h, w)) > 200:
        raise ValueError("aspect ratio > 200")
    hb = max(factor, round(h / factor) * factor); wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor); wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor; wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def per_image_tokens(n):
    return 0 if n <= 0 else int(max(MIN_T, min(MAX_T, BUDGET // n)))


def load_resized(path, max_tokens):
    img = Image.open(path)
    img = img.convert("RGBA").convert("RGB") if img.mode in ("P", "LA", "RGBA") else img.convert("RGB")
    w, h = img.size
    try:
        rh, rw = smart_resize(h, w, PATCH, MIN_T * PATCH ** 2, max_tokens * PATCH ** 2)
    except ValueError:                       # extreme aspect ratio: pad to a square first
        side = max(w, h); canvas = Image.new("RGB", (side, side), (255, 255, 255)); canvas.paste(img, (0, 0)); img = canvas
        rh, rw = smart_resize(side, side, PATCH, MIN_T * PATCH ** 2, max_tokens * PATCH ** 2)
    return img.resize((rw, rh), Image.BICUBIC)


# ---------------------------------------------------------------- model
MODEL = PROCESSOR = None


def load_model():
    global MODEL, PROCESSOR
    if MOCK or MODEL is not None:
        return
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    src = CFG["MODEL_SOURCE"]
    log(f"loading {src} on cuda:0 (physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES')})")
    PROCESSOR = AutoProcessor.from_pretrained(src, min_pixels=MIN_T * PATCH ** 2, max_pixels=MAX_T * PATCH ** 2)
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        device_map={"": 0}, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    out = _generate([{"role": "user", "content": [{"type": "text", "text": "What is 2 + 3? Reply with the number only."}]}], [], 8)
    log(f"sanity check: {out['text']!r}  | GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    assert "5" in out["text"] or "five" in out["text"].lower(), "model sanity check failed"


def _template(messages):
    return PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _count(messages, images):
    if MOCK:
        return int(len(json.dumps(messages)) / 3.6) + sum((i.size[0] // PATCH) * (i.size[1] // PATCH) for i in images)
    return int(PROCESSOR(text=[_template(messages)], images=images or None, return_tensors="pt").input_ids.shape[1])


def _generate(messages, images, max_new):
    if MOCK:
        return mock_generate(messages, max_new)
    import torch
    batch = PROCESSOR(text=[_template(messages)], images=images or None, return_tensors="pt").to(MODEL.device)
    gc = copy.deepcopy(MODEL.generation_config)
    gc.max_new_tokens = max_new; gc.repetition_penalty = CFG["REPETITION_PENALTY"]
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, 1.0, 1.0, 50
    gc.pad_token_id = PROCESSOR.tokenizer.pad_token_id or PROCESSOR.tokenizer.eos_token_id
    with torch.inference_mode():
        out = MODEL.generate(**batch, generation_config=gc)
    new = out[:, batch.input_ids.shape[1]:]
    text = PROCESSOR.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    n_in = int(batch.input_ids.shape[1]); del batch, out
    return {"text": text, "input_tokens": n_in, "output_tokens": int(new.shape[1])}


def _shorten(s, frac):
    keep = max(200, int(len(s) * frac))
    if keep >= len(s):
        return s
    head = int(keep * 0.7)
    return s[:head] + "\n[... evidence shortened to fit the input budget ...]\n" + s[-(keep - head):]


def call_model(spec, stage):
    """spec = {system, user:[{type:text,text,evidence?}|{type:image,path}]}; content-addressed cache = resume."""
    n_img = sum(c["type"] == "image" for c in spec["user"])
    img_tok = per_image_tokens(n_img)
    key = sha256_text(json.dumps({"model": CFG["MODEL_ID"], "max_new": CFG["MAX_NEW_TOKENS"][stage], "max_in": CFG["MAX_INPUT_TOKENS"],
                                  "rep": CFG["REPETITION_PENALTY"], "img_tok": img_tok, "spec": spec}, sort_keys=True, ensure_ascii=False))
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))
    load_model()
    user = [dict(c) for c in spec["user"]]
    images = [load_resized(c["path"], img_tok) for c in user if c["type"] == "image"]
    shortened = 0
    for _ in range(12):
        messages = [{"role": "system", "content": spec["system"]},
                    {"role": "user", "content": [{"type": "image"} if c["type"] == "image" else {"type": "text", "text": c["text"]} for c in user]}]
        if _count(messages, images) + CFG["MAX_NEW_TOKENS"][stage] <= CFG["MAX_INPUT_TOKENS"]:
            break
        ev = [i for i, c in enumerate(user) if c["type"] == "text" and c.get("evidence")]
        if not ev:
            break
        longest = max(ev, key=lambda i: len(user[i]["text"]))
        user[longest]["text"] = _shorten(user[longest]["text"], 0.8); shortened += 1
    t0 = time.perf_counter(); oom = False
    try:
        out = _generate(messages, images, CFG["MAX_NEW_TOKENS"][stage])
    except Exception as e:
        if "out of memory" not in str(e).lower():
            raise
        import torch; torch.cuda.empty_cache(); oom = True
        images = [load_resized(c["path"], max(MIN_T, img_tok // 2)) for c in user if c["type"] == "image"]
        out = _generate(messages, images, CFG["MAX_NEW_TOKENS"][stage])
    rec = {"key": key, "text": out["text"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
           "hit_max_new_tokens": out["output_tokens"] >= CFG["MAX_NEW_TOKENS"][stage], "evidence_shortenings": shortened,
           "n_images": n_img, "tokens_per_image": img_tok, "oom_fallback": oom, "seconds": round(time.perf_counter() - t0, 2)}
    atomic_json(cpath, rec)
    return rec


# ---------------------------------------------------------------- the seven agents (agents.py message formats)
def T(text, evidence=False):
    return {"type": "text", "text": text, "evidence": evidence}


def images_of(item):
    return [{"type": "image", "path": p} for p in item["image_paths"]]


def text_agent(it):
    return {"system": Agent_stage1_system_prompt, "user": [T(f"Here is the text data:\n{it['text_block']}\n Here is the question: {it['question']}", True)]}


def table_agent(it):
    return {"system": Agent_stage1_system_prompt, "user": [T(f"Here is the markdown table data:\n{it['table_block']}\n Here is the question: {it['question']}", True)]}


def image_agent(it):
    return {"system": Agent_stage1_system_prompt, "user": [T(f"Here is the question: {it['question']}")] + images_of(it)}


def text_cross_agent(it, ins):
    return {"system": Agent_stage2_system_prompt, "user": [T(f"Here is the text insight:\n{ins}\n Here is the Question: {it['question']}\n Here is the markdown table:\n{it['table_block']}", True)] + images_of(it)}


def table_cross_agent(it, ins):
    return {"system": Agent_stage2_system_prompt, "user": [T(f"Here is the table insights:\n{ins} \n\n Here is the Question: {it['question']}\n Here is the text data:\n{it['text_block']}", True)] + images_of(it)}


def image_cross_agent(it, ins):
    return {"system": Agent_stage2_system_prompt, "user": [T(f"Here is the image insights: \n {ins} \n Here is the Question: {it['question']}\n Here is the text data:\n{it['text_block']}\n Here is the markdown table: \n{it['table_block']}\n", True)]}


def reasoning_agent(it, t, tb, im):
    return {"system": Agent_stage3_system_prompt, "user": [T(f"Here is the Question: {it['question']}\n Here is the text cross output:\n{t}\n Here is the table cross output:\n{tb}\n Here is the image cross output:\n{im}" + FORMAT_ADDENDUM)]}


ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def final_span(raw):
    tags = [t.strip() for t in ANSWER_TAG.findall(raw or "") if t.strip()]
    if tags:
        return tags[-1], "answer_tag"
    m = re.search(r"(?is)(?:final answer|answer)\s*[:\-]\s*(.+?)(?:\n|$)", raw or "")
    if m:
        return m.group(1).strip().strip("*. "), "answer_colon"
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    return (lines[-1] if lines else ""), "last_line"


def mock_generate(messages, max_new):
    blob = json.dumps(messages, ensure_ascii=False)
    h = int(sha256_text(blob)[:6], 16)
    time.sleep(CFG.get("MOCK_SLEEP", 0.0))
    words = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", blob)
    ans = words[h % len(words)] if words else "insufficient"
    if "final aggregator" in str(messages[0]["content"]):
        cands = re.findall(r"<answer>(.*?)</answer>", blob)
        ans = cands[h % len(cands)] if cands else "unknown"
        return {"text": f"Reasoning.\n<answer>{ans}</answer>", "input_tokens": int(len(blob) / 3.6), "output_tokens": 20}
    return {"text": f"<insights> mock insight mentioning {ans} </insights>\n<answer>{ans}</answer>",
            "input_tokens": int(len(blob) / 3.6), "output_tokens": min(max_new, 40 + h % 30)}


# ---------------------------------------------------------------- run
def run_question(it):
    t0 = time.time()
    s1 = {"text": call_model(text_agent(it), "stage1"), "table": call_model(table_agent(it), "stage1"),
          "image": call_model(image_agent(it), "stage1")}
    s2 = {"text": call_model(text_cross_agent(it, s1["text"]["text"]), "stage2"),
          "table": call_model(table_cross_agent(it, s1["table"]["text"]), "stage2"),
          "image": call_model(image_cross_agent(it, s1["image"]["text"]), "stage2")}
    s3 = call_model(reasoning_agent(it, s2["text"]["text"], s2["table"]["text"], s2["image"]["text"]), "stage3")
    span, how = final_span(s3["text"])
    calls = {f"s1_{k}": v for k, v in s1.items()} | {f"s2_{k}": v for k, v in s2.items()} | {"s3": s3}
    return {
        "id": it["qid"], "question": it["question"], "prediction": span, "parse_method": how,
        "stage1": {"text": {"status": "available", "output": s1["text"]["text"]},
                   "table": {"status": "available" if it["has_table"] else "unavailable", "output": s1["table"]["text"]},
                   "image": {"status": "available" if it["image_paths"] else "unavailable", "output": s1["image"]["text"]}},
        "stage2": {k: v["text"] for k, v in s2.items()},
        "final_raw": s3["text"],
        "calls": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in calls.items()},
        "n_images": len(it["image_paths"]), "missing_images": it["missing_images"],
        "worker": RANK, "seconds": round(time.time() - t0, 1),
    }


def main():
    order = CFG["ORDER"]
    mine = order[RANK::WORLD]
    deadline = CFG["DEADLINE_EPOCH"]
    log(f"{len(mine)} questions assigned; deadline in {(deadline - time.time()) / 3600:.2f} h")
    done = 0
    for qid in mine:
        ck = CKPT / f"{qid}.json"
        if ck.exists():
            continue
        if time.time() > deadline:
            log("deadline reached; not starting new questions"); break
        try:
            rec = run_question(ITEMS[qid])
            atomic_json(ck, rec); done += 1
            log(f"done {qid} in {rec['seconds']}s -> {rec['prediction'][:60]!r}")
        except Exception as e:
            atomic_json(OUT / "errors" / f"{qid}.json", {"qid": qid, "error": repr(e), "trace": traceback.format_exc()})
            log(f"ERROR {qid}: {e!r}")
            if not MOCK:
                import torch; torch.cuda.empty_cache()
    log(f"finished: {done} new questions")


if __name__ == "__main__":
    main()
