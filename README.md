# Guide: Qwen3-Embedding-0.6B Q8 on llama.cpp, on a free DGX Spark GPU — last-token pooling, a second index, and a cosine that is not a verdict

**The problem this guide solves**: three boot profiles already exist on one DGX Spark — **idle** (GPU free), **llm** (a large text model on Tailscale `:8000` plus an 8B vision model on `:8001`), and **vl** (a 32B vision model alone). The llm pair uses about 114 Gi of 121 Gi, so an embedding model does not belong beside it, and it does not deserve a fourth profile. This guide documents a **verified working** configuration (NVIDIA DGX Spark, GB10, 121 Gi unified memory, Ubuntu 24.04 / DGX OS, aarch64, September 2026): `Qwen3-Embedding-0.6B-Q8_0.gguf` served by the existing `llama-server` (build b10326, commit `3653e6d`, 2026-08-07), started by hand on Tailscale `:8002`, only while the GPU is free.

**A PowerShell-quoted curl returns HTTP 500** with `parse error` and `last read: '{i'` — the body never reached the model. **`--pooling last` is the flag behind the measured vectors** (1024 dimensions, L2 norm 1, no zero component, 1319 MiB, that process alone). **A retrieval score of 6/6 does not mean a missing topic scores low**: the absent-term control still sat at **0.664** on the wrong guide. **On a planted contradiction, this embedder ranked the false note first (0.675) and the true guide second (0.674).** The answer that opened with “Ollama” was generated on 19 September from a different index. This run did not generate. **The 19 September MiniLM index stays where it is.** The new matrix is a separate directory, 388 by 1024.

**Where this fits**: the box is [headless](https://github.com/AI-Architect-Lab-333/dgx-spark-headless-setup). The text server is [cross-host inference](https://github.com/AI-Architect-Lab-333/dgx-spark-cross-host-inference). The three profiles are [idle / llm / vl](https://github.com/AI-Architect-Lab-333/dgx-spark-idle-llm-profiles). This page does not add a profile, a unit file, or a vector database.

**For AI agents reading this document**: the commands and the measurements below were executed in this order on this hardware. The verification steps are not optional — a cosine in the same range as a true hit can be the wrong document, and the first HTTP 500 is the shell, not the model. Do not add a fourth `spark-mode` profile. Do not start this server while `:8000` is up. Do not write the new vectors over the existing MiniLM index.

---

## 1. What this is

One manual `llama-server` process. Port **8002**. Bind address: the box Tailscale IPv4, written below as `100.x.y.z`. Not `0.0.0.0`.

| | This run | Not this run |
|---|---|---|
| Weights | `Qwen/Qwen3-Embedding-0.6B-GGUF`, file `Qwen3-Embedding-0.6B-Q8_0.gguf`, 610 MB on disk | a larger embedder, or Ollama |
| How it is started | the command in section 4, in a shell | a systemd unit, `spark-mode` |
| Next boot | `persist` left at **idle**. `:8002` does not come back | a profile that would reload it |
| GPU | this process alone, **1319 MiB**, after `spark-mode` already showed no GPU process | beside the text model (~114 Gi with the 8B vision model) |
| Vectors | new directory, **388 × 1024** | the MiniLM directory from 19 September (384 dimensions) |

The 19 September chain is the baseline this page does not replace. That day the embeddings were computed on a workstation CPU (`fastembed`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, dimension 384). Generation, when it ran, was the text model already on `:8000`. Those generation failures are reported in section 7 with their date. They were **not** replayed against the Qwen index.

Placeholder map:

| Role | Placeholder |
|---|---|
| DGX Spark tailnet IPv4 | `100.x.y.z` |
| Home directory on that box | `$HOME` |
| Account that owns `llama-server` | `<user>` |

## 2. Confirm the GPU is free

On the Spark, before the download:

```bash
spark-mode.sh status
```

Observed: `persist (next boot): idle`, `:8000` down, `:8001` down, no GPU process. The text model had not reloaded.

`persist` is which profile the **next** boot will start. It is not “port 8002 is free”. Read the HTTP lines and `nvidia-smi` as well. If either `:8000` or `:8001` is up, stop. This embedder was not measured in the leftover memory of the llm pair. On 19 September the 8B vision unit beside the text model already failed with `CUDA error: out of memory` and `Failed with result 'core-dump'` (`spark-mode` exit 3, `:8000` still serving, about 100.5 Gi). Text-only work continued that day. An embedder was not added on top.

`spark-mode idle` stops the three profile units only. It does not own a `llama-server` you started by hand, and its GPU-free wait looks for **any** `llama-server` pid. Do not expect that command to clean up `:8002`.

## 3. Download the GGUF

```bash
hf download Qwen/Qwen3-Embedding-0.6B-GGUF --include Qwen3-Embedding-0.6B-Q8_0.gguf
```

Observed file:

```text
$HOME/models/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
```

610 MB. If your `hf` writes somewhere else, keep that path for `--model` in the next section. Do not substitute another file from the same repo.

## 4. Start the server by hand

Same binary as the text server: `llama-server`, build **b10326**, commit **`3653e6d`** (2026-08-07), aarch64. No unit file. `--model` is the spelling already used by that binary on this box; the other flags are the ones this process was started with.

```bash
llama-server \
  --model "$HOME/models/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --host 100.x.y.z \
  --port 8002 \
  --embeddings \
  --pooling last \
  --embd-normalize 2 \
  -ngl 99 \
  --ctx-size 2048 \
  --alias qwen3-embedding-0.6b
```

Replace `100.x.y.z` with the box Tailscale IPv4. Leave `persist` on **idle**. Nothing in this command enables a target, so the next boot does not restart it. While it runs, the GPU is not free for anything else.

Two log lines appeared at load. Neither stopped the server. Do not “fix” them by changing flags that were not part of this measurement.

```text
embeddings enabled with n_batch (2048) > n_ubatch (512)
setting n_batch = n_ubatch = 512
```

```text
control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
```

The first is the binary clamping its default batch to its micro-batch. The second is GGUF metadata. The trial vector afterwards was not the zero vector. Neither line was patched.

The process also warned that CORS is `*` and that there is no API key — the same warning as the other `llama-server` processes on this box. The bind stays the Tailscale address.

### Pitfall #1 — HTTP 500 `parse error`, `last read: '{i'`

Symptom: the first `curl` against `/v1/embeddings` returned HTTP 500, `parse error`, last read `'{i'`. Cause: PowerShell ate the quotes. The model never saw a JSON object. Correction: send the body as JSON with no shell escaping. The request that proved the server was a POST whose body is `{"input": "<text>"}` and whose `Content-Type` is `application/json`. Proof: a vector of length **1024**, L2 norm **1**, no component equal to 0, and `nvidia-smi` showing this `llama-server` alone at **1319 MiB**.

```python
import json
import math
import urllib.request

url = "http://100.x.y.z:8002/v1/embeddings"
payload = {"input": "<text>"}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read().decode("utf-8"))
vec = body["data"][0]["embedding"]
norm = math.sqrt(sum(x * x for x in vec))
zeros = sum(1 for x in vec if x == 0.0)
print(len(vec), norm, zeros)
```

On this box that shape returned `1024`, norm `1`, `zeros` `0`. Section 8 uses a synthetic sentence in place of `<text>` so this page does not carry a private control. The check is the shape of the vector, not the words in `<text>`.

### Pitfall #2 — `n_batch (2048) > n_ubatch (512)`

Symptom: the two lines in section 4, then the server still answers. Cause: a default in this binary, not a bad GGUF path. Correction: none. It was left clamped at 512. Proof: the same vector as pitfall #1.

### Pitfall #3 — `control-looking token: 128247 '</s>'`

Symptom: the override line in section 4. Cause: the GGUF marks that token in a way this build does not treat as a control token. Correction: none applied. Proof: the trial vector was not all zeros. Do not read the word “bug” in that line as “the embedding endpoint is down”.

### Pitfall #4 — CORS `*` and no API key

Symptom: the usual `llama-server` warning. Cause: this binary’s HTTP defaults. Correction used here: bind `100.x.y.z`, never `0.0.0.0`. The tailnet is the access control. This page does not add a key.

## 5. A second index, not a replacement

Passages were embedded **raw**. Only the query received an instruction prefix. The prefix stored with the new index, and the string the query client sent, is:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:<question>
```

There is no space between `Query:` and the question. A passage does not get that prefix.

Chunks were 1400 characters with 150 characters of overlap, split on markdown headings of level 1–3. Each piece was sent as its own POST to `{embed-url}/embeddings` with `{"input": "<chunk>"}`, timeout 120 seconds. The server context is 2048, so a batch of chunks was not used.

The index command that ran, from the lab directory that holds the corpus (that directory is not this repo):

```bash
python rag.py index --embed-url http://100.x.y.z:8002/v1 --index-dir index-qwen3
```

Observed: **20** files, **388** chunks, matrix **`(388, 1024)`**. The MiniLM directory (`index/`, dimension 384, written 2026-09-19) was not replaced. That directory is 388 chunks as well, because the false note of section 7 was already in the corpus when it was last built; the first measurement that morning, before the false note, was 387 chunks of the 19 public guides.

`--index-dir` is required once `--embed-url` is set, and it must not be the MiniLM directory. This repo does not ship `rag.py`. It is wired to a private control file, and that file stays private. What a reader of this page can replay is sections 2–4 and the vector check. The rows in section 6 are the measurement, not a prompt list to copy.

## 6. What retrieval measured on 23 September 2026

Six controls from 19 September, retrieved against `index-qwen3`. Verdict **6/6**. “6/6” means the checks passed. It does not mean six true documents at rank 1. Top of each control:

| Control | Top hit | Score |
|---|---|---|
| 1 | `idle-llm-profiles` | 0.613 |
| 2 | `mujoco-headless-panda` | 0.612 |
| 3 | `bench-saturation-refusal` | 0.618 |
| 4 | `cross-host-inference` | 0.554 |
| 5 | `vl-beside-llm` | 0.670 |
| 6, absent term `k3s` | `idle-llm-profiles` | **0.664** |

Control 6 passed because the string `k3s` was **absent from the top 5 chunks**, not because the score was small. **0.664** sits in the same band as the true hits (0.554–0.670). A cosine does not report absence.

### Pitfall #5 — a high score on the wrong guide

Symptom: top score 0.664, wrong guide, and the term you asked about is not in those chunks. On 19 September the same control, MiniLM index, was already 0.609 on that same wrong guide, and the text model on `:8000` then said the term was not in the context. Cause: nearest-neighbour search always returns something. Correction on the retriever: none. The refusal has to happen when the window is read, and only a generation pass can show that. This Qwen run did not generate. Proof that the score is not a detector: the number above, next to the five true hits.

The contradiction control was retrieval only, **k = 10**. Both sources were inside the window, so the check “both documents are present” passed. Rank did not put the truth first.

| Rank | Document | Score |
|---|---|---|
| 1 | `trap-ollama-serve.md` (local false note: Ollama replaced llama.cpp on `:8000`) | **0.675** |
| 2 | `cross-host-inference` (the true guide) | **0.674** |

The false note is one local file. It is not a guide on the public account, and it is not included here. Generation was not started.

### Pitfall #6 — the false note ranks first, by 0.001

Symptom: 0.675 then 0.674. On 19 September, MiniLM, the same window at k = 10 had the true guide’s chunks at 0.700, 0.638, and 0.595, and this false note **8th at 0.527**. At k = 5 that day the false note did not enter the window. Cause of the 23 September order: a different embedder. The gap is one thousandth. Correction: do not treat rank 1 as the true document, and do not describe this run as the one that answered “Ollama”. That sentence was produced on 19 September. Proof: the two scores, and the fact that no completion was requested after this retrieval.

## 7. What generation already did on 19 September (MiniLM index, `:8000`)

These are limits of the chain. They are not Qwen measurements. The text model was the one already served on `:8000`.

Retrieval alone that morning was **6/6** on 19 public guides (387 chunks, dimension 384). The five factual controls had the right guide first, scores from **0.545 to 0.703**. The absent-term control was pitfall #5 at **0.609**.

Generation with `max_tokens=400` was **4/6**. One empty completion was the token budget, not an empty window. Raised to `max_tokens=2000`, the same chunks finished in 27 seconds with `finish=stop` and the right field names present. The official pass at 2000 was **5/6**: the model still **invented** `refusal_rate` and `refusal_rate_uncertainty` instead of `silent` and `silence_rate`.

### Pitfall #7 — `max_tokens=400` comes back empty

Symptom: an empty completion while the retrieved chunks contained the answer. Cause: the budget was spent before the answer. Correction that was measured: `max_tokens=2000` on the same chunks. Proof: 27 seconds, `finish=stop`, and the field names from the guide present on the replay. The official 2000-token pass still failed the name check (pitfall #8).

### Pitfall #8 — the right chunks, invented names

Symptom: `refusal_rate` / `refusal_rate_uncertainty` in the completion; the guide says `silent` / `silence_rate`. Same chunks as the replay that had used the real names. Cause: retrieval does not bind the generator to the nouns in the window. Correction: none on the retriever. The check has to require the names that are actually in the source. Proof: one pass invented them, the earlier replay of those chunks had not.

### Pitfall #9 — an English stem rejected a French answer

Symptom: a check that required `teleport` failed a completion that contained `téléportée` (and `GRASP_FAIL`). Cause: the check, not the model. Correction: accept the stem that was actually written. This is an instrument failure. It is why a 4/6 at `max_tokens=400` was not four model failures.

### Pitfall #10 — a false chunk below three true ones still wins the answer

Symptom, verbatim from the 19 September completion:

```text
Ollama. Le guide trap-ollama-serve.md … llama.cpp a été essayé puis abandonné
```

The answer opens with “Ollama” and says llama.cpp was tried and then dropped. Cause: at k = 10 the false note was 8th (0.527), under three true chunks, and the model did not say the sources conflict. A check that only required the string `llama.cpp` to appear scored a false pass. Correction: no change on the retriever; the check must reject an opening of “Ollama” and must require the conflict to be named. Verdict of that control: **fooled**.

A later 7-control generation the same day fell to **2/7** (refusals and “not in the context” on chunks that were in the window). That run does not replace the morning **5/6**. The likely cause recorded at the time is a long window on an already hot model, not a new stage of the system.

Section 6 is what changed on 23 September: the **rank** of the same false note, under the Qwen index, with no generation. The 19 September completion stays the 19 September completion.

## 8. Synthetic example

The six controls and the false note are not published here. The pair below is **not** part of that set. It was not embedded on the Spark. It shows the two strings the client treats differently, and nothing else. Do not attach a score to it. The measured “false note beats the true guide by 0.001” result is the table in section 6, not this pair.

Passage, embedded raw:

```text
The north workshop soldering iron stays in drawer B. It is signed out on the paper card. It is not loaned overnight.
```

A second passage, also raw, which contradicts the first:

```text
The north workshop soldering iron was discarded in March. Staff now use a glue gun from drawer C, including overnight.
```

Query, with the prefix from section 5 and no space after `Query:`:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:Where is the north workshop soldering iron kept?
```

A query about something neither passage mentions (`Does the north workshop own a kiln?`) is the public shape of pitfall #5. The nearest of these two notes can still score like a hit. That score does not mean a kiln is documented, and it does not mean the score noticed the kiln was missing.

Embed each passage with `{"input": "<passage>"}`. Embed the query with `{"input": "<prefixed query>"}`. Compare cosines only after both sides used those rules. Rank 1 is not truth. A cosine near the other cosine is not a tie you can ignore: section 6’s true guide lost by 0.001.

## 9. End-to-end verification

Proven on this box: idle GPU, then this server, then one vector of the right shape, then a new 388×1024 index, then retrieval of the six controls at 6/6 with pitfall #5 still true, then the contradiction window at k = 10 with the false note first. Not proven on this index: a completion.

| Step | Expected | Failed |
|---|---|---|
| `spark-mode.sh status` before section 4 | `persist (next boot): idle`; `:8000` and `:8001` down; no GPU process | either port up, or a GPU process. Do not start section 4 |
| Section 4, no unit file | server stays up; `persist` still `idle`; no new profile | a fourth profile or a unit that would return on the next boot |
| POST `/v1/embeddings`, body `{"input": "<text>"}` | length **1024**, L2 norm **1**, **no** component equal to 0; `nvidia-smi` shows this `llama-server` alone at **1319 MiB** | HTTP 500 `parse error` / `'{i'` is pitfall #1. Another length, a zero vector, or a second model on the GPU means the flags or the profile are wrong |
| New index directory | `(388, 1024)`; MiniLM `index/` still dimension 384, still dated 2026-09-19 | the MiniLM directory was replaced |
| Six controls, Qwen index | checks **6/6**, and control 6 is still the wrong guide at **0.664** with `k3s` absent from the top 5 | reading 6/6 as “the score found the absence”, or as “rank 1 is always the true guide” |
| Contradiction, k = 10, retrieval only | both documents in the window; false note **0.675** then true guide **0.674**; no completion | calling that order the 19 September “Ollama” answer, or expecting the higher cosine to be the true document |

The six questions are not on this page. Replaying sections 2–4 and the vector row is the public check. The last two rows are what this corpus did on 23 September.

## Symptom / Cause / Fix

| Symptom | Cause | Fix |
|---|---|---|
| HTTP 500 `parse error`, last read `'{i'` | PowerShell ate the JSON quotes | send `{"input": "<text>"}` with no shell escaping |
| `n_batch (2048) > n_ubatch (512)`, then the server answers | binary default | leave the clamp; not a failed start |
| `control-looking token: 128247 '</s>'` overridden | GGUF metadata | none; the trial vector was not zeros |
| CORS `*` and no API key | `llama-server` defaults | bind the Tailscale IPv4 only |
| Absent term, top score 0.664 on the wrong guide, check still “pass” | a cosine does not report absence | do not treat the score as a detector; 6/6 did not remove this |
| False note 0.675, true guide 0.674 | this embedder’s order, k = 10 | rank 1 is not truth; do not attribute the “Ollama” completion to this run |
| Empty completion at `max_tokens=400` (19 Sep, MiniLM) | budget spent before the answer | 2000 tokens answered; that pass still invented names |
| `refusal_rate` instead of `silent` / `silence_rate` | the generator is not bound by the window | check the source’s names; retrieval cannot do it |
| Check wanted `teleport`, answer had `téléportée` | the check’s language | fix the check |
| Completion opens with “Ollama” (19 Sep, k = 10, false note 8th at 0.527) | one false chunk under three true ones was enough | name the conflict; do not pass on the mere string `llama.cpp` |
| `:8002` gone after reboot, `persist` still idle | the server was never a unit | expected. Start section 4 again. Do not add a profile to “fix” it |

## Known limitations

- **Not a fourth `spark-mode` profile and not a boot service.** `persist` was left at idle. The next boot does not listen on `:8002`. That is the measured choice, not an unfinished unit file. Stopping the server is stopping that process. `spark-mode idle` does not stop it for you.
- **Generation was not run on the Qwen index.** Pitfalls #7, #8, and #10, and the morning 5/6, are the 19 September MiniLM index plus the text model on `:8000`. The Qwen result on the contradiction is the rank in section 6 only.
- **6/6 did not retire pitfall #5 or pitfall #6.** The absent-term score stayed in the true-hit band. The false note moved from 8th (0.527, MiniLM) to 1st (0.675, Qwen). A drop or a flip is a result. It is not a reason, recorded here, to load a third embedder.
- **The private controls, the false note, and the indexer are not in this repo.** Section 8 is the public stand-in. It has no measured score.
- **No pgvector, no Qdrant, no index of private infrastructure notes, no index of a personal catalogue.** The corpus was 19 public guides plus one local false note.
- **Open CORS, no API key.** Safe here only because the socket is the tailnet address.
- **Context 2048, one chunk per request.** Chunks were 1400 characters. This run did not test a larger context or a batch.
- **`--pooling last` is the setting that was measured**, because that is what produced these vectors. This session did not publish an A/B against mean pooling.
- **One GB10 box.** The two embedders are two dated measurements, not a leaderboard.
- **Warnings left as printed:** the `n_batch` clamp and the `</s>` override. The server still returned a unit vector.

## Self-check

Eight questions. Each one is answered by a pitfall above. Write the answer down before you open [self-check-answers.md](self-check-answers.md).

1. **Pitfall #1.** The first request to `/v1/embeddings` returns HTTP 500, `parse error`, last read `'{i'`. Is the GGUF bad? What do you change?
2. **Pitfalls #2 and #3.** The log clamps `n_batch` from 2048 to 512, and it says control-looking token `128247` `'</s>'` will be overridden, calling that a bug in the model. Which of these, if either, is a reason to stop and pick another file?
3. **Pitfall #4.** The server warns that CORS is `*` and that there is no API key. What was the access control actually used, and does this process listen on `:8002` after the next boot?
4. **Pitfall #5.** Retrieval of the six controls is 6/6, and the absent term `k3s` scores 0.664 on `idle-llm-profiles`. Did the score notice that `k3s` is missing?
5. **Pitfall #6.** At k = 10 the false note scores 0.675 and the true guide 0.674. Is the top document the true one? Was this the run that answered “Ollama”?
6. **Pitfalls #7 and #8.** A completion is empty at `max_tokens=400`. A later pass at 2000 tokens writes `refusal_rate` instead of `silent`. Which index was that, and does a better retriever stop the invented name?
7. **Pitfall #9.** A check required `teleport`. The completion contained `téléportée`. Whose failure is that?
8. **Pitfall #10.** The false note is 8th, under three true chunks. A check that only requires the string `llama.cpp` passes. Why is that pass false, and which date does that completion belong to?

## Credits

The weights are [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), license Apache-2.0. This run served `Qwen3-Embedding-0.6B-Q8_0.gguf` and passed `--pooling last`, which is the pooling that repository’s llama.cpp notes require. No text from that page is copied here. The query prefix in section 5 is the string stored with the index this box built.

`llama-server` is build b10326, commit `3653e6d` (2026-08-07), the binary already used for the text server on this machine.

---

*Guide written and verified in September 2026 on an NVIDIA DGX Spark (GB10, 121 Gi unified memory, Ubuntu 24.04 / DGX OS, aarch64). Versions: llama-server b10326 (commit 3653e6d, 2026-08-07); Qwen3-Embedding-0.6B-Q8_0.gguf; Qwen retrieval 2026-09-23; MiniLM generation limits 2026-09-19 (`paraphrase-multilingual-MiniLM-L12-v2`).*
