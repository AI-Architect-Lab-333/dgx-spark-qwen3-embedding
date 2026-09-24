# Self-check answers

Answers for the eight questions in the guide. The questions are the test. This file is the key.

1. **Pitfall #1.** The GGUF is fine and the server is up. PowerShell ate the quotes, so the body that arrived was not a JSON object. Send `{"input": "<text>"}` with `Content-Type: application/json` and no shell escaping. The proof on this box was a vector of length 1024, L2 norm 1, no zero component.

2. **Pitfalls #2 and #3.** Neither. The batch line is this binary clamping its default, and the server still answered. The `</s>` line is GGUF metadata. The trial vector was not all zeros. The word “bug” in that line does not mean the embedding endpoint is down. Both were left as printed.

3. **Pitfall #4.** The access control was the bind address: the box Tailscale IPv4, never `0.0.0.0`. No API key was added. The process does not come back on the next boot. `persist` was left at idle, and this server was never a unit or a fourth profile.

4. **Pitfall #5.** No. The check passed because `k3s` was absent from the top 5 chunks, not because 0.664 is a low score. That score sits in the same band as the true hits (0.554–0.670). A cosine does not report absence. This Qwen run did not generate, so it also did not show a refusal.

5. **Pitfall #6.** No, and no. Rank 1 is the false note, ahead of the true guide by 0.001. No completion was requested after this retrieval. The answer that opened with “Ollama” was produced on 19 September, on the MiniLM index.

6. **Pitfalls #7 and #8.** The MiniLM index of 19 September, with the text model on `:8000`. Not the Qwen index. The empty completion was the token budget: the same chunks at `max_tokens=2000` did answer. The invented name is a different failure. Retrieval does not bind the generator to the nouns in the window. The check has to require `silent` and `silence_rate`. Raising the budget does not fix that, and neither does a better ranking.

7. **Pitfall #9.** The check’s. The completion had the French stem `téléportée`. Requiring the English stem `teleport` rejected a real answer. That is why a 4/6 at `max_tokens=400` was not four model failures.

8. **Pitfall #10.** The string `llama.cpp` is in the true guide and in the false note, so the check cannot tell them apart. The completion opened with “Ollama” and did not say the sources conflict, even though three true chunks outranked the false note (8th, at 0.527). The check has to reject that opening and require the conflict to be named. That completion is 19 September. The 23 September run measured rank only, and on that day the false note was first.
