# UI control audit

Date: 2026-10-13. Scope: every interactive control in the React UI, in the **live** application
(`groundedai serve`, API + UI) and in the **static demo** (`npm run build:demo`, the build used
for the private preview).

Each control is classed as:

* **A, live:** works against the running backend;
* **B, recorded:** works in the static demo by replaying data recorded from a real local run, and
  is labelled as a recording;
* **C, unavailable:** hidden or disabled, with the reason shown on the page.

## Findings before the fix

| Control | Symptom | Root cause |
|---|---|---|
| Hybrid / Lexical / Dense (static demo) | Clicking did nothing | The buttons were deliberately `disabled` in demo mode because only one retrieval setting had been recorded. `.segmented button` had no disabled style and no text explained it, so they looked like working buttons. |
| Passages selector (static demo) | Could not be changed | Same as above: `disabled` in demo mode, without an explanation. |
| Retrieval diagnostics checkbox (static demo) | Ticking or unticking changed nothing | `StaticDemoApi.query` ignored `include_diagnostics` and always returned the recorded diagnostics. |
| Top navigation (any embedding frame) | Reported not to work in a hosted preview | The click handler assigned `window.location.hash` **before** updating the React state. If the hosting frame blocks or ignores that assignment, the state update never runs. **Not reproduced here**: the preview needed a signed-in browser session, which this environment did not have. The dependency was removed. |
| Ask page after visiting another tab | The answer and the chosen settings were lost | Only the active page was mounted; switching tabs unmounted the Ask page and its state. |
| Operations "Refresh" (static demo) | Clicking appeared to do nothing | It reloaded the same recorded snapshot. |
| Settings vs. displayed answer (both modes) | After changing the mode, the old answer stayed on screen with no sign that it came from the previous setting | Nothing recorded which settings produced the answer. |

The live backend itself was correct: `retrieval_mode`, `top_k` and `include_diagnostics` were
already sent by the UI and honoured by `/api/v1/query`. That path had no API test, though.

## Changes

* **Navigation** (`App.tsx`): the view switches first; the tab is then written to the address bar
  with `history.pushState` inside `try/catch`, and `popstate` is handled for Back/Forward. The Ask
  page stays mounted (hidden) so its answer and settings survive tab changes.
* **Static demo recordings** (`scripts/export_demo_data.py`, `api/demo.ts`): each of the 10 demo
  questions is recorded with all 3 retrieval modes × 4 passage counts (120 real responses from a
  local run with bge-small embeddings and the extractive generator). The demo replays the
  recording made with exactly the selected setting, drops diagnostics unless requested, and
  reports a clear error for a setting that was not recorded instead of substituting another one.
* **Ask page** (`AskPage.tsx`, `AnswerView.tsx`): all controls enabled in both modes; a line under
  the controls describes the selected mode; every answer states "Lexical (BM25) retrieval · 3 of
  up to 3 passages used"; when the settings change after an answer, a notice offers "Ask again
  with Dense, 3 passages"; recorded questions are disabled while a request is running.
* **Disabled states** (`styles.css`): every disabled control is visibly dimmed with a
  not-allowed cursor.
* **Operations**: "Refresh" is shown only against a live server.

## Control-by-control result

| Control | Class | Verified by |
|---|---|---|
| Ask / Documents / Evaluation / Operations tabs | A and B | `interactions.test.tsx` (each tab shows its page; still works when `history.pushState` throws; Back via `popstate`); browser: live app at 390 px and 1280 px, static demo inside a sandboxed iframe with `pushState` blocked |
| Hybrid / Lexical / Dense | A, B | tests: each sends `retrieval_mode` to the API; demo replays the matching recording; API test `test_query_uses_the_requested_retrieval_mode` (lexical: 0 dense candidates; dense: 0 lexical candidates; hybrid: both, RRF). Browser, live: diagnostics showed `lexical 20, dense 0`, `lexical 0, dense 20`, `hybrid (rrf fusion)` |
| Passages (3, 5, 8, 10) | A, B | tests: `top_k` sent; API test: 1 vs 2 passages; browser, live: 3 and 8 passages used |
| Retrieval diagnostics | A, B | tests: `include_diagnostics` true/false; demo drops diagnostics when unticked; API test: diagnostics only on request |
| Question box, Ctrl+Enter, Ask question | A (B for recorded questions) | tests: Ctrl+Enter submits; each Ask is a new API call; too-short questions disable the button; demo explains that free-text questions need the live app |
| Recorded / example questions | A, B | tests: clicking asks that question; demo shows the expected result next to the recording |
| "Ask again with …" notice | A, B | tests: appears after a settings change and re-asks with the new settings |
| Theme | A, B | test: cycles system → light → dark and sets `data-theme`; browser in both builds |
| Documents: upload, list, chunks, re-ingest, delete | A; C in the demo | existing API and end-to-end tests; browser, live: upload of a text file → Ready → Chunks shows the text → Hide → Re-ingest → Ready → Delete (asks "Delete … and its chunks?", list goes from 15 to 14). Demo: upload form and row actions hidden, banner explains why (test) |
| Evaluation: report selector, Show table / Show chart | A, B | existing tests; browser in both builds |
| Evaluation: "Evaluate this instance" | A; C in the demo | shown only against a live server |
| Operations: Refresh and 15 s auto-refresh | A; C in the demo | Refresh hidden in the demo (test); the snapshot banner explains it |
| Loading, empty, error, backend-unavailable states | A, B | tests: loading line during a request, empty documents state, API-unreachable banner, model-unavailable banner and error |

With a real language model: the live app was also run with `qwen2.5:1.5b` through Ollama. In the
browser the loading state showed while the model ran (about 6 s), and the answer carried the
provider and model, real token counts, the retrieval line ("Hybrid retrieval · 5 of up to 5
passages used") and, because the model wrote no `[S#]` label, the "Not grounded" badge with
"Treat it as unverified" (`docs/screenshots/ask-ollama-light.png`).

## Added on 2026-10-14

| Element | Class | Verified by |
|---|---|---|
| Verdict badge, claim check list with quotes and source links | A, B | `claims.test.tsx`; browser, live app (extractive generator, bge-small) at 1100 px and 390 px, light and dark |
| Disagreement panel (both quotes, values, dates, which side the answer reports) | A, B | `claims.test.tsx`; browser, live app: the hotel-limit question shows €140/€110 next to €120 with "updated May 2025" |
| Withheld answer (with `GROUNDEDAI_UNVERIFIED_ANSWERS=withhold`) | A | `claims.test.tsx` (not shown in the static demo, which uses the default) |
| Note that dense retrieval uses the hashing embedder | A | `claims.test.tsx`; shown only when the server's embedder is `hashing` |
| Scoring-version-4 tiles and the "recorded outputs re-checked" banner | A, B | `claims.test.tsx`; static demo includes the re-checked reports |

The static demo was re-recorded on 2026-10-14 with these fields.

## Limits of this audit

* The hosted preview itself could not be opened signed-in from this environment. The fix
  removes the suspected cause and was checked in a local sandboxed iframe that blocks history
  changes, which is an approximation of the real host, not the host itself.
* In the static demo the mode and passage controls select between **recordings**; nothing is
  retrieved or generated in the browser. The page says so next to the controls and on every
  answer.
