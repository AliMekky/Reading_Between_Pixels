# Automatic object-addition pilot — completed

Five edits generated through the OpenAI Images API using `gpt-image-2`, medium quality, size auto, one output per input. The bundled imagegen CLI was used. No masks or manual placement were supplied; the editor received the source photograph, insertion word, question, correct answer, and preservation constraints. No output retouching or post-hoc selection among variants was performed.

[Open the self-contained before/after gallery](gallery.html). Exact prompts, input/output hashes, dimensions, selection procedure, and review notes are in [manifest.json](manifest.json). Prompt files are in prompts/ and original/API-output PNGs in output/imagegen/.

## Visual review

| Question ID | Object | Result | Reason |
|---|---|---|---|
| 19358422 | book | Reject | Added to the queried table; book becomes another possible answer. |
| 10720200 | hat | Promising | On the dining table, not worn by the woman; glasses remain visible. |
| 05580253 | vase | Promising | On a background ledge, separate from the flowers and their planter; small object needs recognizability checking. |
| 10872518 | pan | Promising | On the counter beside the stove, while the pot remains on the burner. |
| 10560336 | plates | Reject | Added to the queried upper shelf; plates become another possible answer. |

This is assistant visual inspection, not independent annotation or a measured success rate on representative data. Selection used only metadata from the shared 305 questions: seed-42 shuffle and the first distinct nouns in the predefined small-object set book, hat, vase, pan, plates. Model outcomes were not consulted.

## Interpretation and next step

Automatic insertion is feasible, but a prompt alone does not reliably preserve the answer: two of five edits violate the crucial spatial-relation constraint. Three pass this initial semantic check, not final causal validation. Outputs also have different pixel dimensions and visibly regenerated fine details; unchanged pixels outside the addition have not been established. These images are not ready for the main causal comparison.

For scaling, add an automatic placement-planning step that identifies a permissible region from the image and question, then checks the edited image for the inserted object and for answer contamination. This can remain fully automatic. Evaluate the validator against a small independent audit. If localized preservation remains poor, use automatically generated masks or automated localized compositing with boundary checks. Apply the text overlay only after finalizing the edited image, and retain unrelated-object/sham edit controls.

No VLM behavioral evaluations, overlay comparisons, or extra edits were run. Five successful CLI calls took approximately 33–37 seconds each after submission. The CLI does not save API token usage, so exact billed cost is not available from these artifacts. Earlier built-in-tool attempts failed before producing outputs; their error is retained in the manifest.
