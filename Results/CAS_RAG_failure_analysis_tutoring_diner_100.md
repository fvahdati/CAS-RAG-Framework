# CAS-RAG failure notes — tutoring diner (100 comments)

**File checked:** `algorithmsResult-detail.xlsx`  
**Video:** tutoring diner (`video_02`)  
**Focus:** what kept going wrong for **CAS-RAG** after reading all rows  

---

## Numbers (careful scoring)

- Gold emotion **`None`** counts as a real label (not empty).
- Only 3 gold **`?`** rows left out of emotion scoring.
- Gold routing typo **`ambigues`** treated as **ambiguous**.
- **`pending_human`** counted as wrong for emotion when gold exists.

| Algorithm | Routing mistakes | Routing % | Emotion mistakes | Emotion n | Emotion % |
|-----------|-----------------:|----------:|-----------------:|----------:|----------:|
| Algorithm 1 | 49 | 51.0 | 42 | 97 | 56.7 |
| Algorithm 2 | 47 | 53.0 | 38 | 97 | 60.8 |
| Algorithm 3 | 68 | 32.0 | 52 | 97 | 46.4 |
| Algorithm 4 | 34 | 66.0 | 42 | 97 | 56.7 |
| **CAS-RAG** | **28** | **72.0** | **36** | **97** | **62.9** |

Across Alg1–4 + CAS-RAG: 226 routing mistakes and 210 emotion mistakes (436 combined).

---

## What I noticed when I read the CAS-RAG rows

After going through all 100 records, the mistakes did not feel random. The same kinds of problems came up again and again. It was less like fifty different one-off errors and more like a few habits the model kept repeating.

### Routing: calling things context-dependent too often

This was the biggest routing problem. CAS-RAG often decided a comment needed the video even when the comment already made sense on its own. I saw this a lot in records such as 13, 15, 17, 18, 25, and 48. People were telling a full personal story, giving an opinion, or explaining something clearly, but the label still came out as context-dependent. In short, the model kept overestimating how much outside context was required.

### Routing: missing cases that really do need the video

The opposite also happened, just less often. Some comments really did depend on people or moments in the video, but CAS-RAG treated them as explicit / self-contained. Records 3, 23, and 24 are good examples. Comments about Aubrey or Harper can look clear at sentence level, but you still need the video to know *why* the person is reacting. The model got the surface meaning, but missed that the actual reason sits in the scene.

### Routing: not sitting with uncertainty

A few comments were honestly unclear. Instead of leaving them ambiguous, CAS-RAG forced a more definite interpretation. Records 8, 9, 20, and 22 show this. Sometimes you cannot tell if the writer is joking, being sarcastic, just noticing something, or making a soft suggestion. The model often picked one answer anyway.

### Low confidence / human review

I also saw cases that went to human review when the comment was not junk — it just did not give enough signal for a confident call. Those felt more like “not sure yet” than like something that should be thrown away. In this Excel file those shows up mainly as `pending_human`, not discarded.

### Emotion: reading praise as Empathy

This was the clearest emotion problem. When comments used words like sweet, cute, great, beautiful, or precious, the model often said Empathy. But a lot of those comments were just admiring or liking something — not showing emotional concern for another person. This repeated in records 26, 67, 77, and 99. Praise and empathy are not the same thing, and the model blurs them.

### Emotion: treating someone’s own story as Empathy

Some people were just talking about their own life. The model still often labelled that as empathy toward someone else. Records 44, 51, and 63 are examples. Talking about your memory is not automatically empathy.

### Emotion: taking emojis too seriously

In records such as 20, 38, 45, and 60, emojis like 😭 or 🥲 seemed to pull the prediction toward Empathy or a stronger emotion, even when the words around them did not really support that. Online, those emojis are often joke-y, dramatic, or sarcastic. The model sometimes read them too literally.

### Emotion: reading criticism as Frustration

This showed up less, but it was still there. In records 83 and 92 the writer was mainly judging a situation or giving an opinion, and the model leaned to Frustration. Critical wording is not always emotional frustration.

---

## Short summary

If I had to put it in a few sentences: the biggest routing weakness for CAS-RAG here is over-predicting context dependency. The biggest emotion weakness is mixing praise / admiration with Empathy. Most of the other mistakes sit under those two habits, plus the smaller patterns above (forced certainty, personal stories, emojis, criticism→frustration).

---

## What this points to for fixes

| Problem I kept seeing | Where to harden next |
|-----------------------|----------------------|
| False context dependency | Layer-1 context cues; P1 “resolved” examples; Alg1 RAG skip when comment is already clear |
| Missed real scene references | Keep named/scene prototypes; P1 “unknown” examples |
| Forcing a label when unclear | Keep Ambiguous path; do not collapse it away |
| Praise ≠ Empathy | Stronger Empathy definition + admiration examples in prompts |
| Own story ≠ Empathy | Examples of None / personal recollection in P2/P3 |
| Emoji over-weight | Prompt note: emoji alone is weak evidence |

---

## Related files

- Prompt change log: `prompts/Algorithm 1 latest run prompts_17June2026_CHANGES.docx`
- Metrics workbook: `Results/algorithmsResult-detail.xlsx`
- Row detail: `algorithmsResult-detail.xlsx` (repo root)
