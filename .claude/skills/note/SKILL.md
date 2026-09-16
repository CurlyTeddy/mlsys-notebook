---
name: note
description: Write or revise a concept note in this notebook. Takes a topic as the first argument and optional references (URLs, file paths, or free text) after it. Use when the user asks to write, draft, add, or revise a note on an ML systems concept, or invokes /note.
---

# Writing a concept note

Arguments: `<topic> [reference ...]`

The first argument is the topic. Everything after it is a reference.

## 1. Resolve the folder

Kebab-case the topic into a folder name (`KV cache` → `kv-cache/`). Check whether the folder already exists — if it does, revise `README.md` in place rather than starting over. Check sibling folders for an existing note that covers the same ground under a different name before creating a new one.

## 2. Pull the references

- URL → WebFetch
- Local path → Read
- Anything else → treat as a claim or hint from the user, not a source

References are input, not an outline. Use a reference when it supplies a number, a mechanism detail, or a claim worth citing; ignore the rest. Do not summarize a reference, mirror its structure, or pad the note to cover everything it mentions. A note that uses one line from one reference and nothing else is a correct outcome.

If a reference contradicts another, say so in the note and name which one you followed.

## 3. Write

Three sections, in this order:

1. **The problem** — what breaks or gets slow without this. Name the bottleneck concretely: memory bandwidth, latency, comms volume, tail latency.
2. **The idea** — the mechanism, in the smallest form that is still true.
3. **The cost** — what it trades away. Every ML systems technique trades something; a note without this section is unfinished.

Prefer a real diagram or a worked number over a paragraph. Arithmetic — bytes moved, FLOPs, tokens/s — carries the argument better than adjectives.

Never draw a diagram in markdown. No ASCII art, no box-drawing characters, no mermaid, no figure assembled out of code fences. They read as noise and misalign across fonts. A diagram is either a real image file in `images/` or it does not go in the note; if neither a reference nor the user supplies one, carry the point with a table, a number, or a sentence.

Cite a paper or post inline as a link at the point of the claim, not in a bibliography at the bottom.

### Never state a formula without deriving it

A bare equation is unusable later — it cannot be checked, adapted, or trusted. Any formula in a note carries, in a few lines:

- what each symbol is, and the model being assumed (what is independent, what is constant, what is being counted);
- the steps from that model to the result — the counting argument, the series, the cancellation;
- where the model breaks in practice, since these are approximations of real systems.

Read off whatever the closed form makes obvious — a limit, a floor, a term that vanishes. That is usually the reason the formula is in the note at all.

The same applies to a number: show the quantities and the division, not just the result.

## Style

Compact. Specifically:

- Cut adjectives and intensifiers. "Reduces memory traffic 4x" not "dramatically reduces memory traffic."
- State each idea once. If the mechanism is explained in section 2, section 3 refers to it, it does not re-explain it.
- No preamble, no restatement of the section heading in its first sentence, no closing summary.
- No hedging filler ("it is worth noting that", "essentially", "in order to").
- Prefer a table or list over prose when comparing more than two things.

Target a note that is read in a few minutes and reread later for the numbers.

## 4. Artifacts

Runnable snippets go in `code/`, images in `images/`. Reference each from the prose. Create these folders only when something goes in them; never add a snippet or an image the note does not point at.

### Diagrams

When a reference contains a diagram that carries the mechanism, download it rather than describing it:

1. Find the image URL — WebFetch the page asking for image URLs and their captions, or fetch the raw HTML and pull `<img src>` / `og:image`.
2. Download to `images/` with a descriptive kebab-case filename (`images/draft-verify-loop.png`), not the source's hashed name.
3. Embed with alt text, and credit the source underneath as an italic line linking to the page it came from.

Take a diagram only when it shows the mechanism — a loop, a memory layout, a timeline, a topology. Skip decorative headers, marketing charts, and benchmark bars whose axes the note does not discuss. Two diagrams in a note is usually one too many.

If a diagram would help and no reference has one, say so at the end of the note in a single line (`TODO: diagram of X`) and move on.

## 5. Cross-link

Link to sibling concepts with relative links (`../kv-cache/README.md`) where the connection is real — a shared bottleneck, a technique this one composes with or replaces. Skip the link rather than manufacture a connection.
