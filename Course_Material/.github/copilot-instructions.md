# Copilot Instructions for Language and AI Course Repository

Target: Help AI agents work productively in this Markdown-only course notes repository for "Language and AI" (Tilburg University, Cognitive Science & AI).

## Repository Architecture

- **Pure Markdown content**; no code builds. Structure mirrors the 7-week teaching flow.
- **Weekly folders** (`Week 1/` to `Week 7/`): Each contains:
  - `Topic_Week_N/` subfolder with main lecture notes (`topic.md`) and a `files/` directory for supplementary slides (`.md` transcripts)
  - `Lab_Session_N/` subfolder with practical exercises (`lab_session_n.md`)
  - `Generated/` subfolder reserved for AI model outputs (currently empty; use for generated content)
- **Top-level `Generated/`**: Placeholder for cross-week AI-generated materials.

## File Patterns and Naming Conventions

| Pattern | Purpose | Example |
|---------|---------|---------|
| `Week N/Topic_Week_N/topic.md` | Main lecture notes | `Week 3/Classification_Week_3/classification.md` |
| `Week N/Topic_Week_N/files/*.md` | Slide transcripts and book chapter(s) | `files/week3.md`, `files/week3_inperson_slides.md`, `files/3.md`, `files/4.md`, `files/5.md` |
| `Week N/Lab_Session_N/lab_session_n.md` | Practical exercises and coding tasks | `Lab_Session_1/lab_session_1.md` |
| `Week N/Generated/` | AI-generated content for that week | Place summaries, flashcards here |

## Subject Focus

This course covers **Natural Language Processing (NLP)** and **Text Mining** with emphasis on:
- **Weeks 1-2**: Tokenization, vectorization (tf-idf, BoW), text distances (Euclidean, Cosine, Jaccard), regex, normalization
- **Week 3**: Classification (Naive Bayes, Decision Trees, Logistic Regression, SVMs, k-NN), n-gram language models, smoothing
- **Week 4**: Representation learning (PPMI, word embeddings), neural network fundamentals
- **Week 5**: Sequence labeling (HMMs, Viterbi), information extraction, NER
- **Week 6**: Deep learning (RNNs, RNNLMs, Transformers)
- **Week 7**: Large Language Models (Encoder/Decoder, Attention, Prompt Engineering)

## Style and Notation Guidelines

### Writing Style
- **Strict Academic Tone**: Generated content must adhere to strict academic writing standards. Use formal grammar, precise vocabulary, and objective tone. Avoid colloquialisms, contractions, and emojis (even if present in source files).
- **Content Elevation**: Treat existing lecture notes as a baseline. AI-generated content should improve upon the source material by enhancing clarity, depth, and formality of explanations.
- **Structure**: Use consistent heading levels; `##` for main sections, `###` for subsections.

### Mathematical Notation
- **Inline math**: Use `$...$` for inline expressions (e.g., `$P(w_n | w_{n-1})$`)
- **Display math**: Use `$$...$$` for block equations
- **Vectors**: Denote with `\vec{x}` or bold `\mathbf{x}`
- **Common symbols**: $W$ for sequences, $w_i$ for tokens, $X$ for feature space, $\vec{x}_n$ for instance vectors, $y$ for labels
- **Norms**: Use `\ell_2` notation; dot products as `\bullet` or `\cdot`


### Document Structure Template (Academic book chapter), for generated content

# Chapter Title

| Covered Material | Value |
| :--- | :--- |
| **Chapter** | Chapter N |
| **Book** | Book Title (if applicable) |
| **Related Week** | Week N (Course mapping, if applicable) |
| **Related Slides / Files** | [filename.md](/path_filename) |

---

## Index / Structure (minimal)

- Index (this list) — list of sections and subsections.
- Brief Content Summary — 3–6 sentences; defines acronyms on first use.
- Abstract — 150–250 words (concise aims, methods, results, conclusions).
- Keywords — semicolon-separated.
- 1. Introduction
  - Purpose, scope (one short paragraph).
- 2. Background
  - Key references (bulleted), concise context (one short paragraph).
- 3. Theory / Notation
  - Definitions and symbols (use `$...$` for inline math).
  - Subsections for conventions or lemmas as needed.
- 4. Main chapter content
  - Multiple sections/subsections as needed.
  - Each section begins with a one-line intent statement.
- 5. Conclusion
  - Summary (short paragraph).
- 6. References
  - Consistent citation style; list key sources only.
- 7. Appendices (optional)
  - Extended derivations or code snippets.
- 8. Glossary (optional)
  - Key terms and symbols, one-line definitions.

Styling and formatting rules (concise)
- Use `##` for main sections and `###` for subsections.
- Each section begins with a one-line intent statement.
- Keep paragraphs short (1–3 sentences) unless a worked example requires stepwise detail.
- Allow stylistic freedom beyond these constraints; adhere to concise, formal academic tone.

---

Notes on Mathematical Notation and Style:
- Use `$...$` for inline math and `$$...$$` for display equations.
- Denote vectors as $\vec{x}$ or $\mathbf{x}$.
- Use $\ell_2$ for Euclidean norm and $\cdot$ or $\bullet$ for dot products.
- Maintain strict academic tone: formal grammar, precise vocabulary, and objective register. Define acronyms on first use.
- Ensure the chapter begins with the Index and the Brief Content Summary.

Revision checklist:
- [ ] Document begins with Index.
- [ ] Abstract present and concise.
- [ ] Brief Content Summary present.
- [ ] Acronyms defined on first mention.
- [ ] Mathematical expressions use $...$ or $$...$$.
- [ ] Files to be saved in appropriate folder following repository conventions.

## How to Add or Modify Content

1. **Place content in correct week folder** following naming conventions above.
2. **Always begin with an index** containg all document subsections.
3. **Always write a summary** covering key concepts explained in the document.
5. **Generated content**: Place in appropriate `Generated/` folder; prefix filenames descriptively (e.g., `week3_summary.md`).

## Academic Writing and Tone Guidelines

### Do
- **Elevate the material**: Use existing lecture notes as a conceptual basis but rewrite explanations to meet formal academic standards.
- **Use formal, precise language**: "The model achieves 87% accuracy" rather than "The model does really well."
- **Define terms on first use**: Introduce acronyms with full expansion (e.g., "Natural Language Processing (NLP)").
- **Cite sources explicitly**: Reference Jurafsky & Martin chapters, textbooks, or foundational papers when introducing concepts.
- **Structure logically**: Progress from foundational concepts to applications; use transitions between sections.
- **Maintain consistent voice**: Use third-person passive voice or active voice consistently throughout; avoid "I" or "we."
- **Explain the "why"**: Provide intuition behind methods before mathematical formalism.
- **Use worked examples**: Follow concepts with concrete, step-by-step illustrations.

### Don't
- **Mimic informal source tone**: Do not copy the casual or conversational style found in some lecture notes.
- **Use colloquialisms or slang**: Avoid "cool," "basically," "obviously," "just," or informal register.
- **Make unsupported claims**: Never assert facts, benchmarks, or findings without explicit citations.
- **Oversimplify complex topics**: Acknowledge nuance and edge cases; don't reduce concepts to one-liners.
- **Mix mathematical rigor inconsistently**: If using formal notation, maintain it; don't switch between symbolic and verbal descriptions arbitrarily.
- **Include personal opinions**: Keep content objective and grounded in established NLP literature.
- **Use excessive formatting**: Avoid **bold**, *italics*, or CAPITALS for emphasis beyond academic conventions (e.g., math symbols, section titles).
- **Invent or speculate**: Stick to course scope (Weeks 1–7 content); don't introduce bleeding-edge methods not covered.

## Revision Checklist

Before finalizing content:
- [ ] All mathematical expressions use proper LaTeX (`$...$` inline, `$$...$$` block).
- [ ] Acronyms are defined on first mention.
- [ ] Tone is consistent with existing course materials.
- [ ] Sources (book chapters, slides) are referenced where concepts are introduced.
- [ ] No placeholder links remain unfilled (except intentional `[Video Placeholder]`).
- [ ] Examples align with course datasets and Week N focus.
