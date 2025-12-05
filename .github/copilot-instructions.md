# Copilot Instructions for Language and AI Repository

Target: Help AI agents work productively in this Markdown-only repository for "Language and AI" (Tilburg University, Cognitive Science & AI). The repository supports both course study materials and project-oriented documentation.

## Repository Architecture

- **Pure Markdown content**; no code builds. Structure serves two primary purposes: course study and project documentation.
- **`Course_Material/`**: Contains weekly lecture notes, lab sessions, and generated study content (see `Course_Material/.github/copilot-instructions.md` for detailed guidance).
- **`Misc/`**: Project-oriented documentation, assessment information, and supplementary resources.
  - `Assessment_Information/`: Exam details, grading criteria, and mock exam materials.
  - `Interim_Assignment/`: Research proposal and paper documentation, including data corpus specifications.
  - `Reading/`: Supplementary reading lists and reference materials.
- **`Pdfs/`**: Storage for PDF resources (not processed by AI agents).
- **Top-level `Generated/`**: Placeholder for cross-repository AI-generated materials.

## File Patterns and Naming Conventions

| Pattern | Purpose | Example |
|---------|---------|---------|
| `Course_Material/Week N/...` | Weekly lecture and lab content | See `Course_Material/.github/copilot-instructions.md` |
| `Misc/Assessment_Information/*.md` | Exam schedules, grading rubrics | `assesment_information.md` |
| `Misc/Interim_Assignment/*.md` | Research proposal, paper specs, corpus documentation | `interim_assignment.md`, `SOBR.md` |
| `Misc/Reading/*.md` | Reading lists and references | `reading.md` |
| `Generated/` | AI-generated cross-repository content | Summaries, analyses, project plans |

## Subject Focus

This repository covers **Natural Language Processing (NLP)** and **Text Mining** with emphasis on:
- **Weeks 1–2**: Tokenization, vectorization (tf-idf, BoW), text distances (Euclidean, Cosine, Jaccard), regex, normalization
- **Week 3**: Classification (Naive Bayes, Decision Trees, Logistic Regression, SVMs, k-NN), n-gram language models, smoothing
- **Week 4**: Representation learning (PPMI, word embeddings), neural network fundamentals
- **Week 5**: Sequence labeling (HMMs, Viterbi), information extraction, NER
- **Week 6**: Deep learning (RNNs, RNNLMs, Transformers)
- **Week 7**: Large Language Models (Encoder/Decoder, Attention, Prompt Engineering)

**Applied Research Focus** (Interim Assignment):
- Stylometric profiling and author attribute inference
- Author obfuscation techniques for privacy protection
- Bias detection and mitigation in NLP classifiers
- Corpus construction and distant supervision methodologies

## AI Agent Task Scope

AI agents operating at this repository level are expected to perform:

### Documentation Tasks
- **Specification Writing**: Draft detailed specifications for research proposals, experimental setups, and deliverables.
- **Document Analysis**: Analyze existing Markdown files for completeness, consistency, and adherence to academic standards.
- **Summary Generation**: Produce executive summaries, progress reports, and synthesis documents across multiple sources.
- **Template Creation**: Generate reusable templates for research papers, proposals, and project documentation.

### Project Planning Tasks
- **Milestone Definition**: Break down research objectives into actionable milestones with clear deliverables.
- **Task Decomposition**: Convert high-level goals into specific, measurable tasks.
- **Timeline Estimation**: Propose realistic timelines based on scope and dependencies.
- **Risk Identification**: Flag potential blockers, ambiguities, or missing requirements.

### Analysis Tasks
- **Gap Analysis**: Identify missing sections, undefined terms, or incomplete arguments in documents.
- **Consistency Checking**: Verify alignment between related documents (e.g., proposal vs. paper structure).
- **Citation Verification**: Ensure references are properly formatted and consistently applied.
- **Requirements Extraction**: Extract explicit and implicit requirements from assignment briefs.

## Style and Notation Guidelines

### Writing Style
- **Strict Academic Tone**: All generated content must adhere to formal academic writing standards. Use precise vocabulary, objective tone, and formal grammar. Avoid colloquialisms, contractions, and emojis.
- **Content Elevation**: Treat existing documents as a baseline. AI-generated content should improve clarity, depth, and formality.
- **Structure**: Use consistent heading levels; `##` for main sections, `###` for subsections.

### Mathematical Notation
- **Inline math**: Use `$...$` for inline expressions (e.g., `$P(w_n | w_{n-1})$`)
- **Display math**: Use `$$...$$` for block equations
- **Vectors**: Denote with `\vec{x}` or bold `\mathbf{x}`
- **Common symbols**: $W$ for sequences, $w_i$ for tokens, $X$ for feature space, $\vec{x}_n$ for instance vectors, $y$ for labels
- **Norms**: Use `\ell_2` notation; dot products as `\bullet` or `\cdot`

## Document Structure Templates

### Project Specification Document

```markdown
# [Project/Feature Title] Specification

| Metadata | Value |
| :--- | :--- |
| **Document Type** | Specification |
| **Version** | X.Y |
| **Last Updated** | YYYY-MM-DD |
| **Status** | Draft / In Review / Approved |
| **Related Documents** | [Link to related docs] |

---

## Index

- Index (this list)
- Executive Summary — 3–5 sentences; defines scope and purpose.
- 1. Objectives
  - Primary goals (bulleted).
  - Success criteria.
- 2. Scope
  - In-scope items.
  - Out-of-scope items.
- 3. Requirements
  - Functional requirements.
  - Non-functional requirements.
- 4. Constraints and Assumptions
  - Known constraints.
  - Working assumptions.
- 5. Deliverables
  - List of expected outputs with acceptance criteria.
- 6. Timeline and Milestones
  - Key dates and dependencies.
- 7. Risks and Mitigations
  - Identified risks with mitigation strategies.
- 8. References
  - Related documents, sources, and dependencies.
```

### Project Planning Document

```markdown
# [Project Name] Planning Document

| Metadata | Value |
| :--- | :--- |
| **Document Type** | Planning |
| **Version** | X.Y |
| **Last Updated** | YYYY-MM-DD |
| **Project Lead** | [Name] |
| **Team Members** | [Names] |

---

## Index

- Index (this list)
- Project Overview — Brief description of project aims.
- 1. Objectives and Success Criteria
- 2. Work Breakdown Structure
  - Phase 1: [Phase Name]
    - Task 1.1: [Description]
    - Task 1.2: [Description]
  - Phase 2: [Phase Name]
    - ...
- 3. Timeline
  - Gantt chart or milestone table.
- 4. Resource Allocation
  - Team responsibilities.
  - External dependencies.
- 5. Risk Register
  - Risk | Likelihood | Impact | Mitigation
- 6. Progress Tracking
  - Status updates and completion criteria.
- 7. Review and Approval
  - Sign-off requirements.
```

### Document Analysis Report

```markdown
# Document Analysis: [Document Title]

| Metadata | Value |
| :--- | :--- |
| **Analyzed Document** | [Path/Filename] |
| **Analysis Date** | YYYY-MM-DD |
| **Analyst** | AI Agent |

---

## Index

- Index (this list)
- Summary of Findings — Key observations in 3–5 sentences.
- 1. Document Overview
  - Purpose and scope of analyzed document.
- 2. Structural Analysis
  - Completeness of sections.
  - Logical flow assessment.
- 3. Content Analysis
  - Clarity and precision of language.
  - Technical accuracy.
  - Citation completeness.
- 4. Identified Issues
  - Critical issues (must address).
  - Minor issues (recommended improvements).
- 5. Recommendations
  - Prioritized action items.
- 6. Appendix
  - Detailed findings by section.
```

## How to Add or Modify Content

1. **Place content in correct folder** following naming conventions above.
2. **Always begin with an index** containing all document subsections.
3. **Always write a summary** covering key concepts or findings.
4. **Generated content**: Place in appropriate `Generated/` folder or relevant `Misc/` subdirectory; prefix filenames descriptively (e.g., `interim_spec_v1.md`, `sobr_analysis.md`).
5. **Cross-reference related documents**: Link to related files within the repository using relative paths.

## Academic Writing and Tone Guidelines

### Do
- **Elevate the material**: Use existing documents as a conceptual basis but rewrite explanations to meet formal academic standards.
- **Use formal, precise language**: "The model achieves 87% accuracy" rather than "The model does really well."
- **Define terms on first use**: Introduce acronyms with full expansion (e.g., "Natural Language Processing (NLP)").
- **Cite sources explicitly**: Reference Jurafsky & Martin chapters, textbooks, or foundational papers when introducing concepts.
- **Structure logically**: Progress from foundational concepts to applications; use transitions between sections.
- **Maintain consistent voice**: Use third-person passive voice or active voice consistently throughout; avoid "I" or "we" unless documenting personal contributions.
- **Explain the "why"**: Provide rationale behind decisions, requirements, or methodological choices.
- **Use concrete examples**: Follow abstract concepts with specific, illustrative instances.

### Don't
- **Mimic informal source tone**: Do not copy casual or conversational style from source materials.
- **Use colloquialisms or slang**: Avoid "cool," "basically," "obviously," "just," or informal register.
- **Make unsupported claims**: Never assert facts, benchmarks, or findings without explicit citations.
- **Oversimplify complex topics**: Acknowledge nuance and edge cases; provide sufficient detail.
- **Mix formality inconsistently**: Maintain uniform register throughout documents.
- **Include personal opinions**: Keep content objective and grounded in established literature or explicit requirements.
- **Use excessive formatting**: Avoid overuse of **bold**, *italics*, or CAPITALS beyond academic conventions.
- **Invent or speculate**: Stick to documented scope; flag uncertainties explicitly.

## Revision Checklist

Before finalizing content:
- [ ] Document begins with Index and metadata table.
- [ ] Executive summary or abstract is present and concise.
- [ ] All mathematical expressions use proper LaTeX (`$...$` inline, `$$...$$` block).
- [ ] Acronyms are defined on first mention.
- [ ] Tone is consistent and formally academic.
- [ ] Sources (papers, chapters, specifications) are referenced where concepts are introduced.
- [ ] No placeholder links remain unfilled (except intentional placeholders marked as such).
- [ ] Document structure follows appropriate template for its type.
- [ ] Cross-references to related repository documents are valid.
- [ ] File saved in appropriate folder following repository conventions.
