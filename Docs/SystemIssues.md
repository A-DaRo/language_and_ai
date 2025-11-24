# System Issues Analysis: Recursion, Naming, and Data Integrity

**Last Updated**: November 24, 2025
**Status**: Critical Analysis
**Severity Levels**: Issue 1 (Critical), Issue 2 (High), Issue 3 (Medium)

---

## Executive Summary

Following the recent refactoring, three new distinct issues have emerged that threaten system stability and data quality:

1.  **Infinite Recursion in User Confirmation Phase** (Critical Crash)
2.  **File Extension Loss in Downloads** (Data Quality)
3.  **Ambiguous Page ID Derivation** (Data Integrity)

This document details the root causes and architectural implications of these issues.

---

## Issue 1: Infinite Recursion in UserConfirmationPhase

### Symptom
The application crashes with `RangeError: Maximum call stack size exceeded` during the `UserConfirmationPhase`.

```
RangeError: Maximum call stack size exceeded
    at UserConfirmationPhase._printTreeNode (src\orchestration\phases\UserConfirmationPhase.js:103:13)
    at UserConfirmationPhase._printTreeNode (src\orchestration\phases\UserConfirmationPhase.js:106:22)
    ...
```

### Root Cause: Unbounded Graph Traversal
The `UserConfirmationPhase` attempts to display the discovered page structure as a **tree**, but the underlying data structure is a **graph** which may contain cycles (e.g., Page A links to Page B, which links back to Page A).

**File**: `src/orchestration/phases/UserConfirmationPhase.js`

```javascript
_printTreeNode(context, prefix, isLast, titleRegistry) {
  // ...
  exploredChildren.forEach((child, index) => {
    // RECURSIVE CALL WITHOUT CYCLE DETECTION
    this._printTreeNode(child, childPrefix, childIsLast, titleRegistry);
  });
}
```

When a cycle exists, `_printTreeNode` calls itself indefinitely until the stack overflows. The `PageGraph` correctly models these cycles, but the *visualization logic* assumes a strict tree structure.

### Architectural Implication
The visualization component fails to handle the "Graph -> Tree" projection required for display. It lacks a mechanism to identify "back-edges" or "cross-edges" during traversal.

---

## Issue 2: File Extension Loss in Downloads

### Symptom
Downloaded files are saved without file extensions (e.g., `files/Lecture_Slides` instead of `files/Lecture_Slides.pdf`), making them unusable on the host OS without manual renaming.

### Root Cause: Naive Extension Detection
The `FileNameExtractor` relies on the URL string or link text to determine the filename. It fails when:
1.  The URL does not contain a file extension (e.g., `https://example.com/download?id=123`).
2.  The `linkText` is used as the filename (which typically lacks an extension).
3.  The `_guessExtension` method uses a weak `includes()` check.

**File**: `src/download/file/FileNameExtractor.js`

```javascript
_guessExtension(url) {
  const urlLower = url.toLowerCase();
  for (const ext of this.fileExtensions) {
    if (urlLower.includes(ext)) { // ❌ False positives and misses
      return ext;
    }
  }
  return null;
}
```

### Architectural Implication
The system determines the target filename *before* the download begins. This prevents using the authoritative `Content-Type` or `Content-Disposition` headers from the HTTP response to determine the correct file extension. The architecture needs to support "late-binding" of filenames or a renaming step after headers are received.

---

## Issue 3: Ambiguous Page ID Derivation

### Symptom
The system may fail to correctly identify unique pages or may generate unstable IDs for pages, leading to duplication or "Untitled" pages.

### Root Cause: Strict Regex for ID Extraction
The `GlobalQueueManager` uses a regex that assumes a specific 32-character hex format for Notion IDs.

**File**: `src/orchestration/GlobalQueueManager.js`

```javascript
_derivePageId(url) {
  const match = url && url.match(/([a-f0-9]{32})/i);
  return match ? match[1] : url; // Fallback to full URL is risky
}
```

If the regex fails (e.g., for custom domain URLs or different ID formats), it falls back to the full URL. This is brittle because:
1.  URLs can vary (query params, protocol, trailing slashes).
2.  The same page might be accessed via different URLs, leading to split entities in the graph.

### Architectural Implication
The **Canonicalization Invariant** is threatened. If the ID derivation is unstable, the "Single Source of Truth" principle for page uniqueness is violated.

---
