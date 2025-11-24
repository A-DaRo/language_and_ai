# Refactoring Plan: Recursion, Naming, and Data Integrity

**Status**: Draft
**Target Version**: 1.1.1
**Based on**: `Docs/SystemIssues.md`

## 1. Executive Summary

This plan addresses three critical issues identified in the latest system analysis, plus a user-experience issue regarding dashboard display quality.
1.  **Infinite Recursion** in the User Confirmation Phase (Critical).
2.  **File Extension Loss** during downloads (High).
3.  **Ambiguous Page IDs** causing data integrity issues (Medium).
4.  **Non-sense Dashboard Titles** (Low/UX).

## 2. Detailed Implementation Plan

### Phase 1: Fix Infinite Recursion (Critical)

**Objective**: Prevent `RangeError` by handling cycles in the page graph during tree visualization.

**Component**: `src/orchestration/phases/UserConfirmationPhase.js`

**Proposed Changes**:
Implement a `visited` set to track the current recursion path. If a node is encountered that is already in the current path, stop recursion and mark it as a cycle.

```javascript
/**
 * Recursively prints the tree structure with cycle detection.
 * @private
 * @param {PageContext} context - The current page context.
 * @param {string} prefix - The indentation prefix.
 * @param {boolean} isLast - Whether this node is the last child.
 * @param {Object} titleRegistry - The ID-to-Title map.
 * @param {Set<string>} [pathVisited=new Set()] - IDs visited in the current recursion branch.
 */
_printTreeNode(context, prefix, isLast, titleRegistry, pathVisited = new Set()) {
  const connector = isLast ? '└─ ' : '├─ ';
  const title = titleRegistry[context.id] || context.title || 'Untitled';
  
  // Cycle Detection
  if (pathVisited.has(context.id)) {
    console.log(`${prefix}${connector}${title} ↺ (Cycle)`);
    return;
  }

  // ... (print logic) ...

  const newPathVisited = new Set(pathVisited);
  newPathVisited.add(context.id);

  exploredChildren.forEach((child, index) => {
    const childIsLast = index === exploredChildren.length - 1;
    this._printTreeNode(child, childPrefix, childIsLast, titleRegistry, newPathVisited);
  });
}
```

### Phase 2: Fix File Extension Loss (High)

**Objective**: Ensure all downloaded files have correct extensions, derived from URL or Content-Type headers.

**Components**:
-   `src/download/file/FileNameExtractor.js`
-   `src/download/file/FileDownloadStrategy.js`

**Proposed Changes**:

1.  **Enhance `FileNameExtractor`**:
    Add logic to strictly parse the URL path for an extension before falling back to heuristics.

    ```javascript
    /**
     * Extracts a filename, prioritizing the URL path extension.
     * @param {string} url - The source URL.
     * @param {string} linkText - The text of the link.
     * @returns {string} The best-guess filename.
     */
    extractFilename(url, linkText) {
       // 1. Try path.extname(url.pathname)
       // 2. If valid extension, use basename(pathname)
       // 3. Else, fallback to linkText + guessed extension
    }
    ```

2.  **Header-Based Renaming in `FileDownloadStrategy`**:
    Inspect the `Content-Type` header during download. If the local filename has no extension, append the correct one based on the MIME type.

    ```javascript
    /**
     * Downloads file and renames if extension is missing.
     * @returns {Promise<string>} The actual saved file path.
     */
    async downloadFileWithRetry(url, localPath) {
       // ... axios request ...
       const contentType = response.headers['content-type'];
       const ext = mime.extension(contentType); // Use a mime-types library or map
       
       if (!path.extname(localPath) && ext) {
          localPath += `.${ext}`;
       }
       // ... save file ...
       return localPath;
    }
    ```

### Phase 3: Robust Page ID Derivation (Medium)

**Objective**: Ensure stable and unique Page IDs.

**Component**: `src/orchestration/GlobalQueueManager.js`

**Proposed Changes**:
Normalize URLs before ID extraction to prevent duplicates caused by query parameters or trailing slashes.

```javascript
/**
 * Derives a stable Page ID from a URL.
 * @private
 * @param {string} url - The raw URL.
 * @returns {string} The 32-char ID or normalized path.
 */
_derivePageId(url) {
  try {
    const urlObj = new URL(url);
    // Strip query params and trailing slash
    let cleanPath = urlObj.pathname.replace(/\/$/, '');
    
    // 1. Try strict 32-char hex ID match
    const match = cleanPath.match(/([a-f0-9]{32})$/i);
    if (match) return match[1];
    
    // 2. Fallback: Use the clean path itself (or a hash of it)
    return cleanPath;
  } catch (e) {
    return url; // Last resort
  }
}
```

### Phase 4: Dashboard Title Quality (UX)

**Objective**: Prevent "non-sense" file-like strings (e.g., `Lab_Session_1...pdf_lab_1...`) from appearing as page titles in the dashboard.

**Analysis**:
The issue arises because `PageContext` sanitizes the initial URL/LinkText into a "safe filename" format immediately. When the dashboard displays this *before* the page is scraped (and the true title found), it shows this ugly sanitized string.

**Components**:
-   `src/domain/PageContext.js`
-   `src/ui/TerminalDashboard.js`

**Proposed Changes**:

1.  **Preserve Raw Title**:
    Store the `rawTitle` (link text) separately from the sanitized `title` (filename).

    ```javascript
    // src/domain/PageContext.js
    constructor(url, rawTitle, ...) {
      this.rawTitle = rawTitle || 'Untitled'; // Keep original text
      this.title = FileSystemUtils.sanitizeFilename(rawTitle); // Filesystem safe
      // ...
    }
    ```

2.  **Smart Dashboard Display**:
    Update the dashboard event handlers to prefer `TitleRegistry` > `PageContext.rawTitle` > `PageContext.title`.

    ```javascript
    // In ClusterOrchestrator or DashboardController
    const displayTitle = titleRegistry.get(pageId) || pageContext.rawTitle || 'Discovering...';
    ```

3.  **Truncation**:
    Ensure titles in the dashboard are truncated to a reasonable length (e.g., 50 chars) to prevent layout breaking.

## 3. Architecture Documentation Updates

The `Docs/ARCHITECTURE.md` file must be updated to reflect these changes:

1.  **Graph Visualization**: Add a note in the "Master Process" section about the `UserConfirmationPhase` handling cyclic graphs via path-based visited tracking.
2.  **Data Integrity**: Update the "Canonicalization Invariant" section to describe the new, robust ID derivation strategy (URL normalization).
3.  **File Handling**: Update the "Worker Process" section to mention "Header-Based Extension Detection" in the `FileDownloader` component.
