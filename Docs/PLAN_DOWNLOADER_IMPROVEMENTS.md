# Plan: Intelligent Downloader System Improvements

## 1. Problem Statement
The current downloader system has limitations in identifying and handling specific types of resources:
1.  **Hidden Files:** Some files (e.g., PDFs) are embedded in `div` elements without direct `href` links, requiring interaction to reveal the download URL.
2.  **Unwanted Files:** The system currently downloads `.csv` files, which should be excluded.
3.  **Restricted Domains:** Certain domains (e.g., `surf.nl`) are inaccessible due to firewalls and should be blacklisted to prevent timeouts and errors.
4.  **Ambiguous Resources:** Links without clear file extensions (e.g., LinkedIn profile photos) consume resources but don't yield useful files.

## 2. Proposed Solution
We will enhance the `FileDownloader` and `FileTypeDetector` components to implement "intelligent" downloading behavior. This involves active interaction with the page to discover hidden links, stricter filtering logic, and domain blacklisting.

### 2.1 Architecture Changes
The changes will be localized to the `src/download/` package, specifically within the `FileDownloader` and its helper classes. This respects the separation of concerns, keeping the scraping logic in `PageProcessor` unaware of the low-level download details.

## 3. Detailed Design

### 3.1 Enhanced FileTypeDetector (`src/download/file/FileTypeDetector.js`)
We will expand the detector to support blacklisting and stricter validation.

*   **New Configuration:**
    *   `blacklistedDomains`: Array of domains to skip (e.g., `['surf.nl', 'surfdrive.surf.nl']`).
    *   `blacklistedExtensions`: Array of extensions to skip (e.g., `['.csv']`).
*   **Logic Updates:**
    *   `isDownloadableFile(url, text)`:
        *   Check against `blacklistedDomains`.
        *   Check against `blacklistedExtensions`.
        *   If URL has no extension and doesn't match known patterns (like Notion secure files), reject it (to skip ambiguous resources like LinkedIn images).

### 3.2 Hidden File Discovery (`src/download/FileDownloader.js`)
We will add a new phase to the `downloadAndRewriteFiles` method to handle non-anchor elements.

*   **New Method:** `_processHiddenFiles(page, outputDir)`
*   **Target Selector:** `div[data-popup-origin="true"]` (and potentially others in the future).
*   **Interaction Logic:**
    1.  Identify potential file elements based on text content (e.g., "slides.pdf").
    2.  **Click-and-Intercept Strategy:**
        *   Since the link is not in the DOM, we must trigger the download action.
        *   We will set up a request interceptor on the Puppeteer page.
        *   Click the element.
        *   Capture the resulting request URL.
        *   Abort the browser's navigation/download request (to keep control).
        *   Pass the captured URL to the `FileDownloadStrategy` for handling.
    3.  **DOM Rewriting:**
        *   Replace the opaque `div` with a standard `a` tag pointing to the downloaded local file. This ensures the offline replica is navigable.

### 3.3 Domain Blacklisting
*   The `FileTypeDetector` will be the central place for this logic, ensuring that both `FileDownloader` and potentially `AssetDownloader` (if needed in the future) can respect these rules.

## 4. Implementation Plan

### Step 1: Update `FileTypeDetector.js`
- Add `blacklistedDomains` and `blacklistedExtensions` arrays.
- Implement `isBlacklisted(url)` method.
- Refine `isDownloadableFile` to use these checks and handle extension-less URLs strictly.

### Step 2: Update `FileDownloader.js`
- Implement `_processHiddenFiles` method.
- Add Puppeteer request interception logic to capture URLs from clicks.
- Integrate this new phase into the main `downloadAndRewriteFiles` workflow.
- Implement DOM replacement logic for processed hidden files.

### Step 3: Verification
- Verify that `.csv` files are skipped.
- Verify that `surf.nl` links are ignored.
- Verify that "week1_inperson_slides.pdf" is correctly identified, downloaded, and linked in the final HTML.

## 5. Risk Assessment
*   **Interception Stability:** Request interception can sometimes interfere with page functionality. We must ensure it's enabled only briefly during the hidden file processing phase.
*   **New Tabs:** If clicking the element opens a new tab, simple request interception might miss it. We will need to listen for `targetcreated` events as a fallback.

