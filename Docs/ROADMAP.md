# Project Roadmap

This document tracks the progress of the Notion Recursive Scraper refactoring and development. It compares the original refactoring plan with the current state of the codebase and outlines future improvements.

## Status Overview

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| **Phase 1** | **Structure Setup** | ✅ **Completed** | New directory structure (`src/core`, `src/domain`, etc.) established. |
| **Phase 2** | **Utility Extraction** | ✅ **Completed** | `FileSystemUtils` and `IntegrityAuditor` extracted to `src/utils`. |
| **Phase 3** | **Core & Domain Migration** | ✅ **Completed** | `Config`, `Logger`, `StateManager`, `PageContext` migrated and updated. |
| **Phase 4** | **Component Splitting** | ✅ **Completed** | `PageScraper` split into `PageProcessor` and `LinkRewriter`. `NotionScraper` refactored into Facade + `RecursiveScraper`. |
| **Phase 5** | **JSDoc Enforcement** | ✅ **Completed** | All classes and methods documented with JSDoc. |
| **Phase 6 & 7** | **LOC Limit & JSDoc Refinement** | 📝 **Planned** | Split oversized modules and refine JSDoc for all classes and methods. |
| **Phase 8** | **Method Refactoring** | 📝 **Planned** | Reduce long class methods into smaller, internal methods. |

## Detailed Progress

### 1. Orchestration Layer
*   ✅ **`NotionScraper.js`**: Successfully refactored into a Facade. It now delegates logic to `RecursiveScraper` and `PageProcessor`.
*   ✅ **`RecursiveScraper.js`**: Implements the Strict BFS strategy and handles edge classification (Tree, Back, Forward, Cross).
*   ✅ **`DiscoveryService` / `ExecutionService`**: Merged into `RecursiveScraper` to keep the logic cohesive while maintaining separation from the Facade.

### 2. Scraping Layer
*   ✅ **`PageProcessor.js`**: Focused solely on Puppeteer interactions (navigation, expansion, saving). Link rewriting logic removed.
*   ✅ **`LinkRewriter.js`**: Created in `src/processing` to handle offline HTML manipulation using JSDOM.

### 3. Extraction & Download Layer
*   ✅ **`LinkExtractor.js`**: Logic for extracting links and detecting sections/subsections is isolated.
*   ✅ **`AssetDownloader.js`**: Focused on images. Filename sanitization moved to `FileSystemUtils`.
*   ✅ **`CssDownloader.js`**: Handles CSS downloading and asset localization.
*   ✅ **`FileDownloader.js`**: Handles embedded file downloads (PDFs, zips, etc.).

### 4. Domain Layer
*   ✅ **`PageContext.js`**: Remains the single source of truth for the site graph.

## Future Improvements (Backlog)

This section outlines the planned refactoring for upcoming phases and other future work.

### Phase 6 & 7: LOC Limit Enforcement and JSDoc Refinement

This phase combines module splitting with a comprehensive JSDoc review to ensure code is both modular and well-documented.

1.  **Module Splitting (LOC Limit)**:
    *   **`NotionScraper.js`**:
        *   `src/orchestration/ui/PlanDisplayer.js` (New): Extract `displayTree` and `_printTreeNode`.
        *   `src/orchestration/ui/StatisticsDisplayer.js` (New): Extract `_printStatistics` and `_printHierarchy`.
        *   `src/orchestration/ui/UserPrompt.js` (New): Extract `_promptForPlanDecision`.
    *   **`RecursiveScraper.js`**:
        *   `src/orchestration/analysis/GraphAnalyzer.js` (New): Extract all edge classification logic (`_classifyEdge`, `_logEdgeStatistics`, etc.).
    *   **`CssDownloader.js`**:
        *   `src/download/css/CssContentProcessor.js` (New): Extract CSS parsing logic (`_processCssContent`, `_rewriteImports`, `_rewriteAssetUrls`).
        *   `src/download/css/CssAssetDownloader.js` (New): Extract CSS asset download logic (`_downloadCssAsset`).

2.  **JSDoc Refinement**:
    *   **Goal**: Ensure every class, method, and property is documented clearly and consistently, adhering to the standards outlined in `Docs/JSDocs.md`.
    *   **Class-Level Documentation**:
        *   Use `@classdesc` to provide a detailed explanation of the class's role and responsibilities.
        *   Use `@see` to link to related classes (e.g., `NotionScraper` should `@see RecursiveScraper`).
    *   **Method-Level Documentation**:
        *   Write a concise `@summary` for each method.
        *   Clearly define API contracts with `@param`, `@returns`, and `@throws`. Ensure types are specific (e.g., `{Promise<string>}` instead of just `{Promise}`).
        *   Use `@example` for complex methods to demonstrate usage.
    *   **API Visibility**:
        *   Mark helper methods intended for internal use with `@private` or `@protected`.
        *   For methods created during refactoring (Phase 9), use `@internal` to signify they are not part of the stable public API of the class, even if they are technically public.
    *   **Code Maintenance**:
        *   Use `@todo` to flag areas that require future attention directly in the source code.

### Phase 8: Method Refactoring

To improve readability and maintainability, long methods will be broken down into smaller, single-purpose internal methods, each documented with `@internal`.

*   **Target Method Length**: Aim for methods to be under 75-100 LOC.
*   **Refactoring Candidates**:
    *   **`RecursiveScraper.discover()`**: This large method handles the entire discovery loop.
        *   **Breakdown**: Extract logic for `_discoverLinksForLevel`, `_processDiscoveredLink`, and `_initializeDiscoveryState`.
    *   **`RecursiveScraper.execute()`**: The main execution loop can be simplified.
        *   **Breakdown**: Extract the per-page scraping logic into a helper like `_executeScrapeForPage`.
    *   **`PageProcessor.scrapePage()`**: This method orchestrates navigation, content expansion, asset downloading, and saving.
        *   **Breakdown**: Split into `_navigateAndPreparePage`, `_downloadPageAssets`, and `_savePageContent`.
    *   **`NotionScraper.run()`**: The primary run loop contains complex user interaction logic.
        *   **Breakdown**: Extract the interactive planning loop into `_handleDiscoveryLoop` to separate planning from execution.
    *   **`CssDownloader._downloadCssFile()`**: The retry and processing logic can be separated.
        *   **Breakdown**: Extract the `axios` call and retry mechanism into a dedicated `_fetchWithRetries` method.
    *   **`LinkRewriter.rewriteLinksInFile()`**: This method currently reads the file, processes CSS, and rewrites links.
        *   **Breakdown**: Extract the link iteration and rewriting logic into `_rewriteAnchorTags`.

### High Priority (Post-Phase 8)
*   [ ] **Unit Testing**: Add Jest/Mocha tests for core logic (`PageContext`, `FileSystemUtils`, `LinkRewriter`).
*   [ ] **Error Recovery**: Implement checkpointing to resume interrupted scrapes from the last saved state.
*   [ ] **Parallel Discovery**: Re-introduce parallel discovery (using `puppeteer-cluster`) for the initial scanning phase to speed up large sites.

### Medium Priority
*   [ ] **Config Validation**: Add schema validation for the `Config` class using `joi` or `zod`.
*   [ ] **Interactive CLI**: Enhance the CLI with a progress bar (e.g., `cli-progress`) instead of raw log output.
*   [ ] **Docker Support**: Create a `Dockerfile` and `docker-compose.yml` for containerized execution.

### Low Priority
*   [ ] **Plugin System**: Allow users to write custom extractors or processors.
*   [ ] **Web UI**: Build a simple React/Vue dashboard to monitor the scrape in real-time.
