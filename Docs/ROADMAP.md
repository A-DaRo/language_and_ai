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
| **Micro-Kernel Phase 1** | **Infrastructure Layer** | ✅ **Completed** | Protocol definitions and event bus for Master-Worker IPC. |
| **Micro-Kernel Phase 2** | **Domain Serialization** | ✅ **Completed** | PageContext refactor for JSON serialization. |
| **Micro-Kernel Phase 3** | **Worker Implementation** | ✅ **Completed** | Worker process entry point and task execution. |
| **Micro-Kernel Phase 4** | **Cluster Management** | ✅ **Completed** | Worker pool and resource management. |
| **Micro-Kernel Phase 5** | **Orchestration Logic** | ✅ **Completed** | State machine and queue management. |
| **Micro-Kernel Phase 6** | **Integration** | ✅ **Completed** | Main entry point and system wiring. |
| **Micro-Kernel Phase 7** | **Verification** | ✅ **Completed** | Testing, cleanup, documentation complete. |

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

### 5. Micro-Kernel Architecture (New)
*   ✅ **Phase 1 - Infrastructure Layer**:
    *   ✅ **`src/core/ProtocolDefinitions.js`**: IPC protocol with MESSAGE_TYPES, payload typedefs, error serialization helpers.
    *   ✅ **`src/core/SystemEventBus.js`**: Singleton EventEmitter for Master process coordination with typed events.
*   ✅ **Phase 2 - Domain Serialization**:
    *   ✅ **`src/domain/PageContext.js`**: Refactored with id, parentId, childIds, toJSON(), fromJSON(), targetFilePath for IPC transfer.
    *   ✅ Backward compatibility verified with existing dry-run tests.
*   ✅ **Phase 3 - Worker Layer**:
    *   ✅ **`src/worker/WorkerEntrypoint.js`**: Isolated worker process entry point with browser initialization.
    *   ✅ **`src/worker/TaskRunner.js`**: IPC command router for discovery and download tasks.
    *   ✅ Tested worker process independently - spawns, initializes browser, sends READY signal.
*   ✅ **Phase 4 - Cluster Layer**:
    *   ✅ **`src/cluster/WorkerProxy.js`**: Master-side handle for worker IPC communication and state tracking.
    *   ✅ **`src/cluster/BrowserInitializer.js`**: Resource-aware worker spawning with capacity planning.
    *   ✅ **`src/cluster/BrowserManager.js`**: Worker pool management with idle/busy allocation.
*   ✅ **Phase 5 - Orchestration Layer**:
    *   ✅ **`src/orchestration/GlobalQueueManager.js`**: Centralized queue manager for discovery and download phases.
    *   ✅ **`src/orchestration/analysis/ConflictResolver.js`**: Duplicate detection and canonical path resolution.
    *   ✅ **`src/orchestration/ClusterOrchestrator.js`**: Main state machine with 5-phase workflow (Bootstrap, Discovery, Conflict Resolution, Download, Complete).
*   ✅ **Phase 6 - Integration**:
    *   ✅ **`main-cluster.js`**: New entry point for distributed cluster mode.
    *   ✅ Graceful shutdown handlers for SIGINT/SIGTERM with worker cleanup.
    *   ✅ System initialization via SystemEventBus.
    *   ✅ Command-line argument parsing (--max-depth, --help).
*   ✅ **Phase 7 - Verification**:
    *   ✅ **`test-integration.js`**: Comprehensive test suite with 6 test cases covering all 5 workflow phases.
    *   ✅ **`test-worker-crash.js`**: Fault tolerance testing - verifies system handles worker crashes gracefully.
    *   ✅ **`test-cookie-propagation.js`**: Authentication testing - verifies cookie capture and broadcast.
    *   ✅ **Legacy mode verification**: Tested main.js with --dry-run, all features working correctly.
    *   ✅ **Documentation updates**: README.md, QUICKSTART.md updated with cluster mode; created CLUSTER_MODE.md.
    *   ✅ **npm scripts**: Added start:cluster, test:integration, test:crash, test:cookies.
    *   ✅ **Dependency cleanup**: Marked puppeteer-cluster for deprecation (kept for legacy compatibility).
    *   ✅ See `Docs/PHASE_7_SUMMARY.md` for detailed verification report.

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
*   [ ] **Worker Auto-Respawn**: Automatically replace crashed workers during execution.

### Medium Priority
*   [ ] **Config Validation**: Add schema validation for the `Config` class using `joi` or `zod`.
*   [ ] **Interactive CLI**: Enhance the CLI with a progress bar (e.g., `cli-progress`) instead of raw log output.
*   [ ] **Docker Support**: Create a `Dockerfile` and `docker-compose.yml` for containerized execution.

### Low Priority
*   [ ] **Plugin System**: Allow users to write custom extractors or processors.
*   [ ] **Web UI**: Build a simple React/Vue dashboard to monitor the scrape in real-time.
