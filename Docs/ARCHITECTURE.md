# Architecture Documentation

## Overview

The Notion Scraper is a modular, Node.js-based application designed to recursively scrape Notion pages, download assets (images, files, CSS), and rewrite links for offline browsing. The system is architected around a strict separation of concerns, utilizing a package-based structure to organize core logic, domain models, orchestration, and specific processing tasks.

**Architecture Principles:**
- **Single Responsibility**: Each class has one well-defined purpose
- **Separation of Concerns**: Clear boundaries between orchestration, processing, and utilities
- **Dependency Injection**: Components receive dependencies through constructors
- **Immutable Contexts**: PageContext objects represent immutable page state
- **Strict BFS Traversal**: Breadth-first search ensures proper hierarchy and prevents infinite loops

## System Architecture

The application is divided into the following packages:

*   **`src/core`**: Fundamental infrastructure components (Configuration, Logging, State Management).
*   **`src/domain`**: Domain entities representing the data model (PageContext).
*   **`src/orchestration`**: High-level controllers that coordinate the scraping process.
    *   **`src/orchestration/ui`**: User interface components for CLI interaction.
    *   **`src/orchestration/analysis`**: Graph analysis components for edge classification.
*   **`src/scraping`**: Components responsible for interacting with the live Notion pages via Puppeteer.
*   **`src/processing`**: Components that process page content (expanding toggles, handling cookies, rewriting links).
*   **`src/extraction`**: Components for extracting data from pages (Links).
*   **`src/download`**: Components for downloading external resources (Assets, CSS, Files).
    *   **`src/download/css`**: Specialized CSS processing and downloading components.
*   **`src/utils`**: Shared utility functions (FileSystem, Integrity).

## Package Details & API Reference

### 1. Core Package (`src/core`)

Provides the foundational services used throughout the application.

#### `Config.js`
Centralized configuration management.
*   **Methods**:
    *   `getBaseUrl()`: Returns the protocol and host of the target Notion page.
    *   `isNotionUrl(url)`: Checks if a URL belongs to the Notion domain.
    *   `extractPageNameFromUrl(url)`: Parses a Notion URL to extract a human-readable page name.

#### `Logger.js`
Standardized logging utility with support for categories and timestamps.
*   **Methods**:
    *   `info(category, message)`: Logs informational messages.
    *   `success(category, message)`: Logs success messages.
    *   `warn(category, message)`: Logs warnings.
    *   `error(category, message, error)`: Logs errors with stack traces.
    *   `debug(category, message)`: Logs debug info (if enabled).
    *   `separator(message)`: Prints a visual separator.
    *   `getElapsedTime()`: Returns time since logger initialization.

#### `StateManager.js`
Singleton that manages the global state of the discovery process, including the URL queue and the PageContext registry.
*   **Methods**:
    *   `static getInstance()`: Returns the singleton instance.
    *   `reset(rootUrl, rootTitle)`: Resets the state for a new run.
    *   `bootstrap(rootUrl, rootTitle)`: Initializes the state with a root page.
    *   `getContextByUrl(url)`: Retrieves a `PageContext` by its URL.
    *   `registerOrLink(linkInfo, parentContext)`: Registers a new page or links an existing one to a parent.
    *   `getCurrentLevelQueue()`: Returns URLs to be processed in the current BFS level.
    *   `advanceLevel()`: Moves the next level queue to the current level.

### 2. Domain Package (`src/domain`)

Defines the core data structures.

#### `PageContext.js`
Represents a single Notion page within the hierarchical structure. It is the "Single Source of Truth" for file paths and relationships.
*   **Methods**:
    *   `setDisplayTitle(title)`: Updates the display title.
    *   `setSection(section)`: Sets the section metadata.
    *   `setSubsection(subsection)`: Sets the subsection metadata.
    *   `getRelativePath()`: Calculates the relative path from the root based on the parent chain.
    *   `getDirectoryPath(baseDir)`: Returns the absolute directory path for saving the page.
    *   `getFilePath(baseDir)`: Returns the absolute path to the `index.html` file.
    *   `getRelativePathTo(targetContext)`: Calculates the relative file path from this page to another page (for link rewriting).
    *   `addChild(childContext)`: Adds a child page to the hierarchy.

### 3. Orchestration Package (`src/orchestration`)

Coordinates the workflow of the application.

#### `NotionScraper.js`
Main orchestrator for Notion scraping. Coordinates the initialization, planning, and execution of the scraping process.
*   **Constructor**: `new NotionScraper(config)`
*   **Dependencies**: `PlanDisplayer`, `StatisticsDisplayer`, `UserPrompt`, `RecursiveScraper`
*   **Methods**:
    *   `initialize()`: Initialize browser and page.
    *   `run(options)`: Run the complete scraping process.
        *   `options`: `{ dryRunOnly: boolean, autoConfirm: boolean, initialMaxDepth: number }`
    *   `plan(maxDepth)`: Plan the scraping by discovering pages.
        *   `maxDepth`: Maximum depth to discover.
        *   **Returns**: `Promise<Object>` (The discovery plan).
    *   `cleanup()`: Cleanup resources (close browser).

#### `RecursiveScraper.js`
Orchestrates recursive scraping of Notion pages using strict BFS traversal with edge classification.
*   **Constructor**: `new RecursiveScraper(config, logger, pageProcessor, linkRewriter)`
*   **Dependencies**: `GraphAnalyzer`, `PageProcessor`, `LinkRewriter`
*   **Edge Classification**: Classifies discovered links as tree, back, forward, or cross edges for graph analysis.
*   **Methods**:
    *   `discover(page, rootUrl, maxDepth)`: Discovery phase - build the PageContext tree without heavy downloads.
        *   `page`: Puppeteer page instance.
        *   `rootUrl`: The starting URL.
        *   `maxDepth`: Maximum recursion depth.
        *   **Returns**: `Promise<{rootContext: PageContext, allContexts: PageContext[]}>` (Discovery results).
    *   `execute(page, rootContext)`: Execution phase - traverse planned tree and run full scraping routine.
        *   `page`: Puppeteer page instance.
        *   `rootContext`: The root context from discovery.
        *   **Returns**: `Promise<{rootContext: PageContext, totalLinksRewritten: number, allContexts: PageContext[]}>` (Execution results).
    *   `_processLevelNodes(page, currentLevel, discovered, discoveryOrder)`: Process all nodes at current BFS level. *(Private)*
    *   `_processLink(linkInfo, context, discovered, discoveryOrder)`: Process a single link during discovery. *(Private)*
    *   `_collectContexts(rootContext)`: Collect all contexts from tree using DFS. *(Private)*

#### Orchestration UI Subpackage (`src/orchestration/ui`)

User interface components extracted for clean separation of concerns.

##### `PlanDisplayer.js`
Displays ASCII tree visualization of discovered page hierarchy.
*   **Constructor**: `new PlanDisplayer(logger)`
*   **Methods**:
    *   `displayTree(rootContext, totalPages)`: Display the complete page tree with statistics.
        *   `rootContext`: Root PageContext to visualize.
        *   `totalPages`: Total number of pages discovered.
    *   `_printTreeNode(context, prefix, isLast, isRoot)`: Recursively print tree nodes with proper formatting. *(Private)*

##### `StatisticsDisplayer.js`
Displays post-execution statistics and hierarchy information.
*   **Constructor**: `new StatisticsDisplayer(logger)`
*   **Methods**:
    *   `printStatistics(rootContext, totalLinksRewritten, allContexts, elapsedTime)`: Print comprehensive execution statistics.
        *   `rootContext`: Root PageContext.
        *   `totalLinksRewritten`: Total number of rewritten links.
        *   `allContexts`: Array of all PageContext objects.
        *   `elapsedTime`: Total execution time in milliseconds.
    *   `_printHierarchy(context, indent)`: Recursively print page hierarchy. *(Private)*

##### `UserPrompt.js`
Handles interactive CLI prompting for plan confirmation.
*   **Constructor**: `new UserPrompt(logger)`
*   **Methods**:
    *   `promptForPlanDecision()`: Prompt user to proceed with or cancel the discovered plan.
        *   **Returns**: `Promise<'yes'|'no'>` (User decision).

#### Orchestration Analysis Subpackage (`src/orchestration/analysis`)

Graph analysis components for BFS edge classification.

##### `GraphAnalyzer.js`
Analyzes and classifies edges in the page graph during BFS traversal.
*   **Constructor**: `new GraphAnalyzer(logger)`
*   **Edge Types**:
    *   **Tree edges**: Parent → Child (first discovery)
    *   **Back edges**: To ancestor or self-loop (creates cycles)
    *   **Forward edges**: Skip levels to descendant (shortcuts)
    *   **Cross edges**: Between branches or same level
*   **Methods**:
    *   `classifyEdge(fromContext, toContext, toUrl, linkTitle)`: Classify an edge when encountering an already-discovered node.
        *   **Returns**: `'tree'|'back'|'forward'|'cross'` (Edge type).
    *   `addTreeEdge(fromUrl, toUrl, title)`: Add a tree edge to classification.
    *   `logEdgeStatistics(level)`: Log edge classification statistics for current BFS level.
    *   `logFinalEdgeStatistics()`: Log final edge classification summary with cycle detection.
    *   `resetEdgeClassification()`: Reset all edge classification arrays.
    *   `getStats()`: Get current edge statistics.
        *   **Returns**: `{tree: number, back: number, forward: number, cross: number}` (Edge counts).
    *   `_isAncestor(fromContext, targetUrl)`: Check if targetUrl is an ancestor of fromContext. *(Private)*

### 4. Scraping Package (`src/scraping`)

Handles the direct interaction with Puppeteer for page navigation and saving.

#### `PageProcessor.js`
Handles the scraping of individual Notion pages. Coordinates navigation, content expansion, link extraction, and asset downloads.
*   **Constructor**: `new PageProcessor(config, logger, cookieHandler, contentExpander, linkExtractor, assetDownloader, fileDownloader)`
*   **State Management**: 
    *   Maintains `visitedUrls` Set to prevent duplicate processing
    *   Manages `urlToContextMap` for link rewriting coordination
*   **Methods**:
    *   `scrapePage(page, pageContext, isFirstPage)`: Scrape a single page and return its links.
        *   `page`: Puppeteer page instance.
        *   `pageContext`: Context of the page to scrape.
        *   `isFirstPage`: Whether this is the first page (for cookie handling).
        *   **Returns**: `Promise<Array<Object>>` (List of discovered link objects).
        *   **Workflow**: Navigate → Handle cookies → Expand content → Extract links → Download assets → Save HTML.
    *   `discoverPageInfo(page, url, isFirstPage)`: Lightweight metadata fetch used during discovery.
        *   `page`: Puppeteer page instance.
        *   `url`: URL to discover.
        *   `isFirstPage`: Whether this is the first page.
        *   **Returns**: `Promise<{title: string, links: Array<Object>}>` (Page metadata).
        *   **Note**: Uses 'domcontentloaded' for speed, skips heavy downloads.
    *   `registerPageContext(url, context)`: Register a page context for link rewriting.
    *   `getContextMap()`: Get the map of registered page contexts.
        *   **Returns**: `Map<string, PageContext>` (URL → PageContext mapping).
    *   `isUrlRegistered(url)`: Check if a URL is already registered.
    *   `resetVisited()`: Reset visited state before fresh execution phase.
    *   `resetContextMap()`: Reset the page context map prior to discovery.
    *   `_savePageHtml(page, pageContext)`: Save page HTML to disk. *(Private)*

### 5. Processing Package (`src/processing`)

Manipulates page content and handles specific page interactions.

#### `LinkRewriter.js`
Handles offline link rewriting and CSS localization for scraped pages. Transforms online-dependent content to self-contained offline structure.
*   **Constructor**: `new LinkRewriter(config, logger, cssDownloader)`
*   **Methods**:
    *   `rewriteLinksInFile(pageContext, urlToContextMap)`: Rewrite all internal links in saved HTML files to point to local paths.
        *   `pageContext`: The context of the page to rewrite.
        *   `urlToContextMap`: Map of URLs to PageContexts.
        *   **Returns**: `Promise<number>` (Number of internal links rewritten).
        *   **Workflow**: Load HTML → Identify internal links → Calculate relative paths → Rewrite hrefs → Download CSS → Save.

#### `ContentExpander.js`
Expands toggles, databases, and other collapsible content on Notion pages using aggressive iterative expansion.
*   **Constructor**: `new ContentExpander(config, logger)`
*   **Expansion Strategy**: Multi-iteration click-and-wait cycles to reveal nested content.
*   **Methods**:
    *   `expandAll(page)`: Expand all content on the page.
        *   `page`: Puppeteer page instance.
        *   **Workflow**: Scroll to bottom (lazy-loading) → Iterative toggle expansion.
    *   `_scrollToBottom(page)`: Scroll to the bottom to trigger lazy-loading. *(Private)*
    *   `_expandToggles(page)`: Aggressively expand toggles, buttons, and interactive elements. *(Private)*
        *   **Target Elements**: aria-expanded="false", .notion-toggle-block, buttons, expandable divs.
        *   **Safety**: Excludes destructive actions (delete, remove, share).

#### `CookieHandler.js`
Handles cookie consent banners on Notion pages with intelligent retry and deduplication.
*   **Constructor**: `new CookieHandler(config, logger)`
*   **State Management**: Uses WeakSet to track processed pages (auto garbage-collected).
*   **Methods**:
    *   `ensureConsent(page, label)`: Public entry point that safely handles the cookie banner once per page context.
        *   `page`: Puppeteer page instance.
        *   `label`: Optional label for logging.
        *   **Returns**: `Promise<boolean>` (True if banner was found and handled).
        *   **Features**: Configurable retries, exponential backoff, per-page tracking.
    *   `handle(page, label)`: Backwards-compatibility shim. *(Deprecated)*
    *   `_attemptHandle(page)`: Attempt to handle cookie banner once. *(Private)*
    *   `_isPageUsable(page)`: Check if page is still open and usable. *(Private)*

### 6. Extraction Package (`src/extraction`)

Extracts structured data from the DOM.

#### `LinkExtractor.js`
Extracts and categorizes links from Notion pages with hierarchical context metadata.
*   **Constructor**: `new LinkExtractor(config, logger)`
*   **Extraction Features**:
    *   Filters for internal Notion links (same domain)
    *   Resolves relative URLs to absolute URLs
    *   Deduplicates discovered links
    *   Extracts hierarchical context (sections, subsections) from DOM structure
    *   Excludes self-references and invalid URLs
*   **Methods**:
    *   `extractLinks(page, currentUrl, baseUrl)`: Extract all internal Notion page links with their hierarchical context.
        *   `page`: Puppeteer page object.
        *   `currentUrl`: Current page URL to filter out (prevent self-references).
        *   `baseUrl`: Optional base URL (defaults to config base URL).
        *   **Returns**: `Promise<Array<{url: string, title: string, section?: string, subsection?: string}>>` (Link objects with metadata).

### 7. Download Package (`src/download`)

Manages the retrieval of external resources.

#### `AssetDownloader.js`
Downloads and manages assets (images, files, etc.) with retry logic and content-based hashing.
*   **Constructor**: `new AssetDownloader(config, logger)`
*   **Features**:
    *   Content-based MD5 hashing for unique filenames
    *   Download cache to prevent duplicate requests
    *   Retry logic with exponential backoff
    *   Extracts from both img tags and CSS background properties
*   **Methods**:
    *   `downloadAndRewriteImages(page, outputDir)`: Download all images and assets from a page and rewrite their paths.
        *   `page`: Puppeteer page instance.
        *   `outputDir`: Directory to save images to (images/ subdirectory created).
        *   **Workflow**: Extract URLs → Download with retries → Generate safe filenames → Rewrite DOM references.
    *   `_downloadAsset(url, outputPath)`: Download a single asset with retry logic. *(Private)*
    *   `_generateFilename(url, content)`: Generate safe filename using content hash. *(Private)*

#### `CssDownloader.js`
Coordinator for CSS localization. Downloads external CSS stylesheets and rewrites `<link>` tags. Delegates to specialized components.
*   **Constructor**: `new CssDownloader(config, logger)`
*   **Dependencies**: `CssContentProcessor`, `CssAssetDownloader`
*   **Architecture**: Facade pattern - coordinates CSS processing and asset downloads.
*   **Methods**:
    *   `downloadAndRewriteCss(dom, pageDir, pageUrl)`: Download all CSS files linked in the HTML and rewrite the DOM.
        *   `dom`: JSDOM instance.
        *   `pageDir`: Directory of the HTML file.
        *   `pageUrl`: Original URL of the page.
        *   **Returns**: `Promise<{stylesheets: number, assets: number}>` (Stats on stylesheets and assets).
        *   **Workflow**: Extract stylesheets → Download CSS → Process content → Download assets → Rewrite links.
    *   `_downloadCssFile(cssUrl, outputDir, pageUrl)`: Download and process a single CSS file. *(Private)*
    *   `_cssFileExists(filepath)`: Check if CSS file already downloaded. *(Private)*

#### `FileDownloader.js`
Downloads embedded files (PDFs, code files, documents, etc.) from Notion pages.
*   **Constructor**: `new FileDownloader(config, logger)`
*   **Supported File Types**:
    *   Documents: PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX
    *   Archives: ZIP, RAR, 7Z, TAR, GZ
    *   Code: PY, JS, TS, JAVA, CPP, C, H, IPYNB
    *   Data: JSON, XML, CSV, TXT, MD
    *   Media: MP4, AVI, MOV, MP3, WAV
*   **Methods**:
    *   `downloadAndRewriteFiles(page, outputDir)`: Download all embedded files from a page and rewrite their links.
        *   `page`: Puppeteer page instance.
        *   `outputDir`: Directory to save files to (files/ subdirectory created).
    *   `isDownloadableFile(url, linkText)`: Check if a URL is a downloadable file.
        *   **Detection**: Matches against file extensions and Notion file URL patterns.
    *   `_downloadFile(url, outputPath)`: Download a single file with retry logic. *(Private)*

#### Download CSS Subpackage (`src/download/css`)

Specialized CSS processing components extracted for clean separation.

##### `CssContentProcessor.js`
Handles CSS parsing and URL rewriting for @import rules and asset references.
*   **Constructor**: `new CssContentProcessor(config, logger)`
*   **Methods**:
    *   `processCssContent(cssContent, cssUrl, baseUrl)`: Process CSS content and rewrite URLs.
        *   `cssContent`: Raw CSS content string.
        *   `cssUrl`: URL of the CSS file.
        *   `baseUrl`: Base URL for relative resolution.
        *   **Returns**: `{processedCss: string, assetUrls: Array<string>}` (Rewritten CSS and extracted asset URLs).
        *   **Features**: Rewrites @import statements, rewrites asset URLs (url(...)), preserves data URIs.
    *   `rewriteImports(cssContent, cssUrl)`: Rewrite @import statements to local paths. *(Private)*
    *   `rewriteAssetUrls(cssContent, cssUrl, baseUrl)`: Rewrite asset URLs in CSS. *(Private)*

##### `CssAssetDownloader.js`
Downloads CSS-referenced assets (fonts, images) with retry logic and exponential backoff.
*   **Constructor**: `new CssAssetDownloader(config, logger)`
*   **Features**:
    *   Exponential backoff retry strategy (configurable max retries)
    *   Request timeout handling
    *   Safe filename generation with content hashing
*   **Methods**:
    *   `downloadCssAsset(assetUrl, outputDir)`: Download a CSS asset with retry logic.
        *   `assetUrl`: URL of the asset to download.
        *   `outputDir`: Directory to save asset to.
        *   **Returns**: `Promise<string|null>` (Local filename or null on failure).
    *   `_fetchWithRetries(url, attempt)`: Fetch URL with exponential backoff retry. *(Private)*
    *   `_generateSafeFilename(url)`: Generate safe filename from URL. *(Private)*

### 8. Utils Package (`src/utils`)

Shared helpers.

#### `FileSystemUtils.js`
Centralized logic for file naming.
*   **Methods**:
    *   `static sanitizeFilename(filename)`: Converts a string into a safe filename.

#### `IntegrityAuditor.js`
Performs a post-scrape integrity audit.
*   **Constructor**: `new IntegrityAuditor(config, logger)`
*   **Methods**:
    *   `audit(contexts)`: Audit all saved page contexts.
        *   `contexts`: Array of PageContext objects.
        *   **Returns**: `Promise<Object>` (Audit results including missing files and residual links).

---

## Workflow and Data Flow

### Two-Phase Scraping Architecture

The scraper implements a strict two-phase approach to ensure efficiency and proper hierarchy:

#### Phase 1: Discovery (Lightweight)
**Purpose**: Build the complete PageContext tree without heavy downloads.

**Process**:
1. **Initialization**: Create root PageContext, reset state
2. **BFS Traversal**: Strict level-by-level processing
   - Navigate to page (domcontentloaded only)
   - Extract page title and links
   - Classify edges (tree/back/forward/cross)
   - Register new PageContext objects
   - Build parent-child relationships
3. **Output**: Complete page hierarchy, edge statistics

**Key Characteristics**:
- Fast navigation (no networkidle wait)
- No asset downloads
- No content expansion
- Edge classification for cycle detection
- Builds urlToContextMap for link rewriting

#### Phase 2: Execution (Full Scraping)
**Purpose**: Traverse the planned tree and perform complete page capture.

**Process**:
1. **Preparation**: Reset visited state, validate contexts
2. **BFS Traversal**: Follow tree edges only (ignore back/forward/cross)
   - Navigate to page (networkidle0)
   - Handle cookie consent
   - Expand all content
   - Download assets (images, files)
   - Save HTML to disk
3. **Link Rewriting**: Post-processing to enable offline browsing
   - Load saved HTML files
   - Rewrite internal links to relative paths
   - Download and localize CSS
4. **Output**: Complete offline-ready site

**Key Characteristics**:
- Heavy operations (full page load, asset downloads)
- Follows only tree edges (prevents duplicate work)
- Preserves BFS order for proper depth tracking
- Post-processing link rewriting phase

### Data Flow Diagram

```
User Input (URL, Depth)
        ↓
NotionScraper.run()
        ↓
    Initialize
    - Create Browser
    - Create Dependencies
        ↓
    ┌─────────────────────────────────────────┐
    │         DISCOVERY PHASE                 │
    └─────────────────────────────────────────┘
        ↓
RecursiveScraper.discover()
    - PageProcessor.discoverPageInfo()
    - LinkExtractor.extractLinks()
    - GraphAnalyzer.classifyEdge()
        ↓
    PageContext Tree Built
    (urlToContextMap populated)
        ↓
    PlanDisplayer.displayTree()
    UserPrompt.promptForPlanDecision()
        ↓
    ┌─────────────────────────────────────────┐
    │         EXECUTION PHASE                 │
    └─────────────────────────────────────────┘
        ↓
RecursiveScraper.execute()
    ↓
PageProcessor.scrapePage() (for each page)
    - CookieHandler.ensureConsent()
    - ContentExpander.expandAll()
    - LinkExtractor.extractLinks()
    - AssetDownloader.downloadAndRewriteImages()
    - FileDownloader.downloadAndRewriteFiles()
    - Save HTML to disk
        ↓
LinkRewriter.rewriteLinksInFile() (for each page)
    - Load saved HTML
    - Rewrite internal links
    - CssDownloader.downloadAndRewriteCss()
        - CssContentProcessor.processCssContent()
        - CssAssetDownloader.downloadCssAsset()
    - Save modified HTML
        ↓
    ┌─────────────────────────────────────────┐
    │         COMPLETION                      │
    └─────────────────────────────────────────┘
        ↓
StatisticsDisplayer.printStatistics()
IntegrityAuditor.audit()
NotionScraper.cleanup()
```

### Edge Classification System

During discovery, the GraphAnalyzer classifies each discovered link:

- **Tree Edge** (Green): First discovery, becomes parent-child relationship
  - Action: Create new PageContext, add to queue
  
- **Back Edge** (Red): Points to ancestor or self
  - Indicates: Cycle in page graph
  - Action: Log warning, do not queue
  - Example: Child page linking back to parent
  
- **Forward Edge** (Blue): Skip levels to descendant
  - Indicates: Shortcut in hierarchy
  - Action: Log, do not queue (already discovered)
  - Example: Parent linking to grandchild
  
- **Cross Edge** (Yellow): Between branches or same level
  - Indicates: Connection between sibling branches
  - Action: Log, do not queue (already discovered)
  - Example: Sibling page references

This classification helps identify potential issues (cycles, missing pages) and provides insights into the page graph structure.

### Directory Structure Output

```
output/
├── index.html                 # Root page
├── images/                    # Root page images
├── css/                       # Root page CSS
├── files/                     # Root page files
├── Child_Page_1/
│   ├── index.html
│   ├── images/
│   ├── css/
│   ├── files/
│   └── Grandchild_Page/
│       ├── index.html
│       ├── images/
│       ├── css/
│       └── files/
└── Child_Page_2/
    ├── index.html
    ├── images/
    ├── css/
    └── files/
```

Each page gets its own directory named after its sanitized title, containing an `index.html` and subdirectories for assets. This mirrors the Notion page hierarchy and enables clean relative path resolution for offline browsing.

---

## Design Patterns

### 1. Facade Pattern
**Used in**: `NotionScraper`, `CssDownloader`
- Provides simplified interface to complex subsystems
- Coordinates multiple components
- Hides internal complexity from callers

### 2. Strategy Pattern
**Used in**: Edge classification, content expansion
- Different algorithms for different scenarios
- GraphAnalyzer classifies edges based on graph structure
- ContentExpander adapts to different page types

### 3. Dependency Injection
**Used throughout**: All class constructors
- Components receive dependencies at construction
- Enables testing and loose coupling
- Clear dependency graphs

### 4. Singleton Pattern
**Used in**: `StateManager` (legacy), WeakSet in `CookieHandler`
- Single instance where needed
- Careful use to avoid global state issues

### 5. Template Method Pattern
**Used in**: Two-phase scraping workflow
- Defines algorithm skeleton (discover → execute)
- Subclasses/phases customize specific steps
- Ensures consistent workflow

---

## Code Quality Standards

### File Size Limits
- **Maximum file size**: 200 lines of code
- **Method size**: <100 LOC (most <50 LOC)
- **Rationale**: Improves readability, maintainability, testability

### Documentation Standards
All classes and methods include:
- `@classdesc`: High-level class purpose with implementation details
- `@summary`: Brief one-line method description
- `@description`: Detailed method behavior and workflow
- `@param` with types: Typed parameter documentation
- `@returns` with types: Return value documentation
- `@throws`: Error conditions
- `@see`: Cross-references to related components
- `@private`: Internal method markers
- `@deprecated`: Backwards-compatibility indicators

### Naming Conventions
- **Classes**: PascalCase (e.g., `PageProcessor`)
- **Methods**: camelCase (e.g., `scrapePage`)
- **Private methods**: Prefixed with `_` (e.g., `_savePageHtml`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- **Files**: Match class name (e.g., `PageProcessor.js`)

### Error Handling
- Try-catch blocks around I/O operations
- Graceful degradation where possible
- Detailed error logging with context
- Retry logic for network operations

---

## Performance Considerations

### Optimization Strategies
1. **Two-phase approach**: Discovery is lightweight (no asset downloads)
2. **BFS traversal**: Ensures proper depth ordering, prevents duplicate work
3. **Caching**: Asset download caches prevent duplicate requests
4. **Parallel operations**: Where safe (within same BFS level)
5. **Lazy loading**: Content expansion triggers Notion's lazy-loading

### Bottlenecks
- **Page navigation**: Notion pages can be slow to load
- **Asset downloads**: Many assets per page, network-dependent
- **CSS processing**: Complex stylesheets with many dependencies
- **Content expansion**: Nested toggles require multiple iterations

### Scaling Considerations
- Current: Single-threaded, sequential page processing
- Future: Could parallelize within BFS levels (careful with rate limits)
- Memory: PageContext tree grows with site size
- Disk I/O: Many small file writes (could batch)
