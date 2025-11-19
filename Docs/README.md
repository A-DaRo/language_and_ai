# Notion Recursive Scraper

A sophisticated, object-oriented Node.js application for recursively scraping Notion pages while maintaining their hierarchical structure.

## Features

- **🎯 Perfect 1:1 Replication**: Downloads complete HTML, CSS, and JavaScript for pixel-perfect offline copies
- **🔗 Full Link Rewriting**: All internal links automatically rewritten for seamless offline navigation
- **📂 Deep Hierarchical Structure**: Creates nested folders that mirror Notion's visual hierarchy
- **♻️ Recursive Scraping**: Automatically follows and downloads all linked Notion pages
- **🖼️ Resilient Asset Downloading**: Advanced handling of complex URLs with retry logic and sanitization
- **🎨 Interactive Preservation**: Maintains all styling, layouts, and interactive elements
- **📋 Content Expansion**: Automatically expands toggles, databases, and collapsible content
- **🍪 Cookie Handling**: Manages cookie consent banners automatically
- **📊 Structured Logging**: Professional logging with categories and timestamps
- **⚙️ Configurable Depth**: Control recursion depth and expansion levels
- **⚡ Distributed Architecture**: Micro-kernel design with Master-Worker processes for scalable parallel scraping
- **🛡️ Fault Tolerance**: Worker isolation ensures system stability even when individual processes crash

## Architecture

The application uses a **Micro-Kernel architecture** with distributed Master-Worker processes:

```
src/
├── core/
│   ├── Config.js                    # Configuration
│   ├── Logger.js                    # Logging
│   ├── ProtocolDefinitions.js       # IPC protocol
│   └── SystemEventBus.js            # Event coordination
├── domain/
│   └── PageContext.js               # Serializable page context
├── worker/
│   ├── WorkerEntrypoint.js          # Worker process entry
│   └── TaskRunner.js                # IPC command router
├── cluster/
│   ├── BrowserManager.js            # Worker pool manager
│   ├── BrowserInitializer.js        # Worker spawning
│   └── WorkerProxy.js               # Master-side worker handle
├── orchestration/
│   ├── ClusterOrchestrator.js       # Main state machine
│   ├── GlobalQueueManager.js        # Task queues
│   └── analysis/
│       └── ConflictResolver.js      # Duplicate detection
└── [processing/download/scraping]   # Reused by workers

main-cluster.js                      # Application entry point
tests/                               # Test suite
```

See `Docs/CLUSTER_MODE.md` for detailed architecture documentation.

## Installation

```bash
npm install
# or manually:
npm install puppeteer axios jsdom
```

## Usage

### Basic Usage

```bash
# Run with default settings
npm start

# Or directly
node main-cluster.js

# With custom depth
node main-cluster.js --max-depth 3

# Show help
node main-cluster.js --help
```

### Features

- **Auto-scaled workers**: Automatically determines optimal worker count based on available RAM (~1GB per worker)
- **Fault tolerance**: Worker crashes don't affect the Master process or other workers
- **Parallel processing**: Multiple workers handle discovery and download phases simultaneously
- **Resource efficient**: Isolated browser processes prevent memory leaks

### Testing

```bash
# Run full integration test suite
npm test

# Run specific tests
npm run test:crash      # Worker crash recovery
npm run test:cookies    # Cookie propagation
```

### Configuration

Edit `src/core/Config.js` to customize settings:

```javascript
this.NOTION_PAGE_URL = 'your-notion-page-url';
this.OUTPUT_DIR = 'downloaded_course_material';
this.MAX_RECURSION_DEPTH = 5;  // Maximum depth for following links
this.MAX_EXPANSION_DEPTH = 3;   // Maximum depth for expanding toggles
```

## Output Structure

The scraper creates a **deeply nested** hierarchical folder structure that mirrors the actual Notion page organization:

```
downloaded_course_material/
├── Main_Page/
│   ├── index.html (all links rewritten to relative paths)
│   └── images/
├── Syllabus/
│   ├── index.html
│   ├── images/
│   ├── Course_Overview/
│   │   ├── index.html
│   │   └── images/
│   └── Grading_Policy/
│       ├── index.html
│       └── images/
└── Material/
    ├── Week_1/
    │   ├── index.html
    │   ├── images/
    │   ├── Introduction/         <-- Nested under Week_1
    │   │   ├── index.html
    │   │   ├── images/
    │   │   └── Subtopic_1/       <-- Deeply nested
    │   │       ├── index.html
    │   │       └── images/
    │   └── Lecture_Notes/
    │       ├── index.html
    │       └── images/
    └── Week_2/
        └── ...
```

**All internal Notion links are automatically rewritten** to relative paths like `../Week_2/Introduction/index.html`, making the entire site fully browsable offline!

## How It Works

The scraper executes in **5 distinct phases** orchestrated by the Master process:

### Phase 1: Bootstrap
1. **Initial Worker Spawn**: Master spawns first worker to capture authentication cookies
2. **Cookie Capture**: Worker navigates to target page and handles cookie consent
3. **Worker Pool Creation**: Master spawns remaining workers based on available RAM
4. **Cookie Broadcast**: Captured cookies distributed to all workers via IPC

### Phase 2: Discovery
5. **Parallel Page Discovery**: Workers extract metadata (title, links) from pages in parallel
6. **Queue Management**: Master's `GlobalQueueManager` coordinates task distribution
7. **Context Building**: `PageContext` tree constructed with parent-child relationships

### Phase 3: Conflict Resolution
8. **Duplicate Detection**: `ConflictResolver` identifies pages referenced multiple times
9. **Canonical Selection**: Best version selected for each unique page
10. **Path Calculation**: Target file paths computed based on hierarchy
11. **Link Rewrite Map**: Map generated for transforming links during download

### Phase 4: Download
12. **Parallel Scraping**: Workers download unique pages with full content expansion
13. **Asset Download**: Images, CSS, and files downloaded and localized
14. **Link Rewriting**: Internal links transformed using the rewrite map
15. **File Writing**: Workers write HTML directly to disk (no IPC bottleneck)

### Phase 5: Complete
16. **Statistics**: Master aggregates and displays scraping statistics
17. **Cleanup**: All workers gracefully terminated
18. **Result**: Fully self-contained, browsable offline copy

## Logging

The scraper provides detailed, structured logging:

```
[14:30:15] [MAIN] Starting Notion page download process
[14:30:16] [COOKIE] Cookie banner detected
[14:30:18] [TOGGLE] Expanded 5 toggles at this depth
[14:30:20] [LINKS] Found 12 internal page links
[14:30:22] [IMAGE] Downloaded and rewritten 8 images
```

## Configuration Options

### Timing Settings

- `TIMEOUT_PAGE_LOAD`: Page load timeout (default: 60000ms)
- `TIMEOUT_NAVIGATION`: Navigation timeout (default: 30000ms)
- `WAIT_AFTER_TOGGLE`: Wait after expanding toggles (default: 2000ms)

### Depth Settings

- `MAX_RECURSION_DEPTH`: Maximum depth for following links (default: 5)
- `MAX_EXPANSION_DEPTH`: Maximum depth for expanding toggles (default: 3)

### Selectors

All CSS selectors are configurable in the `SELECTORS` object in `Config.js`.

## Error Handling

- Each component has robust error handling
- Errors are logged but don't stop the entire process
- Already-visited pages are tracked to avoid infinite loops

## Statistics

At the end of scraping, the application provides:

- Total pages scraped
- Total assets downloaded
- **Total internal links rewritten** ✨
- Total time elapsed
- Complete page hierarchy visualization
- Instructions for opening the offline copy

## Limitations

- Only works with publicly accessible Notion pages
- Respects the configured recursion depth to prevent excessive downloads
- Does not handle authentication or private pages

## Key Enhancements in This Version

### ✨ 1:1 Perfect Replication
- Complete HTML/CSS/JavaScript preservation
- All interactive elements remain functional
- Pixel-perfect visual matching

### ✨ Deep Hierarchical Nesting
- True parent-child relationships in folder structure
- Nested toggles create nested folders
- Mirrors visual Notion hierarchy

### ✨ Resilient Asset Downloading
- Advanced filename sanitization for complex URLs
- Retry logic with exponential backoff
- Handles redirects and special characters
- Downloads background images from inline styles

### ✨ Complete Link Rewriting
- All internal links converted to relative paths
- Maintains navigation between pages offline
- External links preserved unchanged
- JSDOM-based HTML parsing for accuracy

### ✨ Distributed Architecture
- Micro-kernel design with Master-Worker IPC
- Native Node.js `child_process` for worker management (no external cluster library)
- Auto-scaled worker pool based on system resources
- Graceful degradation when workers crash

## Contributing

Feel free to extend the functionality by:

1. Adding new asset types (PDFs, videos, embedded content)
2. Implementing authentication for private Notion pages
3. Adding export formats (Markdown, PDF)
4. Optimizing concurrent downloads
5. Adding a progress bar UI

## License

MIT
