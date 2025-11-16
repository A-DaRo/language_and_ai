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
- **🧭 Interactive Discovery Phase**: Plan scrapes with a fast dry run, ASCII tree preview, and depth controls before downloading anything

## Architecture

The application follows object-oriented principles with clear separation of concerns:

```
src/
├── Config.js              # Configuration management
├── Logger.js              # Structured logging utility
├── PageContext.js         # Page hierarchy context
├── CookieHandler.js       # Cookie banner handling
├── ContentExpander.js     # Toggle and database expansion
├── LinkExtractor.js       # Internal link extraction
├── AssetDownloader.js     # Image and asset downloads
├── PageScraper.js         # Individual page scraping
├── RecursiveScraper.js    # Recursive scraping orchestration
└── NotionScraper.js       # Main orchestrator
```

## Installation

```bash
npm install
# or manually:
npm install puppeteer axios jsdom
```

## Usage

### Basic Usage

```bash
node main.js
```

This launches the full two-phase workflow:

1. **Discovery (Dry Run)** quickly maps the entire reachable Notion space.
2. The resulting hierarchy is rendered as an ASCII tree.
3. You're prompted to proceed, abort, or request a deeper crawl before any content is downloaded.

### Planning & Confirmation Options

- `node main.js --dry-run` → perform **only** the discovery phase and exit. Great for planning without downloading files.
- `node main.js --yes` → skip the interactive prompt and proceed directly from discovery to execution when you're already confident.
- `node main.js --max-depth 3` → override the initial discovery depth (you can still request "deeper" interactively later).

### Configuration

Edit `src/Config.js` to customize settings:

```javascript
this.NOTION_PAGE_URL = 'your-notion-page-url';
this.OUTPUT_DIR = 'downloaded_course_material';
this.MAX_RECURSION_DEPTH = 5;  // Maximum depth for following links
this.MAX_EXPANSION_DEPTH = 3;   // Maximum depth for expanding toggles
```

### Programmatic Usage

```javascript
const { NotionScraper } = require('./main');

async function customScrape() {
  const scraper = new NotionScraper();
  await scraper.run();
}

customScrape();
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

### Phase 1: Discovery (Dry Run)
1. **Initialization**: Browser and automation helpers spin up.
2. **Lightweight Visits**: Each page is opened just long enough to capture the title and internal links—no downloads, no expansions.
3. **Tree Assembly**: The discovered hierarchy is stored in `PageContext` objects and rendered as an ASCII tree for review.
4. **Interactive Prompt**: Accept the plan, abort, or type `d`/`deeper` to bump the discovery depth and rebuild the map before continuing.

### Phase 2: Execution (Scraping)
5. **Plan Reuse**: The confirmed tree becomes the authoritative roadmap—no redundant rediscovery.
6. **Content Expansion & Asset Capture**: Each queued page undergoes the full scraping routine (expansions, asset/file downloads, HTML preservation).
7. **Link Rewriting**: Saved HTML is revisited so internal `<a>` tags point to the correct local paths.
8. **Integrity Audit**: The auditor runs last to flag missing files or lingering remote references.

### Result
A fully self-contained, interactive offline copy that mirrors the original Notion site, produced under your direct supervision.

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

### ✨ Interactive Discovery & Confirmation
- Rapid "dry run" pass builds the entire crawl tree with zero downloads
- ASCII plan view keeps you in control before any heavy work begins
- Built-in prompt lets you deepen, abort, or auto-approve with CLI flags (`--dry-run`, `--yes`, `--max-depth`)
- Execution phase reuses the plan for maximum efficiency—no duplicate crawling

## Contributing

Feel free to extend the functionality by:

1. Adding new asset types (PDFs, videos, embedded content)
2. Implementing authentication for private Notion pages
3. Adding export formats (Markdown, PDF)
4. Optimizing concurrent downloads
5. Adding a progress bar UI

## License

MIT
