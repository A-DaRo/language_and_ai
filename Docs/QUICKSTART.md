# Quick Start Guide

## Installation

```bash
# Install dependencies
npm install

# Or manually install each package
npm install puppeteer axios jsdom
```

## Configuration

Edit `src/core/Config.js` to set your target Notion page:

```javascript
this.NOTION_PAGE_URL = 'https://your-notion-site-url-here';
this.OUTPUT_DIR = 'downloaded_course_material';
this.MAX_RECURSION_DEPTH = 5;  // How deep to follow links
this.MAX_EXPANSION_DEPTH = 3;   // How deep to expand toggles
```

## Running the Scraper

```bash
# Start scraping
npm start

# Or directly
node main-cluster.js

# With custom depth
node main-cluster.js --max-depth 3

# Show help
node main-cluster.js --help
```

**System Requirements:**
- 4GB+ RAM recommended for optimal performance
- Each worker requires ~1GB RAM
- Worker count auto-scales based on available memory

### Testing

```bash
# Run full integration test suite
npm test

# Run specific tests
node tests/test-integration.js      # Full workflow test
node tests/test-worker-crash.js     # Crash recovery test
node tests/test-cookie-propagation.js  # Cookie handling test
```

## What Happens

The scraper executes in **5 distinct phases**:

### 1. Bootstrap Phase (10-15 seconds)
- Spawns initial worker for cookie capture
- Spawns remaining workers based on system capacity
- Broadcasts authentication cookies to all workers

### 2. Discovery Phase (parallel, fast)
- Multiple workers discover pages simultaneously
- Extracts metadata, titles, and links only (no heavy assets)
- Builds complete PageContext tree

### 3. Conflict Resolution (5-10 seconds)
- Detects duplicate pages (same URL referenced multiple times)
- Selects canonical version for each unique page
- Generates link rewrite map for download phase

### 4. Download Phase (parallel, varies by site size)
- Multiple workers download unique pages simultaneously
- Full scraping with assets (images, CSS, files)
- Link rewriting using the map from phase 3
- Workers write files directly to disk

### 5. Complete Phase (instant)
- Statistics reporting
- Resource cleanup
- All workers gracefully terminated

## Expected Output

```
downloaded_course_material/
├── Main_Page/
│   ├── index.html          ← Start here!
│   └── images/
├── Section_1/
│   ├── index.html
│   ├── images/
│   └── Subsection/
│       ├── index.html
│       └── images/
└── Section_2/
    └── ...
```

## Opening the Offline Copy

Simply open the main page in any browser:

```bash
# Windows
start downloaded_course_material/Main_Page/index.html

# Mac
open downloaded_course_material/Main_Page/index.html

# Linux
xdg-open downloaded_course_material/Main_Page/index.html
```

Or just double-click `Main_Page/index.html` in your file explorer!

## Verification Checklist

After scraping completes, verify:

- [ ] All pages are present in the folder structure
- [ ] Images load correctly (no broken images)
- [ ] Clicking links navigates to local pages (URL bar shows `file://`)
- [ ] Toggle blocks still work
- [ ] Styling looks identical to online version
- [ ] External links still point to original websites

## Troubleshooting

### "Module not found: jsdom"
```bash
npm install jsdom
```

### "Page load timeout"
Increase timeout in `src/core/Config.js`:
```javascript
this.TIMEOUT_PAGE_LOAD = 120000; // 2 minutes
```

### "Too many open files" or "Out of memory"
Reduce recursion depth in `src/core/Config.js`:
```javascript
this.MAX_RECURSION_DEPTH = 3;
```

Or manually limit worker count by editing `src/cluster/BrowserInitializer.js`.

### Workers not spawning
Check system resources. Each worker needs ~1GB RAM. If system has <4GB, workers may be limited.

### Images not downloading
Check the console for `[DOWNLOAD] ERROR` messages. The scraper will retry up to 3 times automatically.

### Links not working offline
Check that link rewriting completed:
```
[LINK-REWRITE] Total internal links rewritten: XX
```

## Performance Tips

- **Large sites**: Set `MAX_RECURSION_DEPTH = 3` initially in `src/core/Config.js`
- **Slow internet**: Increase timeouts in `src/core/Config.js`
- **Limited RAM**: System auto-scales workers, but you can manually limit in `BrowserInitializer.js`
- **Monitor resources**: Use Task Manager/Activity Monitor to watch memory usage during scraping

## Example Session

```
[14:30:00] [MAIN] ========================================
[14:30:00] [MAIN] Initializing Cluster Mode
[14:30:00] [MAIN] ========================================
[14:30:02] [BOOTSTRAP] Spawning initial worker for cookie capture...
[14:30:05] [WORKER] Worker 1 ready
[14:30:06] [COOKIES] Captured 3 authentication cookies
[14:30:07] [BOOTSTRAP] Spawning 3 additional workers...
[14:30:10] [WORKER] Worker 2 ready
[14:30:10] [WORKER] Worker 3 ready
[14:30:11] [WORKER] Worker 4 ready
[14:30:11] [COOKIES] Broadcast cookies to 4 workers
[14:30:12] [DISCOVERY] Starting parallel discovery phase...
[14:30:45] [DISCOVERY] Discovered 47 pages
[14:30:46] [CONFLICT] Resolving duplicates... found 5 duplicates
[14:30:46] [CONFLICT] Generated rewrite map for 42 unique pages
[14:30:47] [DOWNLOAD] Starting parallel download phase...
[14:32:30] [DOWNLOAD] Completed 42/42 pages
[14:32:31] [STATS] Total pages scraped: 42
[14:32:31] [STATS] Total assets downloaded: 215
[14:32:31] [STATS] Total internal links rewritten: 138
[14:32:31] [STATS] Total time elapsed: 2m 31s
[14:32:31] [STATS] 
[14:32:31] [STATS] The downloaded site is now fully browsable offline!
[14:32:31] [STATS] Open downloaded_course_material/Main_Page/index.html in your browser.
```

## Need Help?

Check these files:
- `Docs/README.md` - Full documentation
- `Docs/CLUSTER_MODE.md` - Detailed cluster architecture
- `src/core/Config.js` - All configuration options

Happy scraping! 🚀
