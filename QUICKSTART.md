# Quick Start Guide

## Installation

```bash
# Install dependencies
npm install

# Or manually install each package
npm install puppeteer axios jsdom
```

## Configuration

Edit `src/Config.js` to set your target Notion page:

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

# Or with Node directly
node main.js

# Enable debug logging
DEBUG=1 node main.js
```

## What Happens

1. **Cookie Handling** (5-10 seconds)
   - Automatically rejects cookie banners
   - Handles confirmation dialog
   - Waits for page reload

2. **Content Expansion** (10-30 seconds per page)
   - Clicks "Load more" buttons in databases
   - Scrolls to trigger lazy-loading
   - Expands all toggle blocks

3. **Recursive Scraping** (varies by site size)
   - Follows all internal links
   - Creates nested folder structure
   - Downloads all images and assets

4. **Link Rewriting** (5-10 seconds per page)
   - Parses all saved HTML files
   - Rewrites internal links to relative paths
   - Creates fully navigable offline copy

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
Increase timeout in `src/Config.js`:
```javascript
this.TIMEOUT_PAGE_LOAD = 120000; // 2 minutes
```

### "Too many open files"
Reduce recursion depth in `src/Config.js`:
```javascript
this.MAX_RECURSION_DEPTH = 3;
```

### Images not downloading
Check the console for `[DOWNLOAD] ERROR` messages. The scraper will retry up to 3 times automatically.

### Links not working offline
Check that link rewriting completed:
```
[LINK-REWRITE] Total internal links rewritten: XX
```

## Performance Tips

- **Large sites**: Set `MAX_RECURSION_DEPTH = 3` initially, then increase if needed
- **Slow internet**: Increase timeouts in `Config.js`
- **Debug issues**: Run with `DEBUG=1` for detailed logs

## Example Session

```
[14:30:00] [MAIN] ========================================
[14:30:00] [MAIN] Starting Notion page download process
[14:30:00] [MAIN] ========================================
[14:30:05] [COOKIE] Cookie banner detected.
[14:30:07] [COOKIE] Page reloaded successfully.
[14:30:10] [DATABASE] No "Load more" buttons found.
[14:30:15] [TOGGLE] Expanded 5 toggles at this depth.
[14:30:20] [LINKS] Found 12 internal page links.
[14:30:25] [IMAGE] Downloaded 8/8 assets and rewritten paths.
[14:30:27] [SCRAPE] Saved HTML to: downloaded_course_material/Main_Page/index.html
...
[14:32:00] [LINK-REWRITE] Rewrote 8 links in Introduction
[14:32:05] [LINK-REWRITE] Total internal links rewritten: 47
[14:32:05] [STATS] Total pages scraped: 13
[14:32:05] [STATS] Total assets downloaded: 89
[14:32:05] [STATS] Total internal links rewritten: 47
[14:32:05] [STATS] Total time elapsed: 2m 5s
[14:32:05] [STATS] 
[14:32:05] [STATS] The downloaded site is now fully browsable offline!
[14:32:05] [STATS] Open downloaded_course_material/Main_Page/index.html in your browser.
```

## Need Help?

Check these files:
- `README.md` - Full documentation
- `ENHANCEMENTS.md` - Technical details of recent improvements
- `src/Config.js` - All configuration options

Happy scraping! 🚀
