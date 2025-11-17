# Testing & Verification Guide

## Pre-Flight Checklist

Before running the scraper, verify your setup:

```bash
# Check Node.js version (should be 14.0.0 or higher)
node --version

# Verify all dependencies are installed
npm list puppeteer axios jsdom

# Expected output:
# notion-recursive-scraper@1.0.0
# ├── axios@1.6.x
# ├── jsdom@23.0.x
# └── puppeteer@21.0.x
```

## Test 1: Configuration Verification

```javascript
// Open src/Config.js and verify:
this.NOTION_PAGE_URL = 'YOUR_NOTION_URL_HERE'; // ← Must be set!
this.OUTPUT_DIR = 'downloaded_course_material';
this.MAX_RECURSION_DEPTH = 5;
```

## Test 2: Small Test Run

Start with a small test to verify the system works:

```javascript
// In src/Config.js, temporarily set:
this.MAX_RECURSION_DEPTH = 1; // Only scrape main page + 1 level
```

Then run:
```bash
npm start
```

### Expected Behavior:
1. Browser launches (may be invisible in headless mode)
2. Cookie banner is handled automatically
3. Content expands (toggles, databases)
4. Images download to `downloaded_course_material/.../images/`
5. HTML files are saved
6. Links are rewritten
7. Statistics are printed

### Look for these log messages:
```
[COOKIE] Cookie banner detected.
[COOKIE] Page reloaded successfully.
[TOGGLE] Expanded X toggles at this depth.
[LINKS] Found X internal page links.
[IMAGE] Downloaded X/X assets and rewritten paths.
[SCRAPE] Saved HTML to: downloaded_course_material/.../index.html
[LINK-REWRITE] Rewrote X links in PageName
[STATS] Total internal links rewritten: X
```

## Test 3: Verify Output Structure

After the test run, check the output:

```bash
# Windows
dir /s /b downloaded_course_material

# Mac/Linux
find downloaded_course_material -type f
```

### Verify:
- [ ] `Main_Page/index.html` exists
- [ ] `Main_Page/images/` folder exists
- [ ] At least one subfolder exists (if MAX_RECURSION_DEPTH > 0)
- [ ] Each folder has an `index.html` file
- [ ] Each folder has an `images/` subfolder (if images were on that page)

## Test 4: Verify HTML Preservation

Open `downloaded_course_material/Main_Page/index.html` in a text editor.

### Check for:
- [ ] `<style>` tags are present
- [ ] `<script>` tags are present
- [ ] CSS variables like `var(--c-texPri)` are present
- [ ] Image src attributes point to `images/1-filename.jpg` (relative paths)

## Test 5: Verify Link Rewriting

Search for `<a href=` in the saved HTML files.

### You should find:
```html
<!-- GOOD: Relative links to local files -->
<a href="../Week_2/index.html">Week 2</a>
<a href="Introduction/index.html">Introduction</a>

<!-- GOOD: External links unchanged -->
<a href="https://google.com">Google</a>

<!-- BAD: Should NOT find absolute Notion URLs -->
<!-- <a href="https://notion.site/...">XXX</a> ← This is wrong! -->
```

## Test 6: Browser Verification

Open the downloaded site in a browser:

```bash
# Open the main page
start downloaded_course_material/Main_Page/index.html
```

### Verify:
1. **Visual Appearance**
   - [ ] Styling looks identical to online version
   - [ ] All images load (no broken images)
   - [ ] Toggle blocks are visible
   - [ ] Buttons and UI elements are styled correctly

2. **Link Navigation**
   - [ ] Click a link to another page
   - [ ] URL bar shows `file:///...` (local file)
   - [ ] Page loads successfully
   - [ ] Images on the new page load
   - [ ] Click "Back" button works

3. **Interactive Elements**
   - [ ] Toggle blocks can expand/collapse (if they still work)
   - [ ] Buttons have hover effects
   - [ ] Tooltips appear (if any)

4. **External Links**
   - [ ] Click an external link (e.g., to a website)
   - [ ] It opens the actual website (not a local file)

## Test 7: Deep Nesting Verification

If your Notion page has nested content:

### Check folder structure:
```
Material/
└── Week_1/
    ├── index.html
    └── Introduction/        ← Nested under Week_1
        ├── index.html
        └── Subtopic/        ← Deeply nested
            └── index.html
```

### Open nested page:
1. Open `Material/Week_1/Introduction/index.html`
2. Click a link to `Subtopic`
3. Verify it navigates to `Subtopic/index.html` (relative path)
4. Click a link back to `Week_1`
5. Verify it navigates to `../../index.html`

## Test 8: Asset Download Verification

Check that all assets were downloaded successfully:

### Look for error messages:
```bash
# Search for download errors in the console output
# Look for: [DOWNLOAD] ERROR: Failed to download...
```

### Verify image files:
```bash
# Check that images are not empty
# Windows
dir downloaded_course_material\Main_Page\images

# Each file should have a size > 0 KB
```

## Test 9: Full Site Run

Once small tests pass, run a full scrape:

```javascript
// In src/Config.js, set:
this.MAX_RECURSION_DEPTH = 5; // Or higher if needed
```

```bash
npm start
```

### Monitor the logs:
- Watch for `[RECURSION] Queue size: X pages remaining`
- This tells you how many more pages need to be scraped
- Process may take 5-30 minutes depending on site size

### Final verification:
Check the statistics output:
```
[STATS] Total pages scraped: 47
[STATS] Total assets downloaded: 312
[STATS] Total internal links rewritten: 189
[STATS] Total time elapsed: 8m 23s
```

If all three numbers are > 0, the scraper worked!

## Troubleshooting Common Issues

### Issue: No pages scraped
**Symptoms**: `Total pages scraped: 1` (only main page)

**Solutions**:
- Check that the main page actually has links to other pages
- Verify `MAX_RECURSION_DEPTH > 0`
- Check logs for `[LINKS] Found 0 internal page links` → page has no links

### Issue: Images not loading in browser
**Symptoms**: Broken image icons in browser

**Solutions**:
- Check browser console for 404 errors
- Verify images were downloaded: check `images/` folders
- Check HTML: `<img src="images/1-file.jpg">` should be relative path
- Check for `[DOWNLOAD] ERROR` messages in scraper output

### Issue: Links don't work offline
**Symptoms**: Clicking links tries to go online or shows 404

**Solutions**:
- Check for `[LINK-REWRITE] Total internal links rewritten: 0` → rewriting failed
- Verify JSDOM is installed: `npm list jsdom`
- Check HTML source: links should be `../Folder/index.html`, not `https://notion.site/...`
- Look for `[LINK-REWRITE] ERROR` messages

### Issue: "Module not found" error
**Symptoms**: `Error: Cannot find module 'jsdom'` (or axios, puppeteer)

**Solution**:
```bash
npm install
```

### Issue: Cookie banner still in HTML
**Symptoms**: Cookie banner visible when opening offline copy

**Solutions**:
- Check logs for `[COOKIE] Cookie banner detected` → handler tried to run
- Check logs for `[COOKIE] Page reloaded successfully` → handler succeeded
- If not present, cookie handling failed → report as bug

### Issue: Scraper hangs or times out
**Symptoms**: Process stops with timeout error

**Solutions**:
- Increase timeouts in `src/Config.js`:
  ```javascript
  this.TIMEOUT_PAGE_LOAD = 120000; // 2 minutes
  this.TIMEOUT_NAVIGATION = 60000; // 1 minute
  ```
- Check your internet connection
- Try scraping a smaller page first

## Performance Benchmarks

### Small Site (< 20 pages):
- Time: 2-5 minutes
- Assets: 50-200
- Links rewritten: 20-100

### Medium Site (20-50 pages):
- Time: 5-15 minutes
- Assets: 200-500
- Links rewritten: 100-300

### Large Site (50+ pages):
- Time: 15-60 minutes
- Assets: 500-2000+
- Links rewritten: 300-1000+

## Success Criteria

✅ All pages scraped (check page count in stats)
✅ All images downloaded (check asset count in stats)
✅ All links rewritten (check link count in stats)
✅ Offline navigation works (test in browser)
✅ Visual appearance matches online version
✅ No error messages in console output

## Reporting Issues

If you encounter problems, gather this information:

1. Console output (full log)
2. Node.js version: `node --version`
3. Package versions: `npm list puppeteer axios jsdom`
4. Target Notion URL structure
5. Operating system
6. Specific error messages

## Next Steps

Once all tests pass:
- Increase `MAX_RECURSION_DEPTH` if needed
- Enable debug logging: `DEBUG=1 node main.js`
- Archive the output folder
- Share the offline copy with others (just zip the folder!)

Happy testing! 🧪
