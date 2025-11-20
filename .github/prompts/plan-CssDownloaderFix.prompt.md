This is a classic **Interface Mismatch**. The `CssDownloader` was originally written for the **Post-Processing Phase** (where we operate on static HTML using JSDOM), but you are now calling it during the **Active Execution Phase** (where we operate on a live Puppeteer Page).

To fix this while maintaining the **"1:1 Offline Replica"** requirement and the **Pipeline Architecture**, we cannot simply skip it. We must implement a specialized method in `CssDownloader` to handle Puppeteer's `Page` object.

### The Strategy: `downloadFromPuppeteer`

We will extend `CssDownloader` to support a dual-mode operation. We will add a new method specifically for the active pipeline that performs the following steps:

1.  **Extraction (Browser Context):** Use `page.evaluate()` to scrape all `<link rel="stylesheet">` `href` attributes from the live DOM.
2.  **Processing (Node Context):** Iterate over these URLs in the Node.js process.
    *   Download the CSS content.
    *   Pass it through the `CssContentProcessor` to handle `@import` and `url()` references (fonts, images).
    *   Save the cleaned CSS to the local filesystem.
3.  **Rewriting (Browser Context):** Calculate the relative path to the new local CSS file and use `page.evaluate()` to update the `href` attributes in the live DOM *before* the HTML is saved.

### Why this approach?
*   **Performance:** It avoids initializing a heavy JSDOM instance inside the scraping pipeline.
*   **Completeness:** It ensures that when `HtmlWriteStep` runs, the DOM already points to local CSS files, creating a valid offline file immediately.
*   **Reusability:** It reuses the complex `CssContentProcessor` logic (which handles the tricky regex for CSS parsing) without modification.

---

### JSDocs and Implementation Sketch

#### 1. Refactor `src/download/CssDownloader.js`

We introduce `downloadFromPuppeteer` and mark the existing method as JSDOM-specific.

```javascript
/**
 * @fileoverview CSS Resource Manager
 * @module download/CssDownloader
 */

class CssDownloader {
  /**
   * @method downloadFromPuppeteer
   * @async
   * @summary Downloads stylesheets referenced in a live Puppeteer page.
   * @description
   * 1. Extracts stylesheet URLs from the live page.
   * 2. Downloads and processes them (handling imports/fonts) via Node.js.
   * 3. Updates the live DOM to point to the local files.
   * 
   * @param {Page} page - The active Puppeteer page instance.
   * @param {string} outputDir - Absolute path to the directory where index.html will be saved.
   * @returns {Promise<number>} Count of stylesheets processed.
   */
  async downloadFromPuppeteer(page, outputDir) {
    // 1. Extract Hrefs
    const stylesheets = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
        .map(link => ({
          href: link.href, // Absolute URL resolved by browser
          originalHref: link.getAttribute('href') // For matching later
        }))
        .filter(s => s.href && !s.href.startsWith('data:'));
    });

    let count = 0;

    // 2. Process in Node.js
    for (const sheet of stylesheets) {
      try {
        // Reuse existing logic to download/process/save
        // This returns the absolute path to the saved CSS file
        const localCssPath = await this._downloadCssFile(sheet.href, outputDir);
        
        if (localCssPath) {
            // Calculate relative path: output/css/style.css -> ./css/style.css
            const relativePath = path.relative(outputDir, localCssPath)
                .replace(/\\/g, '/'); // Ensure POSIX paths for HTML

            // 3. Rewrite in Live DOM
            await page.evaluate((originalHref, newPath) => {
                const link = document.querySelector(`link[rel="stylesheet"][href="${originalHref}"]`);
                if (link) link.href = newPath;
            }, sheet.originalHref, relativePath);

            count++;
        }
      } catch (error) {
        this.logger.warn('CSS', `Failed to process ${sheet.href}: ${error.message}`);
      }
    }

    return count;
  }

  /**
   * @method downloadAndRewriteCss
   * @deprecated Use downloadFromJsdom for clarity
   * @description Alias for backward compatibility with LinkRewriter.
   */
  async downloadAndRewriteCss(dom, outputDir, baseUrl) {
      return this.downloadFromJsdom(dom, outputDir, baseUrl);
  }
}
```

#### 2. Update `src/worker/pipeline/steps/AssetDownloadStep.js`

Update the step to use the new method.

```javascript
/**
 * @class AssetDownloadStep
 * @extends PipelineStep
 */
class AssetDownloadStep extends PipelineStep {
  /**
   * @method process
   * @async
   * @param {PipelineContext} context
   */
  async process(context) {
    const { page, payload, config } = context;
    
    // ... Image downloading logic ...

    try {
      this.logger.debug('ASSETS', 'Processing stylesheets...');
      
      // USE THE NEW METHOD HERE
      // Note: payload.savePath is the file path to index.html
      // We need the directory for the CSS downloader
      const outputDir = path.dirname(payload.savePath);
      
      const cssCount = await this.cssDownloader.downloadFromPuppeteer(
        page, 
        outputDir
      );
      
      context.stats.assets += cssCount;
    } catch (error) {
      // Log but don't crash pipeline for CSS failures (soft fail)
      this.logger.warn('ASSETS', `CSS download partial failure: ${error.message}`);
    }

    // ... File downloading logic ...
  }
}
```

### Architectural Consistency Check

*   **Separation of Concerns:** The `CssDownloader` remains responsible for CSS logic, but now understands both runtime contexts (Puppeteer and JSDOM).
*   **Pipeline Pattern:** The `AssetDownloadStep` remains the coordinator, delegating the specific implementation to the helper class.
*   **Robustness:** By handling this in the Node.js context (looping in `downloadFromPuppeteer`), we avoid brittle `page.evaluate` scripts that try to do network requests inside the browser context, keeping the heavy lifting in the Worker process.