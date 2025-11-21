const fs = require('fs/promises');
const path = require('path');
const { JSDOM } = require('jsdom');

/**
 * @classdesc Handles offline link rewriting and CSS localization for scraped pages.
 * 
 * Performs post-scraping link transformation to enable offline browsing:
 * - Rewrites internal Notion page links to point to local HTML files
 * - Calculates relative paths between pages based on directory structure
 * - Downloads and localizes external CSS stylesheets
 * - Preserves external links unchanged
 * - Uses JSDOM for safe DOM manipulation without browser overhead
 * 
 * This is a critical step that transforms the scraped content from online-dependent
 * to fully self-contained offline browsing.
 * 
 * @see RecursiveScraper#execute
 * @see PageContext#getRelativePathTo
 * @see CssDownloader
 */
class LinkRewriter {
  /**
   * @param {Config} config - Configuration object.
   * @param {Logger} logger - Logger instance.
   * @param {CssDownloader} cssDownloader - CSS downloader instance.
   */
  constructor(config, logger, cssDownloader) {
    this.config = config;
    this.logger = logger;
    this.cssDownloader = cssDownloader;
  }

  /**
   * @summary Rewrite all internal links in saved HTML files to point to local paths.
   * 
   * @description Executes the complete link rewriting workflow:
   * 1. Loads saved HTML using JSDOM
   * 2. Iterates through all anchor elements
   * 3. Identifies internal Notion links using urlToContextMap
   * 4. Calculates relative paths using PageContext.getRelativePathTo()
   * 5. Rewrites href attributes to local paths
   * 6. Downloads and localizes CSS stylesheets
   * 7. Saves modified HTML back to disk
   * 
   * Only rewrites links that point to pages present in urlToContextMap,
   * leaving external links and unscraped pages unchanged.
   * 
   * @param {PageContext} pageContext - The context of the page to rewrite.
   * @param {Map<string, PageContext>} urlToContextMap - Map of URLs to PageContexts.
   * @returns {Promise<number>} Number of internal links successfully rewritten.
   * 
   * @throws {Error} If HTML file cannot be read or written.
   * 
   * @see PageContext#getRelativePathTo
   * @see CssDownloader#downloadAndRewriteCss
   */
  async rewriteLinksInFile(pageContext, urlToContextMap) {
    try {
      const htmlFilePath = pageContext.htmlFilePath;
      if (!htmlFilePath) {
        this.logger.warn('LINK-REWRITE', `No HTML file path for ${pageContext.title}`);
        return 0;
      }
      
      const html = await fs.readFile(htmlFilePath, 'utf-8');
      const dom = new JSDOM(html);
      const document = dom.window.document;
      
      // Create ID-to-Context map for robust lookup
      const idToContextMap = new Map();
      for (const context of urlToContextMap.values()) {
        if (context.id) {
          idToContextMap.set(context.id, context);
        }
      }

      let rewriteCount = 0;
      let modified = false;
      const pageDir = path.dirname(htmlFilePath);
      
      if (this.cssDownloader) {
        const cssResult = await this.cssDownloader.downloadAndRewriteCss(dom, pageDir, pageContext.url);
        if (cssResult.modified) {
          modified = true;
          this.logger.success(
            'CSS',
            `Localized ${cssResult.stylesheets} stylesheet(s), ${cssResult.assets} dependent asset(s), ${cssResult.inlineStyles} inline style block(s) in ${pageContext.title}`
          );
        }
      }
      
      const links = document.querySelectorAll('a[href]');
      
      for (const link of links) {
        const href = link.getAttribute('href');
        if (!href) continue;
        
        let absoluteUrl;
        try {
          if (href.startsWith('/')) {
            absoluteUrl = this.config.getBaseUrl() + href;
          } else if (href.startsWith('http')) {
            absoluteUrl = href;
          } else {
            continue;
          }
          
          // Preserve hash for internal anchors
          const urlParts = absoluteUrl.split('#');
          const urlWithoutHash = urlParts[0].split('?')[0];
          const hash = urlParts.length > 1 ? '#' + urlParts[1] : '';
          
          let targetContext = urlToContextMap.get(urlWithoutHash);
          
          // Fallback: Try to find by ID if URL lookup failed
          if (!targetContext) {
            const idMatch = urlWithoutHash.match(/29[a-f0-9]{30}/i);
            if (idMatch) {
              const id = idMatch[0];
              targetContext = idToContextMap.get(id);
            }
          }
          
          if (targetContext) {
            let newHref;
            
            // If target is the same page, only use hash (if present)
            // This prevents reloading the page when clicking internal anchors
            if (targetContext.id === pageContext.id) {
              newHref = hash || 'index.html';
            } else {
              // Different page: resolve full relative path + hash
              const relativePath = pageContext.getRelativePathTo(targetContext);
              newHref = relativePath + hash;
            }

            link.setAttribute('href', newHref);
            rewriteCount++;
            this.logger.debug('LINK-REWRITE', `${pageContext.title}: ${href} -> ${newHref}`);
          }
        } catch (error) {
          continue;
        }
      }
      
      if (rewriteCount > 0 || modified) {
        const modifiedHtml = dom.serialize();
        await fs.writeFile(htmlFilePath, modifiedHtml, { encoding: 'utf-8' });
        if (rewriteCount > 0) {
          this.logger.success('LINK-REWRITE', `Rewrote ${rewriteCount} links in ${pageContext.title}`);
        }
      }
      
      return rewriteCount;
      
    } catch (error) {
      this.logger.error('LINK-REWRITE', `Error rewriting links in ${pageContext.title}`, error);
      return 0;
    }
  }
}

module.exports = LinkRewriter;
