const fs = require('fs/promises');
const path = require('path');
const { JSDOM } = require('jsdom');
const BlockIDMapper = require('./BlockIDMapper');

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
    this.blockIDMapper = new BlockIDMapper();
  }

  /**
   * @summary Rewrite all internal links in saved HTML files to point to local paths.
   * 
   * @description Executes the complete link rewriting workflow:
   * 1. Loads saved HTML using JSDOM
   * 2. Iterates through all anchor elements
   * 3. Identifies internal Notion links using urlToContextMap
   * 4. Calculates relative paths using PageContext.getRelativePathTo()
   * 5. Rewrites href attributes to local paths with proper block anchors
   * 6. Downloads and localizes CSS stylesheets
   * 7. Saves modified HTML back to disk
   * 
   * Only rewrites links that point to pages present in urlToContextMap,
   * leaving external links and unscraped pages unchanged.
   * 
   * @param {PageContext} pageContext - The context of the page to rewrite.
   * @param {Map<string, PageContext>} urlToContextMap - Map of URLs to PageContexts.
   * @param {PageGraph} [pageGraph=null] - Optional page graph with edge classifications.
   * @param {Map<string, Map>} [blockMapCache=null] - Optional pre-loaded block ID maps.
   * @returns {Promise<number>} Number of internal links successfully rewritten.
   * 
   * @throws {Error} If HTML file cannot be read or written.
   * 
   * @see PageContext#getRelativePathTo
   * @see CssDownloader#downloadAndRewriteCss
   */
  async rewriteLinksInFile(pageContext, urlToContextMap, pageGraph = null, blockMapCache = null) {
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
          
          // Parse href to separate URL and block ID
          const { urlPart, blockIdRaw } = this._parseHref(absoluteUrl);
          
          let targetContext = urlToContextMap.get(urlPart);
          
          // Fallback: Try to find by ID if URL lookup failed
          if (!targetContext) {
            const idMatch = urlPart.match(/29[a-f0-9]{30}/i);
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
              newHref = this._buildAnchorHash(blockIdRaw, targetContext, blockMapCache);
              if (!newHref) {
                newHref = 'index.html';
              }
            } else {
              // Different page: resolve full relative path + hash
              const relativePath = pageContext.getRelativePathTo(targetContext);
              const hash = this._buildAnchorHash(blockIdRaw, targetContext, blockMapCache);
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

  /**
   * Parse href to separate URL and raw block ID
   * @private
   * @param {string} href - Full href attribute
   * @returns {Object} { urlPart, blockIdRaw }
   */
  _parseHref(href) {
    const parts = href.split('#');
    const urlPart = parts[0].split('?')[0];
    const blockIdRaw = parts.length > 1 ? parts[1] : null;
    return { urlPart, blockIdRaw };
  }

  /**
   * Build anchor hash for block ID
   * Converts raw block ID to formatted UUID using block map if available
   * 
   * @private
   * @param {string|null} blockIdRaw - Raw block ID from URL
   * @param {PageContext} targetContext - Target page context
   * @param {Map<string, Map>} blockMapCache - Cache of block ID maps
   * @returns {string} Formatted hash (e.g., '#29d979ee-ca9f-...') or empty string
   */
  _buildAnchorHash(blockIdRaw, targetContext, blockMapCache) {
    if (!blockIdRaw) {
      return '';
    }

    // Get block map for target page if cache provided
    if (blockMapCache && targetContext.id) {
      const blockMap = blockMapCache.get(targetContext.id);
      if (blockMap) {
        const formattedId = this.blockIDMapper.getFormattedId(blockIdRaw, blockMap);
        return '#' + formattedId;
      }
    }

    // Fallback: format the raw ID directly
    const formattedId = this.blockIDMapper.getFormattedId(blockIdRaw, null);
    return '#' + formattedId;
  }
}

module.exports = LinkRewriter;
