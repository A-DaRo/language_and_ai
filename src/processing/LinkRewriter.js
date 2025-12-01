const fs = require('fs/promises');
const path = require('path');
const BlockIDMapper = require('./BlockIDMapper');
const { PathResolverFactory } = require('../domain/path');
const { HtmlFacadeFactory } = require('../html');

/**
 * @classdesc Handles offline link rewriting and CSS localization for scraped pages.
 * 
 * Performs post-scraping link transformation to enable offline browsing:
 * - Rewrites internal Notion page links to point to local HTML files
 * - Calculates relative paths between pages based on directory structure
 * - Downloads and localizes external CSS stylesheets
 * - Preserves external links unchanged
 * - **Preserves block IDs (anchors) during rewriting**
 * - Uses HtmlFacade for context-agnostic DOM manipulation
 * 
 * @design PATH RESOLVER PATTERN
 * Link resolution uses the PathResolverFactory to select the appropriate resolver:
 * - IntraPageResolver: Same-page links (returns anchor hash only)
 * - InterPageResolver: Cross-page links (returns relative path + optional anchor)
 * - ExternalUrlResolver: External URLs (preserved unchanged)
 * 
 * @design HTML FACADE PATTERN
 * DOM manipulation is performed through HtmlFacade abstraction, allowing
 * the same code to work with both Puppeteer pages (browser context) and
 * JSDOM (server context for post-processing).
 * 
 * @see PathResolverFactory - Resolver selection
 * @see HtmlFacadeFactory - DOM abstraction factory
 * @see CssDownloader
 */
class LinkRewriter {
  /**
   * @param {Config} config - Configuration object.
   * @param {Logger} logger - Logger instance.
   * @param {CssDownloader} cssDownloader - CSS downloader instance.
   * @param {PathResolverFactory} [pathResolverFactory=null] - Optional custom factory.
   */
  constructor(config, logger, cssDownloader, pathResolverFactory = null) {
    this.config = config;
    this.logger = logger;
    this.cssDownloader = cssDownloader;
    this.blockIDMapper = new BlockIDMapper();
    
    // Initialize path resolver factory (use provided or create default)
    this.pathResolverFactory = pathResolverFactory || new PathResolverFactory(config, logger);
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
      
      // Use HtmlFacade for context-agnostic DOM manipulation
      const facade = await HtmlFacadeFactory.fromFile(htmlFilePath);
      
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
        // CssDownloader still needs the raw DOM for now
        const cssResult = await this.cssDownloader.downloadAndRewriteCss(facade.getDom(), pageDir, pageContext.url);
        if (cssResult.modified) {
          modified = true;
          this.logger.success(
            'CSS',
            `Localized ${cssResult.stylesheets} stylesheet(s), ${cssResult.assets} dependent asset(s), ${cssResult.inlineStyles} inline style block(s) in ${pageContext.title}`
          );
        }
      }
      
      const links = await facade.query('a[href]');
      
      for (const link of links) {
        const href = await facade.getAttribute(link, 'href');
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
            const idMatch = urlPart.match(/([a-f0-9]{32})/i);
            if (idMatch) {
              const id = idMatch[1];
              targetContext = idToContextMap.get(id);
            }
          }
          
          if (targetContext) {
            // Use PathResolverFactory for unified path resolution
            const newHref = this.pathResolverFactory.resolve({
              source: pageContext,
              target: targetContext,
              blockId: blockIdRaw,
              blockMapCache
            });

            // Handle edge case: same page with no block ID returns empty string
            const finalHref = newHref === '' ? 'index.html' : newHref;

            await facade.setAttribute(link, 'href', finalHref);
            rewriteCount++;
            this.logger.debug('LINK-REWRITE', `${pageContext.title}: ${href} -> ${finalHref}`);
          }
        } catch (error) {
          continue;
        }
      }
      
      if (rewriteCount > 0 || modified) {
        await facade.saveToFile(htmlFilePath);
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
   * Parse href to separate URL and raw block ID.
   * 
   * @description Extracts the URL portion and any block ID anchor from a full href.
   * Notion uses block IDs after the hash symbol (e.g., #29d979eeca9f4abc...).
   * 
   * @private
   * @param {string} href - Full href attribute
   * @returns {Object} { urlPart: string, blockIdRaw: string|null }
   * 
   * @example
   * _parseHref('https://notion.so/page-abc#29d979eeca9f')
   * // Returns: { urlPart: 'https://notion.so/page-abc', blockIdRaw: '29d979eeca9f' }
   */
  _parseHref(href) {
    // Split on hash first to get block ID
    const hashIndex = href.indexOf('#');
    let urlPart = href;
    let blockIdRaw = null;
    
    if (hashIndex !== -1) {
      urlPart = href.substring(0, hashIndex);
      blockIdRaw = href.substring(hashIndex + 1);
      
      // Validate block ID format (should be hex characters)
      if (blockIdRaw && !/^[a-f0-9-]+$/i.test(blockIdRaw)) {
        // Not a valid block ID format, might be a generic anchor
        // Still preserve it for the hash
      }
    }
    
    // Remove query string from URL part
    const queryIndex = urlPart.indexOf('?');
    if (queryIndex !== -1) {
      urlPart = urlPart.substring(0, queryIndex);
    }
    
    return { urlPart, blockIdRaw };
  }

  /**
   * Build anchor hash for block ID.
   * 
   * @description Converts raw block ID to formatted UUID using block map if available.
   * This ensures block anchors work correctly in the offline version.
   * 
   * Algorithm:
   * 1. If no blockIdRaw provided, return empty string
   * 2. If block map cache available for target page, look up formatted ID
   * 3. If found in map, return '#' + formattedId
   * 4. If not found, fall back to formatting raw ID as UUID
   * 5. Always preserve the block ID even without a map (use raw/formatted fallback)
   * 
   * @private
   * @param {string|null} blockIdRaw - Raw block ID from URL (e.g., '29d979eeca9f4abc...')
   * @param {PageContext} targetContext - Target page context
   * @param {Map<string, Map>} blockMapCache - Cache of block ID maps (pageId -> blockMap)
   * @returns {string} Formatted hash (e.g., '#29d979ee-ca9f-...') or empty string
   * 
   * @example
   * // With block map cache hit
   * _buildAnchorHash('29d979eeca9f4abc', targetCtx, cache)
   * // Returns: '#29d979ee-ca9f-4abc-...'
   * 
   * @example
   * // Without block map (fallback formatting)
   * _buildAnchorHash('29d979eeca9f4abc', targetCtx, null)
   * // Returns: '#29d979ee-ca9f-4abc-...' (UUID formatted)
   */
  _buildAnchorHash(blockIdRaw, targetContext, blockMapCache) {
    if (!blockIdRaw) {
      return '';
    }

    // Get block map for target page if cache provided
    if (blockMapCache && targetContext.id) {
      const blockMap = blockMapCache.get(targetContext.id);
      if (blockMap && blockMap.size > 0) {
        const formattedId = this.blockIDMapper.getFormattedId(blockIdRaw, blockMap);
        return '#' + formattedId;
      }
    }

    // Fallback: format the raw ID directly as UUID
    // This ensures block IDs are preserved even without a block map
    const formattedId = this.blockIDMapper.getFormattedId(blockIdRaw, null);
    return '#' + formattedId;
  }
}

module.exports = LinkRewriter;
