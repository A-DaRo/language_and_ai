const path = require('path');
const PipelineStep = require('../PipelineStep');
const { HtmlFacadeFactory } = require('../../../html');

/**
 * @fileoverview Link Rewriting Step
 * @module worker/pipeline/steps/LinkRewriterStep
 * @description Rewrites internal page links to local relative paths.
 * 
 * @design PATH RESOLUTION STRATEGY
 * The linkRewriteMap from Master contains ROOT-RELATIVE paths (e.g., "Syllabus/index.html").
 * This step converts them to SOURCE-RELATIVE paths based on the current page's depth.
 * 
 * Example:
 * - Source page at depth 2: "Section/Page/index.html"
 * - Target root-relative: "OtherSection/Target/index.html"
 * - Computed source-relative: "../../OtherSection/Target/index.html"
 * 
 * Algorithm:
 * 1. Determine source page's directory from pathSegments (depth levels)
 * 2. For each target root-relative path, compute relative path from source
 * 3. Use path.relative() for cross-platform correctness
 * 4. **SAME-PAGE DETECTION**: If target ID matches source page ID, return anchor-only
 * 
 * @fixes Issue #2: Intra-page anchor resolution
 * Before: ToC links were converted to file paths (Lab_Session_1/index.html)
 * After: Same-page links return anchor-only (#heading-block-id)
 */

/**
 * @class LinkRewriterStep
 * @extends PipelineStep
 * @classdesc Rewrites internal page links to point to local HTML files.
 * Uses the linkRewriteMap from the payload to transform Notion URLs
 * into relative local paths, computing proper source-relative navigation.
 * 
 * Handles three link types:
 * 1. **Intra-page links**: Same page, returns #anchor only
 * 2. **Inter-page links**: Different page, returns ../relative/path/index.html#anchor
 * 3. **External links**: Left unchanged (not in linkRewriteMap)
 */
class LinkRewriterStep extends PipelineStep {
  constructor() {
    super('LinkRewriter');
  }
  
  /**
   * @method process
   * @summary Rewrites <a href> attributes for internal navigation.
   * @description
   * 1. Creates HtmlFacade for the Puppeteer page
   * 2. Extracts source page's path context from payload
   * 3. Queries all <a> tags using facade
   * 4. For each href matching a linkRewriteMap key:
   *    - **NEW**: If target ID === source page ID, returns anchor-only (#block-id)
   *    - Otherwise computes relative path from source to target
   *    - Preserves any anchor fragments (#block-id)
   * 5. Updates context.stats with rewrite count
   * 
   * @fixes Issue #2 - ToC links now correctly resolve to anchors on same page
   * 
   * @param {PipelineContext} context - Pipeline context.
   * @returns {Promise<void>}
   */
  async process(context) {
    const { page, payload, logger, stats } = context;
    
    const linkMap = payload.linkRewriteMap || {};
    const mapSize = Object.keys(linkMap).length;
    
    if (mapSize === 0) {
      logger.warn('LINK-REWRITE', 'No link rewrite map provided, skipping link rewriting');
      return;
    }
    
    logger.info('LINK-REWRITE', `Rewriting links using map with ${mapSize} entries...`);
    
    // Extract source path context
    const sourceDepth = payload.depth || 0;
    const sourcePathSegments = payload.pathSegments || [];
    const sourcePageId = payload.pageId || null; // Current page's Notion ID
    
    logger.debug('LINK-REWRITE', `Source page ID: ${sourcePageId?.slice(0, 8)}..., depth: ${sourceDepth}, segments: [${sourcePathSegments.join('/')}]`);
    
    // Build source directory path (for relative path computation)
    const sourceDir = sourcePathSegments.length > 0 
      ? sourcePathSegments.join('/') 
      : '';
    
    // Use HtmlFacade for DOM manipulation
    const facade = HtmlFacadeFactory.forPage(page);
    const links = await facade.query('a[href]');
    
    let rewriteCount = 0;
    let intraPageCount = 0;
    
    for (const link of links) {
      const href = await facade.getAttribute(link, 'href');
      if (!href) continue;
      
      // Try to match against link map keys (Notion page IDs)
      for (const [targetId, targetRootPath] of Object.entries(linkMap)) {
        if (href.includes(targetId)) {
          // Extract anchor fragment if present in original href
          const anchorMatch = href.match(/#([\w-]+)$/);
          const anchor = anchorMatch ? anchorMatch[1] : null;
          
          // ================================================================
          // SAME-PAGE DETECTION (Issue #2 Fix)
          // If the target page ID matches the source page ID, this is an
          // intra-page link (e.g., Table of Contents entry). Return only
          // the anchor hash, not a file path.
          // ================================================================
          if (targetId === sourcePageId) {
            if (anchor) {
              // Same page with anchor - return anchor only
              await facade.setAttribute(link, 'href', `#${anchor}`);
              intraPageCount++;
              logger.debug('LINK-REWRITE', `  INTRA-PAGE: ${targetId.slice(0, 8)}... → #${anchor}`);
            }
            // If no anchor, leave href unchanged (link to self with no anchor is unusual)
            break;
          }
          
          // ================================================================
          // INTER-PAGE LINK
          // Different page - compute relative path from source to target
          // ================================================================
          const anchorSuffix = anchorMatch ? anchorMatch[0] : '';
          
          // Compute source-relative path
          const relativePath = this._computeRelativePath(sourceDir, targetRootPath);
          const finalHref = relativePath + anchorSuffix;
          
          await facade.setAttribute(link, 'href', finalHref);
          rewriteCount++;
          
          logger.debug('LINK-REWRITE', `  ${targetId.slice(0, 8)}... → ${finalHref}`);
          break;
        }
      }
    }
    
    stats.linksRewritten = rewriteCount;
    stats.intraPageLinks = intraPageCount;
    
    const totalRewritten = rewriteCount + intraPageCount;
    logger.success('LINK-REWRITE', `Rewrote ${totalRewritten} link(s): ${rewriteCount} inter-page, ${intraPageCount} intra-page (anchors)`);
  }
  
  /**
   * Compute relative path from source directory to target path.
   * @private
   * @param {string} sourceDir - Source page's directory (e.g., "Section/Page")
   * @param {string} targetRootPath - Target's root-relative path (e.g., "Other/Target/index.html")
   * @returns {string} Source-relative path (e.g., "../../Other/Target/index.html")
   * 
   * @description Uses path.relative() to compute the navigation path.
   * Both paths are treated as relative to the same root directory.
   * 
   * Examples:
   * - sourceDir: "", targetRootPath: "Child/index.html" → "./Child/index.html"
   * - sourceDir: "Section", targetRootPath: "Section/index.html" → "./index.html"
   * - sourceDir: "Section/Page", targetRootPath: "Other/index.html" → "../../Other/index.html"
   */
  _computeRelativePath(sourceDir, targetRootPath) {
    // Handle root page case (empty sourceDir)
    if (!sourceDir) {
      // Source is at root, target path is already relative from root
      return targetRootPath.startsWith('./') ? targetRootPath : './' + targetRootPath;
    }
    
    // Extract target directory (remove /index.html)
    const targetPath = targetRootPath.replace(/\/index\.html$/, '');
    
    // Compute relative path using Node's path module
    // This handles all the ../ navigation correctly
    let relative = path.relative(sourceDir, targetPath || '.');
    
    // Convert backslashes to forward slashes (Windows compatibility)
    relative = relative.replace(/\\/g, '/');
    
    // Add index.html back
    if (relative === '' || relative === '.') {
      // Same directory
      return './index.html';
    }
    
    // Ensure proper format
    if (!relative.startsWith('.') && !relative.startsWith('/')) {
      relative = './' + relative;
    }
    
    return relative + '/index.html';
  }
}

module.exports = LinkRewriterStep;
