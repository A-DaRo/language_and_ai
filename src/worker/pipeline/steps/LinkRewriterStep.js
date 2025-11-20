const PipelineStep = require('../PipelineStep');

/**
 * @fileoverview Link Rewriting Step
 * @module worker/pipeline/steps/LinkRewriterStep
 */

/**
 * @class LinkRewriterStep
 * @extends PipelineStep
 * @classdesc Rewrites internal page links to point to local HTML files.
 * Uses the linkRewriteMap from the payload to transform Notion URLs
 * into relative local paths.
 */
class LinkRewriterStep extends PipelineStep {
  constructor() {
    super('LinkRewriter');
  }
  
  /**
   * @method process
   * @summary Rewrites <a href> attributes for internal navigation.
   * @description
   * 1. Evaluates DOM to find all <a> tags
   * 2. For each href that matches a key in payload.linkRewriteMap
   * 3. Replaces the href with the corresponding local relative path
   * 4. Updates context.stats with rewrite count
   * 
   * This enables offline browsing by converting Notion URLs like:
   *   "https://notion.site/Page-Name-29abc123..."
   * Into local paths like:
   *   "../Page_Name/index.html"
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
    
    // Rewrite links in the browser context
    const rewriteCount = await page.evaluate((map) => {
      let count = 0;
      const links = document.querySelectorAll('a[href]');
      
      links.forEach(link => {
        const href = link.getAttribute('href');
        if (!href) return;
        
        // Try to match against link map keys
        // Keys might be full URLs or page IDs
        for (const [key, value] of Object.entries(map)) {
          if (href.includes(key)) {
            link.setAttribute('href', value);
            count++;
            break;
          }
        }
      });
      
      return count;
    }, linkMap);
    
    stats.linksRewritten = rewriteCount;
    logger.success('LINK-REWRITE', `Rewrote ${rewriteCount} internal link(s)`);
  }
}

module.exports = LinkRewriterStep;
