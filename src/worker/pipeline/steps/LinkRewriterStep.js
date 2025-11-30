const PipelineStep = require('../PipelineStep');
const { HtmlFacadeFactory } = require('../../../html');

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
 * 
 * @design HTML FACADE PATTERN
 * Uses HtmlFacade for context-agnostic DOM manipulation, allowing the
 * same rewriting logic to work with both live Puppeteer pages and
 * saved HTML files.
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
   * 2. Queries all <a> tags using facade
   * 3. For each href that matches a key in payload.linkRewriteMap
   * 4. Replaces the href with the corresponding local relative path
   * 5. Updates context.stats with rewrite count
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
    
    // Use HtmlFacade for DOM manipulation
    const facade = HtmlFacadeFactory.forPage(page);
    const links = await facade.query('a[href]');
    
    let rewriteCount = 0;
    
    for (const link of links) {
      const href = await facade.getAttribute(link, 'href');
      if (!href) continue;
      
      // Try to match against link map keys
      for (const [key, value] of Object.entries(linkMap)) {
        if (href.includes(key)) {
          await facade.setAttribute(link, 'href', value);
          rewriteCount++;
          break;
        }
      }
    }
    
    stats.linksRewritten = rewriteCount;
    logger.success('LINK-REWRITE', `Rewrote ${rewriteCount} internal link(s)`);
  }
}

module.exports = LinkRewriterStep;
