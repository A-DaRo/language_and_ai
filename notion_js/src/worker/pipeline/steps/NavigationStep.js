const PipelineStep = require('../PipelineStep');

/**
 * @fileoverview Navigation Step
 * @module worker/pipeline/steps/NavigationStep
 */

/**
 * @class NavigationStep
 * @extends PipelineStep
 * @classdesc Handles page navigation with strict network idle checks.
 * Sets cookies before navigation and waits for network idle to ensure
 * all initial resources are loaded.
 */
class NavigationStep extends PipelineStep {
  constructor() {
    super('Navigation');
  }
  
  /**
   * @method process
   * @summary Navigates Puppeteer page to target URL.
   * @description Sets cookies from payload. Calls `page.goto` with `networkidle0`.
   * Throws if navigation fails (404, timeout, etc.).
   * 
   * @param {PipelineContext} context - Pipeline context.
   * @returns {Promise<void>}
   * @throws {Error} If navigation fails or times out.
   */
  async process(context) {
    const { page, payload, config, logger } = context;
    
    // Set cookies if provided
    if (payload.cookies && payload.cookies.length > 0) {
      await page.setCookie(...payload.cookies);
      logger.debug('NAVIGATION', `Set ${payload.cookies.length} cookie(s)`);
    }
    
    // Navigate to target URL
    logger.info('NAVIGATION', `Navigating to: ${payload.url}`);
    
    const response = await page.goto(payload.url, {
      waitUntil: 'networkidle0',
      timeout: config.TIMEOUT_PAGE_LOAD
    });
    
    // Check response status
    const status = response.status();
    if (status >= 400) {
      throw new Error(`Navigation failed with status ${status}`);
    }
    
    // Get page title for logging
    const title = await page.title();
    logger.success('NAVIGATION', `Loaded: "${title}" (status: ${status})`);
    
    // Store page title in context for other steps
    context.pageTitle = title;
  }
}

module.exports = NavigationStep;
