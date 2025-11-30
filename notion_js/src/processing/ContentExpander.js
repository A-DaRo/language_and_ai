/**
 * @classdesc Prepares Notion pages for scraping by triggering lazy-loading and closing overlays.
 * 
 * **Architecture Change (v2.0)**: Toggle expansion is now handled separately by:
 * - {@link ToggleStateCapture} - Captures dual-state (collapsed/expanded) content
 * - {@link ToggleCaptureStep} - Pipeline step that orchestrates capture + injection
 * - {@link OfflineToggleController} - Runtime JavaScript for offline interactivity
 * 
 * This class now focuses solely on:
 * - Scrolling to bottom to trigger lazy-loaded content
 * - Closing overlays/modals that might obscure content
 * 
 * The previous aggressive expansion approach was disabled because:
 * - It destroyed toggle interactivity (all toggles permanently expanded)
 * - Risk of clicking destructive actions (delete, share, export)
 * - Breaking page layout with simultaneous expansions
 * 
 * @see ToggleStateCapture
 * @see ToggleCaptureStep
 * @see OfflineToggleController
 * @see PageProcessor#scrapePage
 */

const { HtmlFacadeFactory } = require('../html');

class ContentExpander {
  /**
   * @param {Config} config - Configuration object containing scroll and timeout settings.
   * @param {Logger} logger - Logger instance for expansion progress tracking.
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
  }
  
  /**
   * @summary Expand all content on the page (scroll only).
   * 
   * @description Executes the page preparation workflow:
   * 1. Scrolls to page bottom to trigger lazy-loaded content
   * 
   * **Note**: Toggle expansion is now handled by {@link ToggleCaptureStep}
   * in the scraping pipeline. This method only triggers lazy-loading.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   * 
   * @see _scrollToBottom
   * @see ToggleCaptureStep
   */
  async expandAll(page) {
    await this._scrollToBottom(page);
  }
  
  /**
   * @summary Prepare page for scraping (scroll and close overlays).
   * 
   * @description Replaces the aggressive expansion with a gentler approach:
   * 1. Scrolls to bottom to trigger lazy-loading
   * 2. Attempts to close any open overlays/modals that might obscure content
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   */
  async preparePage(page) {
    await this._scrollToBottom(page);
    await this._closeOverlays(page);
  }

  /**
   * @summary Scroll to the bottom of the page to trigger lazy-loading.
   * 
   * @description Performs incremental scrolling using page.evaluate() to simulate
   * natural user behavior. This triggers Notion's lazy-loading mechanism for images,
   * content blocks, and other dynamic elements.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   * @private
   */
  async _scrollToBottom(page) {
    this.logger.info('SCROLL', 'Scrolling to bottom of page to trigger lazy-loading...');
    
    await page.evaluate((distance, interval) => {
      return new Promise((resolve) => {
        let totalHeight = 0;
        const timer = setInterval(() => {
          const scrollHeight = document.body.scrollHeight;
          window.scrollBy(0, distance);
          totalHeight += distance;
          
          if (totalHeight >= scrollHeight) {
            clearInterval(timer);
            resolve();
          }
        }, interval);
      });
    }, this.config.SCROLL_DISTANCE, this.config.SCROLL_INTERVAL);
    
    this.logger.info('SCROLL', 'Reached bottom of page. Waiting for content to stabilize...');
    await new Promise(resolve => setTimeout(resolve, this.config.WAIT_AFTER_SCROLL));
  }
  
  /**
   * @summary Attempt to close any open overlays, modals, or sidebars.
   * 
   * @description Checks for common overlay selectors and attempts to close them
   * by clicking close buttons or pressing Escape.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   * @private
   */
  async _closeOverlays(page) {
    this.logger.info('CLEANUP', 'Checking for open overlays or sidebars...');
    
    try {
      // Try pressing Escape first (common way to close modals)
      await page.keyboard.press('Escape');
      
      // Create HtmlFacade for element querying
      const facade = HtmlFacadeFactory.forPage(page);
      
      // Check for specific close buttons if Escape didn't work or as backup
      const closeSelectors = [
        '[role="dialog"] button[aria-label="Close"]',
        '.notion-overlay-container [role="button"]',
        '.notion-help-button',
        '[aria-modal="true"] button'
      ];
      
      for (const selector of closeSelectors) {
        const element = await facade.queryOne(selector);
        if (element) {
          // Use click() method on the wrapper element
          await element.click();
        }
      }
      
      // Wait a bit for animations
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (error) {
      this.logger.debug('CLEANUP', `Error closing overlays: ${error.message}`);
    }
  }
}

module.exports = ContentExpander;
