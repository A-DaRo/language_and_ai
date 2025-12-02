/**
 * @classdesc Prepares Notion pages for scraping by triggering lazy-loading,
 * expanding toggles, and closing overlays.
 * 
 * **Architecture Change (v3.0)**: Toggle expansion is now simplified.
 * All toggles are expanded permanently for offline capture. No interactivity
 * is preserved - the offline version shows all content in expanded state.
 * 
 * This class handles:
 * - Scrolling to bottom to trigger lazy-loaded content
 * - Expanding all toggle blocks so their content is visible
 * - Closing overlays/modals that might obscure content
 * 
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
   * @summary Expand all content on the page.
   * 
   * @description Executes the page preparation workflow:
   * 1. Scrolls to page bottom to trigger lazy-loaded content
   * 2. Expands all toggle blocks so content is captured
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   */
  async expandAll(page) {
    await this._scrollToBottom(page);
    await this._expandAllToggles(page);
  }
  
  /**
   * @summary Prepare page for scraping (scroll, expand toggles, close overlays).
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   */
  async preparePage(page) {
    await this._scrollToBottom(page);
    await this._expandAllToggles(page);
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
   * @summary Expand all toggle blocks on the page.
   * 
   * @description Finds all Notion toggle blocks and clicks them to expand.
   * Uses the aria-expanded attribute to identify collapsed toggles.
   * Toggles are expanded permanently for offline capture.
   * 
   * Safety features:
   * - Skips potentially destructive toggles (delete, share, export)
   * - Waits for content to load after each expansion
   * - Limits maximum toggles to prevent infinite loops
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<{expanded: number, skipped: number}>} Expansion statistics
   * @private
   */
  async _expandAllToggles(page) {
    this.logger.info('TOGGLE', 'Expanding all toggle blocks...');
    
    const result = await page.evaluate(async () => {
      const stats = { expanded: 0, skipped: 0, errors: 0 };
      const maxToggles = 100;
      const animationWait = 300;
      const contentWait = 500;
      
      // Patterns to skip (potentially destructive actions)
      const skipPatterns = ['delete', 'remove', 'share', 'export', 'duplicate'];
      
      // Find all toggle buttons that are currently collapsed
      const getCollapsedToggles = () => {
        return Array.from(document.querySelectorAll(
          '.notion-toggle-block [role="button"][aria-expanded="false"]'
        ));
      };
      
      // Check if toggle text contains dangerous patterns
      const shouldSkip = (button) => {
        const text = (button.textContent || '').toLowerCase();
        return skipPatterns.some(pattern => text.includes(pattern));
      };
      
      // Wait for a specified time
      const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
      
      // Expand toggles iteratively (some may be nested)
      let iteration = 0;
      const maxIterations = 5;
      
      while (iteration < maxIterations) {
        const toggles = getCollapsedToggles();
        if (toggles.length === 0) break;
        
        for (const button of toggles) {
          if (stats.expanded >= maxToggles) break;
          
          try {
            if (shouldSkip(button)) {
              stats.skipped++;
              continue;
            }
            
            // Click to expand
            button.click();
            stats.expanded++;
            
            // Wait for animation and content to load
            await wait(animationWait);
          } catch (error) {
            stats.errors++;
          }
        }
        
        // Wait for any async content loading
        await wait(contentWait);
        iteration++;
      }
      
      return stats;
    });
    
    this.logger.success('TOGGLE', 
      `Expanded ${result.expanded} toggle(s), skipped ${result.skipped}, errors ${result.errors}`);
    
    return result;
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
