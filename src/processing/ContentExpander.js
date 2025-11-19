/**
 * @classdesc Expands toggles, databases, and other collapsible content on Notion pages.
 * 
 * Implements an aggressive expansion strategy using iterative click-and-wait cycles
 * to reveal all hidden content. This is critical for ensuring complete page capture,
 * as Notion heavily uses collapsible UI elements that hide content by default.
 * 
 * The expansion process includes:
 * - Scrolling to bottom to trigger lazy-loading
 * - Multi-iteration toggle expansion to handle nested content
 * - Smart element detection to avoid clicking destructive actions
 * 
 * @see PageProcessor#scrapePage
 */
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
   * @description Executes the complete content expansion workflow:
   * 1. Scrolls to page bottom to trigger lazy-loaded content
   * 2. Iteratively expands all toggles and collapsible elements
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   * 
   * @see _scrollToBottom
   * @see _expandToggles
   */
  async expandAll(page) {
    await this._scrollToBottom(page);
    await this._expandToggles(page);
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
   * @summary Aggressively expand ALL toggles, buttons, tabs, and interactive elements.
   * 
   * @description Uses a "click and wait" loop to ensure all hidden content is revealed.
   * Iterates up to maxIterations (default 2) to handle nested collapsible elements.
   * 
   * Selection strategy:
   * - Targets elements with aria-expanded="false"
   * - Looks for Notion-specific toggle classes
   * - Finds generic buttons and expandable elements
   * - Excludes destructive actions (delete, remove, share)
   * 
   * After each iteration, waits for network idle or fixed timeout to allow
   * content to load.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @returns {Promise<void>}
   * @private
   */
  async _expandToggles(page) {
    this.logger.info('TOGGLE', 'Starting AGGRESSIVE content expansion...');
    
    let totalExpanded = 0;
    let iteration = 0;
    const maxIterations = 2; // Prevent infinite loops
    
    while (iteration < maxIterations) {
      iteration++;
      this.logger.info('TOGGLE', `Expansion iteration ${iteration}/${maxIterations}...`);
      
      // Get all potentially expandable elements
      const result = await page.evaluate((scopeSel) => {
        const scopeElement = document.querySelector(scopeSel) || document.body;
        
        // Find ALL potentially expandable elements
        const selectors = [
          '[role="button"][aria-expanded="false"]',  // Collapsed toggles
          '.notion-toggle-block',                     // Toggle blocks
          '[data-block-id*="toggle"]',               // Any toggle blocks
          'div[role="button"]',                       // Generic buttons
          'button:not([disabled])',                   // All enabled buttons
          '[class*="expand"]',                        // Elements with "expand" in class
          '[class*="collapse"]',                      // Elements with "collapse" in class
          '[onclick]',                                // Elements with click handlers
        ];
        
        const allElements = [];
        selectors.forEach(sel => {
          const elements = scopeElement.querySelectorAll(sel);
          elements.forEach(el => {
            // Skip if already in list or if hidden
            if (!allElements.includes(el) && el.offsetParent !== null) {
              allElements.push(el);
            }
          });
        });
        
        // Click all elements
        let clicked = 0;
        allElements.forEach(element => {
          try {
            // Check if element might be expandable
            const text = element.textContent.toLowerCase();
            const ariaExpanded = element.getAttribute('aria-expanded');
            
            // Skip elements that are clearly not expandable
            if (text.includes('delete') || text.includes('remove') || text.includes('share')) {
              return;
            }
            
            // Click if it looks expandable
            if (ariaExpanded === 'false' || 
                text.includes('show') || 
                text.includes('expand') ||
                element.classList.contains('notion-toggle-block')) {
              element.click();
              clicked++;
            }
          } catch (e) {
            // Silently skip elements that can't be clicked
          }
        });
        
        return { clicked, totalFound: allElements.length };
      }, this.config.MAIN_CONTENT_SELECTOR);

      this.logger.info('TOGGLE', `Found ${result.totalFound} interactive elements, clicked ${result.clicked}`);
      
      if (result.clicked === 0) {
        this.logger.success('TOGGLE', 'No more expandable elements found. Content fully expanded.');
        break;
      }

      totalExpanded += result.clicked;
      
      // Wait for content to load and any network requests to complete
      await Promise.race([
        page.waitForNetworkIdle({ timeout: 3000 }).catch(() => {}),
        new Promise(resolve => setTimeout(resolve, 2000))
      ]);
    }

    if (iteration >= maxIterations) {
      this.logger.warn('TOGGLE', `Reached maximum iterations (${maxIterations}). Some content may still be hidden.`);
    }
    
    this.logger.success('TOGGLE', `Aggressive expansion complete. Total elements expanded: ${totalExpanded}`);
  }
}

module.exports = ContentExpander;
