/**
 * Expands toggles, databases, and other collapsible content on Notion pages
 */
class ContentExpander {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
  }
  
  /**
   * Expand all content on the page
   */
  async expandAll(page) {
    await this._expandDatabases(page);
    await this._scrollToBottom(page);
    await this._expandToggles(page);
  }
  
  /**
   * Expand database views by clicking "Load more" buttons
   */
  async _expandDatabases(page) {
    this.logger.info('DATABASE', 'Looking for database/gallery "Load more" buttons...');
    let loadMoreClicked = 0;
    
    try {
      while (true) {
        const clicked = await page.evaluate(() => {
          const loadMoreButtons = Array.from(document.querySelectorAll('div[role="button"]'))
            .filter(btn => {
              const text = btn.textContent.toLowerCase();
              return text.includes('load more') || 
                     text.includes('show more') || 
                     text.includes('view more');
            });
          
          if (loadMoreButtons.length > 0) {
            loadMoreButtons[0].click();
            return true;
          }
          return false;
        });
        
        if (!clicked) break;
        
        loadMoreClicked++;
        this.logger.info('DATABASE', `Clicked "Load more" button (${loadMoreClicked}). Waiting for content...`);
        await new Promise(resolve => setTimeout(resolve, this.config.WAIT_AFTER_LOAD_MORE));
      }
      
      if (loadMoreClicked > 0) {
        this.logger.success('DATABASE', `Expanded ${loadMoreClicked} database views.`);
      } else {
        this.logger.info('DATABASE', 'No "Load more" buttons found.');
      }
    } catch (error) {
      this.logger.error('DATABASE', 'Error while expanding databases', error);
    }
  }
  
  /**
   * Scroll to the bottom of the page to trigger lazy-loading
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
    
    // Scroll back to top
    await page.evaluate(() => window.scrollTo(0, 0));
    this.logger.info('SCROLL', 'Scrolled back to top.');
  }
  
  /**
   * Aggressively expand ALL toggles, buttons, tabs, and interactive elements
   * Uses a "click and wait" loop to ensure all hidden content is revealed
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
