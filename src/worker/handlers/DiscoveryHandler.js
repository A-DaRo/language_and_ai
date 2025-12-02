/**
 * @fileoverview Discovery task handler
 * @module worker/handlers/DiscoveryHandler
 * @description Handles IPC_DISCOVER messages in worker process.
 */

const { MESSAGE_TYPES } = require('../../core/ProtocolDefinitions');
const LinkExtractor = require('../../extraction/LinkExtractor');
const Logger = require('../../core/Logger');

/**
 * @class DiscoveryHandler
 * @classdesc Executes lightweight page analysis (title and links only).
 */
class DiscoveryHandler {
  /**
   * @param {import('puppeteer').Browser} browser - Puppeteer browser
   * @param {Object} context - Worker context
   */
  constructor(browser, context = {}) {
    this.browser = browser;
    this.page = null;
    this.config = context.config;
    this.cookies = context.cookies || [];
    this.titleRegistry = context.titleRegistry || {};
    this.logger = Logger.getInstance();
    this.linkExtractor = new LinkExtractor(this.config, this.logger);
  }

  /**
   * @async
   * @summary Navigates to URL and extracts metadata
   * @description
   * 1. Navigates to payload.url (waitUntil: domcontentloaded)
   * 2. Extracts page title
   * 3. Extracts internal links via LinkExtractor
   * 4. Captures cookies if isFirstPage is true
   * @param {Object} payload - Discovery payload
   * @returns {Promise<Object>} Discovery result
   */
  async handle(payload) {
    // Create or reuse page
    if (!this.page) {
      this.page = await this.browser.newPage();
      await this.page.setViewport({ width: 1920, height: 1080 });
    }

    // Set cookies if available
    if (this.cookies.length > 0) {
      await this.page.setCookie(...this.cookies);
    }

    // Navigate to page
    this.logger.info('Discovery', `Navigating to: ${payload.url}`);
    await this.page.goto(payload.url, {
      waitUntil: 'networkidle0',
      timeout: 60000
    });

    // Extract links
    const links = await this.linkExtractor.extractLinks(this.page, payload.url);

    // Normalize link format
    const normalizedLinks = links.map(link => ({
      url: link.url,
      text: link.title,
      section: link.section,
      subsection: link.subsection
    }));

    // Resolve page name from H1 element (primary) or fallback to URL/title
    let pageName = null;
    
    // Primary: Extract from H1 element with Notion-specific selectors
    // Notion page titles are in <h1> elements with specific attributes
    pageName = await this.page.evaluate(() => {
      // Try to find the main page title H1
      // Notion uses H1 with placeholder="New page" for page titles
      const h1Selectors = [
        'h1[placeholder="New page"]',  // Main page title
        'h1[data-content-editable-leaf="true"]',  // Alternative selector
        '.notion-page-block h1',  // Page block title
        'h1'  // Generic fallback
      ];
      
      for (const selector of h1Selectors) {
        const h1 = document.querySelector(selector);
        if (h1) {
          // Get text content, handling nested spans
          // The text might be directly in H1 or in a nested span
          const text = h1.textContent || h1.innerText || '';
          const trimmed = text.trim();
          
          // Skip if it's just the placeholder or empty
          if (trimmed && trimmed !== 'New page' && trimmed !== 'Untitled') {
            return trimmed;
          }
        }
      }
      
      return null;
    });
    
    // Fallback: Use URL slug if H1 extraction failed
    if (!pageName) {
      const finalUrl = this.page.url();
      const nameMatch = finalUrl.match(/\/([^\/]+?)(-29[a-f0-9]{30})/i);
      if (nameMatch && nameMatch[1]) {
        pageName = nameMatch[1].replace(/-/g, ' ');
      }
    }
    
    // Final fallback: Use page title or URL
    if (!pageName) {
      const pageTitle = await this.page.title();
      if (pageTitle && pageTitle !== 'Untitled' && pageTitle !== 'Notion') {
        pageName = pageTitle;
      } else {
        const origNameMatch = payload.url.match(/\/([^\/]+?)(-29[a-f0-9]{30})/i);
        pageName = origNameMatch && origNameMatch[1]
          ? origNameMatch[1].replace(/-/g, ' ')
          : 'Untitled';
      }
    }

    // Capture cookies on first page
    let capturedCookies = null;
    if (payload.isFirstPage) {
      capturedCookies = await this.page.cookies();
      this.logger.info('Discovery', `Captured ${capturedCookies.length} cookie(s)`);
    }

    return {
      success: true,
      pageId: payload.pageId,
      url: payload.url,
      resolvedTitle: pageName || 'Untitled',
      links: normalizedLinks,
      cookies: capturedCookies,
      metadata: {
        depth: payload.depth,
        parentId: payload.parentId
      }
    };
  }

  /**
   * @async
   * @summary Cleanup resources
   */
  async cleanup() {
    if (this.page) {
      await this.page.close();
      this.page = null;
    }
  }
}

module.exports = DiscoveryHandler;
