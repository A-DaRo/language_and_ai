/**
 * Configuration settings for the Notion scraper
 */
class Config {
  constructor() {
    // Main configuration
    this.NOTION_PAGE_URL = 'https://mctenthij.notion.site/JBC090-Language-AI-29d979eeca9f81469905f51d65beefae';
    this.OUTPUT_DIR = './JBC090-Language-AI';
    this.LOG_DIR = './';
    this.LOG_FILE_ENABLED = true;
    this.MAX_EXPANSION_DEPTH = 100;
    this.MAX_RECURSION_DEPTH = 100;
    this.MAIN_CONTENT_SELECTOR = 'div.notion-page-content';
    
    // Timing configuration (in milliseconds)
    this.TIMEOUT_PAGE_LOAD = 60000;
    this.TIMEOUT_COOKIE_BANNER = 5000;
    this.TIMEOUT_DIALOG = 5000;
    this.TIMEOUT_NAVIGATION = 30000;
    this.WAIT_AFTER_COOKIE = 2000;
    this.WAIT_AFTER_TOGGLE = 2000;
    this.WAIT_AFTER_LOAD_MORE = 1500;
    this.WAIT_AFTER_SCROLL = 2000;
    this.SCROLL_DISTANCE = 100;
    this.SCROLL_INTERVAL = 100;
    this.COOKIE_MAX_RETRIES = 2;
    this.COOKIE_RETRY_DELAY = 750;
    this.COOKIE_HANDLE_ALL_PAGES = true;
    this.CLUSTER_RETRY_LIMIT = 3;
    this.CLUSTER_RETRY_DELAY = 1500;
    this.CLUSTER_TASK_TIMEOUT = Math.max(this.TIMEOUT_PAGE_LOAD, 90000);
    this.ENABLE_PARALLEL_DISCOVERY = true;
    
    // Notion URL patterns
    this.NOTION_DOMAIN_PATTERN = /^https?:\/\/[^\/]*notion\.site/;
    this.NOTION_PAGE_ID_PATTERN = /[a-f0-9]{32}/;
    
    // Selectors
    this.SELECTORS = {
      cookieBanner: 'div[aria-live="polite"]',
      confirmDialog: 'div[role="dialog"][aria-modal="true"]',
      toggleButton: '[role="button"][aria-expanded="false"]',
      toggleIcon: 'svg.arrowCaretDownFillSmall',
      notionLink: 'a[href*="/"]',
      pageIcon: '.notion-record-icon',
      heading: 'h1, h2, h3, [data-block-id]'
    };
  }
  
  /**
   * Get base URL from the Notion page URL
   */
  getBaseUrl() {
    const url = new URL(this.NOTION_PAGE_URL);
    return `${url.protocol}//${url.host}`;
  }
  
  /**
   * Check if a URL is a Notion page
   */
  isNotionUrl(url) {
    return this.NOTION_DOMAIN_PATTERN.test(url);
  }
  
  /**
   * Extract page name from Notion URL
   * Notion URLs follow pattern: /Page-Name-29d979eeca9f81469905f51d65beefae
   * The -[32-char-hex] suffix is always present and serves as a reliable delimiter
   * @param {string} url - Full Notion URL
   * @returns {string} Extracted page name or 'Untitled'
   */
  extractPageNameFromUrl(url) {
    try {
      const urlObj = new URL(url);
      const pathname = urlObj.pathname;
      
      // Remove leading/trailing slashes
      const cleanPath = pathname.replace(/^\/+|\/+$/g, '');
      
      // Match the pattern: everything before -[32 hex chars]
      // Notion page IDs are exactly 32 characters (no hyphens in the ID itself)
      const match = cleanPath.match(/^(.+?)(?:-[a-f0-9]{32})$/i);
      
      if (match && match[1]) {
        // Replace URL hyphens with spaces for readability
        return match[1]
          .replace(/-/g, ' ')
          .replace(/%20/g, ' ')
          .trim();
      }
      
      // Fallback: use the last segment before any hash
      const segments = cleanPath.split('/');
      const lastSegment = segments[segments.length - 1];
      
      if (lastSegment) {
        return lastSegment
          .replace(/-[a-f0-9]{32}$/i, '') // Remove hash if present
          .replace(/-/g, ' ')
          .replace(/%20/g, ' ')
          .trim();
      }
      
      return 'Untitled';
    } catch (e) {
      return 'Untitled';
    }
  }
}

module.exports = Config;
