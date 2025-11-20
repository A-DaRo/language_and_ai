/**
 * Mock Puppeteer browser and page for Worker tests
 */
class MockPage {
  constructor(options = {}) {
    this._url = null;
    this._title = options.title || 'Test Page';
    this._cookies = options.cookies || [];
    this._content = options.content || '<html><body></body></html>';
    this._evaluateResponses = options.evaluateResponses || [];
    this._evaluateIndex = 0;
    this.closed = false;
  }
  
  async goto(url, options) {
    if (!url || typeof url !== 'string') {
      throw new Error('Navigation failed: invalid URL');
    }
    this._url = url;
    return { status: () => 200 };
  }
  
  url() {
    return this._url || 'about:blank';
  }
  
  async title() {
    return this._title || 'Test Page';
  }
  
  async setViewport(viewport) {
    this._viewport = viewport;
  }
  
  async evaluate(fn) {
    // Return pre-configured responses
    if (this._evaluateResponses.length > 0) {
      return this._evaluateResponses[this._evaluateIndex++ % this._evaluateResponses.length];
    }
    
    // Default: execute the function in mock context
    if (typeof fn === 'function') {
      try {
        return fn();
      } catch (e) {
        return [];
      }
    }
    return [];
  }
  
  async content() {
    return this._content;
  }
  
  async cookies() {
    return this._cookies;
  }
  
  async setCookie(...cookies) {
    this._cookies.push(...cookies);
  }
  
  async close() {
    this.closed = true;
  }
  
  async waitForSelector(selector, options) {
    return {}; // Mock element
  }
  
  async $(selector) {
    return {}; // Mock element
  }
  
  async $$(selector) {
    return []; // Mock element array
  }
}

class MockBrowser {
  constructor(options = {}) {
    this.options = options;
    this._pages = [];
    this.closed = false;
  }
  
  async newPage() {
    const page = new MockPage(this.options.pageDefaults || {});
    this._pages.push(page);
    return page;
  }
  
  async close() {
    this.closed = true;
    for (const page of this._pages) {
      if (!page.closed) {
        await page.close();
      }
    }
  }
  
  async pages() {
    return this._pages;
  }
}

/**
 * Factory function matching puppeteer.launch() signature
 */
async function createMockBrowser(options = {}) {
  return new MockBrowser(options);
}

module.exports = {
  MockPage,
  MockBrowser,
  createMockBrowser,
  launch: createMockBrowser // Match puppeteer.launch() API
};
