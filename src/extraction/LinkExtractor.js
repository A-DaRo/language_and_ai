/**
 * @classdesc Extracts and categorizes links from Notion pages.
 * 
 * Identifies internal Notion page links and extracts contextual metadata
 * including hierarchy information (sections, subsections) based on DOM structure.
 * 
 * The extraction process:
 * - Filters for internal Notion links (same domain)
 * - Resolves relative URLs to absolute URLs
 * - Deduplicates discovered links
 * - Extracts hierarchical context from parent elements
 * - Excludes self-references and invalid URLs
 * 
 * @see RecursiveScraper#discover
 * @see PageProcessor#scrapePage
 */
class LinkExtractor {
  /**
   * @param {Config} config - Configuration object for base URL resolution.
   * @param {Logger} logger - Logger instance for extraction progress tracking.
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
  }
  
  /**
   * @summary Extract all internal Notion page links with their hierarchical context.
   * 
   * @description Performs comprehensive link extraction using page.evaluate() to run
   * in the browser context. The extraction process:
   * 1. Queries all anchor elements with href attributes
   * 2. Filters for internal links matching the base domain
   * 3. Resolves relative URLs to absolute URLs
   * 4. Deduplicates based on URL
   * 5. Extracts hierarchical context (sections, subsections) from DOM structure
   * 6. Excludes the current page URL to prevent self-references
   * 
   * @param {Page} page - Puppeteer page object.
   * @param {string} currentUrl - Current page URL to filter out (prevent self-references).
   * @param {string} [baseUrl=null] - Optional base URL for composing internal links (defaults to config base URL).
   * @returns {Promise<Array<{url: string, title: string, section?: string, subsection?: string}>>} Array of link objects with metadata.
   * 
   * @see Config#getBaseUrl
   */
  async extractLinks(page, currentUrl, baseUrl = null) {
    const effectiveBaseUrl = baseUrl || this.config.getBaseUrl();
    this.logger.info('LINKS', `Extracting internal Notion page links (base: ${effectiveBaseUrl})...`);
    
    const links = await page.evaluate((baseUrl, currentUrl) => {
      const results = [];
      const seenUrls = new Set();
      
      // Parse base URL to get hostname for validation
      const baseUrlObj = new URL(baseUrl);
      const baseHostname = baseUrlObj.hostname;
      
      // Find all links
      const allLinks = document.querySelectorAll('a[href]');
      
      allLinks.forEach(link => {
        const href = link.getAttribute('href');
        
        // Skip empty hrefs
        if (!href) {
          return;
        }
        
        // Convert relative URLs to absolute
        let absoluteUrl;
        let isInternal = false;
        
        if (href.startsWith('/')) {
          // Internal relative link - append to base URL
          absoluteUrl = baseUrl + href;
          isInternal = true;
        } else if (href.startsWith('http')) {
          // Full URL - check if it's same domain
          try {
            const hrefObj = new URL(href);
            isInternal = hrefObj.hostname === baseHostname;
            absoluteUrl = href;
          } catch (e) {
            return; // Invalid URL
          }
        } else {
          // Other relative formats (e.g., "./page" or "page")
          // Treat as internal and append to base URL
          const normalizedHref = href.startsWith('./') ? href.substring(2) : href;
          absoluteUrl = baseUrl + (normalizedHref.startsWith('/') ? '' : '/') + normalizedHref;
          isInternal = true;
        }
        
        // Skip external links
        if (!isInternal) {
          return;
        }
        
        // Step 1: Remove everything after '?' (query params and fragments)
        const cleanUrl = absoluteUrl.split('?')[0];
        
        // Step 2: Extract raw page ID - everything after first '29' appearance
        // This handles both /Page-Name-29abc... and /29abc... formats
        const rawIdMatch = cleanUrl.match(/29[a-f0-9]{30}/i);
        if (!rawIdMatch) {
          // No valid page ID found, skip this link
          return;
        }
        
        const rawPageId = rawIdMatch[0];
        
        // Check if this is the current page (same raw ID)
        const currentRawIdMatch = currentUrl.match(/29[a-f0-9]{30}/i);
        const currentRawId = currentRawIdMatch ? currentRawIdMatch[0] : null;
        
        if (currentRawId && rawPageId === currentRawId) {
          // Same page, different section - skip it
          return;
        }
        
        // Skip if already seen (by raw ID)
        if (seenUrls.has(rawPageId)) {
          return;
        }
        
        seenUrls.add(rawPageId);
        
        // Return raw page ID as URL (will be resolved by server later)
        results.push({
          url: baseUrl + '/' + rawPageId,
          title: rawPageId,  // Use raw ID as placeholder, will be resolved later
          section: null,
          subsection: null,
          isInternal: true
        });
      });
      
      return results;
    }, effectiveBaseUrl, currentUrl);
    
    // Filter out the current page (by raw ID)
    const filteredLinks = links.filter(link => link.url !== currentUrl);
    
    this.logger.success('LINKS', `Found ${filteredLinks.length} internal page links.`);
    
    // Log some details about the links
    if (filteredLinks.length > 0) {
      const sections = [...new Set(filteredLinks.map(l => l.section).filter(s => s))];
      if (sections.length > 0) {
        this.logger.info('LINKS', `Detected sections: ${sections.join(', ')}`);
      }
    }
    
    return filteredLinks;
  }
}

module.exports = LinkExtractor;
