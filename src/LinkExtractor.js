/**
 * Extracts and categorizes links from Notion pages
 */
class LinkExtractor {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
  }
  
  /**
   * Check if a URL is an internal Notion link (relative to the scraping domain)
   */
  isInternalLink(url, baseUrl) {
    try {
      // Handle relative URLs
      if (url.startsWith('/')) {
        return true;
      }
      
      // Parse URLs
      const urlObj = new URL(url);
      const baseUrlObj = new URL(baseUrl);
      
      // Check if same domain
      return urlObj.hostname === baseUrlObj.hostname;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Extract all internal Notion page links with their context
   */
  async extractLinks(page, currentUrl) {
    this.logger.info('LINKS', 'Extracting internal Notion page links...');
    
    const links = await page.evaluate((baseUrl) => {
      const results = [];
      const seenUrls = new Set();
      
      // Find all links
      const allLinks = document.querySelectorAll('a[href]');
      
      allLinks.forEach(link => {
        const href = link.getAttribute('href');
        
        // Skip if not a relative Notion page link
        if (!href || href.startsWith('http') && !href.includes('notion.site')) {
          return;
        }
        
        // Convert relative URLs to absolute
        let absoluteUrl;
        if (href.startsWith('/')) {
          absoluteUrl = baseUrl + href;
        } else if (href.startsWith('http')) {
          absoluteUrl = href;
        } else {
          return;
        }
        
        // Remove query parameters and fragments
        absoluteUrl = absoluteUrl.split('?')[0].split('#')[0];
        
        // Skip if already seen
        if (seenUrls.has(absoluteUrl)) {
          return;
        }
        seenUrls.add(absoluteUrl);
        
        // Extract title from the link
        let title = link.textContent.trim();
        
        // Try to find a better title from page icon + text structure
        const parent = link.closest('[data-block-id]');
        if (parent) {
          const titleElement = parent.querySelector('[contenteditable="false"]');
          if (titleElement) {
            title = titleElement.textContent.trim() || title;
          }
        }
        
        // Try to determine the section and subsection
        let section = null;
        let subsection = null;
        
        // Look for parent headings
        let currentElement = link;
        while (currentElement && currentElement !== document.body) {
          currentElement = currentElement.parentElement;
          
          // Check for headings
          if (currentElement) {
            const heading = currentElement.querySelector('h1, h2, h3');
            if (heading && !section) {
              const headingText = heading.textContent.trim();
              
              // Check if it's a week subsection
              if (/week\s+\d+/i.test(headingText)) {
                subsection = headingText;
              } else {
                section = headingText;
              }
            }
          }
        }
        
        // Alternative: Check for section by looking at preceding headings
        if (!section) {
          const allHeadings = Array.from(document.querySelectorAll('h1, h2, h3'));
          const linkPosition = Array.from(document.querySelectorAll('*')).indexOf(link);
          
          for (let i = allHeadings.length - 1; i >= 0; i--) {
            const heading = allHeadings[i];
            const headingPosition = Array.from(document.querySelectorAll('*')).indexOf(heading);
            
            if (headingPosition < linkPosition) {
              const headingText = heading.textContent.trim();
              
              if (/week\s+\d+/i.test(headingText)) {
                if (!subsection) subsection = headingText;
              } else {
                if (!section) section = headingText;
              }
              
              if (section && subsection) break;
            }
          }
        }
        
        results.push({
          url: absoluteUrl,
          title: title || 'Untitled',
          section: section,
          subsection: subsection,
          isInternal: true // Mark as internal by default in this extraction
        });
      });
      
      return results;
    }, this.config.getBaseUrl());
    
    // Filter out the current page
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
