/**
 * @fileoverview External Path Strategy
 * @module domain/path/ExternalPathStrategy
 * @description Handles external URL preservation (pass-through).
 * 
 * @design EXTERNAL LINKS
 * This strategy handles links that point to external resources
 * (not part of the scraped Notion site). These links are passed
 * through unchanged to preserve their original behavior.
 * 
 * External links include:
 * - HTTP/HTTPS URLs pointing to other domains
 * - mailto: links
 * - tel: links
 * - Other protocol schemes (ftp:, etc.)
 * 
 * @see PathStrategy - Base interface
 * @see IntraPathStrategy - Same-page anchor resolution
 * @see InterPathStrategy - Cross-page navigation resolution
 */

const PathStrategy = require('./PathStrategy');

/**
 * @class ExternalPathStrategy
 * @extends PathStrategy
 * @classdesc Handles external URLs by passing them through unchanged.
 * 
 * This is a "null object" strategy that detects external links
 * and returns the original URL without modification. It serves
 * as a fallback when neither IntraPathStrategy nor InterPathStrategy
 * applies.
 * 
 * @example
 * const strategy = new ExternalPathStrategy();
 * 
 * // External URL
 * strategy.resolve(null, null, { originalUrl: 'https://example.com' });
 * // Returns: 'https://example.com'
 */
class ExternalPathStrategy extends PathStrategy {
  /**
   * Create an ExternalPathStrategy instance.
   */
  constructor() {
    super();
  }

  /**
   * Check if this strategy handles the given context pair.
   * 
   * @override
   * @param {PageContext|null} sourceContext - Origin page context (may be null for external)
   * @param {PageContext|null} targetContext - Destination page context (null for external)
   * @returns {boolean} True if target is not an internal page
   * 
   * @description Returns true when targetContext is null or not a valid internal page,
   * indicating an external link that should be preserved unchanged.
   */
  supports(sourceContext, targetContext) {
    // External if no target context (link not found in scraped pages)
    if (!targetContext) {
      return true;
    }

    // External if target has no valid ID
    if (!targetContext.id || typeof targetContext.id !== 'string') {
      return true;
    }

    // Otherwise, it's an internal link - let other strategies handle it
    return false;
  }

  /**
   * Get the path type handled by this strategy.
   * 
   * @override
   * @returns {string} PathType.EXTERNAL
   */
  getType() {
    return PathStrategy.PathType.EXTERNAL;
  }

  /**
   * Resolve the path for an external link.
   * 
   * @override
   * @param {PageContext|null} sourceContext - Origin page context
   * @param {PageContext|null} targetContext - Destination page context (null for external)
   * @param {Object} [options={}] - Resolution options
   * @param {string} [options.originalUrl] - The original URL to preserve
   * @returns {string|null} Original URL unchanged, or null if not provided
   * 
   * @description For external links, simply returns the original URL
   * without any modification. This preserves external links in offline
   * mode (though they may not work without internet connectivity).
   */
  resolve(sourceContext, targetContext, options = {}) {
    const { originalUrl } = options;

    // Return original URL unchanged
    return originalUrl || null;
  }
}

module.exports = ExternalPathStrategy;
