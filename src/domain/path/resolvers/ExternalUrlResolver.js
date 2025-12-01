/**
 * @fileoverview External URL Resolver
 * @module domain/path/resolvers/ExternalUrlResolver
 * @description External URL pass-through using the unified PathResolver interface.
 * 
 * This resolver handles links that point to external resources
 * (not part of the scraped Notion site). These links are passed
 * through unchanged to preserve their original behavior.
 * 
 * External links include:
 * - HTTP/HTTPS URLs pointing to other domains
 * - mailto: links
 * - tel: links
 * - Other protocol schemes (ftp:, etc.)
 * 
 * @see PathResolver - Base interface
 * @see ExternalPathStrategy - Underlying implementation
 */

'use strict';

const PathResolver = require('../PathResolver');

/**
 * @class ExternalUrlResolver
 * @extends PathResolver
 * @classdesc Handles external URLs by passing them through unchanged.
 * 
 * This is a "null object" resolver that detects external links
 * and returns the original URL without modification. It serves
 * as a fallback when neither IntraPageResolver nor InterPageResolver applies.
 * 
 * @example
 * const resolver = new ExternalUrlResolver();
 * 
 * // External URL
 * resolver.resolve({ originalUrl: 'https://example.com' });
 * // Returns: 'https://example.com'
 */
class ExternalUrlResolver extends PathResolver {
  /**
   * Create an ExternalUrlResolver instance.
   */
  constructor() {
    super();
  }

  /**
   * Get the path type handled by this resolver.
   * @override
   * @returns {string} PathResolver.Types.EXTERNAL
   */
  getType() {
    return PathResolver.Types.EXTERNAL;
  }

  /**
   * Check if this resolver handles the given context.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} [context.target] - Target page context (null for external)
   * @returns {boolean} True if target is not an internal page
   * 
   * @description Returns true when targetContext is null or not a valid internal page,
   * indicating an external link that should be preserved unchanged.
   */
  supports(context) {
    const { target } = context || {};

    // External if no target context (link not found in scraped pages)
    if (!target) {
      return true;
    }

    // External if target has no valid ID
    if (!target.id || typeof target.id !== 'string') {
      return true;
    }

    // Otherwise, it's an internal link - let other resolvers handle it
    return false;
  }

  /**
   * Resolve the path for an external link.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {string} [context.originalUrl] - The original URL to preserve
   * @param {string} [context.href] - Alternative: original href
   * @returns {string|null} Original URL unchanged, or null if not provided
   */
  resolve(context) {
    const { originalUrl, href } = context || {};

    // Return original URL unchanged (prefer originalUrl, fallback to href)
    return originalUrl || href || null;
  }
}

module.exports = ExternalUrlResolver;
