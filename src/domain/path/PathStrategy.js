/**
 * @fileoverview Abstract Path Strategy Interface
 * @module domain/path/PathStrategy
 * @description Defines the contract for path resolution strategies.
 * 
 * @design PATH STRATEGY PATTERN
 * This interface establishes a unified contract for resolving paths between pages.
 * Implementations handle specific scenarios:
 * - IntraPathStrategy: Same-page anchor links (#block-id)
 * - InterPathStrategy: Cross-page navigation (../sibling/index.html)
 * - ExternalPathStrategy: External URLs (unchanged)
 * 
 * The pattern addresses the conflation of concerns in the original PathCalculator:
 * 1. Single Responsibility: Each strategy handles one type of path resolution
 * 2. Open/Closed: New strategies can be added without modifying existing code
 * 3. Dependency Inversion: High-level modules depend on this abstraction
 * 
 * @see PathStrategyFactory - For strategy selection and invocation
 * @see IntraPathStrategy - Same-page anchor resolution
 * @see InterPathStrategy - Cross-page navigation resolution
 */

/**
 * @class PathStrategy
 * @classdesc Abstract base class for path resolution strategies.
 * All path strategies must extend this class and implement its abstract methods.
 */
class PathStrategy {
  /**
   * Path type enumeration.
   * Identifies the category of path resolution handled by a strategy.
   * 
   * @static
   * @readonly
   * @enum {string}
   */
  static PathType = Object.freeze({
    /** Same-page anchor links (e.g., #block-id) */
    INTRA: 'intra',
    /** Cross-page navigation links (e.g., ../sibling/index.html) */
    INTER: 'inter',
    /** External URLs - passed through unchanged */
    EXTERNAL: 'external'
  });

  /**
   * Resolve path from source to target context.
   * 
   * @abstract
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @param {Object} [options={}] - Resolution options
   * @param {string} [options.blockId] - Target block ID for anchor links
   * @param {Map<string, Map<string, string>>} [options.blockMapCache] - Block ID mapping cache (pageId → blockMap)
   * @returns {string} Resolved path/href
   * @throws {Error} If not implemented by subclass
   * 
   * @example
   * // IntraPathStrategy returns anchor only
   * strategy.resolve(pageA, pageA, { blockId: '29d979ee...' }); // '#29d979ee-ca9f-...'
   * 
   * @example
   * // InterPathStrategy returns full relative path
   * strategy.resolve(child, sibling, {}); // '../Sibling/index.html'
   */
  resolve(sourceContext, targetContext, options = {}) {
    throw new Error('PathStrategy.resolve() must be implemented by subclass');
  }

  /**
   * Check if this strategy handles the given source-target pair.
   * 
   * @abstract
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @returns {boolean} True if this strategy can resolve the path
   * @throws {Error} If not implemented by subclass
   * 
   * @example
   * // IntraPathStrategy supports same-page links
   * intraStrategy.supports(pageA, pageA); // true
   * intraStrategy.supports(pageA, pageB); // false
   */
  supports(sourceContext, targetContext) {
    throw new Error('PathStrategy.supports() must be implemented by subclass');
  }

  /**
   * Get the path type this strategy handles.
   * 
   * @abstract
   * @returns {string} PathType enumeration value
   * @throws {Error} If not implemented by subclass
   */
  getType() {
    throw new Error('PathStrategy.getType() must be implemented by subclass');
  }

  /**
   * Get human-readable name for this strategy.
   * Useful for logging and debugging.
   * 
   * @returns {string} Strategy name
   */
  getName() {
    return this.constructor.name;
  }
}

module.exports = PathStrategy;
