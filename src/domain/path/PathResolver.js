/**
 * @fileoverview PathResolver Interface
 * @module domain/path/PathResolver
 * @description Abstract interface for all path resolution operations.
 * 
 * PathResolver unifies path computation across the codebase:
 * - Intra-page: Same-page anchor links (#block-id)
 * - Inter-page: Cross-page navigation (../sibling/index.html)
 * - External: Pass-through for external URLs
 * - Filesystem: Output paths for file saving
 * 
 * @design UNIFIED PATH RESOLUTION
 * This interface consolidates fragmented path logic from:
 * - PageContext.getRelativePath() - Hierarchy path (filesystem)
 * - PageContext.getRelativePathTo() - Navigation path (inter-page)
 * - PathCalculator.calculateRelativePath() - Low-level math
 * - IntraPathStrategy.resolve() - Same-page anchors
 * - InterPathStrategy.resolve() - Cross-page navigation
 * - ExternalPathStrategy.resolve() - External URLs
 * - ConflictResolver._calculateFilePath() - Output path
 * - LinkRewriterStep._computeRelativePath() - Source-relative
 * 
 * Benefits:
 * 1. Single source of truth for path logic
 * 2. Easy to use correct method for context
 * 3. Maintains consistency across codebase
 * 4. Factory pattern for automatic strategy selection
 * 
 * @see PathResolverFactory - Factory for creating resolvers
 * @see resolvers/IntraPageResolver - Same-page anchor resolution
 * @see resolvers/InterPageResolver - Cross-page navigation resolution
 * @see resolvers/ExternalUrlResolver - External URL pass-through
 * @see resolvers/FilesystemResolver - Output path for saving
 */

'use strict';

/**
 * Path resolution types enumeration.
 * Used to identify the category of path resolution.
 * 
 * @readonly
 * @enum {string}
 */
const ResolverTypes = Object.freeze({
  /** Same page, anchor only (e.g., #block-id) */
  INTRA: 'intra',
  
  /** Different page, relative path (e.g., ../sibling/index.html) */
  INTER: 'inter',
  
  /** External URL, pass-through */
  EXTERNAL: 'external',
  
  /** Filesystem path for saving files */
  FILESYSTEM: 'filesystem'
});

/**
 * @typedef {Object} ResolutionContext
 * @property {PageContext} [source] - Source page context
 * @property {PageContext} [target] - Target page context (if applicable)
 * @property {string} [href] - Original href being resolved
 * @property {string} [blockId] - Block ID for anchor links
 * @property {string} [originalUrl] - Original URL (for external strategy)
 * @property {Map<string, Map<string, string>>} [blockMapCache] - Block ID mapping cache
 */

/**
 * @class PathResolver
 * @classdesc Abstract base class for path resolution strategies.
 * All path resolvers must extend this class and implement its abstract methods.
 * 
 * @example
 * class IntraPageResolver extends PathResolver {
 *   getType() {
 *     return PathResolver.Types.INTRA;
 *   }
 *   
 *   supports(context) {
 *     return context.source?.id === context.target?.id;
 *   }
 *   
 *   resolve(context) {
 *     return context.blockId ? `#${context.blockId}` : '';
 *   }
 * }
 */
class PathResolver {
  /**
   * Path resolution types.
   * @static
   * @readonly
   * @type {Object}
   */
  static Types = ResolverTypes;

  /**
   * Check if this resolver handles the given scenario.
   * 
   * @abstract
   * @param {ResolutionContext} context - Resolution context
   * @returns {boolean} True if this resolver should handle the resolution
   * @throws {Error} If not implemented by subclass
   * 
   * @example
   * // IntraPageResolver
   * supports({ source, target }) {
   *   return source?.id === target?.id;
   * }
   * 
   * @example
   * // ExternalUrlResolver
   * supports({ target }) {
   *   return !target || !target.id;
   * }
   */
  supports(context) {
    throw new Error('PathResolver.supports() must be implemented by subclass');
  }

  /**
   * Resolve the path/URL.
   * 
   * @abstract
   * @param {ResolutionContext} context - Resolution context
   * @returns {string} Resolved path or URL
   * @throws {Error} If not implemented by subclass
   * 
   * @example
   * // IntraPageResolver resolves to anchor only
   * resolve({ blockId: '29d979ee' }); // '#29d979ee-ca9f-...'
   * 
   * @example
   * // InterPageResolver resolves to relative path
   * resolve({ source: childPage, target: siblingPage }); // '../Sibling/index.html'
   * 
   * @example
   * // FilesystemResolver resolves to output path
   * resolve({ source: pageContext }); // 'Section/Page/index.html'
   */
  resolve(context) {
    throw new Error('PathResolver.resolve() must be implemented by subclass');
  }

  /**
   * Get the type of this resolver.
   * 
   * @abstract
   * @returns {string} One of PathResolver.Types
   * @throws {Error} If not implemented by subclass
   */
  getType() {
    throw new Error('PathResolver.getType() must be implemented by subclass');
  }

  /**
   * Get human-readable name for this resolver.
   * Default implementation derives name from class name.
   * 
   * @returns {string} Resolver name
   */
  getName() {
    return this.constructor.name;
  }
}

module.exports = PathResolver;
