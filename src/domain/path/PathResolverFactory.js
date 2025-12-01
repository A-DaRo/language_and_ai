/**
 * @fileoverview Path Resolver Factory
 * @module domain/path/PathResolverFactory
 * @description Factory for creating and selecting path resolvers.
 * 
 * @design UNIFIED PATH RESOLUTION
 * This factory centralizes all path resolution logic, providing:
 * - Automatic resolver selection based on context
 * - Explicit resolver type invocation
 * 
 * Usage:
 *   const factory = new PathResolverFactory(config, logger);
 *   
 *   // Automatic resolver selection
 *   const path = factory.resolve({
 *       source: currentPage,
 *       target: targetPage,
 *       href: originalHref,
 *       blockId: '29d979ee...'
 *   });
 *   
 *   // Explicit resolver type
 *   const filesystemPath = factory.resolveAs('filesystem', {
 *       source: pageContext
 *   });
 * 
 * Strategy Selection Order (first match wins):
 * 1. IntraPageResolver - Same-page anchor links
 * 2. InterPageResolver - Cross-page navigation
 * 3. ExternalUrlResolver - External URLs (fallback)
 * 
 * Note: FilesystemResolver is NOT in the auto-select chain.
 * Use resolveAs('filesystem', ...) to invoke it explicitly.
 * 
 * @see PathResolver - Base interface
 * @see IntraPageResolver - Same-page anchor resolution
 * @see InterPageResolver - Cross-page navigation resolution
 * @see ExternalUrlResolver - External URL pass-through
 * @see FilesystemResolver - Output path for saving
 */

'use strict';

const PathResolver = require('./PathResolver');
const {
  IntraPageResolver,
  InterPageResolver,
  ExternalUrlResolver,
  FilesystemResolver
} = require('./resolvers');
const BlockIDMapper = require('../../processing/BlockIDMapper');

/**
 * @class PathResolverFactory
 * @classdesc Factory for creating and selecting path resolvers.
 * 
 * Provides a unified interface for path resolution, automatically selecting
 * the appropriate resolver based on the source and target page contexts.
 * 
 * @example
 * const factory = new PathResolverFactory(config, logger);
 * 
 * // Resolve path (resolver selected automatically)
 * const path = factory.resolve({
 *   source: sourcePage,
 *   target: targetPage,
 *   blockId: '29d979eeca9f',
 *   blockMapCache: cache
 * });
 * 
 * // Check path type before resolution
 * const type = factory.getPathType({ source, target });
 * // Returns: 'intra', 'inter', or 'external'
 * 
 * // Explicit filesystem path resolution
 * const outputPath = factory.resolveAs('filesystem', { source: pageContext });
 */
class PathResolverFactory {
  /**
   * Create a PathResolverFactory instance.
   * 
   * @param {Config} [config=null] - Optional configuration object
   * @param {Logger} [logger=null] - Optional logger instance
   * 
   * @description Initializes the factory with default resolvers.
   * Resolvers are registered in priority order (first match wins).
   */
  constructor(config = null, logger = null) {
    this.config = config;
    this.logger = logger;
    this.resolvers = [];
    
    // Filesystem resolver kept separate (not auto-selected)
    this.filesystemResolver = null;

    // Register default resolvers in priority order
    this._registerDefaultResolvers();
  }

  /**
   * Register default resolver implementations.
   * 
   * @private
   * @description Registers resolvers in priority order:
   * 1. IntraPageResolver (same-page anchors)
   * 2. InterPageResolver (cross-page navigation)
   * 3. ExternalUrlResolver (external URLs - fallback)
   * 
   * FilesystemResolver is kept separate and invoked via resolveAs().
   */
  _registerDefaultResolvers() {
    // Shared BlockIDMapper for consistent ID formatting
    const blockIdMapper = new BlockIDMapper();

    // Order matters: first matching resolver is used
    this.registerResolver(new IntraPageResolver(blockIdMapper));
    this.registerResolver(new InterPageResolver(blockIdMapper));
    this.registerResolver(new ExternalUrlResolver());

    // Filesystem resolver is separate
    this.filesystemResolver = new FilesystemResolver();
  }

  /**
   * Register a path resolver.
   * 
   * @param {PathResolver} resolver - Resolver instance to register
   * @throws {Error} If resolver doesn't extend PathResolver
   * 
   * @description Adds a resolver to the list of available resolvers.
   * Resolvers are checked in registration order, so register
   * more specific resolvers before general ones.
   */
  registerResolver(resolver) {
    if (!(resolver instanceof PathResolver)) {
      throw new Error('Resolver must extend PathResolver');
    }
    this.resolvers.push(resolver);

    if (this.logger) {
      this.logger.debug('PathResolverFactory', `Registered resolver: ${resolver.getName()}`);
    }
  }

  /**
   * Clear all registered resolvers.
   * 
   * @description Useful for testing or reconfiguring the factory.
   */
  clearResolvers() {
    this.resolvers = [];
    this.filesystemResolver = null;
  }

  /**
   * Find appropriate resolver for given context.
   * 
   * @param {Object} context - Resolution context
   * @returns {PathResolver|null} First matching resolver, or null if none found
   * 
   * @description Iterates through registered resolvers in order
   * and returns the first one that supports the given context.
   */
  findResolver(context) {
    for (const resolver of this.resolvers) {
      if (resolver.supports(context)) {
        return resolver;
      }
    }
    return null;
  }

  /**
   * Resolve path using appropriate resolver.
   * 
   * @param {Object} context - Resolution context
   * @param {PageContext} [context.source] - Source page context
   * @param {PageContext} [context.target] - Target page context
   * @param {string} [context.href] - Original href being resolved
   * @param {string} [context.blockId] - Target block ID for anchor
   * @param {string} [context.originalUrl] - Original URL (for external resolver)
   * @param {Map<string, Map<string, string>>} [context.blockMapCache] - Block ID mapping cache
   * @returns {string|null} Resolved path/href, or null if no resolver applies
   * 
   * @description Main entry point for path resolution. Automatically
   * selects the appropriate resolver and invokes it.
   * 
   * @example
   * // Same-page anchor
   * factory.resolve({ source: pageA, target: pageA, blockId: '29d979ee...' });
   * // Returns: '#29d979ee-ca9f-...'
   * 
   * // Cross-page navigation
   * factory.resolve({ source: child, target: sibling });
   * // Returns: '../Sibling/index.html'
   * 
   * // Cross-page with anchor
   * factory.resolve({ source: pageA, target: pageB, blockId: '29d979ee...' });
   * // Returns: '../PageB/index.html#29d979ee-ca9f-...'
   */
  resolve(context) {
    const resolver = this.findResolver(context);

    if (!resolver) {
      if (this.logger) {
        this.logger.warn(
          'PathResolverFactory',
          `No resolver found for ${context?.source?.id} -> ${context?.target?.id}`
        );
      }
      return null;
    }

    if (this.logger) {
      this.logger.debug(
        'PathResolverFactory',
        `Using ${resolver.getName()} (${resolver.getType()}) for path resolution`
      );
    }

    return resolver.resolve(context);
  }

  /**
   * Resolve using a specific resolver type.
   * 
   * @param {string} type - One of PathResolver.Types
   * @param {Object} context - Resolution context
   * @returns {string|null} Resolved path
   * 
   * @description Use this method when you need to explicitly invoke
   * a specific resolver type, bypassing automatic selection.
   * 
   * @example
   * // Filesystem path for saving
   * const outputPath = factory.resolveAs('filesystem', { source: pageContext });
   * // Returns: 'Section/Page/index.html'
   */
  resolveAs(type, context) {
    if (type === PathResolver.Types.FILESYSTEM) {
      if (!this.filesystemResolver) {
        this.filesystemResolver = new FilesystemResolver();
      }
      return this.filesystemResolver.resolve(context);
    }

    // Find resolver by type
    const resolver = this.resolvers.find(r => r.getType() === type);
    if (resolver) {
      return resolver.resolve(context);
    }

    if (this.logger) {
      this.logger.warn('PathResolverFactory', `No resolver found for type: ${type}`);
    }
    return null;
  }

  /**
   * Get filesystem resolver instance for direct access.
   * Useful when additional methods like resolveDirectory() are needed.
   * 
   * @returns {FilesystemResolver} Filesystem resolver instance
   */
  getFilesystemResolver() {
    if (!this.filesystemResolver) {
      this.filesystemResolver = new FilesystemResolver();
    }
    return this.filesystemResolver;
  }

  /**
   * Determine path type without full resolution.
   * 
   * @param {Object} context - Resolution context
   * @returns {string} PathType value ('intra', 'inter', 'external', or 'unknown')
   * 
   * @description Useful for logging, debugging, and conditional logic
   * that depends on the type of link without needing the actual path.
   */
  getPathType(context) {
    const resolver = this.findResolver(context);
    return resolver ? resolver.getType() : 'unknown';
  }

  /**
   * Get the resolver that would be used for given context.
   * 
   * @param {Object} context - Resolution context
   * @returns {string|null} Resolver name or null
   */
  getResolverName(context) {
    const resolver = this.findResolver(context);
    return resolver ? resolver.getName() : null;
  }

  /**
   * Get list of registered resolver names.
   * 
   * @returns {string[]} Array of resolver names in registration order
   */
  getRegisteredResolverNames() {
    return this.resolvers.map(r => r.getName());
  }

  /**
   * Create a factory with custom resolvers only.
   * 
   * @static
   * @param {PathResolver[]} resolvers - Array of resolvers to register
   * @param {Config} [config=null] - Optional configuration
   * @param {Logger} [logger=null] - Optional logger
   * @returns {PathResolverFactory} Factory with only the provided resolvers
   * 
   * @description Useful for testing or creating specialized factories
   * with a subset of resolvers.
   */
  static withResolvers(resolvers, config = null, logger = null) {
    const factory = new PathResolverFactory(config, logger);
    factory.clearResolvers();

    for (const resolver of resolvers) {
      factory.registerResolver(resolver);
    }

    return factory;
  }
}

module.exports = PathResolverFactory;
