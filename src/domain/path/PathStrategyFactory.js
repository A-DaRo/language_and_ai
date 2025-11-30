/**
 * @fileoverview Path Strategy Factory
 * @module domain/path/PathStrategyFactory
 * @description Factory for creating and invoking path resolution strategies.
 * 
 * @design FACTORY PATTERN FOR PATH RESOLUTION
 * This factory centralizes path strategy selection and invocation.
 * It encapsulates the logic for determining which strategy to use
 * based on the source and target contexts.
 * 
 * Strategy Selection Order (first match wins):
 * 1. IntraPathStrategy - Same-page anchor links
 * 2. InterPathStrategy - Cross-page navigation
 * 3. ExternalPathStrategy - External URLs (fallback)
 * 
 * Benefits:
 * - Single entry point for all path resolution
 * - Encapsulated strategy selection logic
 * - Easy to add new strategies without modifying callers
 * - Testable strategy registration and selection
 * 
 * @see PathStrategy - Base interface
 * @see IntraPathStrategy - Same-page anchor resolution
 * @see InterPathStrategy - Cross-page navigation resolution
 * @see ExternalPathStrategy - External URL pass-through
 */

const PathStrategy = require('./PathStrategy');
const IntraPathStrategy = require('./IntraPathStrategy');
const InterPathStrategy = require('./InterPathStrategy');
const ExternalPathStrategy = require('./ExternalPathStrategy');
const BlockIDMapper = require('../../processing/BlockIDMapper');

/**
 * @class PathStrategyFactory
 * @classdesc Factory for creating and selecting path resolution strategies.
 * 
 * Provides a unified interface for path resolution, automatically selecting
 * the appropriate strategy based on the source and target page contexts.
 * 
 * @example
 * const factory = new PathStrategyFactory(config, logger);
 * 
 * // Resolve path (strategy selected automatically)
 * const path = factory.resolvePath(sourcePage, targetPage, {
 *   blockId: '29d979eeca9f',
 *   blockMapCache: cache
 * });
 * 
 * // Check path type before resolution
 * const type = factory.getPathType(sourcePage, targetPage);
 * // Returns: 'intra', 'inter', or 'external'
 */
class PathStrategyFactory {
  /**
   * Create a PathStrategyFactory instance.
   * 
   * @param {Config} [config=null] - Optional configuration object
   * @param {Logger} [logger=null] - Optional logger instance
   * 
   * @description Initializes the factory with default strategies.
   * Strategies are registered in priority order (first match wins).
   */
  constructor(config = null, logger = null) {
    this.config = config;
    this.logger = logger;
    this.strategies = [];

    // Register default strategies in priority order
    this._registerDefaultStrategies();
  }

  /**
   * Register default strategy implementations.
   * 
   * @private
   * @description Registers strategies in priority order:
   * 1. IntraPathStrategy (same-page anchors)
   * 2. InterPathStrategy (cross-page navigation)
   * 3. ExternalPathStrategy (external URLs - fallback)
   */
  _registerDefaultStrategies() {
    // Shared BlockIDMapper for consistent ID formatting
    const blockIdMapper = new BlockIDMapper();

    // Order matters: first matching strategy is used
    this.registerStrategy(new IntraPathStrategy(blockIdMapper));
    this.registerStrategy(new InterPathStrategy(blockIdMapper));
    this.registerStrategy(new ExternalPathStrategy());
  }

  /**
   * Register a path strategy.
   * 
   * @param {PathStrategy} strategy - Strategy instance to register
   * @throws {Error} If strategy doesn't extend PathStrategy
   * 
   * @description Adds a strategy to the list of available strategies.
   * Strategies are checked in registration order, so register
   * more specific strategies before general ones.
   */
  registerStrategy(strategy) {
    if (!(strategy instanceof PathStrategy)) {
      throw new Error('Strategy must extend PathStrategy');
    }
    this.strategies.push(strategy);

    if (this.logger) {
      this.logger.debug('PathFactory', `Registered strategy: ${strategy.getName()}`);
    }
  }

  /**
   * Clear all registered strategies.
   * 
   * @description Useful for testing or reconfiguring the factory.
   */
  clearStrategies() {
    this.strategies = [];
  }

  /**
   * Find appropriate strategy for given contexts.
   * 
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @returns {PathStrategy|null} First matching strategy, or null if none found
   * 
   * @description Iterates through registered strategies in order
   * and returns the first one that supports the given context pair.
   */
  findStrategy(sourceContext, targetContext) {
    for (const strategy of this.strategies) {
      if (strategy.supports(sourceContext, targetContext)) {
        return strategy;
      }
    }
    return null;
  }

  /**
   * Resolve path using appropriate strategy.
   * 
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @param {Object} [options={}] - Resolution options
   * @param {string} [options.blockId] - Target block ID for anchor
   * @param {Map<string, Map<string, string>>} [options.blockMapCache] - Block ID mapping cache
   * @param {string} [options.originalUrl] - Original URL (for external strategy)
   * @returns {string|null} Resolved path/href, or null if no strategy applies
   * 
   * @description Main entry point for path resolution. Automatically
   * selects the appropriate strategy and invokes it.
   * 
   * @example
   * // Same-page anchor
   * factory.resolvePath(pageA, pageA, { blockId: '29d979ee...' });
   * // Returns: '#29d979ee-ca9f-...'
   * 
   * // Cross-page navigation
   * factory.resolvePath(child, sibling, {});
   * // Returns: '../Sibling/index.html'
   * 
   * // Cross-page with anchor
   * factory.resolvePath(pageA, pageB, { blockId: '29d979ee...' });
   * // Returns: '../PageB/index.html#29d979ee-ca9f-...'
   */
  resolvePath(sourceContext, targetContext, options = {}) {
    const strategy = this.findStrategy(sourceContext, targetContext);

    if (!strategy) {
      if (this.logger) {
        this.logger.warn(
          'PathFactory',
          `No strategy found for ${sourceContext?.id} -> ${targetContext?.id}`
        );
      }
      return null;
    }

    if (this.logger) {
      this.logger.debug(
        'PathFactory',
        `Using ${strategy.getName()} (${strategy.getType()}) for path resolution`
      );
    }

    return strategy.resolve(sourceContext, targetContext, options);
  }

  /**
   * Determine path type without full resolution.
   * 
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @returns {string} PathType value ('intra', 'inter', 'external', or 'unknown')
   * 
   * @description Useful for logging, debugging, and conditional logic
   * that depends on the type of link without needing the actual path.
   */
  getPathType(sourceContext, targetContext) {
    const strategy = this.findStrategy(sourceContext, targetContext);
    return strategy ? strategy.getType() : 'unknown';
  }

  /**
   * Get the strategy that would be used for given contexts.
   * 
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @returns {string|null} Strategy name or null
   */
  getStrategyName(sourceContext, targetContext) {
    const strategy = this.findStrategy(sourceContext, targetContext);
    return strategy ? strategy.getName() : null;
  }

  /**
   * Get list of registered strategy names.
   * 
   * @returns {string[]} Array of strategy names in registration order
   */
  getRegisteredStrategyNames() {
    return this.strategies.map(s => s.getName());
  }

  /**
   * Create a factory with custom strategies only.
   * 
   * @static
   * @param {PathStrategy[]} strategies - Array of strategies to register
   * @param {Config} [config=null] - Optional configuration
   * @param {Logger} [logger=null] - Optional logger
   * @returns {PathStrategyFactory} Factory with only the provided strategies
   * 
   * @description Useful for testing or creating specialized factories
   * with a subset of strategies.
   */
  static withStrategies(strategies, config = null, logger = null) {
    const factory = new PathStrategyFactory(config, logger);
    factory.clearStrategies();
    
    for (const strategy of strategies) {
      factory.registerStrategy(strategy);
    }
    
    return factory;
  }
}

module.exports = PathStrategyFactory;
