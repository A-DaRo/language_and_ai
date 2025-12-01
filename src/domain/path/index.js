/**
 * @fileoverview Path Module Index
 * @module domain/path
 * @description Exports all path-related components for the Notion scraper.
 * 
 * This module provides the unified PathResolver API for path resolution:
 * 
 * **PathResolver API**
 * Unified interface for all path resolution:
 * - PathResolver - Abstract base class
 * - PathResolverFactory - Factory with automatic resolver selection
 * - resolvers/* - Individual resolver implementations
 * 
 * Usage:
 *   const factory = new PathResolverFactory(config, logger);
 *   const path = factory.resolve({ source, target, ...options });
 *   
 *   // For filesystem paths
 *   const outputPath = factory.resolveAs('filesystem', { source: pageContext });
 * 
 * @see PathResolver - Unified interface
 * @see PathResolverFactory - Factory with resolvers
 */

// PathResolver API
const PathResolver = require('./PathResolver');
const PathResolverFactory = require('./PathResolverFactory');
const {
  IntraPageResolver,
  InterPageResolver,
  ExternalUrlResolver,
  FilesystemResolver
} = require('./resolvers');

module.exports = {
  // PathResolver API
  PathResolver,
  PathResolverFactory,
  IntraPageResolver,
  InterPageResolver,
  ExternalUrlResolver,
  FilesystemResolver
};
