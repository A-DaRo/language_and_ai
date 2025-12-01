/**
 * @fileoverview Filesystem Resolver
 * @module domain/path/resolvers/FilesystemResolver
 * @description Output path generation for file saving using the unified PathResolver interface.
 * 
 * This resolver generates filesystem paths where scraped pages should be saved.
 * It is used by ConflictResolver to calculate targetFilePath for each page.
 * 
 * @design FILESYSTEM PATH GENERATION
 * Unlike other resolvers which compute navigation paths between pages,
 * FilesystemResolver computes the OUTPUT path where a page should be saved.
 * 
 * Algorithm:
 * 1. Get path segments from the page's hierarchy (parent chain)
 * 2. Join segments to form directory path
 * 3. Append 'index.html' for the file
 * 
 * Examples:
 * - Root page (depth 0): 'index.html'
 * - Child page (depth 1): 'ChildTitle/index.html'
 * - Grandchild (depth 2): 'ChildTitle/GrandchildTitle/index.html'
 * 
 * @see PathResolver - Base interface
 * @see ConflictResolver - Primary consumer of this resolver
 */

'use strict';

const path = require('path');
const PathResolver = require('../PathResolver');
const FileSystemUtils = require('../../../utils/FileSystemUtils');

/**
 * @class FilesystemResolver
 * @extends PathResolver
 * @classdesc Generates filesystem output paths for page saving.
 * 
 * This resolver differs from IntraPageResolver and InterPageResolver in that:
 * - It only requires source context (the page being saved)
 * - It produces root-relative paths (not navigation paths)
 * - It is typically used during the analysis phase, not link rewriting
 * 
 * @example
 * const resolver = new FilesystemResolver();
 * 
 * // Root page
 * resolver.resolve({ source: rootPage });
 * // Returns: 'index.html'
 * 
 * // Child page
 * resolver.resolve({ source: childPage });
 * // Returns: 'Section/index.html'
 * 
 * // Grandchild page
 * resolver.resolve({ source: grandchildPage });
 * // Returns: 'Section/Page/index.html'
 */
class FilesystemResolver extends PathResolver {
  /**
   * Create a FilesystemResolver instance.
   */
  constructor() {
    super();
  }

  /**
   * Get the path type handled by this resolver.
   * @override
   * @returns {string} PathResolver.Types.FILESYSTEM
   */
  getType() {
    return PathResolver.Types.FILESYSTEM;
  }

  /**
   * Check if this resolver handles the given context.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} [context.source] - Source page context
   * @returns {boolean} True if source context is valid
   * 
   * @description FilesystemResolver supports any context with a valid source.
   * However, this resolver should only be invoked explicitly via factory.resolveAs('filesystem', ...)
   * since it doesn't participate in the automatic strategy selection chain.
   */
  supports(context) {
    const { source } = context || {};

    // Need a valid source context
    if (!source) {
      return false;
    }

    // Source must have an ID
    return source.id && typeof source.id === 'string';
  }

  /**
   * Resolve the filesystem output path for a page.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} context.source - Source page context
   * @returns {string} Root-relative file path (e.g., 'Section/Page/index.html')
   * 
   * @description Generates the path where the page's HTML file should be saved,
   * relative to the output directory root.
   */
  resolve(context) {
    const { source } = context || {};

    if (!source) {
      return 'index.html';
    }

    // Get path segments for this page
    const segments = this._getPathSegments(source);

    if (segments.length === 0) {
      // Root page - saved directly in output directory
      return 'index.html';
    }

    // Join segments and append index.html
    return segments.join('/') + '/index.html';
  }

  /**
   * Get the directory portion of the output path (without index.html).
   * Useful for creating directories.
   * 
   * @param {Object} context - Resolution context
   * @param {PageContext} context.source - Source page context
   * @returns {string} Directory path (e.g., 'Section/Page' or '' for root)
   */
  resolveDirectory(context) {
    const { source } = context || {};

    if (!source) {
      return '';
    }

    const segments = this._getPathSegments(source);
    return segments.join('/');
  }

  /**
   * Get path segments for a context.
   * @private
   * @param {PageContext} context - Page context
   * @returns {string[]} Array of sanitized path segments (excluding root)
   */
  _getPathSegments(context) {
    // Priority 1: Pre-computed pathSegments (survives IPC serialization)
    if (context.pathSegments && Array.isArray(context.pathSegments) && context.pathSegments.length > 0) {
      return [...context.pathSegments];
    }

    // Priority 2: Use getPathSegments() method if available
    if (typeof context.getPathSegments === 'function') {
      const segments = context.getPathSegments();
      if (segments && segments.length > 0) {
        return [...segments];
      }
    }

    // Priority 3: Use getRelativePath() if available
    if (typeof context.getRelativePath === 'function') {
      const relativePath = context.getRelativePath();
      if (relativePath) {
        // Split on / and filter empty segments
        return relativePath.split('/').filter(s => s.length > 0);
      }
    }

    // Priority 4: Fallback to parent chain traversal
    const segments = [];
    let current = context;

    while (current) {
      if (current.depth > 0 && current.title && current.title !== 'untitled') {
        const safeName = FileSystemUtils.sanitizeFilename(current.title);
        segments.unshift(safeName);
      }
      current = current.parentContext;
    }

    return segments;
  }
}

module.exports = FilesystemResolver;
