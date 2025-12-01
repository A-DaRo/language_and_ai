/**
 * @fileoverview Duplicate Page Detector and Path Resolver
 * @module orchestration/analysis/ConflictResolver
 * @description Analyzes the discovered PageContext tree to detect duplicate pages
 * and generate canonical file paths for the download phase.
 * 
 * **CRITICAL**: This must run BEFORE the download phase begins, as workers need
 * the linkRewriteMap to correctly rewrite links in a single pass.
 */

const Logger = require('../../core/Logger');

/**
 * @class ConflictResolver
 * @classdesc Detects duplicate pages and generates canonical download paths
 */
class ConflictResolver {
  /**
   * Resolve conflicts and generate download plan
   * @static
   * @param {Array<PageContext>} allContexts - All discovered page contexts
   * @param {Object} [titleRegistry={}] - ID-to-title map for human-readable logging
   * @returns {Object} Resolution result
   * @returns {Array<PageContext>} return.canonicalContexts - Unique pages to download
   * @returns {Map<string, string>} return.linkRewriteMap - NotionID -> RelativeFilePath mapping
   * @returns {Object} return.stats - Conflict resolution statistics
   * @example
   * const titleRegistry = queueManager.getTitleRegistry();
   * const { canonicalContexts, linkRewriteMap } = ConflictResolver.resolve(allContexts, titleRegistry);
   * console.log(`Found ${canonicalContexts.length} unique pages`);
   */
  static resolve(allContexts, titleRegistry = {}) {
    const logger = Logger.getInstance();
    logger.debug('ConflictResolver', `Analyzing ${allContexts.length} discovered page(s)`);
    
    // First pass: Update all context titles from the title registry
    // This ensures file paths use human-readable names instead of raw IDs
    logger.debug('ConflictResolver', 'Updating context titles from title registry...');
    let updatedCount = 0;
    for (const context of allContexts) {
      const humanReadableTitle = titleRegistry[context.id];
      if (humanReadableTitle) {
        context.updateTitleFromRegistry(humanReadableTitle);
        updatedCount++;
      }
    }
    logger.debug('ConflictResolver', `Updated ${updatedCount} context title(s) from registry`);
    
    const urlToContexts = new Map(); // url -> Array of PageContext
    const canonicalContexts = [];
    const linkRewriteMap = new Map(); // NotionID -> RelativeFilePath
    
    // Group contexts by URL to detect duplicates
    for (const context of allContexts) {
      if (!urlToContexts.has(context.url)) {
        urlToContexts.set(context.url, []);
      }
      urlToContexts.get(context.url).push(context);
    }
    
    let duplicateCount = 0;
    
    // Process each unique URL
    for (const [url, contexts] of urlToContexts.entries()) {
      // Pick canonical context (first occurrence or shortest path)
      const canonical = ConflictResolver._selectCanonical(contexts, titleRegistry);
      
      if (contexts.length > 1) {
        duplicateCount += contexts.length - 1;
        const displayTitle = titleRegistry[canonical.id] || canonical.title || 'Untitled';
        logger.debug('ConflictResolver', `Found ${contexts.length} references to: ${displayTitle}`);
      }
      
      // Calculate target file path for this page
      const targetFilePath = ConflictResolver._calculateFilePath(canonical);
      canonical.targetFilePath = targetFilePath;
      
      // Add to canonical list
      canonicalContexts.push(canonical);
      
      // Map ALL page IDs (including duplicates) to the same canonical path
      for (const context of contexts) {
        linkRewriteMap.set(context.id, targetFilePath);
      }
    }
    
    const stats = {
      totalDiscovered: allContexts.length,
      uniquePages: canonicalContexts.length,
      duplicates: duplicateCount
    };
    
    logger.info('ConflictResolver', 'Resolution complete');
    logger.info('ConflictResolver', `  - Unique pages: ${stats.uniquePages}`);
    logger.info('ConflictResolver', `  - Duplicates removed: ${stats.duplicates}`);
    
    return {
      canonicalContexts,
      linkRewriteMap,
      stats
    };
  }
  
  /**
   * Select the canonical context from multiple references to the same URL
   * @private
   * @static
   * @param {Array<PageContext>} contexts - Array of contexts pointing to same URL
   * @param {Object} [titleRegistry={}] - ID-to-title map for logging
   * @returns {PageContext} The canonical context to use
   */
  static _selectCanonical(contexts, titleRegistry = {}) {
    if (contexts.length === 1) {
      return contexts[0];
    }
    
    // Selection criteria:
    // 1. Prefer root page (depth 0)
    // 2. Prefer pages with section/subsection metadata
    // 3. Prefer shallower depth
    // 4. Prefer first occurrence
    
    let best = contexts[0];
    
    for (const context of contexts) {
      // Root page always wins
      if (context.depth === 0) {
        return context;
      }
      
      // Has metadata and current best doesn't
      if ((context.section || context.subsection) && !(best.section || best.subsection)) {
        best = context;
        continue;
      }
      
      // Shallower depth
      if (context.depth < best.depth) {
        best = context;
        continue;
      }
    }
    
    return best;
  }
  
  /**
   * Calculate the file path where a page should be saved
   * @private
   * @static
   * @param {PageContext} context - Page context
   * @returns {string} Relative file path (e.g., "Main_Page/Week_1/Lecture/index.html")
   */
  static _calculateFilePath(context) {
    // Use the existing getRelativePath() method from PageContext
    // This builds the path by traversing parent references
    const relativePath = context.getRelativePath();
    
    if (!relativePath) {
      return 'index.html';
    }
    
    // Append index.html to the directory path
    return `${relativePath}/index.html`;
  }
  
  /**
   * Generate a visual conflict report (for logging/debugging)
   * @static
   * @param {Array<PageContext>} allContexts - All discovered contexts
   * @param {Object} [titleRegistry={}] - ID-to-title map for human-readable output
   * @returns {string} Formatted report
   */
  static generateReport(allContexts, titleRegistry = {}) {
    const { canonicalContexts, linkRewriteMap, stats } = ConflictResolver.resolve(allContexts, titleRegistry);
    
    let report = '\n=== Conflict Resolution Report ===\n\n';
    report += `Total Discovered: ${stats.totalDiscovered}\n`;
    report += `Unique Pages: ${stats.uniquePages}\n`;
    report += `Duplicates Removed: ${stats.duplicates}\n\n`;
    
    if (stats.duplicates > 0) {
      report += 'Duplicate URLs:\n';
      
      const urlCounts = new Map();
      for (const context of allContexts) {
        urlCounts.set(context.url, (urlCounts.get(context.url) || 0) + 1);
      }
      
      for (const [url, count] of urlCounts.entries()) {
        if (count > 1) {
          report += `  - ${url.substring(0, 60)}... (${count} references)\n`;
        }
      }
    }
    
    return report;
  }
}

module.exports = ConflictResolver;
