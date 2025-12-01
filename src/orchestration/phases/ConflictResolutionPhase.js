/**
 * @fileoverview Phase 4: Conflict Resolution - Duplicate detection and path generation
 * @module orchestration/phases/ConflictResolutionPhase
 * @description Resolves duplicate pages and generates the link rewrite map.
 */

const ConflictResolver = require('../analysis/ConflictResolver');
const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class ConflictResolutionPhase
 * @extends PhaseStrategy
 * @classdesc Analyzes the graph to resolve duplicates and generate file paths.
 */
class ConflictResolutionPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Prunes the graph and generates the link rewrite map
   * @description
   * 1. Identifies duplicate pages (same URL, different paths)
   * 2. Selects canonical instances based on depth and discovery order
   * 3. Generates linkRewriteMap (URL -> Local Path)
   * 4. Prepares ExecutionQueue with canonical pages
   * @returns {Promise<Object>} { canonicalContexts, linkRewriteMap }
   */
  async execute() {
    this.logger.separator('Phase 4: Conflict Resolution');

    const allContexts = this.queueManager.getAllContexts();
    const titleRegistry = this.queueManager.getTitleRegistry();
    const { canonicalContexts, linkRewriteMap, stats } = ConflictResolver.resolve(
      allContexts,
      titleRegistry
    );

    this.orchestrator.linkRewriteMap = linkRewriteMap;

    this.logger.success(
      'CONFLICT_RESOLUTION',
      `Resolved ${stats.uniquePages} unique page(s) (${stats.duplicates} duplicate(s) removed)`
    );

    return { canonicalContexts, linkRewriteMap };
  }
}

module.exports = ConflictResolutionPhase;
