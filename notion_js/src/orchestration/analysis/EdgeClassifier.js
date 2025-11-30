/**
 * @fileoverview Edge classification for page graph
 * @module orchestration/analysis/EdgeClassifier
 * @description Classifies edges in the discovered page graph based on depth relationships.
 */

/**
 * @class EdgeClassifier
 * @classdesc Classifies edges by analyzing the depth relationship between source and target pages.
 * 
 * BFS guarantees all pages visited, so edge type is deterministic based on depth alone.
 * 
 * Edge types:
 * - FORWARD: source.depth < target.depth (deeper in hierarchy)
 * - BACK: source.depth >= target.depth (returning or crossing)
 */
class EdgeClassifier {
  /**
   * @param {Map<string, PageContext>} contextMap - Map of page ID to PageContext
   */
  constructor(contextMap = null) {
    this.contextMap = contextMap || new Map();
  }

  /**
   * Classify an edge between two discovered pages
   * @param {PageContext} sourceContext - Source page (where link originates)
   * @param {PageContext} targetContext - Target page (where link points)
   * @returns {Object} Classification object
   * @returns {string} .type - 'FORWARD' or 'BACK'
   * @returns {number} .depthDelta - Absolute difference in depths
   * @returns {boolean} .isAncestor - True if target is ancestor of source
   */
  classifyEdge(sourceContext, targetContext) {
    if (!sourceContext || !targetContext) {
      return {
        type: 'UNKNOWN',
        depthDelta: 0,
        isAncestor: false
      };
    }

    const depthDelta = Math.abs(sourceContext.depth - targetContext.depth);

    if (targetContext.depth > sourceContext.depth) {
      // Target is deeper - going forward in hierarchy
      return {
        type: 'FORWARD',
        depthDelta,
        isAncestor: false
      };
    }

    // Target is at same level or shallower - check if ancestor
    const isAncestor = this._isAncestor(sourceContext, targetContext);
    return {
      type: 'BACK',
      depthDelta,
      isAncestor
    };
  }

  /**
   * Check if target is an ancestor of source in the page hierarchy
   * @private
   * @param {PageContext} sourceContext - Source page
   * @param {PageContext} targetContext - Target page
   * @returns {boolean} True if target is an ancestor
   */
  _isAncestor(sourceContext, targetContext) {
    let current = sourceContext;

    // Walk up the parent chain
    while (current.parentContext || current.parentId) {
      if (current.parentContext) {
        current = current.parentContext;
      } else if (current.parentId && this.contextMap) {
        current = this.contextMap.get(current.parentId);
        if (!current) break;
      } else {
        break;
      }

      if (current && current.id === targetContext.id) {
        return true;
      }
    }

    return false;
  }

  /**
   * Set the context map for ancestor lookups
   * @param {Map<string, PageContext>} contextMap - Map of page ID to PageContext
   */
  setContextMap(contextMap) {
    this.contextMap = contextMap;
  }
}

module.exports = EdgeClassifier;
