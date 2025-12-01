/**
 * @fileoverview Centralized page title registry
 * @module orchestration/queues/TitleRegistry
 * @description Decouples title storage from PageContext to reduce IPC overhead.
 */

const Logger = require('../../core/Logger');

/**
 * @class TitleRegistry
 * @classdesc Centralized store for page titles indexed by page ID.
 */
class TitleRegistry {
  constructor() {
    this.logger = Logger.getInstance();
    this.idToTitleMap = new Map();
  }

  /**
   * Associates a human-readable title with a page ID, overriding stale values.
   * @param {string} id - The page ID
   * @param {string} title - The resolved title
   */
  register(id, title) {
    if (!id || !title) {
      return;
    }

    const existing = this.idToTitleMap.get(id);
    if (existing === title) {
      return;
    }

    this.idToTitleMap.set(id, title);
  }

  /**
   * Retrieves the title for a given ID
   * @param {string} id - The page ID
   * @returns {string} The title or a fallback
   */
  get(id) {
    return this.idToTitleMap.get(id) || null;
  }

  /**
   * Returns the plain object representation for IPC transmission
   * @returns {Object} ID-to-Title map
   */
  serialize() {
    return Object.fromEntries(this.idToTitleMap);
  }

  /**
   * Get all entries as Map
   * @returns {Map} Full registry map
   */
  getMap() {
    return new Map(this.idToTitleMap);
  }

  /**
   * Check if a title exists
   * @param {string} id - The page ID
   * @returns {boolean} True if title exists
   */
  has(id) {
    return this.idToTitleMap.has(id);
  }
}

module.exports = TitleRegistry;
