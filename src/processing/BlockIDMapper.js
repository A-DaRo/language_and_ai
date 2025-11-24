/**
 * @fileoverview Block ID mapping persistence
 * @module processing/BlockIDMapper
 * @description Persists and retrieves block ID mappings for offline link rewriting.
 */

const fs = require('fs/promises');
const path = require('path');

/**
 * @class BlockIDMapper
 * @classdesc Manages persistence and retrieval of block ID mappings.
 * Stores mappings in `.block-ids.json` alongside saved HTML.
 */
class BlockIDMapper {
  constructor() {
    this.mapFileName = '.block-ids.json';
  }

  /**
   * Save block ID mapping to disk
   * @param {string} pageId - Page identifier
   * @param {string} saveDir - Directory where page HTML is saved
   * @param {Map<string, string>} blockMap - raw ID → formatted ID mapping
   * @returns {Promise<void>}
   */
  async saveBlockMap(pageId, saveDir, blockMap) {
    try {
      const mapFile = path.join(saveDir, this.mapFileName);
      
      // Convert Map to plain object for JSON serialization
      const mapObj = blockMap instanceof Map
        ? Object.fromEntries(blockMap)
        : blockMap;
      
      await fs.writeFile(
        mapFile,
        JSON.stringify(mapObj, null, 2),
        { encoding: 'utf-8' }
      );
    } catch (error) {
      // Log but don't throw - missing block map shouldn't break the process
      console.warn(`Failed to save block map for ${pageId}:`, error.message);
    }
  }

  /**
   * Load block ID mapping from disk
   * @param {string} saveDir - Directory where page HTML is saved
   * @returns {Promise<Map<string, string>>} raw ID → formatted ID, or empty Map if not found
   */
  async loadBlockMap(saveDir) {
    try {
      const mapFile = path.join(saveDir, this.mapFileName);
      const content = await fs.readFile(mapFile, 'utf-8');
      const mapObj = JSON.parse(content);
      return new Map(Object.entries(mapObj));
    } catch (error) {
      // File not found or invalid JSON - return empty map
      return new Map();
    }
  }

  /**
   * Get formatted ID for a raw block ID
   * Uses block map if available, falls back to formatting
   * 
   * @param {string} rawId - Raw block ID from URL
   * @param {Map<string, string>} blockMap - Loaded block ID mapping
   * @returns {string} Formatted ID for use as HTML anchor
   */
  getFormattedId(rawId, blockMap) {
    if (!rawId) {
      return '';
    }

    // Try to find in map first
    if (blockMap && blockMap.has(rawId)) {
      return blockMap.get(rawId);
    }

    // Fall back to formatting
    return this._fallbackFormat(rawId);
  }

  /**
   * Fallback formatting if block not found in map
   * Applies standard UUID formatting to raw hex
   * 
   * @private
   * @param {string} rawId - Raw block ID
   * @returns {string} Formatted UUID
   */
  _fallbackFormat(rawId) {
    if (!rawId || typeof rawId !== 'string' || rawId.length < 32) {
      return rawId;
    }

    const hex = rawId.toLowerCase();
    return [
      hex.substring(0, 8),
      hex.substring(8, 12),
      hex.substring(12, 16),
      hex.substring(16, 20),
      hex.substring(20, 32)
    ].join('-');
  }

  /**
   * Merge multiple block maps
   * @param {Array<Map<string, string>>} maps - Block maps to merge
   * @returns {Map<string, string>} Merged map
   */
  mergeBlockMaps(maps) {
    const merged = new Map();
    for (const map of maps) {
      if (map instanceof Map) {
        for (const [key, value] of map) {
          merged.set(key, value);
        }
      }
    }
    return merged;
  }

  /**
   * Cache block maps for all pages
   * Pre-loads all block ID mappings before link rewriting
   * 
   * @param {Map<string, PageContext>} contextMap - Map of page ID to PageContext
   * @returns {Promise<Map<string, Map<string, string>>>} page ID → block map
   */
  async loadAllBlockMaps(contextMap) {
    const cache = new Map();

    for (const [pageId, context] of contextMap) {
      try {
        const saveDir = path.dirname(context.htmlFilePath || '');
        if (saveDir && saveDir !== '.' && saveDir !== '') {
          const blockMap = await this.loadBlockMap(saveDir);
          cache.set(pageId, blockMap);
        }
      } catch (error) {
        // Skip pages with errors, use empty map
        cache.set(pageId, new Map());
      }
    }

    return cache;
  }

  /**
   * Get filename for block map file
   * @returns {string} Filename
   */
  getMapFileName() {
    return this.mapFileName;
  }
}

module.exports = BlockIDMapper;
