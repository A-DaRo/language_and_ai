/**
 * @fileoverview Block ID extraction from HTML
 * @module extraction/BlockIDExtractor
 * @description Extracts block IDs from downloaded page HTML.
 * Maps raw block IDs (URL format) to formatted IDs (HTML format).
 */

/**
 * @class BlockIDExtractor
 * @classdesc Extracts and maps block IDs from Notion page HTML.
 * 
 * Block ID format conversion:
 *   Raw:       29d979eeca9f81f7b82fe4b983834212 (32 hex chars)
 *   Formatted: 29d979ee-ca9f-81f7-b82f-e4b983834212 (with dashes at [8,13,18,23])
 */
class BlockIDExtractor {
  /**
   * Extract block ID mapping from saved HTML
   * @param {Document|JSDOM.window.document} document - Parsed HTML DOM
   * @returns {Map<string, string>} raw block ID → formatted block ID
   */
  extractBlockIDs(document) {
    const blockMap = new Map();

    try {
      const blocks = document.querySelectorAll('[data-block-id]');
      
      for (const block of blocks) {
        const formattedId = block.getAttribute('data-block-id');
        
        if (formattedId) {
          // Convert formatted UUID to raw hex for mapping
          const rawId = this._formatToRaw(formattedId);
          blockMap.set(rawId, formattedId);
        }
      }
    } catch (error) {
      // If querySelectorAll fails, return empty map
      return new Map();
    }

    return blockMap;
  }

  /**
   * Convert formatted UUID to raw hex
   * Removes dashes and lowercases
   * 
   * @private
   * @param {string} formattedId - Formatted UUID (e.g., 29d979ee-ca9f-81f7-b82f-e4b983834212)
   * @returns {string} Raw hex (e.g., 29d979eeca9f81f7b82fe4b983834212)
   */
  _formatToRaw(formattedId) {
    if (!formattedId) return '';
    return formattedId.replace(/-/g, '').toLowerCase();
  }

  /**
   * Convert raw hex to formatted UUID
   * Adds dashes at standard UUID positions [8,13,18,23]
   * 
   * @private
   * @param {string} rawId - Raw hex (e.g., 29d979eeca9f81f7b82fe4b983834212)
   * @returns {string} Formatted UUID (e.g., 29d979ee-ca9f-81f7-b82f-e4b983834212)
   */
  _rawToFormatted(rawId) {
    if (!rawId || rawId.length < 32) {
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
   * Check if a string is a valid raw block ID
   * @param {string} rawId - Potential raw block ID
   * @returns {boolean} True if valid format
   */
  isValidRawId(rawId) {
    if (!rawId || typeof rawId !== 'string') {
      return false;
    }
    // 32 hex characters
    return /^[a-f0-9]{32}$/i.test(rawId);
  }

  /**
   * Check if a string is a valid formatted block ID
   * @param {string} formattedId - Potential formatted UUID
   * @returns {boolean} True if valid format
   */
  isValidFormattedId(formattedId) {
    if (!formattedId || typeof formattedId !== 'string') {
      return false;
    }
    // Standard UUID format with dashes
    return /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(formattedId);
  }
}

module.exports = BlockIDExtractor;
