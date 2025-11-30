/**
 * @fileoverview Domain model for scraped page metadata
 * @module domain/PageContext
 * @description Represents the context of a page in the hierarchy.
 * Delegates path calculation to PathCalculator for clean separation.
 * 
 * @design PATH RESOLUTION
 * PageContext provides convenience methods for path operations:
 * - getRelativePath(): Hierarchy-based path (for filesystem)
 * - getDirectoryPath(): Full directory path for saving files
 * - getFilePath(): Full path to index.html
 * - getRelativePathTo(): Navigation path to another page (delegated to PathCalculator)
 * 
 * For more advanced path resolution (same-page anchors, block IDs),
 * use PathStrategyFactory directly in calling code.
 * 
 * @see PathCalculator - Filesystem path calculation
 * @see PathStrategyFactory - Advanced path resolution with strategies
 */

const FileSystemUtils = require('../utils/FileSystemUtils');
const PathCalculator = require('./path/PathCalculator');

/**
 * @class PageContext
 * @classdesc Encapsulates page metadata including URL, title, and hierarchical position.
 */
class PageContext {
  /**
   * @param {string} url - Page URL
   * @param {string} rawTitle - Page title (filesystem-safe)
   * @param {number} [depth=0] - Depth in hierarchy
   * @param {PageContext|null} [parentContext=null] - Parent reference (Master-side)
   * @param {string|null} [parentId=null] - Parent ID (IPC-safe)
   */
  constructor(url, rawTitle, depth = 0, parentContext = null, parentId = null) {
    this.id = this._extractNotionId(url);
    this.url = url;
    this.rawTitle = rawTitle || 'Untitled';
    this.title = FileSystemUtils.sanitizeFilename(this.rawTitle);
    this.depth = depth;
    this.parentContext = parentContext;
    this.parentId = parentId || (parentContext ? parentContext.id : null);
    this.section = null;
    this.subsection = null;
    this.children = [];
    this.childIds = [];
    this.isNestedUnderParent = false;
    this.targetFilePath = null;

    // Initialize path calculator
    this.pathCalculator = new PathCalculator();
  }

  /**
   * Extract Notion page ID from URL
   * @private
   * @param {string} url - Notion page URL
   * @returns {string} Notion page ID
   */
  _extractNotionId(url) {
    const match = url.match(/([a-f0-9]{32})/i);
    return match ? match[1] : url;
  }

  /**
   * Update title from human-readable value
   * @param {string} humanReadableTitle - Title to sanitize and update
   * @returns {void}
   */
  updateTitleFromRegistry(humanReadableTitle) {
    if (!humanReadableTitle) return;
    this.rawTitle = humanReadableTitle;
    this.title = FileSystemUtils.sanitizeFilename(humanReadableTitle);
  }

  /**
   * Get the best display title, preferring registry data and truncating long strings.
   * @param {Object} [titleRegistry={}] - ID-to-title map for resolved titles
   * @returns {string} Display-friendly title
   */
  getDisplayTitle(titleRegistry = {}) {
    const registryTitle = titleRegistry[this.id];
    const fallback = registryTitle || this.rawTitle || this.title || 'Untitled';
    const normalized = fallback.trim() || 'Untitled';

    if (/^[a-f0-9]{32}$/i.test(normalized)) {
      return `Page ${normalized.substring(0, 6)}...`;
    }

    const maxLength = 50;
    if (normalized.length > maxLength) {
      return `${normalized.slice(0, maxLength - 3)}...`;
    }

    return normalized;
  }

  /**
   * Set the section this page belongs to
   * @param {string} section - Section name
   * @returns {void}
   */
  setSection(section) {
    this.section = FileSystemUtils.sanitizeFilename(section);
  }

  /**
   * Set the subsection this page belongs to
   * @param {string} subsection - Subsection name
   * @returns {void}
   */
  setSubsection(subsection) {
    this.subsection = FileSystemUtils.sanitizeFilename(subsection);
  }

  /**
   * Get relative path for this page based on hierarchy
   * @returns {string} Relative path using parent chain
   */
  getRelativePath() {
    return this.pathCalculator.calculateRelativePath(this);
  }

  /**
   * Get full directory path for saving files
   * @param {string} baseDir - Base output directory
   * @returns {string} Full directory path
   */
  getDirectoryPath(baseDir) {
    return this.pathCalculator.calculateDirectoryPath(baseDir, this);
  }

  /**
   * Get full file path for index.html
   * @param {string} baseDir - Base output directory
   * @returns {string} Full file path
   */
  getFilePath(baseDir) {
    return this.pathCalculator.calculateFilePath(baseDir, this);
  }

  /**
   * Get relative path from this page to another
   * @param {PageContext} targetContext - Target page context
   * @returns {string} Relative path like '../sibling/child/index.html'
   */
  getRelativePathTo(targetContext) {
    return this.pathCalculator.calculateRelativePathBetween(this, targetContext);
  }

  /**
   * Add a child page context
   * @param {PageContext} childContext - Child to add
   * @returns {void}
   */
  addChild(childContext) {
    if (!childContext) return;
    const alreadyLinked = this.children.includes(childContext);
    if (!alreadyLinked) {
      this.children.push(childContext);
      if (childContext.id && !this.childIds.includes(childContext.id)) {
        this.childIds.push(childContext.id);
      }
    }
  }

  /**
   * Serialize to JSON for IPC transfer
   * @returns {Object} JSON-serializable object
   */
  toJSON() {
    return {
      id: this.id,
      url: this.url,
      title: this.title,
      rawTitle: this.rawTitle,
      depth: this.depth,
      parentId: this.parentId,
      section: this.section,
      subsection: this.subsection,
      childIds: this.childIds,
      isNestedUnderParent: this.isNestedUnderParent,
      targetFilePath: this.targetFilePath
    };
  }

  /**
   * Deserialize from JSON
   * @static
   * @param {Object} json - Serialized data
   * @param {Map<string, PageContext>} [contextMap] - Optional parent map
   * @returns {PageContext} Reconstructed instance
   */
  static fromJSON(json, contextMap = null) {
    const context = new PageContext(
      json.url,
      json.rawTitle || json.title,
      json.depth || 0,
      null,
      json.parentId
    );

    context.id = json.id;
    context.section = json.section;
    context.subsection = json.subsection;
    context.childIds = json.childIds || [];
    context.isNestedUnderParent = json.isNestedUnderParent || false;
    context.targetFilePath = json.targetFilePath;
    context.rawTitle = json.rawTitle || context.rawTitle;
    context.title = json.title || FileSystemUtils.sanitizeFilename(context.rawTitle);

    if (contextMap && json.parentId) {
      context.parentContext = contextMap.get(json.parentId) || null;
    }

    return context;
  }

  /**
   * Check if this is the root page
   * @returns {boolean} True if root
   */
  isRoot() {
    return this.depth === 0 || !this.parentId;
  }

  /**
   * Get depth label for logging
   * @returns {string} Depth label like '[L0]'
   */
  getDepthLabel() {
    return `[L${this.depth}]`;
  }

  /**
   * Get string representation
   * @returns {string} Debug string
   */
  toString() {
    return `PageContext(id="${this.id}", sanitizedTitle="${this.title}", depth=${this.depth}, section="${this.section}", subsection="${this.subsection}")`;
  }
}

module.exports = PageContext;

