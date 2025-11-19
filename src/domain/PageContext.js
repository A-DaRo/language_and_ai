const path = require('path');
const FileSystemUtils = require('../utils/FileSystemUtils');

/**
 * @fileoverview Domain model for scraped page metadata
 * @module domain/PageContext
 * @description Represents the context of a page in the hierarchy. This class is designed
 * to be serializable for IPC communication in the Micro-Kernel architecture.
 * 
 * **CRITICAL CHANGE**: In the Micro-Kernel architecture, PageContext must be JSON-serializable
 * for IPC transfer between Master and Worker processes. The `parentContext` reference has been
 * replaced with `parentId` (string) to break circular references.
 */

/**
 * Represents the context of a page in the hierarchy
 * @class PageContext
 * @classdesc Encapsulates page metadata including URL, title, hierarchical position,
 * and relationships. Supports both tree-based (parentContext) and flat (parentId) representations.
 */
class PageContext {
  /**
   * @param {string} url - Page URL
   * @param {string} title - Page title (will be sanitized)
   * @param {number} [depth=0] - Depth in the hierarchy
   * @param {PageContext|null} [parentContext=null] - Parent context reference (Master-side only)
   * @param {string|null} [parentId=null] - Parent page ID (IPC-safe alternative)
   */
  constructor(url, title, depth = 0, parentContext = null, parentId = null) {
    // Generate unique ID from URL (Notion ID)
    this.id = this._extractNotionId(url);
    this.url = url;
    this.originalTitle = title || 'Untitled';
    this.displayTitle = null; // Will be set after page is discovered and name is resolved
    this.title = FileSystemUtils.sanitizeFilename(title);
    this.depth = depth;
    
    // Support both tree-based and flat representations
    this.parentContext = parentContext; // Master-side: full object reference
    this.parentId = parentId || (parentContext ? parentContext.id : null); // IPC-safe: string ID
    
    this.section = null; // e.g., "Syllabus", "Material"
    this.subsection = null; // e.g., "Week 1", "Week 2"
    this.children = []; // Array of PageContext (Master-side) or empty (Worker-side)
    this.childIds = []; // Array of child IDs (IPC-safe)
    this.isNestedUnderParent = false; // Flag to indicate if this is a nested sub-page
    this.targetFilePath = null; // Calculated by Master for download phase
  }
  
  /**
   * Extract Notion page ID from URL
   * @private
   * @param {string} url - Notion page URL
   * @returns {string} Notion page ID
   */
  _extractNotionId(url) {
    // Notion URLs format: https://www.notion.so/Title-XXXXX or https://www.notion.so/XXXXX
    const match = url.match(/([a-f0-9]{32})/i);
    return match ? match[1] : url;
  }

  /**
   * Update the human-friendly title while keeping sanitized filesystem name
   */
  setDisplayTitle(title) {
    if (!title) return;
    this.originalTitle = title;
    this.displayTitle = title.trim() || this.displayTitle;
  }
  
  /**
   * Set the section this page belongs to (e.g., "Syllabus")
   */
  setSection(section) {
    this.section = FileSystemUtils.sanitizeFilename(section);
  }
  
  /**
   * Set the subsection this page belongs to (e.g., "Week_1")
   */
  setSubsection(subsection) {
    this.subsection = FileSystemUtils.sanitizeFilename(subsection);
  }
  
  /**
   * Get the relative path for this page with deep nesting support
   * This builds a path that reflects the true hierarchical structure
   * THIS IS THE SINGLE SOURCE OF TRUTH for where a page should be saved
   */
  getRelativePath() {
    const parts = [];
    
    // Build path by traversing up the parent chain
    let current = this;
    while (current) {
      // Add current page's folder name (including Main_Page)
      if (current.title && current.title !== 'untitled') {
        parts.unshift(current.title);
      }
      current = current.parentContext;
    }
    
    return parts.join('/');
  }
  
  /**
   * Get the full directory path for this page
   * Used for saving files to disk
   */
  getDirectoryPath(baseDir) {
    const relativePath = this.getRelativePath();
    if (relativePath) {
      return path.join(baseDir, relativePath);
    }
    return baseDir;
  }
  
  /**
   * Get the full file path for this page's index.html
   */
  getFilePath(baseDir) {
    const dirPath = this.getDirectoryPath(baseDir);
    return path.join(dirPath, 'index.html');
  }
  
  /**
   * Get the relative path from one page to another for link rewriting
   * Returns a relative path string like '../Week_2/index.html'
   */
  getRelativePathTo(targetContext) {
    const sourcePath = this.getRelativePath();
    const targetPath = targetContext.getRelativePath();
    
    // Split paths into segments
    const sourceSegments = sourcePath ? sourcePath.split('/') : [];
    const targetSegments = targetPath ? targetPath.split('/') : [];
    
    // Find common ancestor
    let commonDepth = 0;
    while (commonDepth < sourceSegments.length && 
           commonDepth < targetSegments.length && 
           sourceSegments[commonDepth] === targetSegments[commonDepth]) {
      commonDepth++;
    }
    
    // Build relative path
    const upLevels = sourceSegments.length - commonDepth;
    const downSegments = targetSegments.slice(commonDepth);
    
    const relativeParts = [];
    for (let i = 0; i < upLevels; i++) {
      relativeParts.push('..');
    }
    relativeParts.push(...downSegments);
    relativeParts.push('index.html');
    
    return relativeParts.join('/');
  }
  
  /**
   * Add a child page context
   * @param {PageContext} childContext - Child page context
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
   * Serialize to JSON for IPC transfer (Worker → Master or Master → Worker)
   * @returns {Object} JSON-serializable object
   * @example
   * const json = pageContext.toJSON();
   * // Send via IPC: process.send({ type: 'RESULT', data: json });
   */
  toJSON() {
    return {
      id: this.id,
      url: this.url,
      originalTitle: this.originalTitle,
      displayTitle: this.displayTitle,
      title: this.title,
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
   * Deserialize from JSON to create a PageContext instance with methods
   * @static
   * @param {Object} json - Serialized PageContext data
   * @param {Map<string, PageContext>} [contextMap] - Optional map to resolve parent references
   * @returns {PageContext} Reconstructed PageContext instance
   * @example
   * const json = { id: 'abc123', url: '...', title: 'Page', ... };
   * const pageContext = PageContext.fromJSON(json);
   */
  static fromJSON(json, contextMap = null) {
    // Create instance with IPC-safe parentId
    const context = new PageContext(
      json.url,
      json.originalTitle || json.title,
      json.depth || 0,
      null, // parentContext will be resolved if contextMap provided
      json.parentId
    );
    
    // Restore properties
    context.id = json.id;
    context.displayTitle = json.displayTitle || json.originalTitle;
    context.section = json.section;
    context.subsection = json.subsection;
    context.childIds = json.childIds || [];
    context.isNestedUnderParent = json.isNestedUnderParent || false;
    context.targetFilePath = json.targetFilePath;
    
    // Resolve parent reference if contextMap provided
    if (contextMap && json.parentId) {
      context.parentContext = contextMap.get(json.parentId) || null;
    }
    
    return context;
  }
  
  /**
   * Check if this is the root page
   * @returns {boolean} True if this is the root page
   */
  isRoot() {
    return this.depth === 0 || !this.parentId;
  }
  
  /**
   * Get depth label for logging (e.g., "[L0]", "[L1]")
   * @returns {string} Depth label
   */
  getDepthLabel() {
    return `[L${this.depth}]`;
  }
  
  /**
   * Get a string representation for debugging
   * @returns {string} String representation
   */
  toString() {
    return `PageContext(id="${this.id}", title="${this.title}", depth=${this.depth}, section="${this.section}", subsection="${this.subsection}")`;
  }
}

module.exports = PageContext;
