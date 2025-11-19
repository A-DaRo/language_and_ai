const path = require('path');
const FileSystemUtils = require('../utils/FileSystemUtils');

/**
 * Represents the context of a page in the hierarchy
 */
class PageContext {
  constructor(url, title, depth = 0, parentContext = null) {
    this.url = url;
    this.originalTitle = title || 'Untitled';
    this.displayTitle = this.originalTitle;
    this.title = FileSystemUtils.sanitizeFilename(title);
    this.depth = depth;
    this.parentContext = parentContext;
    this.section = null; // e.g., "Syllabus", "Material"
    this.subsection = null; // e.g., "Week 1", "Week 2"
    this.children = [];
    this.isNestedUnderParent = false; // Flag to indicate if this is a nested sub-page
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
   */
  addChild(childContext) {
    if (!childContext) return;
    const alreadyLinked = this.children.includes(childContext);
    if (!alreadyLinked) {
      this.children.push(childContext);
    }
  }
  
  /**
   * Get a string representation for debugging
   */
  toString() {
    return `PageContext(title="${this.title}", depth=${this.depth}, section="${this.section}", subsection="${this.subsection}")`;
  }
}

module.exports = PageContext;
