/**
 * @fileoverview Centralized Task Queue Manager
 * @module orchestration/GlobalQueueManager
 * @description Manages the discovery and download queues for the distributed scraping system.
 * Tracks visited URLs, manages dependencies, and coordinates the workflow phases.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. This is the "frontier" state manager.
 */

const path = require('path');
const SystemEventBus = require('../core/SystemEventBus');
const PageContext = require('../domain/PageContext');

/**
 * @class GlobalQueueManager
 * @classdesc Centralized queue management for discovery and download phases
 */
class GlobalQueueManager {
  constructor() {
    this.eventBus = SystemEventBus.getInstance();
    
    // Discovery phase queues
    this.discoveryQueue = []; // Array of { pageContext, isFirstPage }
    this.visitedUrls = new Set(); // URLs already discovered
    this.pendingDiscovery = 0; // Count of in-flight discovery tasks
    this.maxDepth = 0; // Track maximum depth discovered (O(1) lookup)
    
    // Download phase queues
    this.downloadQueue = []; // Array of pageContext objects
    this.pendingDownloads = new Map(); // pageId -> { context, childrenCount }
    this.completedDownloads = new Set(); // pageId of completed downloads
    
    // Context registry
    this.allContexts = new Map(); // pageId -> PageContext
    
    // Title registry (ID -> human-readable title)
    this.idToTitleMap = new Map(); // pageId -> resolved title
    
    // Statistics
    this.stats = {
      discovered: 0,
      downloaded: 0,
      failed: 0
    };
  }
  
  /**
   * Enqueue a page for discovery
   * @param {PageContext} pageContext - Page context to discover
   * @param {boolean} [isFirstPage=false] - Whether this is the root page
   * @returns {boolean} True if enqueued, false if already visited
   * @example
   * const wasEnqueued = queueManager.enqueueDiscovery(rootContext, true);
   */
  enqueueDiscovery(pageContext, isFirstPage = false) {
    // Extract raw page ID from URL for deduplication
    const rawIdMatch = pageContext.url.match(/29[a-f0-9]{30}/i);
    const rawPageId = rawIdMatch ? rawIdMatch[0] : null;
    
    if (!rawPageId) {
      // Invalid URL format, skip
      return false;
    }
    
    // Check if already visited (by raw page ID)
    if (this.visitedUrls.has(rawPageId)) {
      return false;
    }
    
    // Mark as visited (using raw page ID)
    this.visitedUrls.add(rawPageId);
    
    // Register context
    this.allContexts.set(pageContext.id, pageContext);
    
    // Track maximum depth (O(1) incremental update)
    if (pageContext.depth > this.maxDepth) {
      this.maxDepth = pageContext.depth;
    }
    
    // Add to queue
    this.discoveryQueue.push({ pageContext, isFirstPage });
    
    this.eventBus.emit('QUEUE:DISCOVERY_ENQUEUED', {
      url: pageContext.url,
      depth: pageContext.depth
    });
    
    return true;
  }
  
  /**
   * Get next discovery task from queue
   * @returns {Object|null} Next task or null if queue is empty
   * @returns {PageContext} return.pageContext - Page context to discover
   * @returns {boolean} return.isFirstPage - Whether this is the first page
   */
  nextDiscovery() {
    if (this.discoveryQueue.length === 0) {
      return null;
    }
    
    const task = this.discoveryQueue.shift();
    this.pendingDiscovery++;
    
    return task;
  }
  
  /**
   * Mark a discovery task as complete and process discovered links
   * @param {string} pageId - ID of the discovered page
   * @param {Array<Object>} discoveredLinks - Array of discovered links
   * @param {Object} [metadata] - Additional metadata
   * @param {string} [resolvedTitle] - Resolved page title from server
   * @returns {Array<PageContext>} New page contexts created from links
   */
  completeDiscovery(pageId, discoveredLinks, metadata = {}, resolvedTitle = null) {
    this.pendingDiscovery--;
    this.stats.discovered++;
    
    const parentContext = this.allContexts.get(pageId);
    if (!parentContext) {
      console.warn(`[GlobalQueueManager] Parent context not found: ${pageId}`);
      return [];
    }
    
    // Store resolved title in centralized registry (immutable after first set)
    if (resolvedTitle && !this.idToTitleMap.has(pageId)) {
      this.idToTitleMap.set(pageId, resolvedTitle);
      // Also update the parent context's title immediately for consistency
      parentContext.updateTitleFromRegistry(resolvedTitle);
    }
    
    const newContexts = [];
    
    // Process each discovered link
    for (const link of discoveredLinks) {
      // Skip if already visited
      if (this.visitedUrls.has(link.url)) {
        continue;
      }
      
      // Create child context
      const childContext = new PageContext(
        link.url,
        link.text || 'Untitled',
        parentContext.depth + 1,
        parentContext,
        parentContext.id
      );
      
      // Add metadata if present
      if (link.section) {
        childContext.setSection(link.section);
      }
      if (link.subsection) {
        childContext.setSubsection(link.subsection);
      }
      
      // Link parent-child relationship
      parentContext.addChild(childContext);
      
      newContexts.push(childContext);
    }
    
    return newContexts;
  }
  
  /**
   * Mark a discovery task as failed
   * @param {string} pageId - ID of the failed page
   * @param {Error} error - Error that occurred
   */
  failDiscovery(pageId, error) {
    this.pendingDiscovery--;
    this.stats.failed++;
    
    console.error(`[GlobalQueueManager] Discovery failed for ${pageId}:`, error.message);
  }
  
  /**
   * Build download queue from discovered contexts
   * @param {Array<PageContext>} contexts - All discovered contexts
   * @returns {number} Number of pages queued for download
   */
  buildDownloadQueue(contexts) {
    console.log(`[GlobalQueueManager] Building download queue from ${contexts.length} context(s)`);
    
    this.downloadQueue = [];
    this.pendingDownloads.clear();
    this.completedDownloads.clear();
    
    // Add all contexts to download queue
    for (const context of contexts) {
      this.downloadQueue.push(context);
      
      // Track children count for dependency management
      this.pendingDownloads.set(context.id, {
        context: context,
        childrenCount: context.children.length,
        completedChildren: 0
      });
    }
    
    this.eventBus.emit('QUEUE:DOWNLOAD_READY', {
      count: this.downloadQueue.length
    });
    
    return this.downloadQueue.length;
  }
  
  /**
   * Get next download task from queue with absolute path calculation
   * @param {string} outputDir - The system's root output directory (absolute)
   * @returns {Object|null} Download payload with absolute savePath or null
   * @returns {PageContext} return.context - Page context to download
   * @returns {string} return.savePath - Absolute filesystem path for index.html
   */
  nextDownload(outputDir) {
    if (this.downloadQueue.length === 0) {
      return null;
    }
    
    // Simple FIFO for now - could be optimized for dependency order
    const context = this.downloadQueue.shift();
    
    // Calculate absolute save path to prevent ghost writes
    const savePath = this._calculateAbsolutePath(outputDir, context);
    
    return {
      context,
      savePath
    };
  }
  
  /**
   * Calculate the absolute filesystem path for a page context
   * @private
   * @param {string} outputDir - Root output directory (relative or absolute)
   * @param {PageContext} context - The page context containing hierarchy info
   * @returns {string} The absolute path ending in 'index.html'
   * @example
   * _calculateAbsolutePath('course_material', context)
   * // Returns: "C:\Users\...\course_material\Main_Page\Week_1\index.html"
   */
  _calculateAbsolutePath(outputDir, context) {
    // Resolve outputDir to absolute path relative to process.cwd()
    const absoluteOutputDir = path.resolve(process.cwd(), outputDir);
    
    // Get relative path from context (e.g., "Main_Page/Week_1")
    const relativePath = context.getFilePath(outputDir);
    
    // Combine to create absolute path
    const absolutePath = path.resolve(absoluteOutputDir, relativePath);
    
    return absolutePath;
  }
  
  /**
   * Mark a download task as complete
   * @param {string} pageId - ID of the downloaded page
   * @param {string} savedPath - Path where the file was saved
   */
  markDownloadComplete(pageId, savedPath) {
    this.completedDownloads.add(pageId);
    this.stats.downloaded++;
    
    // Update parent's completed children count
    const downloadInfo = this.pendingDownloads.get(pageId);
    if (downloadInfo && downloadInfo.context.parentId) {
      const parentInfo = this.pendingDownloads.get(downloadInfo.context.parentId);
      if (parentInfo) {
        parentInfo.completedChildren++;
      }
    }
    
    this.eventBus.emit('QUEUE:DOWNLOAD_COMPLETE_ITEM', {
      pageId,
      savedPath,
      progress: `${this.stats.downloaded}/${this.pendingDownloads.size}`
    });
  }
  
  /**
   * Mark a download task as failed
   * @param {string} pageId - ID of the failed page
   * @param {Error} error - Error that occurred
   */
  failDownload(pageId, error) {
    this.stats.failed++;
    
    console.error(`[GlobalQueueManager] Download failed for ${pageId}:`, error.message);
    
    this.eventBus.emit('QUEUE:DOWNLOAD_FAILED_ITEM', {
      pageId,
      error: error.message
    });
  }
  
  /**
   * Check if discovery phase is complete
   * @returns {boolean} True if all discovery tasks are done
   */
  isDiscoveryComplete() {
    return this.discoveryQueue.length === 0 && this.pendingDiscovery === 0;
  }
  
  /**
   * Check if download phase is complete
   * @returns {boolean} True if all download tasks are done
   */
  isDownloadComplete() {
    return this.downloadQueue.length === 0 && 
           this.completedDownloads.size === this.pendingDownloads.size;
  }
  
  /**
   * Get all discovered contexts
   * @returns {Array<PageContext>} Array of all page contexts
   */
  getAllContexts() {
    return Array.from(this.allContexts.values());
  }
  
  /**
   * Get title registry as serializable object for IPC
   * @returns {Object} Plain object representation of ID-to-title map
   * @example
   * const registry = queueManager.getTitleRegistry();
   * // Returns: { id1: 'Title 1', id2: 'Title 2' }
   */
  getTitleRegistry() {
    return Object.fromEntries(this.idToTitleMap);
  }
  
  /**
   * Get title by page ID
   * @param {string} pageId - Page identifier
   * @returns {string|null} Title or null if not found
   */
  getTitleById(pageId) {
    return this.idToTitleMap.get(pageId) || null;
  }
  
  /**
   * Get maximum depth discovered
   * @returns {number} Maximum depth (0 if no pages discovered)
   */
  getMaxDepth() {
    return this.maxDepth;
  }
  
  /**
   * Set title for a page ID (used for initialization)
   * @param {string} pageId - Page identifier
   * @param {string} title - Human-readable title
   */
  setTitle(pageId, title) {
    if (!this.idToTitleMap.has(pageId)) {
      this.idToTitleMap.set(pageId, title);
    }
  }
  
  /**
   * Get statistics
   * @returns {Object} Current statistics
   */
  getStatistics() {
    return {
      discovered: this.stats.discovered,
      downloaded: this.stats.downloaded,
      failed: this.stats.failed,
      pending: {
        discovery: this.pendingDiscovery,
        download: this.downloadQueue.length
      },
      total: {
        contexts: this.allContexts.size,
        visited: this.visitedUrls.size
      }
    };
  }
  
  /**
   * Reset all queues and state (for testing or restart)
   */
  reset() {
    this.discoveryQueue = [];
    this.visitedUrls.clear();
    this.pendingDiscovery = 0;
    this.maxDepth = 0;
    this.downloadQueue = [];
    this.pendingDownloads.clear();
    this.completedDownloads.clear();
    this.allContexts.clear();
    this.idToTitleMap.clear();
    this.stats = { discovered: 0, downloaded: 0, failed: 0 };
  }
}

module.exports = GlobalQueueManager;
