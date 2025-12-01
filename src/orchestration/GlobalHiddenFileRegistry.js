/**
 * @fileoverview Global Hidden File Registry for Cross-Page Deduplication
 * @module orchestration/GlobalHiddenFileRegistry
 * @description Maintains a global registry of hidden file URLs discovered across all pages,
 * enabling deduplication and preventing redundant downloads.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. Workers query this registry via IPC
 * before processing hidden file elements.
 * 
 * **Deadlock Prevention Strategy:**
 * Hidden files in Notion are often embedded in interactive elements that require
 * clicking and waiting (3+ seconds each). When the same file is referenced on
 * multiple pages, this causes:
 * 1. Redundant waiting time (N pages × 3 seconds per duplicate)
 * 2. Worker pool starvation when all workers block on the same files
 * 
 * The GlobalHiddenFileRegistry solves this by:
 * 1. Tracking all discovered hidden file URLs globally
 * 2. Allowing workers to skip already-discovered files
 * 3. Providing the saved path for link rewriting without re-downloading
 */

const SystemEventBus = require('../core/SystemEventBus');
const Logger = require('../core/Logger');

/**
 * @typedef {Object} HiddenFileEntry
 * @property {'pending'|'downloaded'|'failed'} status - Current processing status
 * @property {string|null} savedPath - Local path where file was saved (null if pending/failed)
 * @property {string} discoveredOn - Page ID that first discovered this URL
 * @property {number} discoveredAt - Unix timestamp of discovery
 * @property {number|null} downloadedAt - Unix timestamp of download completion (null if not downloaded)
 * @property {string|null} error - Error message if failed (null otherwise)
 */

/**
 * @class GlobalHiddenFileRegistry
 * @classdesc Cross-page deduplication registry for hidden file downloads.
 * 
 * **Thread Safety:**
 * Since Node.js is single-threaded in the Master process, concurrent access
 * is not an issue. However, the pending set ensures that two workers don't
 * attempt to process the same URL simultaneously.
 */
class GlobalHiddenFileRegistry {
  constructor() {
    this.logger = Logger.getInstance();
    this.eventBus = SystemEventBus.getInstance();
    
    /**
     * URL → HiddenFileEntry mapping
     * @type {Map<string, HiddenFileEntry>}
     */
    this.registry = new Map();
    
    /**
     * URLs currently being processed by workers
     * @type {Set<string>}
     */
    this.pending = new Set();
    
    /**
     * Statistics for monitoring
     */
    this.stats = {
      totalDiscovered: 0,
      totalDownloaded: 0,
      totalFailed: 0,
      totalSkippedDuplicates: 0,
      bytesDownloaded: 0
    };
    
    this._setupEventListeners();
  }
  
  /**
   * Setup IPC event listeners for worker queries
   * @private
   */
  _setupEventListeners() {
    // Listen for worker queries about hidden files
    this.eventBus.on('HIDDEN_FILE:QUERY', (data) => {
      this._handleWorkerQuery(data);
    });
    
    // Listen for worker registrations of new hidden files
    this.eventBus.on('HIDDEN_FILE:REGISTER', (data) => {
      this._handleWorkerRegistration(data);
    });
    
    // Listen for download completion reports
    this.eventBus.on('HIDDEN_FILE:DOWNLOAD_COMPLETE', (data) => {
      this._handleDownloadComplete(data);
    });
    
    // Listen for download failure reports
    this.eventBus.on('HIDDEN_FILE:DOWNLOAD_FAILED', (data) => {
      this._handleDownloadFailed(data);
    });
  }
  
  /**
   * Check if a URL should be processed by a worker.
   * 
   * @param {string} url - Hidden file URL to check
   * @returns {boolean} True if URL should be processed (not seen before and not pending)
   */
  shouldProcess(url) {
    const normalizedUrl = this._normalizeUrl(url);
    
    // Already in registry (downloaded or failed)
    if (this.registry.has(normalizedUrl)) {
      this.stats.totalSkippedDuplicates++;
      return false;
    }
    
    // Currently being processed by another worker
    if (this.pending.has(normalizedUrl)) {
      this.stats.totalSkippedDuplicates++;
      return false;
    }
    
    return true;
  }
  
  /**
   * Mark a URL as being processed by a worker.
   * 
   * @param {string} url - Hidden file URL
   * @param {string} pageId - Page ID that discovered this URL
   * @returns {boolean} True if successfully marked (false if already exists)
   */
  markPending(url, pageId) {
    const normalizedUrl = this._normalizeUrl(url);
    
    if (this.registry.has(normalizedUrl) || this.pending.has(normalizedUrl)) {
      return false;
    }
    
    this.pending.add(normalizedUrl);
    this.stats.totalDiscovered++;
    
    this.logger.debug('HiddenFileRegistry', 
      `Marked pending: ${this._truncateUrl(normalizedUrl)} (discovered on ${pageId?.substring(0, 8)}...)`);
    
    this.eventBus.emit('HIDDEN_FILE:PENDING', {
      url: normalizedUrl,
      pageId,
      pendingCount: this.pending.size
    });
    
    return true;
  }
  
  /**
   * Record a successful download.
   * 
   * @param {string} url - Hidden file URL
   * @param {string} savedPath - Local path where file was saved
   * @param {string} pageId - Page ID that processed this URL
   * @param {number} [fileSize=0] - Size of downloaded file in bytes
   */
  recordDownload(url, savedPath, pageId, fileSize = 0) {
    const normalizedUrl = this._normalizeUrl(url);
    
    // Remove from pending
    this.pending.delete(normalizedUrl);
    
    // Add to registry
    this.registry.set(normalizedUrl, {
      status: 'downloaded',
      savedPath,
      discoveredOn: pageId,
      discoveredAt: Date.now(),
      downloadedAt: Date.now(),
      error: null
    });
    
    this.stats.totalDownloaded++;
    this.stats.bytesDownloaded += fileSize;
    
    this.logger.debug('HiddenFileRegistry',
      `Recorded download: ${this._truncateUrl(normalizedUrl)} → ${savedPath}`);
    
    this.eventBus.emit('HIDDEN_FILE:REGISTERED', {
      url: normalizedUrl,
      savedPath,
      pageId,
      totalDownloaded: this.stats.totalDownloaded
    });
  }
  
  /**
   * Record a failed download attempt.
   * 
   * @param {string} url - Hidden file URL
   * @param {string} pageId - Page ID that attempted download
   * @param {string} errorMessage - Error description
   */
  recordFailure(url, pageId, errorMessage) {
    const normalizedUrl = this._normalizeUrl(url);
    
    // Remove from pending
    this.pending.delete(normalizedUrl);
    
    // Add to registry as failed
    this.registry.set(normalizedUrl, {
      status: 'failed',
      savedPath: null,
      discoveredOn: pageId,
      discoveredAt: Date.now(),
      downloadedAt: null,
      error: errorMessage
    });
    
    this.stats.totalFailed++;
    
    this.logger.warn('HiddenFileRegistry',
      `Download failed: ${this._truncateUrl(normalizedUrl)} - ${errorMessage}`);
    
    this.eventBus.emit('HIDDEN_FILE:FAILED', {
      url: normalizedUrl,
      pageId,
      error: errorMessage
    });
  }
  
  /**
   * Get the saved path for a URL if already downloaded.
   * 
   * @param {string} url - Hidden file URL
   * @returns {string|null} Saved path or null if not downloaded
   */
  getSavedPath(url) {
    const normalizedUrl = this._normalizeUrl(url);
    const entry = this.registry.get(normalizedUrl);
    
    if (entry?.status === 'downloaded') {
      return entry.savedPath;
    }
    
    return null;
  }
  
  /**
   * Get full entry for a URL.
   * 
   * @param {string} url - Hidden file URL
   * @returns {HiddenFileEntry|null} Entry or null if not found
   */
  getEntry(url) {
    const normalizedUrl = this._normalizeUrl(url);
    return this.registry.get(normalizedUrl) || null;
  }
  
  /**
   * Check if URL is currently being processed.
   * 
   * @param {string} url - Hidden file URL
   * @returns {boolean} True if pending
   */
  isPending(url) {
    const normalizedUrl = this._normalizeUrl(url);
    return this.pending.has(normalizedUrl);
  }
  
  /**
   * Check if URL has been seen (pending, downloaded, or failed).
   * 
   * @param {string} url - Hidden file URL
   * @returns {boolean} True if URL is known
   */
  hasUrl(url) {
    const normalizedUrl = this._normalizeUrl(url);
    return this.registry.has(normalizedUrl) || this.pending.has(normalizedUrl);
  }
  
  /**
   * Get current statistics.
   * 
   * @returns {Object} Registry statistics
   */
  getStatistics() {
    return {
      ...this.stats,
      currentPending: this.pending.size,
      registrySize: this.registry.size,
      downloadedEntries: Array.from(this.registry.values())
        .filter(e => e.status === 'downloaded').length,
      failedEntries: Array.from(this.registry.values())
        .filter(e => e.status === 'failed').length
    };
  }
  
  /**
   * Serialize registry for IPC transfer to workers.
   * 
   * @returns {Object} Serializable registry state
   */
  serialize() {
    const entries = {};
    for (const [url, entry] of this.registry) {
      entries[url] = entry;
    }
    
    return {
      entries,
      pending: Array.from(this.pending),
      stats: this.stats
    };
  }
  
  /**
   * Get all downloaded files as URL → path mapping.
   * Useful for link rewriting.
   * 
   * @returns {Object} URL to local path mapping
   */
  getUrlToPathMap() {
    const map = {};
    for (const [url, entry] of this.registry) {
      if (entry.status === 'downloaded' && entry.savedPath) {
        map[url] = entry.savedPath;
      }
    }
    return map;
  }
  
  /**
   * Handle worker query about a hidden file URL.
   * @private
   */
  _handleWorkerQuery({ workerId, url, responseCallback }) {
    const normalizedUrl = this._normalizeUrl(url);
    
    const response = {
      shouldProcess: this.shouldProcess(url),
      savedPath: this.getSavedPath(url),
      isPending: this.isPending(url),
      status: this.registry.get(normalizedUrl)?.status || null
    };
    
    if (responseCallback) {
      responseCallback(response);
    }
    
    this.eventBus.emit('HIDDEN_FILE:QUERY_RESPONSE', {
      workerId,
      url: normalizedUrl,
      response
    });
  }
  
  /**
   * Handle worker registration of a new hidden file.
   * @private
   */
  _handleWorkerRegistration({ url, pageId }) {
    this.markPending(url, pageId);
  }
  
  /**
   * Handle download completion report from worker.
   * @private
   */
  _handleDownloadComplete({ url, savedPath, pageId, fileSize }) {
    this.recordDownload(url, savedPath, pageId, fileSize);
  }
  
  /**
   * Handle download failure report from worker.
   * @private
   */
  _handleDownloadFailed({ url, pageId, error }) {
    this.recordFailure(url, pageId, error);
  }
  
  /**
   * Normalize URL for consistent comparison.
   * @private
   * @param {string} url - URL to normalize
   * @returns {string} Normalized URL
   */
  _normalizeUrl(url) {
    try {
      const urlObj = new URL(url);
      // Remove query params that don't affect the file
      urlObj.searchParams.delete('cache');
      urlObj.searchParams.delete('t');
      // Normalize to lowercase host
      urlObj.hostname = urlObj.hostname.toLowerCase();
      return urlObj.toString();
    } catch {
      return url;
    }
  }
  
  /**
   * Truncate URL for logging.
   * @private
   */
  _truncateUrl(url, maxLength = 60) {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength - 3) + '...';
  }
  
  /**
   * Reset registry (for testing or restart).
   */
  reset() {
    this.registry.clear();
    this.pending.clear();
    this.stats = {
      totalDiscovered: 0,
      totalDownloaded: 0,
      totalFailed: 0,
      totalSkippedDuplicates: 0,
      bytesDownloaded: 0
    };
    
    this.logger.debug('HiddenFileRegistry', 'Registry reset');
  }
}

module.exports = GlobalHiddenFileRegistry;
