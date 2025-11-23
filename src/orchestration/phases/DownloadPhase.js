/**
 * @fileoverview Phase 5: Download - Parallel page and asset downloading
 * @module orchestration/phases/DownloadPhase
 * @description Manages full-fidelity download of pages and assets.
 */

const { MESSAGE_TYPES } = require('../../core/ProtocolDefinitions');
const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class DownloadPhase
 * @extends PhaseStrategy
 * @classdesc Orchestrates parallel page downloading with link rewriting.
 */
class DownloadPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Orchestrates parallel page downloading
   * @description
   * 1. Consumes tasks from ExecutionQueue (respecting dependencies)
   * 2. Dispatches IPC_DOWNLOAD tasks to workers
   * 3. Tracks progress via dashboard
   * 4. Handles retries for failed downloads
   * @param {Array<PageContext>} canonicalContexts - Pages to download
   * @param {Map} linkRewriteMap - URL -> Local Path map
   * @returns {Promise<void>}
   */
  async execute(canonicalContexts, linkRewriteMap) {
    this.logger.separator('Phase 5: Download');
    this.orchestrator.eventBus.emit('PHASE:CHANGED', {
      phase: 'download',
      data: { total: canonicalContexts.length }
    });

    // Build queue
    this.queueManager.buildDownloadQueue(canonicalContexts);

    // Process queue
    while (!this.queueManager.isDownloadComplete()) {
      const downloadTask = this.queueManager.nextDownload(this.config.OUTPUT_DIR);

      if (!downloadTask) {
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }

      // Emit progress
      const stats = this.queueManager.getStatistics();
      this.orchestrator.eventBus.emit('EXECUTION:PROGRESS', {
        pending: stats.pending.download || 0,
        active: this.browserManager.getAllocatedCount(),
        completed: stats.downloaded,
        total: canonicalContexts.length,
        failed: stats.failed
      });

      const { context, savePath } = downloadTask;

      // Execute download
      const workerId = await this.browserManager.execute(MESSAGE_TYPES.DOWNLOAD, {
        url: context.url,
        pageId: context.id,
        parentId: context.parentId,
        depth: context.depth,
        savePath: savePath,
        cookies: this.orchestrator.cookies,
        linkRewriteMap: Object.fromEntries(linkRewriteMap)
      });

      const titleRegistry = this.queueManager.getTitleRegistry();
      let title = titleRegistry[context.id] || context.title || 'Untitled';
      if (title.match(/^[a-f0-9]{32}$/i)) {
        title = `Page ${title.substring(0, 6)}...`;
      }

      this.orchestrator.eventBus.emit('WORKER:BUSY', {
        workerId,
        task: { description: `Downloading '${title}'...` }
      });
    }

    // Wait for completion
    while (!this.queueManager.isDownloadComplete()) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const stats = this.queueManager.getStatistics();
    this.logger.success('DOWNLOAD', `Complete: ${stats.downloaded} page(s) downloaded`);
  }
}

module.exports = DownloadPhase;
