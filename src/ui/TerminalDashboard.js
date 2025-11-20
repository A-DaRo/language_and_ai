/**
 * @fileoverview Renderer for the Multi-Bar Terminal Dashboard
 * @module ui/TerminalDashboard
 */

const cliProgress = require('cli-progress');

/**
 * @class TerminalDashboard
 * @description Encapsulates the rendering logic for the terminal UI. It is a "dumb"
 * component that receives data and updates its visual state. It does not contain
 * any application logic.
 */
class TerminalDashboard {
  /**
   * @constructor
   * @param {number} workerCount - The number of worker slots to create.
   */
  constructor(workerCount) {
    this.multibar = new cliProgress.MultiBar({
      clearOnComplete: false,
      hideCursor: true,
      format: '{bar} | {label}',
    }, cliProgress.Presets.shades_classic);

    // Header and Footer for static text
    this.header = this.multibar.create(1, 1, { label: 'Initializing...' });
    this.header.setFormat('{label}');

    this.footer = this.multibar.create(1, 1, { label: 'System is starting...' });
    this.footer.setFormat('  \u2514\u2500 {label}');

    // Dynamic bars for progress and workers
    this.progressBars = new Map();
    this.workerBars = new Map();

    for (let i = 0; i < workerCount; i++) {
      const bar = this.multibar.create(1, 0, { label: `Worker ${i + 1}: [IDLE]` });
      bar.setFormat('   {label}');
      this.workerBars.set(i, bar);
    }
  }

  /**
   * @method setMode
   * @summary Reconfigures the dashboard layout for a specific phase.
   * @param {'discovery' | 'download'} mode - The phase to display.
   * @param {Object} [initialData={}] - Initial data for the mode.
   */
  setMode(mode, initialData = {}) {
    this.progressBars.forEach(bar => this.multibar.remove(bar));
    this.progressBars.clear();

    if (mode === 'discovery') {
      const bar = this.multibar.create(1, 0, { label: 'Pages Found: 0 | In Queue: 0 | Conflicts: 0' });
      bar.setFormat('  Progress: {label}');
      this.progressBars.set('discoveryStats', bar);
    } else if (mode === 'download') {
      const bar = this.multibar.create(initialData.total || 1, initialData.completed || 0, {
        label: `[Pending: ${initialData.pending || 0}] [Active: 0] [Complete: 0/${initialData.total || 0}] [Failed: 0]`
      });
      bar.setFormat('  Progress: {label}');
      this.progressBars.set('downloadStats', bar);
    }
  }

  /**
   * @method updateHeader
   * @param {string} title - The main title to display.
   */
  updateHeader(title) {
    this.header.update(1, { label: title });
  }

  /**
   * @method updateDiscoveryStats
   * @param {Object} stats - Discovery statistics.
   * @param {number} stats.pagesFound
   * @param {number} stats.inQueue
   * @param {number} stats.conflicts
   * @param {number} stats.currentDepth
   */
  updateDiscoveryStats({ pagesFound, inQueue, conflicts, currentDepth }) {
    const bar = this.progressBars.get('discoveryStats');
    if (bar) {
      bar.update(1, { label: `Pages Found: ${pagesFound} | In Queue: ${inQueue} | Conflicts: ${conflicts} | Current Depth: ${currentDepth}` });
    }
  }

  /**
   * @method updateDownloadStats
   * @param {Object} stats - Download statistics.
   * @param {number} stats.pending
   * @param {number} stats.active
   * @param {number} stats.completed
   * @param {number} stats.total
   * @param {number} stats.failed
   */
  updateDownloadStats({ pending, active, completed, total, failed }) {
    const bar = this.progressBars.get('downloadStats');
    if (bar) {
      bar.setTotal(total);
      bar.update(completed, { label: `[Pending: ${pending}] [Active: ${active}] [Complete: ${completed}/${total}] [Failed: ${failed}]` });
    }
  }

  /**
   * @method updateWorkerStatus
   * @param {number} slotIndex - The visual slot for the worker.
   * @param {string} statusText - The text to display.
   */
  updateWorkerStatus(slotIndex, statusText) {
    const bar = this.workerBars.get(slotIndex);
    if (bar) {
      bar.update(1, { label: `Worker ${slotIndex + 1}: ${statusText}` });
    }
  }

  /**
   * @method updateFooter
   * @param {string} text - The text for the log ticker.
   */
  updateFooter(text) {
    this.footer.update(1, { label: text });
  }

  /**
   * @method stop
   * @description Stops the renderer and restores the cursor.
   */
  stop() {
    this.multibar.stop();
  }
}

module.exports = TerminalDashboard;
