/**
 * @fileoverview Renderer for the Multi-Bar Terminal Dashboard
 * @module ui/TerminalDashboard
 * 
 * @design DASHBOARD CONSISTENCY
 * This component ensures consistent, clean terminal output:
 * 1. All dynamic text fields are truncated to prevent layout breakage
 * 2. Output goes through process.stdout (never console.log)
 * 3. ANSI escape codes used for formatting
 * 4. Worker status lines are fixed-width for stable layout
 */

const cliProgress = require('cli-progress');

/**
 * Truncate a string to a maximum length with ellipsis.
 * 
 * @param {string} str - String to truncate
 * @param {number} maxLength - Maximum length including ellipsis
 * @returns {string} Truncated string with '...' if needed
 * 
 * @example
 * truncate('Hello World', 8); // 'Hello...'
 * truncate('Short', 10); // 'Short'
 */
function truncate(str, maxLength) {
  if (!str || typeof str !== 'string') return '';
  if (str.length <= maxLength) return str;
  if (maxLength <= 3) return str.substring(0, maxLength);
  return str.substring(0, maxLength - 3) + '...';
}

/**
 * @class TerminalDashboard
 * @description Encapsulates the rendering logic for the terminal UI. It is a "dumb"
 * component that receives data and updates its visual state. It does not contain
 * any application logic.
 * 
 * @design Features for consistent display:
 * - Text truncation on all dynamic fields (URLs, titles, status)
 * - Fixed-width columns for stable layout
 * - No console.log usage (uses process.stdout)
 * - Graceful handling of long Unicode strings
 */
class TerminalDashboard {
  /**
   * Maximum display lengths for various fields.
   * @private
   * @type {Object}
   */
  static DISPLAY_LIMITS = {
    TITLE: 50,
    URL: 60,
    STATUS: 40,
    WORKER_LABEL: 70,
    FOOTER: 80
  };

  /**
   * @constructor
   * @param {number} workerCount - The number of worker slots to create.
   */
  constructor(workerCount) {
    // Clear terminal for clean dashboard display (ANSI reset)
    process.stdout.write('\x1Bc');
    
    this.multibar = new cliProgress.MultiBar({
      clearOnComplete: false,
      hideCursor: true,
      format: '{bar} | {label}',
    }, cliProgress.Presets.shades_classic);

    // Header and Footer for static text (use 1/1 progress to display as static text)
    this.header = this.multibar.create(1, 1, { label: 'Initializing...' }, {
      format: '{label}'
    });

    this.footer = this.multibar.create(1, 1, { label: 'System is starting...' }, {
      format: '  \u2514\u2500 {label}'
    });

    // Dynamic bars for progress and workers
    this.progressBars = new Map();
    this.workerBars = new Map();

    for (let i = 0; i < workerCount; i++) {
      const bar = this.multibar.create(1, 0, { label: `[IDLE] Worker ${i + 1}: Waiting for task...` }, {
        format: '  {label}'
      });
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
      const bar = this.multibar.create(1, 0, { label: '[Pages Found: 0] [In Queue: 0] [Conflicts: 0] [Current Depth: 0]' }, {
        format: ' Progress: {label}'
      });
      this.progressBars.set('discoveryStats', bar);
    } else if (mode === 'download') {
      const bar = this.multibar.create(initialData.total || 1, initialData.completed || 0, {
        label: `[Pending: ${initialData.pending || 0}] [Active: 0] [Complete: 0/${initialData.total || 0}] [Failed: 0]`
      }, {
        format: ' Progress: {label}'
      });
      this.progressBars.set('downloadStats', bar);
    }
  }

  /**
   * @method updateHeader
   * @param {string} title - The main title to display.
   */
  updateHeader(title) {
    const safeTitle = truncate(title, TerminalDashboard.DISPLAY_LIMITS.TITLE);
    this.header.update(1, { label: safeTitle });
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
      bar.update(1, { label: `[Pages Found: ${pagesFound}] [In Queue: ${inQueue}] [Conflicts: ${conflicts || 0}] [Current Depth: ${currentDepth}]` });
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
   * @summary Update a worker's status display with automatic truncation.
   * @param {number} slotIndex - The visual slot for the worker.
   * @param {string} statusText - The text to display.
   */
  updateWorkerStatus(slotIndex, statusText) {
    const bar = this.workerBars.get(slotIndex);
    if (bar) {
      const safeStatus = truncate(statusText, TerminalDashboard.DISPLAY_LIMITS.WORKER_LABEL);
      bar.update(1, { label: safeStatus });
    }
  }

  /**
   * @method updateFooter
   * @summary Update the footer/ticker with automatic truncation.
   * @param {string} text - The text for the log ticker.
   */
  updateFooter(text) {
    const safeText = truncate(text, TerminalDashboard.DISPLAY_LIMITS.FOOTER);
    this.footer.update(1, { label: safeText });
  }

  /**
   * @method stop
   * @description Stops the renderer and restores the cursor.
   */
  stop() {
    this.multibar.stop();
  }
}

/**
 * Truncate utility exported for use by other components.
 * @type {function}
 */
TerminalDashboard.truncate = truncate;

module.exports = TerminalDashboard;
