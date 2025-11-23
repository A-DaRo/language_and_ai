/**
 * @fileoverview Task Execution Router for Worker Process
 * @module worker/TaskRunner
 * @description Routes IPC commands to appropriate task handlers.
 * 
 * **CRITICAL**: Runs in Worker process. Delegates to specialized handlers.
 */

const { MESSAGE_TYPES, serializeError } = require('../core/ProtocolDefinitions');
const Logger = require('../core/Logger');
const DiscoveryHandler = require('./handlers/DiscoveryHandler');
const DownloadHandler = require('./handlers/DownloadHandler');

/**
 * @class TaskRunner
 * @classdesc Routes task commands to appropriate handlers
 */
class TaskRunner {
  /**
   * @param {import('puppeteer').Browser} browser - Puppeteer browser instance
   */
  constructor(browser) {
    this.browser = browser;
    this.config = new (require('../core/Config'))();
    this.cookies = [];
    this.titleRegistry = {};
    this.logger = Logger.getInstance();

    // Initialize handlers
    this.discoveryHandler = new DiscoveryHandler(browser, {
      config: this.config,
      cookies: this.cookies,
      titleRegistry: this.titleRegistry
    });

    this.downloadHandler = new DownloadHandler(browser, {
      config: this.config,
      cookies: this.cookies,
      titleRegistry: this.titleRegistry
    });
  }

  /**
   * Set cookies for this worker
   * @async
   * @param {Array<Object>} cookies - Array of Puppeteer cookie objects
   * @returns {Promise<void>}
   */
  async setCookies(cookies) {
    this.cookies = cookies || [];
    this.discoveryHandler.cookies = this.cookies;
    this.downloadHandler.cookies = this.cookies;
    this.logger.info('TaskRunner', `Received ${this.cookies.length} cookie(s)`);
  }

  /**
   * Initialize or update title registry
   * @param {Object} titleRegistry - ID-to-title map
   * @param {boolean} [isDelta=false] - Whether this is a delta update
   * @returns {void}
   */
  setTitleRegistry(titleRegistry, isDelta = false) {
    if (isDelta) {
      this.titleRegistry = { ...this.titleRegistry, ...titleRegistry };
      this.logger.info('TaskRunner', `Updated title registry with ${Object.keys(titleRegistry).length} delta(s)`);
    } else {
      this.titleRegistry = titleRegistry || {};
      this.logger.info('TaskRunner', `Initialized title registry with ${Object.keys(this.titleRegistry).length} title(s)`);
    }

    this.discoveryHandler.titleRegistry = this.titleRegistry;
    this.downloadHandler.titleRegistry = this.titleRegistry;
  }

  /**
   * Execute a task based on type
   * @async
   * @param {string} taskType - Task type (DISCOVER or DOWNLOAD)
   * @param {Object} payload - Task payload
   * @returns {Promise<Object>} Task result (WorkerResult format)
   */
  async execute(taskType, payload) {
    try {
      const targetId = payload.pageId || (payload.url ? payload.url.split('/').pop() : 'unknown');
      this.logger.info('TaskRunner', `Executing ${taskType} task for: ${targetId}`);

      let result;

      if (taskType === MESSAGE_TYPES.DISCOVER) {
        result = await this.discoveryHandler.handle(payload);
      } else if (taskType === MESSAGE_TYPES.DOWNLOAD) {
        result = await this.downloadHandler.handle(payload);
      } else {
        throw new Error(`Unknown task type: ${taskType}`);
      }

      return {
        type: MESSAGE_TYPES.RESULT,
        taskType,
        data: result
      };

    } catch (error) {
      this.logger.error('TaskRunner', `Task failed`, error);

      return {
        type: MESSAGE_TYPES.RESULT,
        taskType,
        error: serializeError(error)
      };
    }
  }

  /**
   * Cleanup resources
   * @async
   * @returns {Promise<void>}
   */
  async cleanup() {
    await this.discoveryHandler.cleanup();
    await this.downloadHandler.cleanup();
  }
}

module.exports = TaskRunner;


module.exports = TaskRunner;
