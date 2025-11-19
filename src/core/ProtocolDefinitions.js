/**
 * @fileoverview IPC Protocol Definitions for Master-Worker Communication
 * @module core/ProtocolDefinitions
 * @description Defines the strict contract for Inter-Process Communication (IPC) between
 * the Master process and Worker processes. All message types and payload structures are
 * defined here to ensure type safety and consistency across the distributed system.
 */

/**
 * @typedef {Object} MessageType
 * @property {string} INIT - Initialize worker with configuration
 * @property {string} DISCOVER - Execute discovery task (metadata only)
 * @property {string} DOWNLOAD - Execute download task (full scrape with assets)
 * @property {string} SHUTDOWN - Graceful shutdown command
 * @property {string} READY - Worker ready signal
 * @property {string} RESULT - Task result response
 * @property {string} SET_COOKIES - Broadcast cookies to worker
 */

/**
 * Message type constants for IPC communication
 * @constant {MessageType}
 */
const MESSAGE_TYPES = {
  // Master -> Worker commands
  INIT: 'IPC_INIT',
  DISCOVER: 'IPC_DISCOVER',
  DOWNLOAD: 'IPC_DOWNLOAD',
  SHUTDOWN: 'IPC_SHUTDOWN',
  SET_COOKIES: 'IPC_SET_COOKIES',
  
  // Worker -> Master responses
  READY: 'IPC_READY',
  RESULT: 'IPC_RESULT'
};

/**
 * @typedef {Object} InitPayload
 * @property {Object} config - Configuration object for the worker
 * @property {string} config.NOTION_PAGE_URL - Base Notion URL
 * @property {string} config.OUTPUT_DIR - Output directory path
 * @property {number} config.MAX_RECURSION_DEPTH - Maximum recursion depth
 * @property {number} config.MAX_EXPANSION_DEPTH - Maximum toggle expansion depth
 * @property {number} config.TIMEOUT_PAGE_LOAD - Page load timeout in ms
 * @property {number} config.TIMEOUT_EXPAND_TOGGLE - Toggle expansion timeout in ms
 * @property {number} config.EXPAND_WAIT_TIME - Wait time between expansions in ms
 */

/**
 * @typedef {Object} DiscoverPayload
 * @property {string} url - Target URL to discover
 * @property {string} pageId - Unique page identifier (Notion ID)
 * @property {string|null} parentId - Parent page ID (null for root)
 * @property {number} depth - Current depth in the hierarchy
 * @property {boolean} isFirstPage - Whether this is the root page (for cookie capture)
 * @property {Array<Object>} [cookies] - Optional cookies to set before navigation
 */

/**
 * @typedef {Object} DownloadPayload
 * @property {string} url - Target URL to download
 * @property {string} pageId - Unique page identifier (Notion ID)
 * @property {string|null} parentId - Parent page ID (null for root)
 * @property {number} depth - Current depth in the hierarchy
 * @property {string} targetFilePath - Calculated file path where HTML should be saved (relative to OUTPUT_DIR)
 * @property {Array<Object>} cookies - Cookies to set before navigation
 * @property {Object} linkRewriteMap - Map of NotionID -> RelativeFilePath for link rewriting
 */

/**
 * @typedef {Object} SetCookiesPayload
 * @property {Array<Object>} cookies - Array of cookie objects to set
 */

/**
 * @typedef {Object} SerializedError
 * @property {string} message - Error message
 * @property {string} name - Error name/type
 * @property {string} stack - Stack trace
 * @property {string} [code] - Error code if available
 */

/**
 * @typedef {Object} DiscoveryResult
 * @property {boolean} success - Whether the discovery succeeded
 * @property {string} pageId - Page identifier
 * @property {string} url - Page URL
 * @property {string} title - Page title
 * @property {string} [displayTitle] - Display title if different from title
 * @property {Array<Object>} links - Extracted links
 * @property {Array<Object>} [cookies] - Captured cookies (only on first page)
 * @property {Object} [metadata] - Additional metadata
 */

/**
 * @typedef {Object} DownloadResult
 * @property {boolean} success - Whether the download succeeded
 * @property {string} pageId - Page identifier
 * @property {string} url - Page URL
 * @property {string} savedPath - Path where the HTML was saved (relative to OUTPUT_DIR)
 * @property {number} assetsDownloaded - Number of assets downloaded
 * @property {number} linksRewritten - Number of links rewritten
 */

/**
 * @typedef {Object} WorkerResult
 * @property {string} type - Result type (must be MESSAGE_TYPES.RESULT)
 * @property {string} taskType - Original task type (DISCOVER or DOWNLOAD)
 * @property {DiscoveryResult|DownloadResult} [data] - Result data (present on success)
 * @property {SerializedError} [error] - Serialized error object (present on failure)
 */

/**
 * @typedef {Object} IPCMessage
 * @property {string} type - Message type from MESSAGE_TYPES
 * @property {InitPayload|DiscoverPayload|DownloadPayload|SetCookiesPayload|WorkerResult} [payload] - Message payload
 */

/**
 * Serialize an Error object for IPC transmission
 * @param {Error} error - The error to serialize
 * @returns {SerializedError} Serialized error object
 * @example
 * const err = new Error('Failed to load page');
 * const serialized = serializeError(err);
 * // Returns: { message: 'Failed to load page', name: 'Error', stack: '...' }
 */
function serializeError(error) {
  return {
    message: error.message || 'Unknown error',
    name: error.name || 'Error',
    stack: error.stack || '',
    code: error.code || undefined
  };
}

/**
 * Deserialize an error object back into an Error instance
 * @param {SerializedError} serializedError - The serialized error
 * @returns {Error} Error instance
 * @example
 * const error = deserializeError({ message: 'Failed', name: 'Error', stack: '...' });
 * // Returns: Error instance with message and stack
 */
function deserializeError(serializedError) {
  const error = new Error(serializedError.message);
  error.name = serializedError.name;
  error.stack = serializedError.stack;
  if (serializedError.code) {
    error.code = serializedError.code;
  }
  return error;
}

/**
 * Validate that a message conforms to the IPC protocol
 * @param {IPCMessage} message - Message to validate
 * @returns {boolean} True if valid
 * @throws {Error} If message is invalid
 */
function validateMessage(message) {
  if (!message || typeof message !== 'object') {
    throw new Error('Invalid IPC message: must be an object');
  }
  
  if (!message.type || typeof message.type !== 'string') {
    throw new Error('Invalid IPC message: missing or invalid type');
  }
  
  const validTypes = Object.values(MESSAGE_TYPES);
  if (!validTypes.includes(message.type)) {
    throw new Error(`Invalid IPC message type: ${message.type}`);
  }
  
  return true;
}

module.exports = {
  MESSAGE_TYPES,
  serializeError,
  deserializeError,
  validateMessage
};
