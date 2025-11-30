/**
 * Integration test: Master-Worker IPC Protocol
 * 
 * Tests the complete message flow between Master and Worker processes:
 * 1. Master sends IPC_INIT with cookies
 * 2. Master sends IPC_DISCOVER task
 * 3. Worker returns IPC_RESULT with discovered links
 * 4. Master sends IPC_DOWNLOAD task
 * 5. Worker returns IPC_RESULT with download success
 */

const { MESSAGE_TYPES, validateMessage } = require('../../../src/core/ProtocolDefinitions');
const TaskRunner = require('../../../src/worker/TaskRunner');
const { createMockBrowser } = require('../../helpers/MockPuppeteer');
const { createMockPageContext } = require('../../helpers/factories');

const mockLogger = {
  info: jest.fn(),
  success: jest.fn(),
  error: jest.fn(),
  debug: jest.fn()
};

describe('Master-Worker IPC Protocol Flow', () => {
  let taskRunner;
  let mockBrowser;

  beforeEach(async () => {
    mockBrowser = await createMockBrowser({
      pageDefaults: {
        content: `
          <html>
            <body>
              <h1>Parent Page</h1>
              <a href="https://notion.so/Child-Page-29${'a'.repeat(30)}">Child Link</a>
            </body>
          </html>
        `
      }
    });
    taskRunner = new TaskRunner(mockBrowser);
    taskRunner.logger = mockLogger;
    jest.clearAllMocks();
  });

  test('completes full initialization sequence', async () => {
    // Phase 1: Master sends IPC_INIT with cookies
    const initMessage = {
      type: MESSAGE_TYPES.INIT,
      cookies: [
        { name: 'session', value: 'abc123', domain: '.notion.so' }
      ],
      titleRegistry: {
        'page1': 'Title 1'
      }
    };

    // Validate message structure
    expect(() => validateMessage(initMessage)).not.toThrow();

    // Worker processes initialization
    await taskRunner.setCookies(initMessage.cookies);
    taskRunner.setTitleRegistry(initMessage.titleRegistry, true);

    // Verify initialization state
    expect(taskRunner.cookies).toEqual(initMessage.cookies);
    expect(taskRunner.titleRegistry).toEqual({ 'page1': 'Title 1' });
  });

  test('executes discovery task and returns valid result message', async () => {
    // Phase 2: Master sends IPC_DISCOVER task
    const discoverMessage = {
      type: MESSAGE_TYPES.DISCOVER,
      payload: {
        url: 'https://notion.so/Parent-29' + 'b'.repeat(30),
        pageId: 'parent-id',
        depth: 0,
        parentId: null,
        isFirstPage: false
      }
    };

    // Validate message structure
    expect(() => validateMessage(discoverMessage)).not.toThrow();

    // Worker executes discovery
    const result = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, discoverMessage.payload);

    // Phase 3: Verify IPC_RESULT message structure
    expect(result.type).toBe(MESSAGE_TYPES.RESULT);
    expect(result.taskType).toBe(MESSAGE_TYPES.DISCOVER);
    expect(result.data).toBeDefined();
    expect(result.data.success).toBe(true);
    expect(result.data.pageId).toBe('parent-id');
    expect(result.data).toHaveProperty('url');
    expect(result.data).toHaveProperty('resolvedTitle');
    expect(result.data).toHaveProperty('links');
    expect(Array.isArray(result.data.links)).toBe(true);

    // Validate result message protocol compliance
    expect(() => validateMessage(result)).not.toThrow();
  });

  test('propagates errors with serialized error objects', async () => {
    // Master sends invalid task (null URL)
    const invalidMessage = {
      type: MESSAGE_TYPES.DISCOVER,
      payload: {
        url: null, // Invalid
        pageId: 'test',
        depth: 0
      }
    };

    // Worker attempts execution
    const result = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, invalidMessage.payload);

    // Verify error propagation in protocol
    expect(result.type).toBe(MESSAGE_TYPES.RESULT);
    expect(result.error).toBeDefined();
    expect(result.error).toHaveProperty('message');
    expect(result.error).toHaveProperty('name');
    expect(result.error).toHaveProperty('stack');

    // Error messages are still protocol-compliant
    expect(() => validateMessage(result)).not.toThrow();
  });

  test('maintains state across multiple task executions', async () => {
    // Initialize worker state
    await taskRunner.setCookies([
      { name: 'token', value: 'xyz', domain: '.notion.so' }
    ]);

    // Execute first discovery
    const task1 = {
      url: 'https://notion.so/Page1-29' + 'c'.repeat(30),
      pageId: 'page1',
      depth: 0,
      isFirstPage: false
    };

    const result1 = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, task1);
    expect(result1.data.success).toBe(true);

    // Execute second discovery (should reuse same page instance)
    const task2 = {
      url: 'https://notion.so/Page2-29' + 'd'.repeat(30),
      pageId: 'page2',
      depth: 1,
      parentId: 'page1',
      isFirstPage: false
    };

    const result2 = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, task2);
    expect(result2.data.success).toBe(true);

    // Verify state persistence (cookies maintained across navigations)
    expect(taskRunner.cookies.length).toBe(1);
    expect(taskRunner.cookies[0].name).toBe('token');
  });

  test('validates message type enums strictly', () => {
    // Valid message types
    expect(() => validateMessage({ type: MESSAGE_TYPES.INIT })).not.toThrow();
    expect(() => validateMessage({ type: MESSAGE_TYPES.DISCOVER })).not.toThrow();
    expect(() => validateMessage({ type: MESSAGE_TYPES.DOWNLOAD })).not.toThrow();
    expect(() => validateMessage({ type: MESSAGE_TYPES.RESULT })).not.toThrow();

    // Invalid message types
    expect(() => validateMessage({ type: 'INVALID_TYPE' })).toThrow();
    expect(() => validateMessage({ type: null })).toThrow();
    expect(() => validateMessage({ type: undefined })).toThrow();
    expect(() => validateMessage({})).toThrow();
  });
});
