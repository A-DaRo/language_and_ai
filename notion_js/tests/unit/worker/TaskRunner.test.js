const TaskRunner = require('../../../src/worker/TaskRunner');
const { MESSAGE_TYPES } = require('../../../src/core/ProtocolDefinitions');
const { createMockBrowser } = require('../../helpers/MockPuppeteer');

const mockLogger = {
  info: jest.fn(),
  success: jest.fn(),
  error: jest.fn(),
  debug: jest.fn()
};

describe('TaskRunner', () => {
  let taskRunner;
  let mockBrowser;

  beforeEach(async () => {
    mockBrowser = await createMockBrowser({
      pageDefaults: {
        content: '<html><body><h1>Test Page</h1><a href="https://notion.so/link1-29aaaa">Link</a></body></html>'
      }
    });
    taskRunner = new TaskRunner(mockBrowser);
    taskRunner.logger = mockLogger;
    jest.clearAllMocks();
  });

  test('sets cookies for worker', async () => {
    const cookies = [
      { name: 'session', value: 'abc123', domain: '.notion.so' }
    ];
    
    await taskRunner.setCookies(cookies);
    
    expect(taskRunner.cookies).toEqual(cookies);
    expect(taskRunner.cookies.length).toBe(1);
  });

  test('initializes title registry with full map', () => {
    const titleRegistry = {
      'page1': 'Title 1',
      'page2': 'Title 2'
    };
    
    taskRunner.setTitleRegistry(titleRegistry, false);
    
    expect(taskRunner.titleRegistry).toEqual(titleRegistry);
    expect(Object.keys(taskRunner.titleRegistry).length).toBe(2);
  });

  test('updates title registry with delta', () => {
    taskRunner.titleRegistry = { 'page1': 'Title 1' };
    
    taskRunner.setTitleRegistry({ 'page2': 'Title 2' }, true);
    
    expect(taskRunner.titleRegistry).toEqual({
      'page1': 'Title 1',
      'page2': 'Title 2'
    });
  });

  test('executes discovery task and returns result', async () => {
    const payload = {
      url: 'https://notion.so/Test-Page-29' + 'a'.repeat(30),
      pageId: 'test-page-id',
      depth: 1,
      parentId: 'parent-id',
      isFirstPage: false
    };
    
    const result = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, payload);
    
    expect(result.type).toBe(MESSAGE_TYPES.RESULT);
    expect(result.taskType).toBe(MESSAGE_TYPES.DISCOVER);
    expect(result.data.success).toBe(true);
    expect(result.data.pageId).toBe('test-page-id');
    expect(result.data.url).toBe(payload.url);
    expect(result.data).toHaveProperty('resolvedTitle');
    expect(result.data).toHaveProperty('links');
  });

  test('captures cookies on first page discovery', async () => {
    const payload = {
      url: 'https://notion.so/First-29' + 'b'.repeat(30),
      pageId: 'first-page',
      depth: 0,
      parentId: null,
      isFirstPage: true
    };
    
    const result = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, payload);
    
    expect(result.data.cookies).toBeDefined();
    expect(Array.isArray(result.data.cookies)).toBe(true);
  });

  test('serializes errors on task failure', async () => {
    // Force an error by passing invalid payload
    const result = await taskRunner.execute(MESSAGE_TYPES.DISCOVER, {
      url: null // Invalid URL will cause error
    });
    
    expect(result.type).toBe(MESSAGE_TYPES.RESULT);
    expect(result.error).toBeDefined();
    expect(result.error).toHaveProperty('message');
    expect(result.error).toHaveProperty('name');
    expect(result.error).toHaveProperty('stack');
  });

  test('reuses page instance across tasks', async () => {
    const payload1 = {
      url: 'https://notion.so/Page1-29' + 'c'.repeat(30),
      pageId: 'page1',
      depth: 0,
      isFirstPage: false
    };
    
    const payload2 = {
      url: 'https://notion.so/Page2-29' + 'd'.repeat(30),
      pageId: 'page2',
      depth: 0,
      isFirstPage: false
    };
    
    await taskRunner.execute(MESSAGE_TYPES.DISCOVER, payload1);
    const firstPage = taskRunner.page;
    
    await taskRunner.execute(MESSAGE_TYPES.DISCOVER, payload2);
    const secondPage = taskRunner.page;
    
    expect(firstPage).toBe(secondPage); // Same page reused
  });

  test('cleanup closes page', async () => {
    const payload = {
      url: 'https://notion.so/Test-29' + 'e'.repeat(30),
      pageId: 'test',
      depth: 0,
      isFirstPage: false
    };
    
    await taskRunner.execute(MESSAGE_TYPES.DISCOVER, payload);
    expect(taskRunner.page).toBeTruthy();
    
    await taskRunner.cleanup();
    expect(taskRunner.page).toBeNull();
  });
});
