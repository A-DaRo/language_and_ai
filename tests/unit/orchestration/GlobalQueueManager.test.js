const GlobalQueueManager = require('../../../src/orchestration/GlobalQueueManager');
const { createMockPageContext } = require('../../helpers/factories');
const SystemEventBus = require('../../../src/core/SystemEventBus');

describe('GlobalQueueManager', () => {
  let queueManager;

  beforeEach(() => {
    queueManager = new GlobalQueueManager();
  });

  afterEach(() => {
    SystemEventBus._reset();
  });

  test('enqueues discovery tasks and tracks visited URLs', () => {
    const root = createMockPageContext({
      url: 'https://notion.so/Page-Title-29abcdef1234567890123456789012ab',
      title: 'Root',
      depth: 0
    });

    const enqueued = queueManager.enqueueDiscovery(root, true);
    expect(enqueued).toBe(true);

    // duplicate enqueue should fail
    const duplicate = queueManager.enqueueDiscovery(root, false);
    expect(duplicate).toBe(false);

    // Use proper accessors instead of direct property access
    expect(queueManager.discoveryQueue.getLength()).toBe(1);
    expect(queueManager.discoveryQueue.getMaxDepth()).toBe(0);
  });

  test('next disc retrieves task and increments pending count', () => {
    const ctx = createMockPageContext({
      url: 'https://notion.so/Test-29aaaaaabbbbbbccccccddddddeeeeee',
      title: 'Test',
      depth: 1
    });
    queueManager.enqueueDiscovery(ctx);

    const task = queueManager.nextDiscovery();
    expect(task).toBeTruthy();
    expect(task.pageContext).toBe(ctx);
    // Use proper accessor for pending count
    expect(queueManager.discoveryQueue.getPendingCount()).toBe(1);
  });

  test('completeDiscovery creates child contexts from links', () => {
    const parent = createMockPageContext({
      url: 'https://notion.so/Parent-29abcdef1234567890123456789012ab',
      title: 'Parent',
      depth: 0
    });
    queueManager.enqueueDiscovery(parent);
    queueManager.nextDiscovery(); // mark as in-flight

    const links = [
      { url: 'https://notion.so/Child1-29bbbbbb1234567890123456789012ab', text: 'Child 1' },
      { url: 'https://notion.so/Child2-29cccccc1234567890123456789012ab', text: 'Child 2' }
    ];

    const newContexts = queueManager.completeDiscovery(parent.id, links, {}, 'Parent Title');
    expect(newContexts.length).toBe(2);
    expect(newContexts[0].depth).toBe(1);
    expect(newContexts[1].depth).toBe(1);

    // Use proper accessor for title registry
    expect(queueManager.titleRegistry.get(parent.id)).toBe('Parent Title');
  });

  test('isDiscoveryComplete returns true when queue and pending are empty', () => {
    expect(queueManager.isDiscoveryComplete()).toBe(true);

    const ctx = createMockPageContext({
      url: 'https://notion.so/Test-29aaaaaabbbbbbccccccddddddeeeeee',
      title: 'Test',
      depth: 0
    });
    queueManager.enqueueDiscovery(ctx);
    expect(queueManager.isDiscoveryComplete()).toBe(false);

    queueManager.nextDiscovery();
    expect(queueManager.isDiscoveryComplete()).toBe(false);

    queueManager.completeDiscovery(ctx.id, []);
    expect(queueManager.isDiscoveryComplete()).toBe(true);
  });
});
