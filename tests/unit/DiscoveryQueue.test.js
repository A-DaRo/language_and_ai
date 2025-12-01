const DiscoveryQueue = require('../../src/orchestration/queues/DiscoveryQueue');
const SystemEventBus = require('../../src/core/SystemEventBus');
const { createMockPageContext } = require('../helpers/factories');
const { waitForEvent } = require('../helpers/eventHelpers');

const buildNotionUrl = (suffix = 'a') => {
  const normalized = (suffix + (suffix.slice(-1) || 'a').repeat(30)).slice(0, 30);
  return `https://notion.so/page-29${normalized}`;
};

describe('DiscoveryQueue event emissions', () => {
  let queue;
  let bus;

  beforeEach(() => {
    queue = new DiscoveryQueue();
    bus = SystemEventBus.getInstance();
  });

  afterEach(() => {
    SystemEventBus._reset();
  });

  const makeContext = (suffix, overrides = {}) => createMockPageContext({
    url: buildNotionUrl(suffix),
    title: overrides.title || 'Test Page',
    depth: overrides.depth || 0,
    parentId: overrides.parentId || null
  });

  test('emits QUEUE_READY when the first task is enqueued', async () => {
    const emitted = waitForEvent(bus, 'DISCOVERY:QUEUE_READY');
    queue.enqueue(makeContext('a'), true);

    await expect(emitted).resolves.toMatchObject({ queueLength: 1 });
  });

  test('emits QUEUE_EMPTY when the last item is dequeued', async () => {
    queue.enqueue(makeContext('a'));

    const emitted = waitForEvent(bus, 'DISCOVERY:QUEUE_EMPTY');
    queue.next();

    await expect(emitted).resolves.toMatchObject({ queueLength: 0, pendingCount: 1 });
  });

  test('emits ALL_IDLE when queue is empty and pending tasks complete', async () => {
    queue.enqueue(makeContext('a'));
    const task = queue.next();

    const emitted = waitForEvent(bus, 'DISCOVERY:ALL_IDLE');
    queue.markComplete(task.pageContext.id);

    await expect(emitted).resolves.toMatchObject({ queueLength: 0, pendingCount: 0 });
  });

  test('does not emit ALL_IDLE when queue still has items', async () => {
    queue.enqueue(makeContext('a'));
    queue.enqueue(makeContext('b'));
    const task = queue.next();

    const wait = waitForEvent(bus, 'DISCOVERY:ALL_IDLE', 100);
    queue.markComplete(task.pageContext.id);

    await expect(wait).rejects.toThrow('Event DISCOVERY:ALL_IDLE not emitted within 100ms');
  });

  test('markComplete is idempotent and never allows negative pending count', () => {
    queue.enqueue(makeContext('a'));
    const task = queue.next();

    expect(queue.markComplete(task.pageContext.id)).toBe(true);
    expect(queue.markComplete(task.pageContext.id)).toBe(false);
    expect(queue.getPendingCount()).toBe(0);
  });
});
