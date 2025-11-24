const DiscoveryPhase = require('../../src/orchestration/phases/DiscoveryPhase');
const GlobalQueueManager = require('../../src/orchestration/GlobalQueueManager');
const SystemEventBus = require('../../src/core/SystemEventBus');
const { createMockPageContext } = require('../helpers/factories');

const buildNotionUrl = (suffix = 'a') => {
  const normalized = (suffix + (suffix.slice(-1) || 'a').repeat(30)).slice(0, 30);
  return `https://notion.so/page-29${normalized}`;
};

class MockOrchestrator {
  constructor() {
    this.logger = {
      separator: jest.fn(),
      info: jest.fn(),
      debug: jest.fn(),
      error: jest.fn(),
      success: jest.fn()
    };
    this.queueManager = new GlobalQueueManager();
    this.browserManager = {
      execute: jest.fn().mockResolvedValue('worker-1'),
      initializeWorkers: jest.fn().mockResolvedValue(undefined)
    };
    this.eventBus = SystemEventBus.getInstance();
    this.cookies = [];
    this.config = {};
  }
}

describe('DiscoveryPhase event-driven flow', () => {
  afterEach(() => {
    SystemEventBus._reset();
  });

  test('completes when ALL_IDLE event fires', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase(orchestrator);
    const root = createMockPageContext({ url: buildNotionUrl('a'), depth: 0 });
    orchestrator.queueManager.enqueueDiscovery(root, true);

    const inFlight = phase.execute(5);

    setTimeout(() => {
      orchestrator.queueManager.completeDiscovery(root.id, [], {}, 'Root Title');
    }, 50);

    await expect(inFlight).resolves.toBeUndefined();
  });

  test('times out after 30 minutes with no progress', async () => {
    jest.useFakeTimers();
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase(orchestrator);

    const execution = phase.execute(5);
    jest.advanceTimersByTime(30 * 60 * 1000 + 1);

    await expect(execution).rejects.toThrow('Discovery Phase Timeout');
    jest.useRealTimers();
  });

  test('fails fast on negative pending count', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase(orchestrator);

    const execution = phase.execute(5);

    setTimeout(() => {
      orchestrator.eventBus.emit('DISCOVERY:TASK_COMPLETED', {
        pageId: 'page-1',
        success: true,
        pendingCount: -5,
        queueLength: 0
      });
    }, 10);

    await expect(execution).rejects.toThrow('Queue state corrupted');
  });

  test('handles rapid completion without missed events', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase(orchestrator);
    const contexts = Array.from({ length: 100 }, (_, index) =>
      createMockPageContext({ url: buildNotionUrl(index.toString(16)), depth: 1 })
    );

    contexts.forEach(ctx => orchestrator.queueManager.enqueueDiscovery(ctx));

    const execution = phase.execute(5);

    setTimeout(() => {
      for (const ctx of contexts) {
        orchestrator.queueManager.completeDiscovery(ctx.id, [], {}, `Title ${ctx.id}`);
      }
    }, 50);

    await expect(execution).resolves.toBeUndefined();
  });
});
