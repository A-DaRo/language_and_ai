const { WorkerProxy, WorkerState } = require('../../../src/cluster/WorkerProxy');
const MockChildProcess = require('../../helpers/MockChildProcess');
const SystemEventBus = require('../../../src/core/SystemEventBus');
const { MESSAGE_TYPES } = require('../../../src/core/ProtocolDefinitions');

describe('WorkerProxy (master-side)', () => {
  let bus;

  beforeEach(() => {
    // Ensure fresh bus for each test
    bus = SystemEventBus.getInstance();
  });

  afterEach(() => {
    SystemEventBus._reset();
  });

  test('handles READY message and transitions to IDLE', (done) => {
    const child = new MockChildProcess();
    const proxy = new WorkerProxy('worker-1', child);

    bus.once('WORKER:READY', (payload) => {
      try {
        expect(payload.workerId).toBe('worker-1');
        expect(proxy.state).toBe(WorkerState.IDLE);
        done();
      } catch (err) {
        done(err);
      }
    });

    // Simulate worker sending READY message
    child.emit('message', { type: MESSAGE_TYPES.READY, pid: 123 });
  });

  test('sendCommand starts task and handles result success', (done) => {
    const child = new MockChildProcess({ shouldCrash: false });
    const proxy = new WorkerProxy('worker-2', child);

    // Wait until ready
    child.emit('message', { type: MESSAGE_TYPES.READY, pid: 222 });

    bus.once('TASK:STARTED', ({ workerId, taskId, taskType }) => {
      try {
        expect(workerId).toBe('worker-2');
        expect(taskType).toBe(MESSAGE_TYPES.DISCOVER);
        // Simulate worker sending a successful RESULT
        setTimeout(() => {
          child.emit('message', {
            type: MESSAGE_TYPES.RESULT,
            taskType: MESSAGE_TYPES.DISCOVER,
            data: {
              success: true,
              pageId: 'p1',
              url: 'https://notion.so/p1'
            }
          });
        }, 10);
      } catch (e) {
        done(e);
      }
    });

    let seenComplete = false;
    bus.once('TASK:COMPLETE', ({ workerId, taskId, taskType, result }) => {
      try {
        expect(workerId).toBe('worker-2');
        expect(taskType).toBe(MESSAGE_TYPES.DISCOVER);
        expect(result).toHaveProperty('success', true);
        seenComplete = true;
        // wait for idle event to assert final state
      } catch (err) {
        done(err);
      }
    });

    bus.once('WORKER:IDLE', ({ workerId }) => {
      try {
        expect(workerId).toBe('worker-2');
        expect(seenComplete).toBe(true);
        expect(proxy.state).toBe(WorkerState.IDLE);
        done();
      } catch (err) {
        done(err);
      }
    });

    proxy.sendCommand(MESSAGE_TYPES.DISCOVER, { url: 'https://notion.so/p1', pageId: 'p1' });
  });

  test('emits WORKER:CRASHED on exit', (done) => {
    const child = new MockChildProcess();
    const proxy = new WorkerProxy('worker-3', child);

    bus.once('WORKER:CRASHED', (payload) => {
      try {
        expect(payload.workerId).toBe('worker-3');
        done();
      } catch (err) { done(err); }
    });

    // Simulate exit
    child.emit('exit', 1, null);
  });
});
