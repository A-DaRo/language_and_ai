/**
 * @fileoverview Unit tests for DashboardController
 * @module tests/unit/ui/DashboardController.test
 */

const DashboardController = require('../../../src/ui/DashboardController');
const EventEmitter = require('events');

// Mock TerminalDashboard
jest.mock('../../../src/ui/TerminalDashboard', () => {
  return jest.fn().mockImplementation((numWorkers) => {
    return {
      setMode: jest.fn(),
      updateHeader: jest.fn(),
      updateDiscoveryStats: jest.fn(),
      updateDownloadStats: jest.fn(),
      updateWorkerStatus: jest.fn(),
      updateFooter: jest.fn(),
      stop: jest.fn(),
      _numWorkers: numWorkers
    };
  });
});

describe('DashboardController', () => {
  let controller;
  let mockEventBus;
  let mockBrowserManager;
  const TerminalDashboard = require('../../../src/ui/TerminalDashboard');

  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();

    // Create mock event bus
    mockEventBus = new EventEmitter();

    // Create mock browser manager
    mockBrowserManager = {
      getAllWorkerIds: jest.fn(() => ['worker-1', 'worker-2', 'worker-3'])
    };

    // Create controller
    controller = new DashboardController(mockEventBus, mockBrowserManager);
  });

  afterEach(() => {
    if (controller.timerInterval) {
      clearInterval(controller.timerInterval);
    }
  });

  describe('start', () => {
    it('should initialize dashboard with correct worker count', () => {
      const TerminalDashboard = require('../../../src/ui/TerminalDashboard');
      controller.start();

      expect(TerminalDashboard).toHaveBeenCalledWith(3);
      expect(mockBrowserManager.getAllWorkerIds).toHaveBeenCalled();
      expect(controller.workerSlotMap.size).toBe(3);
      expect(controller.workerSlotMap.get('worker-1')).toBe(0);
      expect(controller.workerSlotMap.get('worker-2')).toBe(1);
      expect(controller.workerSlotMap.get('worker-3')).toBe(2);
    });

    it('should start elapsed time timer', (done) => {
      controller.start();

      expect(controller.timerInterval).toBeDefined();
      
      // Wait for timer to tick at least once
      setTimeout(() => {
        expect(controller.dashboard.updateHeader).toHaveBeenCalled();
        done();
      }, 1100);
    });
  });

  describe('event handling', () => {
    beforeEach(() => {
      controller.start();
      // Clear any calls from initialization
      jest.clearAllMocks();
    });

    it('should handle PHASE:CHANGED event', () => {
      const eventData = {
        phase: 'discovery',
        data: { total: 100 }
      };

      mockEventBus.emit('PHASE:CHANGED', eventData);

      expect(controller.dashboard.setMode).toHaveBeenCalledWith('discovery', { total: 100 });
      expect(controller.dashboard.updateHeader).toHaveBeenCalledWith(
        expect.stringContaining('Discovery Phase')
      );
    });

    it('should handle DISCOVERY:PROGRESS event', () => {
      const stats = {
        pagesFound: 42,
        inQueue: 10,
        conflicts: 3,
        currentDepth: 2
      };

      mockEventBus.emit('DISCOVERY:PROGRESS', stats);

      expect(controller.dashboard.updateDiscoveryStats).toHaveBeenCalledWith(stats);
    });

    it('should handle EXECUTION:PROGRESS event', () => {
      const stats = {
        pending: 50,
        active: 3,
        completed: 47,
        total: 100,
        failed: 0
      };

      mockEventBus.emit('EXECUTION:PROGRESS', stats);

      expect(controller.dashboard.updateDownloadStats).toHaveBeenCalledWith(stats);
    });

    it('should handle WORKER:BUSY event', () => {
      const eventData = {
        workerId: 'worker-2',
        task: { description: 'Downloading page X' }
      };

      mockEventBus.emit('WORKER:BUSY', eventData);

      expect(controller.dashboard.updateWorkerStatus).toHaveBeenCalledWith(
        1, // worker-2 maps to slot 1
        '[BUSY] Worker 2: Downloading page X [PagesAssigned: 1]'
      );
    });

    it('should handle WORKER:IDLE event', () => {
      mockEventBus.emit('WORKER:IDLE', { workerId: 'worker-1' });

      expect(controller.dashboard.updateWorkerStatus).toHaveBeenCalledWith(
        0, // worker-1 maps to slot 0
        '[IDLE] Worker 1: Waiting for task...'
      );
    });

    it('should handle WORKER:READY event', () => {
      mockEventBus.emit('WORKER:READY', { workerId: 'worker-3' });

      expect(controller.dashboard.updateWorkerStatus).toHaveBeenCalledWith(
        2, // worker-3 maps to slot 2
        '[IDLE] Worker 3: Waiting for task...'
      );
    });

    it('should ignore events for unknown worker IDs', () => {
      mockEventBus.emit('WORKER:BUSY', {
        workerId: 'unknown-worker',
        task: { description: 'test' }
      });

      expect(controller.dashboard.updateWorkerStatus).not.toHaveBeenCalled();
    });

    it('should handle multiple worker status changes', () => {
      // Worker 1 becomes busy
      mockEventBus.emit('WORKER:BUSY', {
        workerId: 'worker-1',
        task: { description: 'Task A' }
      });

      expect(controller.dashboard.updateWorkerStatus).toHaveBeenCalledWith(0, '[BUSY] Worker 1: Task A [PagesAssigned: 1]');

      // Worker 2 becomes busy
      mockEventBus.emit('WORKER:BUSY', {
        workerId: 'worker-2',
        task: { description: 'Task B' }
      });

      expect(controller.dashboard.updateWorkerStatus).toHaveBeenCalledWith(1, '[BUSY] Worker 2: Task B [PagesAssigned: 1]');

      // Worker 1 becomes idle
      mockEventBus.emit('WORKER:IDLE', { workerId: 'worker-1' });

      expect(controller.dashboard.updateWorkerStatus).toHaveBeenCalledWith(
        0,
        '[IDLE] Worker 1: Waiting for task...'
      );
    });
  });

  describe('getDashboard', () => {
    it('should return the dashboard instance', () => {
      controller.start();
      const dashboard = controller.getDashboard();
      expect(dashboard).toBe(controller.dashboard);
    });
  });

  describe('stop', () => {
    it('should stop the dashboard and clear timer', () => {
      controller.start();
      
      expect(controller.timerInterval).toBeDefined();

      controller.stop();

      expect(controller.timerInterval).toBeNull();
      expect(controller.dashboard.stop).toHaveBeenCalled();
    });

    it('should handle stop when dashboard is not initialized', () => {
      controller.dashboard = null;
      
      expect(() => controller.stop()).not.toThrow();
    });
  });

  describe('elapsed time formatting', () => {
    it('should format elapsed time correctly', (done) => {
      controller.startTime = Date.now() - 125000; // 2 minutes 5 seconds ago
      controller.start();

      setTimeout(() => {
        const headerCalls = controller.dashboard.updateHeader.mock.calls;
        const lastCall = headerCalls[headerCalls.length - 1][0];
        
        // Format is "Xm Ys" (e.g., "2m 5s")
        expect(lastCall).toMatch(/Elapsed: 2m [5-6]s/);
        done();
      }, 1100);
    });

    it('should pad seconds with leading zero', (done) => {
      controller.startTime = Date.now() - 5000; // 5 seconds ago
      controller.start();

      setTimeout(() => {
        const headerCalls = controller.dashboard.updateHeader.mock.calls;
        const lastCall = headerCalls[headerCalls.length - 1][0];
        
        // Format is "Xm Ys" (e.g., "0m 5s")
        expect(lastCall).toMatch(/Elapsed: 0m [5-6]s/);
        done();
      }, 1100);
    });
  });
});
