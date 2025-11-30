/**
 * @fileoverview Unit tests for Logger singleton
 * @module tests/unit/core/Logger.test
 */

const Logger = require('../../../src/core/Logger');
const ConsoleStrategy = require('../../../src/core/logger/ConsoleStrategy');
const FileStrategy = require('../../../src/core/logger/FileStrategy');
const DashboardStrategy = require('../../../src/core/logger/DashboardStrategy');

describe('Logger', () => {
  let logger;

  beforeEach(() => {
    // Reset singleton instance before each test
    if (Logger._instance) {
      Logger._instance.close();
      Logger._instance = null;
    }
    logger = Logger.getInstance();
  });

  afterEach(() => {
    if (logger) {
      logger.close();
    }
  });

  describe('getInstance', () => {
    it('should return a singleton instance', () => {
      const instance1 = Logger.getInstance();
      const instance2 = Logger.getInstance();
      expect(instance1).toBe(instance2);
    });

    it('should throw error if trying to instantiate directly', () => {
      expect(() => new Logger()).toThrow('Logger is a singleton');
    });
  });

  describe('init', () => {
    it('should initialize with console strategy when enabled', () => {
      logger.init({ console: true });
      expect(logger.strategies.length).toBe(1);
      expect(logger.strategies[0]).toBeInstanceOf(ConsoleStrategy);
    });

    it('should initialize with file strategy when enabled', () => {
      logger.init({ file: true, outputDir: './test-output' });
      expect(logger.strategies.length).toBe(1);
      expect(logger.strategies[0]).toBeInstanceOf(FileStrategy);
    });

    it('should initialize with both strategies when both enabled', () => {
      logger.init({ console: true, file: true, outputDir: './test-output' });
      expect(logger.strategies.length).toBe(2);
    });

    it('should not re-initialize if already initialized', () => {
      logger.init({ console: true });
      const strategiesCount = logger.strategies.length;
      logger.init({ file: true });
      expect(logger.strategies.length).toBe(strategiesCount);
    });
  });

  describe('addStrategy', () => {
    it('should add a strategy to the strategies array', () => {
      const consoleStrategy = new ConsoleStrategy();
      logger.addStrategy(consoleStrategy);
      expect(logger.strategies).toContain(consoleStrategy);
    });
  });

  describe('removeStrategy', () => {
    it('should remove a strategy from the strategies array', () => {
      const consoleStrategy = new ConsoleStrategy();
      logger.addStrategy(consoleStrategy);
      expect(logger.strategies).toContain(consoleStrategy);
      
      logger.removeStrategy(consoleStrategy);
      expect(logger.strategies).not.toContain(consoleStrategy);
    });
  });

  describe('switchMode', () => {
    it('should remove ConsoleStrategy and add DashboardStrategy when switching to dashboard', () => {
      // Setup: Initialize with console and file strategies
      const consoleStrategy = new ConsoleStrategy();
      const fileStrategy = new FileStrategy('./test-output');
      logger.addStrategy(consoleStrategy);
      logger.addStrategy(fileStrategy);

      expect(logger.strategies.length).toBe(2);
      expect(logger.strategies).toContain(consoleStrategy);
      expect(logger.strategies).toContain(fileStrategy);

      // Mock dashboard instance
      const mockDashboard = {
        updateFooter: jest.fn()
      };

      // Act: Switch to dashboard mode
      logger.switchMode('dashboard', { dashboardInstance: mockDashboard });

      // Assert: ConsoleStrategy removed, FileStrategy kept, DashboardStrategy added
      expect(logger.strategies.length).toBe(2);
      expect(logger.strategies).not.toContain(consoleStrategy);
      expect(logger.strategies).toContain(fileStrategy);
      expect(logger.strategies.some(s => s instanceof DashboardStrategy)).toBe(true);
    });

    it('should remove DashboardStrategy and add ConsoleStrategy when switching to console', () => {
      // Setup: Initialize with dashboard and file strategies
      const fileStrategy = new FileStrategy('./test-output');
      const mockDashboard = { updateFooter: jest.fn() };
      const dashboardStrategy = new DashboardStrategy(mockDashboard);
      
      logger.addStrategy(fileStrategy);
      logger.addStrategy(dashboardStrategy);

      expect(logger.strategies.length).toBe(2);

      // Act: Switch to console mode
      logger.switchMode('console');

      // Assert: DashboardStrategy removed, FileStrategy kept, ConsoleStrategy added
      expect(logger.strategies.length).toBe(2);
      expect(logger.strategies).toContain(fileStrategy);
      expect(logger.strategies.some(s => s instanceof ConsoleStrategy)).toBe(true);
      expect(logger.strategies.some(s => s instanceof DashboardStrategy)).toBe(false);
    });

    it('should preserve FileStrategy when switching modes', () => {
      // Setup
      const fileStrategy = new FileStrategy('./test-output');
      const consoleStrategy = new ConsoleStrategy();
      logger.addStrategy(fileStrategy);
      logger.addStrategy(consoleStrategy);

      const originalFileStrategy = logger.strategies.find(s => s instanceof FileStrategy);

      // Act: Switch to dashboard and back
      const mockDashboard = { updateFooter: jest.fn() };
      logger.switchMode('dashboard', { dashboardInstance: mockDashboard });
      logger.switchMode('console');

      // Assert: Same FileStrategy instance is preserved
      const currentFileStrategy = logger.strategies.find(s => s instanceof FileStrategy);
      expect(currentFileStrategy).toBe(originalFileStrategy);
    });

    it('should handle switching when no UI strategy exists', () => {
      // Setup: Only file strategy
      const fileStrategy = new FileStrategy('./test-output');
      logger.addStrategy(fileStrategy);

      expect(logger.strategies.length).toBe(1);

      // Act: Switch to console (should add console strategy)
      logger.switchMode('console');

      // Assert
      expect(logger.strategies.length).toBe(2);
      expect(logger.strategies.some(s => s instanceof ConsoleStrategy)).toBe(true);
    });
  });

  describe('logging methods', () => {
    it('should dispatch info messages to all strategies', () => {
      const mockStrategy = {
        log: jest.fn()
      };
      logger.addStrategy(mockStrategy);

      logger.info('TEST', 'test message');

      expect(mockStrategy.log).toHaveBeenCalledWith('info', 'TEST', 'test message', undefined);
    });

    it('should dispatch error messages with metadata', () => {
      const mockStrategy = {
        log: jest.fn()
      };
      logger.addStrategy(mockStrategy);

      const error = new Error('test error');
      logger.error('TEST', 'error message', error);

      expect(mockStrategy.log).toHaveBeenCalledWith(
        'error',
        'TEST',
        'error message',
        expect.objectContaining({
          message: 'test error',
          name: 'Error',
          stack: expect.any(String)
        })
      );
    });

    it('should continue dispatching even if one strategy fails', () => {
      const failingStrategy = {
        log: jest.fn(() => { throw new Error('Strategy failed'); })
      };
      const workingStrategy = {
        log: jest.fn()
      };

      logger.addStrategy(failingStrategy);
      logger.addStrategy(workingStrategy);

      // Should not throw
      expect(() => logger.info('TEST', 'message')).not.toThrow();
      expect(workingStrategy.log).toHaveBeenCalled();
    });
  });

  describe('getElapsedTime', () => {
    it('should return elapsed time since initialization', (done) => {
      const startTime = Date.now();
      
      setTimeout(() => {
        const elapsed = logger.getElapsedTime();
        const actualElapsed = Date.now() - startTime;
        
        expect(elapsed).toBeGreaterThanOrEqual(50);
        expect(elapsed).toBeLessThanOrEqual(actualElapsed + 10);
        done();
      }, 50);
    });
  });
});
