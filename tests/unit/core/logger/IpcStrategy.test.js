/**
 * @fileoverview Unit tests for IpcStrategy
 * @module tests/unit/core/logger/IpcStrategy.test
 */

const IpcStrategy = require('../../../../src/core/logger/IpcStrategy');

describe('IpcStrategy', () => {
  let strategy;
  let originalProcessSend;

  beforeEach(() => {
    strategy = new IpcStrategy();
    // Mock process.send
    originalProcessSend = process.send;
    process.send = jest.fn();
  });

  afterEach(() => {
    // Restore original process.send
    process.send = originalProcessSend;
  });

  describe('log', () => {
    it('should send IPC_LOG message via process.send', () => {
      strategy.log('info', 'TEST', 'test message');

      expect(process.send).toHaveBeenCalledTimes(1);
      expect(process.send).toHaveBeenCalledWith({
        type: 'IPC_LOG',
        payload: expect.objectContaining({
          level: 'info',
          category: 'TEST',
          message: 'test message',
          meta: null,
          timestamp: expect.any(Number)
        })
      });
    });

    it('should include metadata in the message', () => {
      const meta = { key: 'value', count: 42 };
      strategy.log('debug', 'DEBUG', 'debug message', meta);

      expect(process.send).toHaveBeenCalledWith({
        type: 'IPC_LOG',
        payload: expect.objectContaining({
          level: 'debug',
          category: 'DEBUG',
          message: 'debug message',
          meta: { key: 'value', count: 42 }
        })
      });
    });

    it('should handle Error objects in metadata', () => {
      const error = new Error('test error');
      error.code = 'TEST_ERROR';
      
      strategy.log('error', 'ERROR', 'error occurred', error);

      const call = process.send.mock.calls[0][0];
      expect(call.payload.meta).toMatchObject({
        message: 'test error',
        name: 'Error',
        stack: expect.any(String)
      });
    });

    it('should handle circular references in metadata', () => {
      const circular = { a: 1 };
      circular.self = circular;

      // Should not throw
      expect(() => {
        strategy.log('info', 'TEST', 'circular test', circular);
      }).not.toThrow();

      expect(process.send).toHaveBeenCalled();
    });

    it('should not throw if process.send is undefined', () => {
      process.send = undefined;

      expect(() => {
        strategy.log('info', 'TEST', 'message');
      }).not.toThrow();
    });

    it('should include timestamp in the payload', () => {
      const beforeTime = Date.now();
      strategy.log('info', 'TEST', 'message');
      const afterTime = Date.now();

      const call = process.send.mock.calls[0][0];
      expect(call.payload.timestamp).toBeGreaterThanOrEqual(beforeTime);
      expect(call.payload.timestamp).toBeLessThanOrEqual(afterTime);
    });

    it('should handle all log levels', () => {
      const levels = ['info', 'warn', 'error', 'debug', 'success'];

      levels.forEach(level => {
        process.send.mockClear();
        strategy.log(level, 'TEST', `${level} message`);

        expect(process.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'IPC_LOG',
            payload: expect.objectContaining({
              level: level
            })
          })
        );
      });
    });

    it('should serialize metadata as JSON-safe object', () => {
      const meta = {
        string: 'value',
        number: 123,
        boolean: true,
        null: null,
        array: [1, 2, 3],
        nested: { key: 'value' }
      };

      strategy.log('info', 'TEST', 'message', meta);

      const call = process.send.mock.calls[0][0];
      expect(call.payload.meta).toEqual(meta);
    });
  });

  describe('_replacer', () => {
    it('should convert Error objects to plain objects', () => {
      const error = new Error('test error');
      const result = strategy._replacer('error', error);

      expect(result).toEqual({
        message: 'test error',
        name: 'Error',
        stack: expect.any(String)
      });
    });

    it('should pass through non-Error values unchanged', () => {
      expect(strategy._replacer('key', 'string')).toBe('string');
      expect(strategy._replacer('key', 123)).toBe(123);
      expect(strategy._replacer('key', true)).toBe(true);
      expect(strategy._replacer('key', null)).toBe(null);
      expect(strategy._replacer('key', { a: 1 })).toEqual({ a: 1 });
    });
  });
});
