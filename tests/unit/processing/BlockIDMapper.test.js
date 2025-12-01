/**
 * @fileoverview Unit tests for BlockIDMapper
 * @module tests/unit/processing/BlockIDMapper.test.js
 */

const BlockIDMapper = require('../../../src/processing/BlockIDMapper');
const fs = require('fs/promises');
const path = require('path');
const os = require('os');

describe('BlockIDMapper', () => {
  let mapper;
  let tempDir;

  beforeEach(async () => {
    mapper = new BlockIDMapper();
    // Create temp directory for tests
    tempDir = path.join(os.tmpdir(), `test-block-mapper-${Date.now()}`);
    try {
      await fs.mkdir(tempDir, { recursive: true });
    } catch (e) {
      // Directory may already exist
    }
  });

  afterEach(async () => {
    // Clean up temp directory
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (e) {
      // Ignore cleanup errors
    }
  });

  describe('saveBlockMap and loadBlockMap', () => {
    it('should save and load block map', async () => {
      const blockMap = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212'],
        ['12345678123456781234567812345678', '12345678-1234-5678-1234-567812345678']
      ]);

      await mapper.saveBlockMap('test-page', tempDir, blockMap);

      const loaded = await mapper.loadBlockMap(tempDir);

      expect(loaded.size).toBe(2);
      expect(loaded.get('29d979eeca9f81f7b82fe4b983834212')).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should return empty map if file does not exist', async () => {
      const loaded = await mapper.loadBlockMap(tempDir);

      expect(loaded).toEqual(new Map());
    });

    it('should handle plain object input to saveBlockMap', async () => {
      const blockObj = {
        '29d979eeca9f81f7b82fe4b983834212': '29d979ee-ca9f-81f7-b82f-e4b983834212'
      };

      await mapper.saveBlockMap('test-page', tempDir, blockObj);

      const loaded = await mapper.loadBlockMap(tempDir);

      expect(loaded.get('29d979eeca9f81f7b82fe4b983834212')).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });
  });

  describe('getFormattedId', () => {
    it('should get formatted ID from block map', () => {
      const blockMap = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212']
      ]);

      const formatted = mapper.getFormattedId('29d979eeca9f81f7b82fe4b983834212', blockMap);

      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should fallback to formatting if ID not in map', () => {
      const blockMap = new Map();

      const formatted = mapper.getFormattedId('29d979eeca9f81f7b82fe4b983834212', blockMap);

      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should handle null raw ID', () => {
      const blockMap = new Map();

      const formatted = mapper.getFormattedId(null, blockMap);

      expect(formatted).toBe('');
    });

    it('should handle null block map', () => {
      const formatted = mapper.getFormattedId('29d979eeca9f81f7b82fe4b983834212', null);

      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });
  });

  describe('mergeBlockMaps', () => {
    it('should merge multiple block maps', () => {
      const map1 = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212']
      ]);

      const map2 = new Map([
        ['12345678123456781234567812345678', '12345678-1234-5678-1234-567812345678']
      ]);

      const merged = mapper.mergeBlockMaps([map1, map2]);

      expect(merged.size).toBe(2);
      expect(merged.get('29d979eeca9f81f7b82fe4b983834212')).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
      expect(merged.get('12345678123456781234567812345678')).toBe('12345678-1234-5678-1234-567812345678');
    });

    it('should handle overlapping keys (later map wins)', () => {
      const map1 = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212']
      ]);

      const map2 = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834299']
      ]);

      const merged = mapper.mergeBlockMaps([map1, map2]);

      // Map2's value should override Map1's
      expect(merged.get('29d979eeca9f81f7b82fe4b983834212')).toBe('29d979ee-ca9f-81f7-b82f-e4b983834299');
    });

    it('should handle empty array', () => {
      const merged = mapper.mergeBlockMaps([]);

      expect(merged.size).toBe(0);
    });
  });

  describe('getMapFileName', () => {
    it('should return correct filename', () => {
      expect(mapper.getMapFileName()).toBe('.block-ids.json');
    });
  });

  describe('Fallback formatting', () => {
    it('should apply fallback formatting to short IDs', () => {
      const formatted = mapper.getFormattedId('short', null);

      // Should return as-is since it's too short
      expect(formatted).toBe('short');
    });

    it('should apply fallback formatting to valid-length IDs', () => {
      const raw = '29d979eeca9f81f7b82fe4b983834212';
      const formatted = mapper.getFormattedId(raw, null);

      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });
  });

  describe('Integration with JSDOM-like contexts', () => {
    it('should handle loadAllBlockMaps with empty context map', async () => {
      const contextMap = new Map();

      const cache = await mapper.loadAllBlockMaps(contextMap);

      expect(cache.size).toBe(0);
    });
  });

  describe('Error handling', () => {
    it('should not throw when saving to invalid path', async () => {
      // Attempt to save to non-existent parent directory
      // This should be handled gracefully by the implementation
      const result = await mapper.saveBlockMap('test', '/nonexistent/path/that/does/not/exist', new Map());
      
      // Should not throw, just handle gracefully
      expect(result).toBeUndefined();
    });
  });
});
