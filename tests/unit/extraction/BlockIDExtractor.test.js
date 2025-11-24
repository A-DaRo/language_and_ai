/**
 * @fileoverview Unit tests for BlockIDExtractor
 * @module tests/unit/extraction/BlockIDExtractor.test.js
 */

const BlockIDExtractor = require('../../../src/extraction/BlockIDExtractor');
const { JSDOM } = require('jsdom');

describe('BlockIDExtractor', () => {
  let extractor;

  beforeEach(() => {
    extractor = new BlockIDExtractor();
  });

  describe('extractBlockIDs', () => {
    it('should extract block IDs from HTML elements', () => {
      const html = `
        <html>
          <body>
            <div data-block-id="29d979ee-ca9f-81f7-b82f-e4b983834212">Block 1</div>
            <div data-block-id="12345678-1234-5678-1234-567812345678">Block 2</div>
          </body>
        </html>
      `;

      const dom = new JSDOM(html);
      const document = dom.window.document;

      const blockMap = extractor.extractBlockIDs(document);

      expect(blockMap.size).toBe(2);
      expect(blockMap.get('29d979eeca9f81f7b82fe4b983834212')).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
      expect(blockMap.get('12345678123456781234567812345678')).toBe('12345678-1234-5678-1234-567812345678');
    });

    it('should return empty map if no data-block-id elements exist', () => {
      const html = '<html><body><div>No block ID</div></body></html>';
      const dom = new JSDOM(html);
      const document = dom.window.document;

      const blockMap = extractor.extractBlockIDs(document);

      expect(blockMap.size).toBe(0);
    });

    it('should handle elements with empty data-block-id', () => {
      const html = `
        <html>
          <body>
            <div data-block-id="">Empty</div>
            <div data-block-id="29d979ee-ca9f-81f7-b82f-e4b983834212">Valid</div>
          </body>
        </html>
      `;

      const dom = new JSDOM(html);
      const document = dom.window.document;

      const blockMap = extractor.extractBlockIDs(document);

      expect(blockMap.size).toBe(1);
    });

    it('should handle null document gracefully', () => {
      const blockMap = extractor.extractBlockIDs(null);
      expect(blockMap).toEqual(new Map());
    });
  });

  describe('formatToRaw conversion', () => {
    it('should convert formatted UUID to raw hex', () => {
      const formatted = '29d979ee-ca9f-81f7-b82f-e4b983834212';
      const raw = extractor._formatToRaw(formatted);

      expect(raw).toBe('29d979eeca9f81f7b82fe4b983834212');
    });

    it('should handle uppercase input', () => {
      const formatted = '29D979EE-CA9F-81F7-B82F-E4B983834212';
      const raw = extractor._formatToRaw(formatted);

      expect(raw).toBe('29d979eeca9f81f7b82fe4b983834212');
    });

    it('should handle empty string', () => {
      expect(extractor._formatToRaw('')).toBe('');
      expect(extractor._formatToRaw(null)).toBe('');
    });
  });

  describe('rawToFormatted conversion', () => {
    it('should convert raw hex to formatted UUID', () => {
      const raw = '29d979eeca9f81f7b82fe4b983834212';
      const formatted = extractor._rawToFormatted(raw);

      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should handle uppercase input', () => {
      const raw = '29D979EECA9F81F7B82FE4B983834212';
      const formatted = extractor._rawToFormatted(raw);

      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should handle short input gracefully', () => {
      expect(extractor._rawToFormatted('short')).toBe('short');
      expect(extractor._rawToFormatted('')).toBe('');
    });
  });

  describe('Validation methods', () => {
    it('should validate raw block ID format', () => {
      expect(extractor.isValidRawId('29d979eeca9f81f7b82fe4b983834212')).toBe(true);
      expect(extractor.isValidRawId('29D979EECA9F81F7B82FE4B983834212')).toBe(true);
      expect(extractor.isValidRawId('not-a-raw-id')).toBe(false);
      expect(extractor.isValidRawId('')).toBe(false);
      expect(extractor.isValidRawId(null)).toBe(false);
    });

    it('should validate formatted block ID format', () => {
      expect(extractor.isValidFormattedId('29d979ee-ca9f-81f7-b82f-e4b983834212')).toBe(true);
      expect(extractor.isValidFormattedId('29D979EE-CA9F-81F7-B82F-E4B983834212')).toBe(true);
      expect(extractor.isValidFormattedId('29d979eeca9f81f7b82fe4b983834212')).toBe(false);
      expect(extractor.isValidFormattedId('')).toBe(false);
      expect(extractor.isValidFormattedId(null)).toBe(false);
    });
  });

  describe('Round-trip conversion', () => {
    it('should convert raw to formatted and back correctly', () => {
      const originalRaw = '29d979eeca9f81f7b82fe4b983834212';
      const formatted = extractor._rawToFormatted(originalRaw);
      const backToRaw = extractor._formatToRaw(formatted);

      expect(backToRaw).toBe(originalRaw);
    });

    it('should convert formatted to raw and back correctly', () => {
      const originalFormatted = '29d979ee-ca9f-81f7-b82f-e4b983834212';
      const raw = extractor._formatToRaw(originalFormatted);
      const backToFormatted = extractor._rawToFormatted(raw);

      expect(backToFormatted).toBe(originalFormatted);
    });
  });
});
