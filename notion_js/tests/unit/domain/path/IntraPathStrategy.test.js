/**
 * @file IntraPathStrategy.test.js
 * @description Unit tests for IntraPathStrategy - same-page anchor resolution
 * 
 * Tests simulate real Notion ToC links and anchor navigation.
 * Verifies correct anchor-only detection for Table of Contents entries.
 */

const IntraPathStrategy = require('../../../../src/domain/path/IntraPathStrategy');
const PageContext = require('../../../../src/domain/PageContext');

describe('IntraPathStrategy', () => {
  let strategy;

  beforeEach(() => {
    strategy = new IntraPathStrategy();
  });

  describe('supports()', () => {
    test('returns true for same page IDs', () => {
      const page = createContext('https://notion.so/29d979eeca9f8102a85be4dd9007f020', 'Test Page', 1);
      
      expect(strategy.supports(page, page)).toBe(true);
    });

    test('returns true for anchor-only hrefs regardless of target context', () => {
      const page = createContext('https://notion.so/page123', 'Test Page', 1);
      
      // Anchor-only href should always be intra-page
      expect(strategy.supports(page, null, '#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d')).toBe(true);
      expect(strategy.supports(page, page, '#section-heading')).toBe(true);
    });

    test('returns false for different pages without anchor-only href', () => {
      const pageA = createContext('https://notion.so/a123', 'Page A', 1);
      const pageB = createContext('https://notion.so/b456', 'Page B', 1);
      
      expect(strategy.supports(pageA, pageB)).toBe(false);
      expect(strategy.supports(pageA, pageB, './other.html')).toBe(false);
    });

    test('returns false for null source context', () => {
      const page = createContext('https://notion.so/page123', 'Test Page', 1);
      
      expect(strategy.supports(null, page)).toBe(false);
      expect(strategy.supports(null, null)).toBe(false);
    });

    test('handles anchor-only hrefs with various formats', () => {
      const page = createContext('https://notion.so/page123', 'Test Page', 1);
      
      // Notion block ID format
      expect(strategy.supports(page, null, '#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d')).toBe(true);
      
      // Compact block ID (no dashes)
      expect(strategy.supports(page, null, '#29d979eeca9f81cfb69df8e08f3ff10d')).toBe(true);
      
      // Simple anchor
      expect(strategy.supports(page, null, '#introduction')).toBe(true);
      
      // Empty anchor
      expect(strategy.supports(page, null, '#')).toBe(true);
    });
  });

  describe('resolve() - Anchor-Only Links', () => {
    /**
     * CRITICAL: This is the core fix for the ToC link issue.
     * ToC entries like "#29d979ee-ca9f-..." should resolve to anchor-only.
     */
    test('resolves anchor-only href to formatted anchor', () => {
      const page = createContext('https://notion.so/29d979eeca9f8102a85be4dd9007f020', 'Lab_Session_1', 1);
      
      // Simulate a ToC entry link
      const result = strategy.resolve(page, null, {
        targetHref: '#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d'
      });
      
      // Should return the anchor (possibly reformatted)
      expect(result).toMatch(/^#/);
      expect(result.length).toBeGreaterThan(1);
    });

    test('preserves anchor-only href if block ID cannot be parsed', () => {
      const page = createContext('https://notion.so/page123', 'Test Page', 1);
      
      const result = strategy.resolve(page, null, {
        targetHref: '#simple-section-name'
      });
      
      expect(result).toBe('#simple-section-name');
    });

    test('handles Notion ToC block ID format', () => {
      const page = createContext('https://notion.so/page123', 'Lab_Session_1', 1);
      
      // Real Notion ToC entry format
      const result = strategy.resolve(page, page, {
        targetHref: '#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d'
      });
      
      // Result should be an anchor containing the block ID
      expect(result).toMatch(/^#/);
      expect(result).toContain('29d979ee');
    });
  });

  describe('resolve() - Block ID from Options', () => {
    test('resolves with blockId option', () => {
      const page = createContext('https://notion.so/page123', 'Test Page', 1);
      
      const result = strategy.resolve(page, page, {
        blockId: '29d979eeca9f81cfb69df8e08f3ff10d'
      });
      
      expect(result).toMatch(/^#/);
      expect(result).toContain('29d979ee');
    });

    test('returns empty string when no blockId and no anchor href', () => {
      const page = createContext('https://notion.so/page123', 'Test Page', 1);
      
      const result = strategy.resolve(page, page, {});
      
      expect(result).toBe('');
    });
  });

  describe('resolve() - Real Notion Examples', () => {
    /**
     * These tests simulate real links from the JBC090 Language & AI site.
     */
    test('Lab_Session_1 ToC entry: Tokenization and POS tagging', () => {
      // Create page context for Lab_Session_1
      const labSession1 = createContext(
        'https://notion.so/Lab-Session-1-29d979eeca9f8102a85be4dd9007f020',
        'Lab_Session_1',
        1
      );
      
      // ToC entry link format from Notion
      const tocHref = '#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d';
      
      // Should be recognized as intra-page
      expect(strategy.supports(labSession1, null, tocHref)).toBe(true);
      
      // Should resolve to anchor
      const result = strategy.resolve(labSession1, labSession1, { targetHref: tocHref });
      expect(result).toMatch(/^#/);
      expect(result).not.toContain('Lab_Session_1');
      expect(result).not.toContain('index.html');
    });

    test('handles multiple ToC entries on same page', () => {
      const page = createContext('https://notion.so/page123', 'Syllabus', 1);
      
      const tocEntries = [
        '#section-1-introduction',
        '#29d979ee-ca9f-8126-a7fd-c2c749c412ea',
        '#learning-objectives',
        '#assessment-details'
      ];
      
      for (const href of tocEntries) {
        expect(strategy.supports(page, null, href)).toBe(true);
        
        const result = strategy.resolve(page, page, { targetHref: href });
        expect(result).toMatch(/^#/);
      }
    });
  });

  describe('_isAnchorOnly()', () => {
    test('identifies anchor-only hrefs correctly', () => {
      expect(strategy._isAnchorOnly('#section')).toBe(true);
      expect(strategy._isAnchorOnly('#')).toBe(true);
      expect(strategy._isAnchorOnly('#29d979ee-ca9f-81cf')).toBe(true);
      
      expect(strategy._isAnchorOnly('./page.html')).toBeFalsy();
      expect(strategy._isAnchorOnly('../other/index.html')).toBeFalsy();
      expect(strategy._isAnchorOnly('https://example.com')).toBeFalsy();
      expect(strategy._isAnchorOnly(null)).toBeFalsy();
      expect(strategy._isAnchorOnly(undefined)).toBeFalsy();
      expect(strategy._isAnchorOnly('')).toBeFalsy();
    });
  });

  describe('_extractBlockIdFromHref()', () => {
    test('extracts block ID from anchor href', () => {
      // UUID format with dashes
      expect(strategy._extractBlockIdFromHref('#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d'))
        .toBe('29d979ee-ca9f-81cf-b69d-f8e08f3ff10d');
      
      // Compact format without dashes
      expect(strategy._extractBlockIdFromHref('#29d979eeca9f81cfb69df8e08f3ff10d'))
        .toBe('29d979eeca9f81cfb69df8e08f3ff10d');
    });

    test('returns null for non-block-ID anchors', () => {
      expect(strategy._extractBlockIdFromHref('#section-name')).toBeNull();
      expect(strategy._extractBlockIdFromHref('#intro')).toBeNull();
      expect(strategy._extractBlockIdFromHref('#123')).toBeNull();
    });

    test('handles edge cases', () => {
      expect(strategy._extractBlockIdFromHref(null)).toBeNull();
      expect(strategy._extractBlockIdFromHref(undefined)).toBeNull();
      expect(strategy._extractBlockIdFromHref('')).toBeNull();
      expect(strategy._extractBlockIdFromHref('#')).toBeNull();
    });
  });

  describe('getType()', () => {
    test('returns intra type', () => {
      expect(strategy.getType()).toBe('intra');
    });
  });
});

/**
 * Helper to create PageContext.
 */
function createContext(url, title, depth, parentContext = null) {
  return new PageContext(
    url,
    title,
    depth,
    parentContext,
    parentContext?.id || null
  );
}
