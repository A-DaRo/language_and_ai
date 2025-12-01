/**
 * @fileoverview Unit Tests for PathResolver and PathResolverFactory
 * @module tests/unit/domain/path/PathResolver.test
 * @description Tests the unified path resolution system.
 */

const PathResolver = require('../../../../src/domain/path/PathResolver');
const PathResolverFactory = require('../../../../src/domain/path/PathResolverFactory');
const {
  IntraPageResolver,
  InterPageResolver,
  ExternalUrlResolver,
  FilesystemResolver
} = require('../../../../src/domain/path/resolvers');

// Mock PageContext for testing
function createMockContext(id, title, depth = 0, pathSegments = []) {
  return {
    id,
    title,
    depth,
    pathSegments,
    parentContext: null,
    getPathSegments: () => pathSegments
  };
}

describe('PathResolver', () => {
  describe('Types enumeration', () => {
    it('should define all required path types', () => {
      expect(PathResolver.Types.INTRA).toBe('intra');
      expect(PathResolver.Types.INTER).toBe('inter');
      expect(PathResolver.Types.EXTERNAL).toBe('external');
      expect(PathResolver.Types.FILESYSTEM).toBe('filesystem');
    });

    it('should be frozen (immutable)', () => {
      expect(Object.isFrozen(PathResolver.Types)).toBe(true);
    });
  });

  describe('Abstract methods', () => {
    it('should throw when supports() is not implemented', () => {
      const resolver = new PathResolver();
      expect(() => resolver.supports({})).toThrow('must be implemented');
    });

    it('should throw when resolve() is not implemented', () => {
      const resolver = new PathResolver();
      expect(() => resolver.resolve({})).toThrow('must be implemented');
    });

    it('should throw when getType() is not implemented', () => {
      const resolver = new PathResolver();
      expect(() => resolver.getType()).toThrow('must be implemented');
    });

    it('should return class name for getName()', () => {
      const resolver = new PathResolver();
      expect(resolver.getName()).toBe('PathResolver');
    });
  });
});

describe('IntraPageResolver', () => {
  let resolver;

  beforeEach(() => {
    resolver = new IntraPageResolver();
  });

  describe('supports()', () => {
    it('should return true for anchor-only hrefs', () => {
      const context = {
        source: createMockContext('abc123', 'Page'),
        href: '#block-id-123'
      };
      expect(resolver.supports(context)).toBe(true);
    });

    it('should return true for same-page links', () => {
      const page = createMockContext('abc123', 'Page');
      const context = { source: page, target: page };
      expect(resolver.supports(context)).toBe(true);
    });

    it('should return false for different pages', () => {
      const context = {
        source: createMockContext('abc123', 'Page A'),
        target: createMockContext('def456', 'Page B')
      };
      expect(resolver.supports(context)).toBe(false);
    });

    it('should return false for null source', () => {
      const context = { source: null, target: createMockContext('abc', 'Page') };
      expect(resolver.supports(context)).toBe(false);
    });
  });

  describe('resolve()', () => {
    it('should return anchor-only for same-page with blockId', () => {
      const page = createMockContext('abc123', 'Page');
      const context = { source: page, target: page, blockId: '29d979eeca9f' };
      const result = resolver.resolve(context);
      expect(result).toMatch(/^#/);
      expect(result).toContain('29d979ee');
    });

    it('should return empty string for same-page without blockId', () => {
      const page = createMockContext('abc123', 'Page');
      const context = { source: page, target: page };
      expect(resolver.resolve(context)).toBe('');
    });

    it('should preserve anchor-only hrefs', () => {
      const page = createMockContext('abc123', 'Page');
      const context = { source: page, href: '#my-anchor' };
      expect(resolver.resolve(context)).toBe('#my-anchor');
    });
  });

  describe('getType()', () => {
    it('should return INTRA type', () => {
      expect(resolver.getType()).toBe(PathResolver.Types.INTRA);
    });
  });
});

describe('InterPageResolver', () => {
  let resolver;

  beforeEach(() => {
    resolver = new InterPageResolver();
  });

  describe('supports()', () => {
    it('should return true for different internal pages', () => {
      const context = {
        source: createMockContext('abc123', 'Page A'),
        target: createMockContext('def456', 'Page B')
      };
      expect(resolver.supports(context)).toBe(true);
    });

    it('should return false for same page', () => {
      const page = createMockContext('abc123', 'Page');
      const context = { source: page, target: page };
      expect(resolver.supports(context)).toBe(false);
    });

    it('should return false for null target', () => {
      const context = { source: createMockContext('abc', 'Page'), target: null };
      expect(resolver.supports(context)).toBe(false);
    });
  });

  describe('resolve()', () => {
    it('should compute sibling path (same depth)', () => {
      const context = {
        source: createMockContext('abc', 'PageA', 1, ['PageA']),
        target: createMockContext('def', 'PageB', 1, ['PageB'])
      };
      expect(resolver.resolve(context)).toBe('../PageB/index.html');
    });

    it('should compute path from child to root', () => {
      const context = {
        source: createMockContext('child', 'Child', 1, ['Child']),
        target: createMockContext('root', 'Root', 0, [])
      };
      expect(resolver.resolve(context)).toBe('../index.html');
    });

    it('should compute path from grandchild to sibling', () => {
      const context = {
        source: createMockContext('gc', 'Grandchild', 2, ['Section', 'Page']),
        target: createMockContext('other', 'Other', 1, ['Other'])
      };
      expect(resolver.resolve(context)).toBe('../../Other/index.html');
    });

    it('should append block anchor when provided', () => {
      const context = {
        source: createMockContext('abc', 'PageA', 1, ['PageA']),
        target: createMockContext('def', 'PageB', 1, ['PageB']),
        blockId: '29d979ee'
      };
      const result = resolver.resolve(context);
      expect(result).toMatch(/^\.\.\/PageB\/index\.html#/);
    });
  });

  describe('getType()', () => {
    it('should return INTER type', () => {
      expect(resolver.getType()).toBe(PathResolver.Types.INTER);
    });
  });
});

describe('ExternalUrlResolver', () => {
  let resolver;

  beforeEach(() => {
    resolver = new ExternalUrlResolver();
  });

  describe('supports()', () => {
    it('should return true for null target', () => {
      const context = { source: createMockContext('abc', 'Page'), target: null };
      expect(resolver.supports(context)).toBe(true);
    });

    it('should return true for target without ID', () => {
      const context = { target: { title: 'External' } };
      expect(resolver.supports(context)).toBe(true);
    });

    it('should return false for valid internal target', () => {
      const context = { target: createMockContext('abc', 'Page') };
      expect(resolver.supports(context)).toBe(false);
    });
  });

  describe('resolve()', () => {
    it('should return originalUrl unchanged', () => {
      const context = { originalUrl: 'https://example.com/page' };
      expect(resolver.resolve(context)).toBe('https://example.com/page');
    });

    it('should return href as fallback', () => {
      const context = { href: 'https://fallback.com' };
      expect(resolver.resolve(context)).toBe('https://fallback.com');
    });

    it('should return null if no URL provided', () => {
      expect(resolver.resolve({})).toBeNull();
    });
  });

  describe('getType()', () => {
    it('should return EXTERNAL type', () => {
      expect(resolver.getType()).toBe(PathResolver.Types.EXTERNAL);
    });
  });
});

describe('FilesystemResolver', () => {
  let resolver;

  beforeEach(() => {
    resolver = new FilesystemResolver();
  });

  describe('supports()', () => {
    it('should return true for valid source context', () => {
      const context = { source: createMockContext('abc', 'Page') };
      expect(resolver.supports(context)).toBe(true);
    });

    it('should return false for null source', () => {
      const context = { source: null };
      expect(resolver.supports(context)).toBe(false);
    });

    it('should return false for source without ID', () => {
      const context = { source: { title: 'No ID' } };
      // Returns falsy (undefined) when source has no id
      expect(resolver.supports(context)).toBeFalsy();
    });
  });

  describe('resolve()', () => {
    it('should return index.html for root page', () => {
      const context = { source: createMockContext('root', 'Root', 0, []) };
      expect(resolver.resolve(context)).toBe('index.html');
    });

    it('should return correct path for child page', () => {
      const context = { source: createMockContext('child', 'Child', 1, ['Child']) };
      expect(resolver.resolve(context)).toBe('Child/index.html');
    });

    it('should return correct path for grandchild page', () => {
      const context = { 
        source: createMockContext('gc', 'Grandchild', 2, ['Section', 'Page']) 
      };
      expect(resolver.resolve(context)).toBe('Section/Page/index.html');
    });
  });

  describe('resolveDirectory()', () => {
    it('should return empty string for root page', () => {
      const context = { source: createMockContext('root', 'Root', 0, []) };
      expect(resolver.resolveDirectory(context)).toBe('');
    });

    it('should return directory path without index.html', () => {
      const context = { 
        source: createMockContext('gc', 'Grandchild', 2, ['Section', 'Page']) 
      };
      expect(resolver.resolveDirectory(context)).toBe('Section/Page');
    });
  });

  describe('getType()', () => {
    it('should return FILESYSTEM type', () => {
      expect(resolver.getType()).toBe(PathResolver.Types.FILESYSTEM);
    });
  });
});

describe('PathResolverFactory', () => {
  let factory;

  beforeEach(() => {
    factory = new PathResolverFactory();
  });

  describe('constructor', () => {
    it('should register default resolvers', () => {
      const names = factory.getRegisteredResolverNames();
      expect(names).toContain('IntraPageResolver');
      expect(names).toContain('InterPageResolver');
      expect(names).toContain('ExternalUrlResolver');
    });

    it('should create filesystem resolver separately', () => {
      expect(factory.getFilesystemResolver()).toBeInstanceOf(FilesystemResolver);
    });
  });

  describe('resolve()', () => {
    it('should select IntraPageResolver for same-page links', () => {
      const page = createMockContext('abc', 'Page');
      const path = factory.resolve({ source: page, target: page, blockId: '123' });
      expect(path).toMatch(/^#/);
    });

    it('should select InterPageResolver for cross-page links', () => {
      const context = {
        source: createMockContext('abc', 'PageA', 1, ['PageA']),
        target: createMockContext('def', 'PageB', 1, ['PageB'])
      };
      const path = factory.resolve(context);
      expect(path).toBe('../PageB/index.html');
    });

    it('should select ExternalUrlResolver for external links', () => {
      const context = {
        source: createMockContext('abc', 'Page'),
        target: null,
        originalUrl: 'https://example.com'
      };
      const path = factory.resolve(context);
      expect(path).toBe('https://example.com');
    });
  });

  describe('resolveAs()', () => {
    it('should explicitly invoke filesystem resolver', () => {
      const context = { 
        source: createMockContext('child', 'Child', 1, ['Child']) 
      };
      const path = factory.resolveAs('filesystem', context);
      expect(path).toBe('Child/index.html');
    });

    it('should explicitly invoke intra resolver', () => {
      const page = createMockContext('abc', 'Page');
      const path = factory.resolveAs('intra', { source: page, target: page, blockId: 'xyz' });
      expect(path).toMatch(/^#/);
    });
  });

  describe('getPathType()', () => {
    it('should return intra for same-page', () => {
      const page = createMockContext('abc', 'Page');
      expect(factory.getPathType({ source: page, target: page })).toBe('intra');
    });

    it('should return inter for different pages', () => {
      const context = {
        source: createMockContext('abc', 'A'),
        target: createMockContext('def', 'B')
      };
      expect(factory.getPathType(context)).toBe('inter');
    });

    it('should return external for null target', () => {
      const context = { source: createMockContext('abc', 'Page'), target: null };
      expect(factory.getPathType(context)).toBe('external');
    });
  });

  describe('withResolvers() static factory', () => {
    it('should create factory with custom resolvers only', () => {
      const customFactory = PathResolverFactory.withResolvers([
        new IntraPageResolver()
      ]);
      const names = customFactory.getRegisteredResolverNames();
      expect(names).toEqual(['IntraPageResolver']);
    });
  });
});
