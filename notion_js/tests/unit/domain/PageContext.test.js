const PageContext = require('../../../src/domain/PageContext');
const { createMockPageContext } = require('../../helpers/factories');

describe('PageContext', () => {
  test('constructs and serializes to JSON and back', () => {
    const parent = createMockPageContext({ url: 'https://notion.so/parent', title: 'Parent', depth: 0 });
    const child = new PageContext('https://notion.so/child', 'Child Page', 1, parent);
    parent.addChild(child);

    const json = child.toJSON();
    expect(json).toHaveProperty('id');
    expect(json).toHaveProperty('parentId', parent.id);

    const reconstructed = PageContext.fromJSON(json, new Map([[parent.id, parent]]));
    expect(reconstructed.parentId).toBe(parent.id);
    expect(reconstructed.title).toBe(child.title);
  });

  test('getRelativePath and getRelativePathTo produce expected strings', () => {
    const root = createMockPageContext({ url: 'https://notion.so/root', title: 'Root Page', depth: 0 });
    const childA = new PageContext('https://notion.so/a', 'A', 1, root);
    const childB = new PageContext('https://notion.so/b', 'B', 1, root);
    root.addChild(childA);
    root.addChild(childB);

    const relA = childA.getRelativePath();
    const relB = childB.getRelativePath();
    expect(relA).toContain(childA.title);
    expect(relB).toContain(childB.title);

    const relativeFromAtoB = childA.getRelativePathTo(childB);
    expect(relativeFromAtoB).toMatch(/\.\./); // goes up then down
    expect(relativeFromAtoB).toMatch(/index.html$/);
  });

  test('updateTitleFromRegistry updates title with sanitized human-readable name', () => {
    // Create context with raw ID as title
    const context = new PageContext('https://notion.so/29d979eeca9f8102a85be4dd9007f020', '29d979eeca9f8102a85be4dd9007f020', 0);
    
    // Initially, title is the raw ID (sanitized)
    expect(context.title).toContain('29d979eeca9f8102a85be4dd9007f020');
    
    // Update with human-readable title
    context.updateTitleFromRegistry('Introduction to Language and AI');
    
    // Title should now be sanitized human-readable version
    expect(context.title).toBe('Introduction_to_Language_and_AI');
    expect(context.title).not.toContain('29d979eeca9f8102a85be4dd9007f020');
  });

  test('updateTitleFromRegistry handles special characters and spaces', () => {
    const context = new PageContext('https://notion.so/test', 'oldTitle', 0);
    
    // Update with title containing special characters
    context.updateTitleFromRegistry('Week 1: Intro & Overview (Part 1)');
    
    // Should be sanitized to filesystem-safe name
    expect(context.title).toBe('Week_1_Intro_Overview_Part_1');
  });

  test('updateTitleFromRegistry does nothing with null or undefined', () => {
    const context = new PageContext('https://notion.so/test', 'Original Title', 0);
    const originalTitle = context.title;
    
    context.updateTitleFromRegistry(null);
    expect(context.title).toBe(originalTitle);
    
    context.updateTitleFromRegistry(undefined);
    expect(context.title).toBe(originalTitle);
  });
});
