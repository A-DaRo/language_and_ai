const FileSystemUtils = require('../../../src/utils/FileSystemUtils');

describe('FileSystemUtils.sanitizeFilename', () => {
  test('removes illegal Windows characters and control chars', () => {
    const dangerous = 'Page<>:"/\\|?*\x00Name';
    const safe = FileSystemUtils.sanitizeFilename(dangerous);
    expect(safe).not.toMatch(/[<>:"/\\|?*\x00-\x1F]/);
    expect(safe.length).toBeGreaterThan(0);
  });

  test('replaces spaces with underscores and trims dots/underscores', () => {
    const input = '  . My Page Name . ';
    const out = FileSystemUtils.sanitizeFilename(input);
    expect(out).toMatch(/^[A-Za-z0-9_\-\.]+$/);
    expect(out).not.toMatch(/^\.|_$/);
  });

  test('handles unicode and long names by hashing', () => {
    const long = '页'.repeat(500) + '.html';
    const out = FileSystemUtils.sanitizeFilename(long);
    expect(out.length).toBeLessThanOrEqual(160);
    expect(out).not.toContain('\n');
  });

  test('empty or falsy returns untitled/asset fallback', () => {
    expect(FileSystemUtils.sanitizeFilename('')).toBe('untitled');
    expect(FileSystemUtils.sanitizeFilename(null)).toBe('untitled');
  });
});
