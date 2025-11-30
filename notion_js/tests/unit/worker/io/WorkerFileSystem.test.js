const WorkerFileSystem = require('../../../../src/worker/io/WorkerFileSystem');
const path = require('path');

const mockLogger = {
  success: jest.fn(),
  error: jest.fn(),
  debug: jest.fn()
};

// Mock fs/promises
jest.mock('fs/promises');
const fs = require('fs/promises');

describe('WorkerFileSystem', () => {
  let wfs;

  beforeEach(() => {
    wfs = new WorkerFileSystem(mockLogger);
    jest.clearAllMocks();
    fs.writeFile = jest.fn().mockResolvedValue();
    fs.mkdir = jest.fn().mockResolvedValue();
  });

  test('throws error on relative paths (critical safety check)', async () => {
    await expect(wfs.safeWrite('relative/path.html', 'content'))
      .rejects.toThrow('requires absolute path');
    
    await expect(wfs.safeWrite('./relative/path.html', 'content'))
      .rejects.toThrow('requires absolute path');
    
    expect(fs.writeFile).not.toHaveBeenCalled();
  });

  test('allows absolute paths and creates parent directories', async () => {
    const absPath = path.resolve('/tmp/test/nested/file.html');
    await wfs.safeWrite(absPath, '<html>test</html>');
    
    expect(fs.mkdir).toHaveBeenCalledWith(
      path.dirname(absPath),
      { recursive: true }
    );
    expect(fs.writeFile).toHaveBeenCalledWith(
      absPath,
      '<html>test</html>',
      { encoding: 'utf-8' }
    );
    expect(mockLogger.success).toHaveBeenCalledWith(
      'FS',
      expect.stringContaining('Wrote')
    );
  });

  test('tracks written files in session', async () => {
    const path1 = path.resolve('/tmp/file1.html');
    const path2 = path.resolve('/tmp/file2.html');
    
    await wfs.safeWrite(path1, 'content1');
    await wfs.safeWrite(path2, 'content2');
    
    const stats = wfs.getStats();
    expect(stats.filesWritten).toBe(2);
    expect(stats.writtenPaths).toContain(path1);
    expect(stats.writtenPaths).toContain(path2);
  });

  test('calculates content size for logging', async () => {
    const absPath = path.resolve('/tmp/test.html');
    const content = '<html>test</html>';
    
    await wfs.safeWrite(absPath, content);
    
    expect(mockLogger.success).toHaveBeenCalledWith(
      'FS',
      expect.stringMatching(/Wrote \d+ bytes/)
    );
  });

  test('propagates write errors with context', async () => {
    const absPath = path.resolve('/tmp/test.html');
    fs.writeFile.mockRejectedValue(new Error('Disk full'));
    
    await expect(wfs.safeWrite(absPath, 'content'))
      .rejects.toThrow('Disk full');
    
    expect(mockLogger.error).toHaveBeenCalledWith(
      'FS',
      expect.stringContaining('Failed to write'),
      expect.any(Error)
    );
  });
});
