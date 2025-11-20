const ScrapingPipeline = require('../../../../src/worker/pipeline/ScrapingPipeline');
const PipelineStep = require('../../../../src/worker/pipeline/PipelineStep');

const mockLogger = {
  info: jest.fn(),
  success: jest.fn(),
  error: jest.fn(),
  debug: jest.fn()
};

describe('ScrapingPipeline', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('executes steps in order and passes context through', async () => {
    const executionOrder = [];
    
    const step1 = new PipelineStep('Step1');
    step1.process = jest.fn(async (ctx) => {
      executionOrder.push('step1');
      ctx.step1Result = 'data1';
    });
    
    const step2 = new PipelineStep('Step2');
    step2.process = jest.fn(async (ctx) => {
      executionOrder.push('step2');
      ctx.step2Result = 'data2';
      expect(ctx.step1Result).toBe('data1'); // Can access previous step data
    });
    
    const pipeline = new ScrapingPipeline([step1, step2], mockLogger);
    const context = { initialData: 'test' };
    
    await pipeline.execute(context);
    
    expect(executionOrder).toEqual(['step1', 'step2']);
    expect(context.step1Result).toBe('data1');
    expect(context.step2Result).toBe('data2');
    expect(step1.process).toHaveBeenCalledWith(context);
    expect(step2.process).toHaveBeenCalledWith(context);
  });

  test('aborts pipeline if step fails and does not execute remaining steps', async () => {
    const step1 = new PipelineStep('Step1');
    step1.process = jest.fn().mockResolvedValue();
    
    const step2 = new PipelineStep('Step2');
    step2.process = jest.fn().mockRejectedValue(new Error('Step2 failed'));
    
    const step3 = new PipelineStep('Step3');
    step3.process = jest.fn().mockResolvedValue();
    
    const pipeline = new ScrapingPipeline([step1, step2, step3], mockLogger);
    const context = {};
    
    await expect(pipeline.execute(context)).rejects.toThrow('Pipeline failed at step "Step2"');
    
    expect(step1.process).toHaveBeenCalled();
    expect(step2.process).toHaveBeenCalled();
    expect(step3.process).not.toHaveBeenCalled(); // Crucial: Step3 never runs
  });

  test('enhances error with step context for debugging', async () => {
    const failingStep = new PipelineStep('FailingStep');
    failingStep.process = jest.fn().mockRejectedValue(new Error('Original error'));
    
    const pipeline = new ScrapingPipeline([failingStep], mockLogger);
    
    try {
      await pipeline.execute({});
      fail('Should have thrown');
    } catch (error) {
      expect(error.message).toContain('Pipeline failed at step "FailingStep"');
      expect(error.stepName).toBe('FailingStep');
      expect(error.stepNumber).toBe(1);
      expect(error.originalError.message).toBe('Original error');
    }
  });

  test('logs progress after each step completion', async () => {
    const step = new PipelineStep('TestStep');
    step.process = jest.fn().mockResolvedValue();
    
    const pipeline = new ScrapingPipeline([step], mockLogger);
    await pipeline.execute({});
    
    expect(mockLogger.info).toHaveBeenCalledWith('PIPELINE', expect.stringContaining('Starting pipeline'));
    expect(mockLogger.info).toHaveBeenCalledWith('PIPELINE', expect.stringContaining('Executing: TestStep'));
    expect(mockLogger.success).toHaveBeenCalledWith('PIPELINE', expect.stringContaining('Completed: TestStep'));
  });
});
