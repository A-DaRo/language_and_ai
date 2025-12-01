const PD = require('../../../src/core/ProtocolDefinitions');

describe('ProtocolDefinitions', () => {
  test('serialize and deserialize error round-trip', () => {
    const err = new Error('Boom');
    err.code = 'E_TEST';
    const s = PD.serializeError(err);
    expect(s).toHaveProperty('message', 'Boom');
    expect(s).toHaveProperty('name', 'Error');
    expect(s).toHaveProperty('stack');
    expect(s).toHaveProperty('code', 'E_TEST');

    const d = PD.deserializeError(s);
    expect(d).toBeInstanceOf(Error);
    expect(d.message).toBe('Boom');
    expect(d.name).toBe('Error');
    expect(d.code).toBe('E_TEST');
  });

  test('serialize/deserialize title map', () => {
    const map = new Map([['id1', 'Title 1'], ['id2', 'Title 2']]);
    const obj = PD.serializeTitleMap(map);
    expect(obj).toEqual({ id1: 'Title 1', id2: 'Title 2' });

    const map2 = PD.deserializeTitleMap(obj);
    expect(map2).toBeInstanceOf(Map);
    expect(map2.get('id1')).toBe('Title 1');
    expect(map2.get('id2')).toBe('Title 2');
  });

  test('validateMessage accepts valid type and rejects invalid', () => {
    const valid = { type: PD.MESSAGE_TYPES.INIT };
    expect(() => PD.validateMessage(valid)).not.toThrow();

    expect(() => PD.validateMessage(null)).toThrow();
    expect(() => PD.validateMessage({})).toThrow();
    expect(() => PD.validateMessage({ type: 'FOO' })).toThrow();
  });
});
