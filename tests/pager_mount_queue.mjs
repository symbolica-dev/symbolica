/** Transport scheduling tests, with the DOM renderer replaced by a model spy.
 * Run with: node --test tests/pager_mount_queue.mjs
 * SYMBOLICA_PAGER_JS may point to a captured wheel's viewer for validation.
 * The production mount() is evaluated unchanged; no browser is needed.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { setImmediate } from 'node:timers/promises';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';

const source = readFileSync(process.env.SYMBOLICA_PAGER_JS
  || new URL('../src/api/python/display/paging.js', import.meta.url), 'utf8')
  .replace('export function mount(', 'function mount(')
  .replace('export default { render };', '');

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function fixture(transport = () => ['', null]) {
  const calls = [], pages = [], previews = [];
  const initial = { html: '<math><mi>x</mi></math>' };
  let model;
  const mount = runInNewContext(`${source}\nrender = renderStub; mount;`, {
    renderStub({ model: value }) {
      model = value;
      model.on('change:page', () => pages.push(model.get('page')));
      model.on('change:preview', () => previews.push(model.get('preview')));
      model.send({ action: 'attach', view: 'test' });
      return () => model.send({ action: 'dispose', view: 'test' });
    },
  });
  const viewer = mount({ el: {}, page: initial, send: message => {
    calls.push(message);
    return transport(message);
  } });
  return { viewer, model, calls, pages, previews, initial };
}

const hover = request => ({ action: 'preview', request, popup: false });
const popup = request => ({ action: 'preview', request, popup: true });
const names = calls => calls.map(message => message.request || message.action);

test('only the newest queued hover reaches the transport', async () => {
  const f = fixture(message => message.action === 'preview'
    ? ['preview', { request: message.request, html: message.request }]
    : ['', null]);
  f.model.send(hover('a'));
  f.model.send(hover('b'));
  f.model.send(hover('c'));
  await setImmediate();
  assert.deepEqual(names(f.calls), ['attach', 'c']);
  assert.equal(f.previews.length, 1);
  assert.equal(f.previews[0].request, 'c');
  await f.viewer.close();
});

test('running work completes while stale queued hovers are skipped', async () => {
  const running = deferred();
  const f = fixture(message => message.request === 'running'
    ? running.promise : ['', null]);
  await setImmediate();
  f.model.send(hover('running'));
  await setImmediate();
  f.model.send(hover('stale-a'));
  f.model.send(hover('stale-b'));
  f.model.send(popup('pinned'));
  f.model.send({ action: 'next' });
  f.model.send(hover('latest'));
  assert.deepEqual(names(f.calls), ['attach', 'running']);
  const response = { request: 'running', html: 'finished' };
  running.resolve(['preview', response]);
  await setImmediate();
  assert.deepEqual(names(f.calls), ['attach', 'running', 'pinned', 'next', 'latest']);
  assert.equal(f.previews[0], response);
  await f.viewer.close();
});

test('pinned popup and navigation requests retain their order and responses', async () => {
  const pages = new Map();
  const f = fixture(message => {
    if (message.action === 'preview') return ['preview', { request: message.request }];
    if (pages.has(message.action)) return ['page', pages.get(message.action)];
    return ['', null];
  });
  for (const action of ['expand', 'next', 'collapse']) pages.set(action, { action });
  f.model.send(hover('stale'));
  f.model.send(popup('first'));
  f.model.send({ action: 'expand' });
  f.model.send(popup('second'));
  f.model.send({ action: 'next' });
  f.model.send({ action: 'collapse' });
  await setImmediate();
  assert.deepEqual(names(f.calls), ['attach', 'first', 'expand', 'second', 'next', 'collapse']);
  assert.deepEqual(f.pages.map(page => page.action), ['expand', 'next', 'collapse']);
  assert.deepEqual(f.previews.map(preview => preview.request), ['first', 'second']);
  assert.equal(f.model.get('page'), pages.get('collapse'));
  await f.viewer.close();
});

test('a navigation action supersedes a queued hover even without a newer preview', async () => {
  const f = fixture();
  f.model.send(hover('stale'));
  f.model.send({ action: 'next' });
  await setImmediate();
  assert.deepEqual(names(f.calls), ['attach', 'next']);
  await f.viewer.close();
});

test('stale preview failures never overwrite an authoritative main page', async () => {
  const running = deferred();
  const f = fixture(message => message.request === 'stale'
    ? running.promise : ['', null]);
  f.model.send(hover('stale'));
  await setImmediate();
  const updated = { html: 'new page from event transport' };
  f.viewer.update(['page', updated]);
  f.model.send(popup('current'));
  running.reject(new Error('old preview failed'));
  await setImmediate();
  assert.equal(f.model.get('page'), updated);
  assert.equal(f.pages.length, 1);
  assert.equal(f.previews[0].request, 'stale');
  assert.match(f.previews[0].error, /old preview failed/);
  assert.deepEqual(names(f.calls), ['attach', 'stale', 'current']);
  await f.viewer.close();
});

test('current popup errors are scoped and bounded; navigation errors still reach the page', async () => {
  const f = fixture(message => {
    if (message.action === 'preview') throw new Error('x'.repeat(2000));
    if (message.action === 'next') throw new Error('navigation failed');
    return ['', null];
  });
  f.model.send(popup('popup'));
  await setImmediate();
  assert.equal(f.model.get('page'), f.initial);
  assert.equal(f.previews[0].request, 'popup');
  assert.equal(f.previews[0].error.length, 1024);
  f.model.send({ action: 'next' });
  f.model.send({ action: 'back' });
  await setImmediate();
  assert.match(f.pages[0].error, /navigation failed/);
  assert.deepEqual(names(f.calls), ['attach', 'popup', 'next', 'back']);
  await f.viewer.close();
});

test('close waits for running work and sends dispose exactly once', async () => {
  const running = deferred();
  const f = fixture(message => message.request === 'running'
    ? running.promise : ['', null]);
  f.model.send(hover('running'));
  await setImmediate();
  f.model.send(hover('queued'));
  f.model.send(popup('queued-popup'));
  f.model.send({ action: 'next' });
  const firstClose = f.viewer.close();
  const secondClose = f.viewer.close();
  assert.equal(firstClose, secondClose);
  let finished = false;
  firstClose.then(() => { finished = true; });
  f.model.send({ action: 'back' });
  f.viewer.update(['page', { html: 'after close' }]);
  await setImmediate();
  assert.equal(finished, false);
  assert.deepEqual(names(f.calls), ['attach', 'running']);
  running.resolve(['preview', { request: 'running' }]);
  await firstClose;
  assert.equal(finished, true);
  assert.deepEqual(names(f.calls), ['attach', 'running', 'dispose']);
  assert.equal(f.pages.length, 0);
  assert.equal(f.previews.length, 0);
});

test('immediate close retains the original attach/dispose lifecycle', async () => {
  const f = fixture();
  await f.viewer.close();
  assert.deepEqual(names(f.calls), ['dispose']);
});
