// The keyboard registry, at the cheapest layer that can prove it.
//
// A browser cannot show this: the failure being checked is what the map
// holds AFTER a registration was refused, and a refused registration is by
// definition a page that did not finish loading. So it is asserted here,
// in milliseconds, by node's own runner over the real module -- no test
// framework, because node 24 runs TypeScript directly (package.json pins
// the floor for exactly that reason).
//
// The module attaches one document listener when it loads, so a stand-in
// goes in first. It is the only thing stubbed: `register` and `claimant`
// are the real ones.
import assert from "node:assert/strict";
import test from "node:test";

Object.defineProperty(globalThis, "document", {
  value: { addEventListener() {} },
  configurable: true,
});

const { register, claimant } = await import("./keys.ts");

test("a key answers to whoever claimed it", () => {
  const release = register([{ key: "q", by: "first", run: () => {} }]);
  assert.equal(claimant("q"), "first");
  release();
  assert.equal(claimant("q"), null);
});

test("letters are claimed case-insensitively, because caps lock means the same thing", () => {
  const release = register([{ key: "w", by: "first", run: () => {} }]);
  assert.equal(claimant("W"), "first");
  assert.throws(() => register([{ key: "W", by: "second", run: () => {} }]), /second claims "w", which first/);
  release();
});

test("a refused registration leaves the keyboard exactly as it found it", () => {
  // The defect: claiming key by key meant a batch that collided on its
  // fifth left the first four registered by a registration that THREW --
  // a half-claimed keyboard, which is worse than either outcome and
  // unreadable from the error.
  const held = register([{ key: "e", by: "sitting tenant", run: () => {} }]);
  assert.throws(
    () =>
      register([
        { key: "r", by: "late arrival", run: () => {} },
        { key: "t", by: "late arrival", run: () => {} },
        { key: "e", by: "late arrival", run: () => {} },
        { key: "y", by: "late arrival", run: () => {} },
      ]),
    /late arrival claims "e", which sitting tenant/,
  );
  for (const key of ["r", "t", "y"]) {
    assert.equal(claimant(key), null, `${key} was claimed by a registration that failed`);
  }
  assert.equal(claimant("e"), "sitting tenant");
  held();
});

test("a batch that collides with itself is refused before it touches anything", () => {
  assert.throws(
    () =>
      register([
        { key: "u", by: "one half", run: () => {} },
        { key: "i", by: "one half", run: () => {} },
        { key: "u", by: "the other half", run: () => {} },
      ]),
    /the other half and one half both claim "u"/,
  );
  assert.equal(claimant("u"), null);
  assert.equal(claimant("i"), null);
});

test("releasing gives a key back to the next claimant, and only its own keys", () => {
  const mine = register([{ key: "o", by: "mine", run: () => {} }]);
  const yours = register([{ key: "p", by: "yours", run: () => {} }]);
  mine();
  assert.equal(claimant("o"), null);
  assert.equal(claimant("p"), "yours", "one release must not drop another registration's keys");
  register([{ key: "o", by: "next", run: () => {} }])();
  yours();
});
