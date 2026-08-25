// One keystroke, one meaning, proven rather than agreed.
//
// The viewer and the authored strip are separate modules that ship in the
// same bundle on the same surfaces, and each had grown its own
// `document.addEventListener("keydown", ...)`. Nothing compared them, so
// `F` was focus AND favorite, `1` was actual-pixels AND one star, `0` was
// fit AND clear-rating. Every one of those fired both handlers: a person
// looking closely at a photograph was silently rating it.
//
// Two listeners cannot agree about a key by being careful. So there is one
// listener here, modules REGISTER what they respond to, and a second claim
// on a live key throws where it was registered -- naming both claimants.
// The failure is a page that does not load rather than a picture that
// quietly gains a star, and the browser tests hit it on the first mount.
//
// Chords are not modelled on purpose. Every command here is a bare key,
// because a modifier held over the viewer means something else entirely
// (frontend/src/viewer.ts: the wheel walks the library), and a vocabulary
// that mixed the two would have to explain which wins.
//
// This module imports nothing. That is what lets node run it directly
// against a one-method document stand-in (keys.test.ts), which is the only
// layer that can see what the map holds after a registration was REFUSED.

/** One key, and what the surface holding it should do. */
export interface Command {
  /**
   * The `KeyboardEvent.key` this responds to.
   *
   * Single letters are given lower-case and matched case-insensitively:
   * a person with caps lock on means the same thing.
   */
  key: string;
  /** Who claims it, named in the refusal when two modules claim one key. */
  by: string;
  run: () => void;
}

const claimed = new Map<string, Command>();

const spelled = (key: string) => (key.length === 1 ? key.toLowerCase() : key);

/**
 * Claim these keys until the returned release is called.
 *
 * Registration is where a collision is caught, not dispatch: a module that
 * registers has said what it responds to, and the whole point is that the
 * claim is decided once. A remounted surface releases first (the overlay
 * replaces its contents on every open), so re-registering the same keys
 * from the same place is ordinary, not a conflict.
 */
export function register(commands: Command[]): () => void {
  // Proven admissible in full BEFORE anything is written. Claiming as it
  // went meant a batch that collided on its fifth key left the first four
  // registered by a registration that threw -- a half-claimed keyboard,
  // which is worse than either outcome and impossible to reason about
  // from the error. A refused registration changes nothing at all.
  const wanted = new Map<string, Command>();
  for (const command of commands) {
    const key = spelled(command.key);
    const mine = wanted.get(key);
    if (mine) throw new Error(`${command.by} and ${mine.by} both claim "${key}" in one registration`);
    const already = claimed.get(key);
    if (already) throw new Error(`${command.by} claims "${key}", which ${already.by} already answers to`);
    wanted.set(key, command);
  }
  for (const [key, command] of wanted) claimed.set(key, command);
  return () => {
    for (const command of commands) {
      const key = spelled(command.key);
      if (claimed.get(key) === command) claimed.delete(key);
    }
  };
}

/** What responds to a key right now, or null. Exported for the tests. */
export function claimant(key: string): string | null {
  return claimed.get(spelled(key))?.by ?? null;
}

document.addEventListener("keydown", (event) => {
  // Somebody typing means the letters: a place name with an "L" in it is
  // not a request to turn the lights out.
  const target = event.target;
  if (target instanceof Element && target.closest("input, textarea, select, [contenteditable]")) return;
  // A held modifier is a different vocabulary -- the browser's, or the
  // viewer's wheel -- and never a command here.
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const command = claimed.get(spelled(event.key));
  if (!command) return;
  event.preventDefault();
  command.run();
});
