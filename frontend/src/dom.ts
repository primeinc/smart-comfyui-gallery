// Executable DOM expectations: the type proof and the runtime invariant are
// the same fact.
//
// A selector the page must satisfy returns a typed element or throws where it
// was written, naming the selector and the type it wanted -- never `null`
// travelling three calls to die as "cannot read properties of null". A
// selector the page may satisfy returns the element or null, and the caller
// has to say what absence means.
//
// This is what `as HTMLInputElement` pretends to do. The cast asserts the
// fact; `instanceof` proves it, and costs one comparison.

/** A DOM interface's constructor, as a value: `HTMLInputElement`, `Element`. */
export type ElementType<T extends Element> = abstract new (...args: never[]) => T;

/** The element `selector` names, or an error naming what was expected. */
export function requireElement<T extends Element>(root: ParentNode, selector: string, type: ElementType<T>): T {
  const found = root.querySelector(selector);
  if (!(found instanceof type)) {
    throw new Error(`expected ${selector} to be ${type.name}, found ${describe(found)}`);
  }
  return found;
}

/** The element `selector` names, or null when the page does not carry it. */
export function findElement<T extends Element>(root: ParentNode, selector: string, type: ElementType<T>): T | null {
  const found = root.querySelector(selector);
  return found instanceof type ? found : null;
}

/** Every element `selector` names, each proven to be `type`. */
export function everyElement<T extends Element>(root: ParentNode, selector: string, type: ElementType<T>): T[] {
  return [...root.querySelectorAll(selector)].filter((node): node is T => node instanceof type);
}

/**
 * The nearest ancestor-or-self of an event's target matching `selector`.
 *
 * `event.target` is an EventTarget, which need not be a Node at all, so the
 * narrowing belongs here rather than at every listener.
 */
export function closestFrom<T extends Element>(
  target: EventTarget | null,
  selector: string,
  type: ElementType<T>,
): T | null {
  if (!(target instanceof Element)) return null;
  const found = target.closest(selector);
  return found instanceof type ? found : null;
}

/** A dataset value the markup must carry. */
export function requireData(node: HTMLElement, key: string): string {
  const held = node.dataset[key];
  if (held === undefined) {
    throw new Error(`expected a data-${key} on ${node.tagName.toLowerCase()}`);
  }
  return held;
}

function describe(found: Element | null): string {
  return found === null ? "nothing" : found.constructor.name;
}
