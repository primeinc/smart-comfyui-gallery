/**
 * What the person arranged, kept until they rearrange it.
 *
 * This application already has three kinds of state and was missing a
 * fourth, which is why every surface that wanted to remember something
 * had started inventing its own key in localStorage:
 *
 *   server state       rows; the truth about media
 *   URL state          the question being asked -- shareable, bookmarkable
 *   ephemeral state    hover, drag, what is under the pointer right now
 *   WORKSPACE state    how this person has arranged their tools
 *
 * The distinction that matters is lifetime, and it is a judgement about
 * intent rather than about storage. Workspace state is a DELIBERATE
 * arrangement: which panel is open, which sections are disclosed, what is
 * in the comparison tray. It survives navigation and reload, and changes
 * only when the person changes it.
 *
 * Zoom and pan are NOT workspace state. They are how somebody is looking
 * at one picture, they belong to that picture, and carrying them to the
 * next one would be restoring something nobody arranged. The viewer says
 * the same thing about its own transforms; this module exists so the two
 * lifetimes stop being confused for each other.
 *
 * Nothing here is shared. It is this browser's arrangement, per origin,
 * never sent to the server -- another person opening the same library
 * gets their own, and no row anywhere records how somebody likes their
 * panels.
 */

/** Bumped when a stored shape stops being readable by this code. */
const VERSION = 1;

const KEY = `sg.workspace.v${VERSION}`;

/**
 * Everything remembered, as one object.
 *
 * One key rather than a key per setting: a workspace read at startup is
 * a single parse, and a version bump can drop the whole thing rather
 * than leaving a scatter of orphans behind under names nothing reads.
 */
export interface Workspace {
  /** The media viewer's information panel. */
  inspector?: "open" | "closed";
  /**
   * The gallery's filter drawer.
   *
   * Whether the drawer is open is furniture. WHICH FILTERS ARE HELD is
   * never here: that is the question, it lives in the URL, and a filter
   * that outlived its URL would mean one link answering differently for
   * two people.
   */
  filters?: "open" | "closed";
  /**
   * Which inspector sections are disclosed, by their `data-panel` name.
   * Absent means "this person has never said", which is what lets a
   * generated picture and a photograph open different sections until
   * somebody decides otherwise.
   */
  panels?: Record<string, boolean>;
}

/**
 * Read the workspace, or an empty one.
 *
 * Every failure answers the same way: an empty workspace, so a surface
 * gets its defaults. Storage can be unavailable outright (private
 * windows, embedded views, a browser told to block site data), and the
 * stored text can be from a version that no longer parses. Neither is
 * worth an error a person cannot act on -- the cost of not remembering
 * is having to arrange the panels again.
 */
export function workspace(): Workspace {
  try {
    const held = localStorage.getItem(KEY);
    if (!held) return {};
    const parsed: unknown = JSON.parse(held);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return parsed as Workspace;
  } catch {
    return {};
  }
}

/**
 * Change part of the workspace, leaving the rest alone.
 *
 * Read-modify-write on one key, because two surfaces open in two tabs
 * both hold the whole object and a blind overwrite would drop whichever
 * settings the other one had just made.
 */
export function remember(change: Partial<Workspace>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...workspace(), ...change }));
  } catch {
    // Nothing to do and nothing to say: a browser that will not store is
    // a browser where this person rearranges their panels each visit.
  }
}

/** Whether a named section is disclosed, or `undefined` if never said. */
export function panelState(name: string): boolean | undefined {
  return workspace().panels?.[name];
}

/** Record one section's disclosure. */
export function rememberPanel(name: string, open: boolean): void {
  remember({ panels: { ...(workspace().panels ?? {}), [name]: open } });
}

/** Forget everything. Exposed for a settings surface and for tests. */
export function forgetWorkspace(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // as above
  }
}
