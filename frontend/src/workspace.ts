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
   * What has been KEPT to compare, in the order it will be shown.
   *
   * Workspace state in the fullest sense: a person put these here on
   * purpose, one at a time, from several surfaces, and the whole point
   * is that walking away and coming back does not lose them. It empties
   * when they empty it and at no other moment.
   *
   * The name rides along with the slug because the surface that added
   * one knew it, and asking the server to say it again for a strip of
   * five thumbnails would be five round trips for text already on screen.
   */
  compare?: Array<{ slug: string; name: string }>;
  /**
   * How a comparison is shown: everything at once, or one at a time in
   * the same place.
   *
   * Two different questions, which is why it is a choice rather than a
   * better default. SIDE BY SIDE answers "how do these differ", and you
   * read it by moving your eyes. FLIP answers "did this change", and
   * you read it by NOT moving them -- the pictures occupy the same
   * pixels, so a small difference that side-by-side hides in the
   * saccade is the only thing that moves.
   */
  compareMode?: "side" | "flip";
  /** Whether the compare tray is open or collapsed to its tab. */
  tray?: "open" | "closed";
  /**
   * The walk on a timer: how long each picture is shown, in seconds.
   *
   * Workspace state, and the argument is worth stating because the
   * viewer's own header says nothing here is persisted. Zoom and pan are
   * how somebody is looking at ONE picture and belong to that picture.
   * How fast the walk moves, and what happens at its ends, is an
   * arrangement of the tool -- it survives the picture, and being asked
   * it again every time you open a slideshow is the thing this avoids.
   */
  showEvery?: number;
  /**
   * Whether the slideshow is running.
   *
   * Here rather than in a variable BECAUSE the viewer is remounted on
   * every step -- the overlay replaces its contents, the page navigates
   * -- so a timer held in the module dies with each picture. Playing is
   * a fact about the walk, not about the mount, and this is the only
   * place with the walk's lifetime.
   */
  showPlaying?: boolean;
  /**
   * What the ARROWS do at either end of the answer.
   *
   * Off, the walk stops there. On, next from the last member is the
   * first. Never a silent slide into a different question: crossing the
   * end is this same answer starting again.
   */
  wrap?: boolean;
  /**
   * What the SLIDESHOW does when it reaches the end.
   *
   * Off, it stops and stays on the last picture. On, it starts again.
   * Separate from `wrap` because they are different questions: somebody
   * can want a slideshow that repeats all night and arrows that still
   * tell them when they have seen everything.
   */
  loop?: boolean;
  /**
   * Which inspector sections are disclosed, by their `data-panel` name.
   * Absent means "this person has never said", which is what lets a
   * generated picture and a photograph open different sections until
   * somebody decides otherwise.
   */
  panels?: Record<string, boolean>;
  /**
   * The install prompt, told to go away.
   *
   * A deliberate arrangement -- somebody said "not this" -- so it lives
   * with the rest of them. It had its own localStorage key, which is the
   * scatter this module exists to end.
   */
  installDismissed?: boolean;
  /**
   * How tall a row of the timeline's river is, in pixels.
   *
   * Dragged to a size on purpose and expected to stay there, which is
   * the definition above. Also had its own key.
   */
  timelineRow?: number;
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
