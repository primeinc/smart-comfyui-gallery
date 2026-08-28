/**
 * What the person arranged, kept until they rearrange it.
 *
 * Four kinds of state, and this module owns the fourth:
 *
 *   server      rows; the truth about media
 *   URL         the question being asked -- shareable, bookmarkable
 *   ephemeral   hover, drag, what is under the pointer right now
 *   workspace   how this person has arranged their tools
 *
 * What separates workspace state is lifetime, and that is a judgement
 * about intent rather than about where it is stored. A deliberate
 * arrangement -- which panel is open, which sections are disclosed, what
 * is in the compare tray -- survives navigation and reload, and changes
 * only when the person changes it.
 *
 * Zoom and pan are not workspace state. They are how somebody is looking
 * at ONE picture, they belong to that picture, and carrying them to the
 * next would restore something nobody arranged.
 *
 * Nothing here is shared. It is this browser's arrangement, per origin,
 * never sent to the server: another person opening the same library gets
 * their own, and no row records how anybody likes their panels.
 *
 * Every surface that wanted to remember something was inventing its own
 * localStorage key. There is one key now.
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
  /**
   * The board: what this person has put on their canvas, and where.
   *
   * Workspace state in the fullest sense, by this module's own
   * definition. A pin is a deliberate act -- somebody put a question, a
   * person or a photograph somewhere on purpose, and the whole point is
   * that it is still there tomorrow. It changes when they change it and
   * at no other moment.
   *
   * NOT server rows, and the distinction is the one at the top of this
   * file. An album is a collection: it has members, an address, and a
   * place on a shelf everybody shares. A pin is where YOU chose to keep
   * a shortcut to one. Storing pins as rows would make one person's
   * arrangement of their desk into a fact about the library.
   */
  board?: Pin[];
}

/**
 * One thing kept on the board.
 *
 * `kind` says what it stands for and therefore how it opens; `at` is the
 * question or address it opens, in the application's own spelling, so a
 * pin is a bookmark to a surface rather than a copy of one. Nothing
 * about the pictures is stored here -- a pinned query re-answers itself
 * every time it is looked at, which is what keeps a pin from slowly
 * becoming a lie as the library changes underneath it.
 */
export interface Pin {
  /** Unique on this board; also the drawing's identity across a redraw. */
  id: string;
  kind: "query" | "person" | "album" | "folder" | "picture" | "compare";
  /** A compare pin only: the two pins it holds against each other. It
   *  stores their IDS rather than their questions, so moving or renaming
   *  one is not a second place the question has to be kept right. */
  against?: [string, string];
  /** What to call it. The person's own word where they gave one. */
  name: string;
  /** The query string or path this pin opens. */
  at: string;
  /** Where it sits on the board, in board units. */
  x: number;
  y: number;
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

/** What is on the board, in the order it was put there. */
export function board(): Pin[] {
  const held = workspace().board;
  return Array.isArray(held) ? held : [];
}

/**
 * Put something on the board, or move it if it is already there.
 *
 * Keyed by `id` rather than appended, because pinning the same question
 * twice is somebody expecting one card, not two -- and because dragging
 * a card is the same write as making it.
 */
export function pin(one: Pin): void {
  const held = board().filter((other) => other.id !== one.id);
  remember({ board: [...held, one] });
}

/** Take something off the board. */
export function unpin(id: string): void {
  remember({ board: board().filter((one) => one.id !== id) });
}

/** Forget everything. Exposed for a settings surface and for tests. */
export function forgetWorkspace(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // as above
  }
}
