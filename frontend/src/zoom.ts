/**
 * The zoom ceilings, in one place so the two magnifying surfaces state
 * their difference instead of hiding it. The viewer inspects pixels and
 * goes to 40x; the compare tray's glass answers "are these the same
 * picture?", and 16x is past any difference that question needs while
 * keeping the tray's four panes responsive.
 */
export const MAX_SCALE = 40;
export const TRAY_MAX_SCALE = 16;
