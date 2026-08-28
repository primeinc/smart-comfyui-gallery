// Installed on every surface that renders a `<time data-epoch>`, the
// same way the compare tray and the picture fallback are: spelling a
// date is a property of the application, not of one page, and it is
// exactly the kind of rule nobody remembers to apply nine times -- four
// pages were shipping raw Unix floats to the reader because of it.
//
// The observer is for the swaps: htmx replaces a shelf in place, and the
// rows that arrive carry unspelled epochs like the ones that left.
import { spellDays } from "./spelling";

spellDays(document);
new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });
