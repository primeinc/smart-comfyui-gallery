// Asking a person something, in the application, rather than through the
// browser's own 1995 boxes.
//
// `window.prompt`, `window.confirm` and `window.alert` were the last
// surfaces here that did not look like this application. They are worse
// than ugly: prompt has one unlabelled text field and no way to offer
// choices, so "which smart collection?" became a comma-joined list of
// slugs pasted into a sentence with an instruction to type one back. That
// is the application asking a person to remember its internal spelling,
// which is the one thing it exists to do for them.
//
// Built on the native <dialog>, so the parts that are hard to get right
// are the browser's: focus moves in and returns to whatever opened it,
// everything else goes inert, Escape closes, the top layer is above every
// stacking context on the page, and ::backdrop is a real element to style.
// A hand-rolled div reimplements all of that, usually badly -- the MDN
// note is blunt about it ("ensure that all expected default behaviors are
// supported").
//
// One shape underneath: a <form method="dialog"> whose submitting button
// SETS the return value. So "which one" and "yes or no" are the same
// mechanism -- the buttons differ, the reading does not -- and a dismissal
// is the empty return value the browser leaves behind when nobody chose.
import { requireElement } from "./dom";

/** What the browser leaves in `returnValue` when nothing was chosen. */
const DISMISSED = "";

/** The affirmative's return value where the answer is not a choice. */
const TAKEN = "ok";

/** The frame every ask shares. `null` for a button means it is not there. */
export interface Asked {
  /** The heading: what is being asked, in words a person recognises. */
  question: string;
  /** A second line, when the question needs a consequence spelled out. */
  detail?: string;
  /** The affirmative button's words, or null where the body answers. */
  submit: string | null;
  /** The dismissing button's words, or null where there is nothing to refuse. */
  dismiss: string | null;
  /** Whether taking this is destructive, which colours the affirmative. */
  grave?: boolean;
}

/** What a caller may say about the frame. The question is the argument. */
export type Framing = Partial<Omit<Asked, "question">>;

/**
 * The frame a public ask hands `ask`, with its own defaults under the
 * caller's.
 *
 * Written out rather than spread, because `exactOptionalPropertyTypes`
 * makes the two different claims: `{ detail?: string }` says the key may
 * be ABSENT, and spreading an optional over a required key produces
 * `string | undefined`, which is the key being PRESENT and empty. The
 * distinction is the point of the setting, so it is honoured here
 * instead of being widened away.
 */
const framed = (question: string, submit: string | null, dismiss: string | null, said: Framing): Asked => ({
  question,
  submit: said.submit !== undefined ? said.submit : submit,
  dismiss: said.dismiss !== undefined ? said.dismiss : dismiss,
  ...(said.detail !== undefined ? { detail: said.detail } : {}),
  ...(said.grave !== undefined ? { grave: said.grave } : {}),
});

const button = (words: string, value: string, kind: string): HTMLButtonElement => {
  const control = document.createElement("button");
  // No `type`: a button in a form is a submit button, and submitting a
  // form with method="dialog" is what closes the dialog and records which
  // button did it. Setting type="button" here would close nothing.
  control.value = value;
  control.className = kind;
  control.textContent = words;
  return control;
};

/**
 * Put the question on screen and answer with what `read` saw, or null.
 *
 * `build` fills the body and hands back a reader. The reader runs while
 * the dialog's nodes are still in the document -- the alternative is
 * caching every field's value on every keystroke, or reading detached
 * nodes, and neither is worth it for a function that is over in one turn.
 *
 * Dismissal is one case with three doors: the cancel button, Escape (the
 * browser's own, which a modal dialog gets for free) and a click on the
 * backdrop. All three arrive here as the empty return value.
 */
async function ask<T>(asked: Asked, build: (body: HTMLElement, box: HTMLDialogElement) => () => T): Promise<T | null> {
  const box = document.createElement("dialog");
  box.className = "ask-box";
  box.innerHTML = `<form method="dialog" class="ask-form">
      <h2 class="ask-question"></h2>
      <p class="ask-detail" hidden></p>
      <div class="ask-body"></div>
      <div class="ask-feet"></div>
    </form>`;

  requireElement(box, ".ask-question", HTMLElement).textContent = asked.question;
  if (asked.detail !== undefined) {
    const line = requireElement(box, ".ask-detail", HTMLElement);
    line.textContent = asked.detail;
    line.hidden = false;
  }

  const read = build(requireElement(box, ".ask-body", HTMLElement), box);

  // The affirmative goes FIRST in the document and last on the screen
  // (the stylesheet gives the dismissal `order: -1`). Implicit submission
  // -- Enter in a text field -- picks the first submit button in tree
  // order, and a person typing a name and pressing Enter means save.
  const feet = requireElement(box, ".ask-feet", HTMLElement);
  if (asked.submit !== null) {
    feet.append(button(asked.submit, TAKEN, asked.grave === true ? "ask-take is-grave" : "ask-take"));
  }
  if (asked.dismiss !== null) feet.append(button(asked.dismiss, DISMISSED, "ask-drop"));

  // A click whose point is outside the dialog's own box is a click on the
  // backdrop, which has no element of its own to listen on. Measured
  // rather than inferred from the target, because the target of a click
  // on the backdrop IS the dialog. `closedby="any"` says this natively
  // and is Chrome 134 and up; this is the same behaviour everywhere.
  box.addEventListener("click", (event) => {
    const at = box.getBoundingClientRect();
    const inside =
      event.clientX >= at.left && event.clientX <= at.right && event.clientY >= at.top && event.clientY <= at.bottom;
    // A keyboard-activated button reports (0, 0) and a detail of 0, which
    // is outside every dialog and would read as a backdrop click that
    // threw away the answer somebody just pressed Enter on.
    if (!inside && event.detail > 0) box.close(DISMISSED);
  });

  const answer = new Promise<T | null>((settle) => {
    box.addEventListener(
      "close",
      () => {
        const taken = box.returnValue !== DISMISSED ? read() : null;
        box.remove();
        settle(taken);
      },
      { once: true },
    );
  });

  document.body.append(box);
  box.showModal();
  return answer;
}

/**
 * Show something and wait for it to be dismissed.
 *
 * `say` takes a sentence; this takes nodes, for the case where what has
 * to be read is a list or a table rather than a line of prose. One
 * dismissal and no affirmative: there is nothing here to agree to.
 *
 * On `ask` rather than a second dialog, so the focus trap, Escape, the
 * backdrop click and the removal are the ones already proven here.
 */
export async function panel(title: string, fill: (body: HTMLElement) => void, dismiss = "close"): Promise<void> {
  await ask({ question: title, submit: null, dismiss }, (body) => {
    fill(body);
    return () => undefined;
  });
}

/**
 * Say something and wait for it to be read.
 *
 * The replacement for `window.alert`, and awaited rather than fired off:
 * every caller of alert relied on it blocking -- refusal, then reload, or
 * refusal, then navigate -- and a message that vanishes under the page it
 * was explaining is not a message.
 */
export async function say(message: string, framing: Framing = {}): Promise<void> {
  await ask(framed(message, "ok", null, framing), () => () => undefined);
}

/** A yes-or-no question. The replacement for `window.confirm`. */
export async function askYesNo(question: string, framing: Framing = {}): Promise<boolean> {
  return (await ask(framed(question, "yes", "no", framing), () => () => true)) === true;
}

/** What a text ask may say beyond the frame. */
export interface Typed extends Framing {
  /** What the field starts holding. */
  value?: string;
  /** The grey hint inside an empty field. */
  placeholder?: string;
  /** The field's accessible name, when the question is not it. */
  label?: string;
}

/**
 * Ask for a line of text. The replacement for `window.prompt`.
 *
 * Empty is dismissal, not an empty answer: every caller here names
 * something, and nothing in this application is named "".
 */
export async function askText(question: string, typed: Typed = {}): Promise<string | null> {
  const said = await ask(framed(question, "save", "cancel", typed), (body) => {
    const field = document.createElement("input");
    field.type = "text";
    field.className = "ask-field";
    field.value = typed.value ?? "";
    field.placeholder = typed.placeholder ?? "";
    field.autofocus = true;
    field.setAttribute("aria-label", typed.label ?? question);
    body.append(field);
    return () => field.value.trim();
  });
  return said ? said : null;
}

/** One offered answer: what it means, what it is called, and any aside. */
export interface Choice {
  /** The value handed back. Never empty -- empty is dismissal. */
  value: string;
  /** What a person reads. */
  label: string;
  /** A second line: what distinguishes this one from its neighbours. */
  note?: string;
}

/**
 * Ask which of these.
 *
 * Each choice is its own submit button, so choosing is one click and the
 * chosen value IS the return value -- there is no selected-then-confirm
 * step, and no way to submit a choice nobody made. This is the shape
 * `window.prompt` could not have: it has one text field, so offering a
 * list meant printing the list into the question and asking a person to
 * type one of them back.
 */
export async function askChoice(question: string, choices: Choice[], framing: Framing = {}): Promise<string | null> {
  if (choices.length === 0) return null;
  return ask(
    // No affirmative in the feet: the choices are the affirmative.
    framed(question, null, "cancel", { ...framing, submit: null }),
    (body, box) => {
      const list = document.createElement("div");
      list.className = "ask-choices";
      for (const [index, one] of choices.entries()) {
        const control = button(one.label, one.value, "ask-choice");
        control.autofocus = index === 0;
        if (one.note !== undefined) {
          const note = document.createElement("span");
          note.className = "ask-choice-note";
          note.textContent = one.note;
          control.append(note);
        }
        list.append(control);
      }
      body.append(list);
      // The chosen value is the dialog's return value, which the button
      // that submitted set, so it travels the same path as every other
      // answer rather than through a second piece of state.
      return () => box.returnValue;
    },
  );
}
